"""PF-RPN 主模型。

结构（对 Grounding DINO 的最小必要改造）：
    骨干 4 级特征 -> 通道对齐 -> 可学习视觉嵌入 F_T
        -> SIA  (Eq.1，跨层级稀疏路由 + 交叉注意力)
        -> CSP  (Eq.2/3，从深到浅级联自提示)
        -> DETR 式解码器（query 数 = num_queries）
            -> 分类 logit = <query_feat, F~_4^T>   （对比学习，单类 object）
            -> 回归 cxcywh
            -> CG-QS 中心度分数 g_i

与 Grounding DINO 唯一的结构差异：**去掉文本编码器与文本嵌入**，
用一个可学习的视觉嵌入 F_T 扮演同样的角色。这正是「prompt-free」的实现方式：
所有类别被合并为一个 token ``object``，因此不需要任何类别名或样例图。
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import TinyMultiScaleBackbone, sine_positional_embeddings
from .decoder import TransformerDecoder
from .modules.cgqs import CenternessScorer
from .modules.csp import CascadeSelfPrompt
from .modules.sia import SparseImageAwareAdapter


class PFRPN(nn.Module):
    """Prompt-Free Universal Region Proposal Network（精简复现版）。"""

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        d_model = cfg["d_model"]

        self.backbone = TinyMultiScaleBackbone(**cfg.get("backbone", {}))
        in_chs = self.backbone.widths                       # 各阶段原始通道数
        self.level_proj = nn.ModuleList(
            [nn.Conv2d(c, d_model, kernel_size=1, bias=False) for c in in_chs]
        )

        # ---- 可学习视觉嵌入 F_T（1 x C），取代 Grounding DINO 的文本嵌入 ----
        self.embed_token = nn.Parameter(torch.zeros(1, d_model))
        nn.init.normal_(self.embed_token, std=0.02)

        # ---- 三大核心模块 ----
        self.sia = SparseImageAwareAdapter(
            d_model,
            num_levels=len(in_chs),
            topk=cfg.get("sia_topk", 2),
            dropout=cfg.get("dropout", 0.1),
        )
        self.csp = CascadeSelfPrompt(
            d_model,
            threshold=cfg.get("csp_delta", 0.3),
            num_iter=cfg.get("csp_iters", 3),
            order=tuple(range(len(in_chs) - 1, -1, -1)),
        )
        self.cgqs = CenternessScorer(d_model, hidden=cfg.get("ctr_hidden", 256))

        # ---- 解码器与检测头 ----
        self.num_queries = cfg["num_queries"]
        self.query_embed = nn.Embedding(self.num_queries, d_model)
        nn.init.normal_(self.query_embed.weight, std=0.02)
        self.decoder = TransformerDecoder(
            d_model=d_model,
            nhead=cfg.get("nhead", 8),
            num_layers=cfg.get("num_layers", 4),
            dim_ff=cfg.get("dim_ff", 512),
            dropout=cfg.get("dropout", 0.1),
        )
        self.reg_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 4),
        )
        self.scale = d_model ** -0.5

    # ------------------------------------------------------------------ #
    def build_memory(
        self, projs: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """把 4 级特征投影结果拼成 DETR 式 memory 并相加位置编码。

        返回 (B, S, d_model) 的 memory 与位置编码。
        """
        B = projs[0].shape[0]
        mem_parts, pos_parts = [], []
        for p in projs:                                     # (B, C, H, W)
            _, c, h, w = p.shape
            mask = torch.ones(1, h, w, device=p.device, dtype=p.dtype)
            pe = sine_positional_embeddings(mask, p.shape[1])            # (1, C, h, w)
            pe = pe.expand(B, -1, -1, -1)
            if pe.shape[-2:] != p.shape[-2:]:
                pe = F.interpolate(pe, size=p.shape[-2:], mode="bilinear", align_corners=False)
            mem_parts.append(p.flatten(2).transpose(1, 2))               # (B, h*w, C)
            pos_parts.append(pe.flatten(2).transpose(1, 2))
        memory = torch.cat(mem_parts, dim=1)
        pos = torch.cat(pos_parts, dim=1)
        return memory, pos

    # ------------------------------------------------------------------ #
    def forward(self, images: torch.Tensor, return_masks: bool = False) -> Dict[str, torch.Tensor]:
        """images : (B, 3, H, W) 归一化到 [0, 1] 的张量。"""
        B = images.shape[0]

        feats = self.backbone(images)
        projs = [proj(f) for proj, f in zip(self.level_proj, feats)]      # 通道对齐到 d_model

        # ---- ① 可学习嵌入的初始化与第一次更新（SIA）----
        embed0 = self.embed_token.expand(B, -1)                          # (B, C)
        embed1, sia_aux = self.sia(projs, embed0)
        router_w = sia_aux["router_w"]
        # ---- ② 级联自提示（CSP）----
        embed2, masks = self.csp(projs, embed1)

        # ---- ③ 解码 ----
        memory, pos = self.build_memory(projs)
        q = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)       # (B, N, d_model)
        qf = self.decoder(q, memory, pos)

        # 分类分：query 特征与最终可学习嵌入的点积（论文：用对比损失打分）
        cls_logits = (qf * embed2.unsqueeze(1)).sum(-1) * self.scale      # (B, N)
        boxes = self.reg_head(qf).sigmoid()                               # (B, N, 4) 归一化 cxcywh
        ctr = self.cgqs(qf)                                               # (B, N)

        out: Dict[str, torch.Tensor] = {
            "boxes": boxes,
            "cls_logits": cls_logits,
            "ctr": ctr,
            "embed_init": embed0,
            "embed_sia": embed1,
            "embed_final": embed2,
            "router_w": router_w,
            "query_feats": qf,
        }
        if return_masks:
            out["csp_masks"] = masks
        return out

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def inference(
        self,
        images: torch.Tensor,
        topk_list: Tuple[int, ...] = (100, 300, 900),
        nms_thr: float = -1.0,
        return_all: bool = False,
    ):
        """推理。返回 (boxes, scores) 两个 list，每项长度 = batch。

        boxes   : (K, 4) 归一化 cxcywh，按综合分数降序
        scores  : (K,)   = 分类分数 * 中心度分数（论文把两者结合用于 query 选择）

        ``return_all=True`` 时不截断，返回全部 num_queries 个提案——评测时
        AR@100 / AR@300 要各自从第 K 个位置往前切，共用一次前向才划算。
        """
        self.eval()
        out = self(images)
        boxes, scores = [], []
        for i in range(images.shape[0]):
            b = out["boxes"][i]
            s = torch.sigmoid(out["cls_logits"][i]) * torch.sigmoid(out["ctr"][i])
            if nms_thr > 0:
                from .utils.box import box_xyxy_to_cxcywh, box_cxcywh_to_xyxy
                keep = _nms(box_cxcywh_to_xyxy(b), s, nms_thr)
                b, s = b[keep], s[keep]
            order = torch.argsort(s, descending=True)
            if not return_all:
                k = max(int(min(topk_list)), 0) if len(topk_list) else len(s)
                order = order[:k]
            boxes.append(b[order])
            scores.append(s[order])
        return boxes, scores


# --------------------------------------------------------------------------- #
def _nms(boxes_xyxy: torch.Tensor, scores: torch.Tensor, thr: float) -> torch.Tensor:
    """朴素 NMS（不依赖外部库）。输入已归一化 xyxy。"""
    if boxes_xyxy.numel() == 0:
        return torch.zeros(0, dtype=torch.long, device=boxes_xyxy.device)
    x1, y1, x2, y2 = boxes_xyxy.unbind(1)
    areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    order = torch.argsort(scores, descending=True)
    keep: List[int] = []
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break
        rest = order[1:]
        xx1 = torch.max(x1[i], x1[rest])
        yy1 = torch.max(y1[i], y1[rest])
        xx2 = torch.min(x2[i], x2[rest])
        yy2 = torch.min(y2[i], y2[rest])
        wh = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
        inter = wh
        union = areas[i] + areas[rest] - inter
        iou = inter / union.clamp(min=1e-6)
        order = rest[iou <= thr]
    return torch.as_tensor(keep, dtype=torch.long, device=boxes_xyxy.device)
