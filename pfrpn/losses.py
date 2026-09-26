"""PF-RPN 的完整目标函数（论文 Section 3.5，Eq.5）::

    L = L_reg + L_cls + L_rt + lambda * L_ctr

* ``L_reg`` : L1 + GIoU（仅作用于匈牙利匹配成功的 query）
* ``L_cls`` : 对比损失——query 与可学习嵌入 F~_4^T 的点积做单类（object）打分
* ``L_rt``  : MoE 均衡辅助损失 std(w_i)，防止少数特征级被过度激活
* ``L_ctr`` : 中心度损失 sum_i ||g_i - c_i||_1，作用于全部 N 个 query

匹配用 DETR 风格的匈牙利匹配，代价由分类分数、L1 框距离、GIoU、中心度目标加权构成。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils.box import box_giou, box_iou, centerness_target, hungarian_match


def _focal_with_weight(
    logits: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    alpha: float,
    gamma: float,
) -> torch.Tensor:
    """带样本权重的 focal loss，按权重和归一。

    ``FL(p_t) = -alpha_t * (1-p_t)^gamma * log(p_t)``，``p_t`` 只对正样本算正
    概率、对负样本算 1-负概率（标准 focal 的写法）。外层再乘 ``weight`` 是为了
    兼容原来的 pos/neg 权重配置。
    """
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p_t = p * target + (1.0 - p) * (1.0 - target)
    mod = (1.0 - p_t).clamp(min=0.0).pow(gamma)
    a_t = torch.where(target.bool(), alpha, 1.0 - alpha)
    return (ce * mod * a_t * weight).sum() / weight.sum().clamp(min=1e-6)


class PFRPNLoss(nn.Module):
    def __init__(
        self,
        loss_bbox: float = 5.0,
        loss_giou: float = 2.0,
        lam_ctr: float = 5.0,      # 论文 Table 8 最优 = 5
        loss_moe: float = 1.0,
        cls_pos_weight: float = 1.0,
        cls_neg_weight: float = 0.2,
        use_moe: bool = True,
        use_focal: bool = False,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        ctr_pos_only: bool = False,
        cls_pos_only: bool = False,
    ) -> None:
        super().__init__()
        self.loss_bbox = loss_bbox
        self.loss_giou = loss_giou
        self.lam_ctr = lam_ctr
        self.loss_moe = loss_moe
        self.cls_pos_weight = cls_pos_weight
        self.cls_neg_weight = cls_neg_weight
        self.use_moe = use_moe
        self.use_focal = use_focal
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.ctr_pos_only = ctr_pos_only
        self.cls_pos_only = cls_pos_only

    def forward(
        self,
        pred_boxes: torch.Tensor,     # (B, N, 4) 归一化 cxcywh
        cls_logits: torch.Tensor,     # (B, N)  与可学习嵌入的点积
        ctr_pred: torch.Tensor,       # (B, N)
        router_w: torch.Tensor | None,  # (B, L)
        gt_boxes: torch.Tensor,       # (B, M, 4) 归一化 cxcywh（可能已 padding）
        gt_mask: torch.Tensor | None,  # (B, M) bool，False 的位置是 padding，要排除
    ) -> dict[str, torch.Tensor]:
        B, N = pred_boxes.shape[:2]

        losses: dict[str, torch.Tensor] = {}
        reg_l1_all, reg_giou_all, cls_all, ctr_all, ctr_pos_all = [], [], [], [], []

        for b in range(B):
            gt = gt_boxes[b]
            if gt_mask is not None:
                # 只保留真实框，padding 行不参与匹配，否则会被匈牙利算法
                # 当作"必须匹配的目标"，凭空造出 NaN/错配。
                gt = gt[gt_mask[b]]
            M = gt.shape[0]
            if M == 0:
                # 没有 GT：全部 query 都当负样本
                neg = torch.zeros(N, dtype=torch.bool, device=pred_boxes.device)
                cls_all.append(neg)
                ctr_all.append(torch.zeros(N, device=ctr_pred.device))
                continue

            giou = box_giou(pred_boxes[b], gt)                    # (N, M)
            l1 = (pred_boxes[b].unsqueeze(1) - gt.unsqueeze(0)).abs().sum(-1)
            ctr_t = centerness_target(pred_boxes[b], gt)          # (N, M)

            # DETR 匹配代价：越高越该被匹配
            cost = (
                -torch.sigmoid(cls_logits[b]).unsqueeze(1)
                + self.loss_bbox * l1
                + (1.0 - giou)
                - self.lam_ctr * ctr_t
            )
            rows, cols = hungarian_match(cost)
            is_pos = torch.zeros(N, dtype=torch.bool, device=pred_boxes.device)
            is_pos[rows] = True

            reg_l1_all.append(l1[rows, cols].mean())
            reg_giou_all.append((1.0 - giou[rows, cols]).mean())
            cls_all.append(is_pos)
            ctr_t_match = torch.zeros(N, device=ctr_pred.device)
            ctr_t_match[rows] = ctr_t[rows, cols]
            ctr_all.append(ctr_t_match)
            if self.ctr_pos_only:
                # ⚠️ 只统计匹配成功的正样本，并且**按正样本数归一化**。
                #
                # 这里踩过一个很难看出来的坑：最初写成
                #     (ctr_pred - ctr_t_match).abs().sum() / N
                # 分母是全部 N=100 个 query。负样本的目标虽然恒为 0，但 ctr_pred
                # 并不精确等于 0（有 ±0.05 的残余），于是它们贡献了 loss_ctr 约 80%
                # 的数值；而 L1 在 ctr_pred 附近的梯度符号取决于那一小撮残余的正负，
                # 纯随机。实测 32 个 batch 里 ``cgqs.net.2.bias`` 的梯度符号
                # 同号比例只有 0.44（≈抛硬币），Adam 又是按符号更新的，
                # **净漂移 ≈ 0** —— 中心度头一步都没动，``cgqs.net.2.bias``
                # 训练完仍是初始值 -0.0504，``sigmoid(ctr)`` 于是永远是常数 0.5。
                #
                # 按正样本数归一化之后，每一项的目标都是它自己的 ctr_t（均值约 0.43，
                # 明显大于 ctr_pred≈0），符号恒为"往上抬"，Adam 才推得动。
                npos = int(is_pos.sum())
                if npos:
                    sel = is_pos.nonzero(as_tuple=True)[0]
                    ctr_pos_all.append(
                        (ctr_pred[b][sel] - ctr_t_match[sel]).abs().sum() / npos
                    )

        def _mean(xs):
            return torch.stack(xs).mean() if xs else pred_boxes.new_zeros(())

        losses["loss_bbox"] = _mean(reg_l1_all) * self.loss_bbox
        losses["loss_giou"] = _mean(reg_giou_all) * self.loss_giou

        # 对比损失：正样本 logit -> 1，负样本 -> 0（单类 prompt-free 设置）
        pos_flags = torch.stack(cls_all) if cls_all else torch.zeros_like(cls_logits, dtype=torch.bool)
        cls_t = torch.where(pos_flags, torch.ones_like(cls_logits), torch.zeros_like(cls_logits))
        w = torch.where(pos_flags, self.cls_pos_weight, self.cls_neg_weight)
        if self.cls_pos_only:
            # 只在匈牙利匹配成功的 query 上算分类损失（DETR / ATSS 的常规做法）。
            # 动机：一张图约 4 正 : 96 负，负样本目标恒为 0，几乎每个 step 都在
            # 把它们往下压；正样本那 4 个的梯度完全淹没在里面。
            # v2 试过 focal（alpha=0.25）压负样本，实测**每图 logit>0 的个数依然是
            # 0** —— 说明靠"降低负样本权重"这条路走不通，负样本不该参与损失。
            # 保险：整张图一个人都没有时 pos_flags 全 False，BCE 会 NaN，
            # 退化成"不产生梯度"即可。
            if bool(pos_flags.any()):
                losses["loss_cls"] = F.binary_cross_entropy_with_logits(
                    cls_logits[pos_flags], cls_t[pos_flags], reduction="mean"
                )
            else:
                losses["loss_cls"] = cls_logits.sum() * 0.0
        elif self.use_focal:
            # 极不平衡（COCO person 每图约 4 正 : 96 负）下，普通加权 BCE 会把
            # **全部** 100 个 logit 一起压成负数——实测每图 logit>0 的个数恒为 0。
            # 分数全为负时 sigmoid(clf) 全部落在 (0, 0.5)，排序退化成"哪个
            # query 的下界最大"，于是每张图的 top1 都是同一批 query 序号，
            # 框自然全挤在同一个角落。focal loss 用 (1-p_t)^gamma 压掉易分负样本，
            # 正样本才抬得起来。
            losses["loss_cls"] = _focal_with_weight(
                cls_logits, cls_t, w, self.focal_alpha, self.focal_gamma
            )
        else:
            losses["loss_cls"] = F.binary_cross_entropy_with_logits(
                cls_logits, cls_t, weight=w, reduction="mean"
            )

        # 中心度损失
        ctr_t_cat = torch.stack(ctr_all) if ctr_all else torch.zeros_like(ctr_pred)
        if self.ctr_pos_only:
            # 只作用在匹配成功的 query 上。负样本目标恒为 0，若把它们也算进均值，
            # 中心度头会被整体推到原始输出 0 —— 而 sigmoid(0)=0.5，于是
            # sigmoid(ctr) 变成一个常数 0.5，**中心度对排序不再提供任何信息**。
            # FCOS / ATSS 都是这么做的：centerness 只在正样本上监督。
            # 某张图一个人都没有时 ctr_pos_all 里会是 nan，先过滤掉。
            finite = [t for t in ctr_pos_all if torch.isfinite(t)]
            # 退化分支（某 batch 里所有图都没有 GT）造一个干净的常量 0。
            # 别写成 ``ctr_pred.sum() * 0.0``：在真实训练里 ctr_pred 来自模型、
            # 天然带 grad_fn，两者等价；但一旦 ctr_pred 本身不带 grad（自检脚本
            # 里的随机张量就是），``loss_total`` 会被一个常量污染成
            # requires_grad=False，``.backward()`` 报
            # ``element 0 ... does not require grad``，而报错指不到根因是这个分支。
            losses["loss_ctr"] = (_mean(ctr_pos_all) if finite
                                  else ctr_pred.new_zeros(())) * self.lam_ctr
        else:
            diff = (ctr_pred - ctr_t_cat).abs().sum(1)
            losses["loss_ctr"] = diff.mean() / max(N, 1) * self.lam_ctr

        if self.use_moe and router_w is not None:
            losses["loss_moe"] = router_w.std(-1).mean() * self.loss_moe
        else:
            losses["loss_moe"] = pred_boxes.new_zeros(())

        losses["loss_total"] = sum(losses.values())
        losses["pos_ratio"] = pos_flags.float().mean() if cls_all else torch.zeros(())
        return losses
