"""公共工具：配置加载、随机种子、评测指标（AR / AP）。"""
from __future__ import annotations

import importlib.util
import random
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent

# 归一化面积分档阈值：按 COCO 的 32^2 / 96^2 像素、以 224 为基准换算
AREA_SMALL = 0.02
AREA_MED = 0.18


def load_config(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    spec = importlib.util.spec_from_file_location("cfg_module", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


# --------------------------------------------------------------------------- #
# 评测：Average Recall（论文主指标）与 AP50
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate_proposals(
    model,
    dataloader,
    topk_list: Sequence[int] = (100, 300, 900),
    iou_thr: float = 0.5,
    device: torch.device | str = "cpu",
) -> Dict[str, float]:
    """在提案集合上计算 AR@K / AP50。

    AR@K：每张图按分数降序取前 K 个提案，用贪心匹配（IoU >= iou_thr）统计召回，
          再对所有图取平均。这就是论文 Table 1/2 里的 AR100 / AR300 / AR900 口径。
    """
    from pfrpn.utils.box import box_iou

    model.eval()
    max_k = max(topk_list) if topk_list else 0
    buckets: Dict[str, List[float]] = {
        f"AR@{k}": [] for k in topk_list
    }
    buckets["ARs"] = []
    buckets["ARm"] = []
    buckets["ARl"] = []
    buckets["AP50"] = []

    for batch in dataloader:
        images = batch["image"].to(device)
        gts = batch["boxes"].to(device)
        # dataset 的 collate 会把框数不齐的 batch 左对齐 padding。
        # 评测时必须按 mask 只留真实框，否则 padding 行会被算进 recall 的分母，
        # AR 会凭空被稀释掉一截。
        gtm = batch["gt_mask"].to(device) if "gt_mask" in batch else None
        # 一次前向拿全量提案，AR@K 各自切片；否则 num_queries < 900 时
        # 所有 AR@K 都会退化成 AR@num_queries。
        pred_boxes, pred_scores = model.inference(
            images, topk_list=(max_k,), return_all=True
        )

        for i in range(images.shape[0]):
            gt = gts[i] if gtm is None else gts[i][gtm[i]]
            M = gt.shape[0]
            pb_all, ps_all = pred_boxes[i], pred_scores[i]
            order = torch.argsort(ps_all, descending=True)
            pb_all, ps_all = pb_all[order], ps_all[order]

            if M == 0:
                for k in topk_list:
                    buckets[f"AR@{k}"].append(0.0)
                for name in ("ARs", "ARm", "ARl", "AP50"):
                    buckets[name].append(0.0)
                continue

            for k in topk_list:
                matched = torch.zeros(M, dtype=torch.bool)
                buckets[f"AR@{k}"].append(_match_recall(pb_all[:k], gt, matched, iou_thr))

            # 按 GT 面积分档（小 / 中 / 大），对应论文的 ARs / ARm / ARl
            areas = gt[:, 2] * gt[:, 3]
            bins = (
                ("ARs", areas < AREA_SMALL),
                ("ARm", (areas >= AREA_SMALL) & (areas < AREA_MED)),
                ("ARl", areas >= AREA_MED),
            )
            for name, mask in bins:
                if not mask.any():
                    continue
                sub = gt[mask]
                matched = torch.zeros(sub.shape[0], dtype=torch.bool)
                buckets[name].append(_match_recall(pb_all[:max_k], sub, matched, iou_thr))

            matched = torch.zeros(M, dtype=torch.bool)
            buckets["AP50"].append(_ap50(pb_all, ps_all, gt, matched, iou_thr))

    # ⚠️ ``if v else 0.0`` 这个兜底曾经坑过人：val 集被 id_offset 切空时，
    # 每个 bucket 都是空 list，于是静默返回 ``AR@100 = 0.0000``——看着像
    # "模型完全没学会"，实际是一次都没评测过。而且它会让 train.py 里
    # ``res[key] > best(-1.0)`` 成立，用 epoch1 的权重覆盖 best.pth。
    # 宁可报错也不能再让空评测蒙混过关。
    if not buckets["AP50"]:
        raise RuntimeError(
            "评测跑在空数据集上（一张图都没评测）。检查验证集是否被 "
            "id_offset / n_images_cap 切空了。"
        )
    out = {k: (sum(v) / len(v) if v else 0.0) for k, v in buckets.items()}
    return out


def _match_recall(
    pred: torch.Tensor, gt: torch.Tensor, matched: torch.Tensor, thr: float
) -> float:
    """按 GT 顺序贪心匹配（IoU >= thr 且该 GT 尚未被匹配），返回召回率。"""
    from pfrpn.utils.box import box_iou

    if pred.shape[0] == 0:
        return 0.0
    iou = box_iou(pred, gt)                       # (P, M)
    hit = 0
    for m in range(gt.shape[0]):
        if matched[m]:
            continue
        best = int(iou[:, m].argmax())
        if iou[best, m] >= thr:
            matched[m] = True
            hit += 1
    return hit / max(gt.shape[0], 1)


def _ap50(
    pred_boxes: torch.Tensor,
    scores: torch.Tensor,
    gt: torch.Tensor,
    matched: torch.Tensor,
    thr: float,
) -> float:
    """AP50：预测按分数降序，每个 GT 只认一个最大 IoU 的预测；101 点插值。"""
    from pfrpn.utils.box import box_iou

    M = gt.shape[0]
    if M == 0:
        return 0.0
    used = torch.zeros(pred_boxes.shape[0], dtype=torch.bool)
    iou = box_iou(pred_boxes, gt)
    tp = np.zeros(pred_boxes.shape[0], dtype=np.float64)
    for m in range(M):
        if matched[m]:
            continue
        for p in torch.argsort(-iou[:, m]).tolist():
            if used[p] or iou[p, m] < thr:
                continue
            used[p] = True
            tp[p] = 1.0
            break
    total = int(used.sum())
    if total == 0:
        return 0.0
    cum = np.cumsum(tp)
    prec = cum / np.arange(1, tp.size + 1)          # 递增精确率
    recall = cum / max(gt.shape[0], 1)              # 递增召回率

    # ⚠️ 这里是 COCO 的 101 点插值 AP：对 0..1 等分 101 个召回档，每个档取
    #    「召回不低于该档」范围内**最大的**精确率，最后求均值。
    #
    # 原来的写法是
    #     np.mean(prec[:: max(1, (tp.size - 1) // 100 + 1)])
    # 即直接对递增精确率等距采样——少了单调化这一步，是它的悲观近似，会
    # 把 AP 系统性低报（本项目实测 AP50 0.0322 -> 修正后 0.10 量级）。
    # 注意口径修正是**指标变准**，不是模型变好。
    P = np.zeros(101, dtype=np.float64)
    for i, r in enumerate(np.linspace(0.0, 1.0, 101)):
        m = recall >= (r - 1e-12)
        P[i] = prec[m].max() if m.any() else 0.0
    return float(P.mean())


def format_results(res: Dict[str, float]) -> str:
    return "  ".join(f"{k}={v:.4f}" for k, v in res.items())
