"""量化"排序烂"到底损失了多少：top-K IoU 表 + NMS 的效果。

背景：微调后的模型定位能力尚可（100 个提案里最佳 IoU 中位数 0.65），但
top1 几乎打不中人（IoU 中位数 0.088）。所以问题在**排序**，不在定位。
这个脚本回答一个很实际的问题：**不能只信 top1 的话，取前 K 个能到什么水平？
加上 NMS 去重后呢？**

用法::

    python tools/diag_rank.py --ckpt outputs_person_v2/best.pth
    python tools/diag_rank.py --ckpt outputs_person/best.pth --tag v1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pfrpn.datasets import COCOLikeDataset, collate     # noqa: E402
from pfrpn.model import PFRPN                           # noqa: E402
from pfrpn.utils.box import box_cxcywh_to_xyxy          # noqa: E402
from tools.common import load_config                    # noqa: E402

KS = (1, 2, 3, 5, 8, 10, 20, 50, 100)


def iou_mat(pred_xyxy: torch.Tensor, gt_xyxy: torch.Tensor) -> torch.Tensor:
    k, m = pred_xyxy.shape[0], gt_xyxy.shape[0]
    if k == 0 or m == 0:
        return torch.zeros(k, m)
    # ⚠️ 必须分开取 x1/y1 与 x2/y2。这里最初写成 min(...,gt_x2)-max(...,gt_x1)，
    # 于是"真值框被预测框盖住"这类真正该有交叠的情形反而算出 0 ——
    # 结果整张表全是 0.000，看着像模型完全没学会。
    iw = (torch.min(pred_xyxy[:, None, 2], gt_xyxy[None, :, 2])
          - torch.max(pred_xyxy[:, None, 0], gt_xyxy[None, :, 0])).clamp(min=0)
    ih = (torch.min(pred_xyxy[:, None, 3], gt_xyxy[None, :, 3])
          - torch.max(pred_xyxy[:, None, 1], gt_xyxy[None, :, 1])).clamp(min=0)
    inter = iw * ih
    pa = (pred_xyxy[:, None, 2] - pred_xyxy[:, None, 0]).clamp(min=0) * \
         (pred_xyxy[:, None, 3] - pred_xyxy[:, None, 1]).clamp(min=0)
    ga = (gt_xyxy[None, :, 2] - gt_xyxy[None, :, 0]).clamp(min=0) * \
         (gt_xyxy[None, :, 3] - gt_xyxy[None, :, 1]).clamp(min=0)
    return inter / (pa + ga - inter).clamp(min=1e-6)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--nms", type=float, default=-1.0,
                    help="NMS 阈值；-1 表示不加 NMS")
    args = ap.parse_args()

    cfg = load_config("configs/pf_rpn_person_v2.py")
    torch.manual_seed(cfg.get("seed", 0))
    model = PFRPN(cfg)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu")["model"])
    model.eval()

    dcfg = cfg["data"]
    ds = COCOLikeDataset(
        root=dcfg["root"], split="val", img_size=dcfg["img_size"],
        max_objs=dcfg["max_objs"], seed=dcfg.get("seed", 0),
        img_dir=dcfg.get("img_dir"), bbox_fmt=dcfg.get("bbox_fmt", "cxcywh"),
        json_file=dcfg.get("json_file"),
        n_images_cap=dcfg.get("val_n"), id_offset=dcfg.get("val_offset", 0),
    )
    dl = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate)

    tag = args.tag or Path(args.ckpt).parent.name
    nms_thr = args.nms
    if nms_thr > 0:
        from pfrpn.model import _nms
    else:
        _nms = None

    tops: dict[int, list[float]] = {k: [] for k in KS}
    hits: dict[int, list[bool]] = {k: [] for k in KS}
    areas, gareas = [], []

    with torch.no_grad():
        for batch in dl:
            out = model(batch["image"])
            boxes, scores = out["boxes"][0], out["cls_logits"][0]
            s = torch.sigmoid(scores) * torch.sigmoid(out["ctr"][0])
            order = torch.argsort(s, descending=True)
            order = order[:max(KS)]
            b = boxes[order]
            if _nms is not None:
                keep = _nms(box_cxcywh_to_xyxy(b), s[order], nms_thr)
                b, s2 = b[keep], s[order][keep]
                order_b = torch.argsort(s2, descending=True)
                b = b[order_b]
            gt = batch["boxes"][0][batch["gt_mask"][0]]
            if gt.numel() == 0:
                continue
            gt_xyxy = box_cxcywh_to_xyxy(gt)
            pred_xyxy = box_cxcywh_to_xyxy(b)
            areas.append(float(((pred_xyxy[:, 2] - pred_xyxy[:, 0]).clamp(min=0)
                                 * (pred_xyxy[:, 3] - pred_xyxy[:, 1]).clamp(min=0)).mean()))
            gareas.append(float(((gt_xyxy[:, 2] - gt_xyxy[:, 0]).clamp(min=0)
                                 * (gt_xyxy[:, 3] - gt_xyxy[:, 1]).clamp(min=0)).mean()))

            m = iou_mat(pred_xyxy, gt_xyxy)                     # (K, M)
            for k in KS:
                if k > m.shape[0]:
                    continue
                topk = m[:k].max(dim=1).values                   # (k,) 每个框对最近 GT 的 IoU
                tops[k].append(float(torch.median(topk)))
                hits[k].append(bool((topk >= 0.5).any()))

    print("=" * 64)
    print(f"【{tag}】  NMS={'关' if nms_thr < 0 else f' IoU>{nms_thr}'}   验证图 {len(tops[1])} 张")
    print(f"{'K':>4} {'top-K IoU 中位':>16} {'K 个里至少命中一个(IoU>=0.5)':>30}")
    for k in KS:
        if not tops[k]:
            continue
        print(f"{k:>4} {np.median(tops[k]):>16.3f} "
              f"{np.mean(hits[k]) * 100:>28.1f}%")
    print(f"\ntop1 框面积中位 {np.median(areas):.3f}  真值框面积中位 {np.median(gareas):.3f}")
    print("=" * 64)


if __name__ == "__main__":
    main()
