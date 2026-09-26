"""诊断「框全落在角落」到底是谁的问题。

三类可能，得用数据区分开：

1. 数据本身：COCO person 的标注框本来就集中在画面边缘 → 模型没错，是图错了；
2. 模型偏置：提案分布有位置偏好（DETR 系 query 位置先验没学好会这样）；
3. 排序问题：提案质量没差（IoU 高），但高分的恰好在角落 → 是打分函数的问题，
   不是定位的问题。

用法::

    python tools/diag_corner.py --ckpt outputs_person/best.pth
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pfrpn.datasets import COCOLikeDataset, collate      # noqa: E402
from pfrpn.model import PFRPN                            # noqa: E402
from pfrpn.utils.box import box_cxcywh_to_xyxy           # noqa: E402
from tools.common import load_config                     # noqa: E402

BIN = 4    # 把画面切成 4x4 = 16 个格子统计框中心落点


def hist(centers: np.ndarray) -> str:
    """16 格的落点分布，第 0 格是左上角。"""
    if len(centers) == 0:
        return "(无)"
    gx = np.clip((centers[:, 0] * BIN).astype(int), 0, BIN - 1)
    gy = np.clip((centers[:, 1] * BIN).astype(int), 0, BIN - 1)
    cnt = Counter(zip(gy, gx))
    edge_hits = sum(v for (gy_, gx_) in cnt
                    for v in [cnt[(gy_, gx_)]] if gx_ == 0 or gx_ == BIN - 1
                    or gy_ == 0 or gy_ == BIN - 1)
    rows = []
    for r in range(BIN):
        rows.append("  " + "".join(
            f"{cnt.get((r, c), 0):5d}" for c in range(BIN)))
    return ("落点分布(行=上->下, 列=左->右)\n"
            + "\n".join(rows)
            + f"\n  靠边格占比 {edge_hits / len(centers) * 100:.1f}%  "
            f"n={len(centers)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs_person/best.pth")
    ap.add_argument("--config", default="configs/pf_rpn_person.py")
    ap.add_argument("--val-n", type=int, default=200)
    args = ap.parse_args()

    cfg = load_config(args.config)
    dc = cfg["data"]
    ds = COCOLikeDataset(
        ROOT / dc["root"], split="val", img_size=dc["img_size"],
        max_objs=dc["max_objs"], img_dir=dc["img_dir"],
        bbox_fmt=dc["bbox_fmt"], json_file=dc["json_file"],
        # 同 tools/train.py：只有真实数据集（有 json_file）才用 train_n 当偏移。
        # 合成数据的 val 图不是按 [train_n:] 编排的，偏移会把验证集切空 ——
        # 空数据集不会报错，只会让 AR 静默变成 0，见 README 第 23 条。
        n_images_cap=args.val_n,
        id_offset=dc.get("val_offset", dc.get("train_n", 0) if dc.get("json_file") else 0),
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate)

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = PFRPN(ck.get("cfg", cfg)).to("cpu")
    model.load_state_dict(ck["model"])
    model.eval()

    gt_c, t1_c, t20_c, best_c = [], [], [], []
    score_list, iou_t1_list, iou_best_list = [], [], []
    area_t1, area_gt = [], []

    for batch in loader:
        img = batch["image"]
        with torch.no_grad():
            boxes, scores = model.inference(img, topk_list=(100,), return_all=True)
        pb = box_cxcywh_to_xyxy(boxes[0]).numpy()      # (100,4) xyxy
        sc = scores[0].numpy()

        gt = batch["boxes"][0].numpy()
        if "gt_mask" in batch:
            gt = gt[batch["gt_mask"][0].numpy()]
        if not len(gt):
            continue
        gx1 = gt[:, 0] - gt[:, 2] / 2; gy1 = gt[:, 1] - gt[:, 3] / 2
        gx2 = gt[:, 0] + gt[:, 2] / 2; gy2 = gt[:, 1] + gt[:, 3] / 2
        gt_xyxy = np.stack([gx1, gy1, gx2, gy2], axis=1)

        gt_c.append(gt[:, :2])
        area_gt.extend((gt[:, 2] * gt[:, 3]).tolist())

        # 与每个真值框的最大 IoU
        per = []
        for k in range(pb.shape[0]):
            px1, py1, px2, py2 = pb[k]
            ix1 = np.maximum(px1, gx1); iy1 = np.maximum(py1, gy1)
            ix2 = np.minimum(px2, gx2); iy2 = np.minimum(py2, gy2)
            iw = np.maximum(0, ix2 - ix1); ih = np.maximum(0, iy2 - iy1)
            inter = iw * ih
            ua = (px2 - px1) * (py2 - py1) + (gx2 - gx1) * (gy2 - gy1) - inter
            per.append(float((inter / np.maximum(ua, 1e-9)).max()))
        per = np.asarray(per)

        cx = (pb[:, 0] + pb[:, 2]) / 2
        cy = (pb[:, 1] + pb[:, 3]) / 2
        t1_c.append([[cx[0], cy[0]]])
        t20_c.append(np.stack([cx, cy], axis=1))
        bi = int(per.argmax())
        best_c.append([[cx[bi], cy[bi]]])
        score_list.append(sc[0].reshape(-1))
        iou_t1_list.append(per[0])
        iou_best_list.append(per[bi])
        area_t1.append((pb[0, 2] - pb[0, 0]) * (pb[0, 3] - pb[0, 1]))

    gt_c = np.vstack(gt_c); t1_c = np.vstack(t1_c)
    t20_c = np.vstack(t20_c); best_c = np.vstack(best_c)
    score_list = np.concatenate(score_list)
    iou_t1 = np.asarray(iou_t1_list); iou_best = np.asarray(iou_best_list)
    area_t1 = np.asarray(area_t1); area_gt = np.asarray(area_gt)

    print("=" * 62)
    print("【1】真值框（COCO person 标注）的落点")
    print(hist(gt_c))
    print("=" * 62)
    print("【2】模型 top1 提案的落点（按分数排第一的那个）")
    print(hist(t1_c))
    print("=" * 62)
    print("【3】模型全部 100 个提案的落点")
    print(hist(t20_c))
    print("=" * 62)
    print("【4】100 个提案里 IoU 最大的那个的落点")
    print(hist(best_c))
    print("=" * 62)
    print(f"top1 框面积 中位 {np.median(area_t1):.3f}   "
          f"真值框面积 中位 {np.median(area_gt):.3f}")
    print(f"IoU 与分数秩相关: "
          f"{np.corrcoef(score_list, iou_t1)[0, 1]:+.3f}  "
          f"(接近 0 说明打分和定位无关)")

    # 关键判据：如果把 top1 换成"IoU 最大的那个"，效果提升多少
    print("-" * 62)
    print(f"top1 IoU 中位          {np.median(iou_t1):.3f}")
    print(f"100 提案里最佳 IoU 中位 {np.median(iou_best):.3f}")
    print(f"若改成取最佳提案，IoU 中位提升 "
          f"{np.median(iou_best) - np.median(iou_t1):+.3f}")
    print("=" * 62)


if __name__ == "__main__":
    main()
