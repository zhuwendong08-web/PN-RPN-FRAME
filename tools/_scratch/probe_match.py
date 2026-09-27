"""量化 _match_recall 缺少一对一约束造成的虚高。

_match_recall 遍历 GT 时只看「这个 GT 有没有被认领」，不管认领它的那个
预测框是不是已经被别的 GT 用过了。于是同一个预测框可以同时命中多个 GT，
hit 虚高 -> AR 虚高。_ap50 是有 used 数组的（一对一），只有 recall 漏了。
本脚本逐图对比两种口径，看虚了多少。
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from pfrpn.datasets import COCOLikeDataset, collate   # noqa: E402
from pfrpn.model import PFRPN                          # noqa: E402
from pfrpn.utils.box import box_iou                    # noqa: E402
from tools.common import load_config, resolve_device, set_seed  # noqa: E402


def recall_greedy_free(pred, gt, thr):
    """现状：不限制预测框复用（可高估）。"""
    if pred.shape[0] == 0:
        return 0.0
    iou = box_iou(pred, gt)
    hit = 0
    for m in range(gt.shape[0]):
        best = int(iou[:, m].argmax())
        if iou[best, m] >= thr:
            hit += 1
    return hit / max(gt.shape[0], 1)


def recall_one_to_one(pred, gt, thr):
    """修好后：每个预测框只能认领一个 GT。"""
    if pred.shape[0] == 0:
        return 0.0
    iou = box_iou(pred, gt)
    used = torch.zeros(pred.shape[0], dtype=torch.bool)
    hit = 0
    for m in range(gt.shape[0]):
        cand = iou[:, m].masked_fill(used, -1.0)
        best = int(cand.argmax())
        if cand[best] >= thr:
            used[best] = True
            hit += 1
    return hit / max(gt.shape[0], 1)


def main() -> None:
    cfg = load_config("configs/pf_rpn_tiny.py")
    set_seed(cfg.get("seed", 0))
    device = resolve_device(cfg.get("device", "auto"))

    ck = torch.load("outputs/best.pth", map_location=device, weights_only=False)
    model = PFRPN(ck.get("cfg", cfg)).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    dcfg = cfg["data"]
    ds = COCOLikeDataset(
        root=dcfg["root"], split="val", img_size=dcfg["img_size"],
        max_objs=dcfg["max_objs"], seed=cfg.get("seed", 0) + 7,
        n_images=dcfg.get("val_n", 128),
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate)

    per_img = []
    reused = 0          # 有多少图里出现了「一个预测框被多个 GT 认领」
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            gt = batch["boxes"].to(device)[0][batch["gt_mask"][0]]
            pb, ps = model.inference(images, topk_list=(100,), return_all=True)
            pb, ps = pb[0], ps[0]
            order = torch.argsort(ps, descending=True)
            pb = pb[order]

            iou = box_iou(pb, gt)
            # 一个预测框能同时满足几个 GT？>1 说明被复用了
            cols = (iou >= 0.5).sum(dim=0)
            if int((cols > 1).sum()):
                reused += 1
            per_img.append((recall_greedy_free(pb, gt, 0.5),
                            recall_one_to_one(pb, gt, 0.5)))

    n = len(per_img)
    a = sum(x[0] for x in per_img) / n
    b = sum(x[1] for x in per_img) / n
    lose = sum(1 for x in per_img if x[1] < x[0])
    print(f"n={n}")
    print(f"  现状(可复用) 逐图均值 recall@100 = {a:.4f}")
    print(f"  (一对一)     逐图均值 recall@100 = {b:.4f}")
    print(f"  虚高幅度 = {a - b:+.4f}（相对 {(a - b) / max(b, 1e-9) * 100:+.1f}%）")
    print(f"  出现「一个框被多个 GT 认领」的图: {reused}/{n}")
    print(f"  修好之后命中变少的图: {lose}/{n}")


if __name__ == "__main__":
    main()
