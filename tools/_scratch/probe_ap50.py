"""AR@100 = 0.73 和 AP50 = 0.03 同时出现，说明其中一个指标算错了。

本脚本逐图把三个数摊开对照：
  M        ground truth 个数
  R@100    _match_recall(pb[:100]) 命中数（可复用同一预测框）
  AP_tot   _ap50 真正标成 tp 的预测框数（严格一对一）
看两者是否自洽，以及 AP50 里被判 tp 的框排在第几位。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from pfrpn.datasets import COCOLikeDataset, collate   # noqa: E402
from pfrpn.model import PFRPN                          # noqa: E402
from pfrpn.utils.box import box_iou                    # noqa: E402
from tools.common import _ap50, _match_recall, load_config, resolve_device, set_seed  # noqa: E402

THR = 0.5


def ap50_detail(pb, ps, gt):
    """复刻 _ap50，但把 tp 的位置也返回出来。"""
    M = gt.shape[0]
    if M == 0:
        return 0.0, []
    used = torch.zeros(pb.shape[0], dtype=torch.bool)
    iou = box_iou(pb, gt)
    tp = np.zeros(pb.shape[0], dtype=np.float64)
    for m in range(M):
        for p in torch.argsort(-iou[:, m]).tolist():
            if used[p] or iou[p, m] < THR:
                continue
            used[p] = True
            tp[p] = 1.0
            break
    total = int(used.sum())
    if total == 0:
        return 0.0, []
    cum = np.cumsum(tp)
    prec = cum / np.arange(1, tp.size + 1)
    sampled = prec[:: max(1, (tp.size - 1) // 100 + 1)]
    return float(np.mean(sampled)), [int(i) for i in np.flatnonzero(tp)]


def main() -> None:
    ck_path = sys.argv[1] if len(sys.argv) > 1 else "outputs/best.pth"
    cfg = load_config("configs/pf_rpn_tiny.py")
    set_seed(cfg.get("seed", 0))
    device = resolve_device(cfg.get("device", "auto"))

    ck = torch.load(ck_path, map_location=device, weights_only=False)
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

    rows = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            gt = batch["boxes"].to(device)[0][batch["gt_mask"][0]]
            pb, ps = model.inference(images, topk_list=(900,), return_all=True)
            pb, ps = pb[0], ps[0]
            order = torch.argsort(ps, descending=True)
            pb, ps = pb[order], ps[order]
            M = gt.shape[0]
            if M == 0:
                continue
            r100 = _match_recall(pb[:100], gt, torch.zeros(M, dtype=torch.bool), THR)
            ap, pos = ap50_detail(pb, ps, gt)
            rows.append((M, int(round(r100 * M)), len(pos), ap, pos))

    n = len(rows)
    print(f"权重={ck_path}   有 GT 的图数={n}")
    print(f"{'图':>4}{'M':>5}{'R@100命中':>10}{'AP50 tp数':>10}{'AP50值':>9}{'tp所在位序(前12)':>26}")
    for i, (M, hit, tot, ap, pos) in enumerate(rows[:12]):
        head = ",".join(str(x) for x in sorted(pos)[:12])
        print(f"{i:>4}{M:>5}{hit:>10}{tot:>10}{ap:>9.4f}{head:>26}")

    avg = lambda f: sum(f(r) for r in rows) / max(n, 1)  # noqa: E731
    print()
    print(f"平均 M        = {avg(lambda r: r[0]):.2f}")
    print(f"平均 R@100 命中 = {avg(lambda r: r[1]):.2f}")
    print(f"平均 AP50 tp数 = {avg(lambda r: r[2]):.2f}")
    print(f"平均 AP50     = {avg(lambda r: r[3]):.4f}")
    diff = [r for r in rows if r[1] - r[2] > 1]
    print(f"R@100 比 AP50_tp 多出 >1 个的图: {len(diff)}/{n}")


if __name__ == "__main__":
    main()
