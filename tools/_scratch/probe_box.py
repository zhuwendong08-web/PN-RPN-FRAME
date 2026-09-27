"""最佳 IoU 只有 0.53，到底是框的位置没学，还是尺寸没学？

逐个 GT 找出 IoU 最大的那个预测框，把两边数值并排打印：
中心距（归一化）和 宽高比。如果宽高比接近 1 而中心差得多 → 位置问题；
如果中心接近而宽高比差得多 → 尺寸问题。
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

    d_center, ratio_gt, ratio_pd, ious = [], [], [], []
    rows = []
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            images = batch["image"].to(device)
            gt = batch["boxes"].to(device)[0][batch["gt_mask"][0]]
            pb, ps = model.inference(images, topk_list=(900,), return_all=True)
            pb = pb[0]
            if gt.shape[0] == 0:
                continue
            iou = box_iou(pb, gt)
            for m in range(gt.shape[0]):
                p = int(iou[:, m].argmax())
                g, q = gt[m], pb[p]
                dc = ((g[0] - q[0]) ** 2 + (g[1] - q[1]) ** 2).sqrt().item()
                rg = float(g[2] / g[3])
                rp = float(q[2] / q[3])
                d_center.append(dc)
                ratio_gt.append(rg)
                ratio_pd.append(rp)
                ious.append(float(iou[p, m]))
                if bi < 6 and m < 2:
                    rows.append((bi, m, iou[p, m].item(), dc, g.tolist(), q.tolist(), rg, rp))

    med = lambda xs: sorted(xs)[len(xs) // 2]  # noqa: E731
    print(f"权重={ck_path}  样本={len(ious)} 对")
    print(f"{'最佳IoU中位':>12}{'中心距中位':>12}{'宽高比GT中位':>14}{'宽高比预测中位':>15}")
    print(f"{med(ious):>12.4f}{med(d_center):>12.4f}"
          f"{med(ratio_gt):>14.2f}{med(ratio_pd):>15.2f}")
    print()
    print("宽高比绝对误差中位 =", f"{med([abs(a - b) for a, b in zip(ratio_gt, ratio_pd)]):.2f}")
    print()
    print(f"{'图':>4}{'GT号':>5}{'IoU':>8}{'中心距':>8}"
          f"{'GT cx,cy,w,h':>26}{'预测 cx,cy,w,h':>26}{'rg':>7}{'rp':>7}")
    for bi, m, v, dc, g, q, rg, rp in rows:
        gs = ",".join(f"{x:.3f}" for x in g)
        qs = ",".join(f"{x:.3f}" for x in q)
        print(f"{bi:>4}{m:>5}{v:>8.3f}{dc:>8.4f}{gs:>26}{qs:>26}{rg:>7.2f}{rp:>7.2f}")


if __name__ == "__main__":
    main()
