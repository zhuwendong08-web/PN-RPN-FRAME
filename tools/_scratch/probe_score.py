"""v4 修好中心度之后，推理打分公式还该是 sigmoid(cls)*sigmoid(ctr) 吗？

v3 时代实测过「只用分类分 top1 IoU 0.128 > 两者相乘 0.078」，当时归因为
中心度塌成常数。v4 把中心度救活了（范围从死死的 0.5 变成 0.575~0.721），
这个结论可能已经不成立了。本脚本把五种打分口径摊开对比。

用法:
    python tools/_scratch/probe_score.py [checkpoint.pth]
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

IOU_THR = 0.5
VARIANTS = ("乘积 cls*ctr", "只分类 cls", "只中心度 ctr",
            "cls^2*ctr", "cls+ctr 平均")


def recall_at_k(pb, ps, gt, k):
    """按分数降到前 k，算一对一召回。"""
    if pb.shape[0] == 0:
        return 0.0
    order = torch.argsort(ps, descending=True)[:k]
    pb = pb[order]
    iou = box_iou(pb, gt)
    used = torch.zeros(pb.shape[0], dtype=torch.bool)
    hit = 0
    for m in range(gt.shape[0]):
        cand = iou[:, m].masked_fill(used, -1.0)
        best = int(cand.argmax())
        if cand[best] >= IOU_THR:
            used[best] = True
            hit += 1
    return hit / max(gt.shape[0], 1)


def best_iou_median(pb, gt):
    if pb.shape[0] == 0:
        return 0.0
    return float(box_iou(pb, gt).max(dim=0).values.median())


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

    # rec[name][k] = List[float]，逐图记录，最后取均值
    rec = {name: {k: [] for k in (1, 10, 100)} for name in VARIANTS}
    deg = {k: [] for k in
           ("cls_uniq", "ctr_uniq", "prod_uniq",
            "prod_uniq_top100", "prod_std")}
    lo, hi, iou_med = [], [], []
    n = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            gt = batch["boxes"].to(device)[0][batch["gt_mask"][0]]
            out = model(images)
            cls_s = torch.sigmoid(out["cls_logits"][0])
            ctr_s = torch.sigmoid(out["ctr"][0])
            pb = out["boxes"][0]
            n += 1
            lo.append(float(ctr_s.min()))
            hi.append(float(ctr_s.max()))
            iou_med.append(best_iou_median(pb, gt))

            v = {
                "乘积 cls*ctr": cls_s * ctr_s,
                "只分类 cls": cls_s,
                "只中心度 ctr": ctr_s,
                "cls^2*ctr": cls_s * cls_s * ctr_s,
                "cls+ctr 平均": (cls_s + ctr_s) / 2,
            }
            for name, s in v.items():
                for k in (1, 10, 100):
                    rec[name][k].append(recall_at_k(pb, s, gt, k))
            # 退化诊断：这些分数到底有没有区分度？
            deg["cls_uniq"].append(int(cls_s.unique().numel()))
            deg["ctr_uniq"].append(int(ctr_s.unique().numel()))
            prod = v["乘积 cls*ctr"]
            deg["prod_uniq"].append(int(prod.unique().numel()))
            deg["prod_uniq_top100"].append(
                int(prod.topk(min(100, prod.numel())).values.unique().numel()))
            deg["prod_std"].append(float(prod.std()))

    mean = lambda xs: sum(xs) / len(xs)          # noqa: E731
    median = lambda xs: sorted(xs)[len(xs) // 2]  # noqa: E731
    print(f"权重={ck_path}   验证图数={n}   IoU 阈值={IOU_THR}")
    print(f"中心度 sigmoid 范围: 最小 {min(lo):.4f} ~ 最大 {max(hi):.4f}")
    print(f"全 100 提案最佳 IoU 中位 = {mean(iou_med):.4f}")
    print()
    print(f"{'打分口径':<16}{'AR@100':>9}{'AR@10':>9}{'AR@1':>9}")
    for name in VARIANTS:
        print(f"{name:<16}"
              f"{mean(rec[name][100]):>9.4f}"
              f"{mean(rec[name][10]):>9.4f}"
              f"{mean(rec[name][1]):>9.4f}")
    for k in ("cls_uniq", "ctr_uniq", "prod_uniq", "prod_uniq_top100"):
        print(f"{k:<18}每张图唯一值数 中位={median(deg[k])}  均值={mean(deg[k]):.1f}")
    print(f"{"prod_std":<18}标准差 中位={median(deg['prod_std']):.6f}")

    print()
    base = mean(rec["乘积 cls*ctr"][100])
    for name in VARIANTS:
        d = mean(rec[name][100]) - base
        print(f"{name:<16}相对乘积 {'+' if d >= 0 else ''}{d:+.4f}")


if __name__ == "__main__":
    main()
