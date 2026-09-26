"""train.py 和 test.py 的 val 集不一样，这很可能就是 0.0000 的来源。

train.py: COCOLikeDataset(n_images=dcfg["n_images"]=512, seed=seed+7)
test.py : COCOLikeDataset(n_images=dcfg["val_n"]    =128, seed=seed+7)
两者都走 split="val"。本脚本只构造数据集，不训练，直接比 AR。
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from pfrpn.datasets import COCOLikeDataset, collate  # noqa: E402
from pfrpn.model import PFRPN  # noqa: E402
from tools.common import (  # noqa: E402
    evaluate_proposals, format_results, load_config, resolve_device, set_seed,
)


def main() -> None:
    cfg = load_config("configs/pf_rpn_tiny.py")
    set_seed(cfg.get("seed", 0))
    device = resolve_device(cfg.get("device", "auto"))

    ckpt = torch.load("outputs/best.pth", map_location=device, weights_only=False)
    model = PFRPN(ckpt.get("cfg", cfg)).to(device)
    model.load_state_dict(ckpt["model"])
    print("[load] outputs/best.pth")

    dcfg = cfg["data"]
    variants = [
        ("test.py   n_images=val_n =128", 128),
        ("train.py  n_images        =512", 512),
    ]
    for label, n in variants:
        ds = COCOLikeDataset(
            root=dcfg["root"], split="val", img_size=dcfg["img_size"],
            max_objs=dcfg["max_objs"], seed=cfg.get("seed", 0) + 7, n_images=n,
        )
        loader = DataLoader(ds, batch_size=max(cfg["batch_size"], 1),
                            shuffle=False, collate_fn=collate)
        # 统计每张图的 GT 框数
        from pfrpn.datasets import collate as _c
        nb = 0
        counts = []
        for b in loader:
            counts.append(b["gt_mask"].sum(dim=1).tolist())
            nb += b["gt_mask"].sum(dim=0).count_nonzero().item()
        flat = [c for row in counts for c in row]
        print(f"\n===== {label}  图片数={len(ds)}  总框数={nb}  "
              f"每图框数 min={min(flat)} 中位={sorted(flat)[len(flat)//2]} max={max(flat)}")
        res = evaluate_proposals(model, loader, topk_list=(100,), device=device)
        print("[result] " + format_results(res))


if __name__ == "__main__":
    main()
