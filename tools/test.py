"""在验证集上评测（AR@K / AP50）。

    python tools/test.py --config configs/pf_rpn_tiny.py --checkpoint outputs/best.pth
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pfrpn.datasets import COCOLikeDataset, collate        # noqa: E402
from pfrpn.model import PFRPN                              # noqa: E402
from tools.common import (                                 # noqa: E402
    evaluate_proposals,
    format_results,
    load_config,
    resolve_device,
    set_seed,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pf_rpn_tiny.py")
    ap.add_argument("--checkpoint", default="outputs/best.pth")
    ap.add_argument("--split", default="val")
    ap.add_argument("--device", default=None)
    ap.add_argument("--topk", default=None, help="逗号分隔，如 100,300")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg["device"] = args.device or cfg.get("device", "auto")
    device = resolve_device(cfg["device"])
    set_seed(cfg.get("seed", 0))

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = PFRPN(ckpt.get("cfg", cfg)).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"[load] {args.checkpoint}")

    dcfg = cfg["data"]
    ds = COCOLikeDataset(
        root=dcfg["root"], split=args.split, img_size=dcfg["img_size"],
        max_objs=dcfg["max_objs"], seed=cfg.get("seed", 0) + 7,
        n_images=dcfg.get("val_n", 128),
    )
    loader = DataLoader(ds, batch_size=max(cfg["batch_size"], 1), shuffle=False,
                        collate_fn=collate)
    topk = tuple(int(x) for x in args.topk.split(",")) if args.topk else tuple(cfg.get("topk_list", (100,)))

    res = evaluate_proposals(model, loader, topk_list=topk, device=device)
    print("[result] " + format_results(res))


if __name__ == "__main__":
    main()
