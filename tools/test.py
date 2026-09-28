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

from pfrpn.datasets import collate                          # noqa: E402
from pfrpn.model import PFRPN                              # noqa: E402
from tools.common import (                                 # noqa: E402
    build_dataset,
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

    # ⚠️ 数据集构建统一走 tools/common.py::build_dataset，与 train.py 完全一致。
    # 曾经这里自己拼参数、漏了 4 个，把真实 COCO 人权重的权重评到合成
    # data/val.json 上（AR@100 0.6050 -> 0.4180），差点当成模型退化。
    ds = build_dataset(cfg, args.split)
    loader = DataLoader(ds, batch_size=max(cfg["batch_size"], 1), shuffle=False,
                        collate_fn=collate)
    topk = tuple(int(x) for x in args.topk.split(",")) if args.topk else tuple(cfg.get("topk_list", (100,)))

    res = evaluate_proposals(model, loader, topk_list=topk, device=device)
    print("[result] " + format_results(res))


if __name__ == "__main__":
    main()
