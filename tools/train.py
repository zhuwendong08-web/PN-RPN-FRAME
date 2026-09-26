"""训练入口。

    python tools/train.py --config configs/pf_rpn_tiny.py --epochs 4

没有任何数据集时脚本会自动合成一份 COCO 风格数据到 ``data/``。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pfrpn import PFRPN                                    # noqa: E402
from pfrpn.datasets import COCOLikeDataset, collate        # noqa: E402
from pfrpn.losses import PFRPNLoss                         # noqa: E402
from tools.common import (                                 # noqa: E402
    REPO_ROOT,
    evaluate_proposals,
    format_results,
    load_config,
    resolve_device,
    set_seed,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pf_rpn_tiny.py")
    ap.add_argument("--device", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--out", default=None, help="检查点输出目录")
    ap.add_argument("--resume", default=None, help="从已有检查点继续训练")
    ap.add_argument("--force-synth", action="store_true",
                    help="忽略已有的 data/*.json，强制重新合成数据集")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg["device"] = args.device or cfg.get("device", "auto")
    if args.epochs:
        cfg["epochs"] = args.epochs
    if args.batch_size:
        cfg["batch_size"] = args.batch_size
    if args.lr:
        cfg["optimizer"]["lr"] = args.lr

    device = resolve_device(cfg["device"])
    set_seed(cfg.get("seed", 0))
    out_dir = Path(args.out or (REPO_ROOT / cfg.get("out_dir", "outputs")))
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- 数据 ----------------
    dcfg = cfg["data"]
    # 微调真实数据时关键 4 个开关（见 configs/pf_rpn_person.py）：
    #   json_file  换掉默认的 <root>/<split>.json，否则会静默去读合成数据
    #   img_dir    真实图按需下载在别处，不在 root/images 下
    #   bbox_fmt   COCO 官方是 xywh_px
    #   train_n/val_n  真实数据集很大时截断到多少张
    train_ds = COCOLikeDataset(
        root=dcfg["root"], split="train", img_size=dcfg["img_size"],
        max_objs=dcfg["max_objs"], n_images=dcfg.get("n_images", 512),
        seed=cfg.get("seed", 0), force_synth=args.force_synth,
        img_dir=dcfg.get("img_dir"),
        bbox_fmt=dcfg.get("bbox_fmt", "cxcywh"),
        json_file=dcfg.get("json_file"),
        n_images_cap=dcfg.get("train_n"),
        id_offset=dcfg.get("train_offset", 0),
    )
    val_ds = COCOLikeDataset(
        root=dcfg["root"], split="val", img_size=dcfg["img_size"],
        max_objs=dcfg["max_objs"], n_images=dcfg.get("n_images", 512),
        seed=cfg.get("seed", 0) + 7, force_synth=args.force_synth,
        img_dir=dcfg.get("img_dir"),
        bbox_fmt=dcfg.get("bbox_fmt", "cxcywh"),
        json_file=dcfg.get("json_file"),
        n_images_cap=dcfg.get("val_n"),
        # ⚠️ 只有**真实数据集**才用 ``train_n`` 当偏移。COCO 那种 json 是按
        # ``[0:train_n]`` + ``[train_n:train_n+val_n]`` 编排的，val 必须从
        # train_n 之后切，否则训练集和验证集会重叠。
        # 但**合成数据**（dump_synthetic_split）的 val 图 id 是独立编号的
        # 0..val_n-1，偏移 512 会把 128 张 val 图全切没 → 数据集长度 0 →
        # 评测静默返回 AR@100=0.0000，还会拿 epoch1 的权重覆盖 best.pth。
        # 修法：合成数据显式写 ``val_offset: 0``；这里也把兜底收紧成 0。
        id_offset=dcfg.get("val_offset", dcfg.get("train_n", 0) if dcfg.get("json_file") else 0),
    )
    print(f"[data ] 训练集 {len(train_ds)} 张 / 验证集 {len(val_ds)} 张"
          + (f" (bbox_fmt={dcfg.get('bbox_fmt')}, json={dcfg.get('json_file')})"
             if dcfg.get("json_file") else " (合成数据)"))
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                              collate_fn=collate, num_workers=cfg.get("num_workers", 0),
                              drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=max(cfg["batch_size"], 1), shuffle=False,
                            collate_fn=collate, num_workers=cfg.get("num_workers", 0))

    # ---------------- 模型 ----------------
    model = PFRPN(cfg).to(device)
    loss_fn = PFRPNLoss(**cfg.get("loss", {}))
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg["optimizer"]["lr"], weight_decay=cfg["optimizer"]["weight_decay"]
    )
    epochs = cfg["epochs"]
    total_iters = epochs * len(train_loader)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_iters)

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        print(f"[resume] 已加载 {args.resume}")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] 可训练参数 {n_params / 1e6:.3f} M | device={device} | "
          f"queries={cfg['num_queries']} | d_model={cfg['d_model']}")

    # ---------------- 训练循环 ----------------
    best = -1.0
    log_path = out_dir / "train_log.json"
    lines = []
    start = time.time()
    for epoch in range(epochs):
        model.train()
        run = {"loss_total": 0.0, "loss_bbox": 0.0, "loss_giou": 0.0,
               "loss_cls": 0.0, "loss_ctr": 0.0, "loss_moe": 0.0, "pos_ratio": 0.0}
        t0 = time.time()
        for it, batch in enumerate(train_loader):
            images = batch["image"].to(device)
            gts = batch["boxes"].to(device)
            out = model(images)
            losses = loss_fn(out["boxes"], out["cls_logits"], out["ctr"],
                             out["router_w"], gts, batch.get("gt_mask"))
            opt.zero_grad(set_to_none=True)
            losses["loss_total"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            sched.step()
            for k in run:
                run[k] += losses[k].item()
            if (it + 1) % 20 == 0 or it == 0:
                n = it + 1
                msg = (f"[epoch {epoch + 1}/{epochs}][{it + 1}/{len(train_loader)}] "
                       + " ".join(f"{k}={run[k] / n:.4f}" for k in run)
                       + f" | {time.time() - t0:.1f}s")
                print(msg, flush=True)

        n = len(train_loader)
        summary = {"epoch": epoch + 1,
                   **{k: round(run[k] / n, 5) for k in run}}
        res = evaluate_proposals(model, val_loader, topk_list=cfg.get("topk_list", (100,)),
                                 device=device)
        summary["val"] = {k: round(v, 5) for k, v in res.items()}
        summary["lr"] = opt.param_groups[0]["lr"]
        summary["epoch_sec"] = round(time.time() - t0, 1)
        lines.append(summary)
        log_path.write_text(json.dumps(lines, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[eval epoch {epoch + 1}] " + format_results(res)
              + f" | {summary['epoch_sec']}s", flush=True)

        torch.save(
            {"model": model.state_dict(), "optimizer": opt.state_dict(), "cfg": cfg},
            out_dir / "latest.pth",
        )
        key = f"AR@{cfg.get('topk_list', (100,))[0]}"
        if res.get(key, 0) > best:
            best = res[key]
            torch.save({"model": model.state_dict(), "cfg": cfg}, out_dir / "best.pth")
            print(f"[save] best.pth (best {key}={best:.4f})", flush=True)

    print(f"[done] 总耗时 {time.time() - start:.1f}s，日志 -> {log_path}")


if __name__ == "__main__":
    main()
