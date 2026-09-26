"""定位 v2 训练中途网络整体变 NaN 的元凶（第二轮：盯损失分量与梯度）。

第一轮 hook 显示第一个非有限输出在 backbone 最底层，而 clip_grad_norm_ 会
把"某一个梯度 NaN"放大成"全部参数 NaN"，所以得反过来问：

1. 是**前向**哪一环先 NaN（loss 分量谁先非有限）；
2. 还是**反向**某个 op 产出 NaN 梯度（第一个非有限梯度参数是谁）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pfrpn.datasets import COCOLikeDataset, collate  # noqa: E402
from pfrpn.losses import PFRPNLoss                    # noqa: E402
from pfrpn.model import PFRPN                         # noqa: E402
from tools.common import load_config, set_seed        # noqa: E402

cfg = load_config("configs/pf_rpn_person_v2.py")
set_seed(cfg.get("seed", 0))

dcfg = cfg["data"]
train_ds = COCOLikeDataset(
    root=dcfg["root"], split="train", img_size=dcfg["img_size"],
    max_objs=dcfg["max_objs"], n_images=dcfg.get("n_images", 512),
    seed=cfg.get("seed", 0), img_dir=dcfg.get("img_dir"),
    bbox_fmt=dcfg.get("bbox_fmt", "cxcywh"), json_file=dcfg.get("json_file"),
    n_images_cap=dcfg.get("train_n"), id_offset=dcfg.get("train_offset", 0),
)
train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                          collate_fn=collate, num_workers=0, drop_last=False)

model = PFRPN(cfg)
loss_fn = PFRPNLoss(**cfg.get("loss", {}))
opt = torch.optim.AdamW(model.parameters(), lr=cfg["optimizer"]["lr"],
                        weight_decay=cfg["optimizer"]["weight_decay"])
sched = torch.optim.lr_scheduler.CosineAnnealingLR(
    opt, T_max=cfg["epochs"] * len(train_loader))
model.train()

MAXIT = int(sys.argv[1]) if len(sys.argv) > 1 else 400

print(f"开始跑，最多 {MAXIT} 步", flush=True)
for it, batch in enumerate(train_loader):
    if it >= MAXIT:
        break
    images, gts = batch["image"], batch["boxes"]
    images_ok = bool(torch.isfinite(images).all())
    gts_ok = bool(torch.isfinite(gts).all())
    try:
        out = model(images)
    except Exception as e:
        print(f"[step {it}] 前向就炸: {type(e).__name__}: {e} "
              f"images_ok={images_ok} gts_ok={gts_ok}", flush=True)
        break
    out_ok = all(torch.isfinite(v).all().item()
                 for k, v in out.items() if isinstance(v, torch.Tensor))

    try:
        losses = loss_fn(out["boxes"], out["cls_logits"], out["ctr"],
                         out["router_w"], gts, batch.get("gt_mask"))
        bad_loss = [k for k, v in losses.items()
                    if isinstance(v, torch.Tensor) and not torch.isfinite(v).all().item()]
    except Exception as e:
        print(f"[step {it}] 损失就炸: {type(e).__name__}: {e} "
              f"images_ok={images_ok} gts_ok={gts_ok} out_ok={out_ok}", flush=True)
        break

    opt.zero_grad(set_to_none=True)
    try:
        losses["loss_total"].backward()
    except Exception as e:
        print(f"[step {it}] 反向就炸: {type(e).__name__}: {e} "
              f"bad_loss={bad_loss} images_ok={images_ok} gts_ok={gts_ok} out_ok={out_ok}", flush=True)
        break

    gn = float(nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0))
    bad_g = [n for n, p in model.named_parameters()
             if p.grad is not None and not torch.isfinite(p.grad).all().item()]
    bad_p = [n for n, p in model.named_parameters() if not torch.isfinite(p).all().item()]

    if (bad_loss or bad_g or bad_p or not out_ok or not images_ok or not gts_ok
            or not torch.isfinite(torch.as_tensor(gn)).item()):
        print(f"\n[step {it}] 出问题：", flush=True)
        print(f"   images_ok={images_ok} gts_ok={gts_ok} out_ok={out_ok}", flush=True)
        print(f"   bad_loss={bad_loss}", flush=True)
        print(f"   grad_norm={gn}", flush=True)
        print(f"   第一个非有限梯度参数: {bad_g[:5]}", flush=True)
        print(f"   已非有限的参数: {bad_p[:5]}", flush=True)
        if bad_loss:
            for k in bad_loss:
                print(f"     {k} = {losses[k]}", flush=True)
        break

    opt.step()
    sched.step()

print("跑完，全程没触发。" , flush=True)
