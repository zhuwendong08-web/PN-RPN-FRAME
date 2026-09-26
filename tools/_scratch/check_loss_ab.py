"""A/B 自检：普通加权 BCE vs focal，BCE+全query中心度 vs focal+仅正样本中心度。

纯数值检查，不碰数据集。随机张量喂给 PFRPNLoss，确认：
* 不炸、梯度有限；
* 两种口径下 loss_cls / loss_ctr 的量级都还在"和 reg 项可比"的范围。
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pfrpn.losses import PFRPNLoss  # noqa: E402

torch.manual_seed(0)
B, N, M = 2, 100, 4

CASES = [
    ("v1  bce + ctr_all", dict(use_focal=False, ctr_pos_only=False)),
    ("v1' bce + ctr_pos ", dict(use_focal=False, ctr_pos_only=True)),
    ("v2' focal+ ctr_all", dict(use_focal=True, ctr_pos_only=False)),
    ("v2  focal+ ctr_pos", dict(use_focal=True, ctr_pos_only=True)),
]

for name, kw in CASES:
    fn = PFRPNLoss(
        loss_bbox=5.0, loss_giou=2.0, lam_ctr=5.0, loss_moe=1.0,
        cls_pos_weight=1.0, cls_neg_weight=0.2, use_moe=True, **kw
    )
    boxes = torch.rand(B, N, 4, requires_grad=True)      # 归一化 cxcywh
    clf = torch.randn(B, N) * 2.0
    ctr = torch.randn(B, N) * 2.0
    rw = torch.rand(B, 4)
    gt = torch.rand(B, M, 4)                              # 归一化 cxcywh
    # ⚠️ gt_mask 必须是 (B, M) 二维 bool。传 (B,) 的话 gt_mask[b] 是 0 维 bool，
    # gt[0维bool] 会给 (4,4) 的 gt 凭空插一个维度变 (1,4,4)，报错信息完全指不到根因。
    mask = torch.ones(B, M, dtype=torch.bool)

    out = fn(boxes, clf, ctr, rw, gt, mask)
    out["loss_total"].backward()

    print(f"--- {name} ---")
    for k, v in out.items():
        print(f"    {k:14s} {float(v):.4f}")
    print(f"    grad_finite={bool(torch.isfinite(boxes.grad).all())}")
