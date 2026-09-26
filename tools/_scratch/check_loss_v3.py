"""v3 自检：cls_pos_only 分支的数值量级 / 有限性 / 正样本梯度占比。

随机张量喂给 PFRPNLoss，回答三个问题：
1. ``cls_pos_only=True`` 时 loss_cls 会不会 NaN（整图无 GT 的退化分支）；
2. 量级是否和 reg 项可比；
3. **正样本收到的梯度占比**——这才是 v3 想改的东西，比 loss 数值本身重要。
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


def build(kw: dict, cls: torch.Tensor, boxes: torch.Tensor, ctr: torch.Tensor):
    fn = PFRPNLoss(
        loss_bbox=5.0, loss_giou=2.0, lam_ctr=5.0, loss_moe=1.0,
        cls_pos_weight=1.0, cls_neg_weight=1.0, use_moe=True,
        use_focal=True, focal_alpha=0.25, focal_gamma=2.0, **kw
    )
    return fn(boxes, cls, ctr, torch.rand(B, 4), torch.rand(B, M, 4),
              torch.ones(B, M, dtype=torch.bool))


print("=" * 74)
print("A. 有 GT（正常路径）")
print("=" * 74)
for tag, kw in (("v2  focal+ctr_pos", dict(ctr_pos_only=True)),
                ("v3  cls_pos_only  ", dict(ctr_pos_only=True, cls_pos_only=True))):
    boxes = torch.rand(B, N, 4, requires_grad=True)
    clf = (torch.randn(B, N) * 2.0).requires_grad_(True)
    ctr = torch.randn(B, N, requires_grad=True) * 2.0
    out = build(kw, clf, boxes, ctr)
    out["loss_total"].backward()
    finite = all(bool(torch.isfinite(v).all()) for v in out.values())
    print(f"--- {tag} ---")
    for k, v in out.items():
        print(f"    {k:14s} {float(v):.4f}")
    bg = boxes.grad
    print(f"    all_finite={finite}  box_grad={'none' if bg is None else bool(torch.isfinite(bg).all())}")

print()
print("=" * 74)
print("B. 整图无 GT（退化路径：pos_flags 全 False，必须不产生 NaN）")
print("=" * 74)
zero = torch.zeros(B, M, dtype=torch.bool)
for tag, kw in (("v2  focal+ctr_pos", dict(ctr_pos_only=True)),
                ("v3  cls_pos_only  ", dict(ctr_pos_only=True, cls_pos_only=True))):
    boxes = torch.rand(B, N, 4, requires_grad=True)
    clf = (torch.randn(B, N) * 2.0).requires_grad_(True)
    ctr = torch.randn(B, N, requires_grad=True) * 2.0
    fn = PFRPNLoss(
        loss_bbox=5.0, loss_giou=2.0, lam_ctr=5.0, loss_moe=1.0,
        cls_pos_weight=1.0, cls_neg_weight=1.0, use_moe=True,
        use_focal=True, focal_alpha=0.25, focal_gamma=2.0, **kw
    )
    out = fn(boxes, clf, ctr, torch.rand(B, 4), torch.rand(B, M, 4), zero)
    out["loss_total"].backward()
    finite = all(bool(torch.isfinite(v).all()) for v in out.values())
    print(f"--- {tag} ---")
    for k, v in out.items():
        print(f"    {k:14s} {float(v):.4f}")
    bg = boxes.grad
    print(f"    all_finite={finite}  box_grad={'none' if bg is None else bool(torch.isfinite(bg).all())}")

print()
print("=" * 74)
print("C. 正样本 vs 负样本 的 logit 梯度量级（v3 想改的就是这个）")
print("=" * 74)
b, n, m = 1, 100, 4
for tag, neg_w, pos_only in (("v3  neg 不进损失", 1.0, True),
                             ("v2  neg 进损失  ", 0.2, False)):
    fn = PFRPNLoss(
        loss_bbox=5.0, loss_giou=2.0, lam_ctr=5.0, loss_moe=1.0,
        cls_pos_weight=1.0, cls_neg_weight=neg_w, use_moe=True,
        use_focal=True, ctr_pos_only=True, cls_pos_only=pos_only,
    )
    clf = (torch.randn(b, n) * 2.0).requires_grad_(True)
    out = fn(torch.rand(b, n, 4, requires_grad=True), clf, torch.randn(b, n), torch.rand(b, 4),
             torch.rand(b, m, 4), torch.ones(b, m, dtype=torch.bool))
    out["loss_cls"].backward()
    g = clf.grad
    print(f"{tag}: |g| 总和={float(g.abs().sum()):.4f}  "
          f"g>0 个数={int((g > 0).sum()):3d}/100  "
          f"g<0 个数={int((g < 0).sum()):3d}/100")
