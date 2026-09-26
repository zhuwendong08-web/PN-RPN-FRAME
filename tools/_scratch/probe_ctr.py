"""查清"中心度为什么是个常数 0.5"。

问题：v1/v2/v3 训出来的模型，``sigmoid(ctr)`` 永远落在 (0.487, 0.512)，
即原始 logit 只有 ±0.05 的抖动，中心度头看起来完全没被训到。

三个互斥的假设：
  H1 梯度太小：``ctr_pos_only`` 分支里损失除以 N=100，而正样本只有约 2.4 个
     ⇒ 每个正样本的等效权重只有 ``lam_ctr/N = 5/100 = 0.05``。
  H2 目标本身没信息：``centerness_target`` 拿**预测框**和真值算，如果预测框
     都很粗，``min/max(l,r)`` 与 ``min/max(t,b)`` 常常都≈1 ⇒ 目标恒为 1。
     （若真是这样，输出常数就该是 0.73 而不是 0.5，所以这条能证伪。）
  H3 中心度头根本没收到梯度（被冻结 / 名字不匹配）。

用法：python tools/_scratch/probe_ctr.py [ckpt]
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pfrpn.datasets import COCOLikeDataset, collate   # noqa: E402
from pfrpn.losses import PFRPNLoss                    # noqa: E402
from pfrpn.model import PFRPN                         # noqa: E402
from pfrpn.utils.box import centerness_target, box_cxcywh_to_xyxy  # noqa: E402
from tools.common import load_config                  # noqa: E402

CKPT = sys.argv[1] if len(sys.argv) > 1 else "outputs_person_v3/best.pth"
CFG = "configs/pf_rpn_person_v3.py"

cfg = load_config(CFG)
torch.manual_seed(cfg.get("seed", 0))
model = PFRPN(cfg)
model.load_state_dict(torch.load(CKPT, map_location="cpu")["model"], strict=False)

dcfg = cfg["data"]
ds = COCOLikeDataset(
    root=dcfg["root"], split="val", img_size=dcfg["img_size"],
    max_objs=dcfg["max_objs"], seed=dcfg.get("seed", 0),
    img_dir=dcfg.get("img_dir"), bbox_fmt=dcfg.get("bbox_fmt", "cxcywh"),
    json_file=dcfg.get("json_file"),
    n_images_cap=dcfg.get("val_n"), id_offset=dcfg.get("val_offset", 0),
)
dl = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate)

fn = PFRPNLoss(**cfg["loss"])
tgt_all, raw_all = [], []
grad_norm = {}
last_loss = None

for batch in dl:
    x = batch["image"]
    model.train()
    out = model(x)                     # 训练态，为了拿到 logits/ctr/boxes
    losses = fn(
        out["boxes"], out["cls_logits"], out["ctr"],
        out.get("router_w", None), batch["boxes"], batch["gt_mask"],
    )
    losses["loss_total"].backward()
    for n, p in model.named_parameters():
        # 必须在 zero_grad 之前收，否则下面 grad_norm 全空（zero_grad 用了
        # set_to_none=True，grad 直接被置成 None 而不是 0）。
        if p.grad is not None:
            grad_norm[n] = grad_norm.get(n, 0.0) + p.grad.norm().item()
    model.zero_grad(set_to_none=True)
    last_loss = float(losses["loss_ctr"].detach())

    gt = batch["boxes"][0][batch["gt_mask"][0]]          # (M,4) cxcywh
    if gt.numel() == 0:
        continue
    pb = out["boxes"][0]                                  # (N,4) cxcywh
    c = centerness_target(pb, gt)                         # (N,M)
    keep = c > 0                        # (N,M) 有交集的匹配对
    tgt_all.append(c[keep])
    raw_all.append(out["ctr"][0][keep.any(1)])

print("=" * 76)
print(f"模型: {CKPT}")
print("=" * 76)

t = torch.cat(tgt_all)
print(f"【H2】匹配对的 centerness 目标   n={t.numel()}")
print(f"      min={t.min():.4f}  中位={t.median():.4f}  均值={t.mean():.4f}  max={t.max():.4f}")
print(f"      落在 (0.667,1.0] 的比例 = {(t > 0.667).float().mean():.3f}")
print(f"      落在 [0,0.333) 的比例   = {(t < 0.333).float().mean():.3f}")
print("      → 目标有方差 ⇒ 常数 0.5 是 Head 没训起来；目标恒 1 ⇒ 应该是 0.73")

if raw_all:
    r = torch.cat(raw_all)
    print()
    print("【现状】正样本 query 上的中心度原始 logit")
    print(f"      min={r.min():.4f}  中位={r.median():.4f}  max={r.max():.4f}"
          f"  → sigmoid 中位 {torch.sigmoid(r).median():.4f}")

tot = sum(g ** 2 for g in grad_norm.values()) ** 0.5
ctr_keys = [n for n in grad_norm if "ctr" in n.lower() or "cgqs" in n.lower()]
ctr_g = sum(grad_norm[n] ** 2 for n in ctr_keys) ** 0.5
print()
print(f"【H1+H3】梯度：总范数 {tot:.5f}，center 相关 {ctr_g:.5f} "
      f"({ctr_g / tot * 100:.2f}%)")
for n in ctr_keys[:10]:
    print(f"      {n:56s} {grad_norm[n]:.6f}")
if not ctr_keys:
    print("      ⚠️ 参数表里没有任何名字含 ctr/cgqs —— 中心度头没收到梯度。")

zero_n = sum(1 for n in grad_norm if grad_norm[n] == 0)
print()
print(f"      0 梯度的参数 {zero_n}/{len(grad_norm)} 个")
print(f"【损失】loss_ctr = {last_loss:.4f}  "
      f"lam_ctr={cfg['loss']['lam_ctr']}  N={cfg.get('num_queries')}")
