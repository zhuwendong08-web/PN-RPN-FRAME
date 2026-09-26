"""交付前的稳定性体检（纯 CPU，约 30 秒）。

判据是"能不能当交付物交出去"，所以只查会真的把人坑到的那几类问题：

1. **严格加载** —— ``strict=True`` 能不能过。很多项目在 ``strict=False`` 下
   静默丢权重，训练完的效果和交出去的效果对不上，线上才发现。
2. **确定性** —— 同一张图跑两次，top1 必须逐位相同。
   视频逐帧是独立前馈的，只要单帧确定，整段视频就确定。
3. **批量等价** —— batch=2 的结果要和逐张 batch=1 一致。
   不一致说明某处偷偷依赖了 batch 维度（比如 BatchNorm），
   那么视频逐帧跑（batch=1）和训练（batch=2）就会是两套行为。
4. **退化输入** —— 纯黑 / 纯白 / 纯噪声，不能出 NaN/inf。
5. **分辨率鲁棒** —— 160 / 224 / 320 / 448 都不炸，且框仍在 [0,1] 内。
6. **数值范围** —— 分数必须落在 (0,1)，框必须在 [0,1]。

用法：python tools/robust_check.py [--ckpt ...] [--config ...]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pfrpn.model import PFRPN                            # noqa: E402
from tools.common import load_config                     # noqa: E402

PASS, FAIL = "✅ PASS", "❌ FAIL"
results: list[tuple[str, str, str]] = []


def record(name: str, ok: bool, detail: str) -> None:
    results.append((name, PASS if ok else FAIL, detail))


def top1(model: torch.nn.Module, x: torch.Tensor):
    with torch.no_grad():
        out = model(x)
    s = torch.sigmoid(out["cls_logits"]) * torch.sigmoid(out["ctr"])
    i = int(s.argmax(dim=1))
    return out["boxes"][0, i], float(s[0, i])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs_person_v4/best.pth")
    ap.add_argument("--config", default="configs/pf_rpn_person_v4.py")
    args = ap.parse_args()

    cfg = load_config(args.config)
    torch.manual_seed(cfg.get("seed", 0))
    model = PFRPN(cfg)
    ck = torch.load(args.ckpt, map_location="cpu")
    res = model.load_state_dict(ck["model"], strict=True)
    record("严格加载（strict=True，无缺失/多余权重）",
           not res.missing_keys and not res.unexpected_keys,
           f"missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}"
           f"  可训练参数 {sum(p.numel() for p in model.parameters()):,}")
    model.eval()

    torch.manual_seed(1234)
    img = torch.rand(1, 3, cfg["data"]["img_size"], cfg["data"]["img_size"])

    # --- 2. 确定性 ---
    b1, s1 = top1(model, img)
    b2, s2 = top1(model, img)
    same = torch.equal(b1, b2) and s1 == s2
    record("确定性（同帧跑两次结果逐位相同）", same,
           f"top1={b1.tolist()} 分={s1:.6f} / {s2:.6f}")

    # --- 3. 批量等价 ---
    pair = torch.cat([img, torch.rand_like(img)], dim=0)
    with torch.no_grad():
        out_b = model(pair)
    s_b = torch.sigmoid(out_b["cls_logits"]) * torch.sigmoid(out_b["ctr"])
    sb0 = float(s_b[0, int(s_b[0].argmax())])
    record("批量等价（batch=2 的第 0 张 == 单张跑）",
           abs(sb0 - s1) < 1e-6,
           f"单张 {s1:.8f} vs 批量 {sb0:.8f}  Δ={abs(sb0 - s1):.2e}")

    # --- 4/5/6. 退化输入与分辨率 ---
    cases = [
        ("纯黑", torch.zeros(1, 3, cfg["data"]["img_size"], cfg["data"]["img_size"])),
        ("纯白", torch.ones(1, 3, cfg["data"]["img_size"], cfg["data"]["img_size"])),
        ("纯噪声", torch.rand(1, 3, cfg["data"]["img_size"], cfg["data"]["img_size"])),
    ]
    sizes = [160, cfg["data"]["img_size"], 320, 448]
    bad = []
    for tag, x in cases:
        for sz in sizes:
            xx = torch.nn.functional.interpolate(x, size=(sz, sz),
                                                 mode="bilinear", antialias=False)
            with torch.no_grad():
                out = model(xx)
            bx = out["boxes"]
            finite = all(torch.isfinite(v).all()
                         for v in (bx, out["cls_logits"], out["ctr"]))
            inrange = bool((bx >= 0).all() and (bx <= 1).all())
            scores = torch.sigmoid(out["cls_logits"]) * torch.sigmoid(out["ctr"])
            sok = bool((scores > 0).all() and (scores <= 1).all())
            if not (finite and inrange and sok):
                bad.append(f"{tag}@{sz}: finite={finite} box∈[0,1]={inrange} score∈(0,1]={sok}")
    record(f"退化输入 × 分辨率（3 种 × {len(sizes)} 档 = {3 * len(sizes)} 例）",
           not bad, "; ".join(bad) if bad else "全部有限、框与分数都在合法区间")

    # --- 吞吐 ---
    import time
    t0 = time.perf_counter()
    n = 20
    with torch.no_grad():
        for _ in range(n):
            model(img)
    ms = (time.perf_counter() - t0) / n * 1000
    record("吞吐（224px 单帧，纯 CPU）", True,
           f"{ms:.1f} ms/帧 ≈ {1000 / ms:.1f} FPS")

    print("=" * 74)
    print(f"交付前稳定性体检  权重 {args.ckpt}")
    print("=" * 74)
    for name, st, detail in results:
        print(f"{st}  {name}")
        print(f"        {detail}")
    n_ok = sum(1 for _, st, _ in results if st == PASS)
    print("=" * 74)
    print(f"{n_ok}/{len(results)} 项通过")
    print("=" * 74)


if __name__ == "__main__":
    main()
