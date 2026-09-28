"""分别给「数据加载」和「训练一步」计时，定位慢在哪。

用法:
    python tools/_scratch/probe_speed.py [n_steps] [batch_size]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from pfrpn.datasets import COCOLikeDataset, collate  # noqa: E402
from pfrpn.model import PFRPN                          # noqa: E402
from pfrpn.losses import PFRPNLoss                     # noqa: E402
from torch.utils.data import DataLoader                 # noqa: E402
from tools.common import load_config, set_seed          # noqa: E402

NSteps = int(sys.argv[1]) if len(sys.argv) > 1 else 5
BATCH = int(sys.argv[2]) if len(sys.argv) > 2 else 2


def main() -> None:
    cfg = load_config("configs/pf_rpn_person_v4.py")
    set_seed(0)

    dcfg = cfg["data"]
    print(f"[cfg] json_file={dcfg.get('json_file')} img_dir={dcfg.get('img_dir')} "
          f"img_size={dcfg['img_size']} max_objs={dcfg['max_objs']}", flush=True)

    t = time.time()
    ds = COCOLikeDataset(
        root=dcfg["root"], split="train", img_size=dcfg["img_size"],
        max_objs=dcfg["max_objs"], seed=0, n_images=dcfg.get("train_n", 1500),
        json_file=Path(dcfg["json_file"]), img_dir=Path(dcfg["img_dir"]),
        # ⚠️ 必须跟 train.py 一样传 n_images_cap。
        # 漏了它 -> self.ids = sorted(meta.keys()) = 全部 64115 个 id，
        # 其中 94.8% 磁盘上没有图，DataLoader 一 shuffle 就 FileNotFoundError。
        n_images_cap=dcfg.get("train_n"),
        id_offset=dcfg.get("train_offset", 0),
    )
    print(f"[data] 数据集构建 {time.time() - t:.1f}s  图数={len(ds)}", flush=True)

    loader = DataLoader(ds, batch_size=BATCH, shuffle=True, collate_fn=collate,
                        num_workers=0)
    it = iter(loader)

    t = time.time()
    batch = next(it)
    print(f"[data] 首 batch（含图片解码） {time.time() - t:.1f}s  "
          f"image={tuple(batch['image'].shape)}", flush=True)

    model = PFRPN(cfg)
    loss_fn = PFRPNLoss(**cfg["loss"])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # 预热：oneDNN 首次编译很慢，不计入
    for _ in range(2):
        out = model(batch["image"])
        losses = loss_fn(out["boxes"], out["cls_logits"], out["ctr"],
                         out["router_w"], batch["boxes"], batch.get("gt_mask"))
        losses["loss_total"].backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
    opt.zero_grad(set_to_none=True)
    print("[train] 预热完成，开始计时", flush=True)

    ts = []
    for i in range(NSteps):
        t = time.time()
        b = next(it)
        images, gts = b["image"], b["boxes"]
        out = model(images)
        losses = loss_fn(out["boxes"], out["cls_logits"], out["ctr"],
                         out["router_w"], gts, b.get("gt_mask"))
        opt.zero_grad(set_to_none=True)
        losses["loss_total"].backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        ts.append(time.time() - t)
        print(f"  step {i + 1}/{NSteps}: {ts[-1]:.2f}s  loss={float(losses['loss_total']):.4f}",
              flush=True)

    ts_sorted = sorted(ts)
    med = ts_sorted[len(ts) // 2]
    print(f"\n[train] 每步 {med:.2f}s（中位）  最快 {ts_sorted[0]:.2f}s  "
          f"最慢 {ts_sorted[-1]:.2f}s", flush=True)
    print(f"[train] batch_size={BATCH}  => 每秒处理 {BATCH / med:.2f} 张")
    n_per_epoch = 3348 // BATCH
    print(f"[train] 全量 3348 图 1 epoch ≈ {n_per_epoch * med / 60:.0f} 分钟")



if __name__ == "__main__":
    main()
