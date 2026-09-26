"""证实 hypothesis：train.py 的 val_ds 带 id_offset=train_n=512，跳到了不存在的图。

train.py:  id_offset=dcfg.get("val_offset", dcfg.get("train_n", 0)) = 512
test.py :  不传 id_offset，默认 0
data 里只有 512 张训练图 + 128 张 val 图。id>=512 的图不存在 -> gt_mask 全 False
-> evaluate_proposals 的 `if M == 0: buckets[...].append(0.0)` -> AR 恒为 0。
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from pfrpn.datasets import COCOLikeDataset, collate  # noqa: E402
from tools.common import (  # noqa: E402
    evaluate_proposals, load_config, resolve_device, set_seed,
)


def probe(tag: str, **kw) -> None:
    cfg = load_config("configs/pf_rpn_tiny.py")
    set_seed(cfg.get("seed", 0))
    device = resolve_device(cfg.get("device", "auto"))
    dcfg = cfg["data"]

    val_ds = COCOLikeDataset(
        root=dcfg["root"], split="val", img_size=dcfg["img_size"],
        max_objs=dcfg["max_objs"], n_images=dcfg.get("n_images", 512),
        seed=cfg.get("seed", 0) + 7,
        img_dir=dcfg.get("img_dir"),
        bbox_fmt=dcfg.get("bbox_fmt", "cxcywh"),
        json_file=dcfg.get("json_file"),
        **kw,
    )
    loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate)

    n_img = len(val_ds)
    # 直接逐样本看标注，绕开 collate
    n_zero = 0
    first_boxes = None
    for i in range(n_img):
        itm = val_ds[i]
        b = itm["boxes"]
        if first_boxes is None:
            first_boxes = (itm.get("image_id"), b)
        if b.shape[0] == 0:
            n_zero += 1

    tot_mask = 0
    for b in loader:
        tot_mask += int(b["gt_mask"].sum().item())

    print(f"\n===== {tag}")
    print(f"  数据集长度={n_img}  逐样本标注框数=0 的图={n_zero}/{n_img}")
    print(f"  第一张: image_id={first_boxes[0]}  boxes={first_boxes[1]}")
    print(f"  collate 后 gt_mask 真值总数={tot_mask}")
    res = evaluate_proposals(
        _mk_model(cfg, device), loader, topk_list=(100,), device=device
    )
    print(f"  AR@100={res['AR@100']:.6f}  AP50={res['AP50']:.6f}")


def _mk_model(cfg, device):
    from pfrpn.model import PFRPN
    ckpt = torch.load("outputs/best.pth", map_location=device, weights_only=False)
    m = PFRPN(ckpt.get("cfg", cfg)).to(device)
    m.load_state_dict(ckpt["model"])
    m.eval()
    return m


def main() -> None:
    dcfg_all = load_config("configs/pf_rpn_tiny.py")["data"]
    print(f"tiny data 配置: n_images={dcfg_all['n_images']} "
          f"train_n={dcfg_all['train_n']} val_n={dcfg_all['val_n']}")

    probe("test.py 口径（无 id_offset）")
    probe("train.py 口径（id_offset=val_offset|train_n=512）",
          n_images_cap=dcfg_all.get("val_n"),
          id_offset=dcfg_all.get("val_offset", dcfg_all.get("train_n", 0)))


if __name__ == "__main__":
    main()
