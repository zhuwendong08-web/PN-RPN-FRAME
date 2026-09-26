"""微调效果验收：模型到底有没有学会「在人身上出框」。

训练集只有 AR/AP，而 AR 只看召回、不看定位精度，分数也不会校准，
所以单看 AR@100=0.98 不能说明框画对了。这个脚本直接做三件事：

1. 在 COCO person 验证集（200 张，与训练集零重叠）上算**预测框与真值框的 IoU**；
2. 导出带框图，肉眼可核对；
3. 顺带把「合成数据训出来的模型」拉来做同一套对比，看差距是不是真的。

用法::

    python tools/verify_person.py --ckpt outputs_person/best.pth
    python tools/verify_person.py --ckpt outputs/best_SYNTHETIC_backup.pth --tag SYNTH
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pfrpn.datasets import COCOLikeDataset, collate   # noqa: E402
from pfrpn.model import PFRPN                          # noqa: E402
from pfrpn.utils.box import box_cxcywh_to_xyxy         # noqa: E402
from tools.common import load_config                   # noqa: E402

COLORS = [(255, 60, 60), (60, 200, 120), (80, 120, 255),
          (255, 210, 60), (200, 80, 220), (60, 220, 220)]


def iou_of(pred: np.ndarray, gt: np.ndarray) -> float:
    """pred/gt 都是归一化 xyxy，算最大 IoU（预测框与任一真值框的重合度）。"""
    px1, py1, px2, py2 = pred
    gx1, gy1, gx2, gy2 = gt
    ix1, iy1 = max(px1, gx1), max(py1, gy1)
    ix2, iy2 = min(px2, gx2), min(py2, gy2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (px2 - px1) * (py2 - py1) + (gx2 - gx1) * (gy2 - gy1) - inter
    return float(inter / ua) if ua > 0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs_person/best.pth")
    ap.add_argument("--config", default="configs/pf_rpn_person.py")
    ap.add_argument("--tag", default="PERSON")
    ap.add_argument("--val-n", type=int, default=200)
    ap.add_argument("--val-json", default=None,
                    help="显式指定验证标注文件；不填就用配置里的设置")
    ap.add_argument("--dump", type=int, default=8, help="导出几张带框图")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    dc = cfg["data"]
    outdir = Path(args.outdir or (ROOT / f"outputs_verify_{args.tag}"))
    outdir.mkdir(parents=True, exist_ok=True)

    val_ds = COCOLikeDataset(
        ROOT / dc["root"], split="val", img_size=dc["img_size"],
        max_objs=dc.get("max_objs", 4), img_dir=dc.get("img_dir"),
        bbox_fmt=dc.get("bbox_fmt", "cxcywh"),
        json_file=args.val_json or dc.get("json_file"),
        # 同 tools/train.py：只有真实数据集（有 json_file）才用 train_n 当偏移。
        n_images_cap=args.val_n,
        id_offset=dc.get("val_offset", dc.get("train_n", 0) if dc.get("json_file") else 0),
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate)

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = PFRPN(ck.get("cfg", cfg)).to("cpu")
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"[ckpt] {args.ckpt}")

    t0 = time.time()
    iou_top1, iou_best, hits, n_gt, n_frame_hit = [], [], 0, 0, 0
    dump_done = 0
    for idx, batch in enumerate(val_loader):
        img = batch["image"]
        with torch.no_grad():
            boxes, scores = model.inference(img, topk_list=(100,), return_all=True)
        # ⚠️ inference 返回的是**归一化 cxcywh**，不是 xyxy。
        # 这个脚本第一次跑就把它当 xyxy 用，结果 IoU 全是 0、看着像"模型没学会"，
        # 其实是脚本自己的坐标系错误（本项目第 3 次栽在这上面）。
        pb = box_cxcywh_to_xyxy(boxes[0]).numpy()   # (100,4) 归一化 xyxy
        sc = scores[0].numpy()
        gt = batch["boxes"][0].numpy()
        gt = gt[batch["gt_mask"][0].numpy()] if "gt_mask" in batch else gt

        # 真值转 xyxy 用于算 IoU（dataset 里存的是 cxcywh）
        gx1 = gt[:, 0] - gt[:, 2] / 2; gy1 = gt[:, 1] - gt[:, 3] / 2
        gx2 = gt[:, 0] + gt[:, 2] / 2; gy2 = gt[:, 1] + gt[:, 3] / 2
        gt_xyxy = np.stack([gx1, gy1, gx2, gy2], axis=1)

        if len(gt):
            per = [max(iou_of(pb[k], g) for g in gt_xyxy) for k in range(pb.shape[0])]
            iou_top1.append(per[0])
            iou_best.append(max(per))
            n_gt += len(gt)
            if per[0] >= 0.5:
                hits += 1
            if max(per) >= 0.5:
                n_frame_hit += 1

        if idx < args.dump and len(gt):
            # 框可能部分出画，不裁的话 PIL 直接抛
            # `x1 must be greater than or equal to x0`
            def _px(b_, W_, H_):
                a = (float(v) for v in b_)
                x1, y1, x2, y2 = (min(1.0, max(0.0, v)) for v in a)
                if x2 <= x1 or y2 <= y1:
                    return None
                return x1 * W_, y1 * H_, x2 * W_, y2 * H_

            im = Image.fromarray(
                (img[0].permute(1, 2, 0).numpy() * 255).astype("uint8")
            ).convert("RGB")
            W, H = im.size
            d = ImageDraw.Draw(im)
            for b in gt_xyxy:
                r = _px(b, W, H)
                if r is None:
                    continue
                d.rectangle(list(r), outline=(0, 220, 90, 220), width=2)   # 真值：绿
            for i in range(min(6, pb.shape[0])):
                if sc[i] < 0.10:
                    break
                r = _px(pb[i], W, H)
                if r is None:
                    continue
                c = COLORS[i % len(COLORS)]
                d.rectangle(list(r), outline=c + (255,), width=2)
            im.save(outdir / f"v{idx:03d}.png")
            dump_done += 1

    import statistics as st
    el = time.time() - t0
    print(f"[n    ] 验证图 {len(val_ds)} 张，用时 {el:.0f}s")
    print(f"[IoU  ] top1 与最近真值的 IoU: 中位 {st.median(iou_top1):.3f} "
          f"均值 {st.mean(iou_top1):.3f}")
    print(f"[IoU  ] 100 个提案里最佳 IoU: 中位 {st.median(iou_best):.3f}")
    print(f"[hit  ] top1 IoU>=0.5 的图占比 {hits / max(len(iou_top1),1) * 100:.1f}%")
    print(f"[hit  ] 任一提案 IoU>=0.5 的图占比 {n_frame_hit / max(len(iou_top1),1) * 100:.1f}%")
    print(f"[dump ] {dump_done} 张带框图 -> {outdir}  (绿=真值, 彩=模型提案)")
    (outdir / "summary.json").write_text(json.dumps({
        "ckpt": str(args.ckpt), "n_images": len(val_ds),
        "iou_top1_median": round(st.median(iou_top1), 4),
        "iou_top1_mean": round(st.mean(iou_top1), 4),
        "iou_best_median": round(st.median(iou_best), 4),
        "top1_hit_rate": round(hits / max(len(iou_top1), 1), 4),
        "any_hit_rate": round(n_frame_hit / max(len(iou_top1), 1), 4),
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
