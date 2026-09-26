"""诊断：推理打分（cls x ctr）到底还有没有信息？

动机：修好 GT 数据管道后重训，`diag_corner.py` 报出 top1 提案 100% 落在
上/右上两格，148/200 落在同一个格子里。真值并不是这个分布，所以这不是
数据问题，而是**打分本身失去了区分度**——sigmoid(cls)*sigmoid(ctr) 若对
所有 query 都趋近同一个常数，argmax 就退化成"挑数字最大的那个 query 序号"，
于是每张图的 top1 都是同一批 query，落点自然一模一样。

本脚本回答三个问题：

1. 分数是不是饱和了？（min/中位数/max、以及 max/min 比）
2. top1 是不是固定落在同一批 query 序号上？（查 top1 的 query 索引分布）
3. 分类分数和中心度分数，哪个更没信息？（各自单独排序时 top1 IoU 中位数）

用法::

    python tools/diag_score.py --ckpt outputs_person/best.pth
    python tools/diag_score.py --ckpt outputs/best_SYNTHETIC_backup.pth --tag SYNTH
"""
from __future__ import annotations

import argparse
import sys
import torch
from torch.utils.data import DataLoader
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pfrpn.datasets import COCOLikeDataset, collate     # noqa: E402
from pfrpn.model import PFRPN                           # noqa: E402
from pfrpn.utils.box import box_cxcywh_to_xyxy          # noqa: E402
from tools.common import load_config           # noqa: E402


def iou_matrix(pred_xyxy, gt_xyxy):
    """pred:(K,4) gt:(M,4) xyxy -> (K,M)"""
    k = pred_xyxy.shape[0]
    m = gt_xyxy.shape[0]
    if k == 0 or m == 0:
        return torch.zeros(k, m)
    iw = (torch.min(pred_xyxy[:, None, 2], gt_xyxy[None, :, 2])
          - torch.max(pred_xyxy[:, None, 0], gt_xyxy[None, :, 0])).clamp(min=0)
    # ⚠️ 必须分开取 y1 与 y2。写成 min(...,gt_y1)-max(...,gt_y1) 的话，
    # "预测框盖住真值框"这种真正该有交叠的情形反而算出 0，
    # 整份诊断就会全盘报 0.000，看着像模型完全没学会。
    ih = (torch.min(pred_xyxy[:, None, 3], gt_xyxy[None, :, 3])
          - torch.max(pred_xyxy[:, None, 1], gt_xyxy[None, :, 1])).clamp(min=0)
    inter = iw * ih
    pa = (pred_xyxy[:, None, 2] - pred_xyxy[:, None, 0]).clamp(min=0) * \
         (pred_xyxy[:, None, 3] - pred_xyxy[:, None, 1]).clamp(min=0)
    ga = (gt_xyxy[None, :, 2] - gt_xyxy[None, :, 0]).clamp(min=0) * \
         (gt_xyxy[None, :, 3] - gt_xyxy[None, :, 1]).clamp(min=0)
    return inter / (pa + ga - inter).clamp(min=1e-6)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", default="X")
    args = ap.parse_args()

    cfg = load_config("configs/pf_rpn_person.py")
    torch.manual_seed(cfg.get("seed", 0))
    # 配置文件是扁平的，没有 "model" 这一层键；PFRPN 直接吃整个 cfg
    model = PFRPN(cfg)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu")["model"])
    model.eval()

    dcfg = cfg["data"]
    ds = COCOLikeDataset(
        root=dcfg["root"],
        split="val",
        img_size=dcfg["img_size"],
        max_objs=dcfg["max_objs"],
        seed=dcfg.get("seed", 0),
        img_dir=dcfg.get("img_dir"),
        bbox_fmt=dcfg.get("bbox_fmt", "cxcywh"),
        json_file=dcfg.get("json_file"),
        n_images_cap=dcfg.get("val_n"),
        id_offset=dcfg.get("val_offset", dcfg.get("train_n", 0)),
    )
    dl = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate)

    all_scores, cls_s, ctr_s = [], [], []
    top1_idx_hist = {}
    top1_iou_by_cls, top1_iou_by_ctr, top1_iou_by_both = [], [], []
    n_pos_logits = []

    with torch.no_grad():
        for batch in dl:
            x = batch["image"]
            out = model(x)
            logits = out["cls_logits"][0]                      # (N,)
            ctr = out["ctr"][0]                                # (N,)
            boxes = out["boxes"][0]                            # (N,4)

            s_both = torch.sigmoid(logits) * torch.sigmoid(ctr)
            s_cls = torch.sigmoid(logits)
            s_ctr = torch.sigmoid(ctr)

            gt = batch["boxes"][0][batch["gt_mask"][0]]
            gt_xyxy = box_cxcywh_to_xyxy(gt) if gt.numel() else torch.zeros(0, 4)
            pred_xyxy = box_cxcywh_to_xyxy(boxes)

            # 一次算三种口径的 top1 IoU
            for s, sink in ((s_both, top1_iou_by_both),
                            (s_cls, top1_iou_by_cls),
                            (s_ctr, top1_iou_by_ctr)):
                if s.numel() == 0:
                    sink.append(0.0)
                    continue
                idx = int(torch.argsort(s, descending=True)[0])
                if gt_xyxy.numel() == 0:
                    sink.append(0.0)
                    continue
                m = iou_matrix(pred_xyxy[idx:idx + 1], gt_xyxy)
                sink.append(float(m.max()) if m.numel() else 0.0)

            top1_idx_hist[int(torch.argmax(s_both).item())] = \
                top1_idx_hist.get(int(torch.argmax(s_both).item()), 0) + 1

            all_scores.append(s_both.detach().clone())
            cls_s.append(s_cls.detach().clone())
            ctr_s.append(s_ctr.detach().clone())
            n_pos_logits.append(float((logits > 0).sum()))

    flat = torch.cat([t.reshape(-1) for t in all_scores])
    flat_cls = torch.cat([t.reshape(-1) for t in cls_s])
    flat_ctr = torch.cat([t.reshape(-1) for t in ctr_s])

    print("=" * 60)
    print(f"【分数分布】{args.ckpt}")
    print(f"  综合分 c*s   min={flat.min():.6f}  中位={flat.median():.6f}  max={flat.max():.6f}"
          f"  max/min={flat.max()/max(flat.min(),1e-9):.1f}")
    print(f"  分类分 c     min={flat_cls.min():.6f}  中位={flat_cls.median():.6f}  max={flat_cls.max():.6f}"
          f"  max/min={flat_cls.max()/max(flat_cls.min(),1e-9):.1f}")
    print(f"  中心度 s     min={flat_ctr.min():.6f}  中位={flat_ctr.median():.6f}  max={flat_ctr.max():.6f}"
          f"  max/min={flat_ctr.max()/max(flat_ctr.min(),1e-9):.1f}")
    print(f"  每图 logit>0 的 query 数（共 {cls_s[0].numel()} 个）:"
          f" 均值 {sum(n_pos_logits)/len(n_pos_logits):.2f}  最少 {min(n_pos_logits)}  最多 {max(n_pos_logits)}")

    print()
    print(f"【top1 落在哪个 query 序号上】n={sum(top1_idx_hist.values())}，"
          f"有 {len(top1_idx_hist)} 种不同序号")
    top = sorted(top1_idx_hist.items(), key=lambda kv: -kv[1])[:10]
    print("  最常见的 10 个 query 序号: " + ", ".join(f"#{i}:{c}" for i, c in top))
    print(f"  最高频的那个占了 {max(top1_idx_hist.values())/sum(top1_idx_hist.values())*100:.1f}%")

    print()
    print("【top1 IoU 中位数】按不同打分口径")
    print(f"  综合分 c*s      {sorted(top1_iou_by_both)[len(top1_iou_by_both)//2]:.3f}")
    print(f"  只看分类分 c    {sorted(top1_iou_by_cls)[len(top1_iou_by_cls)//2]:.3f}")
    print(f"  只看中心度 s    {sorted(top1_iou_by_ctr)[len(top1_iou_by_ctr)//2]:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
