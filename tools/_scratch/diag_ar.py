"""定位 AR@100=0.0000：在 batch_size=2 上逐图复算 evaluate_proposals。

判据（verbatim）：
    手工算 recall@100 = 1.0
    batch_size=1 走 evaluate_proposals -> 0.6816
    batch_size=2 走 evaluate_proposals -> 0.0000
本脚本把 evaluate_proposals 的内部逐图中间量拆出来看。
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import tools.common as C  # noqa: E402
from pfrpn.datasets import COCOLikeDataset, collate  # noqa: E402
from pfrpn.model import PFRPN  # noqa: E402
from pfrpn.utils.box import box_iou  # noqa: E402

# --------------------------------------------------------------------------- #
# monkeypatch：把 _match_recall 的每次调用记录下来
# --------------------------------------------------------------------------- #
REC: list = []


def _spy(pred, gt, matched, thr):
    p, m = pred.shape[0], gt.shape[0]
    iou = box_iou(pred, gt) if p else torch.zeros(0, m)
    v = _ORIG_MATCH(pred, gt, matched, thr)   # 调原始实现，不能调自己
    REC.append({
        "P": p, "M": m,
        "max_iou": float(iou.max()) if iou.numel() else float("nan"),
        "n_iou05": int((iou >= 0.5).sum()) if iou.numel() else 0,
        "ret": float(v),
    })
    return v


_ORIG_MATCH = C._match_recall
C._match_recall = _spy


def build(bs: int):
    import tools.common as C2
    from pfrpn.build import build_model  # noqa: F401
    return None


def main() -> None:
    cfg = C.load_config("configs/pf_rpn_tiny.py")
    C.set_seed(cfg.get("seed", 0))
    device = C.resolve_device(cfg.get("device", "auto"))

    ckpt = torch.load("outputs/best.pth", map_location=device, weights_only=False)
    model = PFRPN(ckpt.get("cfg", cfg)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[load] outputs/best.pth")

    dcfg = cfg["data"]
    ds = COCOLikeDataset(
        root=dcfg["root"], split="val", img_size=dcfg["img_size"],
        max_objs=dcfg["max_objs"], seed=cfg.get("seed", 0) + 7,
        n_images=dcfg.get("val_n", 128),
    )

    for bs in (1, 2):
        REC.clear()
        loader = DataLoader(ds, batch_size=bs, shuffle=False, collate_fn=collate)
        res = C.evaluate_proposals(model, loader, topk_list=(100,), device=device)
        print(f"\n===== batch_size={bs}  AR@100={res['AR@100']:.6f}  "
              f"ARs={res['ARs']:.4f} ARm={res['ARm']:.4f} ARl={res['ARl']:.4f} "
              f"AP50={res['AP50']:.6f}")
        print(f"  _match_recall 调用次数 = {len(REC)}  "
              f"(图片数={ds.__len__()}，batch={bs})")
        if REC:
            rets = [r["ret"] for r in REC]
            print(f"  逐图 recall 分布: 0 的个数={sum(1 for v in rets if v == 0.0)}"
                  f"  >0 的个数={sum(1 for v in rets if v > 0.0)}"
                  f"  均值={sum(rets) / len(rets):.4f}")
            print(f"  P 的形状集合 = {sorted({r['P'] for r in REC})}")
            print(f"  M 的形状集合 = {sorted({r['M'] for r in REC})}")
            print(f"  前 6 条: {REC[:6]}")

    # ------------------------------------------------------------------ #
    # 逐图手工复算（batch_size=2，复刻 evaluate_proposals 的循环）
    # ------------------------------------------------------------------ #
    print("\n===== 逐图手工复算（batch_size=2 的第一批） =====")
    loader = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=collate)
    model.eval()
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            images = batch["image"].to(device)
            gts = batch["boxes"].to(device)
            gtm = batch["gt_mask"].to(device)
            pb, ps = model.inference(images, topk_list=(100,), return_all=True)
            print(f"  batch {bi}: images={tuple(images.shape)} boxes={tuple(gts.shape)} "
                  f"gt_mask 真值数={gtm.sum(dim=1).tolist()} "
                  f"pred_boxes={tuple(pb.shape)} pred_scores={tuple(ps.shape)}")
            for i in range(images.shape[0]):
                gt = gts[i][gtm[i]]
                pbi, psi = pb[i], ps[i]
                order = torch.argsort(psi, descending=True)
                pbi = pbi[order]
                iou = box_iou(pbi[:100], gt)
                manual = float((iou.max(dim=0).values >= 0.5).sum()) / max(gt.shape[0], 1)
                print(f"    图 {i}: M={gt.shape[0]} P={pbi.shape[0]} "
                      f"手工recall@100={manual:.4f} "
                      f"maxIoU={float(iou.max()) if iou.numel() else float('nan'):.4f} "
                      f"pb范围=[{float(pbi.min()):.3f},{float(pbi.max()):.3f}] "
                      f"gt范围=[{float(gt.min()):.3f},{float(gt.max()):.3f}]")
            if bi >= 1:
                break


if __name__ == "__main__":
    main()
