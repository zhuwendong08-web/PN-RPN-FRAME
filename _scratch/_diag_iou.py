"""临时诊断：对比合成帧的真实框与预测框，确认坐标链路是否对。"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pfrpn.model import PFRPN
from pfrpn.utils.box import box_cxcywh_to_xyxy
from tools.common import load_config, resolve_device
from tools.video_demo import synth_frames

cfg = load_config("configs/pf_rpn_tiny.py")
dev = resolve_device("cpu")
ck = torch.load("outputs/best.pth", map_location=dev, weights_only=False)
model = PFRPN(ck.get("cfg", cfg)).to(dev)
model.load_state_dict(ck["model"])
model.eval()

gen = synth_frames(3, cfg["data"]["img_size"], max_objs=4)
for i, (rgb, gt) in enumerate(gen):
    pil = __import__("PIL.Image", fromlist=["Image"]).fromarray(rgb).convert("RGB") \
        .resize((224, 224), __import__("PIL.Image", fromlist=["Image"]).BILINEAR)
    x = torch.from_numpy(np.asarray(pil, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        boxes, scores = model.inference(x, topk_list=(10,))
    b = boxes[0]
    s = scores[0]
    bxy = np.asarray([box_cxcywh_to_xyxy(v).tolist() for v in b])

    print(f"--- frame {i} ---")
    print(f"  GT  ({len(gt)} 个, cxcywh):")
    for g in gt:
        print(f"    cx,cy,w,h = {[round(float(v),3) for v in g]}")
    print("  预测 top-3 (cxcywh -> xyxy):")
    for j in range(min(3, len(b))):
        print(f"    score={float(s[j]):.4f} cxcywh={[round(float(v),3) for v in b[j]]}"
              f" -> xyxy={[round(float(v),3) for v in bxy[j]]}")
    print(f"  框范围检查: xyxy min={bxy.min():.3f} max={bxy.max():.3f}")
    break
