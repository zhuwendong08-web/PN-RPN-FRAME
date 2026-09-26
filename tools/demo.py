"""单图推理演示：输出提案框，并可选可视化可学习嵌入的热力图。

    python tools/demo.py --checkpoint outputs/latest.pth --image data/images/val_00000.png
    python tools/demo.py --checkpoint outputs/latest.pth --synthesize --visualize-embed

--visualize-embed 会把「可学习嵌入 F~^T 与各层特征的余弦相似度」画出来，
这正是论文 Fig.8 用来论证「物体内部特征比 embedding 本身定位能力更强」的图。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pfrpn.datasets import generate_scene                  # noqa: E402
from pfrpn.model import PFRPN                              # noqa: E402
from pfrpn.utils.box import box_cxcywh_to_xyxy             # noqa: E402
from tools.common import load_config, resolve_device       # noqa: E402

COLORS = [(255, 60, 60), (60, 200, 120), (80, 120, 255), (255, 210, 60),
          (200, 80, 220), (60, 220, 220), (255, 140, 40)]


def draw(img: Image.Image, boxes, scores=None, thr: float = 0.0) -> Image.Image:
    draw = Image.new("RGBA", img.size, (0, 0, 0, 0))
    from PIL import ImageDraw

    d = ImageDraw.Draw(draw)
    for i, b in enumerate(boxes):
        s = 1.0 if scores is None else float(scores[i])
        if s < thr:
            continue
        # 框是归一化 xyxy：x 乘宽、y 乘高，四个分量各乘各的
        x1 = float(b[0]) * img.size[0]
        y1 = float(b[1]) * img.size[1]
        x2 = float(b[2]) * img.size[0]
        y2 = float(b[3]) * img.size[1]
        c = COLORS[i % len(COLORS)]
        # 先铺半透明填充，再压边线；反过来的话边线会被填充色盖成暗红
        d.rectangle([x1, y1, x2, y2], fill=c + (int(45 * s),))
        d.rectangle([x1, y1, x2, y2], outline=c + (255,), width=2)
    return Image.alpha_composite(img.convert("RGBA"), draw)


def save_embed_heatmap(model, image, out_path: Path) -> None:
    """把最终可学习嵌入与第 4 级特征的余弦相似度画成热力图。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    x = image.unsqueeze(0)
    with torch.no_grad():
        feats = model.backbone(x)
        projs = [p(f) for p, f in zip(model.level_proj, feats)]
        e1, _ = model.sia(projs, model.embed_token.expand(1, -1))
        e2, _ = model.csp(projs, e1)
        last = projs[-1]                                  # (1, C, H, W)
        a = e2.view(1, -1, 1, 1).expand_as(last)
        sim = F.cosine_similarity(a, last, dim=1)[0]      # (H, W)
    arr = ((sim - sim.min()) / (sim.max() - sim.min() + 1e-6)).numpy()
    from PIL import Image as _Image

    scale = 8
    arr = (arr * 255).astype("uint8")
    h, w = arr.shape
    arr = Image.fromarray(arr).resize((w * scale, h * scale), Image.NEAREST)
    arr.save(out_path)
    print(f"[embed-heatmap] -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pf_rpn_tiny.py")
    ap.add_argument("--checkpoint", default="outputs/latest.pth")
    ap.add_argument("--image", default=None)
    ap.add_argument("--synthesize", action="store_true", help="没有图片时用合成场景")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--vis-thr", type=float, default=0.05)
    ap.add_argument("--out", default="outputs/demo.png")
    ap.add_argument("--visualize-embed", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(cfg.get("device", "auto"))

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = PFRPN(ckpt.get("cfg", cfg)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[load] {args.checkpoint}")

    size = cfg["data"]["img_size"]
    if args.image:
        img = Image.open(args.image).convert("RGB").resize((size, size), Image.BILINEAR)
        t = torch.from_numpy(np.asarray(img, dtype="float32") / 255.0).permute(2, 0, 1)
    else:
        rng = np.random.default_rng(123)
        t, gt = generate_scene(rng, size=size, max_objs=cfg["data"]["max_objs"])
        img = Image.fromarray((t.permute(1, 2, 0).numpy() * 255).astype("uint8"))
        print("[gt] 真实框:", [[round(v, 3) for v in b] for b in gt.tolist()])
    t = t.unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(t)
        boxes, scores = model.inference(t, topk_list=(args.topk,))
    b, s = boxes[0][: args.topk], scores[0][: args.topk]
    print("[proposals] 分数从高到低：")
    for i in range(min(len(b), 10)):
        x1, y1, x2, y2 = box_cxcywh_to_xyxy(b[i]).tolist()
        print(f"  #{i + 1} score={float(s[i]):.4f} box=[{x1:.3f},{y1:.3f},{x2:.3f},{y2:.3f}]")

    # draw 期待 xyxy，inference 出来的是 cxcywh，必须先转。
    # 这里不用 try/except 兜底：坐标顺序错了就应该当场炸，而不是画出一堆错框。
    b_xyxy = [box_cxcywh_to_xyxy(v).tolist() for v in b]
    assert all(v[2] >= v[0] and v[3] >= v[1] for v in b_xyxy), (
        "提案框不是合法的 xyxy——检查是否漏了 box_cxcywh_to_xyxy"
    )
    out_img = draw(img, b_xyxy, s.tolist(), thr=args.vis_thr)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_img.save(out_path)
    print(f"[demo] -> {out_path}")

    if args.visualize_embed:
        save_embed_heatmap(model, t[0].cpu(), Path("outputs") / "embed_heatmap.png")


if __name__ == "__main__":
    main()
