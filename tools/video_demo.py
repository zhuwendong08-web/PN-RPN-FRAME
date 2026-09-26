"""视频推理：用自己实现的 PF-RPN 逐帧跑提案，输出带框视频。

    # 真实视频
    python tools/video_demo.py --checkpoint outputs/latest.pth --video clip.mp4

    # 没有视频时用自带的合成动态片段（物体在画面里移动，并画出真实框做对照）
    python tools/video_demo.py --checkpoint outputs/latest.pth --synthesize --frames 120

    # 只处理前 60 帧并统计
    python tools/video_demo.py --checkpoint outputs/latest.pth --video clip.mp4 --max-frames 60

只依赖本仓库的 pfrpn/，不碰 mmdet。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    import cv2
except ImportError:                 # pragma: no cover
    cv2 = None

from pfrpn.model import PFRPN                       # noqa: E402
from pfrpn.utils.box import box_cxcywh_to_xyxy      # noqa: E402
from tools.common import load_config, resolve_device  # noqa: E402


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    """两个归一化 xyxy 的 IoU。"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return float(inter / ua) if ua > 0 else 0.0

COLORS = [(255, 60, 60), (60, 200, 120), (80, 120, 255), (255, 210, 60),
          (200, 80, 220), (60, 220, 220), (255, 140, 40)]


# --------------------------------------------------------------------------- #
# 合成动态片段
# --------------------------------------------------------------------------- #
def synth_frames(
    n_frames: int, size: int, max_objs: int = 4, seed: int = 7
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """生成带运动的合成帧，返回 (RGB uint8 (H,W,3), 真实框 M×4 归一化 cxcywh)。

    与 datasets.generate_scene 的区别：这里物体会**移动**、背景渐变会**缓慢漂移**，
    所以每一帧之间是有相关性的连续画面，而不是静态图反复采样。
    视频里的运动是检验时间稳定性的前提——静态图连放看不出丢帧/跳变。
    """
    rng = np.random.default_rng(seed)
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32)

    # 每个物体的初始位置/速度/颜色固定，之后按帧推进
    objs = []
    for _ in range(max_objs):
        w = float(rng.uniform(0.12, 0.34))
        h = float(w * rng.uniform(0.6, 1.6))
        objs.append(dict(
            x=float(rng.uniform(0.0, 1.0 - w)),
            y=float(rng.uniform(0.0, 1.0 - h)),
            w=w, h=h,
            vx=float(rng.uniform(-0.010, 0.010)),
            vy=float(rng.uniform(-0.008, 0.008)),
            color=rng.random(3).astype(np.float32),
        ))
    bg_base = rng.random(3).astype(np.float32) * 0.16
    bg_dir_a, bg_dir_b = rng.random(2), rng.random(2)

    for t in range(n_frames):
        # 背景：双向渐变缓慢漂移 + 固定噪声纹理
        img = np.empty((size, size, 3), dtype=np.float32)
        img[..., 0] = bg_base[0] + bg_dir_a[0] * (ys / size) * 0.20 \
            + 0.02 * np.sin(2 * np.pi * xs / size + t * 0.05)
        img[..., 1] = bg_base[1] + bg_dir_a[1] * (xs / size) * 0.20 \
            + 0.02 * np.sin(2 * np.pi * ys / size + t * 0.04)
        img[..., 2] = bg_base[2] + bg_dir_b[0] * ((xs + ys) / (2 * size)) * 0.16
        img += rng.normal(0.0, 0.015, img.shape).astype(np.float32)

        boxes: List[List[float]] = []
        for o in objs:
            o["x"] += o["vx"]
            o["y"] += o["vy"]
            # 撞到边界就反弹，避免物体飘出画面后凭空消失
            if o["x"] < 0.0 or o["x"] + o["w"] > 1.0:
                o["vx"] = -o["vx"]
                o["x"] = float(np.clip(o["x"], 0.0, 1.0 - o["w"]))
            if o["y"] < 0.0 or o["y"] + o["h"] > 1.0:
                o["vy"] = -o["vy"]
                o["y"] = float(np.clip(o["y"], 0.0, 1.0 - o["h"]))

            x0, y0 = o["x"] * size, o["y"] * size
            x1, y1 = (o["x"] + o["w"]) * size, (o["y"] + o["h"]) * size
            sub = img[int(y0):int(y1), int(x0):int(x1)]
            if sub.size == 0 or sub.shape[0] == 0 or sub.shape[1] == 0:
                continue
            sub *= 0.35
            sub += o["color"] * 0.65
            boxes.append([o["x"] + o["w"] / 2, o["y"] + o["h"] / 2, o["w"], o["h"]])

        yield (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8), \
            np.asarray(boxes, dtype=np.float32).reshape(-1, 4)


# --------------------------------------------------------------------------- #
# 画面绘制
# --------------------------------------------------------------------------- #
def draw_frame(
    frame_rgb: np.ndarray,
    boxes_norm_xyxy: np.ndarray,
    scores: List[float],
    thr: float,
    gt_boxes: Optional[np.ndarray] = None,
) -> np.ndarray:
    """在**原尺寸**帧上画框。

    两个参数都是**归一化 xyxy**；``gt_boxes`` 调用前需由 cxcywh 转换过来。

    坐标映射说明：推理前的预处理是把 (W,H) 拉伸缩放到 (size,size)，
    这种 resize 不改变归一化坐标，所以
        x_pixel = nx * W_orig ,  y_pixel = ny * H_orig
    直接用原图宽高换算即可，不要再乘 size。

    注意：训练/评测时用的是**拉伸**到正方形的图，长宽比失真是有意保留的
    （与原版固定 224² 输入一致），所以这里的框也是拉伸坐标系下的，属预期行为。
    """
    img = Image.fromarray(frame_rgb)
    W, H = img.size
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)

    def _px(b, W, H):
        """归一化 xyxy -> 像素四元组，越界部分裁掉。

        必须先按 [0,1] 裁、再乘宽高：物体部分出画时 x1 会大于 x0，
        不裁的话 PIL 直接抛 `x1 must be greater than or equal to x0`。
        """
        x1, y1, x2, y2 = (float(v) for v in b)
        x1, x2 = max(0.0, min(1.0, x1)), max(0.0, min(1.0, x2))
        y1, y2 = max(0.0, min(1.0, y1)), max(0.0, min(1.0, y2))
        if x2 <= x1 or y2 <= y1:
            return None
        return x1 * W, y1 * H, x2 * W, y2 * H

    if gt_boxes is not None and len(gt_boxes):
        for b in gt_boxes:
            r = _px(b, W, H)
            if r is None:
                continue
            d.rectangle(list(r), outline=(0, 220, 90, 200), width=2)

    shown = 0
    for i, b in enumerate(boxes_norm_xyxy):
        if float(scores[i]) < thr:
            continue
        r = _px(b, W, H)
        if r is None:
            continue
        x1, y1, x2, y2 = r
        c = COLORS[shown % len(COLORS)]
        s = float(scores[i])
        d.rectangle([x1, y1, x2, y2], fill=c + (int(45 * min(s, 1.0)),))
        d.rectangle([x1, y1, x2, y2], outline=c + (255,), width=2)
        shown += 1

    return np.asarray(Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB"))


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pf_rpn_tiny.py")
    ap.add_argument("--checkpoint", default="outputs/latest.pth")
    ap.add_argument("--video", default=None, help="输入视频路径")
    ap.add_argument("--synthesize", action="store_true",
                    help="没有视频时，用自带合成动态片段")
    ap.add_argument("--frames", type=int, default=120, help="合成模式的帧数")
    ap.add_argument("--size", type=int, default=None,
                    help="覆盖 config 里的 img_size")
    ap.add_argument("--out", default=None, help="输出视频路径")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--thr", type=float, default=0.05, help="提案分数阈值")
    ap.add_argument("--max-frames", type=int, default=0, help="最多处理多少帧，0=全部")
    ap.add_argument("--fps-out", type=int, default=10)
    ap.add_argument("--skip", type=int, default=1, help="每隔几帧推理一次（>1 可提速）")
    # 注意写法：用互斥组 + store_true/store_false，不能写 "--gt/--no-gt"。
    # `--gt/--no-gt` 这种简写在 argparse 下会被拆成 option_string='--gt/--no-gt'
    # （usage 里显示为 `[--gt/--no-gt GT]`，还多要一个参数），于是
    # `--gt` / `--no-gt` 都传不进去：前者报 `expected one argument`，
    # 后者报 `unrecognized arguments`。互斥组是最不容易踩坑的写法。
    _gt = ap.add_mutually_exclusive_group()
    _gt.add_argument("--gt", dest="gt", action="store_true", default=True,
                     help="合成模式是否画真实框（绿色）")
    _gt.add_argument("--no-gt", dest="gt", action="store_false",
                     help="不画真实框（真实视频模式下本来就没有真值可画）")
    ap.add_argument("--save-frames-dir", default=None, help="额外导出带框帧为 png")
    ap.add_argument("--stats-json", default=None, help="把每帧分数写进 json")
    args = ap.parse_args()

    if not args.video and not args.synthesize:
        ap.error("需要 --video <路径> 或 --synthesize")
    if cv2 is None:
        raise SystemExit(
            "!! 读/写视频需要 opencv：pip install opencv-python-headless"
        )

    cfg = load_config(args.config)
    device = resolve_device(cfg.get("device", "auto"))
    size = args.size or cfg["data"]["img_size"]

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = PFRPN(ckpt.get("cfg", cfg)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[load] {args.checkpoint}")

    # ---- 打开输入 ----
    use_synth = bool(args.synthesize) or args.video is None
    if use_synth:
        n_total = args.frames
        print(f"[src  ] 合成动态片段: {n_total} 帧 @ {size}px")
    else:
        # 注意：这里不能写 `import cv2`——那会让 cv2 成为 main 的局部变量，
        # 函数体其它位置（含上面的 cv2 is None 判断）就都取不到了。模块级已导入。
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            raise SystemExit(f"!! 打不开视频: {args.video}")
        n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[src  ] {args.video}: {w}x{h}, {n_total} 帧")

    # ---- 输出 ----
    out_path = Path(args.out or "outputs/video_out.mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None

    t_all = time.time()
    n_done = 0
    score_trace: List[float] = []
    per_frame_ms: List[float] = []
    iou_trace: List[float] = []
    frame_iter = _iter_frames(args, use_synth, size)

    for idx, (frame_rgb, gt) in enumerate(frame_iter):
        if args.max_frames and n_done >= args.max_frames:
            break

        # synth_frames 吐的是归一化 cxcywh（与训练数据口径一致）；
        # 但绘制和 IoU 都按 xyxy 处理，所以必须在这里转换一次。
        # 踩过的坑：漏了这步会让绿色真值框画在错误位置、IoU 恒为 0，
        # 却不会报任何错——看起来像"模型定位很差"，其实是坐标系没对齐。
        if gt is not None and len(gt):
            gt_xyxy = box_cxcywh_to_xyxy(torch.as_tensor(gt, dtype=torch.float32))
            gt = gt_xyxy.numpy().astype(np.float32)

        t0 = time.time()
        pil = Image.fromarray(frame_rgb).convert("RGB").resize(
            (size, size), Image.BILINEAR)
        x = torch.from_numpy(np.asarray(pil, dtype=np.float32) / 255.0) \
                 .permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            boxes, scores = model.inference(x, topk_list=(args.topk,))
        boxes, scores = boxes[0], scores[0]
        b_xyxy = np.asarray([box_cxcywh_to_xyxy(v).tolist() for v in boxes])
        s_list = [float(v) for v in scores]

        painted = draw_frame(frame_rgb, b_xyxy, s_list, args.thr,
                             gt if (use_synth and args.gt) else None)

        if use_synth and len(gt):
            gts = np.asarray(gt, dtype=np.float32)
            frame_best = [max((_iou(a, g) for g in gts), default=0.0)
                          for a in b_xyxy]
            iou_trace.append(max(frame_best) if frame_best else 0.0)
        dt = (time.time() - t0) * 1000
        per_frame_ms.append(dt)
        n_done += 1
        if len(s_list):
            score_trace.append(float(s_list[0]))

        if writer is None:
            writer = _open_writer(str(out_path), painted.shape[1],
                                  painted.shape[0], args.fps_out)
        writer.write(cv2.cvtColor(painted, cv2.COLOR_RGB2BGR))

        if args.save_frames_dir and idx % 10 == 0:
            fr = Path(args.save_frames_dir)
            fr.mkdir(parents=True, exist_ok=True)
            Image.fromarray(painted).save(fr / f"f{idx:05d}.png")
        if idx % 10 == 0:
            print(f"  frame {idx:>4}/{n_total} | top1={max(s_list) if s_list else 0.0:.4f}"
                  f" | {dt:.0f} ms/f", flush=True)

    total = time.time() - t_all
    if writer is not None:
        writer.release()
    if not use_synth:
        cap.release()

    print(f"\n[done ] 处理 {n_done} 帧 | 总耗时 {total:.1f}s | "
          f"{total / max(n_done, 1):.2f}s/帧 | 约 {n_done / total:.1f} FPS")
    if score_trace:
        arr = np.asarray(score_trace)
        print(f"[score] top1: 均值 {arr.mean():.4f} / 最小 {arr.min():.4f} / "
              f"最大 {arr.max():.4f} / 抖动(极差) {arr.max() - arr.min():.4f}")
    if iou_trace:
        # 有真实框（合成模式）时才算这个——它比分数更能说明定位是否靠谱
        best = np.asarray(iou_trace)
        hit = (best >= 0.5).mean()
        print(f"[iou  ] 单帧最佳 IoU: 中位数 {np.median(best):.3f} / "
              f"均值 {best.mean():.3f} / >=0.5 的帧占比 {hit:.1%}")
    else:
        print("[iou  ] 无真实框（真实视频输入），跳过定位精度评估")
    if args.stats_json:
        Path(args.stats_json).write_text(json.dumps(
            {"top1": score_trace,
             "ms_per_frame": per_frame_ms,
             "n_frames": n_done}, ensure_ascii=False, indent=2),
            encoding="utf-8")
        print(f"[stats] -> {args.stats_json}")
    print(f"[out  ] -> {out_path}")


def _iter_frames(args, use_synth: bool, size: int):
    """统一两种输入为 (RGB ndarray, gt boxes) 迭代器。"""
    if use_synth:
        for fr in synth_frames(args.frames, size,
                               max_objs=args.topk if args.topk > 0 else 4):
            yield fr
        return

    import cv2
    cap = cv2.VideoCapture(args.video)
    idx = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if idx % max(args.skip, 1) == 0:
            yield cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), None
        idx += 1
    cap.release()


def _open_writer(path: str, w: int, h: int, fps: int):
    import cv2
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w_ = cv2.VideoWriter(path, fourcc, fps, (w, h))
    if not w_.isOpened():                       # mp4v 不支持时降级
        print("[warn ] mp4v 不可用，降级为 AVI/MJPEG")
        path = str(Path(path).with_suffix(".avi"))
        w_ = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"),
                             fps, (w, h))
    return w_


if __name__ == "__main__":
    main()
