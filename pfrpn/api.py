"""PF-RPN 的统一调用入口。

散在 ``tools/*.py`` 里的能力（单图、批量、视频、评测、训练）收敛到这一个类。
用法极简：

    from pfrpn.api import PF
    pf = PF()                                  # 自动找权重
    pf.predict_file("a.jpg", topk=10)          # -> [(框, 分数), ...]
    pf.predict_video("v.mp4", out="v_out.mp4")

命令行等价物见 ``python -m pfrpn --help``。

设计约定（给不想读代码的人）：
* 所有返回的框都是**归一化 cxcywh**——四个数都在 0~1 之间，
  ``[cx, cy, w, h]``，cx/cy 是框中心。要像素坐标用 ``to_pixels()``。
* 所有分数都在 0~1 之间，越大越可信。低于 ``thr`` 的提案默认已被过滤掉。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw

from .model import PFRPN
from .utils.box import box_cxcywh_to_xyxy

try:                                # 视频读写是可选依赖
    import cv2
except ImportError:                 # pragma: no cover
    cv2 = None

__all__ = ["PF", "Pred", "list_checkpoints", "to_pixels", "from_pixels"]


# --------------------------------------------------------------------------- #
# 数据类型
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Pred:
    """一个提案。``box`` 是归一化 cxcywh，``score`` 在 0~1 之间。"""
    box: Tuple[float, float, float, float]
    score: float

    @property
    def xyxy(self) -> Tuple[float, float, float, float]:
        """转成归一化 xyxy（左上角 + 右下角），画图用这个。"""
        x, y, w, h = self.box
        return (x - w / 2, y - h / 2, x + w / 2, y + h / 2)

    def __repr__(self) -> str:      # 让 print(pred) 看得懂
        return (f"Pred(box=[{self.box[0]:.3f},{self.box[1]:.3f},"
                f"{self.box[2]:.3f},{self.box[3]:.3f}], score={self.score:.4f})")


def to_pixels(box, width: int, height: int) -> Tuple[float, float, float, float]:
    """归一化 cxcywh -> 像素 xyxy（整数边界，可直接给 OpenCV/PIL 用）。"""
    cx, cy, w, h = box
    x1 = max(0, int(round((cx - w / 2) * width)))
    y1 = max(0, int(round((cy - h / 2) * height)))
    x2 = min(width, int(round((cx + w / 2) * width)))
    y2 = min(height, int(round((cy + h / 2) * height)))
    return (x1, y1, max(x1 + 1, x2), max(y1 + 1, y2))


def from_pixels(box_px, width: int, height: int) -> Tuple[float, float, float, float]:
    """像素 xyxy -> 归一化 cxcywh，需要喂给模型时用。"""
    x1, y1, x2, y2 = box_px
    return ((x1 + x2) / 2 / width, (y1 + y2) / 2 / height,
            (x2 - x1) / width, (y2 - y1) / height)


COLORS = [(60, 60, 255), (60, 200, 120), (255, 80, 80), (255, 210, 60),
          (200, 80, 220), (60, 220, 220), (255, 140, 40)]


# --------------------------------------------------------------------------- #
# 权重查找
# --------------------------------------------------------------------------- #
#: 自动选权重时的搜索顺序（第一个存在的生效）。
#: 注意 ``*_person*`` 是「微调到能检测人」的权重，``outputs/best.pth`` 是
#: 合成数据上训的通用权重——两者不能混用，见 list_checkpoints() 的说明。
CHECKPOINT_CANDIDATES = (
    "outputs_person_v4/best.pth",
    "outputs_person_v3/best.pth",
    "outputs_person_v2/best.pth",
    "outputs_person/best.pth",
    "outputs/best.pth",
    "outputs/latest.pth",
)

DEFAULT_CONFIG_BY_CKPT = {
    "outputs_person_v4": "configs/pf_rpn_person_v4.py",
    "outputs_person_v3": "configs/pf_rpn_person_v3.py",
    "outputs_person_v2": "configs/pf_rpn_person_v2.py",
    "outputs_person": "configs/pf_rpn_person.py",
}

#: 权重目录不在上表里时的兜底配置（合成数据上的通用超参）。没有它，
#: ``PF(checkpoint='outputs/best.pth')`` 的 self.cfg 会退化成 {}，
#: evaluate() / train() 全都用不了。
DEFAULT_CONFIG = "configs/pf_rpn_tiny.py"


def list_checkpoints(root: Optional[Path] = None) -> List[str]:
    """列出仓库里能找到的权重，标出各自是哪个用途。"""
    root = Path(root) if root else _repo_root()
    found = []
    for rel in CHECKPOINT_CANDIDATES:
        p = root / rel
        if p.exists():
            who = ("微调「检测人」" if "person" in rel
                   else ("合成数据通用" if "outputs/best" in rel else "最新一轮训练"))
            found.append(f"  {rel:<32} {who}   "
                         f"({p.stat().st_size / 1e6:.1f} MB)")
    if not found:
        found.append("  (没找到任何权重，先跑一次训练：python -m pfrpn train)")
    return found


def _repo_root() -> Path:
    """pfrpn/api.py -> 仓库根目录。"""
    return Path(__file__).resolve().parent.parent


def _repo_on_path() -> None:
    """把仓库根目录加进 sys.path（幂等）。

    api 作为 ``pfrpn`` 包被 import 时，兄弟包 ``tools`` 不在路径上，
    evaluate()/train() 里的 ``from tools.common import ...`` 会失败。
    以前在每个方法里各写一遍，现在收敛到这里。
    """
    root = str(_repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)


def _resolve_checkpoint(checkpoint: Optional[str]) -> Tuple[Path, Optional[str]]:
    """确定权重路径；没指定就按 CHECKPOINT_CANDIDATES 自动挑。返回 (路径, 提示信息)。"""
    if checkpoint:
        p = Path(checkpoint)
        if not p.is_absolute():
            p = _repo_root() / p
        if not p.exists():
            raise FileNotFoundError(
                f"找不到权重文件：{p}\n"
                f"可用的在下面这些位置：\n" + "\n".join(list_checkpoints())
            )
        return p, None

    for rel in CHECKPOINT_CANDIDATES:
        p = _repo_root() / rel
        if p.exists():
            note = "微调「检测人」的权重" if "person" in rel else None
            return p, note
    raise FileNotFoundError(
        "没找到任何权重文件。先训练一次：\n"
        "    python -m pfrpn train --epochs 4\n"
        "或者指定一个：PF(checkpoint='outputs/best.pth')"
    )


# --------------------------------------------------------------------------- #
# 主体
# --------------------------------------------------------------------------- #
class PF:
    """PF-RPN 的会话对象。先建一次，之后反复调用，别每次都新建。

        pf = PF()                                  # 自动挑权重
        pf = PF(checkpoint="outputs/best.pth")     # 指定权重
        pf = PF(config="configs/pf_rpn_person.py") # 指定配置
    """

    def __init__(
        self,
        config: Optional[str] = None,
        checkpoint: Optional[str] = None,
        device: Optional[str] = None,
        img_size: Optional[int] = None,
    ) -> None:
        ckpt_path, note = _resolve_checkpoint(checkpoint)
        cfg_path = config
        if cfg_path is None:
            # 权重自带 cfg 时不一定要配文件；但显式指定能让超参更确定。
            # ckpt_path.as_posix() 保证 Windows 下也是 'a/b.pth' 的形态，便于匹配。
            key = next((k for k in DEFAULT_CONFIG_BY_CKPT
                        if ckpt_path.as_posix().startswith(k + "/")), None)
            cfg_path = DEFAULT_CONFIG_BY_CKPT.get(key) if key else None

        _repo_on_path()
        from tools.common import load_config, resolve_device, set_seed

        # ⚠️ 这里踩过一个很难发现的坑：原先写的是
        #       ckpt_path.as_posix().startswith(key + "/")
        # 但 _resolve_checkpoint 返回的是**绝对路径**（"D:/.../outputs_person_v4/best.pth"），
        # 相对前缀永远匹配不上，key 恒为 None -> cfg_path 恒为 None -> self.cfg 恒为 {}。
        # 后果是 evaluate() 在几百行之外抛一个毫无提示的 KeyError('root')。
        # 改成按**权重所在目录名**匹配，绝对路径和相对路径都成立。
        if cfg_path is None:
            cfg_path = DEFAULT_CONFIG_BY_CKPT.get(
                ckpt_path.parent.name, DEFAULT_CONFIG)
        self.cfg_path = cfg_path
        self.cfg = load_config(cfg_path) if cfg_path else {}
        self.device = resolve_device(device or self.cfg.get("device", "auto"))
        self.img_size = int(img_size or self.cfg.get("data", {}).get("img_size", 224))

        ck = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        model = PFRPN(ck.get("cfg", self.cfg)).to(self.device)
        model.load_state_dict(ck["model"])
        model.eval()

        self.model = model
        self.checkpoint = str(ckpt_path)
        self._note = note
        self.warmup()

    def warmup(self, n: int = 2) -> None:
        """预热：用随机噪声空跑几帧。

        ⚠️ 不开这个的话，**第一帧会比后面慢好几倍**（PyTorch 的算子是惰性
        init 的，oneDNN 也会在第一个 batch 上编译）。实测首帧 555 ms、
        预热后 106 ms —— 不预热会让人误判成「这个 API 很慢」。
        代价只有几十毫秒，建对象时自动做了。
        """
        with torch.no_grad():
            for _ in range(n):
                self.model(torch.rand(1, 3, self.img_size, self.img_size))

    # ------------------------------------------------------------------ #
    # 底层：喂一个 (3, H, W) 的张量，拿回提案
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def predict(
        self,
        image: "Image.Image | np.ndarray | torch.Tensor",
        topk: int = 100,
        thr: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """单张图推理。返回 ``(boxes, scores)``，都是 numpy，已按分数降序。

        ``boxes`` 形状 ``(K, 4)``，每行是归一化 cxcywh。
        ``scores`` 形状 ``(K,)``，0~1，越大越可信。
        ``thr`` 是分数门槛，低于它的提案会被丢掉（默认 0，即全留）。
        """
        t = self._to_tensor(image)                      # (3, S, S) float [0,1]
        t = t.unsqueeze(0).to(self.device)
        boxes, scores = self.model.inference(t, topk_list=(100,), return_all=True)
        b = boxes[0].detach().cpu().numpy().astype(np.float32)
        s = scores[0].detach().cpu().numpy().astype(np.float32)

        if thr > 0:
            keep = s >= thr
            b, s = b[keep], s[keep]
        if topk and topk < len(s):
            b, s = b[:topk], s[:topk]
        return b, s

    def predict_p(self, *a, **kw) -> List[Pred]:
        """同 predict()，但返回 ``[Pred(box, score), ...]``，更好直接打印。"""
        b, s = self.predict(*a, **kw)
        return [Pred(tuple(map(float, bb)), float(ss)) for bb, ss in zip(b, s)]

    # ------------------------------------------------------------------ #
    # 文件 / 目录
    # ------------------------------------------------------------------ #
    def predict_file(
        self, path: str, topk: int = 100, thr: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """对一张图片文件推理，同上返回 ``(boxes, scores)``。"""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"找不到图片：{p}")
        im = Image.open(p).convert("RGB")
        return self.predict(im, topk=topk, thr=thr)

    def predict_dir(
        self, path: str, topk: int = 100, thr: float = 0.0,
        pattern: str = "*.jpg",
    ) -> "dict[str, Tuple[np.ndarray, np.ndarray]]":
        """对一个目录里的所有图片批量推理，返回 ``{文件名: (boxes, scores)}``。"""
        d = Path(path)
        if not d.is_dir():
            raise NotADirectoryError(f"不是目录：{d}")
        out = {}
        for p in sorted(d.glob(pattern)):
            try:
                out[p.name] = self.predict_file(p, topk=topk, thr=thr)
            except Exception as e:                     # 单张失败不该中断整批
                out[p.name] = e
                print(f"  [跳过] {p.name}: {e}", file=sys.stderr)
        return out

    # ------------------------------------------------------------------ #
    # 画框
    # ------------------------------------------------------------------ #
    def draw(
        self,
        image,
        boxes: np.ndarray,
        scores: Optional[np.ndarray] = None,
        thr: float = 0.0,
        width: int = 3,
    ) -> np.ndarray:
        """在图上画提案，返回 RGB numpy 数组，可直接 ``cv2.imwrite``。"""
        im = self._to_pil(image)
        W, H = im.size
        overlay = Image.new("RGBA", im.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(overlay)
        shown = 0
        for i in range(len(boxes)):
            if scores is not None and float(scores[i]) < thr:
                continue
            x1, y1, x2, y2 = to_pixels(boxes[i], W, H)
            c = COLORS[shown % len(COLORS)]
            d.rectangle([x1, y1, x2, y2], fill=c + (46,), outline=c + (255,), width=width)
            shown += 1
        return np.asarray(Image.alpha_composite(im.convert("RGBA"), overlay).convert("RGB"))

    def draw_file(self, src: str, dst: str, topk: int = 20, thr: float = 0.0) -> str:
        """读一张图、画框、另存。返回保存路径。"""
        b, s = self.predict_file(src, topk=topk, thr=thr)
        out = np.asarray(self._to_pil(src))
        arr = self.draw(out, b, s, thr=thr)
        dst_p = Path(dst)
        dst_p.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(arr).save(dst_p)
        return str(dst_p)

    # ------------------------------------------------------------------ #
    # 视频
    # ------------------------------------------------------------------ #
    def predict_video(
        self,
        path: str,
        out: Optional[str] = None,
        topk: int = 10,
        thr: float = 0.05,
        max_frames: int = 0,
        fps_out: int = 10,
        skip: int = 1,
        stats_json: Optional[str] = None,
    ) -> dict:
        """逐帧推理一个视频，可选输出带框 mp4。返回统计信息。

        ``skip>1`` 表示每隔几帧推理一次（中间帧不画框），能明显提速。
        """
        if cv2 is None:
            raise RuntimeError(
                "读/写视频需要 opencv，装一下：pip install opencv-python-headless"
            )
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise FileNotFoundError(f"打不开视频：{path}")

        n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[video] {path}: {w}x{h}, {n_total} 帧")

        writer = None
        if out:
            out_p = Path(out)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_p), fourcc, fps_out, (w, h))

        t0 = time.time()
        n_done = 0
        score_trace: List[float] = []
        ms_trace: List[float] = []
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if skip > 1 and idx % skip != 0:
                idx += 1
                continue                                  # 跳过帧也要照贴图
            if max_frames and n_done >= max_frames:
                break

            t_f = time.time()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            im = Image.fromarray(rgb).resize((self.img_size, self.img_size), Image.BILINEAR)
            b, s = self.predict(im, topk=topk, thr=thr)
            ms_trace.append((time.time() - t_f) * 1000)
            if len(s):
                score_trace.append(float(s[0]))

            if writer is not None:
                arr = self.draw(rgb, b, s, thr=thr)
                writer.write(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))

            n_done += 1
            if n_done % 20 == 0:
                print(f"  帧 {n_done}/{n_total or '?'} | top1={score_trace[-1]:.4f} "
                      f"| {ms_trace[-1]:.0f} ms/帧", flush=True)
            idx += 1
        cap.release()
        if writer is not None:
            writer.release()
            print(f"[video] -> {out}")

        elapsed = time.time() - t0
        stats = {
            "frames": n_done,
            "elapsed_sec": round(elapsed, 2),
            "ms_per_frame": round(sum(ms_trace) / max(len(ms_trace), 1), 2),
            "fps": round(n_done / elapsed, 2) if elapsed > 0 else 0.0,
            "top1_mean": round(float(np.mean(score_trace)), 4) if score_trace else 0.0,
            "top1_range": round(float(np.ptp(score_trace)), 4) if score_trace else 0.0,
        }
        if stats_json:
            Path(stats_json).parent.mkdir(parents=True, exist_ok=True)
            Path(stats_json).write_text(
                json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        print("[video] " + json.dumps(stats, ensure_ascii=False))
        return stats

    # ------------------------------------------------------------------ #
    # 训练 / 评测（包装 tools/ 下的脚本，参数沿用配置文件）
    # ------------------------------------------------------------------ #
    def evaluate(self, split: str = "val", topk: Sequence[int] = (100,)) -> dict:
        """在验证集上算 AR@K / AP50，返回字典。"""
        _repo_on_path()
        from torch.utils.data import DataLoader
        from pfrpn.datasets import collate
        from tools.common import build_dataset, evaluate_proposals, format_results

        dcfg = self.cfg.get("data")
        # 别再让 KeyError('root') 这种看不懂的错误蹦出来（曾经因为权重目录名
        # 匹配不上配置文件，self.cfg 退化成 {}，报错点离根因几百行远）。
        if not dcfg or "root" not in dcfg:
            raise RuntimeError(
                "拿不到数据配置（self.cfg_path="
                f"{self.cfg_path!r}，self.cfg keys={sorted(self.cfg.keys())}）。\n"
                "大概率是权重目录名没在 DEFAULT_CONFIG_BY_CKPT 里，或对应的 "
                f"configs/*.py 不存在。已注册的映射：\n  "
                + "\n  ".join(f"{k} -> {v}" for k, v in DEFAULT_CONFIG_BY_CKPT.items())
                + "\n显式指定配置可绕过：PF(checkpoint=..., config='configs/pf_rpn_tiny.py')"
            )
        # ⚠️ 数据集构建统一走 tools/common.py::build_dataset，与 train.py 完全一致。
        # 曾经这里自己拼参数、漏了 4 个，把真实 COCO 人权重的评测跑在合成
        # data/val.json 上（AR@100 0.6050 -> 0.4180），差点当成模型退化。
        ds = build_dataset(self.cfg, split)
        loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate)
        print(f"[eval] 验证集 {len(ds)} 张")
        res = evaluate_proposals(self.model, loader, topk_list=tuple(topk),
                                 device=self.device)
        print("[eval] " + format_results(res))
        return res

    def train(self, epochs: int = 4, batch_size: Optional[int] = None,
              resume: Optional[str] = None) -> None:
        """起一次训练。参数默认读配置，只覆盖 epoch / batch_size。"""
        _repo_on_path()
        import subprocess
        cmd = [sys.executable,
               str(_repo_root() / "tools" / "train.py"),
               "--config", self.cfg_path or "configs/pf_rpn_tiny.py",
               "--epochs", str(epochs)]
        if batch_size:
            cmd += ["--batch-size", str(batch_size)]
        if resume:
            cmd += ["--resume", resume]
        print("[train] " + " ".join(cmd))
        subprocess.run(cmd, check=True)

    # ------------------------------------------------------------------ #
    # 输入格式归一化
    # ------------------------------------------------------------------ #
    def _to_tensor(self, image) -> torch.Tensor:
        """PIL / numpy(HWC uint8) / (3,H,W) float / 路径 -> (3, S, S) float [0,1]。"""
        im = self._to_pil(image)
        if im.size != (self.img_size, self.img_size):
            im = im.resize((self.img_size, self.img_size), Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()

    def _to_pil(self, image) -> "Image.Image":
        """统一转成 RGB 的 PIL 图像。

        ⚠️ 这里**不做 BGR -> RGB 的自动判断**。numpy 数组里谁是 RGB、谁是 BGR
        从数组本身看不出来；本项目内部一律按 **RGB** 处理（PIL 读进来的、
        ``cv2.cvtColor(BGR2RGB)`` 之后的、合成数据生成的，都是 RGB）。
        OpenCV 的 ``imread`` 出来的才是 BGR，那种请用 ``cv2.cvtColor(...,
        cv2.COLOR_BGR2RGB)`` 转一下再传进来。
        """
        if isinstance(image, (str, Path)):
            return Image.open(image).convert("RGB")
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[0] == 3 and image.shape[2] != 3:
                image = image.transpose(1, 2, 0)        # (3, H, W) -> HWC
            if image.dtype != np.uint8:                 # 多半是 float [0,1]
                image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
            return Image.fromarray(image).convert("RGB")
        if isinstance(image, torch.Tensor):
            if image.dtype != torch.uint8:
                image = (image.detach().float().clamp(0, 1) * 255).to(torch.uint8)
            if image.dim() == 3 and image.shape[0] == 3:
                image = image.permute(1, 2, 0)
            return Image.fromarray(image.numpy()).convert("RGB")
        raise TypeError(
            f"不支持的输入类型 {type(image).__name__}。"
            "支持：图片路径 / PIL.Image / numpy 数组 / torch 张量（一律 RGB）"
        )


# --------------------------------------------------------------------------- #
# 命令行
# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    """``python -m pfrpn`` 的入口。"""
    ap = argparse.ArgumentParser(
        prog="python -m pfrpn",
        description="PF-RPN 命令行。不记参数也能跑：默认用 outputs/best.pth。",
    )
    sub = ap.add_subparsers(dest="cmd")

    p_i = sub.add_parser("predict", help="对一张图出提案")
    p_i.add_argument("image")
    p_i.add_argument("--topk", type=int, default=10)
    p_i.add_argument("--thr", type=float, default=0.0)
    p_i.add_argument("--checkpoint", default=None)

    p_v = sub.add_parser("video", help="对视频出提案并输出带框 mp4")
    p_v.add_argument("video")
    p_v.add_argument("-o", "--out", default=None)
    p_v.add_argument("--topk", type=int, default=10)
    p_v.add_argument("--thr", type=float, default=0.05)
    p_v.add_argument("--max-frames", type=int, default=0)
    p_v.add_argument("--skip", type=int, default=1)
    p_v.add_argument("--stats-json", default=None)
    p_v.add_argument("--checkpoint", default=None)

    p_d = sub.add_parser("draw", help="读一张图、画框、另存")
    p_d.add_argument("image"); p_d.add_argument("out")
    p_d.add_argument("--topk", type=int, default=20)
    p_d.add_argument("--thr", type=float, default=0.0)
    p_d.add_argument("--checkpoint", default=None)

    p_e = sub.add_parser("evaluate", help="在验证集上算 AR / AP50")
    p_e.add_argument("--checkpoint", default=None)
    p_e.add_argument("--config", default=None)
    p_e.add_argument("--topk", default="100,300")

    p_t = sub.add_parser("train", help="训练")
    p_t.add_argument("--epochs", type=int, default=4)
    p_t.add_argument("--checkpoint", default=None)
    p_t.add_argument("--config", default=None)

    sub.add_parser("checkpoints", help="列出可用的权重文件")

    args = ap.parse_args(argv)

    if args.cmd == "checkpoints" or args.cmd is None:
        print("可用的权重：")
        print("\n".join(list_checkpoints()))
        return 0

    if args.cmd == "evaluate":
        pf = PF(config=getattr(args, "config", None), checkpoint=args.checkpoint)
        print(f"[load] {pf.checkpoint}"
              + (f"  ({pf._note})" if pf._note else ""))
        pf.evaluate(topk=tuple(int(x) for x in args.topk.split(",")))
        return 0

    if args.cmd == "train":
        pf = PF(config=args.config, checkpoint=args.checkpoint)
        pf.train(epochs=args.epochs)
        return 0

    pf = PF(checkpoint=args.checkpoint)
    # 自动挑权重时把用途讲清楚，免得拿人权重去测合成图还以为是通用权重。
    tag = "（微调「检测人」的权重）" if "person" in pf.checkpoint else "（合成数据上的通用权重）"
    print(f"[load] {pf.checkpoint}  {tag}")

    if args.cmd == "predict":
        b, s = pf.predict_file(args.image, topk=args.topk, thr=args.thr)
        print(f"[predict] {args.image} -> {len(b)} 个提案")
        for i, (bb, ss) in enumerate(zip(b, s)):
            print(f"  #{i + 1:>3} score={float(ss):.4f} box=["
                  f"{bb[0]:.4f},{bb[1]:.4f},{bb[2]:.4f},{bb[3]:.4f}]")
        return 0

    if args.cmd == "draw":
        dst = pf.draw_file(args.image, args.out, topk=args.topk, thr=args.thr)
        print(f"[draw] -> {dst}")
        return 0

    if args.cmd == "video":
        pf.predict_video(args.video, out=args.out, topk=args.topk, thr=args.thr,
                          max_frames=args.max_frames, skip=args.skip,
                          stats_json=args.stats_json)
        return 0

    ap.print_help()
    return 0


if __name__ == "__main__":        # pragma: no cover
    raise SystemExit(main())
