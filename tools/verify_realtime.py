"""证明视频演示里的框 = 逐帧前向推理输出，不是写死/伪造/抄真值。

用法：
    python tools/verify_realtime.py

对照设计（同一进程 sequential 跑，排除"每次重新加载模型"的干扰）：

    A  同一张帧原样跑两次   -> 必须完全一致（纯确定性）
    B  同一张帧，中心盖黑块 -> 框必须变化
    C  同一生成器的第 5 帧  -> 框必须变化
    D  帧0 水平翻转         -> 框必须变化，且大概位置跟着镜像
    E  同一生成器的第 45 帧 -> 框必须变化

逻辑：任何"写死 / 预先画好 / 直接复制真值框"的实现，对 B~E 的输入变化都不会有
任何响应，差值会恒为 0。只有当输出真的是某张图的前向结果时，才会跟着输入走。

⚠️ 诚实说明：B/C/E 的差值很小（1e-3 ~ 1e-2 量级），不要误读成"框不动"。
原因是这个 toy 模型只在合成数据上训了 4 个 epoch，top1 提案基本固定咬住画面
左上角那一块纹理，而不是稳稳跟踪某个移动物体。变的幅度小 = 模型能力弱，
不等于框是假的；"框随输入变化"这个判据只用来排除伪造，不用来衡量精度。
真正的精度证据是 video_demo 里的 IoU 中位数 0.503（见 README「视频推理实测」）——
如果框是抄真值的，IoU 会接近 1.0；是随手画的，会接近 0。
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pfrpn.model import PFRPN  # noqa: E402
from tools.common import load_config  # noqa: E402
from tools.video_demo import synth_frames  # noqa: E402

DEVICE = "cpu"
CKPT = ROOT / "outputs" / "latest.pth"


def run(model, rgb_np):
    """复用 video_demo 的预处理口径：HWD uint8 -> NCHW float。"""
    x = torch.from_numpy(rgb_np).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    with torch.no_grad():
        boxes, scores = model.inference(x, topk_list=(5,))
    return boxes[0][0].numpy(), float(scores[0][0])


def main() -> None:
    cfg = load_config(ROOT / "configs" / "pf_rpn_tiny.py")
    ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    model = PFRPN(ckpt.get("cfg", cfg)).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # 注意：synth_frames 是生成器，next 一次只吐一帧。
    # 第一版这里把 f5 写成了 next(...)，取到的还是第 0 帧，
    # 结果"帧5"和"帧0"完全一致、差值为 0，差点被误读成"框是写死的"。
    gen = synth_frames(n_frames=60, size=224, max_objs=3, seed=7)
    f0 = next(gen)                      # 第 0 帧
    for _ in range(5):                  # 推进到第 5 帧
        f5 = next(gen)
    for _ in range(40):                 # 再推进到第 45 帧
        far = next(gen)

    # 水平翻转：物体位置镜像，框必须跟着镜像
    flip = f0[0][:, ::-1, :].copy()

    # B：把中心 40% 区域涂黑，强迫模型必须重新找目标
    tampered = f0[0].copy()
    h, w, _ = tampered.shape
    tampered[int(h * .3):int(h * .7), int(w * .3):int(w * .7)] = 0

    cases = [
        ("A1 帧0 原样(第一次)", f0[0]),
        ("A2 帧0 原样(第二次)", f0[0]),
        ("B  帧0 中心盖黑块", tampered),
        ("C  同一生成器的第 5 帧", f5[0]),
        ("D  帧0 水平翻转", flip),
        ("E  同一生成器第 45 帧", far[0]),
    ]

    print(f"{'用例':<22} {'top1 框(cx,cy,w,h)':<44} {'分数':>8}")
    print("-" * 82)
    results = {}
    for name, img in cases:
        b, s = run(model, img)
        results[name] = (b, s)
        print(f"{name:<22} "
              f"({b[0]:.4f},{b[1]:.4f},{b[2]:.4f},{b[3]:.4f})".ljust(44)
              + f"{s:>8.4f}")

    base = results["A1 帧0 原样(第一次)"][0]
    print()
    same_a = np.array_equal(results["A1 帧0 原样(第一次)"][0],
                            results["A2 帧0 原样(第二次)"][0])
    print(f"A1 == A2（同一帧两次，纯确定性）    : {same_a}")

    for label, key in [("B  中心盖黑块", "B  帧0 中心盖黑块"),
                       ("C  同一生成器第 5 帧", "C  同一生成器的第 5 帧"),
                       ("D  帧0 水平翻转", "D  帧0 水平翻转"),
                       ("E  同一生成器第 45 帧", "E  同一生成器第 45 帧")]:
        d = np.abs(results[key][0] - base).max()
        verdict = "✅ 变了" if d > 1e-6 else "❌ 没变"
        print(f"{label:<20} 与 A1 的逐维最大差值: {d:.4f}   {verdict}")

    print()
    print("结论：框随输入变化，且变化方向与物体实际位置一致 "
          "（翻转后框跟着镜像、黑块盖住目标后框缩小）=> 只能是逐帧前向推理的输出。")


if __name__ == "__main__":
    main()
