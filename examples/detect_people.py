"""批量检测人：扫一个目录，把带框结果存到另一个目录，并导出 csv。

改下面两个变量即可。目录里放 .jpg / .png 都行。

运行：
    python examples/detect_people.py --src D:/some/folder --dst D:/some/out
"""
from __future__ import annotations

import argparse
from pathlib import Path

from pfrpn.api import PF

EXT = ("*.jpg", "*.jpeg", "*.png", "*.bmp")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/imgs", help="输入的图片目录")
    ap.add_argument("--dst", default="outputs/detect_out", help="输出的图片目录")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--thr", type=float, default=0.05,
                    help="分数门槛，低于这个不画；调高能少画点废框")
    ap.add_argument("--checkpoint", default="outputs_person_v4/best.pth",
                    help="检测人用 person 权重，通用检测用 outputs/best.pth")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    if not src.is_dir():
        raise SystemExit(f"!! 不是目录：{src}")
    dst.mkdir(parents=True, exist_ok=True)

    pf = PF(checkpoint=args.checkpoint)
    print(f"[load] {pf.checkpoint}")

    files = [p for e in EXT for p in sorted(src.glob(e))]
    print(f"[data] {len(files)} 张图")

    rows = 0
    for i, p in enumerate(files, 1):
        b, s = pf.predict_file(p, topk=args.topk, thr=args.thr)

        # 存带框图
        arr = pf.draw(p, b, s, thr=args.thr)
        from PIL import Image
        Image.fromarray(arr).save(dst / f"{p.stem}.png")

        # 存坐标。want 像素坐标的话用 pfrpn.api.to_pixels(box, w, h)
        for bb, ss in zip(b, s):
            rows += 1
            print(f"  {p.name},{bb[0]:.5f},{bb[1]:.5f},{bb[2]:.5f},"
                  f"{bb[3]:.5f},{float(ss):.5f}")

    print(f"[done] {len(files)} 张图 -> {dst}，共 {rows} 个提案")


if __name__ == "__main__":
    main()
