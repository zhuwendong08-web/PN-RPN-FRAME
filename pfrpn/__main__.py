"""``python -m pfrpn`` 的入口。

    python -m pfrpn                       # 列出可用的权重
    python -m pfrpn predict a.jpg
    python -m pfrpn predict a.jpg --topk 20
    python -m pfrpn draw a.jpg out.png
    python -m pfrpn video clip.mp4 -o out.mp4
    python -m pfrpn video clip.mp4 -o out.mp4 --max-frames 60
    python -m pfrpn evaluate
    python -m pfrpn train --epochs 4
详细说明见 USAGE.md。
"""
import sys

from .api import main

if __name__ == "__main__":
    sys.exit(main())
