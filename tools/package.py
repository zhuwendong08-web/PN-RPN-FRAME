"""打出可分发的精简包（不含 data/ 1.1G 和 outputs*/ 185M）。

    python tools/package.py                # 默认打包到 ../pfrpn-release
    python tools/package.py --out D:/tmp/x
    python tools/package.py --with-weights # 连 outputs/best.pth 一起带上

打出来的目录里跑 ``pip install -e . && python -m pfrpn train --epochs 4``
就能复现 README / AGENTS.md 里写的全部数字。
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: 总要带的文件（名字 -> 说明）
MUST_HAVE = {
    "pyproject.toml": None,
    "requirements.txt": None,
    "README.md": None,
    "USAGE.md": None,
    "AGENTS.md": None,
    "run_all.bat": None,
}

#: 整个目录拷过去（tools/ 走下面的白名单过滤）
DIRS = ["pfrpn", "configs", "tests", "examples", "tools"]

#: 只拷 tools/ 下的这些（诊断脚本 *_scratch 不该给对方，里面有一堆中间结论和过期路径）
TOOLS_KEEP = [
    "common.py", "train.py", "test.py", "demo.py", "video_demo.py",
    "verify_person.py", "fetch_coco.py", "package.py",
]


def build(out: Path, with_weights: bool = False) -> None:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    for name in MUST_HAVE:
        src = ROOT / name
        if not src.exists():
            print(f"  [跳过] {name}（不存在）")
            continue
        shutil.copy2(src, out / name)
        print(f"  [文件] {name}")

    for d in DIRS:
        src = ROOT / d
        if not src.exists():
            continue
        dst = out / d
        if d == "tools":                       # 手工过滤，别把 _scratch 带出去
            dst.mkdir(parents=True, exist_ok=True)
            for f in TOOLS_KEEP:
                s = src / f
                if s.exists():
                    shutil.copy2(s, dst / f)
                    print(f"  [工具] tools/{f}")
        else:
            shutil.copytree(src, dst)
            print(f"  [目录] {d}/")

    if with_weights:
        for rel in ("outputs/best.pth",):
            s = ROOT / rel
            if s.exists():
                (out / "outputs").mkdir(parents=True, exist_ok=True)
                shutil.copy2(s, out / "outputs" / "best.pth")
                print(f"  [权重] outputs/best.pth ({s.stat().st_size / 1e6:.1f} MB)")

    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    n = sum(1 for f in out.rglob("*") if f.is_file())
    print(f"\n[done] -> {out}  共 {n} 个文件 / {size / 1e6:.1f} MB")
    print("对方拿到后：")
    print(f"  cd {out}")
    print("  python -m venv .venv && .venv/Scripts/python -m pip install -e .")
    print("  python -m pfrpn train --epochs 4        # 约 17 分钟")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="输出目录，默认 ../pfrpn-release")
    ap.add_argument("--with-weights", action="store_true",
                    help="连 outputs/best.pth 一起带上（省去训练，但多 12 MB）")
    a = ap.parse_args()
    out = Path(a.out) if a.out else ROOT.parent / "pfrpn-release"
    build(out, a.with_weights)


if __name__ == "__main__":
    main()
