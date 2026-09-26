"""按需下载 COCO person 子集对应的图片 —— 不碰整个 18.2 GB 的 train2017 包。

原理：COCO 每张图都有独立 URL
    http://images.cocodataset.org/train2017/{file_name}
而 file_name 就在标注文件里，所以只需要标注 + 下载真正要用的那几千张。

用法：
    python tools/fetch_coco.py --n 5000 --out data/coco/imgs
    python tools/fetch_coco.py --n 5000 --out data/coco/imgs --workers 16 --overwrite

断点续传：已存在且大小正常的文件直接跳过；--overwrite 强制重下。
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = "http://images.cocodataset.org/train2017/"
MIN_BYTES = 4096          # 小于这个值基本是 404 的 XML 错误页，不是真图
RETRIES = 3
UA = {"User-Agent": "Mozilla/5.0 (compatible; pfrpn-fetch/1.0)"}


def fetch_one(item):
    """下载一张图，返回 (file_name, ok, msg)。"""
    fn = item["file_name"]
    dst = item["_dst"]
    if dst.exists() and dst.stat().st_size >= MIN_BYTES:
        return fn, True, f"skip ({dst.stat().st_size:,}B)"
    tmp = dst.with_suffix(dst.suffix + ".part")

    # 用 requests 会更好写，但这台机器没装；标准库 urllib 够用，少一个依赖。
    # 必须带 User-Agent，否则 COCO 的 S3 网关会把默认 Python-urllib UA 挡掉。
    req = urllib.request.Request(BASE + fn, headers=UA)
    for attempt in range(RETRIES):
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                data = resp.read()
        except urllib.error.HTTPError as e:
            return fn, False, f"HTTP {e.code}"
        except Exception as e:    # noqa: BLE001
            if attempt == RETRIES - 1:
                return fn, False, f"{type(e).__name__}: {e}"
            time.sleep(1.5 * (attempt + 1))
            continue

        if len(data) < MIN_BYTES:
            return fn, False, f"too small ({len(data)}B，多半是 404 的 XML)"
        dst.write_bytes(data)      # 直接写最终名：进程被杀也不会留下半个 .part
        return fn, True, f"{len(data):,}B"
    return fn, False, "exhausted retries"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000, help="要下载多少张（从子集头部取）")
    ap.add_argument("--out", default="data/coco/imgs")
    ap.add_argument("--json", default="data/coco_person_mini.json")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    data = json.loads((ROOT / args.json).read_text(encoding="utf-8"))

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    # 必须按 image id 排序后再取前 n 张。COCOLikeDataset 是按 sorted(meta.keys())
    # 切分的，如果这里按原始顺序取，两边选出的图就不是同一个集合——
    # 结果就是报"图缺失"，而且症状看起来像下载失败，很容易误判。
    data["images"].sort(key=lambda im: im["id"])

    items = []
    for im in data["images"][: args.n]:
        items.append({"file_name": im["file_name"],
                      "_dst": out / im["file_name"]})
    if args.overwrite:
        for it in items:
            it["_dst"].unlink(missing_ok=True)

    total = sum(1 for _ in items)
    print(f"[plan ] 目标 {total} 张 -> {out}")
    print(f"[plan ] 并发 {args.workers} | 已存在将自动跳过 | 预计总大小 "
          f"{sum(1 for _ in items) * 0.15 / 1024:.1f} GB 量级")
    print()

    ok, fail = 0, 0
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch_one, it): it["file_name"] for it in items}
        for fu in as_completed(futs):
            fn, good, msg = fu.result()
            done += 1
            if good and not msg.startswith("skip"):
                ok += 1
            elif good:
                ok += 1
            else:
                fail += 1
            if done % 50 == 0 or done == total:
                el = time.time() - t0
                rate = done / el if el > 0 else 0
                eta = (total - done) / rate if rate > 0 else 0
                print(f"  [{done}/{total}] ok={ok} fail={fail} "
                      f"| {rate:.1f} 张/s | 已用 {el/60:.1f} min "
                      f"| 预计剩余 {eta/60:.1f} min")
                sys.stdout.flush()

    el = time.time() - t0
    print()
    print(f"[done ] 成功 {ok} / 失败 {fail}，用时 {el/60:.1f} min，"
          f"平均 {ok/el*1024/1024:.2f} MB/s" if el > 0 else "[done]")
    if fail:
        print("  !! 失败的图在下次运行时会自动重试")


if __name__ == "__main__":
    main()
