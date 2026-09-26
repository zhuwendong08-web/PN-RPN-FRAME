"""数据集：合成场景 + COCO 风格 json。

为什么自带合成数据
------------------
原版 PF-RPN 需要 COCO 5% + ImageNet 5% 才能训练（论文 Table 3）。对没有数据集、
也没有 GPU 的场景，这里提供一个合成「场景下物体的矩形」生成器：
背景随机渐变 + 噪声，物体是随机颜色/尺寸/位置的实心矩形。

关键性质：**物体永远只有一类 ``object``**，正好对应 PF-RPN 的 prompt-free
单类开放集设定——模型不需要任何类别名或样例图就能学会「这里有东西」。
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import write_png

try:
    from PIL import Image  # noqa: F401
except Exception:  # pragma: no cover
    Image = None


# --------------------------------------------------------------------------- #
# 合成生成器
# --------------------------------------------------------------------------- #
def generate_scene(
    rng: np.random.Generator, size: int = 224, max_objs: int = 4
):
    """返回 (image_tensor(3,H,W), boxes(M,4) 归一化 cxcywh)。"""
    img = np.zeros((size, size, 3), dtype=np.float32)
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32)
    # 背景：随机主色 + 双向渐变 + 噪声
    base = rng.random(3).astype(np.float32) * 0.2
    img[..., 0] += base[0] + (ys / size) * 0.18
    img[..., 1] += base[1] + (xs / size) * 0.18
    img[..., 2] += base[2] + rng.normal(0.0, 0.05, (size, size)).astype(np.float32)
    img += rng.normal(0.0, 0.02, img.shape).astype(np.float32)

    boxes: List[List[float]] = []
    for _ in range(max_objs):
        w = float(rng.uniform(0.10, 0.38))
        h = float(w * rng.uniform(0.6, 1.7))
        cx = float(rng.uniform(w / 2, 1 - w / 2))
        cy = float(rng.uniform(h / 2, 1 - h / 2))
        color = rng.random(3).astype(np.float32)
        alpha = float(rng.uniform(0.55, 1.0))
        x0, x1 = int((cx - w / 2) * size), int((cx + w / 2) * size)
        y0, y1 = int((cy - h / 2) * size), int((cy + h / 2) * size)
        if x1 <= x0 or y1 <= y0:
            continue
        sub = img[y0:y1, x0:x1]
        sub *= (1.0 - alpha)
        sub += color * alpha
        # 30% 概率加内部纹理，避免所有物体长得一样
        if rng.random() < 0.3:
            sub[:: max(2, sub.shape[0] // 6), :] *= 0.6
            sub[:, :: max(2, sub.shape[1] // 6)] *= 0.7
        boxes.append([cx, cy, w, h])

    img = np.clip(img, 0.0, 1.0)
    return torch.from_numpy(img).permute(2, 0, 1).contiguous(), torch.tensor(boxes, dtype=torch.float32)


def dump_synthetic_split(
    root: str | Path,
    split: str = "train",
    n_images: int = 512,
    img_size: int = 224,
    max_objs: int = 4,
    seed: int = 0,
) -> dict:
    """生成一份合成数据并落盘成 COCO 风格 json，返回 json 内容。"""
    root = Path(root)
    img_dir = root / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed + (1000 if split == "val" else 0))

    images, annotations, ann_id, ok = [], [], 1, 0
    for i in range(n_images):
        img_t, boxes = generate_scene(rng, size=img_size, max_objs=max_objs)
        if boxes.numel() == 0:
            continue
        name = f"{split}_{i:05d}.png"
        try:
            write_png((img_t * 255).to(torch.uint8).permute(1, 2, 0).cpu(), str(img_dir / name))
        except Exception:
            # torchvision 的 write_png 对某些路径/后端不支持时，退回 PIL
            from PIL import Image as _Image

            _Image.fromarray((img_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)).save(
                img_dir / name
            )
        ok += 1
        images.append({"id": i, "file_name": name, "height": img_size, "width": img_size})
        for b in boxes.tolist():
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": i,
                    "bbox": b,                       # cxcywh，与 COCO 一致
                    "area": max(b[2] * b[3], 0.0),
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    data = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "object", "supercategory": ""}],
    }
    (root / f"{split}.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    print(f"[synthetic] {split}: {ok}/{n_images} 张图，{len(annotations)} 个标注 -> {root}")
    return data


class COCOLikeDataset(Dataset):
    """读取 COCO 风格 json。若 json 不存在，自动合成一份并缓存到磁盘。

    json 结构::

        {"images":[{"id","file_name","height","width"}],
         "annotations":[{"image_id","bbox":[cx,cy,w,h] | [x,y,w,h]}],
         "categories":[{"id":1,"name":"object"}]}

    两种 bbox 口径都支持，靠 ``bbox_fmt`` 区分：

    * ``"cxcywh"``（默认）—— 本项目 ``dump_synthetic_split`` 写的格式，归一化。
    * ``"xywh_px"`` —— **COCO 官方格式**，绝对像素的 ``[x, y, w, h]``。
      微调真实数据时用这个，转换在 ``__init__`` 里一次性做完。

    ``img_dir`` 可另指图片目录：真实 COCO 的图是按需下载的，不在
    ``root/images`` 下（``tools/fetch_coco.py`` 把它们下到了 ``data/coco/imgs``）。
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        img_size: int = 224,
        max_objs: int = 4,
        n_images: int = 512,
        seed: int = 0,
        force_synth: bool = False,
        img_dir: str | Path | None = None,
        bbox_fmt: str = "cxcywh",
        n_images_cap: int | None = None,
        id_offset: int = 0,
        json_file: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.img_size = img_size
        self.bbox_fmt = bbox_fmt
        # 默认读 <root>/<split>.json；微调真实 COCO 时要指到别的标注文件
        # （例如 coco_person_mini.json），否则会静默去读合成数据。
        json_path = Path(json_file) if json_file else self.root / f"{split}.json"
        if (force_synth or not json_path.exists()) and not json_file:
            dump_synthetic_split(
                self.root, split, n_images=n_images, img_size=img_size,
                max_objs=max_objs, seed=seed,
            )
        data = json.loads(json_path.read_text(encoding="utf-8"))
        self.img_dir = Path(img_dir) if img_dir else self.root / "images"

        # 预建 图id -> 标注列表
        self.meta: Dict[int, dict] = {
            im["id"]: im for im in data["images"]
        }
        self.bboxes: Dict[int, torch.Tensor] = {}
        for a in data["annotations"]:
            iid = a["image_id"]
            if iid not in self.meta:
                continue            # 标注指向的图片不在子集里，直接丢
            self.bboxes.setdefault(iid, []).append(a["bbox"])

        # 注意要遍历快照：循环里会 pop，直接迭代 dict 会报
        # `dictionary keys changed during iteration`。
        for iid in list(self.bboxes.keys()):
            raw = self.bboxes.pop(iid)
            boxes = _normalize_boxes(
                raw, self.meta[iid], bbox_fmt, max_objs
            )
            if boxes.numel():
                self.bboxes[iid] = boxes

        # 按图 id 排序后切一段。train 用 [0:train_n]、val 用 [train_n:train_n+val_n]，
        # 这样验证集与训练集**完全不相交**——如果两者都从 0 开始截，同一张图会
        # 既进训练集又进验证集，评测出来的 AR 会被记忆污染，数字虚高。
        #
        # ⚠️ 这套"靠 id 偏移切段"的约定只对**真实数据集**成立（json 里的图 id
        # 是按 [0:train_n] + [train_n:] 编排的，person/full 配置显式写了
        # ``val_offset``）。合成数据（``dump_synthetic_split``）的 val 图 id 是
        # 独立编号的 ``0..val_n-1``，和 train 段重叠；此时如果 id_offset 回退成
        # train_n（512），``sorted(ids)[512:]`` 直接把 128 张 val 图全切没，
        # 数据集长度变成 0。
        #
        # 空数据集是**极危险**的状态：``evaluate_proposals`` 里每个 bucket 都是
        # 空 list，``sum(v)/len(v)`` 走 ``if v else 0.0`` 兜底 → 静默返回
        # ``AR@100 = 0.0000``。看上去"模型完全没学会"，实际是一次都没评测过。
        # 更糟的是 train.py 的 ``if res[key] > best: save(best.pth)``——
        # ``0.0 > -1.0`` 成立，**空评测结果会拿 epoch1 的权重覆盖 best.pth**，
        # 直接把历史最好的权重毁掉。所以这里必须硬失败，不能静默。
        self.ids: List[int] = sorted(self.meta.keys())
        if n_images_cap is not None:
            self.ids = self.ids[id_offset: id_offset + n_images_cap]
        if not self.ids:
            raise RuntimeError(
                f"{self.split} 数据集是空的：id_offset={id_offset} "
                f"n_images_cap={n_images_cap} 时，可用图 id 共 "
                f"{len(self.ids)} 个（meta 里一共 {len(self.meta)} 个）。"
                "检查 data 段：合成数据必须写 val_offset=0，"
                "只有真实 COCO 那种 [0:train_n]/[train_n:] 编排才需要 val_offset。"
            )
        self.image_paths = [
            self.img_dir / self.meta[i]["file_name"] for i in self.ids
        ]

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> dict:
        iid = self.ids[idx]
        path = self.image_paths[idx]
        img_t = _load_image(path, self.img_size)      # (3,H,W) float [0,1]
        # 不能直接写 `if b else`：b 是 Tensor，超过一个元素时
        # `if tensor([...])` 会报 `Boolean value of Tensor with more than one value
        # is ambiguous`。合成数据里每图最多 1 个框所以一直没触发，
        # 一到真实 COCO（一张图好几个人）就崩。
        b = self.bboxes.get(iid)
        if b is None or b.numel() == 0:
            boxes = torch.zeros(0, 4)
        else:
            boxes = torch.tensor(b.tolist(), dtype=torch.float32).reshape(-1, 4)
        return {"image": img_t, "boxes": boxes, "labels": torch.ones(boxes.shape[0], dtype=torch.long)}


def _normalize_boxes(
    raw: List[list], meta: dict, bbox_fmt: str, max_objs: int
) -> torch.Tensor:
    """把一批标注统一成**归一化 cxcywh**，并按面积取前 ``max_objs`` 个。

    两件事必须在真实数据上做，合成数据里都不存在：

    1. **口径转换**。COCO 官方是 ``[x, y, w, h]`` 的绝对像素，本项目训练时喂的是
       归一化 cxcywh。漏了这步会让框的坐标大出几佰倍，模型直接学废——而且是静默的。
    2. **数量截断**。``max_objs`` 是模型侧的硬约束（decoder 的 query 上限相关），
       超出的框必须丢。这里按面积降序保留最大的 ``max_objs`` 个：COCO person
       子集平均每图 4.09 个人，正是 ``max_objs=4`` 的默认值，所以正常情况不会触发；
       但密集人群图（ stadium 里几十号人）一张就能给出 30+ 个框，不截断会炸。
    """
    W = float(meta.get("width", 1.0))
    H = float(meta.get("height", 1.0))
    if W <= 0 or H <= 0:
        W = H = 1.0

    cxcywh: List[List[float]] = []
    for b in raw:
        if bbox_fmt == "xywh_px":
            x, y, w, h = (float(v) for v in b)
            # 明确是像素口径，必须除图像尺寸
            to_norm = True
        else:
            # "cxcywh" 有两种可能：**像素**（真实 COCO）或**已归一化**
            # （本项目合成数据，``dump_synthetic_split`` 直接把
            # ``generate_scene`` 的归一化框写进 json）。
            # ⚠️ 这里曾无条件除以 W/H，于是合成数据被二次缩小 224 倍：
            # 真值框从 0.63 变成 0.0028，全挤成画面左上角一个点。
            # 症状极具迷惑性——``loss_bbox`` 很低（模型确实"学会"复现那个点了）
            # 但 ``loss_giou`` 高到 1.83，两者在数学上互斥，逼着人去查这一行。
            # 后果：tiny 默认路径 4 epoch 训完 AR@100 = 0.0000。
            a, bb, c, d = (float(v) for v in b)
            x, y, w, h = a - c / 2, bb - d / 2, c, d
            # 启发式：归一化口径下 w/h 必然 < 1（合成数据 w∈[0.10,0.38]），
            # 像素口径下不可能全小于 1.5。
            to_norm = max(c, d) > 1.5
        w = max(w, 0.0)
        h = max(h, 0.0)
        # ⚠️ 必须**先归一化再钳位**。这个顺序曾写反：原先在像素尺度上就
        # `cx = min(max(cx, 0.0), 1.0)`，于是 min(max(322.885, 0), 1.0) 直接
        # 把每一个框的中心都压成 1.0 —— 所有真值框被钉死在画面右下角，
        # 模型照着这份坏数据训练，出来的框当然全在角落。
        # 症状：诊断脚本显示连 GT 落点都 100% 在右下角，一开始误判成"数据本身就靠边"。
        if to_norm:
            cx = min(max((x + w / 2) / W, 0.0), 1.0)
            cy = min(max((y + h / 2) / H, 0.0), 1.0)
            cxcywh.append([cx, cy, w / W, h / H])
        else:
            cx = min(max(x + w / 2, 0.0), 1.0)
            cy = min(max(y + h / 2, 0.0), 1.0)
            cxcywh.append([cx, cy, w, h])

    if not cxcywh:
        return torch.zeros(0, 4)
    arr = np.asarray(cxcywh, dtype=np.float32)
    order = np.argsort(-(arr[:, 2] * arr[:, 3]))   # 面积降序
    arr = arr[order[:max_objs]]
    return torch.from_numpy(arr).contiguous()


def _load_image(path: Path, size: int) -> torch.Tensor:
    from PIL import Image

    with Image.open(path) as im:
        im = im.convert("RGB").resize((size, size), Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def collate(batch: List[dict]):
    """合成一个 batch。

    合成数据里每张图的框数恒等于 ``max_objs``，直接 ``torch.stack`` 就行。
    真实 COCO 每张图的人数是变的（1~4），stack 会报 ``stack expects each tensor
    to be equal size``。所以这里做**左对齐 padding**：
    前 M 个位置是真实框，其余填空，并用 ``gt_mask`` 标出哪些是真的。
    loss 端靠这个 mask 把 padding 行排除掉。
    """
    boxes = [b["boxes"] for b in batch]
    M = max(b_.shape[0] for b_ in boxes)
    B = len(batch)
    padded = torch.zeros(B, M, 4, dtype=torch.float32)
    mask = torch.zeros(B, M, dtype=torch.bool)
    for i, b_ in enumerate(boxes):
        n = b_.shape[0]
        if n:
            padded[i, :n] = b_
            mask[i, :n] = True
    return {
        "image": torch.stack([b["image"] for b in batch], dim=0),
        "boxes": padded,
        "gt_mask": mask,
    }


if __name__ == "__main__":  # pragma: no cover
    d = dump_synthetic_split("data", "train", n_images=8, img_size=64)
    print("图片数:", len(d["images"]), "标注数:", len(d["annotations"]))
