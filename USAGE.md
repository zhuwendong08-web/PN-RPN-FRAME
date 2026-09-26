# PF-RPN 使用说明（交付版）

> 面向「只想把它用起来，不想读论文」的人。
> 详细的架构解读、消融实验、踩坑记录都在 **[README.md](README.md)**。
> 这份只讲**怎么用**，以及**用的时候会遇到什么问题**。

---

## 0. 三分钟上手

```bash
# 看看有哪些权重能用
python -m pfrpn

# 跑一张图
python -m pfrpn predict outputs/demo.png --topk 10

# 画框存图
python -m pfrpn draw outputs/demo.png outputs/result.png --topk 10

# 跑视频
python -m pfrpn video clip.mp4 -o outputs/clip_out.mp4
```

Python 里：

```python
from pfrpn.api import PF

pf = PF(checkpoint="outputs/best.pth")       # 建一次，之后反复用
boxes, scores = pf.predict_file("a.jpg", topk=10)
```

跑通了吗？看到 `[predict] xxx.png -> 10 个提案` 就通了。

---

## 1. 命令行：`python -m pfrpn`

| 命令 | 作用 | 常用参数 |
|---|---|---|
| `predict <图>` | 出提案，打印分数和框 | `--topk N`（默认 10）、`--thr X` |
| `draw <图> <输出>` | 画框存图 | `--topk N`、`--thr X` |
| `video <视频>` | 逐帧出提案，可选出带框视频 | `-o/--out`、`--max-frames`、`--skip`、`--thr` |
| `evaluate` | 验证集上算 AR / AP50 | `--checkpoint`、`--topk` |
| `train` | 训练 | `--epochs N` |
| `checkpoints` | 列出可用权重 | — |

**不加 `checkpoints` 参数直接跑 `python -m pfrpn` 就是列权重**，忘了参数时不会报错。

训练：

```bash
python -m pfrpn train --epochs 4                          # 默认 tiny 配置
python -m pfrpn train --epochs 20                         # 想训好一点就加长
```

---

## 2. 权重怎么选（最容易踩的坑）

仓库里有两组权重，**用途完全不同**，别混用：

| 权重 | 训的什么 | 适合 |
|---|---|---|
| `outputs_person_v4/best.pth` | COCO person 子集微调 | **检测人**。默认会优先挑它 |
| `outputs/best.pth` | 合成数据（随机色块 + 人形） | 通用检测 / 想复现论文基线 |

命令行里会自动挑 `person_v4`，并打印一行说明：

```
[load] ...\outputs_person_v4\best.pth  （微调「检测人」的权重）
```

**想用通用权重就显式指定**：

```bash
python -m pfrpn predict outputs/demo.png --checkpoint outputs/best.pth
```

---

## 3. 输出的框到底是什么

**归一化 cxcywh**，四个数都是 0~1 之间的小数：

```
box = [cx, cy, w, h]
      ↑    ↑    ↑  ↑
    框中心x 框中心y   宽 高
```

举例 `box=[0.45, 0.56, 0.79, 0.84]`：框中心在画面 45% 处、56% 处，宽占画面 79%、高占 84%。

要像素坐标，用自带的换算：

```python
from pfrpn.api import to_pixels
x1, y1, x2, y2 = to_pixels(box, width=1280, height=720)
```

反过来，要把像素框喂给模型用 `from_pixels(box_px, w, h)`。

分数 `score` 是 0~1，越大越可信。**实际用的时候 `thr` 设 0.05~0.3 通常比较干净**；设成 0 会把全部 100 个提案都画出来，而它们的分数挤在一起看不出差别。

---

## 4. Python 调用

```python
from pfrpn.api import PF

pf = PF(checkpoint="outputs/best.pth")

# --- 单张图 ---
boxes, scores = pf.predict_file("a.jpg", topk=20, thr=0.05)

# --- 直接喂 PIL / numpy / torch，不用先存盘 ---
from PIL import Image
import numpy as np

b, s = pf.predict(Image.open("a.jpg"))                       # PIL
b, s = pf.predict(np.asarray(Image.open("a.jpg")))           # numpy RGB
b, s = pf.predict(torch.randn(3, 224, 224).clamp(0, 1))      # torch (3,H,W)

# --- 一整个目录 ---
res = pf.predict_dir("D:/photos", topk=20, thr=0.05)         # {文件名: (boxes, scores)}

# --- 出带框图 ---
arr = pf.draw(Image.open("a.jpg"), boxes, scores, thr=0.05)
Image.fromarray(arr).save("out.png")

# --- 视频 ---
stats = pf.predict_video("clip.mp4", out="clip_out.mp4", topk=10, max_frames=60)
print(stats)     # {"frames": 60, "fps": 14.96, "top1_mean": ..., ...}

# --- 评测 ---
res = pf.evaluate(topk=(100, 300))
```

** Screening 建议：建一次 `PF`，后面反复调用。** 每次新建都会重新加载权重（几秒），
而且内部已经做了预热，第一次推理不会慢。

---

## 5. 训练 / 评测

```bash
python -m pfrpn train --epochs 4                     # 约 17 分钟（CPU，tiny 配置）
python -m pfrpn evaluate --checkpoint outputs/best.pth
```

评测输出长这样：

```
[eval] 验证集 128 张
[eval] AR@100=0.7285  AR@300=0.7285  ARs=0.0755  ARm=0.8203  ARl=0.9444  AP50=0.0322
```

怎么看：

* **AR@100 = 0.7285** —— 每张图出 100 个提案，其中能被某个真值框匹配上的比例。
  这是论文的主指标。
* **ARs / ARm / ARl** —— 小 / 中 / 大目标分别的召回。ARs 只有 0.0755 说明
  **小目标还很弱**，这是该模型的已知短板。
* **AR@300 与 AR@100 相同** —— 因为 `num_queries=100`，池子里没有第 101 个提案。
* **AP50 = 0.0322** 偏低 —— 提案召回不错，但单个框的定位/分类精度还不够。

---

## 6. FAQ

**Q：框画出来了，但没框到人，怎么办？**
A：确认用了 person 权重（`--checkpoint outputs_person_v4/best.pth`）。
另外模型只在 224×224 的输入上训过，特别小的人（画面里几像素高）基本框不住——
这是训练数据的局限，不是 bug。

**Q：为什么框位置看着歪的？**
A：先确认坐标口径。本项目内部**一律是归一化 cxcywh**，画图时用 `to_pixels()`。
如果你拿到的框和 PIL/OpenCV 直接给的 `x,y,w,h` 混淆了，画面会完全错位——
本项目在这个坑上栽过三次，详见 README 第 14 / 19 / 22 条。

**Q：很慢？**
A：纯 CPU、224px 下单帧约 **65~107 ms（10~15 FPS）**。想快就：
`--skip 2`（隔帧推理）、`--max-frames N`（只跑一段）、`--topk` 调小。
注意**不要拿第一帧的耗时当性能指标**——第一帧比后面慢好几倍，这是 PyTorch
 lazy init + oneDNN 首次编译造成的，`PF()` 构造时已经自动预热了。

**Q：跑视频报 `needs opencv`？**
A：`pip install opencv-python-headless`。

**Q：能不能 batch 推理？**
A：`pf.predict_dir()` 已经是串行的逐图推理。要真正的整批前向，用底层的
`pf.model.inference(torch.stack(...))`，但它不做 resize。

**Q：`python -m pfrpn` 在别的目录跑不了？**
A：要在仓库根目录跑，或者先 `pip install -e .`。
脚本靠"仓库根"这个相对位置去找权重和配置。

---

## 7. 交付时已知的限制（别对它抱不切实际的期待）

1. **不是检测器，是提案网络（RPN）。** 它不输出类别，只框"大概是物体"。
   想要带类别的检测结果得接一个下游分类器。
2. **在真实视频里框人目前效果有限。** 训练集里的人占画面比例较大，
   远景小目标基本漏。真实素材建议先裁切放大再喂。
3. **小目标召回很低**（ARs 0.0755）。
4. **分类分数区分度弱**：前 10 个提案的分数可能都挤在 0.14~0.17，
   排序能用，绝对数值不能当置信度看。
5. **合成数据训的权重（outputs/best.pth）别用到真实图片上**，
   domain 差太远。通用权重只是用来复现论文基线的。
6. **视频没有跨帧关联**，每帧独立推理，同一物体在相邻帧会轻微跳动。

详细的消融实验与 bug 记录见 README §10（诚实清单）和 §11（微调可行性）。
