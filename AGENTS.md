# AGENTS.md — 给接手这个仓库的 AI Agent

> 这是一份**任务简报**，不是文档目录。
> 你要做的是**把这份仓库跑通**，不是通读所有文档。
> 跑通的定义见下面「验收标准」，做不到就是没完成。

---

## 1. 一句话说明

这个项目是 **PF-RPN**（Prompt-Free Universal Region Proposal Network，
arXiv:2603.17554，CVPR 2026）的**精简复现**：一个不依赖文本提示、直接出候选框的
区域提案网络。参数 2.951 M，纯 CPU 可训练可推理。
框架本身是从零写的，**不依赖 MMDetection / Detectron2 / timm**。

---

## 2. 验收标准（Done when）

**必须逐条跑出下面的结果，任何一条对不上就算没跑通。**
数字是 2026-09-26 在纯 CPU（8 核）上实测的，可复现。

| # | 命令 | 期望输出 |
|---|---|---|
| 1 | `python tests/test_modules.py` | `通过 46 / 46` |
| 2 | `python -m pfrpn` | 列出至少一条权重 |
| 3 | `python -m pfrpn train --epochs 4` | 耗时 **约 17 分钟**，最终 `AR@100=0.72852` |
| 4 | `python -m pfrpn evaluate --checkpoint outputs/best.pth --topk 100` | `AR@100=0.7285 ARs=0.0755 ARm=0.8203 ARl=0.9444 AP50=0.0322` |
| 5 | `python -m pfrpn draw outputs/demo.png outputs/out.png` | 生成 `outputs/out.png` |
| 6 | `python -m pfrpn video <任一 mp4> -o outputs/out.mp4 --max-frames 20` | 约 **58 ms/帧（≈16 FPS）** |

允许的差异：`AP50` 小数点后第 4 位有 ±0.0004 抖动（贪心匹配平局打破顺序不同）。

**如果第 3 步跑出来 `AR@100=0.0000` —— 不要去调模型。**
`0.0000` 在过去是验证集被 `id_offset` 切空导致的，现在已经修好并加了硬断言。
如果你复现出了 0.0000，说明你的环境缺文件或者改动过 `pfrpn/datasets.py`。

---

## 3. 装环境

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -U pip        # Windows
.venv/bin/python -m pip install -U pip            # Linux / macOS
.venv/Scripts/python -m pip install -e .          # Windows
.venv/bin/python -m pip install -e .              # Linux / macOS
```

`pip install -e .` 之后 `python -m pfrpn` 在**任意目录**都能跑。
不想装的话，在仓库根目录跑也可以（`pfrpn/api.py` 会自动把仓库根加进 `sys.path`）。

依赖：torch / torchvision / numpy / scipy / pillow / opencv-python-headless。
本机实测版本 **torch 2.14.0+cpu**。

---

## 4. 数据从哪来

**不需要你准备数据。**

跑训练时如果 `data/` 里没有 COCO 标注，框架会**自动生成合成数据**
（`pfrpn/datasets.py` 里的 `dump_synthetic_split`）：
512 张训练 + 128 张验证，224×224，每图最多 4 个物体。

真实数据（可选）：`python tools/fetch_coco.py` 按需下载 COCO 图，
配 `configs/pf_rpn_person*.py` 做「检测人」微调。整个 COCO 是 18.2 GB，
那个脚本只下标注需要的部分。

---

## 5. 命令速查

```bash
python -m pfrpn                                  # 列出可用权重
python -m pfrpn train --epochs 4                 # 训练（默认 tiny 配置）
python -m pfrpn evaluate --checkpoint outputs/best.pth
python -m pfrpn predict outputs/demo.png --topk 10
python -m pfrpn draw <图> <输出图>
python -m pfrpn video <视频> -o <输出.mp4> --max-frames 60

# 底层脚本（参数更多）
python tools/train.py --config configs/pf_rpn_tiny.py --epochs 4
python tools/test.py --config configs/pf_rpn_tiny.py --checkpoint outputs/best.pth
python tools/demo.py --synthesize --out outputs/demo.png
```

Python 侧：

```python
from pfrpn.api import PF
pf = PF(checkpoint="outputs/best.pth")
boxes, scores = pf.predict_file("a.jpg", topk=10)   # 归一化 cxcywh + 分数
```

---

## 6. 六个已经踩过的坑（**不要再踩一遍**）

这些全部已修复。列出它们，是因为它们的**症状都长得像"模型不行"，
实际全是数据/损失/坐标的问题**，air 很容易浪费几小时往错误方向调。

1. **`AR@100` 恒为 `0.0000`** —— `tools/train.py` 给验证集传了
   `id_offset=train_n=512`，而合成数据的 val 图 id 是 `0..127`，
   于是验证集被切空 → 空评测 → 静默返回 0。
   **现在 `pfrpn/datasets.py` 在数据集为空时直接 `RuntimeError`**，
   `evaluate_proposals` 发现空 bucket 也会 raise。
   如果你又看到 0.0000，先查 `len(val_ds)`，别动模型。

2. **合成数据真值框被二次归一化** —— `_normalize_boxes` 里 `cxcywh` 分支
   无条件除了一次 224，真值框从 0.63 变成 0.002。
   症状：`loss_bbox` 很低但 `loss_giou` 高到 1.83 —— **两者互斥**，这个矛盾才是线索。

3. **中心度头一步都没训动** —— `ctr` 损失除以全部 100 个 query 而不是正样本数，
   负样本 ±0.05 的残余主导了数值，梯度符号同号比例只有 0.44（≈抛硬币），
   而 Adam 按符号更新 → 净漂移为 0。`cgqs.net.2.bias` 训完仍是初始值 -0.0504。
   诊断方法：**对比训练前后某个 bias 是否等于初始值**，比看 loss 更早暴露问题。

4. **所有框落在画面角落** —— 和上面同源（真值框全被钉在右下角）。
   修好后靠边格占比 100% → 0%。
   教训：**当所有输出呈现极端规律时，先查数据管道，别急着调超参。**

5. **钳位顺序写反** —— 先钳位再归一化，`min(max(322.885, 0), 1.0) == 1.0`
   把所有框中心压成 1.0。

6. **坐标系口径** —— 本项目内部一律是**归一化 cxcywh**。
   本项目在这个坑上栽了三次（`box_cxcywh_to_xyxy` 漏转换、demo 绘制、
   verify_person 把归一化框当 xyxy 用）。
   **症状都是「IoU 恒为 0」但完全不报错。**

> 通用教训（这个项目反复验证过）：
> **「指标差到像是完全没学会」时，第一个该怀疑的是自己的数据管道和坐标系，
> 不是模型。** 评分函数里 `x / n if n else 0.0` 这类兜底会把配置错误
> 伪装成模型效果差。

---

## 7. 不要动的东西（除非明确要求）

- **不要 `pip install torch` 覆盖现有版本**。本机锁 torch 2.14.0+cpu。
- **不要在没备份的情况下跑 `tools/train.py`** —— 它每轮都会写
  `outputs/best.pth` 和 `outputs/latest.pth`。之前有过一次误操作，
  把 4 epoch 的成果覆盖成 1 epoch 的。要跑破坏性实验先拷贝。
- **不要给验证集加 `id_offset`**（见坑 1）。只有真实 COCO 那种
  `[0:train_n] + [train_n:]` 编排才需要。
- **不建议给合成数据加 flip / color jitter**。合成数据本身是随机生成的，
  再加增强会让背景分布失真，实测有害。

---

## 8. 当前能力的边界（别对它有错误期待）

- 它是**提案网络（RPN），不是检测器**：不输出类别，只框「大概是物体」。
- 小目标召回很低（`ARs=0.0755`）。
- `AP50=0.0322` —— 提案召回不错，但单个框的定位/分类精度还不够。
- 真实视频里框人效果有限：训练集里的人占画面比例较大，远景小目标基本漏。
- **视频没有跨帧关联**，每帧独立推理。
- `num_queries=100`，所以 `AR@300 / AR@900` 和 `AR@100` 数值相同。

架构解读、消融实验（v1→v4）、完整踩坑记录见 **README.md**（长）。
只想用它跑起来看 **USAGE.md**。
