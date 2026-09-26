"""轻量多尺度骨干网络。

原版 PF-RPN 直接用 Grounding DINO 的 Swin-B / ResNet 编码器输出 4 级多尺度特征。
复现时为了能在纯 CPU 上跑通，这里换成一个极小的 CNN 编码器：
输出 4 个级别，空间下采样率为 1/4、1/8、1/16、1/32。

接口与原版对齐：``forward`` 返回 list of 4 个 (B, C, H_l, W_l)。
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class _Stage(nn.Module):
    """一个阶段 = 步卷积降采样 + 两个 3x3 卷积。"""

    def __init__(self, ch_in: int, ch_out: int, stride: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TinyMultiScaleBackbone(nn.Module):
    """输出 4 级多尺度特征。"""

    def __init__(
        self,
        in_ch: int = 3,
        widths: List[int] = (32, 64, 128, 256),
        strides: List[int] = (4, 2, 2, 2),
    ) -> None:
        super().__init__()
        assert len(widths) == len(strides)
        self.widths = list(widths)
        prev = in_ch
        self.stages = nn.ModuleList()
        for w, s in zip(widths, strides):
            self.stages.append(_Stage(prev, w, s))
            prev = w

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        outs = []
        h = x
        for stage in self.stages:
            h = stage(h)
            outs.append(h)
        return outs


# --------------------------------------------------------------------------- #
# 位置编码（正弦，DETR 风格）
# --------------------------------------------------------------------------- #
def sine_positional_embeddings(
    mask: torch.Tensor, d_model: int, normalize: bool = False
) -> torch.Tensor:
    """标准 DETR 正弦位置编码。

    参数
    ----
    mask : (B, H, W)，0 处为有效像素（这里一律传全 1，只为确定分辨率）
    d_model : 输出通道数（必须是偶数）

    返回 : (B, d_model, H, W)
    """
    assert d_model % 2 == 0, "d_model 必须是偶数（每维用 sin/cos 两个通道）"
    b, h, w = mask.shape

    y_em = mask.cumsum(1, dtype=torch.float32)                  # (B,H,W) 行累计
    x_em = mask.cumsum(2, dtype=torch.float32)                  # (B,H,W) 列累计
    if normalize:                                               # 归一化到 [0,1]
        y_em = y_em / (y_em[:, -1:, :] + 1e-6)
        x_em = x_em / (x_em[:, :, -1:] + 1e-6)

    # 与 DETR 一致的对数频率
    div = 10000 ** (
        torch.arange(0, d_model, 2, dtype=torch.float32, device=mask.device) / d_model
    )                                                            # (d_model/2,)

    def _encode(emb):                                           # (B,H,W) -> (B,H,W,d_model)
        e = emb.unsqueeze(-1) * div                             # (B,H,W,d/2)
        s = torch.stack([e.sin(), e.cos()], dim=-1)             # (B,H,W,d/2,2)
        return s.flatten(start_dim=3)                           # (B,H,W,d_model)

    pe = _encode(y_em) + _encode(x_em)                          # (B,H,W,d_model)
    return pe.permute(0, 3, 1, 2)                               # -> (B,d_model,H,W)
