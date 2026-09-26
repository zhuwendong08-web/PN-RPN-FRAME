"""CG-QS —— Centerness-Guided Query Selection（论文 Section 3.4，Eq.4 / Eq.5）

论文原文：

    c_i     = sqrt( min(l,r)/max(l,r) * min(t,b)/max(t,b) )             (4)
    L_ctr   = sum_{i=1..N} || g_i - c_i ||_1
    L       = L_reg + L_cls + L_rt + lambda * L_ctr                     (5)

其中 ``g_i`` 是轻量 MLP 中心打分网络对 query i 的预测分数，``c_i`` 是中心线监督。

设计要点
--------
1. **动机（论文 Fig.7）**：靠近物体中心的 query 生成的提案框明显比靠近边界的更准。
   因此不能只按分类分数选 query，还要看 query 落在目标中心的程度。
2. **训练/推理一致**：训练时最小化 L_ctr 让打分网络学会「中心度」；
   推理时把中心度分数与分类分数结合，选出最终 query 集合（{100, 300, 900} 分别评估 AR）。
3. **lambda 很敏感**：论文 Table 8 显示 lambda=1 学不会选中心、lambda=9 则中心度损失
   主导优化损害回归，最优 lambda=5。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CenternessScorer(nn.Module):
    """轻量中心打分网络：query 特征 -> 一个标量中心度分数。"""

    def __init__(self, channel: int, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(channel, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, q_feats: torch.Tensor) -> torch.Tensor:
        """q_feats : (B, N, C) -> (B, N)"""
        return self.net(q_feats).squeeze(-1)


if __name__ == "__main__":  # pragma: no cover
    m = CenternessScorer(64)
    print("CG-QS out:", tuple(m(torch.randn(2, 32, 64)).shape))
