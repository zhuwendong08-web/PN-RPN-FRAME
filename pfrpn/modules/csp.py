"""CSP —— Cascade Self-Prompt（论文 Section 3.3，Eq.2 / Eq.3）

论文原文：

    M_i     = 1( cos( F~_{i-1}^T , F_i^I ) > delta )                     (2)
    F~_i^T  = F~_{i-1}^T + MAP( M_i , F_i^I )                            (3)

其中 ``delta`` 为手工阈值（论文设为 0.3），``MAP`` 为带掩码的平均池化。

设计要点
--------
1. **为什么要级联**：论文 Fig.8 的余弦相似度热力图证明——「物体内部特征」比
   「可学习 embedding 本身」定位能力更强（embedding 响应更弥散）。
   所以把已激活的物体特征 iteratively 喂回 embedding，让它不断「自我提示」净化。
2. **从深到浅**：深层编码高层语义、浅层保留结构细节，先聚合语义再补细节。
3. **代价极低**：论文 Table 9 显示迭代 1 -> 3 只增加约 4.6 ms/图（CD-FSOD）。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CascadeSelfPrompt(nn.Module):
    """Cascade Self-Prompt。

    参数
    ----
    channel   : 特征通道数
    threshold : 相似度阈值 delta（论文 0.3）
    num_iter  : 级联轮数（论文 3；Table 9 消融 1/2/3）
    order     : 每一轮遍历的层级顺序，索引由深到浅。默认 (3, 2, 1, 0)
    """

    def __init__(
        self,
        channel: int,
        threshold: float = 0.3,
        num_iter: int = 3,
        order: Tuple[int, ...] = (3, 2, 1, 0),
    ) -> None:
        super().__init__()
        self.channel = channel
        self.threshold = float(threshold)
        self.num_iter = int(num_iter)
        self.order = tuple(order)

    def forward(
        self,
        feats: List[torch.Tensor],
        embed: torch.Tensor,
        return_masks: bool = False,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """参数
        ----
        feats : 长度 4 的特征列表，每项 (B, C, H, W)
        embed : F~_0^T，来自 SIA 的输出 (B, C)

        返回
        ----
        embed_out : (B, C)  最终用于分类打分的可学习嵌入 F~_4^T
        masks     : 每一轮每一步生成的相似度掩码，用于可视化
        """
        f = embed
        masks: List[torch.Tensor] = []

        for _ in range(self.num_iter):
            for li in self.order:
                feat_i = feats[li]                                    # (B,C,H,W)
                a = f.unsqueeze(-1).unsqueeze(-1).expand_as(feat_i)  # (B,C,H,W)
                sim = F.cosine_similarity(a, feat_i, dim=1)          # (B,H,W)

                mask = (sim > self.threshold).to(feat_i.dtype)       # (B,H,W)
                counts = mask.flatten(1).sum(-1).clamp(min=1.0)      # (B,)
                pooled = (feat_i * mask[:, None]).flatten(2).sum(-1) / counts[:, None]
                # 严格按论文 Eq.3 做残差累加
                f = f + pooled
                if return_masks:
                    masks.append(mask.detach())

        if return_masks:
            return f, masks
        return f, []


def masked_avg_pool(feat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """通用版：mask 为 (B,H,W) 的 0/1 权重，feat 为 (B,C,H,W)。"""
    counts = mask.flatten(1).sum(-1).clamp(min=1.0)        # (B,)
    return (feat * mask[:, None, None, :]).flatten(2).sum(-1) / counts[:, None]


if __name__ == "__main__":  # pragma: no cover
    m = CascadeSelfPrompt(32, threshold=0.3, num_iter=3)
    f = [torch.randn(2, 32, s, s) for s in (32, 16, 8, 4)]
    out, masks = m(f, torch.randn(2, 32), return_masks=True)
    print("CSP out:", tuple(out.shape), "masks:", len(masks), tuple(masks[0].shape))
