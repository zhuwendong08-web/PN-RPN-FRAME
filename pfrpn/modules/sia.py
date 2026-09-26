"""SIA —— Sparse Image-Aware Adapter（论文 Section 3.2，Eq.1）

论文原文：

    F~T = sum_{j=1..k} w~_{sigma(j)} * Attn(F_T, [F~_{sigma(j)}^I, F_{sigma(j)}^I])    (1)

其中：
* ``F_T``    : 可学习视觉嵌入（1 x C），取代 Grounding DINO 的文本嵌入
* ``F_i^I``  : 图像编码器第 i 级特征图
* ``F~_i^I`` : 第 i 级特征的全局平均池化向量（粗粒度线索）
* ``Router`` : 轻量 MLP，预测各特征级的重要性权重
* ``top-k``  : 只保留 k 个最相关级别（论文默认 k=2），其余置 0 再 softmax 归一化

设计要点
--------
1. **稀疏是有依据的**：论文 Table 7 消融显示 k=1 信息量不足、k=4 引入冗余噪声，k=2 最优。
2. **一次注意力注入两种粒度**：KV 由 ``[全局向量, 局部特征图]`` 拼接后线性投影得到，
   因此单次 cross-attention 就能同时拿到尺度统计与空间细节。
3. **返回路由权重 w**：供 MoE 均衡损失 ``L_rt = std(w_i), i in {1..4}`` 使用。
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseImageAwareAdapter(nn.Module):
    """Sparse Image-Aware Adapter。"""

    def __init__(
        self,
        channel: int,
        num_levels: int = 4,
        topk: int = 2,
        hidden: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.channel = channel
        self.num_levels = num_levels
        self.topk = min(topk, num_levels)
        hidden = hidden or max(channel // 2, 8)

        # MoE router：为每一级特征图打一个重要性 logit
        self.router = nn.Sequential(
            nn.Linear(channel, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_levels),
        )
        # 把 [全局向量, 局部特征图] 拼成一条 token 序列后投影为 K / V。
        # 拼接发生在 token 维，所以每个 token 的特征维仍是 C。
        self.kv_proj = nn.Linear(channel, channel)
        self.out_proj = nn.Linear(channel, channel)
        self.scale = channel ** -0.5
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, feats: List[torch.Tensor], embed: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """参数
        ----
        feats : 长度 num_levels 的列表，每项 (B, C, H_l, W_l)
        embed : 可学习嵌入 (B, C)

        返回
        ----
        embed_out : (B, C)  更新后的嵌入
        aux       : dict
            router_w : (B, num_levels) 稀疏化**前**的 router 权重，供 L_rt 用。
                       注意 L_rt 必须取这个稠密版本：稀疏化的 w 含 0，
                       其 std 会惩罚「用多级特征」这件事，与 MoE 均衡的意图相反。
            sparse_w : (B, num_levels) top-k 稀疏化后的权重（测试用）
            topk_idx : (B, k)          被选中的特征级下标
        """
        assert len(feats) == self.num_levels, (
            f"期望 {self.num_levels} 级特征，收到 {len(feats)}"
        )

        # 1) 全局平均池化 -> 粗粒度尺度向量        gstack: (L, B, C)
        gavg = [F.adaptive_avg_pool2d(f, 1).flatten(1) for f in feats]
        gstack = torch.stack(gavg, dim=0)

        # 2) router -> softmax -> top-k 稀疏化
        #    注意：router 的输出层已经输出 num_levels 个 logit，
        #    因此只需对「各层全局向量的均值」打一次分，不能再按层堆叠（否则会变成 (B,L,L)）
        pooled = gstack.mean(dim=0)                                      # (B, C)
        logits = self.router(pooled)                                     # (B, L)
        w = torch.softmax(logits, dim=-1)                                # (B, L)
        top_vals, top_idx = w.topk(self.topk, dim=-1)                    # (B, k)

        w_sparse = torch.zeros_like(w)
        w_sparse.scatter_(-1, top_idx, top_vals)                         # 非 top-k 置 0
        w_sparse = w_sparse / w_sparse.sum(-1, keepdim=True).clamp(min=1e-6)

        # 3) 对选中的每一级做 cross-attention，再按权重求和
        #    各级别空间尺寸不同，无法直接 torch.stack，这里按 (level, batch) 逐样本计算
        acc = embed.new_zeros(embed.shape[0], self.channel)

        for j in range(self.topk):
            idx = top_idx[:, j].detach().to(torch.long)                  # (B,) 该级下标
            for b in range(embed.shape[0]):
                lvl = int(idx[b])
                # KV = [该级全局向量, 该级全部空间位置]，拼成 (1 + H*W, C) 的 token 序列
                gvec = gstack[lvl, b]                                     # (C,)
                fmap = feats[lvl][b].flatten(1).t()                       # (H*W, C)
                kv = torch.cat([gvec.unsqueeze(0), fmap], dim=0)          # (1+H*W, C)
                k = self.kv_proj(kv)
                v = self.kv_proj(kv)
                # 单次 query 的缩放点积注意力（全部用 2D 张量，避免多维广播歧义）
                logits = (embed[b] * self.scale) @ k.t()                  # (1+H*W,)
                attn = torch.softmax(logits, dim=0)
                out = attn @ v                                            # (C,)
                acc[b] = acc[b] + w_sparse[b, j] * out

        aux = {"router_w": w, "sparse_w": w_sparse, "topk_idx": top_idx}
        return self.out_proj(self.dropout(acc)), aux


if __name__ == "__main__":  # pragma: no cover
    m = SparseImageAwareAdapter(64, num_levels=4, topk=2)
    f = [torch.randn(2, 64, s, s) for s in (32, 16, 8, 4)]
    out, w = m(f, torch.randn(2, 64))
    print("SIA out:", tuple(out.shape), "router w:", tuple(w.shape))
