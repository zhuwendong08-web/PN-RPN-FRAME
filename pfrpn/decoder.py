"""DETR 式 Transformer 解码器。

原版 PF-RPN 直接复用 Grounding DINO 的跨模态解码器，只是把「文本嵌入」换成
「可学习视觉嵌入 F~^T」。因此这里保留标准 DETR 解码器结构：
每层 = query 自注意力 + query 对 memory 的交叉注意力（K 上叠加位置编码）+ FFN。

query 数量即论文中的 ``num_queries``（评估时取 100 / 300 / 900）。
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_ff)
        self.linear2 = nn.Linear(dim_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # 交叉注意力 K 位置注入：与 Grouping DINO 一致，对 K 加位置编码
        self.pos_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        pos: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q = query
        # --- query 自注意力 ---
        q2, _ = self.self_attn(q, q, q, need_weights=False)
        q = q + self.dropout(q2)
        q = self.norm1(q)

        # --- 对 image memory 的交叉注意力 ---
        k = memory + self.pos_proj(pos)
        q3, _ = self.cross_attn(q, k, memory, attn_mask=attn_mask, need_weights=False)
        q = q + self.dropout(q3)
        q = self.norm2(q)

        # --- FFN ---
        q4 = self.linear2(F.relu(self.linear1(q), inplace=True))
        q = q + self.dropout(q4)
        q = self.norm3(q)
        return q


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_ff: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, nhead, dim_ff, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        pos: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """query : (B, N, d_model)
        memory : (B, S, d_model)
        pos    : (B, S, d_model)
        """
        for layer in self.layers:
            query = layer(query, memory, pos, attn_mask)
        return self.norm(query)
