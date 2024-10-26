'''
Multi-head attention
- paper: Attention Is All You Need http://arxiv.org/abs/1706.03762v7
  - sec 3.2

Reference: x-transformers v1.0.0

NOTE: einops.einsum and torch.einsum input order is different
einops.einsum(A, B, 'ik,kj->ij') vs torch.einsum('ik,kj->ij', A, B)
'''

from pydantic import BaseModel, Field
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import (
    rearrange, 
    repeat,
    reduce,
    einsum
)
from typing import Optional, Tuple, Union, List

from boring_nn.attention.core import AttentionFactory, AttentionConfig
from boring_nn.attention.core import (
    PositionalEncoding,
    AttentionMask,
    TalkingHeads,
    QKNormalization,
    MemoryKeyValue,
)
from boring_utils.nn_utils import (
    default,
    exists,
    max_neg_value
)
from boring_utils.utils import cprint
from boring_utils.helpers import DEBUG


class ComputeAttention(nn.Module):
    def __init__(self, config: AttentionConfig, scale: float, dropout: nn.Dropout):
        super().__init__()
        self.config = config
        self.scale = scale
        self.dropout = dropout
        self.attention_strategy = AttentionFactory.get_strategy(config)
        if config.talking_heads: self.talking_heads = TalkingHeads(config.num_heads)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None):
        if self.config.flash_attention:
            return self._flash_attention(q, k, v, mask)
        
        dots = einsum(q, k, 'b h i d, b h j d -> b h i j') * self.scale
        if self.config.talking_heads: dots = self.talking_heads.pre_softmax(dots)
        dots = AttentionMask.apply_causal_mask(dots, self.config.causal)
        
        if exists(mask):
            mask_value = max_neg_value(dots)
            dots.masked_fill_(~mask, mask_value)
        
        attn = self.attention_strategy(dots)
        attn = self.dropout(attn)
        if self.config.talking_heads: attn = self.talking_heads.post_softmax(attn)
        
        out = einsum(attn, v, 'b h i j, b h j d -> b h i d')
        return out, attn

    def _flash_attention(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None):
        # Implement Flash Attention here
        pass


class BoringAttention(nn.Module):
    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.config = config
        assert config.d_model % config.num_heads == 0, "d_model % num_heads should be zero."

        self.num_heads = config.num_heads
        self.dim_head = config.dim_head
        # self.dim_head = int(config.d_model / config.num_heads)
        self.scale = config.dim_head ** -0.5
        
        inner_dim = config.dim_head * config.num_heads
        self.q_proj = nn.Linear(config.d_model, inner_dim, bias=config.bias)
        self.kv_proj = nn.Linear(config.d_model, inner_dim * 2, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.compute_attention = ComputeAttention(config, self.scale, self.dropout)

        # attention-on-attention mechanism
        if config.attn_on_attn:
            self.to_out = nn.Sequential(nn.Linear(inner_dim, config.d_model * 2), nn.GLU()) 
        else:
            self.to_out = nn.Sequential(nn.Linear(inner_dim, config.d_model))

        # memory key-value store
        if config.num_mem_kv > 0: self.memory_kv = MemoryKeyValue(config.num_heads, config.num_mem_kv, config.dim_head)

    def _project_qkv(self, x: Tensor, context: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]:
        kv_input = default(context, x)
        q = self.q_proj(x)
        k, v = self.kv_proj(kv_input).chunk(2, dim=-1)
        return map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), (q, k, v))

    def forward(
            self, 
            x: Tensor, 
            context: Optional[Tensor] = None, 
            mask: Optional[Tensor] = None, 
            context_mask: Optional[Tensor] = None
        ) -> Tuple[Tensor, Tensor]:
        """
        x: [batch_size, query_len, d_model] -> query
        context: [batch_size, key_len, d_model] -> key, value
        """
        batch_size, q_len = x.size(0), x.size(1)

        q, k, v = self._project_qkv(x, context)
        q, k = QKNormalization.apply(q, k, self.config)
        q, k, v = map(lambda t: PositionalEncoding.apply(t, self.config), (q, k, v))

        if self.config.num_mem_kv > 0: k, v = self.memory_kv.extend(k, v)

        mask = AttentionMask.prepare(mask, context_mask, q_len, k.size(-2), batch_size, self.num_heads)

        out, attn = self.compute_attention(q, k, v, mask)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out), attn
