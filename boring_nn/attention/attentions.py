'''
Multi-head attention
- paper: Attention Is All You Need http://arxiv.org/abs/1706.03762v7
  - sec 3.2

Reference: x-transformers v1.0.0
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

from boring_utils.utils import cprint
from boring_utils.helpers import DEBUG
from boring_utils.nn_utils import (
    default,
    exists,
    max_neg_value
)
from boring_nn.attention.core import AttentionFactory, AttentionConfig
from boring_nn.attention.core import (
    TalkingHeads,
    QKNormalization,
    PositionalEncoding,
    AttentionMask
)


class ComputeAttention:
    def __init__(self, config: AttentionConfig, scale: float, dropout: nn.Dropout):
        self.config = config
        self.scale = scale
        self.dropout = dropout
        self.attention_strategy = AttentionFactory.get_strategy(config)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None):
        if self.config.flash_attention:
            return self._flash_attention(q, k, v, mask)
        
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        dots = self._apply_talking_heads_pre(dots)
        dots = AttentionMask.apply_causal_mask(dots, self.config.causal)
        
        if exists(mask):
            mask_value = max_neg_value(dots)
            dots.masked_fill_(~mask, mask_value)
        
        attn = self.attention_strategy(dots)
        attn = self.dropout(attn)
        attn = self._apply_talking_heads_post(attn)
        
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        return out, attn

    def _flash_attention(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None):
        # Implement Flash Attention here
        pass

    def _apply_talking_heads_pre(self, dots: Tensor) -> Tensor:
        if self.config.talking_heads:
            return self.talking_heads.pre_softmax(dots)
        return dots

    def _apply_talking_heads_post(self, attn: Tensor) -> Tensor:
        if self.config.talking_heads:
            return self.talking_heads.post_softmax(attn)
        return attn


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

        if config.attn_on_attn:  # attention-on-attention mechanism
            self.to_out = nn.Sequential(nn.Linear(inner_dim, config.d_model * 2), nn.GLU()) 
        else:
            self.to_out = nn.Sequential(nn.Linear(inner_dim, config.d_model))

        if config.talking_heads: self.talking_heads = TalkingHeads(config.num_heads)

        if config.num_mem_kv > 0:
            self.mem_k = nn.Parameter(torch.randn(config.num_heads, config.num_mem_kv, config.dim_head))
            self.mem_v = nn.Parameter(torch.randn(config.num_heads, config.num_mem_kv, config.dim_head))

        self.compute_attention = ComputeAttention(config, self.scale, self.dropout)


    def _project_qkv(self, x: Tensor, context: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]:
        kv_input = default(context, x)
        q = self.q_proj(x)
        k, v = self.kv_proj(kv_input).chunk(2, dim=-1)
        return map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), (q, k, v))


    def _add_memory_kv(self, k: Tensor, v: Tensor) -> Tuple[Tensor, Tensor]:
        if self.config.num_mem_kv > 0:
            batch_size = k.shape[0]
            mem_k, mem_v = map(lambda t: repeat(t, 'h n d -> b h n d', b=batch_size), (self.mem_k, self.mem_v))
            k = torch.cat((mem_k, k), dim=-2)
            v = torch.cat((mem_v, v), dim=-2)
        return k, v


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

        k, v = self._add_memory_kv(k, v)

        mask = AttentionMask.prepare(mask, context_mask, q_len, k.size(-2), batch_size, self.num_heads)

        out, attn = self.compute_attention(q, k, v, mask)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out), attn
