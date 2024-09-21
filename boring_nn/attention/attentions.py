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
    TalkingHeads
)

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

        self.attention_strategy = AttentionFactory.get_strategy(config)

        if config.talking_heads:
            self.pre_softmax_proj = nn.Parameter(torch.randn(config.num_heads, config.num_heads))
            self.post_softmax_proj = nn.Parameter(torch.randn(config.num_heads, config.num_heads))

        if config.num_mem_kv > 0:
            self.mem_k = nn.Parameter(torch.randn(config.num_heads, config.num_mem_kv, config.dim_head))
            self.mem_v = nn.Parameter(torch.randn(config.num_heads, config.num_mem_kv, config.dim_head))

    def forward(self, x, context=None, mask=None, context_mask=None):
        """
        x: [batch_size, query_len, d_model] -> query
        context: [batch_size, key_len, d_model] -> key, value
        """
        batch_size, q_len = x.size(0), x.size(1)
        kv_input = default(context, x)

        q = self.q_proj(x)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), (q, k, v))

        if self.config.num_mem_kv > 0:
            mem_k, mem_v = map(lambda t: repeat(t, 'h n d -> b h n d', b=batch_size), (self.mem_k, self.mem_v))
            k = torch.cat((mem_k, k), dim=-2)
            v = torch.cat((mem_v, v), dim=-2)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        if self.config.talking_heads:
            dots = einsum('b h i j, h k -> b k i j', dots, self.pre_softmax_proj)

        if exists(mask) or exists(context_mask):
            mask = default(mask, lambda: torch.ones(b, n, device=x.device).bool())
            if not exists(context):
                context_mask = default(context_mask, mask)
            else: 
                context_mask = default(context_mask, lambda: torch.ones(batch_size, k.shape[-2], device=x.device).bool())
            mask_value = max_neg_value(dots)
            mask = mask[:, None, :, None] * context_mask[:, None, None, :]
            dots.masked_fill_(~mask, mask_value)

        if self.config.causal:
            i, j = dots.shape[-2:]
            causal_mask = torch.ones((i, j), device=device).triu_(j - i + 1).bool()
            dots.masked_fill_(causal_mask, max_neg_value(dots))

        attn = self.attention_strategy(dots)
        attn = self.dropout(attn)

        if self.config.talking_heads:
            attn = einsum('b h i j, h k -> b k i j', attn, self.post_softmax_proj)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
