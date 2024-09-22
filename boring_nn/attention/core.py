from enum import Enum
from pydantic import BaseModel
from typing import Optional, Tuple
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from einops import (
    rearrange, 
    repeat,
    reduce,
    einsum
)


class AttentionType(Enum):
    SOFTMAX = "softmax"
    ENTMAX15 = "entmax15"  # Sparse softmax, potentially improving interpretability
    TOPK = "topk"          # If set, use top-k sparse attention and sets the rest to zero


class AttentionTypeConfig(BaseModel):
    type: AttentionType
    sparse_topk: int = 10


class QKNormConfig(BaseModel):
    enabled: bool = False
    groups: int = 1
    scale: float = 10.0


class AttentionConfig(BaseModel):
    d_model: int = 512              # Input and output dim, usually d_model=dim_head*num_heads
    dim_head: int = 64              # Dimension of each attention head
    num_heads: int = 8              # Number of attention heads
    dropout: float = 0.             # Dropout rate
    bias: bool = False              # Whether to use bias in qkv linear projections
    causal: bool = False            # Whether to apply a causal mask to attention weights
    num_mem_kv: int = 0             # Number of memory key/value pairs, concated to the input kv
    talking_heads: bool = False     # Learned linear projections before and after the softmax
    attn_on_attn: bool = False      # Modified Attention-on-attention mechanism
    flash_attention: bool = False   # Kernelized attention mechanism
    rotary_pos_emb: bool = False    # RoPE positional embeddings
    attn_type_config: AttentionTypeConfig
    qk_norm: QKNormConfig = QKNormConfig()  # l2 normalization of qk before softmax


class AttentionStrategy(nn.Module):
    def forward(self, qk_dots: Tensor) -> Tensor:
        raise NotImplementedError


class SoftmaxStrategy(AttentionStrategy):
    def forward(self, qk_dots: Tensor) -> Tensor:
        return F.softmax(qk_dots, dim=-1)


class Entmax15Strategy(AttentionStrategy):
    """
    Entmax15 is a generalization of softmax and sparsemax.
    https://github.com/deep-spin/entmax
    """
    def __init__(self):
        super().__init__()
        from entmax import entmax15
        self.entmax15 = entmax15

    def forward(self, qk_dots: Tensor) -> Tensor:
        return self.entmax15(qk_dots, dim=-1)


class TopKStrategy(AttentionStrategy):
    def __init__(self, topk):
        super().__init__()
        self.topk = topk

    def forward(self, qk_dots: Tensor) -> Tensor:
        top, _ = qk_dots.topk(self.topk, dim=-1)
        vk = top[..., -1].unsqueeze(-1).expand_as(qk_dots)
        mask = qk_dots < vk
        return qk_dots.masked_fill(mask, float('-inf')).softmax(dim=-1)


class AttentionFactory:
    @staticmethod
    def get_strategy(config: AttentionConfig) -> AttentionStrategy:
        attn_type = config.attn_type_config.type
        if attn_type == AttentionType.ENTMAX15:
            return Entmax15Strategy()
        elif attn_type == AttentionType.TOPK:
            return TopKStrategy(config.attn_type_config.sparse_topk)
        else:
            return SoftmaxStrategy()


class PositionalEncoding:
    @staticmethod
    def apply(x, config):
        if config.rotary_pos_emb:
            return PositionalEncoding.apply_rotary(x)
        return x

    @staticmethod
    def apply_rotary(x):
        pass


class AttentionMask:
    @staticmethod
    def prepare(attn_mask, query_len, key_len, batch_size, num_heads):
        pass

    @staticmethod
    def apply_causal_mask(dots, causal):
        pass


class QKNormalization:
    @staticmethod
    def apply(q, k, config):
        if config.qk_norm:
            pass
        return q, k


class TalkingHeads:
    def __init__(self, num_heads):
        super().__init__()
        self.pre_softmax_proj = nn.Parameter(torch.randn(num_heads, num_heads))
        self.post_softmax_proj = nn.Parameter(torch.randn(num_heads, num_heads))

    def pre_softmax(self, dots):
        return einsum('b h i j, h k -> b k i j', dots, self.pre_softmax_proj)

    def post_softmax(self, attn):
        return einsum('b h i j, h k -> b k i j', attn, self.post_softmax_proj)


class MemoryKeyValue:
    def __init__(self, num_heads: int, num_mem_kv: int, dim_head: int):
        self.mem_k = nn.Parameter(torch.randn(num_heads, num_mem_kv, dim_head))
        self.mem_v = nn.Parameter(torch.randn(num_heads, num_mem_kv, dim_head))

    def extend(self, k: Tensor, v: Tensor) -> Tuple[Tensor, Tensor]:
        batch_size = k.shape[0]
        mem_k, mem_v = map(lambda t: repeat(t, 'h n d -> b h n d', b=batch_size), (self.mem_k, self.mem_v))
        return torch.cat((mem_k, k), dim=-2), torch.cat((mem_v, v), dim=-2)