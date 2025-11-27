"""
Attention mechanism registry and transforms

This module provides various attention enhancements from x-transformers:
- Multi-Query Attention (MQA) / Grouped-Query Attention (GQA)
- QK Normalization for training stability
- Talking Heads (pre/post attention head mixing)
- Cosine Similarity Attention
- Value Gating (from AlphaFold2)
- Sparse TopK Attention
"""

from typing import Optional, Callable
from pydantic import Field
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange, repeat

from boring_llm.base.component_registry import ComponentTransform, ComponentRegistry


class AttentionTransform(ComponentTransform):
    """Base class for attention mechanism transformations"""

    def apply(self, q: Tensor, k: Tensor, v: Tensor, **kwargs) -> Tensor:
        """Apply attention mechanism"""
        raise NotImplementedError


attention_registry = ComponentRegistry[AttentionTransform]("Attention")


# ============================================================================
# Core Attention Variants
# ============================================================================

@attention_registry.register("standard", {
    "dropout": (float, Field(default=0., description="Attention dropout")),
    "causal": (bool, Field(default=False, description="Use causal masking"))
})
class StandardAttention(AttentionTransform):
    """Standard scaled dot-product attention"""

    def __init__(self, dim_model: int, num_heads: int = 8, dropout: float = 0.,
                 causal: bool = False, **kwargs):
        super().__init__()
        self.scale = (dim_model // num_heads) ** -0.5
        self.dropout = nn.Dropout(dropout)
        self.causal = causal
        self.num_heads = num_heads

    def apply(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None, **kwargs) -> Tensor:
        """
        Args:
            q: Query tensor [batch, heads, seq_q, dim_head]
            k: Key tensor [batch, heads, seq_k, dim_head]
            v: Value tensor [batch, heads, seq_k, dim_head]
            mask: Optional attention mask
        """
        # Compute attention scores
        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        # Apply causal mask if needed
        if self.causal:
            seq_q, seq_k = sim.shape[-2:]
            causal_mask = torch.ones((seq_q, seq_k), device=sim.device, dtype=torch.bool).triu(seq_k - seq_q + 1)
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        # Apply custom mask
        if mask is not None:
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        # Attention weights
        attn = F.softmax(sim, dim=-1)
        attn = self.dropout(attn)

        # Weighted sum
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        return out


@attention_registry.register("cosine_sim", {
    "temperature": (float, Field(default=1.0, description="Temperature for cosine similarity")),
    "dropout": (float, Field(default=0., description="Attention dropout")),
    "causal": (bool, Field(default=False, description="Use causal masking"))
})
class CosineSimilarityAttention(AttentionTransform):
    """
    Cosine similarity attention

    Uses cosine similarity instead of dot product for attention scores.
    More stable for training and doesn't require scaling.
    """

    def __init__(self, dim_model: int, num_heads: int = 8, temperature: float = 1.0,
                 dropout: float = 0., causal: bool = False, **kwargs):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)
        self.causal = causal
        self.num_heads = num_heads

    def apply(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None, **kwargs) -> Tensor:
        """Cosine similarity attention"""
        # Normalize q and k
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # Cosine similarity
        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k) / self.temperature

        # Apply causal mask
        if self.causal:
            seq_q, seq_k = sim.shape[-2:]
            causal_mask = torch.ones((seq_q, seq_k), device=sim.device, dtype=torch.bool).triu(seq_k - seq_q + 1)
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        if mask is not None:
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        attn = F.softmax(sim, dim=-1)
        attn = self.dropout(attn)

        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        return out


@attention_registry.register("sparse_topk", {
    "topk": (int, Field(default=8, description="Number of top values to keep")),
    "dropout": (float, Field(default=0., description="Attention dropout")),
    "causal": (bool, Field(default=False, description="Use causal masking")),
    "straight_through": (bool, Field(default=True, description="Use straight-through estimator"))
})
class SparseTopKAttention(AttentionTransform):
    """
    Sparse Top-K Attention from 'Explicit Sparse Transformer'

    Only keeps top-k attention scores, zeros out the rest.
    More efficient than full attention.
    """

    def __init__(self, dim_model: int, num_heads: int = 8, topk: int = 8,
                 dropout: float = 0., causal: bool = False,
                 straight_through: bool = True, **kwargs):
        super().__init__()
        self.scale = (dim_model // num_heads) ** -0.5
        self.topk = topk
        self.dropout = nn.Dropout(dropout)
        self.causal = causal
        self.straight_through = straight_through
        self.num_heads = num_heads

    def apply(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None, **kwargs) -> Tensor:
        """Sparse top-k attention"""
        # Compute scores
        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        # Apply masks
        if self.causal:
            seq_q, seq_k = sim.shape[-2:]
            causal_mask = torch.ones((seq_q, seq_k), device=sim.device, dtype=torch.bool).triu(seq_k - seq_q + 1)
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        if mask is not None:
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        # Top-k selection
        topk_values, topk_indices = sim.topk(min(self.topk, sim.shape[-1]), dim=-1)

        # Create sparse mask
        sparse_mask = torch.zeros_like(sim, dtype=torch.bool)
        sparse_mask.scatter_(-1, topk_indices, True)

        # Apply sparse mask
        if self.straight_through:
            # Straight-through: use sparse in forward, but full gradients in backward
            sim_sparse = sim.masked_fill(~sparse_mask, -torch.finfo(sim.dtype).max)
            sim = sim_sparse + (sim - sim.detach())
        else:
            sim = sim.masked_fill(~sparse_mask, -torch.finfo(sim.dtype).max)

        attn = F.softmax(sim, dim=-1)
        attn = self.dropout(attn)

        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        return out


# ============================================================================
# Normalization Enhancements
# ============================================================================

class QKNormalization(nn.Module):
    """
    Query-Key Normalization for training stability

    Normalizes queries and keys before attention computation.
    Helps with training stability and allows higher learning rates.
    """

    def __init__(self, dim_head: int, num_groups: int = 1, scale: float = 10.0,
                 learnable_scale: bool = False, num_heads: int = 8, kv_heads: int = 8):
        super().__init__()
        self.num_groups = num_groups
        self.scale = scale
        self.learnable_scale = learnable_scale

        if learnable_scale:
            self.q_scale = nn.Parameter(torch.ones(num_heads, 1, dim_head))
            self.k_scale = nn.Parameter(torch.ones(kv_heads, 1, dim_head))
        else:
            self.q_scale = 1.
            self.k_scale = 1.

    def forward(self, q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
        """Apply QK normalization"""
        # Group normalization
        if self.num_groups > 1:
            q = rearrange(q, 'b h n (g d) -> b h n g d', g=self.num_groups)
            k = rearrange(k, 'b h n (g d) -> b h n g d', g=self.num_groups)

        # L2 normalize
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        if self.num_groups > 1:
            q = rearrange(q, 'b h n g d -> b h n (g d)')
            k = rearrange(k, 'b h n g d -> b h n (g d)')

        # Apply learnable scale
        q = q * self.q_scale
        k = k * self.k_scale

        # Apply fixed scale
        q = q * self.scale
        k = k * self.scale

        return q, k


# ============================================================================
# Value Processing
# ============================================================================

class ValueGating(nn.Module):
    """
    Value gating from AlphaFold2

    Gates the aggregated values with input features for better control.
    """

    def __init__(self, dim: int, dim_out: int, use_swiglu: bool = False):
        super().__init__()
        self.to_gate = nn.Linear(dim, dim_out)
        self.activation = F.silu if use_swiglu else F.sigmoid

        # Initialize to bias towards passing through values
        nn.init.constant_(self.to_gate.weight, 0)
        nn.init.constant_(self.to_gate.bias, 10)

    def forward(self, x: Tensor, values: Tensor) -> Tensor:
        """Apply value gating"""
        gate = self.activation(self.to_gate(x))
        return values * gate


class ValueHeadGating(nn.Module):
    """
    Per-head value gating from 'Attend to Nothing' paper

    Allows each head to gate its output independently.
    """

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.to_head_gate = nn.Linear(dim, num_heads)

        nn.init.constant_(self.to_head_gate.weight, 0)
        nn.init.constant_(self.to_head_gate.bias, 10)

    def forward(self, x: Tensor, values: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor [batch, seq, dim]
            values: Values tensor [batch, heads, seq, dim_head]
        """
        gate = self.to_head_gate(x).sigmoid()  # [batch, seq, heads]
        gate = rearrange(gate, 'b n h -> b h n 1')
        return values * gate


# ============================================================================
# Talking Heads
# ============================================================================

class TalkingHeads(nn.Module):
    """
    Talking Heads from Noam Shazeer

    Allows information mixing between attention heads before and/or after softmax.
    """

    def __init__(self, num_heads: int, pre_softmax: bool = False, post_softmax: bool = False):
        super().__init__()
        self.pre_softmax = pre_softmax
        self.post_softmax = post_softmax

        if pre_softmax:
            self.pre_talking = nn.Conv2d(num_heads, num_heads, 1, bias=False)

        if post_softmax:
            self.post_talking = nn.Conv2d(num_heads, num_heads, 1, bias=False)

    def forward(self, attn: Tensor, pre: bool = True) -> Tensor:
        """
        Apply talking heads

        Args:
            attn: Attention tensor [batch, heads, seq_q, seq_k]
            pre: If True, apply pre-softmax mixing; else post-softmax
        """
        if pre and self.pre_softmax:
            return self.pre_talking(attn)
        elif not pre and self.post_softmax:
            return self.post_talking(attn)
        return attn
