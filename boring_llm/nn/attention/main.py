"""
Main attention module with MQA/GQA support

Provides a flexible attention interface supporting:
- Multi-Query Attention (MQA)
- Grouped-Query Attention (GQA)
- Various attention mechanisms (standard, cosine sim, sparse topk)
- QK normalization
- Talking heads
- Value gating
"""

from typing import Optional
from pydantic import Field
import torch
import torch.nn as nn
from torch import Tensor
from einops import rearrange, repeat

from boring_llm.base.component_registry import ComponentConfig
from boring_llm.nn.attention.registry import (
    attention_registry,
    QKNormalization,
    ValueGating,
    ValueHeadGating,
    TalkingHeads
)


class AttentionConfig(ComponentConfig):
    """
    Attention Configuration

    Supports various attention types and enhancements:
    - standard: Standard scaled dot-product attention
    - cosine_sim: Cosine similarity attention
    - sparse_topk: Sparse top-k attention
    """
    # Model dimensions
    num_heads: int = Field(default=8, description="Number of query heads")
    dim_head: int = Field(default=64, description="Dimension per head")

    # Multi-Query / Grouped-Query Attention
    kv_heads: Optional[int] = Field(default=None, description="Number of KV heads (GQA). If None, same as num_heads")
    one_kv_head: bool = Field(default=False, description="Use single KV head (MQA)")

    # Attention mechanism
    dropout: float = Field(default=0., description="Attention dropout")
    causal: bool = Field(default=False, description="Use causal masking")

    # Sparse TopK
    topk: int = Field(default=8, description="Top-k values for sparse attention")
    topk_straight_through: bool = Field(default=True, description="Straight-through estimator")

    # Cosine similarity
    temperature: float = Field(default=1.0, description="Temperature for cosine sim")

    # QK Normalization
    qk_norm: bool = Field(default=False, description="Apply QK normalization")
    qk_norm_groups: int = Field(default=1, description="Groups for QK norm")
    qk_norm_scale: float = Field(default=10., description="QK norm scale factor")
    qk_norm_learnable: bool = Field(default=False, description="Learnable QK scale")

    # Talking Heads
    pre_talking_heads: bool = Field(default=False, description="Mix heads before softmax")
    post_talking_heads: bool = Field(default=False, description="Mix heads after softmax")

    # Value processing
    gate_values: bool = Field(default=False, description="Gate values with input (AlphaFold2)")
    gate_value_heads: bool = Field(default=False, description="Per-head value gating")
    swiglu_values: bool = Field(default=False, description="Use SwiGLU for value gating")

    # Cross attention
    dim_context: Optional[int] = Field(default=None, description="Context dimension for cross-attention")

    def __init__(self, **data):
        """Custom init for Pydantic v1/v2 compatibility"""
        # Handle MQA/GQA
        if data.get('one_kv_head', False):
            data['kv_heads'] = 1
        elif data.get('kv_heads') is None:
            data['kv_heads'] = data.get('num_heads', 8)

        # Validate before calling super
        num_heads = data.get('num_heads', 8)
        kv_heads = data.get('kv_heads', num_heads)
        assert num_heads % kv_heads == 0, \
            f"num_heads ({num_heads}) must be divisible by kv_heads ({kv_heads})"

        super().__init__(**data)


class BoringMultiHeadAttention(nn.Module):
    """
    Multi-Head Attention with MQA/GQA support

    Example:
        # Standard multi-head attention
        attn = BoringMultiHeadAttention(AttentionConfig(
            type="standard",
            dim_model=512,
            num_heads=8,
            dim_head=64
        ))

        # Multi-Query Attention (MQA)
        attn = BoringMultiHeadAttention(AttentionConfig(
            type="standard",
            dim_model=512,
            num_heads=8,
            one_kv_head=True  # Single KV head for all queries
        ))

        # Grouped-Query Attention (GQA)
        attn = BoringMultiHeadAttention(AttentionConfig(
            type="standard",
            dim_model=512,
            num_heads=8,
            kv_heads=2  # 8 query heads, 2 KV heads
        ))
    """

    def __init__(self, config: AttentionConfig = None, **kwargs):
        super().__init__()
        config = AttentionConfig(**kwargs) if not config else config.model_copy(update=kwargs)

        self.config = config
        self.dim_model = config.dim_model
        self.num_heads = config.num_heads
        self.kv_heads = config.kv_heads
        self.dim_head = config.dim_head
        self.causal = config.causal

        # Determine context dimension
        dim_context = config.dim_context if config.dim_context is not None else config.dim_model

        # Query, Key, Value projections
        q_dim = config.dim_head * config.num_heads
        k_dim = config.dim_head * config.kv_heads
        v_dim = config.dim_head * config.kv_heads

        self.to_q = nn.Linear(config.dim_model, q_dim, bias=False)
        self.to_k = nn.Linear(dim_context, k_dim, bias=False)
        self.to_v = nn.Linear(dim_context, v_dim, bias=False)

        # Output projection
        self.to_out = nn.Linear(q_dim, config.dim_model, bias=False)

        # QK Normalization
        self.qk_norm = None
        if config.qk_norm:
            self.qk_norm = QKNormalization(
                dim_head=config.dim_head,
                num_groups=config.qk_norm_groups,
                scale=config.qk_norm_scale,
                learnable_scale=config.qk_norm_learnable,
                num_heads=config.num_heads,
                kv_heads=config.kv_heads
            )

        # Talking Heads
        self.talking_heads = None
        if config.pre_talking_heads or config.post_talking_heads:
            self.talking_heads = TalkingHeads(
                num_heads=config.num_heads,
                pre_softmax=config.pre_talking_heads,
                post_softmax=config.post_talking_heads
            )

        # Value gating
        self.value_gate = None
        if config.gate_values:
            self.value_gate = ValueGating(
                dim=config.dim_model,
                dim_out=q_dim,
                use_swiglu=config.swiglu_values
            )

        self.value_head_gate = None
        if config.gate_value_heads:
            self.value_head_gate = ValueHeadGating(
                dim=config.dim_model,
                num_heads=config.num_heads
            )

        # Create attention mechanism
        attn_kwargs = {
            'dim_model': config.dim_model,
            'num_heads': config.num_heads,
        }

        if config.type == "standard":
            attn_kwargs.update({
                'dropout': config.dropout,
                'causal': config.causal
            })
        elif config.type == "cosine_sim":
            attn_kwargs.update({
                'temperature': config.temperature,
                'dropout': config.dropout,
                'causal': config.causal
            })
        elif config.type == "sparse_topk":
            attn_kwargs.update({
                'topk': config.topk,
                'dropout': config.dropout,
                'causal': config.causal,
                'straight_through': config.topk_straight_through
            })

        self.attention_fn = attention_registry.create_strategy(config.type, **attn_kwargs)

    def forward(
        self,
        x: Tensor,
        context: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        **kwargs
    ) -> Tensor:
        """
        Forward pass

        Args:
            x: Input tensor [batch, seq, dim]
            context: Context tensor for cross-attention [batch, seq_ctx, dim_ctx]
            mask: Attention mask [batch, seq_q, seq_k]

        Returns:
            Output tensor [batch, seq, dim]
        """
        batch, seq_len, _ = x.shape

        # Determine context
        context = x if context is None else context

        # Project to Q, K, V
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        # Reshape to multi-head format
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.kv_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.kv_heads)

        # Expand KV heads if using MQA/GQA
        if self.kv_heads < self.num_heads:
            # Repeat each KV head for its group of query heads
            repeats = self.num_heads // self.kv_heads
            k = repeat(k, 'b h n d -> b (h r) n d', r=repeats)
            v = repeat(v, 'b h n d -> b (h r) n d', r=repeats)

        # Apply QK normalization if enabled
        if self.qk_norm is not None:
            q, k = self.qk_norm(q, k)

        # Apply attention
        out = self.attention_fn.apply(q, k, v, mask=mask, **kwargs)

        # Merge heads
        out = rearrange(out, 'b h n d -> b n (h d)')

        # Apply value gating if enabled
        if self.value_gate is not None:
            out = self.value_gate(x, out)

        if self.value_head_gate is not None:
            out = rearrange(out, 'b n (h d) -> b h n d', h=self.num_heads)
            out = self.value_head_gate(x, out)
            out = rearrange(out, 'b h n d -> b n (h d)')

        # Output projection
        return self.to_out(out)


def create_attention(attn_type: str = "standard", **kwargs) -> BoringMultiHeadAttention:
    """
    Convenience function to create attention module

    Args:
        attn_type: Type of attention (standard, cosine_sim, sparse_topk)
        **kwargs: Additional configuration

    Returns:
        BoringMultiHeadAttention instance

    Example:
        # Multi-Query Attention
        attn = create_attention(
            "standard",
            dim_model=512,
            num_heads=8,
            one_kv_head=True
        )

        # Grouped-Query Attention with QK norm
        attn = create_attention(
            "standard",
            dim_model=512,
            num_heads=8,
            kv_heads=2,
            qk_norm=True
        )
    """
    if 'type' in kwargs:
        attn_type = kwargs.pop('type')

    config = AttentionConfig(type=attn_type, **kwargs)
    return BoringMultiHeadAttention(config)


if __name__ == "__main__":
    print("Testing attention implementations...")

    # Test 1: Standard attention
    attn1 = create_attention(
        "standard",
        dim_model=512,
        num_heads=8,
        dim_head=64
    )

    # Test 2: Multi-Query Attention (MQA)
    attn2 = create_attention(
        "standard",
        dim_model=512,
        num_heads=8,
        one_kv_head=True
    )

    # Test 3: Grouped-Query Attention (GQA)
    attn3 = create_attention(
        "standard",
        dim_model=512,
        num_heads=8,
        kv_heads=2
    )

    # Test 4: Cosine similarity attention with QK norm
    attn4 = create_attention(
        "cosine_sim",
        dim_model=512,
        num_heads=8,
        qk_norm=True
    )

    # Test 5: Sparse TopK attention
    attn5 = create_attention(
        "sparse_topk",
        dim_model=512,
        num_heads=8,
        topk=8
    )

    # Test 6: With value gating
    attn6 = create_attention(
        "standard",
        dim_model=512,
        num_heads=8,
        gate_values=True
    )

    # Test forward pass
    x = torch.randn(2, 10, 512)

    print("\n1. Standard Attention:")
    y1 = attn1(x)
    print(f"   Output shape: {y1.shape}")

    print("\n2. Multi-Query Attention (MQA):")
    y2 = attn2(x)
    print(f"   Output shape: {y2.shape}")
    print(f"   KV heads: {attn2.kv_heads} (vs {attn2.num_heads} query heads)")

    print("\n3. Grouped-Query Attention (GQA):")
    y3 = attn3(x)
    print(f"   Output shape: {y3.shape}")
    print(f"   KV heads: {attn3.kv_heads} (vs {attn3.num_heads} query heads)")

    print("\n4. Cosine Similarity Attention:")
    y4 = attn4(x)
    print(f"   Output shape: {y4.shape}")

    print("\n5. Sparse TopK Attention:")
    y5 = attn5(x)
    print(f"   Output shape: {y5.shape}")

    print("\n6. Attention with Value Gating:")
    y6 = attn6(x)
    print(f"   Output shape: {y6.shape}")

    print("\n✅ All attention tests passed!")
