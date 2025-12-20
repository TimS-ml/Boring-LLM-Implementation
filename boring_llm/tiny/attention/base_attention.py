"""
Tiny Transformer with Customizable Attention

Demonstrates how to use different attention mechanisms from boring_llm.nn.attention
with the base transformer implementation.
"""
from __future__ import annotations
import torch
from torch import nn, Tensor
from typing import Optional, Type
from jaxtyping import Float

from boring_utils.utils import get_device
from boring_llm.tiny.tiny_base import TinyTransformBlock, TinyDecoder, TinyFeedForward
from boring_llm.nn.attention import create_attention, AttentionConfig

device = get_device()


class AttentionTransformerBlock(nn.Module):
    """
    Transformer block with customizable attention

    Instead of hardcoded standard attention, uses the flexible attention from nn.attention
    """
    def __init__(
            self,
            dim: int,
            n_layers: int,
            n_head: int,
            d_head: int,
            ffn_mul: int,
            causal: bool = False,
            cross_attend: bool = False,
            dropout: float = 0.,
            attn_type: str = "standard",
            attn_config: Optional[AttentionConfig] = None,
            **attn_kwargs
        ):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList([])
        self.causal = causal
        self.cross_attend = cross_attend

        for _ in range(n_layers):
            # Create attention layers
            if attn_config is not None:
                self_attn = create_attention(attn_config.type, **attn_config.model_dump())
                cross_attn = create_attention(attn_config.type, **attn_config.model_dump()) if cross_attend else None
            else:
                self_attn = create_attention(
                    attn_type,
                    dim_model=dim,
                    num_heads=n_head,
                    dim_head=d_head,
                    causal=causal,
                    dropout=dropout,
                    **attn_kwargs
                )
                cross_attn = create_attention(
                    attn_type,
                    dim_model=dim,
                    num_heads=n_head,
                    dim_head=d_head,
                    causal=False,  # Cross attention is not causal
                    dropout=dropout,
                    **attn_kwargs
                ) if cross_attend else None

            self.layers.append(
                nn.ModuleList([
                    nn.LayerNorm(dim),
                    self_attn,
                    nn.LayerNorm(dim) if cross_attend else None,
                    cross_attn,
                    nn.LayerNorm(dim),
                    TinyFeedForward(dim, mul=ffn_mul, dropout=dropout)
                ])
            )

    def forward(
            self,
            x: Float[Tensor, "batch seq embd"],
            context: Optional[Float[Tensor, "batch seq embd"]] = None,
            **kwargs
        ) -> Float[Tensor, "batch seq embd"]:

        for norm1, attn, norm2, cross_attn, norm3, ff in self.layers:
            # Self-attention with normalization
            x = x + attn(norm1(x))

            # Cross-attention (if available)
            if cross_attn:
                x = x + cross_attn(norm2(x), context=context)

            # Feedforward with normalization
            x = x + ff(norm3(x))

        return x


class AttentionTransformerWrapper(nn.Module):
    """TinyTransformBlock with customizable attention"""
    def __init__(
            self,
            num_tokens: int,
            max_seq_len: int,
            dim: int,
            n_layers: int,
            n_head: int,
            d_head: int,
            ffn_mul: int,
            dropout: float = 0.,
            cross_attend: bool = False,
            transform_layer: Type[nn.Module] = TinyDecoder,
            return_only_embed: bool = False,
            attn_type: str = "standard",
            attn_config: Optional[AttentionConfig] = None,
            **attn_kwargs
        ):
        super().__init__()
        self.dim = dim
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Parameter(torch.randn(1, max_seq_len, dim))
        self.max_seq_len = max_seq_len

        # Use custom attention transformer
        self.transformer = AttentionTransformerBlock(
            dim=dim,
            n_layers=n_layers,
            n_head=n_head,
            d_head=d_head,
            ffn_mul=ffn_mul,
            dropout=dropout,
            cross_attend=cross_attend,
            causal=True,  # Assuming decoder
            attn_type=attn_type,
            attn_config=attn_config,
            **attn_kwargs
        )

        if return_only_embed:
            self.to_logits = nn.Identity()
        else:
            self.to_logits = nn.Linear(dim, num_tokens, bias=False)

    def forward(
            self,
            x: Float[Tensor, "batch seq"],
            context: Optional[Float[Tensor, "batch seq embd"]] = None,
            **kwargs
        ) -> Float[Tensor, "embd num_tokens"]:
        batch, seq_len = x.shape

        # Truncate sequence if it exceeds max length
        if seq_len > self.max_seq_len:
            x = x[:, -self.max_seq_len:]
            seq_len = self.max_seq_len

        x = self.token_emb(x)
        x = x + self.pos_emb[:, :seq_len]

        if self.transformer.cross_attend:
            x = self.transformer(x, context=context)
        else:
            x = self.transformer(x)

        return self.to_logits(x)


def create_attention_transformer(
    num_tokens: int,
    max_seq_len: int,
    dim: int,
    n_layers: int,
    n_head: int,
    d_head: int,
    ffn_mul: int,
    attn_type: str = "standard",
    dropout: float = 0.0,
    **attn_kwargs
) -> AttentionTransformerWrapper:
    """Convenience function to create transformer with custom attention"""
    return AttentionTransformerWrapper(
        num_tokens=num_tokens,
        max_seq_len=max_seq_len,
        dim=dim,
        n_layers=n_layers,
        n_head=n_head,
        d_head=d_head,
        ffn_mul=ffn_mul,
        dropout=dropout,
        attn_type=attn_type,
        **attn_kwargs
    )


def test_attention_transformers():
    """Test different attention strategies"""
    from boring_llm.base.tiny_config import (
        NUM_TOKENS, BLOCK_SIZE, EMBEDDING_DIM,
        N_LAYER, N_HEAD, D_HEAD, FFN_MUL,
        BATCH_SIZE, DROPOUT
    )

    test_configs = [
        {
            "name": "Standard Attention",
            "attn_type": "standard",
        },
        {
            "name": "Multi-Query Attention (MQA)",
            "attn_type": "standard",
            "one_kv_head": True,
        },
        {
            "name": "Grouped-Query Attention (GQA)",
            "attn_type": "standard",
            "kv_heads": 2,
        },
        {
            "name": "Cosine Similarity Attention",
            "attn_type": "cosine_sim",
            "temperature": 1.0,
        },
        {
            "name": "Sparse TopK Attention",
            "attn_type": "sparse_topk",
            "topk": 8,
        },
        {
            "name": "QK Normalized Attention",
            "attn_type": "standard",
            "qk_norm": True,
            "qk_norm_groups": 8,
        },
        {
            "name": "Attention with Talking Heads",
            "attn_type": "standard",
            "pre_talking_heads": True,
            "post_talking_heads": True,
        },
        {
            "name": "Attention with Value Gating",
            "attn_type": "standard",
            "gate_values": True,
        },
    ]

    models = {}

    for config in test_configs:
        name = config.pop("name")
        print(f"\nCreating {name}...")

        model = create_attention_transformer(
            num_tokens=NUM_TOKENS,
            max_seq_len=BLOCK_SIZE,
            dim=EMBEDDING_DIM,
            n_layers=N_LAYER,
            n_head=N_HEAD,
            d_head=D_HEAD,
            ffn_mul=FFN_MUL,
            dropout=DROPOUT,
            **config
        ).to(device)

        models[name] = model

    # Create test input
    x = torch.randint(0, NUM_TOKENS, (BATCH_SIZE, BLOCK_SIZE)).to(device)

    # Test all models
    for name, model in models.items():
        print(f"\nTesting {name}...")
        logits = model(x)

        # Check output shape
        expected_shape = (BATCH_SIZE, BLOCK_SIZE, NUM_TOKENS)
        assert logits.shape == expected_shape, \
            f"{name}: Expected {expected_shape}, got {logits.shape}"

        print(f"{name} output shape: {logits.shape} ✓")

    print("\n✅ All attention tests passed!")


if __name__ == "__main__":
    test_attention_transformers()
