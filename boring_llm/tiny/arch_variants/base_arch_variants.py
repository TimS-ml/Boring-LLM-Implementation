"""
Tiny Transformer with Architecture Variants

Demonstrates different architectural patterns from boring_llm.nn.arch_variants
with the base transformer implementation.
"""
from __future__ import annotations
import torch
from torch import nn, Tensor
from typing import Optional, Type
from jaxtyping import Float

from boring_utils.utils import get_device
from boring_llm.tiny.tiny_base import TinyTransformBlock, TinyDecoder, TinyMultiHeadAttention, TinyFeedForward

device = get_device()


class SandwichNormTransformerBlock(nn.Module):
    """
    Transformer block with Sandwich Normalization

    Applies normalization both before and after the attention/FFN layers
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
            norm_class: Type[nn.Module] = nn.LayerNorm,
        ):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList([])
        self.causal = causal
        self.cross_attend = cross_attend

        for _ in range(n_layers):
            self.layers.append(
                nn.ModuleList([
                    # Pre-norm
                    norm_class(dim),
                    # Attention
                    TinyMultiHeadAttention(dim, n_head, d_head, causal=causal, dropout=dropout),
                    # Post-norm (sandwich)
                    norm_class(dim),
                    # Cross attention (if needed)
                    norm_class(dim) if cross_attend else None,
                    TinyMultiHeadAttention(dim, n_head, d_head, causal=False, dropout=dropout) if cross_attend else None,
                    norm_class(dim) if cross_attend else None,
                    # FFN with sandwich norm
                    norm_class(dim),
                    TinyFeedForward(dim, mul=ffn_mul, dropout=dropout),
                    norm_class(dim),
                ])
            )

    def forward(
            self,
            x: Float[Tensor, "batch seq embd"],
            context: Optional[Float[Tensor, "batch seq embd"]] = None,
            **kwargs
        ) -> Float[Tensor, "batch seq embd"]:

        for parts in self.layers:
            if self.cross_attend:
                norm1, attn, norm2, norm3, cross_attn, norm4, norm5, ff, norm6 = parts

                # Sandwich norm for self-attention
                attn_out = attn(norm1(x))
                x = x + norm2(attn_out)

                # Sandwich norm for cross-attention
                cross_attn_out = cross_attn(norm3(x), context=context)
                x = x + norm4(cross_attn_out)

                # Sandwich norm for FFN
                ff_out = ff(norm5(x))
                x = x + norm6(ff_out)
            else:
                norm1, attn, norm2, _, _, _, norm5, ff, norm6 = parts

                # Sandwich norm for self-attention
                attn_out = attn(norm1(x))
                x = x + norm2(attn_out)

                # Sandwich norm for FFN
                ff_out = ff(norm5(x))
                x = x + norm6(ff_out)

        return x


class MacaronTransformerBlock(nn.Module):
    """
    Transformer block with Macaron structure

    FFN(1/2) -> Attention -> FFN(1/2)
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
        ):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList([])
        self.causal = causal
        self.cross_attend = cross_attend

        # Macaron structure: half FFN before attention, half after
        ffn_mul_half = ffn_mul // 2 if ffn_mul >= 2 else ffn_mul

        for _ in range(n_layers):
            self.layers.append(
                nn.ModuleList([
                    # First half of FFN (before attention)
                    nn.LayerNorm(dim),
                    TinyFeedForward(dim, mul=ffn_mul_half, dropout=dropout),
                    # Attention
                    nn.LayerNorm(dim),
                    TinyMultiHeadAttention(dim, n_head, d_head, causal=causal, dropout=dropout),
                    # Cross attention (if needed)
                    nn.LayerNorm(dim) if cross_attend else None,
                    TinyMultiHeadAttention(dim, n_head, d_head, causal=False, dropout=dropout) if cross_attend else None,
                    # Second half of FFN (after attention)
                    nn.LayerNorm(dim),
                    TinyFeedForward(dim, mul=ffn_mul_half, dropout=dropout),
                ])
            )

    def forward(
            self,
            x: Float[Tensor, "batch seq embd"],
            context: Optional[Float[Tensor, "batch seq embd"]] = None,
            **kwargs
        ) -> Float[Tensor, "batch seq embd"]:

        for parts in self.layers:
            if self.cross_attend:
                norm1, ff1, norm2, attn, norm3, cross_attn, norm4, ff2 = parts

                # First half FFN
                x = x + ff1(norm1(x))

                # Self-attention
                x = x + attn(norm2(x))

                # Cross-attention
                x = x + cross_attn(norm3(x), context=context)

                # Second half FFN
                x = x + ff2(norm4(x))
            else:
                norm1, ff1, norm2, attn, _, _, norm4, ff2 = parts

                # First half FFN
                x = x + ff1(norm1(x))

                # Self-attention
                x = x + attn(norm2(x))

                # Second half FFN
                x = x + ff2(norm4(x))

        return x


class SandwichNormTransformer(nn.Module):
    """Transformer with Sandwich Normalization"""
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
            return_only_embed: bool = False,
        ):
        super().__init__()
        self.dim = dim
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Parameter(torch.randn(1, max_seq_len, dim))
        self.max_seq_len = max_seq_len

        self.transformer = SandwichNormTransformerBlock(
            dim=dim,
            n_layers=n_layers,
            n_head=n_head,
            d_head=d_head,
            ffn_mul=ffn_mul,
            dropout=dropout,
            cross_attend=cross_attend,
            causal=True,
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


class MacaronTransformer(nn.Module):
    """Transformer with Macaron structure"""
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
            return_only_embed: bool = False,
        ):
        super().__init__()
        self.dim = dim
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Parameter(torch.randn(1, max_seq_len, dim))
        self.max_seq_len = max_seq_len

        self.transformer = MacaronTransformerBlock(
            dim=dim,
            n_layers=n_layers,
            n_head=n_head,
            d_head=d_head,
            ffn_mul=ffn_mul,
            dropout=dropout,
            cross_attend=cross_attend,
            causal=True,
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


def create_sandwich_norm_transformer(
    num_tokens: int,
    max_seq_len: int,
    dim: int,
    n_layers: int,
    n_head: int,
    d_head: int,
    ffn_mul: int,
    dropout: float = 0.0,
    **kwargs
) -> SandwichNormTransformer:
    """Convenience function to create Sandwich Norm transformer"""
    return SandwichNormTransformer(
        num_tokens=num_tokens,
        max_seq_len=max_seq_len,
        dim=dim,
        n_layers=n_layers,
        n_head=n_head,
        d_head=d_head,
        ffn_mul=ffn_mul,
        dropout=dropout,
        **kwargs
    )


def create_macaron_transformer(
    num_tokens: int,
    max_seq_len: int,
    dim: int,
    n_layers: int,
    n_head: int,
    d_head: int,
    ffn_mul: int,
    dropout: float = 0.0,
    **kwargs
) -> MacaronTransformer:
    """Convenience function to create Macaron transformer"""
    return MacaronTransformer(
        num_tokens=num_tokens,
        max_seq_len=max_seq_len,
        dim=dim,
        n_layers=n_layers,
        n_head=n_head,
        d_head=d_head,
        ffn_mul=ffn_mul,
        dropout=dropout,
        **kwargs
    )


def test_arch_variant_transformers():
    """Test architecture variant transformers"""
    from boring_llm.base.tiny_config import (
        NUM_TOKENS, BLOCK_SIZE, EMBEDDING_DIM,
        N_LAYER, N_HEAD, D_HEAD, FFN_MUL,
        BATCH_SIZE, DROPOUT
    )

    test_configs = [
        {
            "name": "Sandwich Norm Transformer",
            "model_fn": create_sandwich_norm_transformer,
        },
        {
            "name": "Macaron Transformer",
            "model_fn": create_macaron_transformer,
        },
    ]

    models = {}

    for config in test_configs:
        name = config["name"]
        model_fn = config["model_fn"]
        print(f"\nCreating {name}...")

        model = model_fn(
            num_tokens=NUM_TOKENS,
            max_seq_len=BLOCK_SIZE,
            dim=EMBEDDING_DIM,
            n_layers=2,  # Use fewer layers for faster testing
            n_head=N_HEAD,
            d_head=D_HEAD,
            ffn_mul=FFN_MUL,
            dropout=DROPOUT,
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

    print("\n✅ All architecture variant tests passed!")


if __name__ == "__main__":
    test_arch_variant_transformers()
