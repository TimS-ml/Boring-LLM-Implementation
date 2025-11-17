"""
Tiny Transformer with Customizable Normalization

Demonstrates how to use different normalization strategies from boring_llm.nn.norm
with the base transformer implementation.
"""
from __future__ import annotations
import torch
from torch import nn, Tensor
from typing import Optional, Type
from jaxtyping import Float

from boring_utils.utils import get_device
from boring_llm.tiny.tiny_base import TinyTransformBlock, TinyDecoder
from boring_llm.nn.norm import create_norm, NormConfig

device = get_device()


class NormTransformerBlock(nn.Module):
    """
    Transformer block with customizable normalization

    Instead of hardcoded LayerNorm, uses the flexible normalization from nn.norm
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
            norm_type: str = "rmsnorm",
            norm_config: Optional[NormConfig] = None,
            **norm_kwargs
        ):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList([])
        self.causal = causal
        self.cross_attend = cross_attend

        # Import attention and FFN from tiny_base
        from boring_llm.tiny.tiny_base import (
            TinyMultiHeadAttention,
            TinyMultiHeadCrossAttention,
            TinyFeedForward
        )

        for _ in range(n_layers):
            # Create normalization layers
            if norm_config is not None:
                norm1 = create_norm(norm_config.type, **norm_config.model_dump())
                norm2 = create_norm(norm_config.type, **norm_config.model_dump()) if cross_attend else None
                norm3 = create_norm(norm_config.type, **norm_config.model_dump())
            else:
                norm1 = create_norm(norm_type, dim_model=dim, **norm_kwargs)
                norm2 = create_norm(norm_type, dim_model=dim, **norm_kwargs) if cross_attend else None
                norm3 = create_norm(norm_type, dim_model=dim, **norm_kwargs)

            self.layers.append(
                nn.ModuleList([
                    norm1,
                    TinyMultiHeadAttention(dim, n_head, d_head, causal=causal, dropout=dropout),
                    norm2,
                    TinyMultiHeadCrossAttention(dim, n_head, d_head) if cross_attend else None,
                    norm3,
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


class NormTransformerWrapper(nn.Module):
    """TinyTransformBlock with customizable normalization"""
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
            norm_type: str = "rmsnorm",
            norm_config: Optional[NormConfig] = None,
            **norm_kwargs
        ):
        super().__init__()
        self.dim = dim
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Parameter(torch.randn(1, max_seq_len, dim))
        self.max_seq_len = max_seq_len

        # Use custom normalization transformer
        self.transformer = NormTransformerBlock(
            dim=dim,
            n_layers=n_layers,
            n_head=n_head,
            d_head=d_head,
            ffn_mul=ffn_mul,
            dropout=dropout,
            cross_attend=cross_attend,
            causal=True,  # Assuming decoder
            norm_type=norm_type,
            norm_config=norm_config,
            **norm_kwargs
        )

        # Final normalization
        if norm_config is not None:
            self.final_norm = create_norm(norm_config.type, **norm_config.model_dump())
        else:
            self.final_norm = create_norm(norm_type, dim_model=dim, **norm_kwargs)

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

        x = self.final_norm(x)
        return self.to_logits(x)


def create_norm_transformer(
    num_tokens: int,
    max_seq_len: int,
    dim: int,
    n_layers: int,
    n_head: int,
    d_head: int,
    ffn_mul: int,
    norm_type: str = "rmsnorm",
    dropout: float = 0.0,
    **norm_kwargs
) -> NormTransformerWrapper:
    """Convenience function to create transformer with custom normalization"""
    return NormTransformerWrapper(
        num_tokens=num_tokens,
        max_seq_len=max_seq_len,
        dim=dim,
        n_layers=n_layers,
        n_head=n_head,
        d_head=d_head,
        ffn_mul=ffn_mul,
        dropout=dropout,
        norm_type=norm_type,
        **norm_kwargs
    )


def test_norm_transformers():
    """Test different normalization strategies"""
    from boring_llm.base.tiny_config import (
        NUM_TOKENS, BLOCK_SIZE, EMBEDDING_DIM,
        N_LAYER, N_HEAD, D_HEAD, FFN_MUL,
        BATCH_SIZE, DROPOUT
    )

    test_configs = [
        {
            "name": "RMSNorm",
            "norm_type": "rmsnorm",
            "unit_offset": False
        },
        {
            "name": "RMSNorm with unit offset",
            "norm_type": "rmsnorm",
            "unit_offset": True
        },
        {
            "name": "LayerNorm",
            "norm_type": "layernorm",
        },
        {
            "name": "ScaleNorm",
            "norm_type": "scalenorm",
        },
        {
            "name": "Simple RMSNorm",
            "norm_type": "simple_rmsnorm",
        },
    ]

    models = {}

    for config in test_configs:
        name = config.pop("name")
        print(f"Creating {name}...")

        model = create_norm_transformer(
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

    print("\n✅ All normalization tests passed!")


if __name__ == "__main__":
    test_norm_transformers()
