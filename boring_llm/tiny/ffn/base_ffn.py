"""
Tiny Transformer with Customizable Feed-Forward Networks

Demonstrates how to use different FFN mechanisms from boring_llm.nn.ffn
with the base transformer implementation.
"""
from __future__ import annotations
import torch
from torch import nn, Tensor
from typing import Optional, Type
from jaxtyping import Float

from boring_utils.utils import get_device
from boring_llm.tiny.tiny_base import TinyTransformBlock, TinyDecoder, TinyMultiHeadAttention
from boring_llm.nn.ffn import create_ffn, create_moe_ffn, FFNConfig

device = get_device()


class FFNTransformerBlock(nn.Module):
    """
    Transformer block with customizable FFN

    Instead of hardcoded standard FFN, uses the flexible FFN from nn.ffn
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
            ffn_type: str = "standard",
            ffn_config: Optional[FFNConfig] = None,
            use_moe: bool = False,
            **ffn_kwargs
        ):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList([])
        self.causal = causal
        self.cross_attend = cross_attend

        for _ in range(n_layers):
            # Create FFN layer
            if use_moe:
                # Create MoE FFN
                if ffn_config is not None:
                    ffn = create_moe_ffn(**ffn_config.model_dump())
                else:
                    ffn = create_moe_ffn(
                        dim_model=dim,
                        mult_dim=ffn_mul,
                        dropout=dropout,
                        **ffn_kwargs
                    )
            else:
                # Create standard or GLU FFN
                if ffn_config is not None:
                    ffn = create_ffn(ffn_config.type, **ffn_config.model_dump())
                else:
                    ffn = create_ffn(
                        ffn_type,
                        dim_model=dim,
                        mult_dim=ffn_mul,
                        dropout=dropout,
                        **ffn_kwargs
                    )

            self.layers.append(
                nn.ModuleList([
                    nn.LayerNorm(dim),
                    TinyMultiHeadAttention(dim, n_head, d_head, causal=causal, dropout=dropout),
                    nn.LayerNorm(dim) if cross_attend else None,
                    TinyMultiHeadAttention(dim, n_head, d_head, causal=False, dropout=dropout) if cross_attend else None,
                    nn.LayerNorm(dim),
                    ffn
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


class FFNTransformerWrapper(nn.Module):
    """TinyTransformBlock with customizable FFN"""
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
            ffn_type: str = "standard",
            ffn_config: Optional[FFNConfig] = None,
            use_moe: bool = False,
            **ffn_kwargs
        ):
        super().__init__()
        self.dim = dim
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Parameter(torch.randn(1, max_seq_len, dim))
        self.max_seq_len = max_seq_len

        # Use custom FFN transformer
        self.transformer = FFNTransformerBlock(
            dim=dim,
            n_layers=n_layers,
            n_head=n_head,
            d_head=d_head,
            ffn_mul=ffn_mul,
            dropout=dropout,
            cross_attend=cross_attend,
            causal=True,  # Assuming decoder
            ffn_type=ffn_type,
            ffn_config=ffn_config,
            use_moe=use_moe,
            **ffn_kwargs
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


def create_ffn_transformer(
    num_tokens: int,
    max_seq_len: int,
    dim: int,
    n_layers: int,
    n_head: int,
    d_head: int,
    ffn_mul: int,
    ffn_type: str = "standard",
    dropout: float = 0.0,
    use_moe: bool = False,
    **ffn_kwargs
) -> FFNTransformerWrapper:
    """Convenience function to create transformer with custom FFN"""
    return FFNTransformerWrapper(
        num_tokens=num_tokens,
        max_seq_len=max_seq_len,
        dim=dim,
        n_layers=n_layers,
        n_head=n_head,
        d_head=d_head,
        ffn_mul=ffn_mul,
        dropout=dropout,
        ffn_type=ffn_type,
        use_moe=use_moe,
        **ffn_kwargs
    )


def test_ffn_transformers():
    """Test different FFN strategies"""
    from boring_llm.base.tiny_config import (
        NUM_TOKENS, BLOCK_SIZE, EMBEDDING_DIM,
        N_LAYER, N_HEAD, D_HEAD, FFN_MUL,
        BATCH_SIZE, DROPOUT
    )

    test_configs = [
        {
            "name": "Standard FFN",
            "ffn_type": "standard",
        },
        {
            "name": "GLU FFN",
            "ffn_type": "glu",
            "activation": "GELU",
        },
        {
            "name": "SwiGLU FFN",
            "ffn_type": "glu",
            "activation": "SiLU",
        },
        {
            "name": "FFN with No Bias",
            "ffn_type": "standard",
            "no_bias": True,
        },
        {
            "name": "FFN with Post-Activation LayerNorm",
            "ffn_type": "standard",
            "post_type": "post_standard",
            "post_act_ln": True,
        },
        {
            "name": "FFN with Scaled Output",
            "ffn_type": "standard",
            "post_type": "post_scaled",
            "scale_factor": 0.5,
            "learnable_scale": True,
        },
        {
            "name": "MoE FFN with Soft Router",
            "ffn_type": "standard",
            "use_moe": True,
            "num_experts": 4,
            "top_k": 2,
            "router_type": "soft_router",
        },
        {
            "name": "MoE FFN with Hard Router",
            "ffn_type": "standard",
            "use_moe": True,
            "num_experts": 4,
            "top_k": 1,
            "router_type": "hard_router",
        },
    ]

    models = {}

    for config in test_configs:
        name = config.pop("name")
        print(f"\nCreating {name}...")

        model = create_ffn_transformer(
            num_tokens=NUM_TOKENS,
            max_seq_len=BLOCK_SIZE,
            dim=EMBEDDING_DIM,
            n_layers=2,  # Use fewer layers for faster testing
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

    print("\n✅ All FFN tests passed!")


if __name__ == "__main__":
    test_ffn_transformers()
