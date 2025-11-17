"""
Tiny Transformer with Customizable Connection Strategies

Demonstrates how to use different connection strategies from boring_llm.nn.connections
with the base transformer implementation.
"""
from __future__ import annotations
import torch
from torch import nn, Tensor
from typing import Optional, Type
from jaxtyping import Float

from boring_utils.utils import get_device
from boring_llm.tiny.tiny_base import TinyTransformBlock, TinyDecoder, TinyMultiHeadAttention, TinyFeedForward
from boring_llm.nn.connections import create_connection, ConnectionConfig

device = get_device()


class ConnectionsTransformerBlock(nn.Module):
    """
    Transformer block with customizable connection strategies

    Instead of hardcoded residual connections, uses flexible connections from nn.connections
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
            conn_type: str = "residual",
            conn_config: Optional[ConnectionConfig] = None,
            **conn_kwargs
        ):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList([])
        self.causal = causal
        self.cross_attend = cross_attend

        for layer_idx in range(n_layers):
            # Create connection layers for attention and FFN
            if conn_config is not None:
                # Update layer_index for hyper_connection
                config_dict = conn_config.model_dump()
                if conn_type == "hyper_connection":
                    config_dict['layer_index'] = layer_idx
                attn_conn = create_connection(conn_config.type, **config_dict)
                ffn_conn = create_connection(conn_config.type, **config_dict)
                cross_attn_conn = create_connection(conn_config.type, **config_dict) if cross_attend else None
            else:
                conn_kwargs_with_idx = {**conn_kwargs}
                if conn_type == "hyper_connection":
                    conn_kwargs_with_idx['layer_index'] = layer_idx

                attn_conn = create_connection(
                    conn_type,
                    dim_model=dim,
                    **conn_kwargs_with_idx
                )
                ffn_conn = create_connection(
                    conn_type,
                    dim_model=dim,
                    **conn_kwargs_with_idx
                )
                cross_attn_conn = create_connection(
                    conn_type,
                    dim_model=dim,
                    **conn_kwargs_with_idx
                ) if cross_attend else None

            self.layers.append(
                nn.ModuleList([
                    nn.LayerNorm(dim),
                    TinyMultiHeadAttention(dim, n_head, d_head, causal=causal, dropout=dropout),
                    attn_conn,
                    nn.LayerNorm(dim) if cross_attend else None,
                    TinyMultiHeadAttention(dim, n_head, d_head, causal=False, dropout=dropout) if cross_attend else None,
                    cross_attn_conn,
                    nn.LayerNorm(dim),
                    TinyFeedForward(dim, mul=ffn_mul, dropout=dropout),
                    ffn_conn,
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
                norm1, attn, attn_conn, norm2, cross_attn, cross_attn_conn, norm3, ff, ffn_conn = parts

                # Self-attention with custom connection
                attn_out = attn(norm1(x))
                x = attn_conn(attn_out, residual=x)

                # Cross-attention with custom connection
                cross_attn_out = cross_attn(norm2(x), context=context)
                x = cross_attn_conn(cross_attn_out, residual=x)

                # Feedforward with custom connection
                ff_out = ff(norm3(x))
                x = ffn_conn(ff_out, residual=x)
            else:
                norm1, attn, attn_conn, _, _, _, norm3, ff, ffn_conn = parts

                # Self-attention with custom connection
                attn_out = attn(norm1(x))
                x = attn_conn(attn_out, residual=x)

                # Feedforward with custom connection
                ff_out = ff(norm3(x))
                x = ffn_conn(ff_out, residual=x)

        return x


class ConnectionsTransformerWrapper(nn.Module):
    """TinyTransformBlock with customizable connections"""
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
            conn_type: str = "residual",
            conn_config: Optional[ConnectionConfig] = None,
            **conn_kwargs
        ):
        super().__init__()
        self.dim = dim
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Parameter(torch.randn(1, max_seq_len, dim))
        self.max_seq_len = max_seq_len

        # Use custom connections transformer
        self.transformer = ConnectionsTransformerBlock(
            dim=dim,
            n_layers=n_layers,
            n_head=n_head,
            d_head=d_head,
            ffn_mul=ffn_mul,
            dropout=dropout,
            cross_attend=cross_attend,
            causal=True,  # Assuming decoder
            conn_type=conn_type,
            conn_config=conn_config,
            **conn_kwargs
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


def create_connections_transformer(
    num_tokens: int,
    max_seq_len: int,
    dim: int,
    n_layers: int,
    n_head: int,
    d_head: int,
    ffn_mul: int,
    conn_type: str = "residual",
    dropout: float = 0.0,
    **conn_kwargs
) -> ConnectionsTransformerWrapper:
    """Convenience function to create transformer with custom connections"""
    return ConnectionsTransformerWrapper(
        num_tokens=num_tokens,
        max_seq_len=max_seq_len,
        dim=dim,
        n_layers=n_layers,
        n_head=n_head,
        d_head=d_head,
        ffn_mul=ffn_mul,
        dropout=dropout,
        conn_type=conn_type,
        **conn_kwargs
    )


def test_connections_transformers():
    """Test different connection strategies"""
    from boring_llm.base.tiny_config import (
        NUM_TOKENS, BLOCK_SIZE, EMBEDDING_DIM,
        N_LAYER, N_HEAD, D_HEAD, FFN_MUL,
        BATCH_SIZE, DROPOUT
    )

    test_configs = [
        {
            "name": "Standard Residual",
            "conn_type": "residual",
        },
        {
            "name": "Scaled Residual",
            "conn_type": "residual",
            "scale_residual": True,
        },
        {
            "name": "GRU Gated Residual",
            "conn_type": "gru_gating",
        },
        {
            "name": "Layer Scale",
            "conn_type": "layer_scale",
            "init_value": 1e-4,
        },
        {
            "name": "Layer Scale with Unit Offset",
            "conn_type": "layer_scale",
            "init_value": 0.0,
            "unit_offset": True,
        },
    ]

    models = {}

    for config in test_configs:
        name = config.pop("name")
        print(f"\nCreating {name}...")

        model = create_connections_transformer(
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

    print("\n✅ All connection tests passed!")


if __name__ == "__main__":
    test_connections_transformers()
