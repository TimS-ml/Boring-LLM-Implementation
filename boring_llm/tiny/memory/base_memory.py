"""
Tiny Transformer with Memory Features

Demonstrates how to use memory mechanisms from boring_llm.nn.memory
with the base transformer implementation.
"""
from __future__ import annotations
import torch
from torch import nn, Tensor
from typing import Optional, Type
from jaxtyping import Float

from boring_utils.utils import get_device
from boring_llm.tiny.tiny_base import TinyTransformBlock, TinyDecoder, TinyMultiHeadAttention, TinyFeedForward
from boring_llm.nn.memory.memory import MemoryTokens, PersistentMemoryKV

device = get_device()


class MemoryTransformerWrapper(nn.Module):
    """
    Transformer with memory features

    Supports:
    - Memory Tokens (register tokens)
    - Persistent Memory Key/Values
    """
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
            # Memory features
            num_memory_tokens: int = 0,
            num_mem_kv: int = 0,
        ):
        super().__init__()
        self.dim = dim
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Parameter(torch.randn(1, max_seq_len, dim))
        self.max_seq_len = max_seq_len
        self.num_memory_tokens = num_memory_tokens

        # Memory tokens
        self.memory_tokens = None
        if num_memory_tokens > 0:
            self.memory_tokens = MemoryTokens(dim, num_memory_tokens)

        # Persistent memory KV (if needed, integrate into custom attention)
        self.num_mem_kv = num_mem_kv
        if num_mem_kv > 0:
            print(f"Note: Persistent Memory KV ({num_mem_kv}) requires custom attention implementation.")
            print("This demo shows memory tokens only. For memory KV, use with custom attention.")

        # Use standard transformer
        self.transformer = transform_layer(
            dim=dim,
            n_layers=n_layers,
            n_head=n_head,
            d_head=d_head,
            ffn_mul=ffn_mul,
            dropout=dropout,
            cross_attend=cross_attend,
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

        # Token and positional embeddings
        x = self.token_emb(x)
        x = x + self.pos_emb[:, :seq_len]

        # Add memory tokens
        if self.memory_tokens is not None:
            x = self.memory_tokens(x)

        # Transformer
        if self.transformer.cross_attend:
            x = self.transformer(x, context=context)
        else:
            x = self.transformer(x)

        # Remove memory tokens if present
        if self.memory_tokens is not None:
            x = self.memory_tokens.remove_memory(x)

        return self.to_logits(x)


def create_memory_transformer(
    num_tokens: int,
    max_seq_len: int,
    dim: int,
    n_layers: int,
    n_head: int,
    d_head: int,
    ffn_mul: int,
    dropout: float = 0.0,
    num_memory_tokens: int = 0,
    num_mem_kv: int = 0,
    **kwargs
) -> MemoryTransformerWrapper:
    """Convenience function to create transformer with memory features"""
    return MemoryTransformerWrapper(
        num_tokens=num_tokens,
        max_seq_len=max_seq_len,
        dim=dim,
        n_layers=n_layers,
        n_head=n_head,
        d_head=d_head,
        ffn_mul=ffn_mul,
        dropout=dropout,
        num_memory_tokens=num_memory_tokens,
        num_mem_kv=num_mem_kv,
        **kwargs
    )


def test_memory_transformers():
    """Test memory transformers"""
    from boring_llm.base.tiny_config import (
        NUM_TOKENS, BLOCK_SIZE, EMBEDDING_DIM,
        N_LAYER, N_HEAD, D_HEAD, FFN_MUL,
        BATCH_SIZE, DROPOUT
    )

    test_configs = [
        {
            "name": "No Memory",
            "num_memory_tokens": 0,
        },
        {
            "name": "With 10 Memory Tokens",
            "num_memory_tokens": 10,
        },
        {
            "name": "With 20 Memory Tokens",
            "num_memory_tokens": 20,
        },
    ]

    models = {}

    for config in test_configs:
        name = config.pop("name")
        print(f"\nCreating {name}...")

        model = create_memory_transformer(
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

        # Check output shape (should match input shape regardless of memory tokens)
        expected_shape = (BATCH_SIZE, BLOCK_SIZE, NUM_TOKENS)
        assert logits.shape == expected_shape, \
            f"{name}: Expected {expected_shape}, got {logits.shape}"

        print(f"{name} output shape: {logits.shape} ✓")
        if model.num_memory_tokens > 0:
            print(f"  (Used {model.num_memory_tokens} memory tokens internally)")

    print("\n✅ All memory tests passed!")


if __name__ == "__main__":
    test_memory_transformers()
