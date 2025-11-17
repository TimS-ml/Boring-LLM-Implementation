"""
Memory mechanisms for transformers

Includes:
- Memory Tokens (register tokens, meta tokens)
- Persistent Memory Key/Values
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
from einops import repeat


class MemoryTokens(nn.Module):
    """
    Memory Tokens (Register Tokens / Meta Tokens)

    Learnable tokens that are prepended to the input sequence and attend
    to all input tokens. Helps with:
    - Reducing outliers in attention (MetaAI finding)
    - Global information aggregation
    - Attending to "nothing" when needed

    Used in:
    - Vision in Transformers (Meta AI)
    - Hymba (Nvidia) - termed "meta tokens"
    - Various recent models for stability

    Args:
        dim: Model dimension
        num_memory_tokens: Number of memory tokens to add
    """

    def __init__(self, dim: int, num_memory_tokens: int = 20):
        super().__init__()
        self.num_memory_tokens = num_memory_tokens

        # Learnable memory tokens
        self.memory_tokens = nn.Parameter(torch.randn(num_memory_tokens, dim))
        nn.init.normal_(self.memory_tokens, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        """
        Prepend memory tokens to input

        Args:
            x: Input tensor [batch, seq, dim]

        Returns:
            Tensor with memory tokens prepended [batch, seq + num_memory, dim]
        """
        batch_size = x.shape[0]

        # Expand memory tokens for batch
        memory = repeat(self.memory_tokens, 'n d -> b n d', b=batch_size)

        # Prepend to input
        return torch.cat([memory, x], dim=1)

    def remove_memory(self, x: Tensor) -> Tensor:
        """Remove memory tokens from output"""
        return x[:, self.num_memory_tokens:]


class PersistentMemoryKV(nn.Module):
    """
    Persistent Memory Key/Values for Attention

    From "Augmenting Self-attention with Persistent Memory"
    https://arxiv.org/abs/1907.01470

    Adds learned memory key/value pairs that attend to all queries.
    Can help replace or augment feedforward layers.

    Args:
        dim_head: Dimension per attention head
        num_heads: Number of attention heads (or kv_heads for GQA)
        num_mem_kv: Number of memory key/value pairs
    """

    def __init__(self, dim_head: int, num_heads: int = 8, num_mem_kv: int = 16):
        super().__init__()
        self.num_mem_kv = num_mem_kv

        # Learnable memory keys and values
        self.mem_k = nn.Parameter(torch.randn(num_heads, num_mem_kv, dim_head))
        self.mem_v = nn.Parameter(torch.randn(num_heads, num_mem_kv, dim_head))

        # Initialize
        nn.init.normal_(self.mem_k, std=dim_head ** -0.5)
        nn.init.normal_(self.mem_v, std=dim_head ** -0.5)

    def forward(self, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """
        Prepend memory key/values to attention keys/values

        Args:
            k: Key tensor [batch, heads, seq, dim_head]
            v: Value tensor [batch, heads, seq, dim_head]

        Returns:
            Tuple of (k_with_mem, v_with_mem)
        """
        batch_size = k.shape[0]

        # Expand memory for batch
        mem_k = repeat(self.mem_k, 'h n d -> b h n d', b=batch_size)
        mem_v = repeat(self.mem_v, 'h n d -> b h n d', b=batch_size)

        # Prepend to keys and values
        k = torch.cat([mem_k, k], dim=2)
        v = torch.cat([mem_v, v], dim=2)

        return k, v


if __name__ == "__main__":
    print("Testing Memory modules...")

    # Test Memory Tokens
    memory_tokens = MemoryTokens(dim=512, num_memory_tokens=20)
    x = torch.randn(2, 100, 512)
    x_with_mem = memory_tokens(x)
    print(f"\nMemory Tokens:")
    print(f"  Input shape: {x.shape}")
    print(f"  With memory: {x_with_mem.shape}")
    print(f"  Removed: {memory_tokens.remove_memory(x_with_mem).shape}")

    # Test Persistent Memory KV
    mem_kv = PersistentMemoryKV(dim_head=64, num_heads=8, num_mem_kv=16)
    k = torch.randn(2, 8, 100, 64)
    v = torch.randn(2, 8, 100, 64)
    k_mem, v_mem = mem_kv(k, v)
    print(f"\nPersistent Memory KV:")
    print(f"  Original K: {k.shape}")
    print(f"  With memory: {k_mem.shape}")
    print(f"  Memory size: {mem_kv.num_mem_kv}")

    print("\n✅ Memory module tests passed!")
