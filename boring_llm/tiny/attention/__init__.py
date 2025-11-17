"""
Tiny Transformer with Customizable Attention

Provides simple transformer implementations showcasing different attention mechanisms.
"""

from boring_llm.tiny.attention.base_attention import (
    AttentionTransformerWrapper,
    create_attention_transformer,
)

__all__ = [
    "AttentionTransformerWrapper",
    "create_attention_transformer",
]
