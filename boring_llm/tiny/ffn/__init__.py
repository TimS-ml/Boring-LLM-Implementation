"""
Tiny Transformer with Customizable Feed-Forward Networks

Provides simple transformer implementations showcasing different FFN mechanisms.
"""

from boring_llm.tiny.ffn.base_ffn import (
    FFNTransformerWrapper,
    create_ffn_transformer,
)

__all__ = [
    "FFNTransformerWrapper",
    "create_ffn_transformer",
]
