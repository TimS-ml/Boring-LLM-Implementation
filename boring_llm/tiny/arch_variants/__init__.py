"""
Tiny Transformer with Architecture Variants

Provides simple transformer implementations showcasing different architectural patterns.
"""

from boring_llm.tiny.arch_variants.base_arch_variants import (
    SandwichNormTransformer,
    MacaronTransformer,
    create_sandwich_norm_transformer,
    create_macaron_transformer,
)

__all__ = [
    "SandwichNormTransformer",
    "MacaronTransformer",
    "create_sandwich_norm_transformer",
    "create_macaron_transformer",
]
