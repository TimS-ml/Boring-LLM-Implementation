"""
FFN Module - Simplified Feed Forward Network Implementation

This module provides:
- Registry-based FFN transforms (standard, glu, post_standard)
- Configuration management with inheritance from BaseConfig
- Simplified main module with convenience functions

Usage:
    from boring_llm.nn.ffn import create_ffn, SimplifiedFeedForward, FFNConfig
    
    # Quick creation
    ffn = create_ffn("glu", dim_model=512, mult_dim=2)
    
    # With config
    config = FFNConfig(type="standard", dim_model=512)
    ffn = SimplifiedFeedForward(config)
"""

from .main import BoringFeedForward, FFNConfig, create_ffn
from .registry import ffn_registry, FFNTransform

__all__ = [
    # Main components
    "BoringFeedForward",
    "FFNConfig", 
    "create_ffn",
    
    # Registry and base classes
    "ffn_registry",
    "FFNTransform",
]
