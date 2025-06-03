"""
FFN Module - Simplified Feed Forward Network Implementation

This module provides:
- Registry-based FFN transforms (standard, glu, post_standard, router)
- Configuration management with inheritance from BaseConfig
- Simplified main module with convenience functions
- Separate MOE implementation for Mixture of Experts

Usage:
    from boring_llm.nn.ffn import create_ffn, BoringFeedForward, FFNConfig, create_moe_ffn, BoringFeedForwardMOE
    
    # Quick creation
    ffn = create_ffn("glu", dim_model=512, mult_dim=2)
    
    # MOE creation with different expert types
    moe_ffn = create_moe_ffn(num_experts=8, top_k=2, expert_type="standard", dim_model=512)
    moe_glu_ffn = create_moe_ffn(num_experts=8, top_k=2, expert_type="glu", dim_model=512)
    
    # With config
    config = FFNConfig(type="standard", dim_model=512)
    ffn = BoringFeedForward(config)
    
    moe_config = MOEConfig(expert_type="glu", num_experts=8, dim_model=512)
    moe_ffn = BoringFeedForwardMOE(moe_config)
"""

from .main import BoringFeedForward, BoringFeedForwardMOE, FFNConfig, create_ffn, create_moe_ffn
from .registry import ffn_registry, FFNTransform

__all__ = [
    # Main components
    "BoringFeedForward",
    "BoringFeedForwardMOE",
    "FFNConfig", 
    "create_ffn",
    "create_moe_ffn",
    
    # Registry and base classes
    "ffn_registry",
    "FFNTransform",
]
