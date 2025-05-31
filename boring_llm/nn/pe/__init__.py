"""
Positional Encoding Module - Simplified PE Implementation

This module provides:
- Registry-based PE transforms (fixed, absolute, rotary, alibi, none)
- Configuration management with inheritance from BaseConfig
- Simplified main module with convenience functions

Usage:
    from boring_llm.nn.pe import create_pe, SimplifiedPositionalEncoding, PEConfig
    
    # Quick creation
    pe = create_pe("rotary", dim_model=512, rotary_percentage=0.5)
    
    # With config
    config = PEConfig(type="fixed", dim_model=512)
    pe = SimplifiedPositionalEncoding(config)
"""

from .main import BoringPositionalEncoding, PEConfig, create_pe
from .registry import pe_registry, PETransform

__all__ = [
    # Main components
    "BoringPositionalEncoding", 
    "PEConfig",
    "create_pe",
    
    # Registry and base classes
    "pe_registry",
    "PETransform",
]
