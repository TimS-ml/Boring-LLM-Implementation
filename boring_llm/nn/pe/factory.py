from typing import Optional, Dict, Type
import torch.nn as nn

from boring_llm.nn.pe.base import PositionalEncoding


class PositionalEncodingFactory:
    """Factory for creating positional encoding modules"""
    
    _registry: Dict[str, Type[PositionalEncoding]] = {}
    
    @classmethod
    def register(cls, name: str):
        """Register a positional encoding implementation"""
        def decorator(strategy_class: Type[PositionalEncoding]):
            cls._registry[name] = strategy_class
            return strategy_class
        return decorator
    
    @classmethod
    def create(cls, encoding_type: str, **kwargs) -> PositionalEncoding:
        """
        Create a positional encoding module
        
        Args:
            encoding_type: Type of positional encoding ('none', 'fixed', 'absolute', 'rotary', 'alibi', etc.)
            **kwargs: Arguments for the specific encoding type
        
        Returns:
            A positional encoding module
        """
        import boring_llm.nn.pe.strategies
        if encoding_type not in cls._registry:
            raise ValueError(f"Unknown positional encoding type: {encoding_type}")
            
        return cls._registry[encoding_type](**kwargs)