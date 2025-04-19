from typing import Dict, Type
import torch.nn as nn

from boring_llm.nn.ffn.base import FeedForward


class FeedForwardFactory:
    """Factory for creating feed forward network modules"""
    
    _registry: Dict[str, Type[FeedForward]] = {}
    
    @classmethod
    def register(cls, name: str):
        """Register a feed forward network implementation"""
        def decorator(strategy_class: Type[FeedForward]):
            cls._registry[name] = strategy_class
            return strategy_class
        return decorator
    
    @classmethod
    def create(cls, ffn_type: str, **kwargs) -> FeedForward:
        """
        Create a feed forward network module
        
        Args:
            ffn_type: Type of feed forward network ('standard', 'glu', etc.)
            **kwargs: Arguments for the specific feed forward type
        
        Returns:
            A feed forward network module
        """
        import boring_llm.nn.ffn.strategies
        if ffn_type not in cls._registry:
            raise ValueError(f"Unknown feed forward network type: {ffn_type}")
            
        return cls._registry[ffn_type](**kwargs)
