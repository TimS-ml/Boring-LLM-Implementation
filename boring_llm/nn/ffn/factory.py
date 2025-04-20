from typing import Dict, Type, Any, ClassVar
import torch.nn as nn

from boring_llm.base.base_factory import BaseFactory, BaseConfigFactory
from boring_llm.nn.ffn.base import FeedForwardTransform


class FeedForwardFactory(BaseFactory[FeedForwardTransform]):
    """Factory for creating feed forward transformation modules"""
    
    _registry: Dict[str, Type[FeedForwardTransform]] = {}

    @classmethod
    def is_post_processor(cls, ffn_type: str) -> bool:
        """Check if the given type is a post-processor"""
        return ffn_type.startswith("post_") 

    @classmethod
    def create(cls, ffn_type: str, **kwargs) -> FeedForwardTransform:
        """
        Create a feed forward transformation module
        
        Args:
            ffn_type: Type of feed forward transformation ('standard', 'glu', etc.)
            **kwargs: Arguments for the specific transformation type
        
        Returns:
            A feed forward transformation module
        """
        import boring_llm.nn.ffn.strategies
        return super().create(ffn_type, **kwargs)


class FeedForwardConfigFactory(BaseConfigFactory[Dict[str, Any]]):
    """Factory for managing configuration parameters for different feed forward types"""
    
    _type_configs: ClassVar[Dict[str, Dict[str, Any]]] = {}
