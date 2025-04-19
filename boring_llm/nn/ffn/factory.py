from typing import Dict, Type, Any, ClassVar
from boring_llm.base.base_factory import BaseFactory, BaseConfigFactory
from boring_llm.nn.ffn.base import FeedForward


class FeedForwardFactory(BaseFactory[FeedForward]):
    """Factory for creating feed forward network modules"""
    
    _registry: Dict[str, Type[FeedForward]] = {}
    
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
        return super().create(ffn_type, **kwargs)


class FeedForwardConfigFactory(BaseConfigFactory[Dict[str, Any]]):
    """Factory for managing configuration parameters for different feed forward types"""
    
    _type_configs: ClassVar[Dict[str, Dict[str, Any]]] = {}
