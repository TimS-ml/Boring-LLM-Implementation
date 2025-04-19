from typing import Dict, Type, Any, ClassVar
from boring_llm.base.base_factory import BaseFactory, BaseConfigFactory
from boring_llm.nn.pe.base import PositionalEncoding

class PositionalEncodingFactory(BaseFactory[PositionalEncoding]):
    """Factory for creating positional encoding modules"""
    
    _registry: Dict[str, Type[PositionalEncoding]] = {}
    
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
        return super().create(encoding_type, **kwargs)


class PositionalEncodingConfigFactory(BaseConfigFactory[Dict[str, Any]]):
    """Factory for managing configuration parameters for different PE types"""
    
    _type_configs: ClassVar[Dict[str, Dict[str, Any]]] = {}
