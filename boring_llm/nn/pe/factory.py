from typing import Optional, Dict, Type, List, Any, ClassVar
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
    def get_available_types(cls) -> List[str]:
        """Return all registered encoding types"""
        return list(cls._registry.keys())
    
    @classmethod
    def is_valid_type(cls, encoding_type: str) -> bool:
        """Check if the encoding type is valid"""
        return encoding_type in cls._registry
    
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


class PositionalEncodingConfigFactory:
    """Factory for managing configuration parameters for different PE types"""
    
    _type_configs: ClassVar[Dict[str, Dict[str, Any]]] = {}
    
    @classmethod
    def register(cls, pe_type: str):
        """Decorator for registering specific configuration parameters for a PE type"""
        def decorator(config_dict: Dict[str, Any]):
            cls._type_configs[pe_type] = config_dict
            return config_dict
        return decorator
    
    @classmethod
    def get_config_fields(cls, pe_type: str) -> Dict[str, Any]:
        """Get configuration fields for a specific PE type"""
        return cls._type_configs.get(pe_type, {})
    
    @classmethod
    def get_all_config_fields(cls) -> Dict[str, Dict[str, Any]]:
        """Get all configuration fields for all PE types"""
        return cls._type_configs