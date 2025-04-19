from typing import Dict, Type, List, Any, TypeVar, Generic, ClassVar

T = TypeVar('T')

class BaseFactory(Generic[T]):
    """Base factory class for creating various modules"""

    _registry: Dict[str, Type[T]] = {}
    
    @classmethod
    def register(cls, name: str):
        """Register an implementation"""
        def decorator(strategy_class: Type[T]):
            cls._registry[name] = strategy_class
            return strategy_class
        return decorator
    
    @classmethod
    def get_available_types(cls) -> List[str]:
        """Return all registered types"""
        return list(cls._registry.keys())
    
    @classmethod
    def is_valid_type(cls, type_name: str) -> bool:
        """Check if the type is valid"""
        return type_name in cls._registry
    
    @classmethod
    def create(cls, type_name: str, **kwargs) -> T:
        """
        Create a module
        
        Args:
            type_name: Type of module
            **kwargs: Arguments for the specific type
        
        Returns:
            A module instance
        """
        if type_name not in cls._registry:
            raise ValueError(f"Unknown type: {type_name}")
            
        return cls._registry[type_name](**kwargs)


class BaseConfigFactory(Generic[T]):
    """Base config factory class for managing different type configurations"""
    
    _type_configs: ClassVar[Dict[str, T]] = {}
    
    @classmethod
    def register(cls, type_name: str):
        """Decorator for registering specific configuration parameters for a type"""
        def decorator(config_dict: T):
            cls._type_configs[type_name] = config_dict
            return config_dict
        return decorator
    
    @classmethod
    def get_config_fields(cls, type_name: str) -> T:
        """Get configuration fields for a specific type"""
        return cls._type_configs.get(type_name, {})
    
    @classmethod
    def get_all_config_fields(cls) -> Dict[str, T]:
        """Get all configuration fields for all types"""
        return cls._type_configs