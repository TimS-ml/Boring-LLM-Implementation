from typing import Dict, Type, Any, Generic, TypeVar, Optional, Callable
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field, create_model
import torch.nn as nn
from torch import Tensor

from boring_llm.base.base_config import BaseConfig
from boring_llm.utils.utils import PrintInitParamsMixin

from boring_utils.helpers import VERBOSE
from boring_llm.utils.utils import debug_init


T = TypeVar('T', bound=nn.Module)


class ComponentTransform(nn.Module, ABC, PrintInitParamsMixin):
    """Base abstract class for all component transformations"""
    
    @abstractmethod
    def apply(self, x: Tensor, **kwargs) -> Tensor:
        """Apply transformation to input tensor"""
        pass
    
    @property
    def output_dim(self) -> Optional[int]:
        """Optional output dimension for components that change dimensionality"""
        return None


class ComponentRegistry(Generic[T]):
    """Universal registry for neural network components"""
    
    def __init__(self, component_name: str):
        self.component_name = component_name
        self._strategies: Dict[str, Type[T]] = {}
        self._configs: Dict[str, Dict[str, Any]] = {}
    
    def register(self, name: str, config_fields: Dict[str, Any] = None):
        """Register both strategy and its configuration in one step"""
        def decorator(strategy_class):
            if VERBOSE: strategy_class = debug_init(strategy_class)
            self._strategies[name] = strategy_class
            if config_fields:
                self._configs[name] = config_fields
            return strategy_class
        return decorator
    
    def create_strategy(self, type_name: str, **kwargs) -> T:
        """Create a strategy instance"""
        if type_name not in self._strategies:
            raise ValueError(f"Unknown {self.component_name} type: {type_name}")
        return self._strategies[type_name](**kwargs)
    
    def get_config_fields(self, type_name: str) -> Dict[str, Any]:
        """Get configuration fields for a specific type"""
        return self._configs.get(type_name, {})
    
    def get_available_types(self) -> list[str]:
        """Return all registered types"""
        return list(self._strategies.keys())
    
    def is_valid_type(self, type_name: str) -> bool:
        """Check if the type is valid"""
        return type_name in self._strategies


class ComponentConfig(BaseConfig):
    """Base configuration for neural network components"""
    type: str = Field(default="standard")
    
    @classmethod
    def create_typed_config(cls, registry: ComponentRegistry, component_type: str, **base_fields):
        """Create a type-specific configuration class"""
        fields = {
            "type": (str, Field(default=component_type)),
        }
        
        # Add base fields
        fields.update(base_fields)
        
        # Add type-specific fields
        type_fields = registry.get_config_fields(component_type)
        fields.update(type_fields)
        
        return create_model(
            f"{component_type.capitalize()}{cls.__name__}", 
            __base__=cls, 
            **fields
        )


class ComponentModule(nn.Module, Generic[T]):
    """Generic main module that uses strategy pattern"""
    
    def __init__(self, type_name: str, registry: ComponentRegistry[T], **kwargs):
        super().__init__()
        self.type_name = type_name
        self.registry = registry
        
        # Create the strategy based on type and kwargs
        self.strategy = registry.create_strategy(type_name, **kwargs)
    
    def forward(self, x: Tensor, **kwargs) -> Tensor:
        """Apply component transformation"""
        return self.strategy.apply(x, **kwargs) 