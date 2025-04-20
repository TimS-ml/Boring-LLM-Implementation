import torch
import torch.nn as nn
from torch import Tensor
from typing import Union, Any

from boring_llm.nn.activation.config import ActivationType
from boring_llm.nn.activation.config import ActivationConfig


def get_activation(activation_type: Union[str, ActivationType], **kwargs: Any) -> nn.Module:
    """
    Get activation function module
    
    Args:
        activation_type: Activation type (string or enum)
        **kwargs: Additional arguments for activation function
    
    Returns:
        Activation function module
    """
    # Convert enum to string if needed
    if isinstance(activation_type, ActivationType):
        activation_type = activation_type.value
    
    # Handle built-in PyTorch activations    
    if activation_type == "relu":
        return nn.ReLU(**kwargs)
    elif activation_type == "gelu":
        return nn.GELU(**kwargs)
    elif activation_type == "silu" or activation_type == "swish":
        return nn.SiLU(**kwargs)
    elif activation_type == "sigmoid":
        return nn.Sigmoid(**kwargs)
    elif activation_type == "tanh":
        return nn.Tanh(**kwargs)
    elif activation_type == "relu_squared":
        from boring_llm.nn.activation.activation import ReluSquared
        return ReluSquared()
    else:
        raise ValueError(f"Unknown activation type: {activation_type}")


def get_activation_from_config(config: ActivationConfig) -> nn.Module:
    """
    Create activation function from config object
    
    Args:
        config: Activation function configuration
        
    Returns:
        Activation function module
    """
    activation_type = config.get_type_value()
    kwargs = {}
    
    # Add activation-specific parameters
    if hasattr(config, 'inplace'):
        kwargs['inplace'] = config.inplace
        
    return get_activation(activation_type, **kwargs)


if __name__ == "__main__":
    config = ActivationConfig(type=ActivationType.GELU)
    activation = get_activation_from_config(config)
    
    x = torch.randn(2, 3)
    y = activation(x)
    print(f"Input: {x}")
    print(f"Output: {y}")