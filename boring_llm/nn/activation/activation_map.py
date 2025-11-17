from typing import Type
import torch.nn as nn
import torch.nn.functional as F


def get_activation_by_name(name: str) -> Type[nn.Module]:
    """Get activation function by name using PyTorch's standard naming convention.
    
    Args:
        name: Standard PyTorch activation name (e.g., "ReLU", "GELU", "SiLU")
        
    Returns:
        Activation function class from torch.nn
        
    Examples:
        >>> relu_cls = get_activation_by_name("ReLU")
        >>> gelu_cls = get_activation_by_name("GELU") 
        >>> custom_cls = get_activation_by_name("ReluSquared")
    """
    name = name.strip()
    
    # Try to get from torch.nn first
    if hasattr(nn, name): return getattr(nn, name)
    elif hasattr(F, name): return getattr(F, name) 

    # Try custom activation functions
    try:
        from boring_llm.nn.activation import ReluSquared, SoLU
        custom_activations = {
            'ReluSquared': ReluSquared,
            'SoLU': SoLU,
        }
        if name in custom_activations:
            return custom_activations[name]
    except ImportError:
        pass
    
    raise ValueError(f"Unknown activation: {name}, please using PyTorch's standard naming convention")
