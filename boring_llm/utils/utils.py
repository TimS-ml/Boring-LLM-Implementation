from typing import Dict, Any
import inspect
import torch

from boring_utils.colorprint import tprint
from boring_utils.utils import get_device
from boring_utils.helpers import VERBOSE


def debug_init(cls):
    """
    Decorator to add debug initialization logging to classes
    Usage: check out ComponentRegistry.register
    Affects only classes that are registered
    """
    original_init = cls.__init__
    
    def new_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if VERBOSE: self.__print_init_args__()
    
    cls.__init__ = new_init
    return cls


class DebugInitMeta(type):
    """
    Metaclass that automatically adds debug initialization logging
    Usage: 
        class MyClass(nn.Module, ABC, PrintInitParamsMixin, metaclass=DebugInitMeta): ...
    Affects all classes that inherit MyClass
    """
    
    def __call__(cls, *args, **kwargs):
        instance = super().__call__(*args, **kwargs)
        if VERBOSE and hasattr(instance, '__print_init_args__'):
            # We need to manually get the initialization parameters because the metaclass is called after the instance is created
            instance._debug_init_args = args
            instance._debug_init_kwargs = kwargs
            instance.__print_init_args_meta__()
        return instance


class PrintInitParamsMixin:
    """A mixin that logs initialization parameters for debugging purposes"""
    
    def __print_init_args__(self):
        frame = inspect.currentframe().f_back
        args, _, _, local_vars = inspect.getargvalues(frame)
        
        if 'self' in args:
            args.remove('self')
            
        # Separate regular args and kwargs
        self._init_args = {arg: local_vars[arg] for arg in args}
        self._init_kwargs = local_vars.get('kwargs', {})
            
        # class_name = self.__class__.__name__
        # tprint(f"Initializing {class_name}", sep='*')
        
        if self._init_args:
            tprint("Args", sep='*', c='gray')
            for k, v in self._init_args.items():
                print(f"    {k}: {v}")

        if self._init_kwargs:
            tprint("Kwargs", sep='*', c='gray')
            for k, v in self._init_kwargs.items():
                print(f"    {k}: {v}")
    
    def get_init_args(self) -> Dict[str, Any]:
        """Get all initialization arguments (both positional and keyword)"""
        args = self._init_args.copy() if hasattr(self, '_init_args') else {}
        kwargs = self._init_kwargs.copy() if hasattr(self, '_init_kwargs') else {}
        return {**args, **kwargs}

    def __print_init_args_meta__(self):
        """Version for metaclass that uses stored args"""
        class_name = self.__class__.__name__
        
        if hasattr(self, '_debug_init_args') and self._debug_init_args:
            tprint(f"{class_name} Args (Meta)", sep='*', c='blue')
            for i, arg in enumerate(self._debug_init_args):
                print(f"    arg{i}: {arg}")
        
        if hasattr(self, '_debug_init_kwargs') and self._debug_init_kwargs:
            tprint(f"{class_name} Kwargs (Meta)", sep='*', c='blue')
            for k, v in self._debug_init_kwargs.items():
                print(f"    {k}: {v}")


def create_causal_mask(seq_q: int, seq_k: int, device: torch.device = get_device()):
    """
    Create a causal mask for attention.
    Args:
        seq_q: sequence length of query
        seq_k: sequence length of key
        device: device to create mask on
    Returns:
        Upper triangular boolean mask of shape (seq_q, seq_k)
    """
    return torch.triu(
            torch.ones(
                (seq_q, seq_k), 
                device=device, 
                dtype=torch.bool
            ), 
            diagonal=1
        )