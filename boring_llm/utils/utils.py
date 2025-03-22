from typing import Dict, Any
import inspect
from boring_utils.colorprint import tprint


class PrintInitParamsMixin:
    """A mixin that logs initialization parameters for debugging purposes"""
    
    def __init_debug__(self):
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