import os
import importlib
import inspect
from boring_llm.nn.ffn.base import FeedForwardTransform

strategies_dir = os.path.dirname(__file__)
py_files = [f[:-3] for f in os.listdir(strategies_dir) 
           if f.endswith('.py') and f != '__init__.py']

__all__ = []

# Import all modules in the directory
for module_name in py_files:
    module = importlib.import_module(f"boring_llm.nn.ffn.strategies.{module_name}")
    
    # Find all classes in the module that inherit from FeedForwardTransformation
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if issubclass(obj, FeedForwardTransform) and obj != FeedForwardTransform:
            __all__.append(name)
            globals()[name] = obj

# print(f"Loaded feed forward strategies: {__all__}")