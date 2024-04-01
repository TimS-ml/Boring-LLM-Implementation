'''
- copyed from tinygrad/docs/env_vars.md
'''

from __future__ import annotations
import os, contextlib
from typing import Dict, Tuple, Union, List, ClassVar


def getenv(key:str, default=0): return type(default)(os.getenv(key, default))


class Context(contextlib.ContextDecorator):
    stack: ClassVar[List[dict[str, int]]] = [{}]

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __enter__(self):
        Context.stack[-1] = {
            k: o.value
            for k, o in ContextVar._cache.items()
        }  # Store current state.
        for k, v in self.kwargs.items():
            ContextVar._cache[k].value = v  # Update to new temporary state.
        Context.stack.append(
            self.kwargs
        )  # Store the temporary state so we know what to undo later.

    def __exit__(self, *args):
        for k in Context.stack.pop():
            ContextVar._cache[k].value = Context.stack[-1].get(
                k, ContextVar._cache[k].value)


class ContextVar:
    _cache: ClassVar[Dict[str, ContextVar]] = {}
    value: int

    def __new__(cls, key, default_value):
        if key in ContextVar._cache: return ContextVar._cache[key]
        instance = ContextVar._cache[key] = super().__new__(cls)
        instance.value = getenv(key, default_value)
        return instance

    def __bool__(self):
        return bool(self.value)

    def __ge__(self, x):
        return self.value >= x

    def __gt__(self, x):
        return self.value > x

    def __lt__(self, x):
        return self.value < x


DEBUG = ContextVar("DEBUG", 0)

