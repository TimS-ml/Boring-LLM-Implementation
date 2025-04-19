# from .feedforwards import *
from boring_llm.nn.ffn.main import BoringFeedForward
from boring_llm.nn.ffn.config import FeedForwardConfig, ActivationType

__all__ = ["BoringFeedForward", "FeedForwardConfig", "ActivationType"]
