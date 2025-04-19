import torch
import torch.nn as nn
from torch import Tensor
from abc import ABC, abstractmethod
from typing import Optional

from boring_llm.utils.utils import PrintInitParamsMixin


class PositionalEncoding(nn.Module, ABC, PrintInitParamsMixin):
    """Base abstract class for all positional encoding implementations
    NOTE: we need nn.Module to be able to use register_buffer
    """
    
    @abstractmethod
    def apply(self, x: Tensor, **kwargs) -> Tensor:
        """
        Apply positional encoding to input tensor
        
        Args:
            x: Input tensor
            **kwargs: Additional arguments specific to implementation
            
        Returns:
            Tensor with positional information
        """
        pass