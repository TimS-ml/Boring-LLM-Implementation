import torch.nn as nn
from torch import Tensor
from abc import ABC, abstractmethod
from typing import Optional

from boring_llm.utils.utils import PrintInitParamsMixin


class PositionalEncodingTransform(nn.Module, ABC, PrintInitParamsMixin):
    """Base abstract class for all positional encoding implementations
    NOTE: we need nn.Module to be able to use register_buffer
    """
    
    @abstractmethod
    def apply(self, pos: Tensor, **kwargs) -> Tensor:
        """
        Apply positional encoding to input tensor
        
        Args:
            pos: Position indices
            **kwargs: Additional arguments specific to implementation
            
        Returns:
            Tensor with positional information
        """
        pass