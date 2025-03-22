import torch
import torch.nn as nn
from torch import Tensor
from abc import ABC, abstractmethod
from typing import Optional


class PositionalEncoding(nn.Module, ABC):
    """Base abstract class for all positional encoding implementations"""
    
    @abstractmethod
    def forward(self, x: Tensor, **kwargs) -> Tensor:
        """
        Apply positional encoding to input tensor
        
        Args:
            x: Input tensor
            **kwargs: Additional arguments specific to implementation
            
        Returns:
            Tensor with positional information
        """
        pass