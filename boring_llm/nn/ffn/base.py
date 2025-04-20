import torch.nn as nn
from torch import Tensor
from abc import ABC, abstractmethod
from typing import Optional, Tuple

from boring_llm.utils.utils import PrintInitParamsMixin


class FeedForwardTransform(nn.Module, ABC, PrintInitParamsMixin):
    """
    Base abstract class for all feed forward transformation strategies
    """
    
    @abstractmethod
    def apply(self, x: Tensor, **kwargs) -> Tensor:
        """
        Apply core transformation to input tensor
        
        Args:
            x: Input tensor
            **kwargs: Additional arguments specific to implementation
            
        Returns:
            Transformed tensor
        """
        pass
    
    @property
    @abstractmethod
    def output_dim(self) -> int:
        pass

    # more entry points for feed forward could be added