import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

from boring_llm.nn.pe.base import PositionalEncoding
from boring_llm.nn.pe.config import PositionalEncodingConfig, PositionalEncodingType
from boring_llm.nn.pe.factory import PositionalEncodingFactory


class BoringPositionalEncoding(PositionalEncoding):
    """
    Main positional encoding module that uses strategy pattern
    to support different types of positional encodings
    """
    def __init__(self, config: PositionalEncodingConfig):
        super().__init__()
        self.config = config
        
        # Get proper dimension
        dim = config.dim_model if config.dim_model is not None else config.d_model
        
        # Create the appropriate PE implementation based on config
        self.pe_strategy = PositionalEncodingFactory.create(
            encoding_type=config.type.value,
            dim=dim,
            max_seq_len=config.max_seq_len,
            l2norm_embed=config.l2norm_embed,
            rotary_percentage=config.rotary_percentage,
            alibi_num_heads=config.alibi_num_heads
        )
    
    def forward(self, x: Tensor, **kwargs) -> Tensor:
        """
        Apply positional encoding to input tensor
        
        Args:
            x: Input tensor of shape [batch, seq_len, dim]
            **kwargs: Additional arguments passed to the strategy
            
        Returns:
            Tensor with positional information
        """
        return self.pe_strategy(x, **kwargs)