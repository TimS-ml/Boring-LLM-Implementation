import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

from boring_llm.nn.pe.base import PositionalEncoding
from boring_llm.nn.pe.config import PositionalEncodingConfig
from boring_llm.nn.pe.factory import PositionalEncodingFactory, PositionalEncodingConfigFactory


class BoringPositionalEncoding(PositionalEncoding):
    """
    Main positional encoding module that uses strategy pattern
    to support different types of positional encodings
    """
    def __init__(self, config: PositionalEncodingConfig):
        super().__init__()
        self.config = config

        pe_args = {
            "dim": config.dim_model,
            "max_seq_len": config.max_seq_len,
        }
    
        type_fields = PositionalEncodingConfigFactory.get_config_fields(config.type)
        for field_name in type_fields:
            if hasattr(config, field_name):
                pe_args[field_name] = getattr(config, field_name)
    
        # Create the appropriate PE implementation based on config
        self.pe_strategy = PositionalEncodingFactory.create(
            encoding_type=config.type,
            **pe_args
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
