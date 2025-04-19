# TODO: I might merge all the factory pattern into BoringPositionalEncoding?

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

from boring_llm.nn.pe.base import PositionalEncoding
from boring_llm.nn.pe.config import PositionalEncodingConfig
from boring_llm.nn.pe.factory import PositionalEncodingFactory
from boring_llm.nn.pe.config import create_pe_config


class BoringPositionalEncoding(PositionalEncoding):
    """
    Main positional encoding module that uses strategy pattern
    to support different types of positional encodings
    """
    def __init__(self, config: PositionalEncodingConfig):
        super().__init__()
        self.config = config
        
        pe_type = config.type
        factory_args = config.model_dump(exclude={"type"})
        self.pe_strategy = PositionalEncodingFactory.create(
            encoding_type=pe_type,
            **factory_args
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


if __name__ == "__main__":
    from boring_llm.base.tiny_config import *
    pe_type="absolute"
    pe_args = create_pe_config(pe_type)(
                    dim_model=EMBEDDING_DIM,
                    max_seq_len=BLOCK_SIZE,
                    l2norm_embed=True
                )
    pe = BoringPositionalEncoding(pe_args)
    x = torch.randn(1, BLOCK_SIZE, EMBEDDING_DIM)
    print(pe(x).shape)
