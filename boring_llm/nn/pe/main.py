import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

from boring_llm.nn.pe.base import PositionalEncoding
from boring_llm.nn.pe.config import PositionalEncodingConfig
from boring_llm.nn.pe.factory import PositionalEncodingFactory
from boring_llm.nn.pe.config import create_pe_config


class BoringPositionalEncoding(nn.Module):
    """
    Main positional encoding module that uses strategy pattern
    to support different types of positional encodings
    """
    def __init__(self, config: PositionalEncodingConfig):
        super().__init__()
        self.config = config
        self.max_seq_len = config.max_seq_len
        
        pe_type = config.type
        factory_args = config.model_dump(exclude={"type", "max_seq_len"})
        self.pe_strategy = PositionalEncodingFactory.create(
            encoding_type=pe_type,
            **factory_args
        )
    
    def forward(self, x: Tensor, pos: Optional[Tensor] = None, **kwargs) -> Tensor:
        """
        Apply positional encoding to input tensor
        
        Args:
            x: Input tensor of shape [batch, seq_len, dim]
            pos: Optional position indices. If None, uses sequential positions
            **kwargs: Additional arguments passed to the strategy
            
        Returns:
            Tensor with positional information
        """
        seq_len, device = x.shape[1], x.device
        
        # Common validation and default generation
        assert seq_len <= self.max_seq_len, f'Sequence length {seq_len} exceeds maximum sequence length {self.max_seq_len}'
        
        if pos is None:
            pos = torch.arange(seq_len, device=device)
            
        # Delegate to the strategy implementation
        return self.pe_strategy.apply(pos=pos, **kwargs)


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
