import math
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
# from einops import rearrange

from boring_utils.utils import cprint
from boring_utils.helpers import DEBUG

from torch import Tensor
from typing import Optional, Tuple, Union


def SimpleSinusoidalPositionalEncoding(num_hiddens: int, device: Union[str, torch.device] = "cpu", max_len: int = 1000) -> Tensor:
    """
    Create sinusoidal positional encoding with the specified maximum length `max_len` 
    and embedding size `num_hiddens`.
    """
    # Create positional encoding matrix with size max_len x num_hiddens
    pe = torch.zeros(max_len, num_hiddens, device=device)
    
    # Create a tensor that represents the positions
    pos = torch.arange(max_len, dtype=torch.float32, device=device).unsqueeze(1)
    
    # Compute the positional encodings once, using the formula provided
    # two implementations of the same formula
    div_term = torch.exp(-math.log(10000) * torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens).to(device)
    # div_term = 1 / torch.pow(10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens).to(device)
    pe[:, 0::2] = torch.sin(pos * div_term)
    pe[:, 1::2] = torch.cos(pos * div_term)
    
    return pe


class SinusoidalPositionalEncoding(nn.Module):
    """
    Create sinusoidal positional encoding with the specified maximum length `max_len` 
    and embedding size `num_hiddens`.
    """

    def __init__(self, num_hiddens: int, dropout: Optional[float] = None, max_len: int = 1000):
        super().__init__()
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        # Create a long enough P
        self.pe = torch.zeros((1, max_len, num_hiddens))

        pos = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) 
        div_term = torch.exp(torch.arange(0, num_hiddens, 2, dtype=torch.float32) * (-math.log(10000) / num_hiddens))
        self.pe[:, :, 0::2] = torch.sin(pos * div_term)
        self.pe[:, :, 1::2] = torch.cos(pos * div_term)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:, :x.shape[1], :].to(x.device)

        # [batch size, time steps (seq length), channels]
        if self.dropout is not None:
            return self.dropout(x)
        return x


class LearnedPositionalEncoding(nn.Module):
    """
    Create learned positional encoding with the specified maximum length `max_len`
    and embedding size `num_hiddens`.
    This version uses an nn.Embedding layer for positional encoding, similar to the nanoGPT.
    """

    def __init__(self, num_hiddens: int, dropout: Optional[float] = None, max_len: int = 1000):
        super().__init__()
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        self.pe = nn.Embedding(max_len, num_hiddens)
        self.num_hiddens = num_hiddens

    def forward(self, x: Tensor) -> Tensor:
        # B, T = x.shape, where B is batch size and T is sequence length (block size)
        pos = torch.arange(x.shape[1], device=x.device).expand(x.shape[0], x.shape[1])
        x = x + self.pe(pos)

        # [batch size, time steps (seq length), channels]
        if self.dropout is not None:
            return self.dropout(x)
        return x

