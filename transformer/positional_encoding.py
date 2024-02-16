import math
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
# from einops import rearrange

from utils import *
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

    def __init__(self, num_hiddens: int, dropout: float, max_len: int = 1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # Create a long enough P
        self.pe = torch.zeros((1, max_len, num_hiddens))

        pos = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) 
        div_term = torch.exp(torch.arange(0, num_hiddens, 2, dtype=torch.float32) * (-math.log(10000) / num_hiddens))
        self.pe[:, :, 0::2] = torch.sin(pos * div_term)
        self.pe[:, :, 1::2] = torch.cos(pos * div_term)

    def forward(self, x):
        x = x + self.pe[:, :x.shape[1], :].to(x.device)
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """
    Create learned positional encoding with the specified maximum length `max_len`
    and embedding size `num_hiddens`.
    This version uses an nn.Embedding layer for positional encoding, similar to the nanoGPT example.
    """

    def __init__(self, num_hiddens: int, dropout: float, max_len: int = 1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.positional_encodings = nn.Embedding(max_len, num_hiddens)
        self.num_hiddens = num_hiddens

    def forward(self, x: torch.Tensor):
        # where B is batch size and T is sequence length (block size)
        B, T = x.shape
        positions = torch.arange(T, device=x.device).expand(B, T)  # Create a sequence of positions
        x = x + self.positional_encodings(positions)  # [B, T, C]
        return self.dropout(x)

