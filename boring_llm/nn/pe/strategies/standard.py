from typing import Optional
from pydantic import Field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange

from boring_llm.nn.norm.core import l2norm
from boring_llm.nn.pe.base import PositionalEncoding
from boring_llm.nn.pe.factory import PositionalEncodingFactory, PositionalEncodingConfigFactory
from boring_utils.helpers import VERBOSE


PositionalEncodingConfigFactory.register("fixed")({})
@PositionalEncodingFactory.register("fixed")
class FixedPositionalEncoding(PositionalEncoding):
    """
    Sinusoidal positional embeddings from the "Attention Is All You Need" paper
    - Even indices (2i):   $PE_{(pos, 2i)}   = \sin(pos/10000^{2i/d})$
    - Odd indices (2i+1):  $PE_{(pos, 2i+1)} = \cos(pos/10000^{2i/d})$
    """
    def __init__(self, dim: int, **kwargs):
        super().__init__()
        if VERBOSE: self.__print_init_args__()
        # [0, 2, 4, ..., dim] / dim = [0, 2/dim, 4/dim, ..., 1]
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))  # [dim/2]
        # NOTE: if we use self.inv_freq, then it won't be included in the state_dict
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x: Tensor, pos: Optional[Tensor] = None, seq_dim: int = 1, offset: int = 0) -> Tensor:
        """
        Args:
            x: Input tensor to determine sequence length
            pos: Optional position indices. If None, uses sequential positions
            seq_dim: Dimension containing sequence length
            offset: Offset for positions
        
        Returns:
            Positional embeddings of shape matching input
        """
        if pos is None: pos = torch.arange(x.shape[seq_dim], device=x.device)
        pos = pos.type_as(self.inv_freq) + offset  # [seq_len]
        sinusoid_inp = pos.unsqueeze(-1) * self.inv_freq  # [seq_len, 1] * [dim/2] -> [seq_len, dim/2]
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)  # [seq_len, dim]
        return emb


PositionalEncodingConfigFactory.register("absolute")({
    "l2norm_embed": (bool, Field(default=False, description="Whether to L2 normalize embeddings"))
})
@PositionalEncodingFactory.register("absolute")
class AbsolutePositionalEncoding(PositionalEncoding):
    """
    Learnable absolute positional embeddings
    """
    def __init__(self, dim: int, max_seq_len: int, l2norm_embed: bool = False, **kwargs):
        super().__init__()
        if VERBOSE: self.__print_init_args__()
        self.l2norm_embed = l2norm_embed
        self.scale = dim ** -0.5 if not l2norm_embed else 1.
        self.max_seq_len = max_seq_len
        self.emb = nn.Embedding(max_seq_len, dim)

    def forward(self, x: Tensor, pos: Optional[Tensor] = None, seq_start_pos: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x: Input tensor to determine sequence length
            pos: Optional position indices. If None, uses sequential positions
            seq_start_pos: Optional sequence start position for offset calculation
        
        Returns:
            Positional embeddings of shape matching input
        """
        seq_len, device = x.shape[1], x.device
        assert seq_len <= self.max_seq_len, f'Sequence length {seq_len} exceeds maximum sequence length {self.max_seq_len} for absolute positional embedding'

        if pos is None: pos = torch.arange(seq_len, device=device)
        if seq_start_pos is not None:
            pos = (pos - seq_start_pos[..., None]).clamp(min=0)

        pos_emb = self.emb(pos)
        pos_emb = pos_emb * self.scale
        return l2norm(pos_emb) if self.l2norm_embed else pos_emb


PositionalEncodingConfigFactory.register("none")({})
@PositionalEncodingFactory.register("none")
class NonePositionalEncoding(PositionalEncoding):
    """
    No positional encoding - identity function
    """
    def __init__(self, **kwargs):
        super().__init__()
        if VERBOSE: self.__print_init_args__()
        
    def forward(self, x: Tensor, **kwargs) -> Tensor:
        """
        Return input as is (no positional encoding)
        
        Args:
            x: Input tensor
            
        Returns:
            The input tensor unchanged
        """
        return x