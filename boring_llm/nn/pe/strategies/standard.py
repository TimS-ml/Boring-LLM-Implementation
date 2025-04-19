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
    def __init__(self, dim_model: int, **kwargs):
        super().__init__()
        if VERBOSE: self.__print_init_args__()
        # [0, 2, 4, ..., dim] / dim = [0, 2/dim, 4/dim, ..., 1]
        inv_freq = 1. / (10000 ** (torch.arange(0, dim_model, 2).float() / dim_model))  # [dim/2]
        # NOTE: if we use self.inv_freq, then it won't be included in the state_dict
        self.register_buffer('inv_freq', inv_freq)

    def apply(self, pos: Tensor, offset: int = 0, **kwargs) -> Tensor:
        """
        Apply fixed sinusoidal positional encoding
        
        Args:
            pos: Position indices
            offset: Offset for positions
        
        Returns:
            Positional embeddings
        """
        pos = pos.type_as(self.inv_freq) + offset  # [seq_len]
        sinusoid_inp = pos.unsqueeze(-1) * self.inv_freq  # [seq_len, 1] * [dim/2] -> [seq_len, dim/2]
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)  # [seq_len, dim]
        return emb


PositionalEncodingConfigFactory.register("absolute")({
    "l2norm_embed": (bool, Field(default=False, description="Whether to L2 normalize embeddings"))
})
@PositionalEncodingFactory.register("absolute")
class AbsolutePositionalEncoding(PositionalEncoding):
    """Learnable absolute positional embeddings"""
    def __init__(self, dim_model: int, l2norm_embed: bool = False, **kwargs):
        super().__init__()
        if VERBOSE: self.__print_init_args__()
        self.l2norm_embed = l2norm_embed
        self.scale = dim_model ** -0.5 if not l2norm_embed else 1.
        self.emb = nn.Embedding(kwargs.get('max_seq_len', 1024), dim_model)

    def apply(self, pos: Tensor, **kwargs) -> Tensor:
        """
        Apply absolute positional encoding
        
        Args:
            pos: Position indices
        
        Returns:
            Positional embeddings of shape matching input
        """
        pos_emb = self.emb(pos)
        pos_emb = pos_emb * self.scale
        return l2norm(pos_emb) if self.l2norm_embed else pos_emb


PositionalEncodingConfigFactory.register("none")({})
@PositionalEncodingFactory.register("none")
class NonePositionalEncoding(PositionalEncoding):
    """No positional encoding - identity function"""
    def __init__(self, **kwargs):
        super().__init__()
        if VERBOSE: self.__print_init_args__()
        
    def apply(self, **kwargs) -> Tensor:
        return None