import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange
from typing import Optional
from boring_llm.nn.norm.core import l2norm


class FixedPositionalEmbedding(nn.Module):
    """
    Sinusoidal positional embeddings from the "Attention Is All You Need" paper
    """
    def __init__(self, dim: int):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
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
        if pos is None:
            pos = torch.arange(x.shape[seq_dim], device=x.device)

        pos = pos.type_as(self.inv_freq) + offset
        sinusoid_inp = pos.unsqueeze(-1) * self.inv_freq
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        return emb


class AbsolutePositionalEmbedding(nn.Module):
    """
    Learnable absolute positional embeddings
    """
    def __init__(self, dim: int, max_seq_len: int, l2norm_embed: bool = False):
        super().__init__()
        self.scale = dim ** -0.5 if not l2norm_embed else 1.
        self.max_seq_len = max_seq_len
        self.l2norm_embed = l2norm_embed
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

        if pos is None:
            pos = torch.arange(seq_len, device=device)

        if seq_start_pos is not None:
            pos = (pos - seq_start_pos[..., None]).clamp(min=0)

        pos_emb = self.emb(pos)
        pos_emb = pos_emb * self.scale
        return l2norm(pos_emb) if self.l2norm_embed else pos_emb