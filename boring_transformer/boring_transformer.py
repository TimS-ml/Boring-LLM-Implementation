'''
On LayerNorm's location: https://arxiv.org/abs/2002.04745

- https://nn.labml.ai/transformers/models.html
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from einops import rearrange

from torch import Tensor, Size
from typing import Optional, Tuple, Union, List

from boring_nn.attention import MultiHeadAttention
from boring_nn.norm import LayerNorm
from boring_nn.pe import SinusoidalPositionalEncoding, LearnedPositionalEncoding
from boring_nn.ffn import FeedForward
from boring_utils.utils import cprint
from boring_utils.helpers import DEBUG


class AttentionLayers(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, causal=False, **kwargs):
        super(AttentionLayers, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model, d_ff, dropout=dropout)
        self.causal = causal

    def forward(self, x, attn_mask=None, **kwargs):
        if self.causal and attn_mask is None:
            attn_mask = torch.ones((x.shape[1], x.shape[1]), device=x.device, dtype=torch.bool).triu(1)

        attn_output, _ = self.mha(x, x, x, attn_mask=attn_mask)
        attn_output = self.dropout(attn_output)
        x = self.layer_norm(x + attn_output)

        ff_output = self.feed_forward(x)
        ff_output = self.dropout(ff_output)
        x = self.layer_norm(x + ff_output)

        return x


class Encoder(AttentionLayers):
    def __init__(self, **kwargs):
        assert 'causal' not in kwargs, 'cannot set causality on encoder'
        super().__init__(causal=False, **kwargs)


class Decoder(AttentionLayers):
    def __init__(self, **kwargs):
        assert 'causal' not in kwargs, 'cannot set causality on decoder'
        super().__init__(causal=True, **kwargs)


# TODO
# class PrefixDecoder(AttentionLayers):
#     def __init__(self, **kwargs):
#         assert 'causal' not in kwargs, 'cannot set causality on decoder'
#         super().__init__(causal=False, **kwargs)

#     def forward(self, x, attn_mask=None, prefix_attn_len=None, **kwargs):
#         b, n, device = x.shape[0], x.shape[1], x.device
#         causal_mask = torch.ones((n, n), device=device, dtype=torch.bool).triu(1)

#         forwarded_mask = ~causal_mask

#         if prefix_attn_len is not None:
#             if isinstance(prefix_attn_len, int):
#                 prefix_attn_len = torch.full((b,), prefix_attn_len, device=device)

#             prefix_mask = torch.arange(n, device=device) < rearrange(prefix_attn_len, 'b -> b 1 1 1')
#             forwarded_mask = forwarded_mask | prefix_mask

#         if attn_mask is not None:
#             forwarded_mask = forwarded_mask & attn_mask

#         return super().forward(x, attn_mask=forwarded_mask, **kwargs)


class BoringTransformerBlock(nn.Module):
    def __init__(
        self,
        *,
        d_model,
        num_heads,
        d_ff,
        dropout=0.1,
        max_seq_len,
        num_layers=1,
        use_abs_pos_emb=True,
        scaled_sinu_pos_emb=False,
        l2norm_embed=False,
        causal=False,
        **kwargs
    ):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.num_layers = num_layers

        self.max_seq_len = max_seq_len
        self.l2norm_embed = l2norm_embed

        no_abs_pos_emb = max_seq_len == 0 or not use_abs_pos_emb

        if no_abs_pos_emb:
            self.pos_emb = always(0)
        elif scaled_sinu_pos_emb:
            self.pos_emb = ScaledSinusoidalEmbedding(d_model)
        else:
            self.pos_emb = AbsolutePositionalEmbedding(d_model, max_seq_len, l2norm_embed=l2norm_embed)

        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            AttentionLayers(d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=dropout, causal=causal)
            for _ in range(num_layers)
        ])

    def init_(self):
        if self.l2norm_embed:
            if not isinstance(self.pos_emb, always):
                nn.init.normal_(self.pos_emb.emb.weight, std=1e-5)

    def forward(
        self,
        x,
        return_embeddings=False,
        mask=None,
        return_attn=False,
        return_intermediates=False,
        pos=None,
        **kwargs
    ):
        b, n, device = x.shape[0], x.shape[1], x.device

        external_pos_emb = exists(pos) and pos.dtype != torch.long
        pos_emb = self.pos_emb(x, pos=pos) if not external_pos_emb else pos
        x = x + pos_emb

        x = self.dropout(x)

        intermediates = []
        for layer in self.layers:
            x, layer_intermediates = layer(x, mask=mask, return_hiddens=True, **kwargs)
            intermediates.append(layer_intermediates)

        if not return_intermediates:
            return x

        return x, intermediates
