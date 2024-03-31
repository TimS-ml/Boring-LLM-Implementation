'''
Multi-head attention
- paper: Attention Is All You Need http://arxiv.org/abs/1706.03762v7
  - sec 3.2

- https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
- https://nn.labml.ai/transformers/mha.html
- https://github.com/sooftware/attentions/blob/master/attentions.py
- https://github.com/xmu-xiaoma666/External-Attention-pytorch/blob/master/model/attention/SelfAttention.py

Note:
- In the Transformer architecture, the dimensionality of keys, queries, and values is typically the same, so...
- The attention mask is used to selectively ignore or pay less attention to certain elements 
    in the input sequences during the attention calculation.
- The attention weights in these attention functions have a shape of [batch_size, seq_len (query), seq_len (key)]
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from einops import rearrange

from boring_nn.utils import cprint

from torch import Tensor
from typing import Optional, Tuple, Union


def SimpleScaledDotProductAttention(query: Tensor,
                                    key: Tensor,
                                    value: Tensor,
                                    dropout: Optional[float] = None,
                                    attn_mask: Optional[Tensor] = None,
                                    is_causal: bool = False,
                                    d_k: int = 0) -> Tensor:
    '''
    Scaled Dot-Product Attention

    input:
      [batch_size, seq_len, d_model]

    d_k is related to the scaling factor, which is the square root of the dimension of the key vectors

    Casual Masking (is_causal=True):
      Creating a lower triangular matrix (torch.ones(L, S).tril(diagonal=0)) 
      where only the current and past positions are marked as True (to be attended to), and future positions are masked out
    
    Boolean Masking (attn_mask.dtype == torch.bool):
      fills positions in attn_bias with -inf where attn_mask is False
      The logical negation .logical_not() is applied to attn_mask, so positions that should be ignored (originally False) are marked with -inf. 
      This is because adding -inf to those positions before the softmax operation effectively zeros out those positions in the attention weights, 
      ensuring that no attention is paid to masked positions.

    Additive Masking (attn_mask.dtype != torch.bool):
      use -inf to mask out positions in attn_bias where attn_mask is not zero
    '''
    # calculate the scaling factor
    if d_k == 0:
        scale_factor = 1 / np.sqrt(key.size(-1))
    else:
        scale_factor = 1 / np.sqrt(d_k)

    # match the shape of query @ key.transpose(-2, -1)
    query_len, key_len = query.size(-2), key.size(-2)
    attn_bias = torch.zeros(query_len, key_len, dtype=query.dtype)

    # Generate a lower triangular matrix for causal masking
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(query_len, key_len,
                               dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = F.softmax(attn_weight, dim=-1)
    if dropout is not None:
        attn_weight = F.dropout(attn_weight, dropout, training=True)
    return attn_weight @ value, attn_weight


class ScaledDotProductAttention(nn.Module):
    '''
    Scaled Dot-Product Attention, does not contain learnable parameters.

    input:
      [batch_size, seq_len, d_model] or [batch_size, d_model]

    score shape:
      [batch_size, seq_len, seq_len] or [batch_size, seq_len]

    d_k is related to the scaling factor, which is the square root of the dimension of the key vectors
    '''

    def __init__(self, d_k: int = 0, dropout: Optional[float] = None):
        super().__init__()
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        self.d_k = d_k

    def forward(self,
                query: Tensor,
                key: Tensor,
                value: Tensor,
                attn_mask: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:

        # calculate the scaling factor
        if self.d_k == 0:
            scale_factor = 1 / np.sqrt(key.size(-1))
        else:
            scale_factor = 1 / np.sqrt(self.d_k)

        scores = torch.matmul(query, key.transpose(-2, -1)) * scale_factor
        # cprint(scores.shape)

        if attn_mask is not None:
            # broadcasted to the shape of scores
            scores = scores.masked_fill(attn_mask == 0, float("-inf"))

        if self.dropout is not None:
            scores = self.dropout(scores)

        attn_weight = torch.softmax(scores, dim=-1)

        # cprint(attn_weight.shape)
        # cprint(value.shape)
        context = torch.matmul(attn_weight, value)
        # return context, attn_weight
        return context, attn_weight


class MultiHeadAttention(nn.Module):
    '''
    Multi-Head Attention
    "we found it beneficial to linearly project the 
    queries, keys and values h times with different, learned linear projections 
    to dk, dk and dv dimensions, respectively."

    head_i = Attention(Q*W_q, K*W_k, V*W_v)

    input:
      [batch_size, seq_len, d_model]
    '''

    def __init__(self,
                 d_model: int = 512,
                 num_heads: int = 8,
                 dropout: Optional[float] = None):
        super().__init__()
        # if dropout is not None:
        #     self.dropout = nn.Dropout(dropout)
        # else:
        #     self.dropout = None

        assert d_model % num_heads == 0, "d_model % num_heads should be zero."

        self.num_heads = num_heads
        self.d_head = int(d_model / num_heads)
        self.scaled_dot_attn = ScaledDotProductAttention(self.d_head, dropout)

        # usually d_model == d_head * num_heads, some times it's also written as nn.Linear(d_model, d_model)
        self.query_proj = nn.Linear(d_model, self.d_head * num_heads)
        self.key_proj = nn.Linear(d_model, self.d_head * num_heads)
        self.value_proj = nn.Linear(d_model, self.d_head * num_heads)

    def forward(self,
                query: Tensor,
                key: Tensor,
                value: Tensor,
                attn_mask: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:

        # split d_model (the last dimention) into num_heads * d_head
        # batch_size, seq_len, num_heads, d_head
        batch_size = query.size(0)
        seq_len = query.size(1)
        query = self.query_proj(query)
        query = query.view(batch_size, -1, self.num_heads, self.d_head)
        key = self.key_proj(key)
        key = key.view(batch_size, -1, self.num_heads, self.d_head)
        value = self.value_proj(value)
        value = value.view(batch_size, -1, self.num_heads, self.d_head)

        # permute the dimensions into batch_size * num_heads, seq_len, d_head
        # because in ScaledDotProductAttention we only care about the last two dims
        query = query.permute(2, 0, 1, 3).contiguous()
        query = query.view(batch_size * self.num_heads, -1, self.d_head)
        key = key.permute(2, 0, 1, 3).contiguous()
        key = key.view(batch_size * self.num_heads, -1, self.d_head)
        value = value.permute(2, 0, 1, 3).contiguous()
        value = value.view(batch_size * self.num_heads, -1, self.d_head)

        # the Einops way
        # query = self.query_proj(query)
        # key = self.key_proj(key)
        # value = self.value_proj(value)

        # # Reshape and permute the dimensions using Einops rearrange
        # query = rearrange(query, 'b q (n d) -> (b n) q d', n=self.num_heads)
        # key = rearrange(key, 'b k (n d) -> (b n) k d', n=self.num_heads)
        # value = rearrange(value, 'b v (n d) -> (b n) v d', n=self.num_heads)

        if attn_mask is not None:
            # batch_size, seq_len (Q), seq_len (K) -> batch_size, 1, seq_len (Q), seq_len (K)
            attn_mask = attn_mask.unsqueeze(1)

            # repeat operation duplicates the tensor along specified dimensions
            # batch_size, num_heads, seq_len (Q), seq_len (K)
            attn_mask = attn_mask.repeat(1, self.num_heads, 1, 1)

        context, attn_weights = self.scaled_dot_attn(query, key, value, attn_mask)

        # Reshape the context tensor back to the original form
        # unpack batch_size * num_heads, seq_len, d_head -> batch_size, seq_len, d_model
        context = context.view(self.num_heads, batch_size, -1, self.d_head)
        context = context.permute(1, 2, 0, 3).contiguous().view(
            batch_size, -1, self.num_heads * self.d_head)

        # the Einops way
        # context = rearrange(context, '(b n) v d -> b v (n d)', b=batch_size, n=self.num_heads)

        attn_weights = attn_weights.view(batch_size * self.num_heads, seq_len, -1)
        attn_weights = attn_weights.view(batch_size, self.num_heads, seq_len, -1)
        return context, attn_weights

