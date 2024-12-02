'''
Multi-head attention
- paper: Attention Is All You Need http://arxiv.org/abs/1706.03762v7
  - sec 3.2

- https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
- https://pytorch.org/docs/master/_modules/torch/nn/modules/activation.html#MultiheadAttention
  - "use_separate_proj_weight=True" when different q k v
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

from boring_utils.utils import cprint
from boring_utils.helpers import DEBUG

from torch import Tensor
from typing import Optional, Tuple, Union, List
from jaxtyping import Float, Bool


def SimpleScaledDotProductAttention(
    query: Float[Tensor, "batch seq_q dim"],
    key: Float[Tensor, "batch seq_k dim"],
    value: Float[Tensor, "batch seq_k dim"],
    dropout: Optional[float] = None,
    attn_mask: Optional[Union[Bool[Tensor, "seq_q seq_k"], Float[Tensor, "seq_q seq_k"]]] = None,
    is_causal: bool = False,
    d_k: int = 0
) -> Tuple[Float[Tensor, "batch seq_q dim"], Float[Tensor, "batch seq_q seq_k"]]:
    '''
    Scaled Dot-Product Attention

    input:
      [batch_size, seq_len, d_model]

    d_k is related to the scaling factor, which is the square root of the dimension of the key vectors

    NOTE: ScaledDotProductAttention assumes that its input Q, K, V are already in the appropriate attention space!

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
    if DEBUG >= 1:
        # print('=' * 10 + 'ScaledDotProductAttention' + '=' * 10)
        cprint(query.shape, key.shape, attn_bias.shape)

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
      [batch_size, seq_len, d_model]

    score shape:
      [batch_size, seq_len, seq_len]

    d_k is related to the scaling factor, which is the square root of the dimension of the key vectors
    '''

    def __init__(self, d_k: int = 0, dropout: Optional[float] = None):
        super().__init__()
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        self.d_k = d_k

    def forward(
        self,
        query: Float[Tensor, "batch seq_q dim"],
        key: Float[Tensor, "batch seq_k dim"],
        value: Float[Tensor, "batch seq_k dim"],
        attn_mask: Optional[Union[Bool[Tensor, "seq_q seq_k"], Float[Tensor, "seq_q seq_k"]]] = None
    ) -> Tuple[Float[Tensor, "batch seq_q dim"], Float[Tensor, "batch seq_q seq_k"]]:

        # calculate the scaling factor
        if self.d_k == 0:
            scale_factor = 1 / np.sqrt(key.size(-1))
        else:
            scale_factor = 1 / np.sqrt(self.d_k)

        # match the shape of query @ key.transpose(-2, -1)
        attn_weight = query @ key.transpose(-2, -1) * scale_factor

        if attn_mask is not None:
            # print('=' * 10 + 'ScaledDotProductAttention' + '=' * 10)
            if DEBUG >= 1:
                cprint(query.shape)
                cprint(key.shape)
                cprint(attn_mask.shape)
            if attn_mask.dtype == torch.bool:
                attn_weight.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                attn_weight += attn_mask

        if DEBUG >= 1:
            cprint(attn_weight.shape)
        attn_weight = F.softmax(attn_weight, dim=-1)
        if self.dropout is not None:
            attn_weight = self.dropout(attn_weight)
        return attn_weight @ value, attn_weight


# need d_k?
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

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        dropout: Optional[float] = None,
        bias: Optional[bool] = False
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model % num_heads should be zero."

        self.num_heads = num_heads
        self.d_head = int(d_model / num_heads)
        self.scaled_dot_attn = ScaledDotProductAttention(self.d_head, dropout)

        # usually d_model == d_head * num_heads, some times it's also written as nn.Linear(d_model, d_model)
        self.query_proj = nn.Linear(d_model, self.d_head * num_heads, bias=bias)
        self.key_proj = nn.Linear(d_model, self.d_head * num_heads, bias=bias)
        self.value_proj = nn.Linear(d_model, self.d_head * num_heads, bias=bias)

        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
    
    def transpose_qkv(
        self,
        x: Float[Tensor, "batch seq_len dim_model"]
    ) -> Float[Tensor, "batch num_heads seq_len d_head"]:
        '''
        input: 
          [batch_size, seq_len, d_model]
        output:
          [batch_size * num_heads, seq_len, d_head], d_head = int(d_model / num_heads)

        because in ScaledDotProductAttention we only care about the last two dims

        # the Einops way
        # Reshape and permute the dimensions using Einops rearrange
        query = rearrange(query, 'b q (n d) -> (b n) q d', n=self.num_heads)
        key = rearrange(key, 'b k (n d) -> (b n) k d', n=self.num_heads)
        value = rearrange(value, 'b v (n d) -> (b n) v d', n=self.num_heads)
        '''
        x = x.reshape(x.shape[0], x.shape[1], self.num_heads, self.d_head)
        x = x.permute(0, 2, 1, 3)
        return x.reshape(-1, x.shape[2], x.shape[3])

    def transpose_output(
        self,
        x: Float[Tensor, "batch num_heads seq_len d_head"]
    ) -> Float[Tensor, "batch seq_len dim"]:
        '''
        reverse transpose_qkv
        '''
        x = x.reshape(-1, self.num_heads, x.shape[1], x.shape[2])
        x = x.permute(0, 2, 1, 3)
        return x.reshape(x.shape[0], x.shape[1], -1)

    def prepare_mask(
        self,
        attn_mask: Optional[Union[Bool[Tensor, "seq_q seq_k"], Float[Tensor, "seq_q seq_k"]]],
        query_len: int,
        key_len: int,
        batch_size: int
    ) -> Optional[Union[Bool[Tensor, "batch num_heads seq_q seq_k"], Float[Tensor, "batch num_heads seq_q seq_k"]]]:
        if attn_mask is None:
            return None

        if attn_mask.dim() == 2:
            correct_2d_size = (query_len, key_len)
            if attn_mask.shape != correct_2d_size:
                raise RuntimeError(f"The shape of the 2D attention mask is {attn_mask.shape}, but should be {correct_2d_size}.")
            attn_mask = attn_mask.unsqueeze(0)
        elif attn_mask.dim() == 3:
            correct_3d_size = (attn_mask.size(0), query_len, key_len)
            if attn_mask.shape != correct_3d_size:
                raise RuntimeError(f"The shape of the 3D attention mask is {attn_mask.shape}, but should be {correct_3d_size}.")
        else:
            raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")

        # expand the attention mask to match the shape of the attention weights
        # attn_mask = attn_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        # attn_mask = attn_mask.reshape(-1, query_len, key_len)
        attn_mask = attn_mask.unsqueeze(1).expand(batch_size, self.num_heads, -1, -1)
        attn_mask = attn_mask.reshape(batch_size * self.num_heads, query_len, key_len)

        return attn_mask

    def forward(
        self,
        query: Float[Tensor, "batch seq_q dim_model"],
        key: Float[Tensor, "batch seq_k dim_model"],
        value: Float[Tensor, "batch seq_k dim_model"],
        attn_mask: Optional[Union[Bool[Tensor, "seq_q seq_k"], Float[Tensor, "seq_q seq_k"]]] = None
    ) -> Tuple[Float[Tensor, "batch seq_q dim_model"], Float[Tensor, "batch num_heads seq_q seq_k"]]:

        # split d_model (the last dimention) into num_heads * d_head
        # batch_size, seq_len, num_heads, d_head
        batch_size = query.size(0)
        query_len, key_len = query.size(1), key.size(1)

        if DEBUG >= 1:
            # print('=' * 10 + 'MHA' + '=' * 10)
            cprint(query.shape)
            cprint(key.shape)
            if attn_mask is not None:
                cprint(attn_mask.shape)

        query = self.query_proj(query)
        query = self.transpose_qkv(query)
        key = self.key_proj(key)
        key = self.transpose_qkv(key)
        value = self.value_proj(value)
        value = self.transpose_qkv(value)

        attn_mask = self.prepare_mask(attn_mask, query_len, key_len, batch_size)

        context, attn_weights = self.scaled_dot_attn(query, key, value, attn_mask)

        context = self.transpose_output(context)
        attn_weights = attn_weights.view(batch_size, self.num_heads, query_len, key_len)

        return context, attn_weights


