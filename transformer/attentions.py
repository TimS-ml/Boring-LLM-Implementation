'''
Multi-head attention
- paper: Attention Is All You Need http://arxiv.org/abs/1706.03762v7
  - sec 3.2

- https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
- https://nn.labml.ai/transformers/mha.html
- https://github.com/sooftware/attentions/blob/master/attentions.py
- https://github.com/xmu-xiaoma666/External-Attention-pytorch/blob/master/model/attention/SelfAttention.py
'''

import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
# from einops import rearrange

from utils import *
from torch import Tensor
from typing import Optional, Tuple, Union


class ScaledDotProductAttention(nn.Module):
    '''
    Scaled Dot-Product Attention

    input:
    [batch_size, seq_len, d_model] or [batch_size, d_model]

    score shape:
    [batch_size, seq_len, seq_len] or [batch_size, seq_len]
    '''
    def __init__(self, d_k: int = 0):
        super().__init__()
        self.d_k = d_k  # scaling factor
    
    def forward(
            self, query: Tensor, key: Tensor, value: Tensor, 
            mask: Optional[Tensor] = None, dropout_p: Optional[float] = None
        ) -> Tuple[Tensor, Tensor]:
        if self.d_k == 0:
            scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(key.size(-1))
        else:
            scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(self.d_k)
        # cprint(scores.shape)

        if mask is not None:
            # broadcasted to the shape of scores
            scores = scores.masked_fill(mask == 0, float("-inf"))
        
        if dropout_p is not None:
            dropout = nn.Dropout(p=dropout_p)
            scores = dropout(scores)

        attention = torch.softmax(scores, dim=-1)

        cprint(attention.shape)
        cprint(value.shape)
        context = torch.matmul(attention, value)
        return context, attention


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
    def __init__(self, d_model: int = 512, num_heads: int = 8):
        super().__init__()

        assert d_model % num_heads == 0, "d_model % num_heads should be zero."

        self.num_heads = num_heads
        self.d_head = int(d_model / num_heads)
        self.scaled_dot_attn = ScaledDotProductAttention(self.d_head)

        # usually d_model == d_head * num_heads, some times it's also written as nn.Linear(d_model, d_model)
        self.query_proj = nn.Linear(d_model, self.d_head * num_heads)
        self.key_proj = nn.Linear(d_model, self.d_head * num_heads)
        self.value_proj = nn.Linear(d_model, self.d_head * num_heads)

    def forward(
            self, query: Tensor, key: Tensor, value: Tensor, 
            mask: Optional[Tensor] = None, dropout_p: Optional[float] = None
        ) -> Tuple[Tensor, Tensor]:
        batch_size = query.size(0)

        # split d_model (the last dimention) into num_heads * d_head
        # batch_size, seq_len, num_heads, d_head
        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.d_head)
        key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.d_head)
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.d_head)

        # permute the dimensions into batch_size * num_heads, seq_len, d_head
        # because in ScaledDotProductAttention we only care about the last two dims
        query = query.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.d_head)
        key = key.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.d_head)
        value = value.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.d_head)

        # the Einops way
        # query = self.query_proj(query)
        # key = self.key_proj(key)
        # value = self.value_proj(value)

        # # Reshape and permute the dimensions using Einops rearrange
        # query = rearrange(query, 'b q (n d) -> (b n) q d', n=self.num_heads)
        # key = rearrange(key, 'b k (n d) -> (b n) k d', n=self.num_heads)
        # value = rearrange(value, 'b v (n d) -> (b n) v d', n=self.num_heads)
    
        if mask is not None:
            # batch_size, seq_len (Q), seq_len (K) -> batch_size, 1, seq_len (Q), seq_len (K)
            mask = mask.unsqueeze(1)

            # repeat operation duplicates the tensor along specified dimensions
            # batch_size, num_heads, seq_len (Q), seq_len (K)
            mask = mask.repeat(1, self.num_heads, 1, 1)
    
        context, attn = self.scaled_dot_attn(query, key, value, mask, dropout_p)
    
        # Reshape the context tensor back to the original form
        # unpack batch_size * num_heads, seq_len, d_head -> batch_size, seq_len, d_model
        context = context.view(self.num_heads, batch_size, -1, self.d_head)
        context = context.permute(1, 2, 0, 3).contiguous().view(batch_size, -1, self.num_heads * self.d_head)

        # the Einops way
        # context = rearrange(context, '(b n) v d -> b v (n d)', b=batch_size, n=self.num_heads)
    
        return context, attn



if __name__ == '__main__':
    # Generating random inputs
    B, T, C = 4, 8, 32  # batch size, time steps (seq length), channels
    x = torch.rand(B, T, C)
    tril = torch.tril(torch.ones(T, T))  # for mask

    def test_scaled_dot_product_attention():
        # head_size = 1  # (B, T, C) -> (B, T, head_size)
        # query = nn.Linear(C, head_size, bias=False)
        # key = nn.Linear(C, head_size, bias=False)
        # value = nn.Linear(C, head_size, bias=False)
        # q, k, v = query(x), key(x), value(x)

        # cprint(x.shape)
        # cprint(q.shape)

        q = torch.randn(T, C)
        k = torch.randn(T, C)
        v = torch.randn(T, C)

        # Calling the attention function
        att = ScaledDotProductAttention()
        output, attn_weights = att(q, k, v, mask=tril)

        cprint(output.shape)
        cprint(attn_weights.shape)
        cprint(attn_weights)

    def test_multihead_attention():
        head_size = 8  # (B, T, C) -> (B, T, head_size)

        # Calling the attention function
        att = MultiHeadAttention(d_model=C, num_heads=head_size)
        output, attn_weights = att(x, x, x)

        cprint(output.shape)
        cprint(attn_weights.shape)
        cprint(attn_weights)

    test_scaled_dot_product_attention()
    # test_multihead_attention()
