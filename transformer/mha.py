'''
Multi-head attention
- paper: Attention Is All You Need http://arxiv.org/abs/1706.03762v7
  - sec 3.2
- https://nn.labml.ai/transformers/mha.html
- https://github.com/sooftware/attentions/blob/master/attentions.py
- https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
'''

import math
import numpy as np
import torch 
import torch.nn as nn

import sys
sys.path.append('..')
from utils.utils import cprint, cprint_str

from typing import Optional, Tuple



class ScaledDotProductAttention(nn.Module):
    '''
    Scaled Dot-Product Attention
    [batch_size, seq_len, d_model] or [batch_size, d_model]
    '''
    def __init__(self, d_k: int = 0):
        super().__init__()
        self.d_k = d_k  # scaling factor
    
    def forward(self, query, key, value, mask=None):
        if self.d_k == 0:
            scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(key.size(-1))
        else:
            scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        if mask is not None:
            # broadcasted to the shape of scores
            scores.masked_fill_(mask.view(scores.size()), -float('Inf'))
        
        attention = torch.softmax(scores, dim=-1)
        context = torch.matmul(attention, value)
        return context, attention


if __name__ == '__main__':
    # Generating random inputs
    B, T, C = 4, 8, 32  # batch size, time steps (seq length), channels (d_model)
    x = torch.rand(B, T, C)
    
    tril = torch.tril(torch.ones(T, T))
    
    # single Head perform self-attention
    # head_size = 16
    head_size = 1
    key = nn.Linear(C, head_size, bias=False)
    query = nn.Linear(C, head_size, bias=False)
    value = nn.Linear(C, head_size, bias=False)
    
    k, q, v = key(x), query(x), value(x)

    # Calling the attention function
    att = ScaledDotProductAttention()
    output, attn_weights = att(q, k, v)
    cprint(attn_weights)
