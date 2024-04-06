'''
Gated: GLU
- https://arxiv.org/abs/2002.05202

- https://nn.labml.ai/transformers/feed_forward.html
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
# from einops import rearrange

from boring_utils.utils import cprint

from torch import Tensor
from typing import Optional, Tuple, Union


class FeedForward(nn.Module):
    """
    d_model is the number of features in a token embedding
    d_ff is the number of features in the hidden layer of the FFN
    is_gated is related to GLU
    bias1/bias2/bias_gate are reltaed to whether the 1st/2nd/gate fc layer have a learnable bias
    """
    def __init__(self, 
                 d_model: int, 
                 d_ff: int,
                 dropout: float = 0.1,
                 activation=nn.ReLU(),
                 is_gated: bool = False,
                 bias1: bool = True,
                 bias2: bool = True,
                 bias_gate: bool = True):
        super().__init__()
        self.ln1 = nn.Linear(d_model, d_ff, bias=bias1)
        self.ln2 = nn.Linear(d_ff, d_model, bias=bias2)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.is_gated = is_gated
        if is_gated:
            # If there is a gate the linear layer to transform inputs to
            # be multiplied by the gate, parameterized by weight $V$ and bias $c$
            self.linear_v = nn.Linear(d_model, d_ff, bias=bias_gate)

    def forward(self, x: Tensor) -> Tensor:
        g = self.activation(self.ln1(x))

        if self.is_gated:
            x = g * self.linear_v(x)
        else:
            x = g

        x = self.dropout(x)
        x = self.ln2(x)
        return x

