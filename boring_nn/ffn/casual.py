'''
Gated: GLU
- https://arxiv.org/abs/2002.05202

- https://nn.labml.ai/transformers/feed_forward.html
'''
from pydantic import BaseModel, Field
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
# from einops import rearrange

from torch import Tensor
from typing import Optional, Tuple, Union

from boring_utils.utils import cprint
from boring_utils.helpers import DEBUG


class FeedForward(nn.Module):
    """
    d_model is the number of features in a token embedding
    d_ff is the number of features in the hidden layer of the FFN
    is_gated is related to GLU
    bias1/bias2/bias_gate are reltaed to whether the 1st/2nd/gate fc layer have a learnable bias
    """
    def __init__(self, 
                 d_model: int, 
                 ffn_dim: int,
                 dropout: float = 0.1,
                 activation=nn.ReLU(),
                 is_gated: bool = False,
                 bias1: bool = True,
                 bias2: bool = True,
                 bias_gate: bool = True):
        super().__init__()
        self.linear1 = nn.Linear(d_model, ffn_dim, bias=bias1)
        self.linear2 = nn.Linear(ffn_dim, d_model, bias=bias2)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.is_gated = is_gated
        if is_gated:
            # If there is a gate the linear layer to transform inputs to
            # be multiplied by the gate, parameterized by weight $V$ and bias $c$
            self.linear_v = nn.Linear(d_model, ffn_dim, bias=bias_gate)

        if DEBUG >= 1:
            # print('=' * 10 + 'FFN' + '=' * 10)
            cprint(self.is_gated, self.activation)

    def forward(self, x: Tensor) -> Tensor:
        g = self.activation(self.linear1(x))

        if self.is_gated:
            x = g * self.linear_v(x)
        else:
            x = g

        x = self.dropout(x)
        x = self.linear2(x)
        return x

