import torch
import torch.nn as nn
import torch.nn.functional as F
# from einops import rearrange

from torch import Tensor
from typing import Optional, Tuple, Union, List

from nn.ffn.core import FeedForwardConfig, ActivationType
from boring_utils.utils import cprint
from boring_utils.helpers import DEBUG


class GLU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, config: FeedForwardConfig):
        super().__init__()
        activation_type = config.activation.type
        no_bias = config.no_bias
        self.act = BoringFeedForward.get_activation(activation_type)
        self.proj = nn.Linear(dim_in, dim_out * 2, bias=not no_bias)
        self.mult_bias = nn.Parameter(torch.ones(dim_out)) if config.activation.mult_bias else 1.

    def forward(self, x: Tensor) -> Tensor:
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * self.act(gate) * self.mult_bias


class ReluSquared(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.relu(x)**2


class BoringFeedForward(nn.Module):
    def __init__(self, config: FeedForwardConfig):
        super().__init__()
        dim_in = config.d_model
        dim_out = config.ffn_dim_out or dim_in
        mult_dim = config.mult_dim
        inner_dim = int(dim_in * mult_dim)

        activation_type = config.activation.type
        use_glu = config.activation.use_glu

        # project_in is the first layer of the FFN
        if use_glu:
            project_in = GLU(dim_in, inner_dim, config)
        else:
            project_in = nn.Sequential(
                nn.Linear(dim_in, inner_dim, bias=not config.no_bias),
                self.get_activation(activation_type)
            )

        self.feedforward = nn.Sequential(
            project_in,
            nn.LayerNorm(inner_dim) if config.post_act_ln else nn.Identity(),
            nn.Dropout(config.dropout),
            nn.Linear(inner_dim, dim_out, bias=not config.no_bias)
        )

        if config.zero_init_output:
            nn.init.zeros_(self.feedforward[-1].weight)

        self.config = config

    @staticmethod
    def get_activation(activation_type: ActivationType):
        if activation_type == ActivationType.RELU:
            return nn.ReLU()
        elif activation_type == ActivationType.GELU:
            return nn.GELU()
        elif activation_type == ActivationType.SWISH:
            return nn.SiLU()
        elif activation_type == ActivationType.RELU_SQUARED:
            # return lambda x: F.relu(x)**2  # causes an error for not being a nn.Module
            return ReluSquared()
        else:
            raise ValueError(f"Unsupported activation type: {activation_type}")

    def forward(self, x: Tensor) -> Tensor:
        return self.feedforward(x)
