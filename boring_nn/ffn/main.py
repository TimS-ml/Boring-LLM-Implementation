import torch
import torch.nn as nn
import torch.nn.functional as F
# from einops import rearrange

from torch import Tensor
from typing import Optional, Tuple, Union, List

from boring_nn.ffn.core import FeedForwardConfig, ActivationType
from boring_utils.utils import cprint
from boring_utils.helpers import DEBUG


class GLU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, activation_type: ActivationType, use_bias: bool = True):
        super().__init__()
        self.act = BoringFeedForward.get_activation(activation_type)
        self.proj = nn.Linear(dim_in, dim_out * 2, bias=use_bias)

    def forward(self, x: Tensor) -> Tensor:
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * self.act(gate)


class ReluSquared(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.relu(x)**2


class BoringFeedForward(nn.Module):
    def __init__(self, config: FeedForwardConfig):
        super().__init__()
        dim = config.d_model
        dim_out = config.ffn_dim_out or dim
        mult = config.mult
        inner_dim = int(dim * mult)

        activation_type = config.activation.type
        use_glu = config.activation.use_glu

        # project_in is the first layer of the FFN
        if use_glu:
            self.glu = GLU(dim, inner_dim, activation_type, use_bias=not config.no_bias)
            project_in = self.glu
        else:
            activation = self.get_activation(activation_type)
            project_in = nn.Sequential(
                nn.Linear(dim, inner_dim, bias=not config.no_bias),
                activation
            )

        self.net = nn.Sequential(
            project_in,
            nn.LayerNorm(inner_dim) if config.post_act_ln else nn.Identity(),
            nn.Dropout(config.dropout),
            nn.Linear(inner_dim, dim_out, bias=not config.no_bias)
        )

        if config.zero_init_output:
            nn.init.zeros_(self.net[-1].weight)

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
        return self.net(x)
