'''
- https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
- https://nn.labml.ai/normalization/layer_norm/index.html
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from einops import rearrange

import sys; from pathlib import Path; sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import *
from utils import *
from torch import Tensor, Size
from typing import Optional, Tuple, Union, List


class SimpleLayerNorm1d:

    def __init__(self, dim: int, eps: float = 1e-5):
        """
        dim (int): The number of dimensions in the input tensor.
        eps (float, optional): A small value added to the denominator for numerical stability. Defaults to 1e-5.
        momentum (float, optional): The value used for the running mean and variance computation. Defaults to 0.1.
        """
        self.eps = eps
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)

    def __call__(self, x: Tensor) -> Tensor:
        # since this is 1d, dim=1
        x_mean = x.mean(dim=1, keepdim=True)
        x_var = x.var(dim=1, keepdim=True)
        x_hat = (x - x_mean) / torch.sqrt(
            x_var + self.eps)  # norm to unit variance
        self.out = self.gamma * x_hat + self.beta  # scale and shift
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]


class LayerNorm(nn.Module):

    def __init__(self,
                 normalized_shape: Union[int, List[int], Size],
                 elementwise_affine: bool = True,
                 eps: float = 1e-5):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = torch.Size([normalized_shape])
        elif isinstance(normalized_shape, list):
            normalized_shape = torch.Size(normalized_shape)
        assert isinstance(normalized_shape, torch.Size)

        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.gamma = nn.Parameter(torch.ones(normalized_shape))
            self.beta = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x: Tensor) -> Tensor:
        # x = rearrange(x, '... d -> ... d')
        dims = [-(i + 1) for i in range(len(self.normalized_shape))]
        mean = x.mean(dim=dims, keepdim=True)
        # std = x.std(dim=dims, keepdim=True, unbiased=False)
        var = x.var(dim=dims, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        if self.elementwise_affine:
            x = x * self.gamma + self.beta
        return x


if __name__ == '__main__':

    # def test_LayerNorm1d():
    #     module = SimpleLayerNorm1d(100)
    #     x = torch.randn(32, 100)
    #     x = module(x)
    #     cprint(x.shape)

    def test_LayerNorm():
        # NLP Example
        batch, sentence_length, embedding_dim = 20, 5, 10
        embedding = torch.randn(batch, sentence_length, embedding_dim)
        # layer_norm = nn.LayerNorm(embedding_dim)
        layer_norm = LayerNorm(embedding_dim)

        # Activate module
        layer_norm(embedding)
        # Image Example
        N, C, H, W = 20, 5, 10, 10
        input = torch.randn(N, C, H, W)
        # Normalize over the last three dimensions (i.e. the channel and spatial dimensions)
        # as shown in the image below
        # layer_norm = nn.LayerNorm([C, H, W])
        layer_norm = LayerNorm([C, H, W])
        output = layer_norm(input)
        cprint(output.shape)

    # test_LayerNorm1d()
    test_LayerNorm()
