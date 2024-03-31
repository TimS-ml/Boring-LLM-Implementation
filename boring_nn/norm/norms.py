'''
- https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
- https://nn.labml.ai/normalization/layer_norm/index.html
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from einops import rearrange

from boring_utils.utils import cprint

from torch import Tensor, Size
from typing import Optional, Tuple, Union, List


class SimpleLayerNorm1d:

    def __init__(self, dim: int, eps: float = 1e-5):
        """
        dim (int): The number of dimensions in the input tensor.
        eps (float, optional): A small value added to the denominator for numerical stability. 
        In this implementation, self.gamma and self.beta are initialized as regular tensors 
        using torch.ones(dim) and torch.zeros(dim), respectively. 
        As regular tensors, they do not automatically get updated during the training process when using gradient descent methods.
        """
        self.eps = eps
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)

    def __call__(self, x: Tensor) -> Tensor:
        # since this is 1d, dim=1
        x_mean = x.mean(dim=1, keepdim=True)
        x_var = x.var(dim=1, keepdim=True)

        # norm to unit variance
        x = (x - x_mean) / torch.sqrt(x_var + self.eps)
        x = self.gamma * x + self.beta  # scale and shift
        return x

    def parameters(self):
        return [self.gamma, self.beta]


class LayerNorm(nn.Module):

    def __init__(self,
                 normalized_shape: Union[int, List[int], Size],
                 elementwise_affine: bool = True,
                 eps: float = 1e-5):
        super().__init__()
        '''
        normalized_shape: The shape of the input tensor that should be normalized. 
        This specifies the shape of the last dimension(s) of the input tensor to be normalized. 
        elementwise_affine: If True, this module has learnable per-element
        affine parameters initialized to ones (for weights) and zeros (for biases). This
        allows the layer to learn an elementwise scale and shift for the normalized output.
        Defaults to True.
        eps (float, optional): A value added to the denominator for numerical stability.
        
        gamma (Tensor): The learnable weights of the module of shape equal to
        normalized_shape if elementwise_affine is True.
        beta (Tensor): The learnable bias of the module of shape equal to
        normalized_shape if elementwise_affine is True.
        '''
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
        # assert self.normalized_shape == x.shape[-len(self.normalized_shape):]
        dims = [-(i + 1) for i in range(len(self.normalized_shape))]
        mean = x.mean(dim=dims, keepdim=True)
        # std = x.std(dim=dims, keepdim=True, unbiased=False)
        var = x.var(dim=dims, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        if self.elementwise_affine:
            x = x * self.gamma + self.beta
        return x


class BatchNorm2d(nn.Module):

    def __init__(self,
                 channels: int,
                 eps: float = 1e-5,
                 momentum: float = 0.1,
                 affine: bool = True,
                 track_running_stats: bool = True):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.gamma = nn.Parameter(torch.ones(channels))
            self.beta = nn.Parameter(torch.zeros(channels))
        if self.track_running_stats:
            self.register_buffer('exp_mean', torch.zeros(channels))
            self.register_buffer('exp_var', torch.ones(channels))

    def forward(self, x: Tensor) -> Tensor:
        x_shape = x.shape
        batch_size = x_shape[0]
        assert self.channels == x.shape[1]
        x = x.view(batch_size, self.channels, -1)

        if self.training or not self.track_running_stats:
            mean = x.mean(dim=[0, 2])
            mean_x2 = (x**2).mean(dim=[0, 2])
            var = mean_x2 - mean**2
            if self.training and self.track_running_stats:
                self.exp_mean = (
                    1 - self.momentum) * self.exp_mean + self.momentum * mean
                self.exp_var = (
                    1 - self.momentum) * self.exp_var + self.momentum * var
        else:
            mean = self.exp_mean
            var = self.exp_var

        x_norm = (x - mean.view(1, -1, 1)) / torch.sqrt(var + self.eps)
        x_norm = x_norm.view(1, -1, 1)

        if self.affine:
            x_norm = self.gamma.view(1, -1, 1) * x_norm + \
                        self.beta.view(1, -1, 1)
        return x_norm.view(x_shape)

