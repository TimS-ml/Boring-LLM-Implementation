import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
# from einops import rearrange

from utils import *
from torch import Tensor
from typing import Optional, Tuple, Union


class LayerNorm1d:
    def __init__(self, dim: int, eps: float = 1e-5, momentum: float = 0.1):
        """
        dim (int): The number of dimensions in the input tensor.
        eps (float, optional): A small value added to the denominator for numerical stability. Defaults to 1e-5.
        momentum (float, optional): The value used for the running mean and variance computation. Defaults to 0.1.
        """
        self.eps = eps
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)

    def __call__(self, x):
        xmean = x.mean(dim=1, keepdim=True)
        xvar = x.var(dim=1, keepdim=True)
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)  # norm to unit variance
        self.out = self.gamma * xhat + self.beta  # scale and shift
        return self.out 
    
    def parameters(self):
        return [self.gamma, self.beta]


if __name__ == '__main__':

    def test_LayerNorm1d():
        module = LayerNorm1d(100)
        x = torch.randn(32, 100)
        x = module(x)
        cprint(x.shape)

    test_LayerNorm1d()
