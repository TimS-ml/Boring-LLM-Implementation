import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ReluSquared(nn.Module):
    """ReLU squared activation function: f(x) = ReLU(x)^2"""
    def __init__(self):
        super().__init__()
    
    def forward(self, x: Tensor) -> Tensor:
        return F.relu(x) ** 2
