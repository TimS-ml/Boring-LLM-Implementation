import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ReluSquared(nn.Module):
    """ReLU squared activation function: f(x) = ReLU(x)^2"""
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return F.relu(x) ** 2


class SoLU(nn.Module):
    """
    Softmax Linear Unit (SoLU) activation.

    Proposed by Anthropic, combines softmax gating with layer normalization.
    f(x) = softmax(x) * x, then normalized.

    Args:
        dim: Feature dimension for layer normalization
    """
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        # Softmax gating: each feature is weighted by its softmax probability
        activated = x.softmax(dim=-1) * x
        return self.norm(activated)
