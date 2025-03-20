import torch.nn.functional as F
from torch import Tensor
from einops import rearrange


def l2norm(t: Tensor, groups: int = 1) -> Tensor:
    """Apply L2 normalization across groups of features"""
    t = rearrange(t, '... (g d) -> ... g d', g=groups)
    t = F.normalize(t, p=2, dim=-1)
    return rearrange(t, '... g d -> ... (g d)')