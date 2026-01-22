from typing import Optional
from pydantic import Field
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange

from boring_llm.base.component_registry import ComponentTransform, ComponentRegistry


class NormTransform(ComponentTransform):
    """Base class for normalization transformations"""

    def apply(self, x: Tensor, **kwargs) -> Tensor:
        """Apply normalization to input tensor"""
        raise NotImplementedError


norm_registry = ComponentRegistry[NormTransform]("Normalization")


@norm_registry.register("layernorm", {
    "unit_offset": (bool, Field(default=False, description="Use unit offset for better weight decay compatibility"))
})
class LayerNormTransform(NormTransform):
    """
    Layer Normalization without bias.

    Bias-less layernorm has been shown to be more stable. Most newer models
    have moved towards RMSNorm, which is also bias-less.
    """

    def __init__(self, dim_model: int, unit_offset: bool = False, **kwargs):
        super().__init__()
        self.unit_offset = unit_offset

        # Use LayerNorm without learnable affine parameters
        self.ln = nn.LayerNorm(dim_model, elementwise_affine=False)
        self.gamma = nn.Parameter(torch.ones(dim_model))
        # Initialize gamma accounting for unit offset
        nn.init.constant_(self.gamma, 1. - float(unit_offset))

    def apply(self, x: Tensor, **kwargs) -> Tensor:
        """Apply layer normalization"""
        normed = self.ln(x)
        gamma = self.gamma + float(self.unit_offset)
        return normed * gamma


@norm_registry.register("rmsnorm", {
    "unit_offset": (bool, Field(default=False, description="Use unit offset for better weight decay compatibility"))
})
class RMSNormTransform(NormTransform):
    """Root Mean Square Layer Normalization"""

    def __init__(self, dim_model: int, unit_offset: bool = False, **kwargs):
        super().__init__()
        self.unit_offset = unit_offset
        self.scale = dim_model ** 0.5

        self.g = nn.Parameter(torch.zeros(dim_model))
        nn.init.constant_(self.g, 1. - float(unit_offset))

    def apply(self, x: Tensor, **kwargs) -> Tensor:
        """Apply RMS normalization"""
        gamma = self.g + float(self.unit_offset)
        return F.normalize(x, dim=-1) * self.scale * gamma


@norm_registry.register("simple_rmsnorm")
class SimpleRMSNormTransform(NormTransform):
    """Simplest RMS normalization without learnable parameters"""

    def __init__(self, dim_model: int, **kwargs):
        super().__init__()
        self.scale = dim_model ** 0.5

    def apply(self, x: Tensor, **kwargs) -> Tensor:
        """Apply simple RMS normalization"""
        return F.normalize(x, dim=-1) * self.scale


@norm_registry.register("scalenorm", {
    "unit_offset": (bool, Field(default=False, description="Use unit offset for better weight decay compatibility"))
})
class ScaleNormTransform(NormTransform):
    """
    Scale Normalization

    Simpler than LayerNorm, just normalizes to unit length and scales.
    """

    def __init__(self, dim_model: int, unit_offset: bool = False, **kwargs):
        super().__init__()
        self.unit_offset = unit_offset
        self.scale = dim_model ** 0.5

        self.g = nn.Parameter(torch.zeros(1))
        nn.init.constant_(self.g, 1. - float(unit_offset))

    def apply(self, x: Tensor, **kwargs) -> Tensor:
        """Apply scale normalization"""
        gamma = self.g + float(self.unit_offset)
        return F.normalize(x, dim=-1) * self.scale * gamma


@norm_registry.register("multihead_rmsnorm", {
    "num_heads": (int, Field(default=8, description="Number of attention heads"))
})
class MultiheadRMSNormTransform(NormTransform):
    """RMS Normalization with per-head scaling"""

    def __init__(self, dim_model: int, num_heads: int = 8, **kwargs):
        super().__init__()
        self.scale = dim_model ** 0.5
        self.gamma = nn.Parameter(torch.zeros(num_heads, 1, dim_model))

    def apply(self, x: Tensor, **kwargs) -> Tensor:
        """Apply multihead RMS normalization"""
        normed = F.normalize(x, dim=-1) * self.scale
        return normed * (self.gamma + 1.)


@norm_registry.register("adaptive_layernorm", {
    "dim_condition": (Optional[int], Field(default=None, description="Dimension of conditioning vector"))
})
class AdaptiveLayerNormTransform(NormTransform):
    """Layer Normalization conditioned on external input"""

    def __init__(self, dim_model: int, dim_condition: Optional[int] = None, **kwargs):
        super().__init__()
        dim_condition = dim_condition if dim_condition is not None else dim_model

        self.ln = nn.LayerNorm(dim_model, elementwise_affine=False)
        self.to_gamma = nn.Linear(dim_condition, dim_model, bias=False)
        nn.init.zeros_(self.to_gamma.weight)

    def apply(self, x: Tensor, condition: Optional[Tensor] = None, **kwargs) -> Tensor:
        """Apply adaptive layer normalization"""
        if condition is None:
            raise ValueError("AdaptiveLayerNorm requires 'condition' argument")

        if condition.ndim == 2:
            condition = rearrange(condition, 'b d -> b 1 d')

        normed = self.ln(x)
        gamma = self.to_gamma(condition)
        return normed * (gamma + 1.)


@norm_registry.register("adaptive_rmsnorm", {
    "dim_condition": (Optional[int], Field(default=None, description="Dimension of conditioning vector"))
})
class AdaptiveRMSNormTransform(NormTransform):
    """RMS Normalization conditioned on external input"""

    def __init__(self, dim_model: int, dim_condition: Optional[int] = None, **kwargs):
        super().__init__()
        self.scale = dim_model ** 0.5
        dim_condition = dim_condition if dim_condition is not None else dim_model

        self.to_gamma = nn.Linear(dim_condition, dim_model, bias=False)
        nn.init.zeros_(self.to_gamma.weight)

    def apply(self, x: Tensor, condition: Optional[Tensor] = None, **kwargs) -> Tensor:
        """Apply adaptive RMS normalization"""
        if condition is None:
            raise ValueError("AdaptiveRMSNorm requires 'condition' argument")

        if condition.ndim == 2:
            condition = rearrange(condition, 'b d -> b 1 d')

        normed = F.normalize(x, dim=-1)
        gamma = self.to_gamma(condition)
        return normed * self.scale * (gamma + 1.)


@norm_registry.register("dynamic_tanh", {
    "init_alpha": (float, Field(default=1., description="Initial pre-tanh scale")),
    "unit_offset": (bool, Field(default=False, description="Use unit offset"))
})
class DynamicTanhTransform(NormTransform):
    """
    Dynamic Tanh normalization from https://arxiv.org/abs/2503.10622

    Applies learnable tanh-based transformation with scaling.
    """

    def __init__(self, dim_model: int, init_alpha: float = 1., unit_offset: bool = False, **kwargs):
        super().__init__()
        self.pre_tanh_scale = nn.Parameter(torch.tensor(init_alpha))

        self.gamma = nn.Parameter(torch.ones(dim_model))
        self.beta = nn.Parameter(torch.zeros(dim_model))

        self.pre_tanh_scale_offset = init_alpha if unit_offset else 0.
        self.gamma_offset = float(unit_offset)

        nn.init.constant_(self.pre_tanh_scale, 0 if unit_offset else init_alpha)
        nn.init.constant_(self.gamma, 1. - float(unit_offset))

    def apply(self, x: Tensor, **kwargs) -> Tensor:
        """Apply dynamic tanh transformation"""
        pre_tanh_scale = self.pre_tanh_scale + self.pre_tanh_scale_offset
        gamma = self.gamma + self.gamma_offset
        return (x * pre_tanh_scale).tanh() * gamma + self.beta


@norm_registry.register("derf", {
    "init_alpha": (float, Field(default=0.5, description="Initial pre-erf scaling parameter")),
    "init_bias": (float, Field(default=0., description="Initial bias before erf")),
    "unit_offset": (bool, Field(default=False, description="Use unit offset for identity-like initial behavior"))
})
class DerfTransform(NormTransform):
    """
    Derf (Derivative of Error Function) Normalization
    Paper: https://arxiv.org/abs/2512.10938

    Uses the error function (erf) as the activation, which is the integral of the
    Gaussian distribution. The erf function has smooth, bounded behavior similar to
    tanh but with different gradient characteristics that may be beneficial for training.

    The error function is defined as: erf(x) = (2/√π) * ∫₀ˣ e^(-t²) dt

    This applies learnable scaling and shifting both before and after the erf:
    Forward computation: erf(x * alpha + s) * gamma + beta

    Args:
        dim_model: Feature dimension (for per-dimension gamma and beta parameters)
        init_alpha: Initial value for the pre-erf scaling parameter (default: 0.5)
        init_bias: Initial value for the shift term 's' applied before erf (default: 0.0)
        unit_offset: If True, uses unit initialization for identity-like initial behavior
    """

    def __init__(
        self,
        dim_model: int,
        init_alpha: float = 0.5,
        init_bias: float = 0.,
        unit_offset: bool = False,
        **kwargs
    ):
        super().__init__()
        scale_offset = 1. if unit_offset else 0.

        self.alpha = nn.Parameter(torch.tensor(init_alpha) - scale_offset)
        self.s = nn.Parameter(torch.tensor(init_bias))

        self.gamma = nn.Parameter(torch.ones(dim_model) - scale_offset)
        self.beta = nn.Parameter(torch.zeros(dim_model))

        self.scale_offset = scale_offset

    def apply(self, x: Tensor, **kwargs) -> Tensor:
        """Apply Derf normalization"""
        x = x * (self.alpha + self.scale_offset) + self.s
        activated = torch.erf(x)
        return activated * (self.gamma + self.scale_offset) + self.beta


# Utility function for L2 normalization (keep existing)
def l2norm(t: Tensor, groups: int = 1) -> Tensor:
    """Apply L2 normalization across groups of features"""
    t = rearrange(t, '... (g d) -> ... g d', g=groups)
    t = F.normalize(t, p=2, dim=-1)
    return rearrange(t, '... g d -> ... (g d)')
