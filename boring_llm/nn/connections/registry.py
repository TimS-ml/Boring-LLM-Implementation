from typing import Optional, Callable
from pydantic import Field
import math
import torch
import torch.nn as nn
from torch import Tensor, cat, stack
from torch.nn import Module
from einops import rearrange

from boring_llm.base.component_registry import ComponentTransform, ComponentRegistry


class ConnectionTransform(ComponentTransform):
    """Base class for connection/wrapper transformations"""

    def apply(self, x: Tensor, **kwargs) -> Tensor:
        """Apply connection transformation"""
        raise NotImplementedError


connection_registry = ComponentRegistry[ConnectionTransform]("Connection")


# ============================================================================
# Residual Connection Variants
# ============================================================================

@connection_registry.register("residual", {
    "scale_residual": (bool, Field(default=False, description="Use learnable residual scaling")),
    "scale_residual_constant": (float, Field(default=1., description="Constant residual scale factor"))
})
class ResidualTransform(ConnectionTransform):
    """
    Standard or scaled residual connection

    Supports learnable per-dimension scaling and constant scaling.
    """

    def __init__(self, dim_model: int, scale_residual: bool = False,
                 scale_residual_constant: float = 1., **kwargs):
        super().__init__()
        self.residual_scale = nn.Parameter(torch.ones(dim_model)) if scale_residual else None
        self.scale_residual_constant = scale_residual_constant

    def prepare(self, residual: Tensor):
        """Prepare residual for use - returns branch input, residual, and extra kwargs"""
        return residual, residual, dict()

    def apply(self, x: Tensor, residual: Tensor, **kwargs) -> Tensor:
        """Apply residual connection"""
        if self.residual_scale is not None:
            residual = residual * self.residual_scale

        if self.scale_residual_constant != 1:
            residual = residual * self.scale_residual_constant

        return x + residual


@connection_registry.register("gru_gating", {
    "scale_residual": (bool, Field(default=False, description="Use learnable residual scaling"))
})
class GRUGatingTransform(ConnectionTransform):
    """
    GRU-based gating for residual connections

    Uses a GRU cell to dynamically gate the residual connection.
    """

    def __init__(self, dim_model: int, scale_residual: bool = False, **kwargs):
        super().__init__()
        self.gru = nn.GRUCell(dim_model, dim_model)
        self.residual_scale = nn.Parameter(torch.ones(dim_model)) if scale_residual else None

    def prepare(self, residual: Tensor):
        """Prepare residual for use"""
        return residual, residual, dict()

    def apply(self, x: Tensor, residual: Tensor, **kwargs) -> Tensor:
        """Apply GRU gating"""
        if self.residual_scale is not None:
            residual = residual * self.residual_scale

        gated_output = self.gru(
            rearrange(x, 'b n d -> (b n) d'),
            rearrange(residual, 'b n d -> (b n) d')
        )

        return gated_output.reshape_as(x)


@connection_registry.register("hyper_connection", {
    "layer_index": (int, Field(default=0, description="Index of current layer")),
    "num_residual_streams": (int, Field(default=4, description="Number of residual streams")),
    "num_input_views": (int, Field(default=1, description="Number of input views")),
    "use_tanh": (bool, Field(default=True, description="Use tanh activation"))
})
class HyperConnectionTransform(ConnectionTransform):
    """
    Hyper-connections from https://arxiv.org/abs/2409.19606

    Dynamic multi-stream residual connections with learned mixing.
    """

    def __init__(
        self,
        dim_model: int,
        layer_index: int = 0,
        num_residual_streams: int = 4,
        num_input_views: int = 1,
        use_tanh: bool = True,
        **kwargs
    ):
        super().__init__()

        self.act = nn.Tanh() if use_tanh else nn.Identity()
        self.norm = nn.LayerNorm(dim_model, bias=False)

        self.num_residual_streams = num_residual_streams
        self.layer_index = layer_index
        self.num_input_views = num_input_views

        self.static_beta = nn.Parameter(torch.ones(num_residual_streams))

        init_alpha0 = torch.zeros((num_residual_streams, num_input_views))
        init_alpha0[layer_index % num_residual_streams, :] = 1.

        self.static_alpha = nn.Parameter(
            cat([init_alpha0, torch.eye(num_residual_streams)], dim=1)
        )

        self.dynamic_alpha_fn = nn.Parameter(
            torch.zeros(dim_model, num_residual_streams + num_input_views)
        )
        self.dynamic_alpha_scale = nn.Parameter(torch.ones(()) * 1e-2)

        self.dynamic_beta_fn = nn.Parameter(torch.zeros(dim_model))
        self.dynamic_beta_scale = nn.Parameter(torch.ones(()) * 1e-2)

    def prepare(self, residuals: Tensor):
        """Prepare residuals for hyper-connection"""
        residuals = rearrange(residuals, '(b s) n d -> b n s d', s=self.num_residual_streams)

        normed = self.norm(residuals)

        wc_weight = self.act(normed @ self.dynamic_alpha_fn)
        dynamic_alpha = wc_weight * self.dynamic_alpha_scale
        alpha = dynamic_alpha + self.static_alpha

        dc_weight = self.act(normed @ self.dynamic_beta_fn)
        dynamic_beta = dc_weight * self.dynamic_beta_scale
        beta = dynamic_beta + self.static_beta

        # Width connection
        mix_h = torch.einsum('... s t, ... s d -> ... t d', alpha, residuals)

        views = self.num_input_views

        if views == 1:
            branch_input, residuals = mix_h[..., 0, :], mix_h[..., 1:, :]
        else:
            branch_input, residuals = mix_h[..., :views, :], mix_h[..., views:, :]
            branch_input = rearrange(branch_input, '... v d -> v ... d')

        return branch_input, residuals, dict(beta=beta)

    def apply(self, x: Tensor, residuals: Tensor, beta: Tensor, **kwargs) -> Tensor:
        """Apply hyper-connection"""
        residuals = torch.einsum('b n d, b n s -> b n s d', x, beta) + residuals
        return rearrange(residuals, 'b n s d -> (b s) n d')


# ============================================================================
# Layer Wrappers
# ============================================================================

@connection_registry.register("layer_scale", {
    "init_value": (float, Field(default=0., description="Initial scale value")),
    "unit_offset": (bool, Field(default=False, description="Use unit offset"))
})
class LayerScaleTransform(ConnectionTransform):
    """
    Layer Scale from https://arxiv.org/abs/2103.17239

    Learnable per-channel scaling applied to layer outputs.
    Helps with training stability in deep networks.
    """

    def __init__(self, dim_model: int, init_value: float = 0.,
                 unit_offset: bool = False, **kwargs):
        super().__init__()
        self.unit_offset = unit_offset
        self.gamma = nn.Parameter(torch.zeros(dim_model))
        nn.init.constant_(self.gamma, init_value - float(unit_offset))

    def apply(self, x: Tensor, **kwargs) -> Tensor:
        """Apply layer scale"""
        gamma = self.gamma + float(self.unit_offset)
        return x * gamma


@connection_registry.register("adaptive_layer_scale", {
    "dim_condition": (Optional[int], Field(default=None, description="Dimension of conditioning vector")),
    "init_bias_value": (float, Field(default=-2., description="Initial bias value"))
})
class AdaptiveLayerScaleTransform(ConnectionTransform):
    """
    Adaptive Layer Scale conditioned on external input

    Scale is dynamically computed from conditioning input.
    """

    def __init__(self, dim_model: int, dim_condition: Optional[int] = None,
                 init_bias_value: float = -2., **kwargs):
        super().__init__()
        dim_condition = dim_condition if dim_condition is not None else dim_model
        self.to_gamma = nn.Linear(dim_condition, dim_model)

        nn.init.zeros_(self.to_gamma.weight)
        nn.init.constant_(self.to_gamma.bias, init_bias_value)

    def apply(self, x: Tensor, condition: Optional[Tensor] = None, **kwargs) -> Tensor:
        """Apply adaptive layer scale"""
        if condition is None:
            raise ValueError("AdaptiveLayerScale requires 'condition' argument")

        if condition.ndim == 2:
            condition = rearrange(condition, 'b d -> b 1 d')

        gamma = self.to_gamma(condition).sigmoid()
        return x * gamma


# ============================================================================
# Layer Aggregators
# ============================================================================

@connection_registry.register("dynamic_lime", {
    "num_layers": (int, Field(default=12, description="Number of layers to aggregate")),
    "num_views": (int, Field(default=1, description="Number of output views")),
    "use_norm": (bool, Field(default=True, description="Use normalization before projection")),
    "use_softmax": (bool, Field(default=True, description="Use softmax (else ReLU)"))
})
class DynamicLIMeTransform(ConnectionTransform):
    """
    Dynamic Layer Integrated Memory (LIMe)

    Dynamically aggregates hidden states from multiple layers using
    learned attention-like weights.
    """

    def __init__(
        self,
        dim_model: int,
        num_layers: int = 12,
        num_views: int = 1,
        use_norm: bool = True,
        use_softmax: bool = True,
        **kwargs
    ):
        super().__init__()
        from boring_llm.nn.norm.registry import RMSNormTransform

        self.num_layers = num_layers
        self.multiple_views = num_views > 1

        layers = []
        if use_norm:
            layers.append(RMSNormTransform(dim_model))

        self.norm = layers[0] if use_norm else None

        self.to_weights_proj = nn.Linear(dim_model, num_views * num_layers)
        self.num_views = num_views
        self.activation = nn.Softmax(dim=-1) if use_softmax else nn.ReLU()

    def apply(self, x: Tensor, hiddens: list[Tensor], **kwargs) -> Tensor:
        """
        Apply dynamic LIMe aggregation

        Args:
            x: Current hidden state [batch, seq, dim]
            hiddens: List of hidden states from previous layers
        """
        if not isinstance(hiddens, Tensor):
            hiddens = stack(hiddens)

        assert hiddens.shape[0] == self.num_layers, \
            f'Expected {self.num_layers} layers but got {hiddens.shape[0]}'

        # Compute attention weights
        if self.norm is not None:
            x_normed = self.norm.apply(x)
        else:
            x_normed = x

        weights = self.to_weights_proj(x_normed)  # [batch, seq, views*layers]
        weights = rearrange(weights, 'b n (v l) -> v b n l', v=self.num_views)
        weights = self.activation(weights)

        # Aggregate hidden states
        out = torch.einsum('l b n d, v b n l -> v b n d', hiddens, weights)

        if self.multiple_views:
            return out

        return rearrange(out, '1 ... -> ...')


# ============================================================================
# Utility Transforms
# ============================================================================

def shift(t: Tensor, amount: int, mask: Optional[Tensor] = None) -> Tensor:
    """Shift tokens along sequence dimension"""
    if amount == 0:
        return t

    amount = min(amount, t.shape[1])

    if mask is not None:
        t = t.masked_fill(~mask[..., None], 0.)

    # Pad and crop to shift
    padding = (amount, -amount) if amount > 0 else (-amount, amount)
    return nn.functional.pad(t[:, :t.shape[1]-abs(amount)], (0, 0, max(0, amount), max(0, -amount)))


@connection_registry.register("shift_tokens", {
    "shifts": (tuple, Field(default=(0,), description="Shift amounts for each segment"))
})
class ShiftTokensTransform(ConnectionTransform):
    """
    Token shifting along sequence dimension

    Splits features into segments and shifts each by different amounts.
    Useful for causal relationships in sequence modeling.
    """

    def __init__(self, dim_model: int, shifts: tuple = (0,), **kwargs):
        super().__init__()
        self.shifts = tuple(shifts)
        self.segments = len(shifts)
        self.feats_per_shift = dim_model // self.segments

    def apply(self, x: Tensor, mask: Optional[Tensor] = None, **kwargs) -> Tensor:
        """Apply token shifting"""
        shifts = self.shifts
        segments = self.segments
        feats_per_shift = self.feats_per_shift

        splitted = x.split(feats_per_shift, dim=-1)
        segments_to_shift, rest = splitted[:segments], splitted[segments:]

        segments_to_shift = [
            shift(seg, amount, mask=mask)
            for seg, amount in zip(segments_to_shift, shifts)
        ]

        return cat((*segments_to_shift, *rest), dim=-1)


@connection_registry.register("fold_axially", {
    "axial_dim": (int, Field(default=1, description="Axial dimension for folding"))
})
class FoldAxiallyTransform(ConnectionTransform):
    """
    Axial folding for sequence processing

    Folds sequence dimension by axial_dim for efficient processing
    of long sequences.
    """

    def __init__(self, dim_model: int, axial_dim: int = 1, **kwargs):
        super().__init__()
        self.axial_dim = axial_dim

    def apply(self, x: Tensor, **kwargs) -> Tensor:
        """Apply axial folding"""
        if self.axial_dim == 1:
            return x

        seq_len = x.shape[1]
        axial_dim = self.axial_dim

        # Pad to next multiple
        next_multiple = math.ceil(seq_len / axial_dim) * axial_dim
        if seq_len < next_multiple:
            padding = (0, 0, 0, next_multiple - seq_len)
            x = nn.functional.pad(x, padding)

        # Fold
        x = rearrange(x, 'b (n a) ... -> (b a) n ...', a=axial_dim)
        return x
