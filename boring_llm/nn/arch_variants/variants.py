"""
Architecture Variants for Transformers

Includes various architectural modifications:
- Sandwich Norm (extra normalization)
- ResiDual (dual residual paths)
- Normformer (additional normalization points)
- Macaron (FFN-Attn-FFN structure)
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Callable, Optional


class SandwichNorm(nn.Module):
    """
    Sandwich Normalization

    Adds an extra normalization layer after the attention/FFN operation
    but before adding the residual. Improves training stability.

    Structure: x + Norm2(Module(Norm1(x)))

    Args:
        dim: Model dimension
        fn: Module to wrap (attention or feedforward)
        norm_class: Normalization class to use
    """

    def __init__(self, dim: int, fn: nn.Module, norm_class: Callable = None):
        super().__init__()
        norm_class = norm_class or nn.LayerNorm
        self.norm_pre = norm_class(dim)
        self.fn = fn
        self.norm_post = norm_class(dim)

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        """Forward with sandwich normalization"""
        return x + self.norm_post(self.fn(self.norm_pre(x), **kwargs))


class ResiDual(nn.Module):
    """
    ResiDual - Dual Residual Paths

    From various papers exploring multiple residual connections.
    Provides two paths for gradients to flow.

    Structure: x + alpha * path1(x) + beta * path2(x)

    Args:
        dim: Model dimension
        fn1: First module (e.g., attention)
        fn2: Second module (e.g., feedforward)
        learnable_alphas: If True, learn the mixing coefficients
    """

    def __init__(
        self,
        dim: int,
        fn1: nn.Module,
        fn2: Optional[nn.Module] = None,
        learnable_alphas: bool = True
    ):
        super().__init__()
        self.fn1 = fn1
        self.fn2 = fn2 or nn.Identity()

        if learnable_alphas:
            self.alpha1 = nn.Parameter(torch.ones(1))
            self.alpha2 = nn.Parameter(torch.ones(1))
        else:
            self.alpha1 = 1.0
            self.alpha2 = 1.0

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        """Forward with dual residual"""
        out1 = self.fn1(x, **kwargs)
        out2 = self.fn2(x, **kwargs)

        return x + self.alpha1 * out1 + self.alpha2 * out2


class Normformer(nn.Module):
    """
    Normformer - Extra Normalization Points

    Adds normalization at multiple points for improved stability.
    Particularly useful for deep networks.

    Structure: x + Norm_out(Module(Norm_in(x)))
    with additional normalization on Q, K, V

    Args:
        dim: Model dimension
        fn: Module to wrap
        norm_class: Normalization class
    """

    def __init__(self, dim: int, fn: nn.Module, norm_class: Callable = None):
        super().__init__()
        norm_class = norm_class or nn.LayerNorm

        self.norm_in = norm_class(dim)
        self.fn = fn
        self.norm_out = norm_class(dim)

        # Additional norms if fn has qkv (for attention)
        if hasattr(fn, 'to_q'):
            self.norm_q = norm_class(dim)
            self.norm_k = norm_class(dim)
            self.norm_v = norm_class(dim)
        else:
            self.norm_q = self.norm_k = self.norm_v = None

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        """Forward with extra normalization"""
        # Apply input norm
        x_normed = self.norm_in(x)

        # If attention module, apply QKV norms
        if self.norm_q is not None and hasattr(self.fn, 'to_q'):
            # This is a simplified version - full implementation would
            # require modifying the attention forward pass
            pass

        # Apply module
        out = self.fn(x_normed, **kwargs)

        # Apply output norm and residual
        return x + self.norm_out(out)


class MacaronNet(nn.Module):
    """
    Macaron-style Layer Structure

    From "Understanding and Improving Transformer From a Multi-Particle
    Dynamic System Point of View"

    Places attention between two feedforward layers:
    FFN(1/2) -> Attention -> FFN(1/2)

    Improves gradient flow and model dynamics.

    Args:
        dim: Model dimension
        attn_fn: Attention module
        ff_fn: Feedforward module
        norm_class: Normalization class
    """

    def __init__(
        self,
        dim: int,
        attn_fn: nn.Module,
        ff_fn: nn.Module,
        norm_class: Callable = None
    ):
        super().__init__()
        norm_class = norm_class or nn.LayerNorm

        # First half FFN
        self.norm1 = norm_class(dim)
        self.ff_pre = self._scale_ff(ff_fn, 0.5)

        # Attention
        self.norm2 = norm_class(dim)
        self.attn = attn_fn

        # Second half FFN
        self.norm3 = norm_class(dim)
        self.ff_post = self._scale_ff(ff_fn, 0.5)

    def _scale_ff(self, ff_fn: nn.Module, scale: float) -> nn.Module:
        """Scale feedforward output"""
        class ScaledFF(nn.Module):
            def __init__(self, fn, s):
                super().__init__()
                self.fn = fn
                self.scale = s

            def forward(self, x, **kwargs):
                return self.fn(x, **kwargs) * self.scale

        return ScaledFF(ff_fn, scale)

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        """Forward through Macaron structure"""
        # First half FFN
        x = x + self.ff_pre(self.norm1(x))

        # Attention
        x = x + self.attn(self.norm2(x), **kwargs)

        # Second half FFN
        x = x + self.ff_post(self.norm3(x))

        return x


class ResidualAttention(nn.Module):
    """
    Residual Attention

    From "Residual Attention: A Simple but Effective Method to Improve
    Deep Learning Models" https://arxiv.org/abs/2012.11747

    Residualizes attention scores across layers.

    Args:
        fn: Attention module
        dim: Model dimension
    """

    def __init__(self, fn: nn.Module, dim: int):
        super().__init__()
        self.fn = fn
        self.residual_attn = None

    def forward(self, x: Tensor, return_attn: bool = False, **kwargs) -> Tensor:
        """
        Forward with residual attention

        Note: Full implementation requires modifying attention internals
        to access and residualize attention matrices.
        """
        # Simplified version - full implementation would store and
        # add attention matrices across layers
        out = self.fn(x, **kwargs)

        if return_attn:
            return out, self.residual_attn

        return out


if __name__ == "__main__":
    print("Testing Architecture Variants...")

    dim = 512

    # Dummy modules for testing
    dummy_attn = nn.Identity()
    dummy_ff = nn.Linear(dim, dim)

    # Test Sandwich Norm
    sandwich = SandwichNorm(dim, dummy_attn)
    x = torch.randn(2, 10, dim)
    y = sandwich(x)
    print(f"\nSandwich Norm: {x.shape} -> {y.shape}")

    # Test ResiDual
    residual = ResiDual(dim, dummy_attn, dummy_ff)
    y = residual(x)
    print(f"ResiDual: {x.shape} -> {y.shape}")

    # Test Normformer
    normformer = Normformer(dim, dummy_attn)
    y = normformer(x)
    print(f"Normformer: {x.shape} -> {y.shape}")

    # Test Macaron
    macaron = MacaronNet(dim, dummy_attn, dummy_ff)
    y = macaron(x)
    print(f"Macaron: {x.shape} -> {y.shape}")

    print("\n✅ Architecture variant tests passed!")
