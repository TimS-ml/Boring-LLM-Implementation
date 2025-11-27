from typing import Optional
from pydantic import Field
import torch
import torch.nn as nn
from torch import Tensor

from boring_llm.base.component_registry import ComponentConfig
from boring_llm.nn.norm.registry import norm_registry


class NormConfig(ComponentConfig):
    """
    Normalization Configuration

    Supports multiple normalization types:
    - layernorm: Layer normalization without bias
    - rmsnorm: Root mean square normalization
    - simple_rmsnorm: Simplest RMSNorm without learnable parameters
    - scalenorm: Scale normalization
    - multihead_rmsnorm: RMSNorm with per-head scaling
    - adaptive_layernorm: LayerNorm conditioned on external input
    - adaptive_rmsnorm: RMSNorm conditioned on external input
    - dynamic_tanh: Dynamic tanh-based normalization
    """
    # Common fields
    unit_offset: bool = Field(default=False, description="Use unit offset for better weight decay compatibility")

    # MultiheadRMSNorm fields
    num_heads: int = Field(default=8, description="Number of attention heads for multihead normalization")

    # Adaptive normalization fields
    dim_condition: Optional[int] = Field(default=None, description="Dimension of conditioning vector")

    # DynamicTanh fields
    init_alpha: float = Field(default=1., description="Initial pre-tanh scale for dynamic_tanh")


class BoringNorm(nn.Module):
    """
    Unified normalization module supporting multiple normalization strategies

    Example:
        # RMSNorm
        norm = BoringNorm(NormConfig(type="rmsnorm", dim_model=512))

        # Adaptive LayerNorm
        norm = BoringNorm(NormConfig(
            type="adaptive_layernorm",
            dim_model=512,
            dim_condition=256
        ))
    """

    def __init__(self, config: NormConfig = None, **kwargs):
        super().__init__()
        config = NormConfig(**kwargs) if not config else config.model_copy(update=kwargs)

        # Create normalization strategy
        strategy_kwargs = {
            'dim_model': config.dim_model,
        }

        # Add type-specific kwargs
        if config.type in ["layernorm", "rmsnorm", "scalenorm"]:
            strategy_kwargs['unit_offset'] = config.unit_offset
        elif config.type == "multihead_rmsnorm":
            strategy_kwargs['num_heads'] = config.num_heads
        elif config.type in ["adaptive_layernorm", "adaptive_rmsnorm"]:
            strategy_kwargs['dim_condition'] = config.dim_condition
        elif config.type == "dynamic_tanh":
            strategy_kwargs['init_alpha'] = config.init_alpha
            strategy_kwargs['unit_offset'] = config.unit_offset

        self.norm_strategy = norm_registry.create_strategy(config.type, **strategy_kwargs)
        self.norm_type = config.type

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        """
        Apply normalization to input tensor

        Args:
            x: Input tensor of shape [batch, seq_len, dim] or [..., dim]
            **kwargs: Additional arguments (e.g., condition for adaptive norms)

        Returns:
            Normalized tensor
        """
        return self.norm_strategy.apply(x, **kwargs)


def create_norm(norm_type: str = "rmsnorm", **kwargs) -> BoringNorm:
    """
    Convenience function to create normalization layer

    Args:
        norm_type: Type of normalization (rmsnorm, layernorm, etc.)
        **kwargs: Additional configuration parameters

    Returns:
        BoringNorm instance

    Example:
        norm = create_norm("rmsnorm", dim_model=512, unit_offset=True)
    """
    if 'type' in kwargs:
        norm_type = kwargs.pop('type')

    config = NormConfig(type=norm_type, **kwargs)
    return BoringNorm(config)


if __name__ == "__main__":
    # Example 1: RMSNorm
    norm1 = create_norm(
        norm_type="rmsnorm",
        dim_model=512,
        unit_offset=False
    )

    # Example 2: LayerNorm with unit offset
    norm2 = create_norm(
        norm_type="layernorm",
        dim_model=512,
        unit_offset=True
    )

    # Example 3: Simple RMSNorm
    norm3 = create_norm(
        norm_type="simple_rmsnorm",
        dim_model=512
    )

    # Example 4: ScaleNorm
    norm4 = create_norm(
        norm_type="scalenorm",
        dim_model=512
    )

    # Example 5: Multihead RMSNorm
    norm5 = create_norm(
        norm_type="multihead_rmsnorm",
        dim_model=512,
        num_heads=8
    )

    # Example 6: Adaptive LayerNorm
    norm6 = create_norm(
        norm_type="adaptive_layernorm",
        dim_model=512,
        dim_condition=256
    )

    # Example 7: Adaptive RMSNorm
    norm7 = create_norm(
        norm_type="adaptive_rmsnorm",
        dim_model=512,
        dim_condition=256
    )

    # Example 8: Dynamic Tanh
    norm8 = create_norm(
        norm_type="dynamic_tanh",
        dim_model=512,
        init_alpha=1.0,
        unit_offset=False
    )

    # Test
    x = torch.randn(2, 10, 512)
    condition = torch.randn(2, 256)  # For adaptive norms

    print("Testing normalization implementations...")

    y1 = norm1(x)
    print(f"RMSNorm output: {y1.shape}")

    y2 = norm2(x)
    print(f"LayerNorm output: {y2.shape}")

    y3 = norm3(x)
    print(f"Simple RMSNorm output: {y3.shape}")

    y4 = norm4(x)
    print(f"ScaleNorm output: {y4.shape}")

    y5 = norm5(x)
    print(f"Multihead RMSNorm output: {y5.shape}")

    y6 = norm6(x, condition=condition)
    print(f"Adaptive LayerNorm output: {y6.shape}")

    y7 = norm7(x, condition=condition)
    print(f"Adaptive RMSNorm output: {y7.shape}")

    y8 = norm8(x)
    print(f"Dynamic Tanh output: {y8.shape}")

    print("\nAll normalization tests passed! ✓")
