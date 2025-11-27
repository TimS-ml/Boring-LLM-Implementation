from typing import Optional
from pydantic import Field
import torch
import torch.nn as nn
from torch import Tensor

from boring_llm.base.component_registry import ComponentConfig
from boring_llm.nn.connections.registry import connection_registry


class ConnectionConfig(ComponentConfig):
    """
    Connection/Wrapper Configuration

    Supports various connection strategies:
    - residual: Standard or scaled residual connection
    - gru_gating: GRU-gated residual connection
    - hyper_connection: Dynamic multi-stream residual
    - layer_scale: Learnable per-channel scaling
    - adaptive_layer_scale: Condition-dependent scaling
    - dynamic_lime: Dynamic layer aggregation
    - shift_tokens: Token shifting
    - fold_axially: Axial folding
    """
    # Residual connection fields
    scale_residual: bool = Field(default=False, description="Use learnable residual scaling")
    scale_residual_constant: float = Field(default=1., description="Constant residual scale")

    # HyperConnection fields
    layer_index: int = Field(default=0, description="Current layer index")
    num_residual_streams: int = Field(default=4, description="Number of residual streams")
    num_input_views: int = Field(default=1, description="Number of input views")
    use_tanh: bool = Field(default=True, description="Use tanh activation")

    # LayerScale fields
    init_value: float = Field(default=0., description="Initial scale value")
    unit_offset: bool = Field(default=False, description="Use unit offset")

    # AdaptiveLayerScale fields
    dim_condition: Optional[int] = Field(default=None, description="Conditioning dimension")
    init_bias_value: float = Field(default=-2., description="Initial bias value")

    # DynamicLIMe fields
    num_layers: int = Field(default=12, description="Number of layers to aggregate")
    num_views: int = Field(default=1, description="Number of output views")
    use_norm: bool = Field(default=True, description="Use normalization")
    use_softmax: bool = Field(default=True, description="Use softmax activation")

    # ShiftTokens fields
    shifts: tuple = Field(default=(0,), description="Shift amounts")

    # FoldAxially fields
    axial_dim: int = Field(default=1, description="Axial dimension")


class BoringConnection(nn.Module):
    """
    Unified connection/wrapper module

    Example:
        # Residual connection with scaling
        conn = BoringConnection(ConnectionConfig(
            type="residual",
            dim_model=512,
            scale_residual=True
        ))

        # Layer scale
        conn = BoringConnection(ConnectionConfig(
            type="layer_scale",
            dim_model=512,
            init_value=1e-4
        ))
    """

    def __init__(self, config: ConnectionConfig = None, **kwargs):
        super().__init__()
        config = ConnectionConfig(**kwargs) if not config else config.model_copy(update=kwargs)

        # Create strategy
        strategy_kwargs = {
            'dim_model': config.dim_model,
        }

        # Add type-specific kwargs
        if config.type == "residual":
            strategy_kwargs.update({
                'scale_residual': config.scale_residual,
                'scale_residual_constant': config.scale_residual_constant
            })
        elif config.type == "gru_gating":
            strategy_kwargs['scale_residual'] = config.scale_residual
        elif config.type == "hyper_connection":
            strategy_kwargs.update({
                'layer_index': config.layer_index,
                'num_residual_streams': config.num_residual_streams,
                'num_input_views': config.num_input_views,
                'use_tanh': config.use_tanh
            })
        elif config.type == "layer_scale":
            strategy_kwargs.update({
                'init_value': config.init_value,
                'unit_offset': config.unit_offset
            })
        elif config.type == "adaptive_layer_scale":
            strategy_kwargs.update({
                'dim_condition': config.dim_condition,
                'init_bias_value': config.init_bias_value
            })
        elif config.type == "dynamic_lime":
            strategy_kwargs.update({
                'num_layers': config.num_layers,
                'num_views': config.num_views,
                'use_norm': config.use_norm,
                'use_softmax': config.use_softmax
            })
        elif config.type == "shift_tokens":
            strategy_kwargs['shifts'] = config.shifts
        elif config.type == "fold_axially":
            strategy_kwargs['axial_dim'] = config.axial_dim

        self.connection_strategy = connection_registry.create_strategy(config.type, **strategy_kwargs)
        self.conn_type = config.type

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        """
        Apply connection transformation

        Args:
            x: Input tensor
            **kwargs: Additional arguments (residual, condition, hiddens, etc.)

        Returns:
            Transformed tensor
        """
        return self.connection_strategy.apply(x, **kwargs)

    def prepare(self, *args, **kwargs):
        """
        Prepare inputs for connection (for residual-type connections)

        Returns:
            Tuple of (branch_input, residual, extra_kwargs)
        """
        if hasattr(self.connection_strategy, 'prepare'):
            return self.connection_strategy.prepare(*args, **kwargs)
        else:
            raise AttributeError(f"{self.conn_type} does not support prepare()")


def create_connection(conn_type: str = "residual", **kwargs) -> BoringConnection:
    """
    Convenience function to create connection module

    Args:
        conn_type: Type of connection
        **kwargs: Additional configuration

    Returns:
        BoringConnection instance

    Example:
        conn = create_connection("layer_scale", dim_model=512, init_value=1e-4)
    """
    if 'type' in kwargs:
        conn_type = kwargs.pop('type')

    config = ConnectionConfig(type=conn_type, **kwargs)
    return BoringConnection(config)


if __name__ == "__main__":
    print("Testing connection implementations...")

    # Example 1: Residual connection
    conn1 = create_connection(
        conn_type="residual",
        dim_model=512,
        scale_residual=True
    )

    # Example 2: GRU gating
    conn2 = create_connection(
        conn_type="gru_gating",
        dim_model=512
    )

    # Example 3: Layer scale
    conn3 = create_connection(
        conn_type="layer_scale",
        dim_model=512,
        init_value=1e-4
    )

    # Example 4: Adaptive layer scale
    conn4 = create_connection(
        conn_type="adaptive_layer_scale",
        dim_model=512,
        dim_condition=256
    )

    # Example 5: Dynamic LIMe
    conn5 = create_connection(
        conn_type="dynamic_lime",
        dim_model=512,
        num_layers=6
    )

    # Example 6: Shift tokens
    conn6 = create_connection(
        conn_type="shift_tokens",
        dim_model=512,
        shifts=(0, 1, 2)
    )

    # Test
    x = torch.randn(2, 10, 512)
    residual = torch.randn(2, 10, 512)
    condition = torch.randn(2, 256)
    hiddens = [torch.randn(2, 10, 512) for _ in range(6)]

    print("\nTesting Residual...")
    y1 = conn1(x, residual=residual)
    print(f"Residual output: {y1.shape}")

    print("\nTesting GRU Gating...")
    y2 = conn2(x, residual=residual)
    print(f"GRU Gating output: {y2.shape}")

    print("\nTesting Layer Scale...")
    y3 = conn3(x)
    print(f"Layer Scale output: {y3.shape}")

    print("\nTesting Adaptive Layer Scale...")
    y4 = conn4(x, condition=condition)
    print(f"Adaptive Layer Scale output: {y4.shape}")

    print("\nTesting Dynamic LIMe...")
    y5 = conn5(x, hiddens=hiddens)
    print(f"Dynamic LIMe output: {y5.shape}")

    print("\nTesting Shift Tokens...")
    y6 = conn6(x)
    print(f"Shift Tokens output: {y6.shape}")

    print("\nAll connection tests passed! ✓")
