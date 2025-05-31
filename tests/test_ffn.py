import pytest
import torch
import torch.nn as nn
from boring_llm.nn.ffn.config import FeedForwardConfig, create_ffn_config
from boring_llm.nn.ffn.main import BoringFeedForward
from boring_llm.nn.activation.activation import ReluSquared

# ------------------------------
# Base Config
# ------------------------------
@pytest.fixture
def default_config():
    return FeedForwardConfig(
        d_model=512,
        dropout=0.1,
        activation=nn.GELU()
    )

@pytest.fixture
def glu_config():
    return FeedForwardConfig(
        d_model=512,
        dropout=0.1,
        activation=nn.GELU()
    )


# ------------------------------
# Shape Tests 
# ------------------------------
def test_output_shape(default_config):
    ffn = BoringFeedForward(default_config)
    x = torch.randn(2, 10, 512)
    output = ffn(x)
    assert output.shape == (2, 10, 512)

def test_zero_init_output():
    config = FeedForwardConfig(
        d_model=512,
        dropout=0.1,
        activation=nn.GELU(),
        zero_init_output=True
    )
    ffn = BoringFeedForward(config)
    assert torch.allclose(ffn.post_processor.proj.weight, torch.zeros_like(ffn.post_processor.proj.weight))


# ------------------------------
# Config Tests 
# ------------------------------
def test_different_activation_types():
    activations = [
        nn.GELU(),
        nn.ReLU(),
        nn.SiLU(),
        nn.Tanh(),
        ReluSquared(),
        nn.Identity()
    ]
    
    for activation in activations:
        config = FeedForwardConfig(
            d_model=512,
            dropout=0.1,
            activation=activation
        )
        ffn = BoringFeedForward(config)
        x = torch.randn(2, 10, 512)
        output = ffn(x)
        assert output.shape == (2, 10, 512)

def test_no_bias():
    config = FeedForwardConfig(
        d_model=512,
        dropout=0.1,
        no_bias=True,
        activation=nn.GELU()
    )
    ffn = BoringFeedForward(config)
    
    # Check standard transform projection has no bias
    assert ffn.ffn_transform.proj.bias is None
    # Check post processor projection has no bias
    assert ffn.post_processor.proj.bias is None


# ------------------------------
# Config Tests (GLU)
# ------------------------------
def test_glu_activation():
    config = create_ffn_config("glu")(
        dim_model=512,
        mult_dim=2,
        post_type="post_standard",
        mult_bias=False,
        activation=nn.SiLU()
    )
    ffn = BoringFeedForward(config)
    x = torch.randn(2, 10, 512)
    output = ffn(x)
    assert output.shape == (2, 10, 512)

def test_standard_activation():
    config = create_ffn_config("standard")(
        dim_model=512,
        mult_dim=4,
        post_type="post_standard",
        activation=nn.GELU()
    )
    ffn = BoringFeedForward(config)
    x = torch.randn(2, 10, 512)
    output = ffn(x)
    assert output.shape == (2, 10, 512)

def test_relu_squared_activation():
    config = create_ffn_config("standard")(
        dim_model=512,
        mult_dim=4,
        post_type="post_standard",
        activation=ReluSquared()
    )
    ffn = BoringFeedForward(config)
    x = torch.randn(2, 10, 512)
    output = ffn(x)
    assert output.shape == (2, 10, 512)

def test_activation_callable_instantiation():
    # Test that activation classes are properly instantiated
    config = FeedForwardConfig(
        dim_model=512,
        activation=nn.GELU  # Pass class, not instance
    )
    # Should be instantiated in model_post_init
    assert isinstance(config.activation, nn.Module)
    
    ffn = BoringFeedForward(config)
    x = torch.randn(2, 10, 512)
    output = ffn(x)
    assert output.shape == (2, 10, 512)


# ------------------------------
# Integration Tests
# ------------------------------
def test_different_ffn_types():
    """Test different FFN types with various activations"""
    ffn_types = ["standard", "glu"]
    activations = [nn.GELU(), nn.ReLU(), nn.SiLU(), ReluSquared()]
    
    for ffn_type in ffn_types:
        for activation in activations:
            if ffn_type == "glu":
                config = create_ffn_config(ffn_type)(
                    dim_model=256,
                    mult_dim=2,
                    post_type="post_standard",
                    mult_bias=False,
                    activation=activation
                )
            else:
                config = create_ffn_config(ffn_type)(
                    dim_model=256,
                    mult_dim=4,
                    post_type="post_standard",
                    activation=activation
                )
            
            ffn = BoringFeedForward(config)
            x = torch.randn(2, 8, 256)
            output = ffn(x)
            assert output.shape == (2, 8, 256)


if __name__ == "__main__":
    pytest.main([__file__])
