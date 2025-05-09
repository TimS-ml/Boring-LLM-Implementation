import pytest
import torch
from boring_llm.nn.ffn.config import (
    FeedForwardConfig, ActivationType, ActivationConfig,
)
from boring_llm.nn.ffn.main import BoringFeedForward

# ------------------------------
# Base Config
# ------------------------------
@pytest.fixture
def default_config():
    return FeedForwardConfig(
        d_model=512,
        dropout=0.1,
        activation=ActivationConfig(type=ActivationType.GELU)
    )

@pytest.fixture
def glu_config():
    return FeedForwardConfig(
        d_model=512,
        dropout=0.1,
        activation=ActivationConfig(use_glu=True, type=ActivationType.GELU)
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
        activation=ActivationConfig(type=ActivationType.GELU),
        zero_init_output=True
    )
    ffn = BoringFeedForward(config)
    assert torch.allclose(ffn.feedforward[-1].weight, torch.zeros_like(ffn.feedforward[-1].weight))


# ------------------------------
# Config Tests 
# ------------------------------
def test_different_activation_types():
    for activation_type in ActivationType:
        config = FeedForwardConfig(
            d_model=512,
            dropout=0.1,
            activation=ActivationConfig(type=activation_type)
        )
        ffn = BoringFeedForward(config)
        x = torch.randn(2, 10, 512)
        output = ffn(x)
        assert output.shape == (2, 10, 512)

def test_no_bias():
    config = FeedForwardConfig(
        d_model=512,
        dropout=0.1,
        no_bias=True
    )
    ffn = BoringFeedForward(config)

    assert ffn.feedforward[-1].bias is None


# ------------------------------
# Config Tests (GLU)
# ------------------------------
def test_glu_activation(glu_config):
    ffn = BoringFeedForward(glu_config)
    assert isinstance(ffn.feedforward[0], ffn.glu.__class__)

def test_glu_activation_types(glu_config):
    ffn = BoringFeedForward(glu_config)
    x = torch.randn(2, 10, 512)
    output = ffn(x)
    assert output.shape == (2, 10, 512)

def test_glu_no_bias():
    config = FeedForwardConfig(
        d_model=512,
        dropout=0.1,
        activation=ActivationConfig(use_glu=True),
        no_bias=True
    )
    ffn = BoringFeedForward(config)

    assert ffn.feedforward[0].proj.bias is None
    assert ffn.feedforward[-1].bias is None


# import IPython; IPython.embed()
