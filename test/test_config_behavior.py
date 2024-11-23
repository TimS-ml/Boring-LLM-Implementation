import pytest
from pydantic import ValidationError
from boring_nn.ffn.core import FeedForwardConfig, ActivationType, ActivationConfig
from boring_nn.attention.config import (
    AttentionConfig, AttentionType, AttentionTypeConfig, QKNormConfig
)
from boring_transformer.core import TransformerLayerWrapConfig, TransformerLayersConfig
from boring_llm_base.base_config import BaseConfig


# def test_config_validation():
#     """Requires Field gt
#     ffn_dim: int = Field(default=2048, gt=0, description="Feed-forward network dimension")
#     """
#     with pytest.raises(ValidationError):
#         FeedForwardConfig(ffn_dim=-1)
#
#     with pytest.raises(ValidationError):
#         AttentionConfig(num_heads=-1)
#
#     with pytest.raises(ValidationError):
#         TransformerLayersConfig(depth=-1)

def test_config_inheritance():
    class ChildConfig(BaseConfig):
        child_param: int = 10

    config = ChildConfig()
    assert config.d_model == 512  # Inherited from BaseConfig
    assert config.child_param == 10  # New parameter

def test_config_override():
    config = BaseConfig(d_model=1024, dropout=0.2)
    assert config.d_model == 1024
    assert config.dropout == 0.2


# import IPython; IPython.embed()
