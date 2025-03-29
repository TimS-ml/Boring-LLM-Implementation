import pytest
from boring_llm.base.base_config import BaseConfig
from boring_llm.nn.attention.config import AttentionConfig
from pathlib import Path
import yaml

from boring_utils.utils import cprint


def test_base_config_defaults():
    config = BaseConfig()
    assert config.d_model == 512
    assert config.num_tokens == 20000
    assert config.dropout == 0.1


def test_load_gpt2_config():
    base_config = BaseConfig.from_name("gpt2", config_key="base_config")
    cprint(base_config)
    assert base_config.d_model == 768
    assert base_config.num_tokens == 50257
    assert base_config.dropout == 0.1

    attn_config = AttentionConfig.from_name("gpt2", config_key="attention_config")
    cprint(attn_config)
    assert attn_config.dim_head == 64
    assert attn_config.num_heads == 12
    assert attn_config.bias == True


def test_load_llama_config():
    base_config = BaseConfig.from_name("meta-llama/Llama-3.2-1B", config_key="base_config")
    cprint(base_config)
    assert base_config.d_model == 2048
    assert base_config.num_tokens == 128000
    
    attn_config = AttentionConfig.from_name("meta-llama/Llama-3.2-1B", config_key="attention_config")
    cprint(attn_config)
    assert attn_config.dim_head == 64
    assert attn_config.num_heads == 32
    assert attn_config.bias == False
    assert attn_config.n_query_groups == 8


def test_config_override():
    config = BaseConfig.from_name("gpt2", config_key="base_config", dropout=0.2)
    cprint(config)
    assert config.dropout == 0.2
    assert config.d_model == 768


def test_invalid_model_name():
    with pytest.raises(ValueError, match="is not a supported config name"):
        BaseConfig.from_name("invalid-model")


# NOTE: tmp_path is a pytest fixture
@pytest.fixture
def temp_config_file(tmp_path):
    config = {
        "base_config": {
            "d_model": 1024,
            "num_tokens": 32000,
            "dropout": 0.1
        },
        "attention_config": {
            "dim_head": 64,
            "num_heads": 16,
            "bias": False
        }
    }
    
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    
    return config_path


def test_load_from_file(temp_config_file):
    base_config = BaseConfig.from_file(temp_config_file)
    cprint(base_config)
    assert base_config.d_model == 1024
    assert base_config.num_tokens == 32000


def test_load_from_checkpoint(tmp_path):
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    
    config = {
        "base_config": {
            "d_model": 2048,
            "num_tokens": 50000,
            "dropout": 0.1
        }
    }
    
    config_path = checkpoint_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    
    base_config = BaseConfig.from_checkpoint(checkpoint_dir)
    cprint(base_config)
    assert base_config.d_model == 2048
    assert base_config.num_tokens == 50000


def test_nested_config_inheritance():
    attn_config = AttentionConfig.from_name("gpt2", config_key="attention_config")
    cprint(attn_config)
    assert hasattr(attn_config, "d_model")
    assert hasattr(attn_config, "dropout")
    assert hasattr(attn_config, "dim_head")
    assert hasattr(attn_config, "num_heads")


def test_config_type_validation():
    with pytest.raises(ValueError):
        BaseConfig(d_model="invalid")
    
    with pytest.raises(ValueError):
        BaseConfig(dropout="invalid")


if __name__ == "__main__":
    pytest.main([__file__])


# import IPython; IPython.embed()
