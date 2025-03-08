"""
TODO: I want to keep the current config hierarchy, but also want to load all configs at once.
Potential way to achieve this:

class BoringLLM:
    @classmethod
    def from_pretrained(cls, name: str, **kwargs):
        base_config = BaseConfig.from_name(name, config_key="base_config")
        attn_config = AttentionConfig.from_name(name, config_key="attention_config")
        ffn_config = FFNConfig.from_name(name, config_key="ffn_config")
        # ... other configs

        model_config = {
            "base": base_config,
            "attention": attn_config,
            # ... other configs
        }

        return cls(config=model_config)
"""

from pathlib import Path
from pydantic import BaseModel, Field
from typing import Any
from copy import deepcopy

from base.model_configs import MODEL_CONFIGS
name_to_config = {config["name"]: config for config in MODEL_CONFIGS}


class BaseConfig(BaseModel):
    """
    Usage:
        config = BaseConfig.from_name("microsoft/phi-2")  # nested config is supported
        config = BaseConfig.from_name("microsoft/phi-2", config_key="attention_config")  # load attention config only
        config = BaseConfig.from_name("phi-2", dropout=0.2)  # override dropout rate
        config = BaseConfig.from_file("config.yaml")  # load from yaml file
        config = BaseConfig.from_checkpoint("checkpoints/model-v1")  # load from checkpoint dir
    """
    d_model: int                    = Field(default=512,   description="Input and output dim")
    num_tokens: int                 = Field(default=20000, description="Tokenizer's vocab size")
    dropout: float                  = Field(default=0.1,   description="Global dropout rate")

    @classmethod
    def from_name(cls, name: str, config_key: str = "base_config", **kwargs: Any) -> "BaseConfig":
        """
        Create config object from predefined config name
        
        Args:
            name: Model name (e.g. "microsoft/phi-2")
            config_key: Which nested config to load (e.g. "base_config", "attention_config")
            **kwargs: Additional overrides
        """
        if name not in name_to_config:
            try:
                conf_dict = next(
                    config for config in MODEL_CONFIGS
                    if name == config["hf_config"]["name"]
                    or config["hf_config"]["org"] + "/" + config["hf_config"]["name"] == name
                )
            except StopIteration:
                raise ValueError(f"{name!r} is not a supported config name")
        else:
            conf_dict = name_to_config[name]

        if config_key in conf_dict:
            nested_config = conf_dict[config_key]
        else:
            # we can still use the whole config dict as the nested config
            nested_config = conf_dict
            
        nested_config = deepcopy(nested_config)
        nested_config.update(kwargs)
        return cls(**nested_config)

    @classmethod
    def from_file(cls, path: str, config_key: str = "base_config", **kwargs: Any) -> "BaseConfig":
        """Load from config file"""
        import yaml
        with open(path) as f:
            file_kwargs = yaml.safe_load(f)
            if file_kwargs is None:
                raise ValueError(f"{path} is empty which is likely unexpected.")
        
        if config_key in file_kwargs:
            nested_config = file_kwargs[config_key]
            nested_config.update(kwargs)
            return cls(**nested_config)
        else:
            file_kwargs.update(kwargs)
            return cls(**file_kwargs)

    @classmethod
    def from_checkpoint(cls, path: Path, **kwargs: Any) -> "BaseConfig":
        """Load config from checkpoint directory"""
        if (config_path := path / "config.yaml").is_file():
            return cls.from_file(config_path, **kwargs)
        if (model_name := path.name) in name_to_config:
            return cls.from_name(model_name, **kwargs)
        raise FileNotFoundError(f"For {str(path)!r} neither 'config.yaml' nor matching config exists.")
