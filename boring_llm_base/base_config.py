from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional, Tuple, Type, Any, Dict, Self, List
from copy import deepcopy

from boring_llm_base.model_configs import MODEL_CONFIGS

name_to_config = {config["name"]: config for config in MODEL_CONFIGS}

class BaseConfig(BaseModel):
    d_model: int                    = Field(default=512,   description="Input and output dim")
    num_tokens: int                 = Field(default=20000, description="Tokenizer's vocab size")
    dropout: float                  = Field(default=0.1,   description="Global dropout rate")

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> "BaseConfig":
        """Create config object from predefined config name"""
        if name not in name_to_config:
            # Search hf_config
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

        # Create config object
        conf_dict = deepcopy(conf_dict)
        # Update with any additional kwargs
        conf_dict.update(kwargs)
        return cls(**conf_dict)

    @classmethod
    def from_file(cls, path: str) -> "BaseConfig":
        """Load from config file"""
        import yaml
        with open(path) as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    @classmethod 
    def from_checkpoint(cls, checkpoint_dir: str) -> "BaseConfig":
        """Load config from checkpoint directory"""
        from pathlib import Path
        config_path = Path(checkpoint_dir) / "config.yaml"
        return cls.from_file(config_path)

