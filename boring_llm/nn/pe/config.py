from pydantic import Field, field_validator, create_model
from typing import Optional, Type

from boring_llm.base.base_config import BaseConfig
from boring_llm.nn.pe.factory import PositionalEncodingFactory, PositionalEncodingConfigFactory


class PositionalEncodingConfig(BaseConfig):
    """Configuration for positional encoding"""
    type: str = Field(default="fixed")
    
    @field_validator('type')
    def validate_type(cls, v):
        if not PositionalEncodingFactory.is_valid_type(v):
            valid_types = PositionalEncodingFactory.get_available_types()
            raise ValueError(f"Invalid positional encoding type: {v}. Valid types: {valid_types}")
        return v

    max_seq_len: int = Field(
        default=1024,
        description="Maximum sequence length for positional embeddings"
    )
    # dim_model: Optional[int] = Field(
    #     default=None,
    #     description="Model dimension (if None, uses d_model from parent config)"
    # )


def create_pe_config(pe_type: str) -> Type[PositionalEncodingConfig]:
    fields = {
        "type": (str, Field(default=pe_type, const=True)),
        "max_seq_len": (int, Field(default=1024)),
        "dim_model": (Optional[int], Field(default=None)),
    }
    
    type_fields = PositionalEncodingConfigFactory.get_config_fields(pe_type)
    fields.update(type_fields)
    
    return create_model(f"{pe_type.capitalize()}PositionalEncodingConfig", 
                       __base__=PositionalEncodingConfig, 
                       **fields)