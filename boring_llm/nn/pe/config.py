from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional, Union

from boring_llm.base.base_config import BaseConfig

class PositionalEncodingType(str, Enum):
    NONE = "none"
    ABSOLUTE = "absolute"
    FIXED = "fixed"
    ROTARY = "rotary"
    ALIBI = "alibi"


class PositionalEncodingConfig(BaseConfig):
    """Configuration for positional encoding"""
    type: PositionalEncodingType = Field(
        default=PositionalEncodingType.FIXED,
        description="Type of positional encoding to use"
    )
    max_seq_len: int = Field(
        default=1024,
        description="Maximum sequence length for positional embeddings"
    )
    dim_model: Optional[int] = Field(
        default=None,
        description="Model dimension (if None, uses d_model from parent config)"
    )
    l2norm_embed: bool = Field(
        default=False,
        description="Whether to L2 normalize embeddings"
    )
    
    # Rotary PE specific
    rotary_percentage: float = Field(
        default=1.0,
        description="Percentage of dimensions to apply rotary encoding to"
    )
    
    # ALiBi specific
    alibi_num_heads: Optional[int] = Field(
        default=None,
        description="Number of attention heads for ALiBi"
    )