from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional, Tuple


class BaseConfig(BaseModel):
    d_model: Optional[int]          = Field(default=512,   description="Input and output dim, usually d_model=dim_head*num_heads")
    num_tokens: Optional[int]       = Field(default=20000, description="Tokenizer's vocab size")
    dropout: Optional[float]        = Field(default=0.1,   description="Global dropout rate")
