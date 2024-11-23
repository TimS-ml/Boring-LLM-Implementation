from typing import Optional, Tuple
from enum import Enum
from pydantic import BaseModel, Field

from boring_llm_base.base_config import BaseConfig


class AttentionType(Enum):
    SOFTMAX = "softmax"
    ENTMAX15 = "entmax15"  # Sparse softmax, potentially improving interpretability
    TOPK = "topk"          # If set, use top-k sparse attention and sets the rest to zero


class AttentionTypeConfig(BaseModel):
    type: AttentionType = Field(default=AttentionType.SOFTMAX, description="Type of attention mechanism")
    sparse_topk: int    = Field(default=10,                    description="Top-k value for sparse attention")


class QKNormConfig(BaseModel):
    enabled: bool  = Field(default=False, description="Whether to enable QK normalization")
    groups: int    = Field(default=1,     description="Number of groups for QK normalization")
    scale: float   = Field(default=10.0,  description="Scale factor for QK normalization")


class AttentionConfig(BaseConfig):
    # basic
    dim_head: Optional[int]         = Field(default=64,    description="Dimension of each attention head")
    num_heads: Optional[int]        = Field(default=8,     description="Number of attention heads")
    causal: Optional[bool]          = Field(default=False, description="Whether to apply a causal mask to attention weights")
    bias: Optional[bool]            = Field(default=False, description="Whether to use bias in qkv linear projections")

    # advanced
    num_mem_kv: Optional[int]       = Field(default=0,     description="Number of memory key/value pairs, concated to the input kv")
    talking_heads: Optional[bool]   = Field(default=False, description="Learned linear projections before and after the softmax")
    attn_on_attn: Optional[bool]    = Field(default=False, description="Modified Attention-on-attention mechanism")
    flash_attention: Optional[bool] = Field(default=False, description="Kernelized attention mechanism")
    rotary_pos_emb: Optional[bool]  = Field(default=False, description="RoPE positional embeddings")
    attn_type_config: AttentionTypeConfig = Field(default_factory=AttentionTypeConfig, description="Attention type configuration")
    qk_norm: QKNormConfig           = Field(default_factory=QKNormConfig, description="l2 normalization of qk before softmax")


# TODO
class CrossAttentionConfig(AttentionConfig):
    dim_context: Optional[int]      = Field(default=None,  description="Context dimension, defaults to same as input dim")
    shared_kv: bool                 = Field(default=False, description="Share key/value projections for memory efficiency")
    one_kv_head: bool               = Field(default=False, description="Use single head for key/values")
    kv_heads: Optional[int]         = Field(default=None,  description="Number of key/value heads, defaults to same as query heads")
    value_dim_head: Optional[int]   = Field(default=None,  description="Dimension of value heads, defaults to same as key dim")
    
    gate_values: bool               = Field(default=False, description="Use gating for aggregated values (from AlphaFold2)")
    gate_value_heads: bool          = Field(default=False, description="Per head gating of output values")
    swiglu_values: bool             = Field(default=False, description="Use SwiGLU activation for value gating")
    tensor_product: bool            = Field(default=False, description="Use tensor product attention")
    add_zero_kv: bool               = Field(default=False, description="Add zero attention key/value pair")

    class Config:
        validate_assignment = True