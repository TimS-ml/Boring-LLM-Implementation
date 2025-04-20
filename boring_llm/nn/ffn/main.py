import torch
import torch.nn as nn
import torch.nn.functional as F
# from einops import rearrange

from torch import Tensor
from typing import Optional, Tuple, Union, List, Literal, Any

from boring_llm.nn.ffn.config import FeedForwardConfig, create_ffn_config
from boring_llm.nn.ffn.base import FeedForwardTransform
from boring_llm.nn.ffn.factory import FeedForwardFactory
from boring_llm.nn.activation.config import ActivationConfig
from boring_llm.nn.activation.main import get_activation

from boring_utils.utils import cprint, tprint
from boring_utils.helpers import DEBUG


class BoringFeedForward(nn.Module):
    """
    Main feed-forward network module that uses strategy pattern
    to support different types of feed-forward implementations
    """
    def __init__(self, config: FeedForwardConfig):
        super().__init__()
        self.config = config
        
        dim = config.dim_model
        dim_out = config.ffn_dim_out or dim
        mult = config.mult_dim
        inner_dim = int(dim * mult) if config.inner_dim is None else config.inner_dim

        ffn_type = config.type
        ffn_post_type = config.post_type
        
        if DEBUG: cprint(config)
        transform_args = create_ffn_config(ffn_type)(
            dim_model=dim,
            inner_dim=inner_dim,
            activation=config.activation,
            no_bias=config.no_bias
        )
        factory_args = transform_args.model_dump(exclude={"type", "post_type"})
        # if ffn_type == "glu": factory_args["mult_bias"] = config.mult_bias 
        if ffn_type == "glu" and hasattr(config, "mult_bias"):
            factory_args["mult_bias"] = config.mult_bias
        if DEBUG: cprint(factory_args)
        self.ffn_transform = FeedForwardFactory.create(
            ffn_type=ffn_type,
            **factory_args
        )
        trans_dim_out = self.ffn_transform.output_dim
        
        post_args = create_ffn_config(ffn_post_type)(
            dim_model=dim_out,
            inner_dim=trans_dim_out,
            dropout=config.dropout,
            post_act_ln=config.post_act_ln,
            no_bias=config.no_bias,
            zero_init_output=config.zero_init_output
        )
        post_factory_args = post_args.model_dump(exclude={"type", "post_type"})
        if DEBUG: cprint(post_factory_args)
        self.post_processor = FeedForwardFactory.create(
            ffn_type=ffn_post_type,
            **post_factory_args
        )

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        """
        Apply feed-forward transformation to input tensor
        
        Args:
            x: Input tensor of shape [batch, seq_len, dim]
            **kwargs: Additional arguments passed to the transformation
            
        Returns:
            Transformed tensor
        """
        transformed = self.ffn_transform.apply(x, **kwargs)
        return self.post_processor.apply(transformed)


if __name__ == "__main__":
    from boring_llm.base.tiny_config import *
    
    tprint("Standard FFN")
    ffn_type = "standard"
    ffn_args = create_ffn_config(ffn_type)(
        dim_model=EMBEDDING_DIM,
        mult_dim=4,
        post_type="post_standard",
        activation=nn.GELU  # using callable
    )
    ffn = BoringFeedForward(ffn_args)
    x = torch.randn(2, 3, EMBEDDING_DIM)
    y = ffn(x)
    print(f"Standard FFN output shape: {y.shape}")

    tprint("GLU FFN")
    ffn_type = "glu"
    ffn_args = create_ffn_config(ffn_type)(
        dim_model=EMBEDDING_DIM,
        mult_dim=2,
        post_type="post_standard",
        mult_bias=False,
        activation="silu"  # using str
    )
    ffn = BoringFeedForward(ffn_args)
    y = ffn(x)
    print(f"GLU FFN output shape: {y.shape}") 
