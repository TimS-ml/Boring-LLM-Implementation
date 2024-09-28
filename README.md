# 😑😑😑 Boring LLM 😑😑😑 

(WIP) A simple language model that generates boring text. Implemented everything from scratch (norm layer, cuda acc, llm training).


# Install
```bash
pip install git+https://github.com/TimS-ML/Boring-Utils.git
git clone https://github.com/TimS-ml/Boring-LLM-Implementation.git && cd Boring-LLM-Implementation
pip install -e .
```


# Usage
## Casual Attention
```python
import torch
from boring_utils.utils import cprint
from boring_nn.attention import MultiHeadAttention


B, T, C = 4, 8, 32  # batch size, time steps (seq length), channels
x = torch.rand(B, T, C)

tril = torch.tril(torch.ones(T, T))
mask = tril.float().masked_fill(tril == 0, float('-inf'))
cprint(mask)

# Calling the attention function
att = MultiHeadAttention(d_model=C, num_heads=8)
output = att(x, x, x)

cprint(output.shape)
```


## Complex Attention
```python
from boring_nn.attention import BoringAttention 
from boring_nn.attention.core import AttentionConfig, AttentionType

# ... same as above

# Calling the attention function
cfg = AttentionConfig(
    d_model=C,
    num_mem_kv=2,       # enable memory key-value
    attn_on_attn=True,  # attention on attention
    attn_type=AttentionType.TOPK  # sparse attention
)
att = MultiHeadAttention(d_model=C, num_heads=8)
output = att(x, x, x)

cprint(output.shape)
```


## Transformer Block (WIP)
```python
import torch
from boring_utils.utils import cprint
from boring_transformer.core import TransformerLayersConfig, TransformerLayerWrapConfig
from boring_nn.attention.core import AttentionConfig, AttentionType

config = TransformerLayersConfig(
    d_model=512,
    depth=6,
    num_heads=8,
    causal=True,
    layer_config=TransformerLayerWrapConfig(
        attention=AttentionConfig(
            dim_head=64,
            dropout=0.1
        ),
        ffn_dim=2048
    )
)
```


## LLM Block (WIP)


## Built-In DEBUG Flag
```bash
DEBUG=1 python -m test_legacy_transformer_encoder-block
```

Will print out the more detailed and colorful information about the model like this:
```
LayerNorm -> self.normalized_shape:
torch.Size([512])
LayerNorm -> self.elementwise_affine:
True
FeedForward -> self.is_gated:
False
FeedForward -> self.activation:
ReLU()
encoder_block_with_mask -> attn_mask.shape:
torch.Size([3, 7, 7])
```


# Demo Data and Preprocessing
From nanoGPT, improved downloading for openwebtext.

