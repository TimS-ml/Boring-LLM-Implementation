# 😑😑😑 Boring LLM 😑😑😑 

(WIP) A simple language model that generates boring text. Implemented everything from scratch (norm layer, cuda acc, llm training).


# Install
```bash
git clone https://github.com/TimS-ml/Boring-Utils.git && cd Boring-Utils
pip install -e .

git clone https://github.com/TimS-ml/Boring-LLM-Implementation.git && cd Boring-LLM-Implementation
pip install -e .
```


# Usage
## Attention
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


## Transformer Block
```python
import torch
from boring_utils.utils import cprint
from boring_transformer.legacy.boring_transformer import BoringEncoderBlock

input_seq = torch.randn(batch_size, max_seq_len, d_model)
output_seq = encoder_block(input_seq)
cprint(output_seq.shape == (batch_size, max_seq_len, d_model))
```


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

