# 😑😑😑 Boring LLM 😑😑😑 

(WIP) A simple language model that generates boring text. Implemented everything from scratch (norm layer, cuda acc, llm training).


# Install
```bash
git clone https://github.com/TimS-ml/Boring-LLM-Implementation.git && cd Boring-LLM-Implementation
pip install -e .
```


# Use
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

```python
import torch
from boring_utils.utils import cprint
from boring_transformer.boring_transformer import BoringEncoderBlock

input_seq = torch.randn(batch_size, max_seq_len, d_model)
output_seq = encoder_block(input_seq)
cprint(output_seq.shape == (batch_size, max_seq_len, d_model))
```


# Demo Data and Preprocessing
From nanoGPT, improved downloading for openwebtext.

