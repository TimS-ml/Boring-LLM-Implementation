# 😑😑😑 Boring LLM 😑😑😑 

Inspired by [x-transformer](https://github.com/lucidrains/x-transformers) and [litgpt](https://github.com/Lightning-AI/litgpt/) and [gpt-fast](https://github.com/pytorch-labs/gpt-fast).

Implemented all components from scratch (normalization layers, CUDA acceleration, LLM training pipeline).

(WIP) Unified 40+ research papers into a single transformer implementation:
- Configure multiple optimizations simultaneously (LoRA, FlashAttention, RMSNorm, Memory Transformers) through simple config changes
- Easily modify architecture designs across 20+ mainstream LLM architectures (Deepseek, LLaMA, etc.)

(WIP) Developing CUDA acceleration modules.

## Docs

- Paper notes: `docs/relu-squared.md`
- Architecture / patterns: `docs/design-docs/ROADMAP.md`


# Install
```bash
pip install git+https://github.com/TimS-ML/Boring-Utils.git
git clone https://github.com/TimS-ml/Boring-LLM-Implementation.git && cd Boring-LLM-Implementation
pip install -e .
```


# Usage
## Simple Attention
```python
from boring_llm.tiny.tiny_base import TinyMultiHeadAttention

mha = TinyMultiHeadAttention(dim=EMBEDDING_DIM, n_head=N_HEAD, d_head=D_HEAD, causal=False)
x = torch.randn(BATCH_SIZE, BLOCK_SIZE, EMBEDDING_DIM)
out = mha(x)
```


## Complex Attention
```python
from boring_llm.nn.attention import BoringMultiHeadAttention 
from boring_llm.nn.attention.config import AttentionConfig, AttentionType

cfg = AttentionConfig(
    d_model=EMBEDDING_DIM,
    num_mem_kv=2,       # enable memory key-value
    attn_on_attn=True,  # attention on attention
    attn_type=AttentionType.TOPK  # sparse attention
)
boring_mha = BoringMultiHeadAttention(cfg)
output = boring_mha(x)
```


## Simple Encoder-Decoder Transformer
```python
from boring_llm.tiny.tiny_base import TinyEncDecTransformer

model = TinyEncDecTransformer(
    dim=EMBEDDING_DIM,
    # encoder
    enc_num_tokens=ENC_NUM_TOKENS,
    enc_n_layers=ENC_N_LAYER,
    enc_n_head=ENC_N_HEAD,
    enc_max_seq_len=BLOCK_SIZE,
    # decoder
    dec_num_tokens=DEC_NUM_TOKENS,
    dec_n_layers=DEC_N_LAYER,
    dec_n_head=DEC_N_HEAD,
    dec_max_seq_len=BLOCK_SIZE,
    # misc
    tie_token_emb=False,
    ffn_mul=FFN_MUL,
    dropout=DROPOUT
).to(device)

# Create test inputs
src = torch.randint(0, ENC_NUM_TOKENS, (BATCH_SIZE, BLOCK_SIZE)).to(device)
tgt = torch.randint(0, DEC_NUM_TOKENS, (BATCH_SIZE, BLOCK_SIZE)).to(device)

# Forward pass
logits = model(src, tgt)

# Test generation
tgt_start = torch.randint(0, DEC_NUM_TOKENS, (BATCH_SIZE, 1)).to(device)
generated = model.generate(src, tgt_start, seq_len=BLOCK_SIZE)
```


## Built-In Flag
```bash
VERBOSE=1 DEBUG=2 python boring_llm/tiny/pe/base_pe.py
```

Will print out the more detailed and colorful information about the model like this:
```
[2025-05-31 18:08:30] ===== test_positional_embedding_transformer -> Creating Absolute PE... =====
[2025-05-31 18:08:30] ***** AbsolutePositionalEncoding.__print_init_args__ -> Kwargs *****
    dim_model: 96
    max_seq_len: 128
    l2norm_embed: True
[2025-05-31 18:08:30] ===== test_positional_embedding_transformer -> Creating Fixed PE... =====
[2025-05-31 18:08:30] ***** FixedPositionalEncoding.__print_init_args__ -> Kwargs *****
    dim_model: 96
    max_seq_len: 128
[2025-05-31 18:08:30] ===== test_positional_embedding_transformer -> Creating No PE... =====
[2025-05-31 18:08:30] ===== test_positional_embedding_transformer -> Creating Rotary PE... =====
[2025-05-31 18:08:30] ***** RotaryPositionalEncoding.__print_init_args__ -> Kwargs *****
    dim_model: 96
    max_seq_len: 128
    rotary_percentage: 0.5
    rope_base: 10000
[2025-05-31 18:08:30] ===== test_positional_embedding_transformer -> Testing Absolute PE... =====
[2025-05-31 18:08:31] ===== test_positional_embedding_transformer -> Absolute PE output shape: torch.Size([8, 128, 128]) ✓ =====
[2025-05-31 18:08:31] ===== test_positional_embedding_transformer -> Testing Fixed PE... =====
[2025-05-31 18:08:31] ===== test_positional_embedding_transformer -> Fixed PE output shape: torch.Size([8, 128, 128]) ✓ =====
[2025-05-31 18:08:31] ===== test_positional_embedding_transformer -> Testing No PE... =====
[2025-05-31 18:08:31] ===== test_positional_embedding_transformer -> No PE output shape: torch.Size([8, 128, 128]) ✓ =====
[2025-05-31 18:08:31] ===== test_positional_embedding_transformer -> Testing Rotary PE... =====
[2025-05-31 18:08:31] ===== test_positional_embedding_transformer -> Rotary PE output shape: torch.Size([8, 128, 128]) ✓ =====
[2025-05-31 18:08:31] ===== test_positional_embedding_transformer -> All tests passed! 🎉 =====
[2025-05-31 18:08:31] ===== <module> -> Testing SimplifiedPETransformer... =====
[2025-05-31 18:08:31] ***** FixedPositionalEncoding.__print_init_args__ -> Kwargs *****
    dim_model: 96
    max_seq_len: 128
[2025-05-31 18:08:31] ===== <module> -> SimplifiedPETransformer output shape: torch.Size([8, 128, 128]) ✓ =====
```
