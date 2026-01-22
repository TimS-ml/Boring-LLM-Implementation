# Residual Attention

- Paper: https://arxiv.org/abs/2012.11747

## Paper
Residual attention carries attention information across layers by residualizing attention scores (or related attention-state) instead of recomputing them independently at every layer. This can improve optimization and performance in deep transformer stacks.

## Repo implementation
Implementation file: `boring_llm/nn/arch_variants/variants.py`

- Compared to standard Transformers that compute attention independently in each layer, residual attention reuses/adds attention-state across layers. It targets optimization in deep stacks by providing a cross-layer shortcut for attention information (note: this repo’s wrapper is not fully wired to accumulate attention matrices yet).
- `ResidualAttention` wraps an attention module `fn` and provides a `residual_attn` slot for cross-layer attention state.
- `ResidualAttention.forward(x, return_attn=...)` currently calls `fn(x, ...)` and returns `(out, residual_attn)` when requested.
- The code notes that a full implementation would require access to attention matrices to accumulate/residualize them across layers.
