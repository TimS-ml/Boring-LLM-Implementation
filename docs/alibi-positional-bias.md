# ALiBi Positional Bias

## Paper
ALiBi (Attention with Linear Biases) adds a static, head-specific linear bias to attention logits based on relative position. It acts as a relative positional encoding and is often used to help transformers generalize to longer sequences than seen during training.

## Repo implementation
Implementation file: `boring_llm/nn/pe/registry.py`

- Compared to standard Transformers with absolute position embeddings, ALiBi adds a fixed distance-based bias directly to attention logits. It targets length extrapolation and relative-position inductive bias without learned position vectors.
- `AlibiPositionalEncoding._get_slopes(num_heads)` computes a per-head slope schedule.
- `AlibiPositionalEncoding.apply(pos)` builds a relative position matrix and multiplies by slopes to produce a bias tensor of shape `[num_heads, seq, seq]`.
