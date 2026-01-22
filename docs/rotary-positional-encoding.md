# Rotary Positional Encoding (RoPE)

## Paper
Rotary positional encoding injects relative position information by rotating query/key feature pairs by position-dependent angles. This encodes position through rotations (rather than addition) and yields relative-position behavior without adding learned position embeddings.

## Repo implementation
Implementation file: `boring_llm/nn/pe/registry.py`

- Compared to standard absolute positional embeddings (added to token embeddings), RoPE injects positions by rotating query/key features inside attention. It targets relative-position behavior without adding learned position vectors.
- `RotaryPositionalEncoding.__init__` computes `inv_freq` from `rope_base` and precomputes cached `cos`/`sin` values up to `max_seq_len`.
- `RotaryPositionalEncoding.apply(pos)` returns the cached `cos` and `sin` tensors for the requested positions.
- The returned values are meant to be applied inside attention to rotate query/key vectors (this repo stores the precomputed trigonometric terms).
