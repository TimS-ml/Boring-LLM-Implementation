# Polar Positional Encoding (PoPE)

- Paper: https://arxiv.org/abs/2509.10534

## Paper
PoPE encodes position using polar coordinates: a phase based on sinusoidal frequencies (similar to rotary ideas) and a magnitude that is constrained to be non-negative. This polar formulation aims to improve length extrapolation by giving position signals a different inductive bias than standard additive embeddings.

## Repo implementation
Implementation file: `boring_llm/nn/pe/registry.py`

- Compared to standard additive sinusoidal/learned positional embeddings, this applies a polar-coordinate transform and uses per-head learned phase shifts. It targets improved length extrapolation by changing the inductive bias of how positions are represented.
- `PolarPositionalEncoding.apply(pos, offset=...)` returns `(freqs, bias)` where `bias` is a per-head learned phase shift clamped to `[-2π, 0]`.
- `apply_polar_pos_emb(t, freqs)` converts inputs to non-negative magnitudes with `F.softplus` and then concatenates `(r*cos(θ), r*sin(θ))`.
- `PolarPositionalEncoding.apply_to_qk(q, k, freqs, bias)` applies the polar embedding to queries and keys, adding the learned bias to key frequencies.
