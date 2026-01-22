# Dynamic Tanh Normalization

- Paper: https://arxiv.org/abs/2503.10622

## Paper
Dynamic Tanh normalization applies a tanh nonlinearity with learnable scaling so the transformation can adapt during training. The bounded tanh output can improve stability while still allowing learnable amplitude via per-channel parameters.

## Repo implementation
Implementation file: `boring_llm/nn/norm/registry.py`

- Compared to standard normalization layers (e.g. LayerNorm/RMSNorm), this applies a bounded `tanh` transform with learnable scaling and affine parameters. It targets stability by limiting activations while preserving learnable amplitude/shift.
- `DynamicTanhTransform` learns a scalar `pre_tanh_scale` plus per-channel `gamma` and `beta`.
- `DynamicTanhTransform.apply(x)` computes `(x * pre_tanh_scale).tanh() * gamma + beta` (with optional unit offsets).
