# Derf Normalization (erf-based)

- Paper: https://arxiv.org/abs/2512.10938

## Paper
Derf normalization uses the Gaussian error function (`erf`) as a smooth bounded nonlinearity, with learnable scaling and shifting before and after the `erf`. This provides a tunable, saturation-bounded transform that can be used as a normalization/activation-style component.

## Repo implementation
Implementation file: `boring_llm/nn/norm/registry.py`

- Compared to standard normalization layers, this uses a bounded `erf` nonlinearity with learnable pre/post affine parameters. It targets smoother, saturation-bounded behavior that may improve training stability.
- `DerfTransform` learns pre-activation parameters `alpha` and `s`, and post-activation parameters `gamma` and `beta`.
- `DerfTransform.apply(x)` computes `torch.erf(x * (alpha + offset) + s) * (gamma + offset) + beta`.
