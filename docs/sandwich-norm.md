# Sandwich Normalization

## Paper
Sandwich normalization adds an extra normalization after a sub-layer (attention or MLP) but before adding the residual. This “norm-module-norm” structure can improve stability by ensuring the residual branch output is normalized before it is merged back into the main stream.

## Repo implementation
Implementation file: `boring_llm/nn/arch_variants/variants.py`

- Compared to standard residual wrapping (e.g. pre-norm or post-norm alone), this adds an additional normalization on the residual branch output before addition. It targets training stability by controlling the distribution of the branch output.
- `SandwichNorm` wraps an arbitrary module `fn` with `norm_pre` and `norm_post`.
- `SandwichNorm.forward(x, ...)` computes `x + norm_post(fn(norm_pre(x)))`.
