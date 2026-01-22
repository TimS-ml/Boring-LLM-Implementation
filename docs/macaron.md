# Macaron Network (FFN-Attn-FFN)

## Paper
Macaron-style transformer layers place the attention block between two half-step feedforward blocks. Splitting the feedforward into two scaled sub-steps can improve the dynamical behavior of deep transformer stacks while keeping the same overall computation pattern.

## Repo implementation
Implementation file: `boring_llm/nn/arch_variants/variants.py`

- Compared to a standard Transformer block (Attention + one FFN), this uses two half-step FFNs around attention. It targets improved layer dynamics / gradient flow while keeping a similar overall compute profile.
- `MacaronNet` applies `FFN * 0.5 -> Attention -> FFN * 0.5`, each wrapped with its own normalization.
- `_scale_ff(ff_fn, 0.5)` wraps the feedforward module so its output is scaled by `0.5`.
- `forward(x, ...)` uses residual additions at each stage (`x = x + ...`).
