# Shifted Tokens

- Paper: https://arxiv.org/abs/2106.07477

## Paper
Shifted tokens improve sequence modeling by shifting parts of the feature channels along the sequence dimension. By giving different channel groups slightly different temporal alignment, the model can more easily represent local causal relationships.

## Repo implementation
Implementation file: `boring_llm/nn/connections/registry.py`

- Compared to a standard Transformer block (no temporal/channel shifting), this deterministically shifts channel groups along sequence length. It targets better local inductive bias and can improve optimization/convergence in some settings.
- `ShiftTokensTransform` splits the feature dimension into segments and shifts each segment by a configured amount.
- `ShiftTokensTransform.apply(x, mask=...)` uses the helper `shift(t, amount, mask)` to pad/crop along the sequence dimension.
- The shifted segments are concatenated back together with any remaining unshifted features.
