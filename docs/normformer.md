# NormFormer (Extra Normalization Points)

## Paper
NormFormer adds extra normalization points within transformer layers (beyond the standard pre/post norm layouts). The goal is improved stability, especially for deep networks, by normalizing intermediate signals more aggressively.

## Repo implementation
Implementation file: `boring_llm/nn/arch_variants/variants.py`

- Compared to standard pre/post norm layouts, this adds extra normalization points around the wrapped sub-layer. It targets improved stability in deeper networks by normalizing more intermediate signals.
- `Normformer.forward(x, ...)` applies `norm_in`, calls the wrapped module, then applies `norm_out` and adds the residual (`x + norm_out(out)`).
- If the wrapped module exposes `to_q`, `to_k`, `to_v`, the wrapper constructs `norm_q`, `norm_k`, and `norm_v` (but this repo does not currently wire them into attention internals).
