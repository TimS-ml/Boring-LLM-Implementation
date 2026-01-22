# LayerScale

- Paper: https://arxiv.org/abs/2103.17239

## Paper
LayerScale adds a learnable per-channel scaling vector to a residual branch output. Initializing the scale near zero (or a small value) can stabilize training for deep transformer-like networks.

## Repo implementation
Implementation file: `boring_llm/nn/connections/registry.py`

- Compared to a standard residual branch (added unscaled), this introduces a learnable per-channel multiplier on the branch output. It targets training stability for very deep stacks by controlling residual magnitudes early in training.
- `LayerScaleTransform` stores a learnable `gamma` parameter (per channel) initialized from `init_value` and `unit_offset`.
- `LayerScaleTransform.apply(x)` multiplies the layer output by `gamma` (with optional offset) before it is added to the residual stream.
