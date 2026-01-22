# Dynamic Position Bias (MLP)

## Paper
Dynamic position bias learns a function from relative distance to per-head attention bias using a small MLP. This produces a relative-position bias that can generalize beyond the training sequence length without relying on fixed positional embeddings.

## Repo implementation
Implementation file: `boring_llm/nn/pe/registry.py`

- Compared to fixed positional embeddings/biases, this learns a function from distance to bias (via an MLP). It targets length extrapolation by generating biases for distances beyond what was seen in training.
- `DynamicPositionBiasEncoding.__init__` builds a depth-`depth` MLP ending in a `num_heads` output.
- `apply(..., seq_len_q, seq_len_k)` constructs a relative-distance grid and computes biases by passing a distance range through the MLP.
- With `log_distance=True`, distances are log-scaled before being fed to the MLP.
