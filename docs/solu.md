# SoLU (Softmax Linear Unit)

## Paper
SoLU replaces standard pointwise activations with a softmax-based gating: each feature is weighted by its softmax probability and then normalized. The intent is to introduce competition between features while keeping the transformation simple.

## Repo implementation
Implementation file: `boring_llm/nn/activation/activation.py`

- Compared to standard pointwise activations (e.g. GELU/ReLU), this gates each feature by its softmax probability and then normalizes. It targets feature competition and more controlled activations within the FFN.
- `SoLU.forward(x)` computes `x.softmax(dim=-1) * x` and then applies `nn.LayerNorm`.
- `SoLU.__init__(dim)` creates an internal `LayerNorm(dim)` used after the softmax gating.
