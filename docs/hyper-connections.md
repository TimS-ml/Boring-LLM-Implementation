# Hyper-Connections (Multi-Stream Residual Mixing)

- Paper: https://arxiv.org/abs/2409.19606

## Paper
Hyper-connections introduce multiple residual streams and learn to dynamically mix between them. Instead of a single residual path, each layer can route information across streams via learned mixing matrices, increasing expressivity and improving gradient flow.

## Repo implementation
Implementation file: `boring_llm/nn/connections/registry.py`

- Compared to a standard Transformer (single residual stream per layer), this maintains multiple residual streams and learns dynamic mixing between them. It targets more expressive routing and improved gradient flow.
- `HyperConnectionTransform.prepare(residuals)` reshapes residual streams to `[batch, seq, streams, dim]` and computes dynamic mixing weights from normalized residuals.
- It produces mixing matrices (`alpha`) and output mixing coefficients (`beta`) and returns a branch input plus updated residual streams.
- `HyperConnectionTransform.apply(x, residuals, beta=...)` merges the branch output back into the residual streams using `einsum`.
