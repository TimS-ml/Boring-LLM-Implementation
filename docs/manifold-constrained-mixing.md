# Manifold Constrained Mixing (Sinkhorn Constraint)

- Paper: https://arxiv.org/abs/2512.24880

## Paper
The paper constrains learned mixing matrices to lie on a well-behaved manifold (e.g. doubly-stochastic matrices). Enforcing these constraints (via Sinkhorn normalization) can stabilize training and make multi-stream mixing more robust.

## Repo implementation
Implementation file: `boring_llm/nn/connections/registry.py`

- Compared to unconstrained mixing weights, this constrains mixing matrices toward doubly-stochastic structure via Sinkhorn normalization. It targets more stable / well-conditioned multi-stream mixing.
- `sinkhorn(t, iters)` alternates L1-normalization across rows and columns to produce a doubly-stochastic matrix.
- `HyperConnectionTransform.prepare(...)` reshapes the residual mixing block to `[streams, streams]` and applies `sinkhorn(...)` before using it as part of `alpha`.
- The constrained mixing is applied only to the residual-stream mixing portion (not the input-view mixing).
