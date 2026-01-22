# Explicit Sparse Transformer (Top-k Attention)

- Paper: https://arxiv.org/abs/1912.11637

## Paper
The paper sparsifies attention by keeping only the top-k attention logits per query and masking the rest before softmax. This produces a sparse attention pattern with a single extra hyperparameter (`k`).

## Repo implementation
Implementation file: `boring_llm/nn/attention/registry.py`

- Compared to standard dense softmax attention, this masks all but the per-query top-k logits before softmax. It targets efficiency and inductive bias toward sparse attention patterns.
- `SparseTopKAttention.apply(q, k, v, ...)` computes scaled dot-product logits and then selects `topk` values along the key dimension.
- It builds a boolean `sparse_mask` via `scatter_` and masks out non-topk logits before softmax.
- With `straight_through=True`, it uses a straight-through trick to keep dense gradients while using sparse logits in the forward pass.
