# Cosine Similarity Attention

- Paper: https://arxiv.org/abs/2010.04245

## Paper
Cosine similarity attention normalizes queries and keys before computing attention scores, replacing dot products with cosine similarity. This reduces numerical issues (e.g. overflow) and can improve training stability.

## Repo implementation
Implementation file: `boring_llm/nn/attention/registry.py`

- Compared to standard scaled dot-product attention, this variant L2-normalizes `q`/`k` and uses cosine similarity for logits. It mainly targets numerical stability (overflow/instability from large dot products) while keeping the rest of attention unchanged.
- `CosineSimilarityAttention.apply(q, k, v, ...)` L2-normalizes `q` and `k` with `F.normalize(..., dim=-1)`.
- It computes similarity with `einsum('b h i d, b h j d -> b h i j', q, k) / temperature`.
- It supports causal masking, optional external masks, and returns `softmax(sim) @ v` with dropout applied to attention weights.
