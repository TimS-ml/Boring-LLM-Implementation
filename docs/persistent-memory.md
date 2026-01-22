# Augmenting Self-attention with Persistent Memory

- Paper: https://arxiv.org/abs/1907.01470

## Paper
The paper proposes adding a small set of learned key/value vectors to every self-attention layer. These persistent memories are prepended to the attention K/V so every query can attend to them, improving capacity without increasing sequence length.

## Repo implementation
Implementation file: `boring_llm/nn/memory/memory.py`

- Compared to standard self-attention (keys/values come only from the current sequence), this prepends learned K/V slots that every token can attend to. It targets extra global capacity without increasing sequence length.
- `PersistentMemoryKV.__init__` creates learned `mem_k` / `mem_v` parameters with shape `[num_heads, num_mem_kv, dim_head]`.
- `PersistentMemoryKV.forward(k, v)` repeats the memory across the batch and concatenates it to `k` and `v` along the sequence axis (`dim=2`).
- Inputs are expected to be shaped like attention projections: `k, v` are `[batch, heads, seq, dim_head]`.
