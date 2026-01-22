# Contextual Position Encoding (CoPE)

- Paper: https://arxiv.org/abs/2405.18719

## Paper
CoPE computes positional information from attention patterns rather than fixed indices. It estimates a contextual “count” of relevant tokens using attention-derived gates, enabling position signals that depend on the current context.

## Repo implementation
Implementation file: `boring_llm/nn/pe/registry.py`

- Compared to standard positional encodings that depend only on token index, CoPE computes positions from attention-derived gates. It targets “contextual counting” (positions depend on which tokens are relevant) rather than fixed absolute/relative indices.
- `CoPEEncoding.apply(..., query=..., attn_logits=...)` derives gating values from `attn_logits.sigmoid()` and uses cumulative sums to compute contextual positions.
- It projects the query with `pos_emb` to get per-position logits, then uses interpolation (hard or soft one-hot) to turn computed positions into an added bias term.
- If `talking_heads=True`, it applies a head-mixing 1×1 conv over attention logits before computing gates.
