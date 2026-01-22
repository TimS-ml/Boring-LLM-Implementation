# Relative Position Bias (T5-style)

## Paper
T5-style relative position bias adds a learned bias to attention logits based on relative distance between tokens. Distances are bucketed so nearby positions use exact buckets while farther positions share logarithmic buckets.

## Repo implementation
Implementation file: `boring_llm/nn/pe/registry.py`

- Compared to no positional bias (or absolute embeddings), this adds a learned bias term to attention logits based on bucketed relative distance. It targets stronger relative-position generalization while keeping parameter count small.
- `RelativePositionBiasEncoding` stores a `nn.Embedding(num_buckets, num_heads)` used to map bucket ids to per-head biases.
- `_relative_position_bucket(relative_position, ...)` implements the exact+log bucketization scheme.
- `apply(..., seq_len_q, seq_len_k)` builds the relative position matrix, buckets it, looks up biases, and returns `[num_heads, seq_len_q, seq_len_k]` (scaled by `scale`).
