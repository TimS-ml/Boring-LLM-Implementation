# Memory Tokens (Memory Transformers)

- Paper: https://arxiv.org/abs/2006.11527

## Paper
Memory Transformers introduce a set of learned “memory tokens” that are prepended to the input sequence. These tokens participate in attention like normal tokens and can serve as global scratchpad / aggregation slots across layers.

## Repo implementation
Implementation file: `boring_llm/nn/memory/memory.py`

- Compared to a standard Transformer input sequence, this prepends extra learned tokens that participate in attention like regular tokens. It targets providing dedicated global scratchpad/aggregation slots (and can help with stability/outliers depending on usage).
- `MemoryTokens` stores learnable `memory_tokens` of shape `[num_memory_tokens, dim]`.
- `MemoryTokens.forward(x)` repeats memory tokens across batch and concatenates them before the input sequence (`dim=1`).
- `MemoryTokens.remove_memory(x)` slices off the prepended memory tokens from the output (`x[:, num_memory_tokens:]`).
