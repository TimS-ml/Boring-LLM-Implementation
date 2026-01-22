# Talking-Heads Attention

- Paper: https://arxiv.org/abs/2003.02436

## Paper
Talking-Heads attention mixes information across attention heads by applying learned linear projections to the attention matrix, either before softmax, after softmax, or both. This lets heads communicate through learned head-to-head mixing.

## Repo implementation
Implementation file: `boring_llm/nn/attention/registry.py`

- Compared to standard multi-head attention (independent heads combined only at the output projection), this adds explicit head-to-head mixing on the attention matrix. It targets better coordination / information sharing across heads.
- `TalkingHeads(num_heads, pre_softmax=True, post_softmax=True)` creates 1×1 conv projections over the head dimension.
- `TalkingHeads.forward(attn, pre=True)` applies the pre-softmax mixing if enabled; `pre=False` applies post-softmax mixing.
- This module is provided as a building block and must be called by an attention implementation that has access to attention logits / weights.
