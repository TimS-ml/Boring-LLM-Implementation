# Data-Dependent ALiBi

- Paper: https://openreview.net/forum?id=q2Lnyegkr8

## Paper
Data-dependent ALiBi replaces static distance-based slopes with forget gates predicted from the input. The resulting position bias is computed from these gates so the model can adapt its effective decay pattern based on the current sequence content.

## Repo implementation
Implementation file: `boring_llm/nn/pe/registry.py`

- Compared to standard ALiBi (static slopes based only on distance), this predicts per-head forget gates from the input and derives a bias matrix from them. It targets adapting the effective decay/recency bias to the current sequence content.
- `DataDependentAlibiEncoding.__init__` builds `to_forget_gates`, producing per-head log-forget-gates from the input sequence.
- `apply(pos, x=...)` uses `torch.logcumsumexp` to construct a position-bias matrix from the predicted gates.
- For causal mode, it masks future positions after computing the bias.
