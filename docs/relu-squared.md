# ReLU² (ReLU Squared)

- Paper: https://arxiv.org/abs/2109.08668

## What the paper does

Proposes using a simple activation for transformer feedforward layers: `ReLU(x)^2`, discovered via neural architecture search, as a drop-in alternative to more complex activations (e.g., GELU).

## Implementation in this repo

Implementation file: `boring_llm/nn/activation/activation.py`

- `ReluSquared.forward(x)` returns `F.relu(x) ** 2`
- No extra parameters (empty constructor)
