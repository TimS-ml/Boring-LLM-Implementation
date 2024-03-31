import torch
from boring_utils.utils import cprint
from boring_nn.attention import *


B, T, C = 4, 8, 32  # batch size, time steps (seq length), channels
x = torch.rand(B, T, C)

# Generate a lower triangular matrix for causal masking
# Then convert the lower triangular matrix to float; positions to attend to are marked as 0, others as -inf
tril = torch.tril(torch.ones(T, T))
mask = tril.float().masked_fill(tril == 0, float('-inf'))
cprint(mask)

def test_scaled_dot_product_attention():
    # head_size = 1  # (B, T, C) -> (B, T, head_size)
    # query = nn.Linear(C, head_size, bias=False)
    # key = nn.Linear(C, head_size, bias=False)
    # value = nn.Linear(C, head_size, bias=False)
    # q, k, v = query(x), key(x), value(x)

    # cprint(x.shape)
    # cprint(q.shape)

    q = torch.randn(T, C)
    k = torch.randn(T, C)
    v = torch.randn(T, C)

    # Calling the attention function
    att = ScaledDotProductAttention()
    output, wei = att(q, k, v, attn_mask=tril)

    cprint(output.shape)
    cprint(wei.shape)
    # cprint(attn_weights)

def test_multihead_attention():
    head_size = 8  # (B, T, C) -> (B, T, head_size)

    # Calling the attention function
    att = MultiHeadAttention(d_model=C, num_heads=head_size)
    output, wei = att(x, x, x)

    cprint(output.shape)
    cprint(wei.shape)


print("test_scaled_dot_product_attention")
test_scaled_dot_product_attention()

print("test_multihead_attention")
test_multihead_attention()

