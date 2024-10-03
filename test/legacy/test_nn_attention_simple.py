import torch
from boring_utils.utils import cprint
from boring_nn.attention import *

B, T, C = 4, 8, 36  # batch size, time steps (seq length), channels
T_ENC, T_DEC = 10, T  # time steps (seq length) for encoder and decoder
HEAD_SIZE = 6  # assert d_model % num_heads == 0

x = torch.rand(B, T, C)

# Generate a lower triangular matrix for causal masking
# Then convert the lower triangular matrix to float; positions to attend to are marked as 0, others as -inf
tril = torch.tril(torch.ones(T, T))
mask = tril.float().masked_fill(tril == 0, float('-inf'))
# cprint(mask, c='green')
# print()

def test_simple_attention_1():
    q = torch.randn(T, C)
    k = torch.randn(T, C)
    v = torch.randn(T, C)
    # q = torch.randn(B, T, C)
    # k = torch.randn(B, T, C)
    # v = torch.randn(B, T, C)

    # Calling the attention function
    output, wei = SimpleScaledDotProductAttention(q, k, v, attn_mask=mask)

    cprint(output.shape, c='blue')
    cprint(wei.shape, c='blue')
    print()

    casual_output, casual_wei = SimpleScaledDotProductAttention(q, k, v, is_causal=True)

    cprint(casual_output.shape, c='blue')
    cprint(casual_wei.shape, c='blue')
    # cprint(attn_weights)


def test_simple_attention_2():
    head_size = HEAD_SIZE  # (B, T, C) -> (B, T, head_size)
    query = nn.Linear(C, head_size, bias=False)
    key = nn.Linear(C, head_size, bias=False)
    value = nn.Linear(C, head_size, bias=False)
    q, k, v = query(x), key(x), value(x)

    # Calling the attention function
    output, wei = SimpleScaledDotProductAttention(q, k, v, attn_mask=mask)

    cprint(output.shape, c='blue')
    cprint(wei.shape, c='blue')
    print()

    casual_output, casual_wei = SimpleScaledDotProductAttention(q, k, v, is_causal=True)

    cprint(casual_output.shape, c='blue')
    cprint(casual_wei.shape, c='blue')
    # cprint(attn_weights)


def test_simple_attention_3():
    q = torch.randn(B, T_DEC, C)
    k = torch.randn(B, T_ENC, C)
    v = torch.randn(B, T_ENC, C)
    tril = torch.tril(torch.ones(T_DEC, T_ENC))
    mask = tril.float().masked_fill(tril == 0, float('-inf'))

    # Calling the attention function
    output, wei = SimpleScaledDotProductAttention(q, k, v, attn_mask=mask)

    cprint(output.shape, c='blue')
    cprint(wei.shape, c='blue')
    print()

    casual_output, casual_wei = SimpleScaledDotProductAttention(q, k, v, is_causal=True)

    cprint(casual_output.shape, c='blue')
    cprint(casual_wei.shape, c='blue')
    # cprint(attn_weights)


def test_scaled_dot_product_attention():
    # head_size = 1  # (B, T, C) -> (B, T, head_size)
    # query = nn.Linear(C, head_size, bias=False)
    # key = nn.Linear(C, head_size, bias=False)
    # value = nn.Linear(C, head_size, bias=False)
    # q, k, v = query(x), key(x), value(x)

    # cprint(x.shape, c='blue')
    # cprint(q.shape, c='blue')

    q = torch.randn(T, C)
    k = torch.randn(T, C)
    v = torch.randn(T, C)

    # Calling the attention function
    att = ScaledDotProductAttention()
    output, wei = att(q, k, v, attn_mask=tril)

    cprint(output.shape, c='blue')
    cprint(wei.shape, c='blue')
    # cprint(attn_weights)


cprint("test_simple_attention", c='normal')
#  test_simple_attention_1()
#  test_simple_attention_2()
test_simple_attention_3()
print()

# cprint("test_scaled_dot_product_attention", c='normal')
# test_scaled_dot_product_attention()
# print()

