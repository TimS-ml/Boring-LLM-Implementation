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
cprint(mask, c='green')
print()


def test_multihead_attention_1():
    head_size = HEAD_SIZE  # (B, T, C) -> (B, T, head_size)

    # Calling the attention function
    att = MultiHeadAttention(d_model=C, num_heads=head_size)
    output, wei = att(x, x, x)

    cprint(output.shape, c='blue')
    cprint(wei.shape, c='blue')


def test_multihead_attention_2():
    '''
    Q, K with different seq length: 
    - no mask
    - 2d mask
    - 3d mask
    '''
    head_size = HEAD_SIZE  # (B, T, C) -> (B, T, head_size)

    q = torch.randn(B, T_DEC, C)
    k = torch.randn(B, T_ENC, C)
    v = torch.randn(B, T_ENC, C)

    # Generate a lower triangular matrix for causal masking
    tril = torch.tril(torch.ones(T_DEC, T_ENC))
    mask = tril.float().masked_fill(tril == 0, float('-inf'))

    print('No mask:')
    # Calling the attention function without mask
    att = MultiHeadAttention(d_model=C, num_heads=head_size)
    output, wei = att(q, k, v)

    cprint(output.shape, c='blue')
    cprint(wei.shape, c='blue')
    print()


    print('2d mask:')
    # Calling the attention function with mask
    output_2d, wei_2d = att(q, k, v, attn_mask=mask)
    # output, wei = att(q, k, v, attn_mask=tril)

    cprint(output_2d.shape, c='blue')
    cprint(wei_2d.shape, c='blue')
    print()

    print('3d mask:')
    new_mask = mask.expand(B, T_DEC, T_ENC)
    output_3d, wei_3d = att(q, k, v, attn_mask=new_mask)

    cprint(output_3d.shape, c='blue')
    cprint(wei_3d.shape, c='blue')
    print()


cprint("test_multihead_attention", c='normal')
# test_multihead_attention_1()
test_multihead_attention_2()

