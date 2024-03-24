import torch
from boring_transformer.utils import cprint
from boring_transformer.pe import *


# Generating random inputs
B, T, C = 4, 8, 32  # batch size, time steps (seq length), channels
x = torch.rand(B, T, C)
# x = torch.rand(B, T)
tril = torch.tril(torch.ones(T, T))  # for mask

def test_simple_sinusoidal_positional_encoding():
    pe = SimpleSinusoidalPositionalEncoding(C, max_len=T)

    assert pe.shape == (T, C)
    
    cprint(pe)
    cprint(pe.shape)

def test_sinusoidal_positional_encoding():
    pe = SinusoidalPositionalEncoding(C, dropout=0.1, max_len=T)
    y = pe(x)

    assert y.shape == x.shape, "Output shape should match input shape."
    # assert y.sum() >= x.sum(), "Expecting learned positional encoding to increase the sum of elements. Even after Dropout."

    cprint(y)
    cprint(y.shape)
    # cprint(y.sum(), ' ', x.sum())

def test_learned_positional_encoding():
    pe = LearnedPositionalEncoding(C, dropout=0.1, max_len=T)
    y = pe(x)

    assert y.shape == x.shape, "Output shape should match input shape."
    # assert y.sum() >= x.sum(), "Expecting learned positional encoding to increase the sum of elements. Even after Dropout."

    cprint(y)
    cprint(y.shape)
    # cprint(y.sum(), ' ', x.sum())

# test_simple_sinusoidal_positional_encoding()
test_sinusoidal_positional_encoding()
# test_learned_positional_encoding()

