import torch
from boring_utils.utils import cprint, get_layers
from boring_utils.helpers import *

from boring_nn import attention
from boring_nn import pe

# Get all layers in the module
attn_layers = get_layers(attention)
pe_layers = get_layers(pe)

cprint(attn_layers, c='green')
cprint(pe_layers, c='green')


def test_get_layers_pe():
    # Generating random inputs
    B, T, C = 4, 8, 32  # batch size, time steps (seq length), channels
    x = torch.rand(B, T, C)
    # x = torch.rand(B, T)
    # tril = torch.tril(torch.ones(T, T))  # for mask

    pe = pe_layers['SinusoidalPositionalEncoding'](C, dropout=0.1, max_len=T)
    y = pe(x)

    assert y.shape == x.shape, "Output shape should match input shape."
    
    cprint(y)
    cprint(y.shape)


test_get_layers_pe()

