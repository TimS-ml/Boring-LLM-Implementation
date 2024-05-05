import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def l2_norm(tok, groups=1):
    batch_size, seq_len, hidden_dim = tok.size()
    tok = tok.view(batch_size, seq_len, groups, hidden_dim // groups)
    tok = F.normalize(tok, p=2, dim=-1)
    tok = tok.view(batch_size, seq_len, hidden_dim)
    return tok
