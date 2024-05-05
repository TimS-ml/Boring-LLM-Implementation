'''
some of them grabed from x_transformer
'''

import os
from pathlib import Path
import math
import numpy as np
import matplotlib.pyplot as plt
from functools import partial, wraps
from collections import namedtuple
from dataclasses import dataclass
from typing import List, Dict, Tuple, Callable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def deriv(func, input_, delta=0.001):
    return (func(input_ + delta) - func(input_ - delta)) / (2 * delta)


def get_batch_np(data, block_size=1024, batch_size=32, device='cpu'):
    # Returns a tensor filled with random integers generated uniformly between low (inclusive) and high (exclusive)
    # The shape of the tensor is defined by the variable argument size
    # 0 ~ len(data) - block_size with output shape of (batch_size,)
    ix = torch.randint(len(data) - block_size, (batch_size, ))
    x = torch.stack([
        torch.from_numpy((data[i:i + block_size]).astype(np.int64)) for i in ix
    ])
    y = torch.stack([
        torch.from_numpy((data[i + 1:i + 1 + block_size]).astype(np.int64))
        for i in ix
    ])
    if device == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(
            device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss_np(model,
                     eval_iters,
                     train_data,
                     val_data,
                     block_size=1024,
                     batch_size=32,
                     device='cpu'):
    out = {}
    model.eval()
    data_dic = {'train': train_data, 'val': val_data}
    for split, data in data_dic.items():
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch_np(data,
                                block_size=block_size,
                                batch_size=batch_size,
                                device=device)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# learning rate decay scheduler (cosine with warmup)
def get_lr(it, learning_rate, warmup_iters, lr_decay_iters, min_lr):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


class always():

    def __init__(self, val):
        self.val = val

    def __call__(self, *args, **kwargs):
        return self.val


def exists(val):
    return val is not None


'''
From X-transformer, need update
'''

def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max


def pad_at_dim(t, pad: Tuple[int, int], dim = -1, value = 0.):
    if pad == (0, 0):
        return t

    dims_from_right = (- dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = ((0, 0) * dims_from_right)
    return F.pad(t, (*zeros, *pad), value = value)

# init helpers
def init_zero_(layer):
    nn.init.constant_(layer.weight, 0.)
    if exists(layer.bias):
        nn.init.constant_(layer.bias, 0.)


# keyword argument helpers
def pick_and_pop(keys, d):
    values = list(map(lambda key: d.pop(key), keys))
    return dict(zip(keys, values))


def group_dict_by_key(cond, d):
    return_val = [dict(), dict()]
    for key in d.keys():
        match = bool(cond(key))
        ind = int(not match)
        return_val[ind][key] = d[key]
    return (*return_val, )


def string_begins_with(prefix, str):
    return str.startswith(prefix)


def group_by_key_prefix(prefix, d):
    return group_dict_by_key(partial(string_begins_with, prefix), d)


def groupby_prefix_and_trim(prefix, d):
    kwargs_with_prefix, kwargs = group_dict_by_key(
        partial(string_begins_with, prefix), d)
    kwargs_without_prefix = dict(
        map(lambda x: (x[0][len(prefix):], x[1]),
            tuple(kwargs_with_prefix.items())))
    return kwargs_without_prefix, kwargs
