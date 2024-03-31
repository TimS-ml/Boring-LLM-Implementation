import torch
from boring_nn.utils import cprint
from boring_nn.norm import *


def test_LayerNorm1d():
    module = SimpleLayerNorm1d(100)
    x = torch.randn(32, 100)
    x = module(x)
    cprint(x.shape)


def test_LayerNorm():
    # [1] NLP Example
    batch, sentence_length, embedding_dim = 20, 5, 10
    embedding = torch.randn(batch, sentence_length, embedding_dim)
    # layer_norm = nn.LayerNorm(embedding_dim)
    layer_norm_nlp = LayerNorm(embedding_dim)
    cprint(layer_norm_nlp(embedding).shape)

    # [2] CV Example
    batch, channel, H, W = 20, 5, 10, 10
    img_batch = torch.randn(batch, channel, H, W)
    # Normalize over the last three dimensions (i.e. the channel and spatial dimensions)
    layer_norm_cv = LayerNorm(img_batch.shape[1:])
    cprint(layer_norm_cv(img_batch).shape)


test_LayerNorm()


