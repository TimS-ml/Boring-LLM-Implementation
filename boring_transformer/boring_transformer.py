'''
On LayerNorm's location: https://arxiv.org/abs/2002.04745

- https://nn.labml.ai/transformers/models.html
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from einops import rearrange

from torch import Tensor, Size
from typing import Optional, Tuple, Union, List

from boring_nn import attention, pe, norm, ffn 
from boring_nn.norm import l2_norm

from boring_utils.utils import cprint, get_layers, get_device
from boring_utils.nn_utils import (
    always, 
    exists, 
    groupby_prefix_and_trim, 
    pick_and_pop,
    max_neg_value,
    pad_at_dim
)
from boring_utils.helpers import DEBUG

attn_layers = get_layers(attention)
norm_layers = get_layers(norm)
ffn_layers = get_layers(ffn)
pe_layers = get_layers(pe)

ATTN = attn_layers['MultiHeadAttention']
FFN = ffn_layers['FeedForward']
LN = norm_layers['LayerNorm']
PE = pe_layers['LearnedPositionalEncoding']

if DEBUG >= 2:
    cprint(attn_layers, norm_layers, ffn_layers, pe_layers)


def dropout_seq(seq, mask, dropout):
    b, n = seq.shape[:2]
    device = seq.device
    logits = torch.randn(b, n, device=device)

    if mask is not None:
        mask_value = max_neg_value(logits)
        logits = logits.masked_fill(~mask, mask_value)

    keep_prob = 1. - dropout
    num_keep = max(1, int(keep_prob * n))
    keep_indices = logits.topk(num_keep, dim=1).indices

    batch_indices = torch.arange(b, device=device).unsqueeze(1)

    seq = seq[batch_indices, keep_indices]

    if mask is not None:
        seq_counts = mask.sum(dim=-1)
        seq_keep_counts = torch.ceil(seq_counts * keep_prob).int()
        keep_mask = torch.arange(num_keep, device=device).unsqueeze(0) < seq_keep_counts.unsqueeze(1)

        mask = mask[batch_indices, keep_indices] & keep_mask

    return seq, mask


class TokenEmbedding(nn.Module):
    def __init__(self, dim, num_tokens, l2_norm=False):
        super().__init__()
        self.l2_norm = l2_norm
        self.emb = nn.Embedding(num_tokens, dim)

    def forward(self, x):
        token_emb = self.emb(x.long())
        if self.l2_norm:
            return l2_norm(token_emb)
        return token_emb


# TODO: add prefix_attn_len
class AttentionLayers(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, causal=False, **kwargs):
        super(AttentionLayers, self).__init__()
        self.mha = ATTN(d_model, num_heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LN(d_model)
        self.feed_forward = FFN(d_model, d_ff, dropout=dropout)
        self.causal = causal

    def forward(self, x, attn_mask=None, return_hiddens=False, **kwargs):
        '''
        kv cache:
        in XTransformer, we got:
            hiddens = []
            layer_hiddens = []
            intermediates = []

        when return_hiddens=True,
        intermediates = LayerIntermediates(
            hiddens = hiddens,
            last_hidden = x,
            attn_intermediates = intermediates,
            layer_hiddens = layer_hiddens,
        )
        
        then intermediates can be used as "cache" in the next layer
        '''
        if self.causal and attn_mask is None:
            attn_mask = torch.ones((x.shape[1], x.shape[1]), device=x.device, dtype=torch.bool).triu(1)

        attn_output, attn_weights = self.mha(x, x, x, attn_mask=attn_mask)
        attn_output = self.dropout(attn_output)
        x_ln1 = self.layer_norm(x + attn_output)

        ff_output = self.feed_forward(x_ln1)
        ff_output = self.dropout(ff_output)
        x_ln2 = self.layer_norm(x_ln1 + ff_output)

        intermediates = {
            'attn_weights': attn_weights,
            'layer_norm_1': x_ln1,
            'layer_norm_2': x_ln2
        }

        if return_hiddens:
            return x_ln2, intermediates 
        return x_ln2


class Encoder(AttentionLayers):
    def __init__(self, **kwargs):
        assert 'causal' not in kwargs, 'cannot set causality on encoder'
        super().__init__(causal=False, **kwargs)


class Decoder(AttentionLayers):
    def __init__(self, **kwargs):
        assert 'causal' not in kwargs, 'cannot set causality on decoder'
        super().__init__(causal=True, **kwargs)


# class PrefixDecoder(AttentionLayers):
#     def __init__(self, **kwargs):
#         assert 'causal' not in kwargs, 'cannot set causality on decoder'
#         super().__init__(causal=False, **kwargs)

#     def forward(self, x, attn_mask=None, prefix_attn_len=None, **kwargs):
#         b, n, device = x.shape[0], x.shape[1], x.device
#         causal_mask = torch.ones((n, n), device=device, dtype=torch.bool).triu(1)

#         forwarded_mask = ~causal_mask

#         if prefix_attn_len is not None:
#             if isinstance(prefix_attn_len, int):
#                 prefix_attn_len = torch.full((b,), prefix_attn_len, device=device)

#             prefix_mask = torch.arange(n, device=device) < rearrange(prefix_attn_len, 'b -> b 1 1 1')
#             forwarded_mask = forwarded_mask | prefix_mask

#         if attn_mask is not None:
#             forwarded_mask = forwarded_mask & attn_mask

#         return super().forward(x, attn_mask=forwarded_mask, **kwargs)


'''
TODO: 
- in which case max_seq_len == 0?
- add prepend_mask + prepend_embeds:
    prepend_mask is used in conjunction with prepend_embeds to mask out certain tokens in the prepended embeddings.
    https://github.com/lucidrains/x-transformers/issues/211
'''
class BoringTransformerWrapper(nn.Module):
    '''
    Positional arguments are not allowed after the * in the parameter list.
    Simplified version of TransformerWrapper
    There's no need for Encoder-Decoder attention

    model = BoringTransformerWrapper(
        attn_layers = Decoder(
            d_model = 12, 
            num_heads = 8, 
            d_ff = 512
        ),
        num_tokens = 20000,
        max_seq_len = 1024,
    ).cuda()


    model = BoringTransformerWrapper(
        attn_layers = Encoder(
            d_model = 12, 
            num_heads = 8, 
            d_ff = 512
        ),
        num_tokens = 20000,
        max_seq_len = 1024,
    ).cuda()


    '''
    def __init__(
        self,
        *,
        attn_layers,
        d_model,
        num_tokens,
        max_seq_len,  # max sequence length
        l2norm_embed=False,
        dropout=0.1,
        **kwargs
    ):
        super().__init__()

        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.l2norm_embed = l2norm_embed
        self.dropout = nn.Dropout(dropout)
        
        # if l2norm_embed = False, then this is a normal nn.Embedding
        self.token_emb = TokenEmbedding(d_model, num_tokens, l2_norm = l2norm_embed)

        no_abs_pos_emb = max_seq_len == 0
        if no_abs_pos_emb:
            self.pos_emb = always(0)
        else:
            self.pos_emb = PE(d_model, dropout, max_seq_len)

        self.layers = attn_layers

        # TODO: KV Cache

    def init_(self):
        if self.l2norm_embed:
            nn.init.normal_(self.token_emb.emb.weight, std = 1e-5)

            if not isinstance(self.pos_emb, always):
                nn.init.normal_(self.pos_emb.emb.weight, std = 1e-5)
            return

        nn.init.kaiming_normal_(self.token_emb.emb.weight)

    '''
    we can use external positional embedding, 
    for example RoPE
    '''
    def forward(
        self,
        x,
        mask=None,
        pos=None,
        return_intermediates=False,
        **kwargs
    ):
        batch_size, seq_len, device = x.shape[0], x.shape[1], x.device  # used for mask

        external_pos_emb = exists(pos) and pos.dtype != torch.long
        pos_emb = self.pos_emb(x, pos=pos) if not external_pos_emb else pos
        x = self.token_emb(x) + pos_emb
        x = self.dropout(x)
        
        # recursively feed x to self.layers
        intermediates = []
        for layer in self.layers:
            x, layer_intermediates = layer(x, mask=mask, return_hiddens=True, **kwargs)
            intermediates.append(layer_intermediates)

        if not return_intermediates:
            return x

        return x, intermediates


'''
TODO: WIP
need update, add enc / dec layer number
check out the causal flag's location
'''
class BoringTransformer(nn.Module):
    '''
    Positional arguments are not allowed after the * in the parameter list.
    Simplified version of XTransformer
    **Need Encoder-Decoder attention**

    model = BoringTransformer(
        dim = 512,
        enc_num_tokens = 256,
        enc_depth = 6,
        enc_heads = 8,
        enc_max_seq_len = 1024,
        dec_num_tokens = 256,
        dec_depth = 6,
        dec_heads = 8,
        dec_max_seq_len = 1024,
    )
    '''
    def __init__(
        self,
        *,
        d_model,
        **kwargs
    ):
        super().__init__()

        enc_kwargs, kwargs = groupby_prefix_and_trim('enc_', kwargs)
        dec_kwargs, kwargs = groupby_prefix_and_trim('dec_', kwargs)

        enc_transformer_kwargs = pick_and_pop(['num_tokens', 'max_seq_len'], enc_kwargs)
        enc_transformer_kwargs['emb_dropout'] = enc_kwargs.pop('emb_dropout', 0)
        enc_transformer_kwargs['num_memory_tokens'] = enc_kwargs.pop('num_memory_tokens', None)
        enc_transformer_kwargs['scaled_sinu_pos_emb'] = enc_kwargs.pop('scaled_sinu_pos_emb', False)
        enc_transformer_kwargs['use_abs_pos_emb'] = enc_kwargs.pop('use_abs_pos_emb', True)

        dec_transformer_kwargs = pick_and_pop(['num_tokens', 'max_seq_len'], dec_kwargs)
        dec_transformer_kwargs['emb_dropout'] = dec_kwargs.pop('emb_dropout', 0)
        dec_transformer_kwargs['scaled_sinu_pos_emb'] = dec_kwargs.pop('scaled_sinu_pos_emb', False)
        dec_transformer_kwargs['use_abs_pos_emb'] = dec_kwargs.pop('use_abs_pos_emb', True)

        self.encoder = BoringTransformerWrapper(
            **enc_transformer_kwargs,
            attn_layers = Encoder(dim = d_model, **enc_kwargs)
        )

        self.decoder = BoringTransformerWrapper(
            **dec_transformer_kwargs,
            attn_layers = Decoder(dim = d_model, cross_attend = True, **dec_kwargs)
        )

    def init_(self):
        if self.l2norm_embed:
            if not isinstance(self.pos_emb, always):
                nn.init.normal_(self.pos_emb.emb.weight, std=1e-5)

    def forward(
        self,
        src, 
        tgt,
        mask=None,
        attn_mask = None,
        src_prepend_embeds = None,
        **kwargs
    ):

        enc = self.encoder(
            src, mask = mask, attn_mask = attn_mask, return_embeddings = True)

        if exists(src_prepend_embeds) and exists(mask):
            mask = pad_at_dim(mask, (src_prepend_embeds.shape[-2], 0), dim = -1, value = True)

        if self.training and self.cross_attn_tokens_dropout > 0:
            enc, mask = dropout_seq(enc, mask, self.cross_attn_tokens_dropout)

        out = self.decoder(tgt, context = enc, context_mask = mask)
        return out
