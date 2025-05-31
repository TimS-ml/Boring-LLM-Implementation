import torch
from boring_llm.tiny.tiny_base import (
    TinyFeedForward, TinyScaleDotProductAttention,
    TinyMultiHeadAttention, TinyTransformBlock,
    TinyEncoder, TinyDecoder,
    TinyTransformerWrapper, TinyEncDecTransformer
)
from boring_llm.base.tiny_config import *
from boring_llm.utils.utils import create_causal_mask
from boring_utils.utils import get_device, cprint, tprint
# from boring_utils.helpers import DEBUG
# import os; os.environ['DEBUG'] = '3'

device = get_device()

def test_Tiny_feed_forward():
    model = TinyFeedForward(EMBEDDING_DIM)
    x = torch.randn(BATCH_SIZE, BLOCK_SIZE, EMBEDDING_DIM)
    out = model(x)
    assert out.shape == x.shape, "Output shape should match input shape"
    
    mul = 2
    model = TinyFeedForward(EMBEDDING_DIM, mul=mul)
    assert model.fc1.weight.shape == (EMBEDDING_DIM * mul, EMBEDDING_DIM), "Hidden layer dimensions should be correct"
    assert model.fc2.weight.shape == (EMBEDDING_DIM, EMBEDDING_DIM * mul), "Output layer dimensions should be correct"
    
    tprint("All tests passed!")

def test_scale_dot_product_attention():
    seq_q, seq_k = BLOCK_SIZE, BLOCK_SIZE
    q = torch.randn(BATCH_SIZE, N_HEAD, seq_q, D_HEAD)
    k = torch.randn(BATCH_SIZE, N_HEAD, seq_k, D_HEAD)
    v = torch.randn(BATCH_SIZE, N_HEAD, seq_k, D_HEAD)
    
    # Test non-causal
    attn = TinyScaleDotProductAttention(causal=False)
    out = attn(q, k, v)
    assert out.shape == (BATCH_SIZE, N_HEAD, seq_q, D_HEAD), "Output shape should match input shape"
    
    # Test causal
    attn = TinyScaleDotProductAttention(causal=True)
    out = attn(q, k, v)
    assert out.shape == (BATCH_SIZE, N_HEAD, seq_q, D_HEAD), "Output shape should match input shape"
    
    # Test mask values
    mask = create_causal_mask(seq_q, seq_k)
    assert mask.shape == (seq_q, seq_k), "Mask shape should be (seq_q, seq_k)"
    assert mask.dtype == torch.bool, "Mask should be boolean"
    assert torch.all(mask == torch.triu(torch.ones_like(mask), diagonal=1)), "Mask should be upper triangular"
    
    tprint("All tests passed!")

def test_multi_head_attention():
    # Test self-attention
    mha = TinyMultiHeadAttention(dim=EMBEDDING_DIM, n_head=N_HEAD, d_head=D_HEAD, causal=False)
    x = torch.randn(BATCH_SIZE, BLOCK_SIZE, EMBEDDING_DIM)
    out = mha(x)
    assert out.shape == (BATCH_SIZE, BLOCK_SIZE, EMBEDDING_DIM), "Output shape should match input shape"
    
    # Test cross-attention
    context = torch.randn(BATCH_SIZE, BLOCK_SIZE, EMBEDDING_DIM)
    out = mha(x, context=context)
    assert out.shape == (BATCH_SIZE, BLOCK_SIZE, EMBEDDING_DIM), "Output shape should match input shape"
    
    # Test causal masking
    mha_causal = TinyMultiHeadAttention(dim=EMBEDDING_DIM, n_head=N_HEAD, d_head=D_HEAD, causal=True)
    out = mha_causal(x)
    assert out.shape == (BATCH_SIZE, BLOCK_SIZE, EMBEDDING_DIM), "Output shape should match input shape"
    
    tprint("All tests passed!")

def test_encoder_decoder():
    # Test Encoder
    encoder = TinyEncoder(
        dim=EMBEDDING_DIM,
        n_layers=N_LAYER,
        n_head=N_HEAD,
        d_head=D_HEAD,
        ffn_mul=FFN_MUL
    ).to(device)
    
    x = torch.randn(BATCH_SIZE, BLOCK_SIZE, EMBEDDING_DIM).to(device)
    enc_out = encoder(x)
    assert enc_out.shape == (BATCH_SIZE, BLOCK_SIZE, EMBEDDING_DIM), "Encoder output shape should match input shape"
    
    # Test Decoder
    decoder = TinyDecoder(
        dim=EMBEDDING_DIM,
        n_layers=N_LAYER,
        n_head=N_HEAD,
        d_head=D_HEAD,
        ffn_mul=FFN_MUL
    ).to(device)
    
    # Test decoder with context (cross attention)
    context = torch.randn(BATCH_SIZE, BLOCK_SIZE, EMBEDDING_DIM).to(device)
    dec_out = decoder(x, context=context)
    assert dec_out.shape == (BATCH_SIZE, BLOCK_SIZE, EMBEDDING_DIM), "Decoder output shape should match input shape with context"
    
    # Test decoder without context
    dec_out = decoder(x)
    assert dec_out.shape == (BATCH_SIZE, BLOCK_SIZE, EMBEDDING_DIM), "Decoder output shape should match input shape without context"
    
    tprint("All tests passed!")

def test_Tiny_transform():
    # Test basic forward pass
    model = TinyTransformBlock(
        dim=EMBEDDING_DIM,
        n_layers=N_LAYER,
        n_head=N_HEAD,
        d_head=D_HEAD,
        causal=False,
        ffn_mul=FFN_MUL
    ).to(device)
    
    x = torch.randn(BATCH_SIZE, BLOCK_SIZE, EMBEDDING_DIM).to(device)
    out = model(x)
    assert out.shape == (BATCH_SIZE, BLOCK_SIZE, EMBEDDING_DIM), "Output shape should match input shape"
    
    # Test with context (cross attention)
    context = torch.randn(BATCH_SIZE, BLOCK_SIZE, EMBEDDING_DIM).to(device)
    out = model(x, context=context)
    assert out.shape == (BATCH_SIZE, BLOCK_SIZE, EMBEDDING_DIM), "Output shape should match input shape with context"
    
    # Test causal masking
    model_causal = TinyTransformBlock(
        dim=EMBEDDING_DIM,
        n_layers=N_LAYER,
        n_head=N_HEAD,
        d_head=D_HEAD,
        causal=True,
        ffn_mul=FFN_MUL
    ).to(device)
    
    out = model_causal(x)
    assert out.shape == (BATCH_SIZE, BLOCK_SIZE, EMBEDDING_DIM), "Output shape should match input shape with causal masking"
    
    tprint("All tests passed!")

def test_Tiny_transformer_wrapper():
    # Create model
    model = TinyTransformerWrapper(
        num_tokens=NUM_TOKENS,
        max_seq_len=BLOCK_SIZE,
        dim=EMBEDDING_DIM,
        n_layers=N_LAYER,
        n_head=N_HEAD,
        d_head=D_HEAD,
        ffn_mul=FFN_MUL,
        dropout=DROPOUT
    ).to(device)
    
    # Create test input
    x = torch.randint(0, NUM_TOKENS, (BATCH_SIZE, BLOCK_SIZE)).to(device)
    
    # Forward pass
    logits = model(x)
    
    # Check output shape
    assert logits.shape == (BATCH_SIZE, BLOCK_SIZE, NUM_TOKENS), \
        f"Expected output shape {(BATCH_SIZE, BLOCK_SIZE, NUM_TOKENS)}, got {logits.shape}"
    
    tprint("All tests passed!")

def test_Tiny_enc_dec_transformer():
    # Test config
    ENC_NUM_TOKENS = int(NUM_TOKENS * 0.5)
    DEC_NUM_TOKENS = NUM_TOKENS
    ENC_N_LAYER = N_LAYER
    DEC_N_LAYER = N_LAYER
    ENC_N_HEAD = N_HEAD
    DEC_N_HEAD = N_HEAD
    
    # Create model
    model = TinyEncDecTransformer(
        dim=EMBEDDING_DIM,
        # encoder
        enc_num_tokens=ENC_NUM_TOKENS,
        enc_n_layers=ENC_N_LAYER,
        enc_n_head=ENC_N_HEAD,
        enc_max_seq_len=BLOCK_SIZE,
        # decoder
        dec_num_tokens=DEC_NUM_TOKENS,
        dec_n_layers=DEC_N_LAYER,
        dec_n_head=DEC_N_HEAD,
        dec_max_seq_len=BLOCK_SIZE,
        # misc
        tie_token_emb=False,
        ffn_mul=FFN_MUL,
        dropout=DROPOUT
    ).to(device)
    
    # Create test inputs
    src = torch.randint(0, ENC_NUM_TOKENS, (BATCH_SIZE, BLOCK_SIZE)).to(device)
    tgt = torch.randint(0, DEC_NUM_TOKENS, (BATCH_SIZE, BLOCK_SIZE)).to(device)
    
    # Forward pass
    logits = model(src, tgt)
    
    # Check output shape
    assert isinstance(logits, torch.Tensor) and logits.dim() == 0, \
        f"Expected output to be a scalar loss value, got shape {logits.shape}"
    assert torch.isfinite(logits), f"Expected a finite loss value, got {logits}"
    
    # Test generation
    tgt_start = torch.randint(0, DEC_NUM_TOKENS, (BATCH_SIZE, 1)).to(device)
    generated = model.generate(src, tgt_start, seq_len=BLOCK_SIZE)
    
    # Check generation shape
    assert generated.shape == (BATCH_SIZE, BLOCK_SIZE), \
        f"Expected generation shape {(BATCH_SIZE, BLOCK_SIZE)}, got {generated.shape}"
    
    tprint("All tests passed!")

if __name__ == "__main__":
    test_Tiny_feed_forward()
    test_scale_dot_product_attention()
    test_multi_head_attention()
    test_encoder_decoder()
    test_Tiny_transform()
    test_Tiny_transformer_wrapper()
    test_Tiny_enc_dec_transformer()
