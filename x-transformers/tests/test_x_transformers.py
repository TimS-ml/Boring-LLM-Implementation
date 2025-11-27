# Import pytest framework and parametrize decorator for test parameterization
import pytest
param = pytest.mark.parametrize

# Import PyTorch core modules for tensor operations and neural network building blocks
import torch
from torch import nn
from torch.nn import Module

# Import x-transformers core components for transformer architecture testing
from x_transformers.x_transformers import (
    XTransformer,           # Full encoder-decoder transformer architecture
    TransformerWrapper,     # Wrapper for transformer with token embeddings and position encoding
    Encoder,               # Encoder-only transformer layers
    Decoder,               # Decoder-only transformer layers (autoregressive)
    LinearNoBias,          # Linear layer without bias term
)

# Import NeoMLP for testing alternative MLP architectures
from x_transformers.neo_mlp import (
    NeoMLP
)

# Import multi-input transformer wrapper for handling multiple token types
from x_transformers.multi_input import MultiInputTransformerWrapper

def test_readme():
    """
    Test basic XTransformer encoder-decoder model functionality.

    This test validates the core use case from the README: creating a full
    encoder-decoder transformer model with tied embeddings, computing a loss,
    and performing a backward pass. This ensures the basic seq2seq architecture
    works correctly for tasks like machine translation.
    """
    # Create a full encoder-decoder transformer with tied token embeddings
    model = XTransformer(
        dim = 512,                    # Model dimension
        enc_num_tokens = 256,         # Encoder vocabulary size
        enc_depth = 6,                # Number of encoder layers
        enc_heads = 8,                # Number of attention heads in encoder
        enc_max_seq_len = 1024,       # Maximum encoder sequence length
        dec_num_tokens = 256,         # Decoder vocabulary size
        dec_depth = 6,                # Number of decoder layers
        dec_heads = 8,                # Number of attention heads in decoder
        dec_max_seq_len = 1024,       # Maximum decoder sequence length
        tie_token_emb = True          # Share embeddings between encoder and decoder
    )

    # Create source sequence: random token IDs in range [0, 256)
    src = torch.randint(0, 256, (1, 1024))
    # Create source mask: all tokens are valid (not padding)
    src_mask = torch.ones_like(src).bool()
    # Create target sequence: random token IDs in range [0, 256)
    tgt = torch.randint(0, 256, (1, 1024))

    # Forward pass computes cross-entropy loss for sequence-to-sequence task
    loss = model(src, tgt, mask = src_mask)
    # Backward pass computes gradients - validates model is differentiable
    loss.backward()

def test_kv_cache():
    """
    Test key-value caching for efficient autoregressive generation.

    This test validates that the KV cache mechanism produces identical results
    to the full computation while enabling faster inference. When generating
    tokens sequentially, we can cache previous key-value pairs to avoid
    redundant computation.
    """
    # Create decoder model with cross-attention capability
    model = TransformerWrapper(
        num_tokens = 20000,           # Vocabulary size
        max_seq_len = 1024,           # Maximum sequence length
        attn_layers = Decoder(
            dim = 8,                  # Small dimension for testing
            depth = 2,                # 2 decoder layers
            heads = 4,                # 4 attention heads
            cross_attend = True       # Enable cross-attention to context
        )
    )

    # Set to evaluation mode to ensure consistent behavior
    model.eval()

    # Create initial prompts (batch_size=2, seq_len=16) initialized to zeros
    prompts = torch.zeros((2, 16))
    # Create random context for cross-attention (batch_size=2, context_len=8, dim=8)
    context = torch.randn(2, 8, 8)

    # Forward pass with intermediate values (including KV cache) returned
    logits, cache = model(
        prompts,
        context = context,
        return_intermediates = True  # Returns cache for reuse
    )

    # Sample next token by taking argmax of last position logits
    sampled = logits[:, -1].argmax(dim = -1, keepdim = True)
    # Append sampled token to prompts for next iteration
    prompts = torch.cat((prompts, sampled), dim = -1)

    # Compute logits without using cache (full recomputation)
    next_logits = model(prompts, context = context)
    # Compute logits using cached key-value pairs (efficient)
    next_logits_with_cache = model(prompts, context = context, cache = cache)

    # Verify that cached computation produces identical results to full computation
    # Only check last position since that's what we're generating
    assert torch.allclose(next_logits[:, -1], next_logits_with_cache[:, -1], atol = 1e-6)

def test_cope():
    """
    Test Contextual Position Encoding (CoPE) in attention layers.

    CoPE is a position encoding method that conditions positions on content,
    allowing the model to learn content-dependent position representations.
    This test ensures the model can forward pass with CoPE enabled.
    """
    # Create decoder model with Contextual Position Encoding
    model = TransformerWrapper(
        num_tokens = 256,             # Vocabulary size
        max_seq_len = 1024,           # Maximum sequence length
        attn_layers = Decoder(
            dim = 8,                  # Model dimension
            depth = 2,                # Number of layers
            heads = 4,                # Number of attention heads
            attn_use_cope = True      # Enable Contextual Position Encoding
        )
    )

    # Create random sequence of token IDs (batch_size=1, seq_len=1024)
    seq = torch.randint(0, 256, (1, 1024))
    # Forward pass - validates CoPE implementation works correctly
    logits = model(seq)

def test_adaptive_layernorm():
    """
    Test Adaptive LayerNorm conditioned on external input.

    Adaptive LayerNorm allows the normalization parameters (scale and shift)
    to be modulated by an external conditioning vector, enabling the model to
    adapt its representations based on auxiliary information. This is useful
    for conditional generation tasks.
    """
    # Create decoder with adaptive normalization conditioned on external signal
    model = TransformerWrapper(
        num_tokens = 20000,                 # Vocabulary size
        max_seq_len = 1024,                 # Maximum sequence length
        attn_layers = Decoder(
            dim = 512,                      # Model dimension
            dim_condition = 768,            # Dimension of conditioning vector
            depth = 12,                     # Number of decoder layers
            heads = 8,                      # Number of attention heads
            use_adaptive_layernorm = True,  # Enable adaptive layer normalization
            use_adaptive_layerscale = True  # Enable adaptive layer scaling
        )
    )

    # Create batch of random token sequences (batch_size=2, seq_len=1024)
    x = torch.randint(0, 256, (2, 1024))
    # Create conditioning vectors (batch_size=2, condition_dim=768)
    # These vectors modulate the normalization parameters
    condition = torch.randn(2, 768)

    # Forward pass with conditioning - validates adaptive norm implementation
    model(x, condition = condition)

def test_adaptive_rmsnorm():
    """
    Test Adaptive RMSNorm (Root Mean Square Normalization) with conditioning.

    RMSNorm is a simpler alternative to LayerNorm that only rescales activations
    without centering. Adaptive RMSNorm allows the scale parameter to be modulated
    by a conditioning vector, with an optional MLP to process the condition.
    """
    # Create decoder with adaptive RMS normalization
    model = TransformerWrapper(
        num_tokens = 20000,                  # Vocabulary size
        max_seq_len = 1024,                  # Maximum sequence length
        attn_layers = Decoder(
            dim = 512,                       # Model dimension
            dim_condition = 768,             # Dimension of conditioning vector
            depth = 12,                      # Number of decoder layers
            heads = 8,                       # Number of attention heads
            use_adaptive_rmsnorm = True,     # Enable adaptive RMS normalization
            adaptive_condition_mlp = True    # Use MLP to process conditioning vector
        )
    )

    # Create batch of random token sequences (batch_size=2, seq_len=1024)
    x = torch.randint(0, 256, (2, 1024))
    # Create conditioning vectors (batch_size=2, condition_dim=768)
    condition = torch.randn(2, 768)

    # Forward pass with conditioning - validates adaptive RMSNorm implementation
    model(x, condition = condition)

def test_attn_softclamp_logits():
    """
    Test soft clamping of attention logits.

    Soft clamping applies a smooth limiting function to attention logits to
    prevent extreme values that could lead to numerical instability or
    overly peaked attention distributions. This helps stabilize training.
    """
    # Create decoder with soft clamping enabled on attention logits
    model = TransformerWrapper(
        num_tokens = 20000,                  # Vocabulary size
        max_seq_len = 1024,                  # Maximum sequence length
        attn_layers = Decoder(
            dim = 512,                       # Model dimension
            dim_condition = 768,             # Dimension of conditioning vector
            depth = 12,                      # Number of decoder layers
            heads = 8,                       # Number of attention heads
            attn_softclamp_logits = True,    # Enable soft clamping of attention logits
        )
    )

    # Create random token sequence (batch_size=1, seq_len=1024)
    x = torch.randint(0, 256, (1, 1024))

    # Forward pass - validates soft clamping works without errors
    model(x)

def test_multiple_input_embeds():
    """
    Test multi-input transformer that handles multiple token types simultaneously.

    This test validates a model that can process multiple parallel sequences with
    different vocabularies (e.g., musical notes, pitch values, and tones). Each
    input type has its own embedding table, and the embeddings are combined before
    being fed to the transformer. Useful for multimodal or structured inputs.
    """
    # Create multi-input transformer with separate embeddings for each input type
    model = MultiInputTransformerWrapper(
        num_tokens = dict(
            note = 20000,            # Large vocabulary for note symbols
            pitch = 32,              # Smaller vocabulary for pitch values
            tone = 16                # Smaller vocabulary for tone types
        ),
        max_seq_len = 1024,          # Maximum sequence length
        return_only_embed = True,    # Return embeddings instead of logits
        attn_layers = Decoder(
            dim = 128,               # Model dimension
            depth = 6,               # Number of decoder layers
            heads = 8                # Number of attention heads
        )
    )

    # Create input dictionary with parallel sequences for each modality
    x = dict(
        note = torch.randint(0, 20000, (2, 1024)),   # Note tokens (batch=2, seq=1024)
        pitch = torch.randint(0, 32, (2, 1024)),     # Pitch values (batch=2, seq=1024)
        tone = torch.randint(0, 16, (2, 1024))       # Tone types (batch=2, seq=1024)
    )

    # Forward pass combines all embeddings and processes through transformer
    embed = model(x)

    # Verify output has correct shape: (batch_size, seq_len, model_dim)
    assert embed.shape == (2, 1024, 128)

def test_average_pool_embed():
    """
    Test average pooling of embeddings for sequence-level classification.

    Instead of using a CLS token or the last position, this test validates
    average pooling across the sequence (respecting the mask) to produce a
    single embedding vector per sequence. This is useful for classification
    tasks where all tokens contribute equally.
    """
    # Create encoder with average pooling enabled
    model = TransformerWrapper(
        num_tokens = 20000,              # Vocabulary size
        max_seq_len = 1024,              # Maximum sequence length
        num_memory_tokens = 2,           # Additional memory tokens
        average_pool_embed = True,       # Enable average pooling across sequence
        attn_layers = Encoder(
            dim = 128,                   # Model dimension
            depth = 6,                   # Number of encoder layers
            heads = 8                    # Number of attention heads
        )
    )

    # Create batch of random token sequences (batch_size=2, seq_len=1024)
    x = torch.randint(0, 20000, (2, 1024))
    # Create random mask: True=valid token, False=padding (batch_size=2, seq_len=1024)
    mask = torch.randint(0, 2, (2, 1024)).bool()

    # Forward pass produces classification logits from pooled embeddings
    logits = model(x, mask = mask)

    # Verify output has shape (batch_size, num_classes) - one vector per sequence
    assert logits.shape == (2, 20000)

@param('num_cls_tokens', (1, 2))
def test_cls_token(num_cls_tokens):
    """
    Test CLS token(s) for sequence-level representation.

    CLS tokens are special tokens prepended to the sequence whose final
    representations are used for classification. This test validates both
    single and multiple CLS tokens, where multiple CLS tokens can capture
    different aspects of the sequence.

    Args:
        num_cls_tokens: Number of CLS tokens to prepend (1 or 2)
    """
    # Create encoder with CLS token(s)
    model = TransformerWrapper(
        num_tokens = 20000,              # Vocabulary size
        max_seq_len = 1024,              # Maximum sequence length
        num_memory_tokens = 2,           # Additional memory tokens
        use_cls_token = True,            # Enable CLS token prepending
        num_cls_tokens=num_cls_tokens,   # Number of CLS tokens to use
        attn_layers = Encoder(
            dim = 128,                   # Model dimension
            depth = 6,                   # Number of encoder layers
            heads = 8                    # Number of attention heads
        )
    )

    # Create batch of random token sequences (batch_size=2, seq_len=1024)
    x = torch.randint(0, 20000, (2, 1024))
    # Create random attention mask (batch_size=2, seq_len=1024)
    mask = torch.randint(0, 2, (2, 1024)).bool()

    # Forward pass - CLS token representations used for classification
    logits = model(x, mask = mask)

    # Expected output shape depends on number of CLS tokens
    if num_cls_tokens == 1:
        # Single CLS token: (batch_size, num_classes)
        expected_shape = (2, 20000)
    else:
        # Multiple CLS tokens: (batch_size, num_cls_tokens, num_classes)
        expected_shape = (2, num_cls_tokens, 20000)

    # Verify output shape matches expected
    assert logits.shape == expected_shape

def test_squeeze_logit_dim_one():
    """
    Test squeezing singleton dimension for scalar predictions.

    When the logits dimension is 1 (e.g., binary classification or regression),
    we can optionally squeeze out the last dimension to get a simpler output
    shape. This test validates that with logits_dim=1 and squeeze_out_last_dim=True,
    the output is a scalar per sequence rather than a (batch, 1) tensor.
    """
    # Create encoder for scalar prediction with dimension squeezing
    model = TransformerWrapper(
        num_tokens = 20000,              # Vocabulary size
        max_seq_len = 1024,              # Maximum sequence length
        logits_dim = 1,                  # Single output value per sequence
        average_pool_embed = True,       # Pool embeddings for sequence representation
        squeeze_out_last_dim = True,     # Remove singleton last dimension
        attn_layers = Encoder(
            dim = 128,                   # Model dimension
            depth = 6,                   # Number of encoder layers
            heads = 8                    # Number of attention heads
        )
    )

    # Create batch of random token sequences (batch_size=2, seq_len=1024)
    x = torch.randint(0, 20000, (2, 1024))
    # Create random attention mask (batch_size=2, seq_len=1024)
    mask = torch.randint(0, 2, (2, 1024)).bool()

    # Forward pass produces scalar predictions
    logits = model(x, mask = mask)

    # Verify output has shape (batch_size,) - one scalar per sequence
    assert logits.shape == (2,)

@param('depth', (4, 5))
def test_unet_skip(depth):
    """
    Test U-Net style skip connections in transformer layers.

    U-Net skip connections connect early layers to later layers symmetrically,
    similar to the U-Net architecture. This helps preserve fine-grained information
    and can improve gradient flow. The test validates both even (4) and odd (5)
    depth configurations to ensure skip connections work in both cases.

    Args:
        depth: Number of encoder layers (4 or 5) - tests even and odd depths
    """
    # Create encoder with U-Net style skip connections
    model = TransformerWrapper(
        num_tokens = 20000,              # Vocabulary size
        max_seq_len = 1024,              # Maximum sequence length
        attn_layers = Encoder(
            dim = 128,                   # Model dimension
            depth = depth,               # Number of layers (parameterized)
            heads = 8,                   # Number of attention heads
            unet_skips = True            # Enable U-Net skip connections
        )
    )

    # Create batch of random token sequences (batch_size=2, seq_len=1024)
    x = torch.randint(0, 20000, (2, 1024))
    # Create random attention mask (batch_size=2, seq_len=1024)
    mask = torch.randint(0, 2, (2, 1024)).bool()

    # Forward pass with U-Net skip connections
    model(x, mask = mask)

def test_recycling():
    """
    Test recycling mechanism for iterative refinement.

    Recycling passes the output through the model multiple times, using
    each iteration's output as input to the next. This allows iterative
    refinement of predictions. During training, the number of recycle steps
    is randomized; during evaluation, it can be explicitly specified.
    """
    # Create decoder with recycling enabled for iterative refinement
    model = TransformerWrapper(
        num_tokens = 20000,              # Vocabulary size
        max_seq_len = 1024,              # Maximum sequence length
        recycling = True,                # Enable recycling mechanism
        train_max_recycle_steps = 5,     # Max recycle iterations during training
        attn_layers = Decoder(
            dim = 128,                   # Model dimension
            depth = 6,                   # Number of decoder layers
            heads = 8                    # Number of attention heads
        )
    )

    # Create batch of random token sequences (batch_size=2, seq_len=1024)
    x = torch.randint(0, 20000, (2, 1024))

    # Training mode: random number of recycle steps (up to train_max_recycle_steps)
    logits = model(x)

    # Switch to evaluation mode
    model.eval()

    # Evaluation mode: explicitly specify number of recycle steps
    eval_logits = model(x, recycle_steps = 3)

def test_mos():
    """
    Test Mixture of Softmaxes (MoS) output layer.

    MoS uses multiple softmax distributions and mixes them with learned weights,
    allowing the model to represent multimodal output distributions more effectively.
    This can improve language modeling by capturing different contexts or meanings.
    """
    # Create decoder with Mixture of Softmaxes output layer
    model = TransformerWrapper(
        num_tokens = 20000,              # Vocabulary size
        max_seq_len = 1024,              # Maximum sequence length
        mixture_of_softmax = True,       # Enable Mixture of Softmaxes
        attn_layers = Decoder(
            dim = 128,                   # Model dimension
            depth = 6,                   # Number of decoder layers
            heads = 8                    # Number of attention heads
        )
    )

    # Create batch of random token sequences (batch_size=2, seq_len=1024)
    x = torch.randint(0, 20000, (2, 1024))

    # Training mode forward pass
    logits = model(x)

    # Switch to evaluation mode
    model.eval()

    # Evaluation mode forward pass
    eval_logits = model(x)

@param('attn_one_kv_head', (True, False))
def test_l2_distance(attn_one_kv_head):
    """
    Test L2 distance-based attention instead of dot-product attention.

    Instead of computing attention weights via dot-product similarity,
    this uses L2 (Euclidean) distance. Closer representations get higher
    attention weights. Tests both multi-head KV and single shared KV head.

    Args:
        attn_one_kv_head: If True, use single shared key-value head for all queries
    """
    # Create decoder with L2 distance-based attention
    model = TransformerWrapper(
        num_tokens = 20000,              # Vocabulary size
        max_seq_len = 1024,              # Maximum sequence length
        attn_layers = Decoder(
            dim = 512,                   # Model dimension
            depth = 12,                  # Number of decoder layers
            heads = 8,                   # Number of query heads
            attn_l2_distance = True,     # Use L2 distance instead of dot-product
            attn_one_kv_head = attn_one_kv_head,  # Single or multi KV heads
        )
    )

    # Create batch of random token sequences (batch_size=1, seq_len=1024)
    x = torch.randint(0, 256, (1, 1024))

    # Forward pass with L2 distance attention
    model(x)

def test_reinject_input():
    """
    Test input reinjection at each layer.

    Input reinjection adds the original input embeddings to each transformer
    layer, providing a direct path from input to all layers. This can improve
    gradient flow and help the model retain input information throughout
    the network. Useful when combined with recycling.
    """
    # Create decoder with input reinjection enabled
    model = TransformerWrapper(
        num_tokens = 20000,              # Vocabulary size
        max_seq_len = 1024,              # Maximum sequence length
        recycling = True,                # Enable recycling for iterative refinement
        attn_layers = Decoder(
            dim = 512,                   # Model dimension
            depth = 12,                  # Number of decoder layers
            heads = 8,                   # Number of attention heads
            reinject_input = True        # Add input embeddings to each layer
        )
    )

    # Create batch of random token sequences (batch_size=1, seq_len=1024)
    x = torch.randint(0, 256, (1, 1024))

    # Forward pass with input reinjection - output shape: (1, 1024, 20000)
    model(x)

@param('learned_value_residual_mix', (False, True))
def test_value_residual(
    learned_value_residual_mix: bool
):
    """
    Test value residual connections in attention.

    Value residuals add the pre-attention values back to the attention output,
    creating an additional residual path within the attention mechanism itself.
    This can be either a fixed addition or a learned weighted mixture.

    Args:
        learned_value_residual_mix: If True, learn mixing weight; if False, use fixed addition
    """
    # Create decoder with value residual connections
    model = TransformerWrapper(
        num_tokens = 20000,                  # Vocabulary size
        max_seq_len = 1024,                  # Maximum sequence length
        attn_layers = Decoder(
            dim = 128,                       # Model dimension
            depth = 6,                       # Number of decoder layers
            heads = 8,                       # Number of attention heads
            add_value_residual = True,       # Enable value residual connections
            learned_value_residual_mix = learned_value_residual_mix  # Fixed or learned mix
        )
    )

    # Create batch of random token sequences (batch_size=2, seq_len=1024)
    x = torch.randint(0, 20000, (2, 1024))

    # Forward pass with value residuals
    model(x)

@param('has_num_mem_kv', (False, True))
def test_forgetting_transformer(
    has_num_mem_kv: bool
):
    """
    Test forgetting transformer with data-dependent ALiBi and memory keys/values.

    This tests a transformer variant designed for forgetting earlier context,
    using data-dependent ALiBi (Attention with Linear Biases) that adapts
    position biases based on content. Optionally includes learnable memory
    key-value pairs that all positions can attend to.

    Args:
        has_num_mem_kv: If True, add 1 memory key-value pair; if False, use 0
    """
    # Create decoder with forgetting mechanism and data-dependent position biases
    model = TransformerWrapper(
        num_tokens = 20000,                      # Vocabulary size
        max_seq_len = 1024,                      # Maximum sequence length
        attn_layers = Decoder(
            dim = 128,                           # Model dimension
            depth = 6,                           # Number of decoder layers
            heads = 8,                           # Number of attention heads
            attn_num_mem_kv = 1 if has_num_mem_kv else 0,  # Memory key-value pairs
            attn_data_dependent_alibi = True     # Content-dependent position biases
        )
    )

    # Create batch of random token sequences (batch_size=2, seq_len=1024)
    x = torch.randint(0, 20000, (2, 1024))

    # Forward pass with forgetting mechanism
    embed = model(x)

def test_neo_mlp():
    """
    Test NeoMLP architecture.

    NeoMLP is an alternative MLP architecture that uses a model dimension
    separate from the input/output dimensions, allowing more expressive
    transformations. This test validates that NeoMLP correctly transforms
    inputs through multiple hidden layers.
    """
    # Create NeoMLP with specified dimensions and depth
    mlp = NeoMLP(
        dim_in = 5,              # Input dimension
        dim_out = 7,             # Output dimension
        dim_hidden = 16,         # Hidden layer dimension
        depth = 5,               # Number of hidden layers
        dim_model = 64,          # Model dimension (internal representation)
    )

    # Create random input batch (batch_size=3, input_dim=5)
    x = torch.randn(3, 5)

    # Forward pass through NeoMLP
    out = mlp(x)
    # Verify output has correct shape: (batch_size, output_dim)
    assert out.shape == (3, 7)

@param('flash', (True, False))
def test_custom_alibi(flash: bool):
    """
    Test custom position specification with ALiBi (Attention with Linear Biases).

    ALiBi adds position-dependent biases to attention scores without using
    position embeddings. This test validates that custom (non-sequential)
    position indices work correctly with ALiBi, both with and without
    Flash Attention optimization.

    Args:
        flash: If True, use Flash Attention; if False, use standard attention
    """
    # Create decoder with ALiBi position biases
    model = TransformerWrapper(
        num_tokens = 20_000,             # Vocabulary size
        max_seq_len = 1024,              # Maximum sequence length
        attn_layers = Decoder(
            dim = 512,                   # Model dimension
            depth = 2,                   # Number of decoder layers
            heads = 8,                   # Number of attention heads
            alibi_pos_bias = True,       # Enable ALiBi position biases
            attn_flash = flash           # Use Flash Attention if True
        )
    )

    # Create batch of random token sequences (batch_size=2, seq_len=4)
    x = torch.randint(0, 20000, (2, 4))

    # Custom position indices (non-sequential) to test ALiBi flexibility
    # First sequence positions: [0, 1, 2, 4], Second: [1, 3, 5, 7]
    pos = torch.tensor([[0, 1, 2, 4], [1, 3, 5, 7]])

    # Forward pass with custom positions
    logits = model(x, pos = pos)

@param('rotary_xpos', (True, False))
def test_custom_rotary_pos_emb(rotary_xpos):
    """
    Test custom position specification with Rotary Position Embeddings (RoPE).

    RoPE encodes position information by rotating query and key vectors.
    This test validates that providing explicit sequential position indices
    produces the same results as the default behavior. Also tests with
    and without xPos (extrapolation for longer sequences).

    Args:
        rotary_xpos: If True, enable xPos for better length extrapolation
    """
    from einops import repeat

    # Create decoder with Rotary Position Embeddings
    model = TransformerWrapper(
        num_tokens = 20_000,             # Vocabulary size
        max_seq_len = 1024,              # Maximum sequence length
        attn_layers = Decoder(
            dim = 512,                   # Model dimension
            depth = 2,                   # Number of decoder layers
            heads = 8,                   # Number of attention heads
            rotary_pos_emb = True,       # Enable Rotary Position Embeddings
            rotary_xpos = rotary_xpos    # Enable xPos if True
        )
    )

    # Create batch of random token sequences (batch_size=4, seq_len=4)
    x = torch.randint(0, 20000, (4, 4))

    # Create explicit sequential position indices: [[0,1,2,3], [0,1,2,3], ...]
    pos = repeat(torch.arange(0, 4), "n -> b n", b=4)

    # Forward pass with explicit positions
    logits1 = model(x, pos = pos)
    # Forward pass with default (implicit) positions
    logits2 = model(x)
    # Verify that explicit sequential positions match implicit behavior
    assert torch.allclose(logits1, logits2)

@param('flash', (True, False))
def test_custom_alibi_across_heads(flash: bool):
    """
    Test per-head custom ALiBi slopes and position indices.

    This test validates ALiBi with different position indices for each
    attention head, allowing each head to have its own position sequence.
    Custom slopes control the position bias strength per head.

    Args:
        flash: If True, use Flash Attention; if False, use standard attention
    """
    # Create decoder with per-head ALiBi configuration
    model = Decoder(
        dim = 512,                       # Model dimension
        depth = 2,                       # Number of decoder layers
        heads = 2,                       # Number of attention heads
        alibi_pos_bias = True,           # Enable ALiBi position biases
        rel_pos_kwargs = dict(
            slopes = [1, 1]              # ALiBi slope for each head
        ),
        attn_flash = flash               # Use Flash Attention if True
    )

    # Create batch of random embeddings (batch_size=2, seq_len=4, dim=512)
    x = torch.randn(2, 4, 512)

    # Per-head custom position indices: shape (batch, heads, seq_len)
    # Each head in each batch element has its own position sequence
    pos = torch.tensor([
        [[0, 1, 2, 4], [1, 3, 5, 7]],    # Batch 0: head 0 and head 1 positions
        [[2, 3, 4, 5], [6, 8, 9, 10]]    # Batch 1: head 0 and head 1 positions
    ])

    # Forward pass with per-head positions
    embed = model(x, pos = pos)

@param('embedder_type', ('embedding', 'none', 'custom'))
def test_embedder(embedder_type):
    """
    Test custom token embedding modules.

    This test validates that TransformerWrapper can work with different
    types of embedding modules: standard PyTorch Embedding, None (uses
    default), and custom embedders that may require additional inputs.

    Args:
        embedder_type: Type of embedder to test ('embedding', 'none', or 'custom')
    """
    num_tokens = 20000
    dim = 128
    token_emb_kwargs = {}

    if embedder_type == 'embedding':
        # Use standard PyTorch embedding layer
        embedder = nn.Embedding(num_tokens, dim)
    elif embedder_type == 'none':
        # Let TransformerWrapper create default embedding
        embedder = None
    else:
        # Test custom embedder that requires additional inputs
        class CustomEmbedder(Module):
            """
            Custom embedder that sums two separate embeddings.

            This validates that we can pass additional inputs to the embedder's
            forward pass through token_emb_kwargs without breaking the model.
            """
            def __init__(self, num_tokens, dim):
                super().__init__()
                self.embed_x = nn.Embedding(num_tokens, dim)  # Primary embedding
                self.embed_y = nn.Embedding(num_tokens, dim)  # Secondary embedding

            def forward(self, x, y):
                # Combine both embeddings by addition
                return self.embed_x(x) + self.embed_y(y)

            def init_(self):
                # Required interface for initialization
                pass

        embedder = CustomEmbedder(num_tokens, dim)
        # Provide additional 'y' input required by custom embedder
        token_emb_kwargs['y'] = torch.randint(0, num_tokens, (2, 1024))

    # Create model with specified embedder type
    model = TransformerWrapper(
        num_tokens = num_tokens,         # Vocabulary size
        max_seq_len = 1024,              # Maximum sequence length
        attn_layers = Decoder(
            dim = dim,                   # Model dimension
            depth = 6,                   # Number of decoder layers
            heads = 8,                   # Number of attention heads
        ),
        token_emb = embedder,            # Custom or standard embedder
    )

    # Create batch of random token sequences (batch_size=2, seq_len=1024)
    x = torch.randint(0, 20000, (2, 1024))

    # Forward pass with optional embedder kwargs
    output = model(x, token_emb_kwargs=token_emb_kwargs)
    # Verify output has correct shape: (batch_size, seq_len, num_tokens)
    assert output.shape == (2, 1024, 20000)


@param("to_logits", ('linear', 'none', 'pointer'))
def test_to_logits(to_logits):
    """
    Test custom logits projection layers.

    This test validates that TransformerWrapper can use different methods
    to convert hidden states to logits: standard linear layer, default
    (created internally), or custom modules like pointer networks.

    Args:
        to_logits: Type of logits layer ('linear', 'none', or 'pointer')
    """
    num_tokens = 20000
    dim = 128

    to_logits_kwargs = {}

    if to_logits == 'linear':
        # Use standard linear layer without bias
        logit_mapper = LinearNoBias(dim, num_tokens)
    elif to_logits == 'none':
        # Let TransformerWrapper create default logits projection
        logit_mapper = None
    else:
        # Test custom pointer network logits module
        class PointerNetworkLogits(Module):
            """
            Pointer network that computes logits via attention to input embeddings.

            Instead of a fixed output vocabulary, this points to positions in
            the input by computing similarity between model embeddings and
            input embeddings. Useful for tasks like summarization or copying.
            """
            def __init__(self, dim):
                super().__init__()
                self.proj_to_pointers = nn.Linear(dim, dim)

            def forward(self, model_embeddings, input_embeddings):
                # Project model embeddings to pointer space
                pointers = self.proj_to_pointers(model_embeddings)
                # Compute attention scores to input positions
                logits = torch.matmul(pointers, input_embeddings.permute(0, 2, 1))
                return logits

        logit_mapper = PointerNetworkLogits(dim)
        # Provide input embeddings required by pointer network
        to_logits_kwargs['input_embeddings'] = torch.randn(2, 20000, dim)

    # Create model with specified logits projection
    model = TransformerWrapper(
        num_tokens = num_tokens,         # Vocabulary size
        max_seq_len = 1024,              # Maximum sequence length
        attn_layers = Decoder(
            dim = dim,                   # Model dimension
            depth = 6,                   # Number of decoder layers
            heads = 8,                   # Number of attention heads
        ),
        to_logits = logit_mapper,        # Custom or standard logits layer
    )

    # Create batch of random token sequences (batch_size=2, seq_len=1024)
    x = torch.randint(0, num_tokens, (2, 1024))

    # Forward pass with optional logits kwargs
    output = model(x, to_logits_kwargs=to_logits_kwargs)

    # Verify output has correct shape: (batch_size, seq_len, num_tokens)
    assert output.shape == (2, 1024, 20000)

def test_laser():
    """
    Test LASER (Layer-Selective Rank Reduction) attention.

    LASER is a technique for selectively reducing the rank of attention
    layers, which can improve model efficiency and potentially performance
    by focusing on more important attention patterns.
    """
    # Create decoder with LASER attention enabled
    model = TransformerWrapper(
        num_tokens = 20000,              # Vocabulary size
        max_seq_len = 1024,              # Maximum sequence length
        attn_layers = Decoder(
            dim = 128,                   # Model dimension
            depth = 6,                   # Number of decoder layers
            heads = 8,                   # Number of attention heads
            attn_laser = True            # Enable LASER attention
        )
    )

    # Create batch of random token sequences (batch_size=2, seq_len=1024)
    x = torch.randint(0, 20000, (2, 1024))

    # Forward pass with LASER attention
    model(x)

@param('self_attn_custom_pos', (True, False))
@param('cross_attn_rotary', (True, False))
def test_cross_attn_rotary(
    self_attn_custom_pos: bool,
    cross_attn_rotary: bool
):
    """
    Test Rotary Position Embeddings in cross-attention context.

    This test validates that RoPE works correctly in cross-attention scenarios,
    where the model attends to a separate context sequence. Tests custom position
    indices for both self-attention and cross-attention independently.

    Args:
        self_attn_custom_pos: If True, provide custom positions for self-attention
        cross_attn_rotary: If True, provide custom positions for cross-attention
    """
    # Create input embeddings (batch=1, seq_len=64, dim=256)
    x = torch.randn((1, 64, 256))
    # Self-attention mask: all positions valid
    mask = torch.ones((1, 64)).bool()
    # Cross-attention context (batch=1, context_len=128, context_dim=512)
    context = torch.randn((1, 128, 512))
    # Cross-attention mask: all context positions valid
    context_mask = torch.ones((1, 128)).bool()

    # Create encoder with cross-attention and rotary position embeddings
    model = Encoder(
        dim = 256,                       # Model dimension
        depth = 4,                       # Number of encoder layers
        heads = 4,                       # Number of attention heads
        rotary_pos_emb = True,           # Enable RoPE for self-attention
        cross_attend = True,             # Enable cross-attention to context
        cross_attn_dim_context = 512     # Context dimension
    )

    # Self-attention positions: use custom if enabled, otherwise default
    pos = torch.arange(64) if self_attn_custom_pos else None
    # Cross-attention positions: use custom if enabled, otherwise default
    context_pos = torch.arange(128) if cross_attn_rotary else None

    # Forward pass with optional custom positions
    embed = model(
        x = x,
        mask = mask,
        context = context,
        pos = pos,                       # Self-attention positions
        context_pos = context_pos,       # Cross-attention positions
        context_mask = context_mask
    )

@param('tanh', (True, False))
def test_hyper_connections(tanh):
    """
    Test hyper connections (multiple residual streams).

    Hyper connections use multiple parallel residual streams instead of a
    single residual connection, allowing the model to maintain multiple
    information pathways. Optional tanh activation can be applied to control
    the magnitude of residual contributions.

    Args:
        tanh: If True, apply tanh activation to residual connections
    """
    # Create decoder with multiple residual streams (hyper connections)
    model = TransformerWrapper(
        num_tokens = 20000,              # Vocabulary size
        max_seq_len = 1024,              # Maximum sequence length
        attn_layers = Decoder(
            dim = 128,                   # Model dimension
            depth = 6,                   # Number of decoder layers
            heads = 8,                   # Number of attention heads
            num_residual_streams = 8,    # 8 parallel residual streams
            residual_fn_kwargs = dict(
                tanh = tanh              # Apply tanh activation if True
            )
        )
    )

    # Create batch of random token sequences (batch_size=2, seq_len=1024)
    x = torch.randint(0, 20000, (2, 1024))

    # Forward pass with hyper connections
    model(x)

@param('hybrid_axial_dim', (1, 4))
def test_hybrid(hybrid_axial_dim):
    """
    Test hybrid attention with RNN modules (mixing attention and recurrence).

    Hybrid attention alternates or combines transformer attention with RNN
    modules (like GRU or LSTM). This can capture both long-range dependencies
    (attention) and sequential patterns (RNN). The axial dimension controls
    how the sequence is folded for the RNN processing.

    Args:
        hybrid_axial_dim: Axial folding dimension for RNN processing (1 or 4)
    """
    from torch.nn import GRU

    # Create decoder with hybrid attention + GRU
    dec = TransformerWrapper(
        num_tokens = 20000,                      # Vocabulary size
        max_seq_len = 1024,                      # Maximum sequence length
        attn_layers = Decoder(
            dim = 128,                           # Model dimension
            depth = 6,                           # Number of decoder layers
            heads = 8,                           # Number of attention heads
            attn_dim_head = 64,                  # Dimension per attention head
            attn_hybrid_fold_axial_dim = hybrid_axial_dim,  # Folding dimension
            attn_hybrid_module = GRU(128, 64 * 8, batch_first = True)  # GRU module
        )
    )

    # Create batch of random token sequences (batch_size=2, seq_len=1024)
    x = torch.randint(0, 20000, (2, 1024))

    # Forward pass through hybrid decoder
    embed = dec(x)

    # Create encoder with hybrid attention + bidirectional GRU
    enc = TransformerWrapper(
        num_tokens = 20000,                      # Vocabulary size
        max_seq_len = 1024,                      # Maximum sequence length
        attn_layers = Encoder(
            dim = 128,                           # Model dimension
            depth = 6,                           # Number of encoder layers
            heads = 8,                           # Number of attention heads
            attn_dim_head = 64,                  # Dimension per attention head
            attn_hybrid_fold_axial_dim = hybrid_axial_dim,  # Folding dimension
            attn_hybrid_module = GRU(128, 64 * 4, batch_first = True, bidirectional = True)  # Bidirectional GRU
        )
    )

    # Create random attention mask (batch_size=2, seq_len=1024)
    mask = torch.randint(0, 2, (2, 1024)).bool()
    # Forward pass through hybrid encoder with mask
    embed = enc(x, mask = mask)

def test_hybrid_cache():
    """
    Test caching with hybrid attention (attention + RNN).

    This test validates that KV caching works correctly with hybrid models
    that combine attention and RNN modules. It compares parallel processing
    (all tokens at once) with sequential processing (using cache) to ensure
    they produce identical results.
    """
    from torch.nn import GRU

    # Create hybrid decoder (attention + GRU)
    model = TransformerWrapper(
        num_tokens = 20000,                      # Vocabulary size
        max_seq_len = 1024,                      # Maximum sequence length
        attn_layers = Decoder(
            dim = 128,                           # Model dimension
            depth = 6,                           # Number of decoder layers
            heads = 8,                           # Number of attention heads
            attn_dim_head = 64,                  # Dimension per attention head
            attn_hybrid_fold_axial_dim = 1,      # Folding dimension
            attn_hybrid_module = GRU(128, 64 * 8, batch_first = True)  # GRU module
        )
    )

    # Create small batch for testing (batch_size=2, seq_len=4)
    x = torch.randint(0, 20000, (2, 4))

    # Parallel processing: process all tokens at once
    out_parallel = model(x)

    # Sequential processing with caching:
    # Process first 3 tokens and cache intermediate states
    x_without_last = x[:, :-1]

    # Process first 3 tokens, returning cache
    out1, cache = model(x_without_last, return_intermediates = True)
    # Process all 4 tokens using cached states (only computes last token)
    out2 = model(x, cache = cache)

    # Concatenate sequential outputs
    out_seq = torch.cat((out1, out2), dim = 1)

    # Verify parallel and sequential processing produce identical results
    assert torch.allclose(out_parallel, out_seq, atol = 1e-5)

def test_caching_when_inputs_not_include_past():
    """
    Test incremental caching where each input is only the new tokens.

    This validates a more efficient caching mode where instead of passing
    all previous tokens plus new tokens, we only pass the new tokens and
    specify that the input doesn't include past cached content. This is
    more efficient for autoregressive generation.
    """
    from torch.nn import GRU

    # Create hybrid decoder with rotary position embeddings
    model = TransformerWrapper(
        num_tokens = 20000,                      # Vocabulary size
        max_seq_len = 1024,                      # Maximum sequence length
        attn_layers = Decoder(
            dim = 128,                           # Model dimension
            depth = 6,                           # Number of decoder layers
            heads = 8,                           # Number of attention heads
            attn_dim_head = 64,                  # Dimension per attention head
            rotary_pos_emb = True,               # Enable rotary position embeddings
            attn_hybrid_fold_axial_dim = 1,      # Folding dimension
            attn_hybrid_module = GRU(128, 64 * 8, batch_first = True)  # GRU module
        )
    )

    # Create test sequence (batch_size=2, seq_len=4)
    x = torch.randint(0, 20000, (2, 4))

    # Parallel: process all tokens at once
    out_parallel = model(x)

    # Split sequence into individual tokens for incremental processing
    x1, x2, x3 = x[:, :2], x[:, 2:3], x[:, 3:4]

    # Process first 2 tokens, get cache
    out1, cache = model(x1, return_intermediates = True)
    # Process token 3 only (input doesn't include cached tokens)
    out2, cache = model(x2, cache = cache, return_intermediates = True, input_not_include_cache = True)
    # Process token 4 only (input doesn't include cached tokens)
    out3, cache = model(x3, cache = cache, return_intermediates = True, input_not_include_cache = True)

    # Concatenate all incremental outputs
    out_seq = torch.cat((out1, out2, out3), dim = 1)

    # Verify incremental generation matches parallel processing
    assert torch.allclose(out_parallel, out_seq, atol = 1e-5)

def test_caching_when_inputs_not_include_past_continuous():
    """
    Test incremental caching with continuous (non-tokenized) inputs.

    Similar to test_caching_when_inputs_not_include_past but for continuous-valued
    inputs (e.g., embeddings) rather than discrete tokens. Validates that caching
    works correctly for models processing continuous data streams.
    """
    from torch.nn import GRU
    from x_transformers.continuous import ContinuousTransformerWrapper

    # Create continuous transformer (processes continuous embeddings, not tokens)
    model = ContinuousTransformerWrapper(
        dim_in = 77,                             # Input embedding dimension
        max_seq_len = 1024,                      # Maximum sequence length
        attn_layers = Decoder(
            dim = 128,                           # Model dimension
            depth = 6,                           # Number of decoder layers
            heads = 8,                           # Number of attention heads
            attn_dim_head = 64,                  # Dimension per attention head
            rotary_pos_emb = False,              # Disable rotary embeddings
            attn_hybrid_fold_axial_dim = 1,      # Folding dimension
            attn_hybrid_module = GRU(128, 64 * 8, batch_first = True)  # GRU module
        )
    )

    # Create continuous input (batch=1, seq_len=4, dim=77)
    x = torch.randn(1, 4, 77)

    # Parallel processing
    out_parallel = model(x)

    # Split for incremental processing
    x1, x2, x3 = x[:, :2], x[:, 2:3], x[:, 3:4]

    # Incremental processing with cache
    out1, cache = model(x1, return_intermediates = True)
    out2, cache = model(x2, cache = cache, return_intermediates = True, input_not_include_cache = True)
    out3, cache = model(x3, cache = cache, return_intermediates = True, input_not_include_cache = True)

    # Concatenate incremental outputs
    out_seq = torch.cat((out1, out2, out3), dim = 1)

    # Verify incremental matches parallel processing
    assert torch.allclose(out_parallel, out_seq, atol = 1e-5)

def test_multi_latent_attention():
    """
    Test multi-latent attention with learnable query/key-value latents.

    Multi-latent attention uses learnable latent tokens that attend to or are
    attended by the sequence. Separate latents for queries and key-values
    allow for more flexible attention patterns. RoPE can be applied to
    latent subheads for position awareness.
    """
    # Create decoder with multi-latent attention
    model = TransformerWrapper(
        num_tokens = 20000,                  # Vocabulary size
        max_seq_len = 1024,                  # Maximum sequence length
        attn_layers = Decoder(
            dim = 128,                       # Model dimension
            depth = 6,                       # Number of decoder layers
            heads = 8,                       # Number of attention heads
            attn_use_latent_q = True,        # Enable latent queries
            attn_dim_latent_q = 128,         # Dimension of latent queries
            attn_use_latent_kv = True,       # Enable latent key-values
            attn_dim_latent_kv = 128,        # Dimension of latent key-values
            attn_latent_rope_subheads = 4,   # Number of latent RoPE subheads
            rotary_pos_emb = False           # Disable main RoPE (using latent RoPE)
        )
    )

    # Create batch of random token sequences (batch_size=2, seq_len=1024)
    x = torch.randint(0, 20000, (2, 1024))

    # Forward pass with multi-latent attention
    model(x)

@param('num_residual_streams', (1, 4))
@param('integrate_layers', (False, True))
def test_lime(
    num_residual_streams,
    integrate_layers
):
    """
    Test LIME (Layer Integration via Multi-stream Ensembles).

    LIME uses multiple residual streams with optional layer integration,
    which allows information from different layers to be selectively combined.
    This tests both single and multi-stream configurations, with and without
    layer integration mechanisms.

    Args:
        num_residual_streams: Number of parallel residual streams (1 or 4)
        integrate_layers: If True, enable layer integration across streams
    """
    # Create decoder with LIME configuration
    model = TransformerWrapper(
        num_tokens = 20000,                      # Vocabulary size
        max_seq_len = 1024,                      # Maximum sequence length
        attn_layers = Decoder(
            dim = 128,                           # Model dimension
            depth = 6,                           # Number of decoder layers
            heads = 8,                           # Number of attention heads
            num_residual_streams = num_residual_streams,  # Number of residual streams
            integrate_layers = integrate_layers  # Enable layer integration if True
        )
    )

    # Create batch of random token sequences (batch_size=2, seq_len=1024)
    x = torch.randint(0, 20000, (2, 1024))

    # Forward pass with LIME architecture
    model(x)

@param('backward_ar_loss_weight', (1., 0.5))
@param('goal_suffix', (False, True))
@param('pred_distance', (False, True))
@param('variable_len', (False, True))
def test_belief_state_wrapper(
    backward_ar_loss_weight,
    goal_suffix,
    pred_distance,
    variable_len
):
    """
    Test Belief State Wrapper for bidirectional sequence modeling.

    Belief State Wrapper combines forward and backward models to create
    better representations by modeling sequences in both directions.
    It can optionally predict distances and generate with goal suffixes.
    Tests various configurations and generation modes.

    Args:
        backward_ar_loss_weight: Weight for backward autoregressive loss (1.0 or 0.5)
        goal_suffix: If True, test generation conditioned on goal suffix
        pred_distance: If True, enable distance prediction
        variable_len: If True, use variable-length sequences
    """
    from x_transformers.belief_state_wrapper import BeliefStateWrapper

    # Create forward (left-to-right) decoder model
    forward_model = TransformerWrapper(
        num_tokens = 20000,              # Vocabulary size
        max_seq_len = 1024,              # Maximum sequence length
        attn_layers = Decoder(
            dim = 512,                   # Model dimension
            depth = 6,                   # Number of decoder layers
            heads = 8,                   # Number of attention heads
            rotary_pos_emb = True        # Enable rotary position embeddings
        )
    )

    # Create backward (right-to-left) decoder model
    backward_model = TransformerWrapper(
        num_tokens = 20000,              # Vocabulary size
        max_seq_len = 1024,              # Maximum sequence length
        attn_layers = Decoder(
            dim = 512,                   # Model dimension
            depth = 6,                   # Number of decoder layers
            heads = 8,                   # Number of attention heads
            rotary_pos_emb = True        # Enable rotary position embeddings
        )
    )

    # Wrap forward and backward models together
    model = BeliefStateWrapper(
        forward_decoder = forward_model,
        backward_decoder = backward_model,
        backward_ar_loss_weight = backward_ar_loss_weight,  # Weight for backward loss
        pred_distance = pred_distance    # Enable distance prediction if True
    )

    # Create batch of random token sequences (batch_size=2, seq_len=16)
    seq = torch.randint(0, 20000, (2, 16))

    # Optional variable-length sequences
    lens = None
    if variable_len:
        # Random lengths for each sequence (between 4 and 16)
        lens = torch.randint(4, 16, (2,))

    # Compute bidirectional loss (backward pass happens automatically)
    loss = model(seq, lens = lens)
    loss.backward()

    # Test generation with optional goal suffix conditioning
    suffix = None
    if goal_suffix:
        # Goal suffix to condition generation on (batch_size=2, suffix_len=2)
        suffix = torch.randint(0, 20000, (2, 2))

    # Generate sequences conditioned on initial token and optional suffix
    sampled = model.generate_with_suffix_cond(seq[:, :1], 16, suffix = suffix)
    # Verify generated sequence has correct shape
    assert sampled.shape == (2, 16)

def test_dynamic_tanh():
    """
    Test dynamic tanh activation with learnable alpha parameter.

    Dynamic tanh uses a learnable scaling parameter (alpha) that allows the
    model to adaptively control the saturation point of the tanh activation.
    This can help with training stability and expressiveness compared to
    fixed tanh activation.
    """
    # Create decoder with dynamic tanh activation
    model = TransformerWrapper(
        num_tokens = 20000,                  # Vocabulary size
        max_seq_len = 1024,                  # Maximum sequence length
        attn_layers = Decoder(
            dim = 128,                       # Model dimension
            depth = 6,                       # Number of decoder layers
            heads = 8,                       # Number of attention heads
            use_dynamic_tanh = True,         # Enable dynamic tanh activation
            dynamic_tanh_init_alpha = 1.5    # Initial alpha value for tanh scaling
        )
    )

    # Create batch of random token sequences (batch_size=2, seq_len=1024)
    x = torch.randint(0, 20000, (2, 1024))

    # Forward pass with dynamic tanh
    model(x)

@param('var_length', (False, True))
def test_entropy_based_tokenizer(
    var_length
):
    """
    Test entropy-based dynamic tokenization.

    Entropy-based tokenizer adaptively segments sequences based on the model's
    prediction entropy. Low entropy (confident predictions) allows merging tokens,
    while high entropy (uncertain predictions) maintains fine granularity. Tests
    both fixed and variable-length sequences.

    Args:
        var_length: If True, use variable-length sequences with different lengths
    """
    from x_transformers.entropy_based_tokenizer import EntropyBasedTokenizer

    # Create decoder model for entropy-based tokenization
    model = TransformerWrapper(
        num_tokens = 20000,              # Vocabulary size
        max_seq_len = 1024,              # Maximum sequence length
        attn_layers = Decoder(
            dim = 128,                   # Model dimension
            depth = 6,                   # Number of decoder layers
            heads = 8,                   # Number of attention heads
            attn_dim_head = 64,          # Dimension per attention head
        )
    )

    # Create tokenizer that segments based on entropy threshold
    tokenizer = EntropyBasedTokenizer(model, entropy_threshold = 9.738)

    # Create batch of random token sequences (batch_size=2, seq_len=1024)
    seq = torch.randint(0, 20000, (2, 1024))

    # Optional variable-length sequences
    lens = None
    if var_length:
        # Different lengths for each sequence (between 512 and 768)
        lens = torch.randint(512, 768, (2,))

    # Segment sequences based on model entropy
    segmented_seq = tokenizer(seq, lens, return_segmented_seq = True)

    # Verify we get one segmented sequence per input sequence
    assert len(segmented_seq) == seq.shape[0]

    # Test that tokenizer can handle single sequence (no batch dimension)
    tokenizer(seq[0])

def test_entropy_based_tokenizer_max_token_len():
    """
    Test entropy-based tokenizer with maximum token length constraint.

    This test validates that the entropy-based tokenizer respects a maximum
    token size limit, ensuring that merged tokens don't exceed the specified
    maximum length. Useful for controlling computational costs.
    """
    from x_transformers.entropy_based_tokenizer import EntropyBasedTokenizer

    # Create decoder model
    model = TransformerWrapper(
        num_tokens = 20000,              # Vocabulary size
        max_seq_len = 1024,              # Maximum sequence length
        attn_layers = Decoder(
            dim = 128,                   # Model dimension
            depth = 6,                   # Number of decoder layers
            heads = 8,                   # Number of attention heads
            attn_dim_head = 64,          # Dimension per attention head
        )
    )

    # Create tokenizer with very high entropy threshold and max token size
    tokenizer = EntropyBasedTokenizer(
        model,
        entropy_threshold = 100,         # Very high threshold (encourages merging)
        max_token_size = 4               # Maximum merged token length
    )

    # Create test sequence (batch_size=1, seq_len=16)
    seq = torch.randint(0, 20000, (1, 16,))
    # Actual sequence length is 14
    lens = torch.tensor([14])

    # Get token lengths after entropy-based segmentation
    token_lengths = tokenizer(seq, lens = lens)

    # Verify no token exceeds maximum size
    assert token_lengths.amax().item() <= 4
    # Verify total length equals input length
    assert token_lengths.sum().item() == 14

def test_custom_ff_activation():
    """
    Test custom activation function in feedforward layers.

    This test validates that custom activation functions (e.g., Sigmoid instead
    of GELU or SwiGLU) can be used in the feedforward layers of the transformer.
    Allows experimentation with different activation functions.
    """
    # Create decoder with custom feedforward activation
    model = TransformerWrapper(
        num_tokens = 20000,                  # Vocabulary size
        max_seq_len = 1024,                  # Maximum sequence length
        attn_layers = Decoder(
            dim = 128,                       # Model dimension
            depth = 6,                       # Number of decoder layers
            heads = 8,                       # Number of attention heads
            attn_dim_head = 64,              # Dimension per attention head
            ff_custom_activation = nn.Sigmoid()  # Use Sigmoid activation instead of default
        )
    )

    # Create batch of random token sequences (batch_size=2, seq_len=1024)
    seq = torch.randint(0, 20000, (2, 1024))

    # Forward pass with custom activation
    logits = model(seq)

    # Verify output shape: (batch_size, seq_len, vocab_size)
    assert logits.shape == (2, 1024, 20000)

def test_ff_deep_embed():
    """
    Test deep embedding with feedforward layer before transformer.

    Deep embedding processes token embeddings through an additional feedforward
    layer before passing them to the transformer layers. This can help create
    richer initial representations and improve model capacity.
    """
    # Create decoder with deep embedding enabled
    model = TransformerWrapper(
        num_tokens = 20000,              # Vocabulary size
        max_seq_len = 1024,              # Maximum sequence length
        ff_deep_embed = True,            # Enable feedforward layer on embeddings
        attn_layers = Decoder(
            dim = 512,                   # Model dimension
            depth = 6,                   # Number of decoder layers
            heads = 8,                   # Number of attention heads
            rotary_pos_emb = True,       # Enable rotary position embeddings
        )
    )

    # Create batch of random token sequences (batch_size=2, seq_len=1024)
    seq = torch.randint(0, 20000, (2, 1024))

    # Forward pass with deep embeddings
    logits = model(seq)

    # Verify output shape: (batch_size, seq_len, vocab_size)
    assert logits.shape == (2, 1024, 20000)

@param('probabilistic', (False, True))
@param('cache_kv', (False, True))
@param('rollout_steps', (1, 4))
def test_continuous(
    probabilistic,
    cache_kv,
    rollout_steps
):
    """
    Test continuous-valued autoregressive transformer.

    Continuous transformers process and generate continuous-valued embeddings
    rather than discrete tokens. Useful for tasks like speech, music, or
    continuous control. Tests both deterministic and probabilistic outputs,
    with and without KV caching during generation.

    Args:
        probabilistic: If True, output probabilistic distribution over continuous values
        cache_kv: If True, use KV caching during generation for efficiency
        rollout_steps: Number of rollout steps for training (1 or 4)
    """
    from x_transformers import (
        ContinuousTransformerWrapper,
        Decoder,
        ContinuousAutoregressiveWrapper
    )

    # Create continuous transformer (processes continuous embeddings)
    model = ContinuousTransformerWrapper(
        dim_in = 777,                    # Input embedding dimension
        dim_out = 777,                   # Output embedding dimension
        max_seq_len = 1024,              # Maximum sequence length
        probabilistic = probabilistic,   # Output distribution if True, else deterministic
        attn_layers = Decoder(
            dim = 512,                   # Model dimension
            depth = 12,                  # Number of decoder layers
            heads = 8                    # Number of attention heads
        )
    )

    # Wrap with continuous autoregressive wrapper for training and generation
    model = ContinuousAutoregressiveWrapper(model)

    # Create mock continuous data (batch=1, seq_len=1024, dim=777)
    x = torch.randn((1, 1024, 777))
    # Attention mask: all positions valid
    mask = torch.ones(1, 1024).bool()

    # Training: compute loss with multiple rollout steps
    loss = model(x, mask = mask, rollout_steps = rollout_steps)
    loss.backward()

    # Generation: autoregressively generate continuous embeddings
    start_emb = torch.randn(1, 777)  # Initial embedding
    generated = model.generate(start_emb, 17, cache_kv = cache_kv)
    # Verify generated sequence has correct shape: (seq_len, dim)
    assert generated.shape == (17, 777)

@param('add_continuous_pred_head', (False, True))
def test_autoregressive_wrapper(
    add_continuous_pred_head
):
    """
    Test AutoregressiveWrapper for training language models.

    AutoregressiveWrapper handles the shifted input/target setup required
    for autoregressive language modeling. Optionally adds a continuous
    prediction head for auxiliary continuous-valued outputs.

    Args:
        add_continuous_pred_head: If True, add head for continuous predictions
    """
    from x_transformers import AutoregressiveWrapper

    # Create decoder with optional continuous prediction head
    model = TransformerWrapper(
        num_tokens = 20000,                      # Vocabulary size
        max_seq_len = 1024,                      # Maximum sequence length
        add_continuous_pred_head = add_continuous_pred_head,  # Continuous head if True
        attn_layers = Decoder(
            dim = 512,                           # Model dimension
            depth = 6,                           # Number of decoder layers
            heads = 8,                           # Number of attention heads
        )
    )

    # Create batch of random token sequences (batch_size=2, seq_len=1024)
    x = torch.randint(0, 20000, (2, 1024))

    # Wrap model for autoregressive training
    wrapper = AutoregressiveWrapper(model)
    # Compute autoregressive loss (shifts input/target automatically)
    loss = wrapper(x)

    # Backward pass
    loss.backward()

def test_prepend_embed():
    """
    Test prepending embeddings for conditioning.

    Prepend embeddings allow conditioning the model on arbitrary continuous
    representations (e.g., image embeddings, audio features) before the token
    sequence. This is useful for multimodal or conditioned generation tasks.
    Tests both training and generation with prepended embeddings.
    """
    from x_transformers import AutoregressiveWrapper

    # Create decoder model
    model = TransformerWrapper(
        num_tokens = 256,                # Vocabulary size
        max_seq_len = 1024,              # Maximum sequence length
        attn_layers = Decoder(
            dim = 512,                   # Model dimension
            depth = 12,                  # Number of decoder layers
            heads = 8                    # Number of attention heads
        )
    )

    # Wrap for autoregressive training and generation
    model = AutoregressiveWrapper(model)

    # Create batch of random token sequences (batch_size=2, seq_len=10)
    x = torch.randint(0, 256, (2, 10))
    # Prepend embeddings to condition on (batch_size=2, num_prepend=3, dim=512)
    prepend_embeds = torch.randn(2, 3, 512)
    # Mask for prepend embeddings (which prepended tokens are valid)
    prepend_mask = torch.randint(0, 2, (2, 3)).bool()

    # Training: compute loss with prepended conditioning embeddings
    loss = model(x, prepend_mask = prepend_mask, prepend_embeds = prepend_embeds)
    loss.backward()

    # Generation with KV caching
    sample = model.generate(
        prompts = x[:, :1],              # Initial token
        seq_len = 100,                   # Generate 100 tokens
        temperature = 0.,                # Greedy sampling
        prepend_embeds = prepend_embeds, # Conditioning embeddings
        prepend_mask = prepend_mask,     # Prepend mask
        cache_kv = True,                 # Use KV caching for efficiency
    )

    # Generation without KV caching (should produce same results)
    sample_no_cache = model.generate(
        prompts = x[:, :1],
        seq_len = 100,
        temperature = 0.,
        prepend_embeds = prepend_embeds,
        prepend_mask = prepend_mask,
        cache_kv = False,
    )

    # Verify cached and non-cached generation produce identical results
    assert torch.allclose(sample, sample_no_cache)

def add_attn_pool():
    """
    Test attention pooling to extract pooled tokens.

    Attention pooling uses learnable query tokens that attend to the sequence
    to extract a fixed number of summary tokens. These pooled tokens can be
    used for classification or as compact sequence representations.
    """
    # Create decoder with attention pooling
    model = TransformerWrapper(
        num_tokens = 256,                # Vocabulary size
        max_seq_len = 1024,              # Maximum sequence length
        attn_pool = True,                # Enable attention pooling
        num_pooled_tokens = 3,           # Number of pooled tokens to extract
        attn_layers = Decoder(
            dim = 512,                   # Model dimension
            depth = 12,                  # Number of decoder layers
            heads = 8                    # Number of attention heads
        ),
    )

    # Create batch of random token sequences (batch_size=1, seq_len=10)
    x = torch.randint(0, 256, (1, 10))

    # Forward pass with intermediate values returned
    logits, intermediates = model(x, return_intermediates = True)

    # Verify we get the correct number of pooled tokens
    assert intermediates.attn_pooled_tokens.shape[1] == 3

@param('keep_buffer_on_cpu', (False, True))
def test_up(
    keep_buffer_on_cpu
):
    """
    Test Universal Pretraining (UP) wrapper.

    UP wrapper enables continual pretraining by maintaining a buffer of
    data that the model continuously trains on. Buffer can be kept on CPU
    to save GPU memory for large-scale pretraining scenarios.

    Args:
        keep_buffer_on_cpu: If True, keep data buffer on CPU to save GPU memory
    """
    from x_transformers.up_wrapper import UniversalPretrainWrapper

    # Create decoder with attention pooling
    model = TransformerWrapper(
        num_tokens = 256,                # Vocabulary size
        max_seq_len = 1024,              # Maximum sequence length
        attn_pool = True,                # Enable attention pooling
        attn_layers = Decoder(
            dim = 512,                   # Model dimension
            depth = 12,                  # Number of decoder layers
            heads = 8                    # Number of attention heads
        ),
    )

    # Wrap for universal pretraining
    up_wrapper = UniversalPretrainWrapper(
        model,
        seq_len = 16,                            # Sequence length for pretraining
        keep_buffer_on_cpu = keep_buffer_on_cpu  # Buffer location (CPU or GPU)
    )

    # Training step (generates data internally from buffer)
    loss = up_wrapper()
    loss.backward()

@param('stochastic', (False, True))
def test_beam_search(stochastic):
    """
    Test beam search decoding for sequence generation.

    Beam search maintains multiple hypotheses (beams) during generation,
    selecting the overall best sequence rather than greedily choosing at
    each step. Stochastic beam search adds randomness for diversity.

    Args:
        stochastic: If True, use stochastic beam search; if False, deterministic
    """
    from x_transformers import TransformerWrapper, Decoder, AutoregressiveWrapper

    # Create decoder model
    model = TransformerWrapper(
        num_tokens = 256,                # Vocabulary size
        max_seq_len = 1024,              # Maximum sequence length
        attn_layers = Decoder(
            dim = 512,                   # Model dimension
            depth = 12,                  # Number of decoder layers
            heads = 8                    # Number of attention heads
        ),
    )

    # Create batch of prompts (batch_size=2, prompt_len=10)
    x = torch.randint(0, 256, (2, 10))

    # Wrap for autoregressive generation
    wrapper = AutoregressiveWrapper(model)

    # Generate using beam search with 4 beams, return best sequence
    generated = wrapper.beam_search(x[:, :1], 10, beams = 4, stochastic = stochastic)

    # Verify output shape: (batch_size, seq_len)
    assert generated.shape == (2, 10)

    # Generate with all beams and scores returned
    beams, scores = wrapper.beam_search(x[:, :1], 10, beams = 4, return_beams_and_scores = True, stochastic = stochastic)

    # Verify beams shape: (num_beams, batch_size, seq_len)
    assert beams.shape == (4, 2, 10)
    # Verify scores shape: (num_beams, batch_size)
    assert scores.shape == (4, 2)


@param('num_pooled_tokens', (1, 3))
@param('attn_pool_depth', (1, 3))
def test_attn_pooler(
    num_pooled_tokens,
    attn_pool_depth
):
    """
    Test attention pooler with configurable pooling depth and token count.

    The attention pooler uses multiple layers of cross-attention to progressively
    refine pooled tokens. Tests different numbers of pooled tokens and different
    pooling depths to validate flexible configuration.

    Args:
        num_pooled_tokens: Number of tokens to pool from sequence (1 or 3)
        attn_pool_depth: Number of pooling layers to apply (1 or 3)
    """
    # Create encoder with configurable attention pooling
    model = TransformerWrapper(
        num_tokens = 256,                        # Vocabulary size
        max_seq_len = 1024,                      # Maximum sequence length
        attn_pool = True,                        # Enable attention pooling
        num_pooled_tokens = num_pooled_tokens,   # Number of pooled tokens
        attn_pool_depth = attn_pool_depth,       # Pooling layer depth
        dim_pooled_tokens = 77,                  # Dimension of pooled tokens
        attn_layers = Encoder(
            dim = 512,                           # Model dimension
            depth = 12,                          # Number of encoder layers
            heads = 8,                           # Number of attention heads
            attn_value_rmsnorm = True            # Apply RMSNorm to attention values
        ),
    )

    # Create batch of random token sequences (batch_size=2, seq_len=10)
    x = torch.randint(0, 256, (2, 10))

    # Forward pass produces pooled tokens
    out = model(x)

    # Verify output shape: (batch_size, num_pooled_tokens, pooled_dim)
    assert out.shape == (2, num_pooled_tokens, 77)

def test_prompts_given_as_list_tensor():
    """
    Test generation with variable-length prompts provided as a list.

    This test validates that the model can handle a list of prompts with
    different lengths (ragged batching). Each prompt can have a different
    length, and they are automatically batched and padded internally.
    """
    from x_transformers import AutoregressiveWrapper

    # Create decoder model
    model = TransformerWrapper(
        num_tokens = 20000,              # Vocabulary size
        max_seq_len = 1024,              # Maximum sequence length
        attn_layers = Decoder(
            dim = 512,                   # Model dimension
            depth = 12,                  # Number of decoder layers
            heads = 8                    # Number of attention heads
        )
    )

    # Wrap for autoregressive generation
    wrapped = AutoregressiveWrapper(model)

    # Training with fixed-length batch (batch_size=3, seq_len=1024)
    seq = torch.randint(0, 20000, (3, 1024))

    # Compute loss
    loss = wrapped(seq)
    loss.backward()

    # Generation with variable-length prompts (list of tensors)
    # Each prompt has different length: 3, 5, 2, and 7 tokens
    sampled = wrapped.generate([
        torch.randint(0, 20000, (3,)),   # Prompt 1: 3 tokens
        torch.randint(0, 20000, (5,)),   # Prompt 2: 5 tokens
        torch.randint(0, 20000, (2,)),   # Prompt 3: 2 tokens
        torch.randint(0, 20000, (7,)),   # Prompt 4: 7 tokens
    ], 256)

    # Verify output shape: (num_prompts, target_seq_len)
    assert sampled.shape == (4, 256)

def test_external_key_values():
    """
    Test providing external key-value pairs for attention.

    External key-values allow the model to attend to additional context
    beyond the input sequence. This is useful for retrieval-augmented
    generation or conditioning on external memory. Tests that external
    key-values can be provided per layer.
    """
    from x_transformers import AutoregressiveWrapper

    # Create decoder model
    model = TransformerWrapper(
        num_tokens = 20000,              # Vocabulary size
        max_seq_len = 1024,              # Maximum sequence length
        attn_layers = Decoder(
            dim = 512,                   # Model dimension
            depth = 3,                   # Number of decoder layers
            heads = 8,                   # Number of attention heads
            attn_dim_head = 16           # Dimension per attention head
        )
    )

    # Create batch of random token sequences (batch_size=3, seq_len=1024)
    seq = torch.randint(0, 20000, (3, 1024))

    # External key-value pairs for first 2 layers
    # Each layer gets (keys, values) of shape (batch, heads, kv_len, dim_head)
    key_values = [
        (torch.randn(3, 2, 32, 16), torch.randn(3, 2, 32, 16)),  # Layer 0 KV
        (torch.randn(3, 2, 32, 16), torch.randn(3, 2, 32, 16)),  # Layer 1 KV
    ]

    # Mask for external key-values (batch_size=3, kv_len=32)
    additional_kv_mask = torch.randint(0, 2, (3, 32)).bool()

    # Forward pass with external key-values
    logits = model(seq, self_attn_additional_kv = key_values, additional_kv_mask = additional_kv_mask)

def test_learned_head_attn_sink():
    """
    Test learned attention sink tokens per head.

    Attention sink tokens are learnable tokens that each attention head can
    attend to, acting as a form of learned bias or memory. This can help
    stabilize attention and provide a default attention target.
    """
    # Create decoder with learned attention sink tokens
    model = TransformerWrapper(
        num_tokens = 20000,                  # Vocabulary size
        max_seq_len = 1024,                  # Maximum sequence length
        attn_layers = Decoder(
            dim = 512,                       # Model dimension
            depth = 12,                      # Number of decoder layers
            heads = 8,                       # Number of attention heads
            attn_head_learned_sink = True    # Enable learned sink tokens per head
        )
    )

    # Create batch of random token sequences (batch_size=3, seq_len=1024)
    seq = torch.randint(0, 20000, (3, 1024))

    # Forward pass with learned attention sinks
    logits = model(seq)

def test_accept_layer_intermediates():
    """
    Test using layer intermediates as additional key-values for another model.

    This test validates a pattern useful for hierarchical models or
    vision-language models, where one model's intermediate representations
    are used as additional context for another model to attend to.
    Tests detaching intermediates to prevent gradient flow.
    """
    from x_transformers import TransformerWrapper, Decoder, AutoregressiveWrapper

    # Create vision-language model (or first-stage model)
    vlm = TransformerWrapper(
        num_tokens = 20000,              # Vocabulary size
        max_seq_len = 1024,              # Maximum sequence length
        attn_layers = Decoder(
            dim = 512,                   # Model dimension
            depth = 3,                   # Number of decoder layers
            heads = 4,                   # Number of attention heads
        )
    )

    # Create input sequence and mask for VLM
    seq = torch.randint(0, 20000, (3, 1024))
    mask = torch.randint(0, 2, (3, 1024)).bool()

    # Get intermediates from VLM (layer-wise key-value pairs)
    _, intermediates = vlm(seq, return_intermediates = True)

    # Create action model (or second-stage model)
    action_model = Decoder(
        dim = 512,                       # Model dimension
        depth = 6,                       # Number of decoder layers
        heads = 8,                       # Number of attention heads
    )

    # Create continuous input for action model (batch=3, seq=32, dim=512)
    seq = torch.randn(3, 32, 512)

    # Action model attends to VLM intermediates as additional context
    embeds = action_model(
        seq,
        self_attn_additional_kv = intermediates,  # VLM layer intermediates
        detach_additional_kv = True,              # Detach to stop gradients
        additional_kv_mask = mask                 # Mask from VLM
    )

    # Verify output shape: (batch_size, seq_len, dim)
    assert embeds.shape == (3, 32, 512)

@param('use_loss_weight', (False, True))
def test_simple_mdlm(
    use_loss_weight
):
    """
    Test Simple Masked Discrete Language Model (MDLM).

    MDLM is a non-autoregressive model that predicts masked tokens.
    Simple MDLM uses a simplified masking strategy. Optional loss weighting
    can emphasize certain tokens during training.

    Args:
        use_loss_weight: If True, use simple MDLM loss weighting scheme
    """
    from x_transformers.nonautoregressive_wrapper import NonAutoregressiveWrapper

    # Create encoder model (non-autoregressive, no causal masking)
    # Note: vocab size is 257 (256 tokens + 1 mask token)
    model = TransformerWrapper(
        num_tokens = 256 + 1,            # Vocabulary + mask token
        max_seq_len = 1024,              # Maximum sequence length
        attn_layers = Encoder(
            dim = 512,                   # Model dimension
            depth = 4,                   # Number of encoder layers
            rotary_pos_emb = True        # Enable rotary position embeddings
        )
    )

    # Wrap for non-autoregressive masked language modeling
    nar = NonAutoregressiveWrapper(
        model,
        mask_id = 256,                            # ID of mask token
        use_simple_mdlm_loss_weight = use_loss_weight  # Use loss weighting if True
    )

    # Create batch of random token sequences (batch_size=1, seq_len=1024)
    seq = torch.randint(0, 256, (1, 1024))

    # Compute masked language modeling loss
    loss = nar(seq)
    loss.loss.backward()

def test_qk_clip_attn():
    """
    Test query-key clipping in attention layer.

    QK clipping limits the magnitude of query-key dot products before
    softmax, helping prevent attention from becoming too peaked or
    overconfident. This test validates the clipping mechanism.
    """
    from x_transformers import Attention

    # Create random input (batch=1, seq=1024, dim=512)
    x = torch.randn(1, 1024, 512)

    # Create attention module
    attn = Attention(dim = 512, dim_out = 384)

    # Forward pass with intermediates returned
    out, intermediates = attn(x, return_intermediates = True)

    # Apply query-key clipping to intermediates
    attn.qk_clip_(intermediates, tau = 100)

def test_qk_clip_attn_layers():
    """
    Test query-key clipping across all attention layers in a model.

    This test validates that QK clipping can be applied to all attention
    layers in a multi-layer transformer, useful for post-training analysis
    or optimization.
    """
    from x_transformers import TransformerWrapper, Decoder

    # Create decoder model
    model = TransformerWrapper(
        num_tokens = 256,                # Vocabulary size
        max_seq_len = 1024,              # Maximum sequence length
        attn_layers = Decoder(dim = 512, depth = 2)  # 2 decoder layers
    )

    # Create random sequence (batch=1, seq=1024)
    seq = torch.randint(0, 256, (1, 1024))

    # Forward pass with intermediates returned
    out, intermediates = model(seq, return_intermediates = True)

    # Apply QK clipping to all attention layers
    model.attn_qk_clip_(intermediates)

def test_vae():
    """
    Test GPT-VAE (Variational Autoencoder based on GPT architecture).

    GPT-VAE combines autoregressive language modeling with variational
    autoencoders, learning latent representations of sequences. Can
    generate sequences conditioned on latent codes extracted from
    style sequences.
    """
    from x_transformers.gpt_vae import GPTVAE

    # Create GPT-VAE model
    model = GPTVAE(
        num_tokens = 256,                # Vocabulary size
        max_seq_len = 1024,              # Maximum sequence length
        dim = 512,                       # Model dimension
        depth = 4,                       # Decoder depth
        enc_depth = 2                    # Encoder depth for latent extraction
    )

    # Create random sequence for training (batch=1, seq=1024)
    seq = torch.randint(0, 256, (1, 1024))

    # Compute VAE loss (reconstruction + KL divergence)
    loss = model(seq)
    loss.backward()

    # Create style sequence to extract latents from
    style = torch.randint(0, 256, (1, 1024))

    # Generate sequence conditioned on style latents
    out = model.generate(seq[:, :512], 512, seq_for_latents = style)

def test_muon_params():
    """
    Test extraction of Muon optimizer parameters.

    Muon optimizer is designed for large-scale training and requires
    identifying specific parameter groups. This test validates that
    Attention, FeedForward, and Encoder modules correctly expose
    their Muon-eligible parameters.
    """
    from x_transformers import Attention, FeedForward, Encoder

    # Test Attention module
    attn = Attention(dim = 512, dim_out = 384)
    # Verify it has 2 Muon parameters (typically Q and K projections)
    assert len(list(attn.muon_parameters())) == 2

    # Test FeedForward module
    ff = FeedForward(dim = 512)
    # Verify it has 2 Muon parameters (typically up and down projections)
    assert len(list(ff.muon_parameters())) == 2

    # Test Encoder module
    enc = Encoder(dim = 512, depth = 2)
    # Verify encoder has Muon parameters from all layers
    assert len(enc.muon_parameters()) > 0

def test_stochastic_attn():
    """
    Test stochastic attention using Gumbel-Softmax.

    Stochastic attention uses Gumbel-Softmax to create discrete attention
    patterns while maintaining differentiability. This enables training
    with discrete attention decisions and computing log probabilities.
    """
    from x_transformers import Attention

    # Create attention with Gumbel-Softmax enabled
    attn = Attention(dim = 512, gumbel_softmax = True)
    # Forward pass with random input
    out, intermediate = attn(torch.randn(1, 1024, 512), return_intermediates = True)

    # Verify output shape
    assert out.shape == (1, 1024, 512)

    # Extract log probabilities from hard (discrete) attention decisions
    from x_transformers.attend import log_prob_from_hard_attend
    log_probs = log_prob_from_hard_attend(intermediate)
    # Verify log_probs shape: (batch, heads, seq_len)
    assert log_probs.shape == (1, 8, 1024)

@param('head_learned_sink', (True, False))
def test_attn_negative_weights(
    head_learned_sink
):
    """
    Test attention with signed (negative) weights using CoG mechanism.

    CoG (Context-Oriented Grouping) attention allows negative attention
    weights, enabling the model to actively suppress certain positions.
    This test validates signed attention with learned sink tokens.

    Args:
        head_learned_sink: If True, enable learned sink tokens per head
    """
    from x_transformers import TransformerWrapper, Decoder

    # Create decoder with signed attention weights
    model = TransformerWrapper(
        num_tokens = 256,                # Vocabulary size
        max_seq_len = 1024,              # Maximum sequence length
        attn_layers = Decoder(
            dim = 512,                   # Model dimension
            depth = 12,                  # Number of decoder layers
            heads = 8,                   # Number of attention heads
            attn_cog_signed = True,      # Enable signed (negative) attention weights
            attn_head_learned_sink = True  # Enable learned sink tokens
        ),
    )

    # Create random sequence (batch=1, seq=10)
    x = torch.randint(0, 256, (1, 10))

    # Forward pass with signed attention
    logits = model(x)

@param('per_token_latents', (False, True))
@param('dec_head_depth', (0, 4))
@param('separate_seq_for_latents', (False, True))
def test_free(
    dec_head_depth,
    per_token_latents,
    separate_seq_for_latents
):
    """
    Test FREE (Factorized Randomized Embedding Ensemble) Transformer.

    FREE uses discrete latent codes with VQ (Vector Quantization) to
    create structured latent representations. Tests various configurations:
    per-token vs sequence-level latents, decoder head depth, and using
    separate sequences for latent extraction.

    Args:
        dec_head_depth: Decoder head layers (0 or 4)
        per_token_latents: If True, use per-token latents; else sequence-level
        separate_seq_for_latents: If True, use separate sequence for latent extraction
    """
    from x_transformers.free_transformer import FreeTransformer

    # Create FREE transformer with discrete latents
    model = FreeTransformer(
        num_tokens = 256,                    # Vocabulary size
        max_seq_len = 1024,                  # Maximum sequence length
        dim = 512,                           # Model dimension
        heads = 8,                           # Number of attention heads
        dec_head_depth = dec_head_depth,     # Decoder head layers
        dec_tail_depth = 4,                  # Decoder tail layers
        enc_depth = 2,                       # Encoder depth for latents
        kl_loss_weight = 1.,                 # KL divergence loss weight
        per_token_latents = per_token_latents,  # Per-token or sequence-level latents
        latent_bits = 8                      # 8 bits = 256 latent codes
    )

    # Create random sequence for training (batch=1, seq=1024)
    seq = torch.randint(0, 256, (1, 1024))

    # Optional separate sequence for latent extraction
    separate_seq_for_latents = torch.randint(0, 256, (1, 32)) if separate_seq_for_latents else None

    # Compute total loss (autoregressive + auxiliary latent loss)
    loss, (ar_loss, aux_loss) = model(seq, separate_seq_for_latents, return_all_losses = True)
    loss.backward()

    # Verify auxiliary loss is scalar
    assert aux_loss.numel() == 1

    # Generate sequence conditioned on random latent code
    rand_indices = torch.randint(0, 2 ** 8, ())  # Random latent index
    generated = model.generate(seq[:, :1], 32, latents = rand_indices)

    # Verify generated sequence shape
    assert generated.shape == (1, 32)

def test_kv_input_residual():
    """
    Test key-value residual connections in cross-attention.

    KV residuals add conditioning information directly to cross-attention
    key-value pairs, allowing external signals to modulate cross-attention.
    This is useful for multi-layer conditioning or hierarchical models.
    """
    # Create decoder with cross-attention capability
    attn = Decoder(
        dim = 256,                   # Model dimension
        depth = 2,                   # Number of decoder layers
        heads = 4,                   # Number of attention heads
        cross_attend = True          # Enable cross-attention
    )

    # Create input tokens (batch=3, seq=32, dim=256)
    tokens = torch.randn(3, 32, 256)
    # Create cross-attention context (batch=3, context_len=64, dim=256)
    context = torch.randn(3, 64, 256)

    # Create KV residuals: (num_layers, batch, context_len, dim)
    # These residuals are added to cross-attention key-values at each layer
    condition = torch.randn(2, 3, 64, 256)

    # Forward pass with KV residuals
    out = attn(tokens, context = context, cross_attn_kv_residuals = condition)

    # Verify output has same shape as input tokens
    assert tokens.shape == out.shape

@param('orthog_project', (False, True))
@param('orthog_project_per_head', (False, True))
def test_belief_attn(
    orthog_project,
    orthog_project_per_head
):
    """
    Test belief attention with orthogonal value projections.

    Belief attention uses orthogonal projections on attention values to
    ensure different attention heads capture diverse, non-redundant features.
    Tests both shared and per-head orthogonal projections with grouped
    query attention (GQA).

    Args:
        orthog_project: If True, apply orthogonal projection to values
        orthog_project_per_head: If True, use separate projection per head
    """
    from x_transformers import TransformerWrapper, Decoder

    # Create decoder with belief attention (orthogonal value projections)
    model = TransformerWrapper(
        num_tokens = 256,                                    # Vocabulary size
        max_seq_len = 1024,                                  # Maximum sequence length
        attn_layers = Decoder(
            dim = 512,                                       # Model dimension
            depth = 6,                                       # Number of decoder layers
            heads = 8,                                       # Number of query heads
            attn_kv_heads = 4,                               # Number of key-value heads (GQA)
            rotary_pos_emb = True,                           # Enable rotary position embeddings
            attn_orthog_projected_values = orthog_project,   # Enable orthogonal value projection
            attn_orthog_projected_values_per_head = orthog_project_per_head  # Per-head projection
        )
    )

    # Create random sequence (batch=1, seq=10)
    x = torch.randint(0, 256, (1, 10))

    # Forward pass with belief attention (orthogonal projections)
    logits = model(x)
