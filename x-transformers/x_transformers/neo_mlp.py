"""
NeoMLP: Neural Network Architecture using Message Passing instead of Traditional MLP

This module implements the NeoMLP architecture from the paper:
"Learning to Learn with Generative Models of Neural Network Checkpoints"
https://openreview.net/forum?id=A8Vuf2e8y6
https://haian-jin.github.io/projects/LVSM/

The key innovation is replacing traditional MLP hidden layers with nodes that communicate
via self-attention (message passing), treating the network as a fully connected graph.
"""

from collections import namedtuple

import torch
from torch import nn, tensor, pi, is_tensor
import torch.nn.functional as F
from torch.nn import Module, ModuleList

from einops import rearrange, repeat, einsum, pack, unpack

from x_transformers.x_transformers import (
    Encoder
)

# helpers

def exists(v):
    """
    Check if a value is not None.

    Args:
        v: Any value to check

    Returns:
        bool: True if v is not None, False otherwise
    """
    return v is not None

def default(v, d):
    """
    Return the value v if it exists, otherwise return default value d.

    Args:
        v: The value to check
        d: The default value to return if v is None

    Returns:
        Either v or d depending on whether v exists
    """
    return v if exists(v) else d

# random fourier

class RandomFourierEmbed(Module):
    """
    Random Fourier Feature Embedding for continuous values.

    This layer projects continuous input values through random Fourier features,
    which helps the network learn better representations of continuous features.
    The projection weights are frozen (not trainable) to maintain the random
    Fourier property.

    Args:
        dim (int): Dimension of the output embedding

    Input:
        times: Tensor of continuous values of any shape

    Output:
        Tensor: Cosine-transformed random projections of shape (..., dim)
    """

    def __init__(self, dim):
        super().__init__()
        # Linear projection with frozen random weights for Fourier features
        self.proj = nn.Linear(1, dim)
        self.proj.requires_grad_(False)

    def forward(
        self,
        times,
    ):
        """
        Forward pass applying random Fourier features.

        Args:
            times: Continuous values to embed

        Returns:
            Tensor: cos(2π * random_projection(times))
        """
        # Reshape to add feature dimension
        times = rearrange(times, '... -> ... 1')
        # Apply random projection
        rand_proj = self.proj(times)
        # Apply cosine transformation for Fourier features
        return torch.cos(2 * pi * rand_proj)

# class

class NeoMLP(Module):
    """
    NeoMLP: A novel neural network architecture that replaces traditional MLP layers
    with self-attention based message passing.

    References:
        https://openreview.net/forum?id=A8Vuf2e8y6
        https://haian-jin.github.io/projects/LVSM/

    Instead of using traditional hidden layers with matrix multiplications, NeoMLP
    represents the network as a graph where:
    - Input features are nodes
    - Hidden units are nodes
    - Output units are nodes
    All nodes communicate via self-attention (message passing), treating the network
    as a fully connected graph.

    Args:
        dim_in (int): Number of input features
        dim_hidden (int): Number of hidden nodes/units
        dim_out (int): Number of output features
        dim_model (int): Embedding dimension for all nodes
        depth (int): Number of transformer layers for message passing
        encoder_kwargs (dict): Additional arguments for the Encoder (attention layers).
            Defaults to {'attn_dim_head': 16, 'heads': 4}

    Attributes:
        input_embed: Learnable embeddings for input nodes
        hidden_embed: Learnable embeddings for hidden nodes
        output_embed: Learnable embeddings for output nodes
        random_fourier: Random Fourier feature transformation for continuous inputs
        transformer: Encoder that performs message passing via self-attention
        to_output_weights: Output projection weights
        to_output_bias: Output projection bias
    """

    def __init__(
        self,
        *,
        dim_in,
        dim_hidden,
        dim_out,
        dim_model,
        depth,
        encoder_kwargs: dict = dict(
            attn_dim_head = 16,
            heads = 4
        )
    ):
        super().__init__()

        # input and output embeddings

        # Learnable embedding parameters for each type of node
        self.input_embed = nn.Parameter(torch.zeros(dim_in, dim_model))
        self.hidden_embed = nn.Parameter(torch.zeros(dim_hidden, dim_model))
        self.output_embed = nn.Parameter(torch.zeros(dim_out, dim_model))

        # Initialize embeddings with small random values (std=0.02)
        nn.init.normal_(self.input_embed, std = 0.02)
        nn.init.normal_(self.hidden_embed, std = 0.02)
        nn.init.normal_(self.output_embed, std = 0.02)

        # they use random fourier for continuous features

        # Transform continuous input features using random Fourier features
        # followed by a linear layer for additional expressiveness
        self.random_fourier = nn.Sequential(
            RandomFourierEmbed(dim_model),
            nn.Linear(dim_model, dim_model)
        )

        # hidden dimensions of mlp replaced with nodes with message passing
        # which comes back to self attention as a fully connected graph.

        # Transformer encoder performs message passing between all nodes
        # This replaces traditional MLP hidden layer computations
        self.transformer = Encoder(
            dim = dim_model,
            depth = depth,
            **encoder_kwargs
        )

        # output

        # Linear projection to produce final output values from output node embeddings
        self.to_output_weights = nn.Parameter(torch.randn(dim_out, dim_model))
        self.to_output_bias = nn.Parameter(torch.zeros(dim_out))

    def forward(
        self,
        x,
        return_embeds = False
    ):
        """
        Forward pass through the NeoMLP network.

        The forward pass consists of:
        1. Transform input features via random Fourier features
        2. Add transformed features to input embeddings
        3. Concatenate input, hidden, and output embeddings
        4. Apply transformer (message passing via self-attention)
        5. Extract and project output embeddings to get final predictions

        Args:
            x (Tensor): Input features of shape (batch, dim_in) or (dim_in,)
            return_embeds (bool): If True, return both output and intermediate embeddings.
                Defaults to False.

        Returns:
            If return_embeds is False:
                Tensor: Output predictions of shape (batch, dim_out) or (dim_out,)
            If return_embeds is True:
                tuple: (output, (input_embed, hidden_embed, output_embed))
                    - output: Tensor of shape (batch, dim_out) or (dim_out,)
                    - input_embed: Transformed input embeddings
                    - hidden_embed: Transformed hidden embeddings
                    - output_embed: Transformed output embeddings
        """
        # Check if input has no batch dimension
        no_batch = x.ndim == 1

        # Add batch dimension if input is 1D
        if no_batch:
            x = rearrange(x, '... -> 1 ...')

        batch = x.shape[0]

        # Transform input through random Fourier features
        fouriered_input = self.random_fourier(x)

        # add fouriered input to the input embedding

        # Combine Fourier-transformed input with learned input embeddings
        input_embed = fouriered_input + self.input_embed

        # Repeat hidden and output embeddings for each batch sample
        hidden_embed, output_embed = tuple(repeat(t, '... -> b ...', b = batch) for t in (self.hidden_embed, self.output_embed))

        # pack all the inputs into one string of tokens for self attention

        # Concatenate all node embeddings (input, hidden, output) into a single sequence
        # This creates a fully connected graph where all nodes can attend to each other
        embed, packed_shape = pack([input_embed, hidden_embed, output_embed], 'b * d')

        # attention is all you need

        # Apply transformer layers for message passing between all nodes
        embed = self.transformer(embed)

        # unpack

        # Separate the transformed embeddings back into input, hidden, and output nodes
        input_embed, hidden_embed, output_embed = unpack(embed, packed_shape, 'b * d')

        # project for output

        # Compute final output by projecting output node embeddings
        # Einstein summation: for each output node, compute weighted sum of its embedding
        output = einsum(output_embed, self.to_output_weights, 'b n d, n d -> b n')
        output = output + self.to_output_bias

        # Remove batch dimension if input had no batch dimension
        if no_batch:
            output = rearrange(output, '1 ... -> ...')

        # Return just the output if embeddings are not requested
        if not return_embeds:
            return output

        # Return both output and all transformed embeddings
        return output, (input_embed, hidden_embed, output_embed)
