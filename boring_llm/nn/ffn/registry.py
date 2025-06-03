from typing import Callable
from pydantic import Field
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from boring_llm.base.component_registry import ComponentTransform, ComponentRegistry


class FFNTransform(ComponentTransform):
    """Base class for FFN transformations"""
    
    @property
    def output_dim(self) -> int:
        """FFN transforms must specify output dimension"""
        raise NotImplementedError


ffn_registry = ComponentRegistry[FFNTransform]("FFN")


##############################
# FFN Transform
##############################

@ffn_registry.register("standard")
class StandardFFN(FFNTransform):
    """Standard feed-forward transformation"""
    
    def __init__(self, dim_model: int, inner_dim: int, activation: Callable = nn.GELU, 
                 no_bias: bool = False, **kwargs):
        super().__init__()
        self.proj = nn.Linear(dim_model, inner_dim, bias=not no_bias)
        self.act = activation() if callable(activation) and not isinstance(activation, nn.Module) else activation
        
    def apply(self, x: Tensor, **kwargs) -> Tensor:
        return self.act(self.proj(x))
    
    @property
    def output_dim(self) -> int:
        return self.proj.out_features


@ffn_registry.register("glu", {
    "mult_bias": (bool, Field(default=True, description="Whether to use multiplicative bias in GLU"))
})
class GLUFFN(FFNTransform):
    """Feed-forward transformation with Gated Linear Unit"""
    
    def __init__(self, dim_model: int, inner_dim: int, activation: Callable = nn.SiLU, 
                 mult_bias: bool = True, no_bias: bool = False, **kwargs):
        super().__init__()
        self.proj = nn.Linear(dim_model, inner_dim * 2, bias=not no_bias)
        self.act = activation() if callable(activation) and not isinstance(activation, nn.Module) else activation
        self.mult_bias = nn.Parameter(torch.ones(inner_dim)) if mult_bias else 1.0
        self._inner_dim = inner_dim
        
    def apply(self, x: Tensor, **kwargs) -> Tensor:
        value, gate = self.proj(x).chunk(2, dim=-1)
        return value * self.act(gate) * self.mult_bias
    
    @property
    def output_dim(self) -> int:
        return self._inner_dim


##############################
# MoE Router
##############################

@ffn_registry.register("soft_router", {
    "num_experts": (int, Field(default=8, description="Number of experts")),
    "top_k": (int, Field(default=2, description="Number of experts to route to")),
    "noise_std": (float, Field(default=1.0, description="Noise standard deviation for load balancing"))
})
class RouterTransform(FFNTransform):
    """Router/Gating network for MOE"""
    
    def __init__(self, dim_model: int, num_experts: int = 8, top_k: int = 2, 
                 noise_std: float = 1.0, no_bias: bool = False, **kwargs):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_std = noise_std
        
        # Main routing layer
        self.gate = nn.Linear(dim_model, num_experts, bias=not no_bias)
        # Noise layer for load balancing (only used if noise_std > 0)
        if noise_std > 0:
            self.noise_gate = nn.Linear(dim_model, num_experts, bias=not no_bias)
        else:
            self.noise_gate = None
        
    def apply(self, x: Tensor, **kwargs) -> tuple:
        """Returns routing weights and indices"""
        logits = self.gate(x)
        
        # Add noise for load balancing during training
        if self.training and self.noise_std > 0 and self.noise_gate is not None:
            noise_logits = self.noise_gate(x)
            noise = torch.randn_like(logits) * F.softplus(noise_logits) * self.noise_std
            logits = logits + noise
            
        # Get top-k experts
        top_k_logits, indices = torch.topk(logits, self.top_k, dim=-1)
        
        # Create sparse representation
        zeros = torch.full_like(logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        routing_weights = F.softmax(sparse_logits, dim=-1)
        
        return routing_weights, indices
    
    @property
    def output_dim(self) -> int:
        return self.num_experts


@ffn_registry.register("hard_router", {
    "num_experts": (int, Field(default=8, description="Number of experts")),
    "top_k": (int, Field(default=1, description="Number of experts to route to (typically 1 for hard routing)")),
    "temperature": (float, Field(default=1.0, description="Temperature for Gumbel softmax, usually lower than soft router")),
    "straight_through": (bool, Field(default=True, description="Use straight-through estimator"))
})
class HardRouterTransform(FFNTransform):
    """Hard Router/Gating network for MOE using discrete routing"""
    
    def __init__(self, dim_model: int, num_experts: int = 8, top_k: int = 1, 
                 temperature: float = 1.0, straight_through: bool = True, 
                 no_bias: bool = False, **kwargs):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k  # usually 1
        self.temperature = temperature
        self.straight_through = straight_through
        
        # Main routing layer
        self.gate = nn.Linear(dim_model, num_experts, bias=not no_bias)
        
    def apply(self, x: Tensor, **kwargs) -> tuple:
        """Returns hard routing weights and indices"""
        logits = self.gate(x)
        
        if self.training and self.straight_through:
            # Gumbel-Softmax with straight-through estimator
            gumbel_logits = logits + self._gumbel_noise(logits)
            soft_weights = F.softmax(gumbel_logits / self.temperature, dim=-1)
            
            # Hard selection with straight-through
            _, indices = torch.topk(soft_weights, self.top_k, dim=-1)
            hard_weights = torch.zeros_like(logits)
            hard_weights.scatter_(-1, indices, 1.0)
            
            # Straight-through: forward hard, backward soft
            routing_weights = hard_weights + soft_weights - soft_weights.detach()
        else:
            # Pure hard routing for inference
            _, indices = torch.topk(logits, self.top_k, dim=-1)
            routing_weights = torch.zeros_like(logits)
            routing_weights.scatter_(-1, indices, 1.0)
            
        return routing_weights, indices
    
    def _gumbel_noise(self, logits):
        uniform = torch.rand_like(logits)
        return -torch.log(-torch.log(uniform + 1e-20) + 1e-20)


##############################
# Post-processor
##############################

class BasePostProcessor(FFNTransform):
    """Base class for post-processors with common functionality"""
    
    def __init__(self, dim_model: int, inner_dim: int, dropout: float = 0.0, 
                 post_act_ln: bool = False, no_bias: bool = False, 
                 zero_init_output: bool = False, **kwargs):
        super().__init__()
        self.dim_model = dim_model
        self.inner_dim = inner_dim
        
        # Pre-projection layers
        self.pre_layers = nn.ModuleList()
        if post_act_ln: 
            self.pre_layers.append(nn.LayerNorm(inner_dim))
        if dropout > 0: 
            self.pre_layers.append(nn.Dropout(dropout))
        
        # Main projection layer
        self.proj = self._create_projection_layer(inner_dim, dim_model, no_bias)
        
        # Initialize projection
        if zero_init_output:
            self._zero_init_projection()
    
    def _create_projection_layer(self, input_dim: int, output_dim: int, no_bias: bool) -> nn.Module:
        """Create the main projection layer - can be overridden by subclasses"""
        return nn.Linear(input_dim, output_dim, bias=not no_bias)
    
    def _zero_init_projection(self):
        """Zero initialize the projection layer"""
        nn.init.zeros_(self.proj.weight)
        if hasattr(self.proj, 'bias') and self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)
    
    def _apply_pre_layers(self, x: Tensor) -> Tensor:
        """Apply pre-processing layers"""
        for layer in self.pre_layers:
            x = layer(x)
        return x
    
    def _apply_projection(self, x: Tensor) -> Tensor:
        """Apply main projection - can be overridden by subclasses"""
        return self.proj(x)
    
    def apply(self, x: Tensor, **kwargs) -> Tensor:
        """Main forward pass"""
        x = self._apply_pre_layers(x)
        x = self._apply_projection(x)
        return self._post_process(x, **kwargs)
    
    def _post_process(self, x: Tensor, **kwargs) -> Tensor:
        """Post-process the output - can be overridden by subclasses"""
        return x
    
    @property
    def output_dim(self) -> int:
        return self.dim_model


@ffn_registry.register("post_standard")
class StandardPostProcessor(BasePostProcessor):
    """Standard post-processor for feed-forward networks"""
    pass  # Uses all base functionality


@ffn_registry.register("post_regularized", {
    "weight_decay": (float, Field(default=0.01, description="Weight decay for output layer")),
    "grad_clip_norm": (float, Field(default=0.0, description="Gradient clipping norm")),
    "spectral_norm": (bool, Field(default=False, description="Apply spectral normalization"))
})
class RegularizedPostProcessor(BasePostProcessor):
    """Post-processor with regularization techniques for training stability"""
    
    def __init__(self, dim_model: int, inner_dim: int, weight_decay: float = 0.01,
                 grad_clip_norm: float = 0.0, spectral_norm: bool = False, **kwargs):
        self.weight_decay = weight_decay
        self.grad_clip_norm = grad_clip_norm
        self.use_spectral_norm = spectral_norm
        
        super().__init__(dim_model, inner_dim, **kwargs)
        
        # Apply spectral normalization if requested
        if spectral_norm:
            self.proj = nn.utils.spectral_norm(self.proj)
    
    def _apply_projection(self, x: Tensor) -> Tensor:
        """Apply projection with gradient clipping"""
        # Apply projection
        output = self.proj(x)
        
        # Apply gradient clipping during training
        if self.training and self.grad_clip_norm > 0:
            # Register hook for gradient clipping
            if output.requires_grad:
                def clip_grad_hook(grad):
                    if grad is not None:
                        return torch.clamp(grad, -self.grad_clip_norm, self.grad_clip_norm)
                    return grad
                output.register_hook(clip_grad_hook)
        
        return output
    
    def get_weight_decay_params(self):
        """Return parameters that should have weight decay applied"""
        if self.weight_decay > 0:
            return [self.proj.weight]
        return []


@ffn_registry.register("post_scaled", {
    "scale_factor": (float, Field(default=1.0, description="Output scaling factor")),
    "learnable_scale": (bool, Field(default=False, description="Make scale learnable parameter")),
    "residual_scale": (bool, Field(default=False, description="Scale for residual connection compatibility")),
    "layer_scale_init": (float, Field(default=1e-4, description="Initial value for learnable layer scale"))
})
class ScaledPostProcessor(BasePostProcessor):
    """Post-processor with output scaling for deep network support"""
    
    def __init__(self, dim_model: int, inner_dim: int, scale_factor: float = 1.0,
                 learnable_scale: bool = False, residual_scale: bool = False,
                 layer_scale_init: float = 1e-4, **kwargs):
        super().__init__(dim_model, inner_dim, **kwargs)
        
        self.base_scale = scale_factor
        
        # Learnable scaling parameters
        if learnable_scale:
            self.scale = nn.Parameter(torch.ones(dim_model) * layer_scale_init)
        else:
            self.scale = scale_factor
            
        # Residual connection scaling (for very deep networks)
        if residual_scale:
            self.residual_scale = nn.Parameter(torch.ones(1) * layer_scale_init)
        else:
            self.residual_scale = None
    
    def _post_process(self, x: Tensor, **kwargs) -> Tensor:
        """Apply scaling to the output"""
        # Apply main scaling
        if isinstance(self.scale, nn.Parameter):
            x = x * self.scale
        else:
            x = x * self.scale
            
        # Apply residual scaling if enabled
        if self.residual_scale is not None:
            x = x * self.residual_scale
            
        return x
    
    def set_scale(self, scale: float):
        """Dynamically adjust the scale factor"""
        if isinstance(self.scale, nn.Parameter):
            with torch.no_grad():
                self.scale.fill_(scale)
        else:
            self.scale = scale
