"""Transformer block for Qwen2 model"""
from typing import Dict, Optional
from ..tensor import Tensor
from ..ops import Ops
from .attention import RMSNorm, MultiHeadAttention
from .ffn import FeedForwardNetwork


class TransformerBlock:
    """Transformer decoder block with attention and FFN"""
    
    def __init__(
        self,
        layer_idx: int,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        intermediate_size: int,
        rope_theta: float,
        rms_norm_eps: float,
        weights: Dict[str, Tensor],
    ):
        """Initialize transformer block
        
        Args:
            layer_idx: Layer index in the model
            hidden_size: Hidden size
            num_attention_heads: Number of attention heads
            num_key_value_heads: Number of key-value heads
            intermediate_size: Intermediate size in FFN
            rope_theta: Theta for rotary embeddings
            rms_norm_eps: Epsilon for layer norm
            weights: Dictionary of all model weights
        """
        self.layer_idx = layer_idx
        
        # Layer norm for attention
        self.attn_norm = RMSNorm(
            weights[f"model.layers.{layer_idx}.input_layernorm.weight"],
            rms_norm_eps
        )
        
        # Multi-head attention with separate Q, K, V projections
        self.attention = MultiHeadAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            q_weight=weights[f"model.layers.{layer_idx}.self_attn.q_proj.weight"],
            q_bias=weights.get(f"model.layers.{layer_idx}.self_attn.q_proj.bias"),
            k_weight=weights[f"model.layers.{layer_idx}.self_attn.k_proj.weight"],
            k_bias=weights.get(f"model.layers.{layer_idx}.self_attn.k_proj.bias"),
            v_weight=weights[f"model.layers.{layer_idx}.self_attn.v_proj.weight"],
            v_bias=weights.get(f"model.layers.{layer_idx}.self_attn.v_proj.bias"),
            out_weight=weights[f"model.layers.{layer_idx}.self_attn.o_proj.weight"],
            out_bias=weights.get(f"model.layers.{layer_idx}.self_attn.o_proj.bias"),
            rope_theta=rope_theta,
            eps=rms_norm_eps,
        )
        
        # Layer norm for FFN
        self.ffn_norm = RMSNorm(
            weights[f"model.layers.{layer_idx}.post_attention_layernorm.weight"],
            rms_norm_eps
        )
        
        # Feed-forward network
        self.ffn = FeedForwardNetwork(
            gate_weight=weights[f"model.layers.{layer_idx}.mlp.gate_proj.weight"],
            gate_bias=weights.get(f"model.layers.{layer_idx}.mlp.gate_proj.bias"),
            up_weight=weights[f"model.layers.{layer_idx}.mlp.up_proj.weight"],
            up_bias=weights.get(f"model.layers.{layer_idx}.mlp.up_proj.bias"),
            down_weight=weights[f"model.layers.{layer_idx}.mlp.down_proj.weight"],
            down_bias=weights.get(f"model.layers.{layer_idx}.mlp.down_proj.bias"),
        )
    
    def forward(
        self,
        x: Tensor,
        pos_ids: Tensor,
        kv_cache: Optional[object] = None,
    ) -> Tensor:
        """Forward pass of transformer block
        
        Args:
            x: Input tensor of shape (batch, seq_len, hidden_size)
            pos_ids: Position IDs tensor
            kv_cache: Optional KV cache
            
        Returns:
            Output tensor of shape (batch, seq_len, hidden_size)
        """
        # Self-attention with residual connection
        attn_input = self.attn_norm.forward(x)
        attn_output = self.attention.forward(attn_input, pos_ids, kv_cache)
        x = self._add_residual(x, attn_output)
        
        # Feed-forward with residual connection
        ffn_input = self.ffn_norm.forward(x)
        ffn_output = self.ffn.forward(ffn_input)
        x = self._add_residual(x, ffn_output)
        
        return x
    
    def _add_residual(self, x: Tensor, y: Tensor) -> Tensor:
        """Add residual connection (x + y)
        
        Args:
            x: First tensor
            y: Second tensor
            
        Returns:
            Sum tensor
        """
        output = Tensor(
            shape=x.shape(),
            dtype=x.dtype(),
            device=x.device_type(),
            device_id=x.device_id()
        )
        Ops.add(output, x, y)
        return output
