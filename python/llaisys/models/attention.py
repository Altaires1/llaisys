"""Attention layer for Qwen2 model"""
import math
from ..tensor import Tensor
from .utils import linear_nd, rms_norm_nd, rope_nd, self_attention_nd


class RMSNorm:
    """Root Mean Square Layer Normalization"""
    
    def __init__(self, weight: Tensor, eps: float = 1e-6):
        """Initialize RMSNorm
        
        Args:
            weight: Layer norm weight of shape (hidden_size,)
            eps: Epsilon for numerical stability
        """
        self.weight = weight
        self.eps = eps
    
    def forward(self, x: Tensor) -> Tensor:
        """Apply RMSNorm
        
        Args:
            x: Input tensor
            
        Returns:
            Normalized tensor
        """
        return rms_norm_nd(x, self.weight, self.eps)


class RotaryEmbedding:
    """Rotary position embedding (RoPE)"""
    
    def __init__(self, theta: float = 10000.0):
        """Initialize rotary embedding
        
        Args:
            theta: Base for the exponential (default: 10000)
        """
        self.theta = theta
    
    def forward(self, q: Tensor, pos_ids: Tensor) -> Tensor:
        """Apply rotary embedding to queries
        
        Args:
            q: Query tensor
            pos_ids: Position IDs tensor
            
        Returns:
            Rotated query tensor
        """
        return rope_nd(q, pos_ids, self.theta)


class MultiHeadAttention:
    """Multi-head attention layer"""
    
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        q_weight: Tensor,
        k_weight: Tensor,
        v_weight: Tensor,
        out_weight: Tensor,
        q_bias: Tensor = None,
        k_bias: Tensor = None,
        v_bias: Tensor = None,
        out_bias: Tensor = None,
        rope_theta: float = 10000.0,
        eps: float = 1e-6,
    ):
        """Initialize multi-head attention
        
        Args:
            hidden_size: Hidden size of the model
            num_attention_heads: Number of query attention heads
            num_key_value_heads: Number of key-value heads
            q_weight: Weight for Q projection
            k_weight: Weight for K projection
            v_weight: Weight for V projection
            out_weight: Weight for output projection
            q_bias: Optional bias for Q projection
            k_bias: Optional bias for K projection
            v_bias: Optional bias for V projection
            out_bias: Optional bias for output projection
            rope_theta: Theta for rotary embeddings
            eps: Epsilon for layer norm
        """
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = hidden_size // num_attention_heads
        
        self.q_weight = q_weight
        self.q_bias = q_bias
        self.k_weight = k_weight
        self.k_bias = k_bias
        self.v_weight = v_weight
        self.v_bias = v_bias
        self.out_weight = out_weight
        self.out_bias = out_bias
        
        self.rope = RotaryEmbedding(rope_theta)
    
    def forward(
        self,
        x: Tensor,
        pos_ids: Tensor,
        kv_cache=None,
    ) -> Tensor:
        """Forward pass of attention
        
        Args:
            x: Input tensor of shape (batch, seq_len, hidden_size)
            pos_ids: Position IDs tensor
            kv_cache: Optional KV cache for inference
            
        Returns:
            Output tensor of shape (batch, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = x.shape()

        # Project to Q, K, V
        q = linear_nd(x, self.q_weight, self.q_bias)
        k = linear_nd(x, self.k_weight, self.k_bias)
        v = linear_nd(x, self.v_weight, self.v_bias)
        
        # Reshape to (batch, seq_len, n_heads, head_dim) before RoPE
        q = q.view(batch_size, seq_len, self.num_attention_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)

        # Apply RoPE to Q and K
        q = self.rope.forward(q, pos_ids)
        k = self.rope.forward(k, pos_ids)
        
        # Compute attention scores
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_output = self_attention_nd(q, k, v, scale)
        
        # Reshape output back to (batch, seq_len, hidden_size)
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        
        # Output projection
        output = linear_nd(attn_output, self.out_weight, self.out_bias)
        
        return output
