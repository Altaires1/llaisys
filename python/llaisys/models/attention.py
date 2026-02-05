"""Attention layer for Qwen2 model"""
import math
from ..tensor import Tensor
from ..ops import Ops
from ..libllaisys import DeviceType, DataType
from .utils import linear_nd


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
        output = Tensor(
            shape=x.shape(),
            dtype=x.dtype(),
            device=x.device_type(),
            device_id=x.device_id()
        )
        
        Ops.rms_norm(output, x, self.weight, self.eps)
        
        return output


class RotaryEmbedding:
    """Rotary position embedding (RoPE)"""
    
    def __init__(self, dim: int, theta: float = 10000.0):
        """Initialize rotary embedding
        
        Args:
            dim: Dimension of the embedding
            theta: Base for the exponential (default: 10000)
        """
        self.dim = dim
        self.theta = theta
    
    def forward(self, q: Tensor, pos_ids: Tensor) -> Tensor:
        """Apply rotary embedding to queries
        
        Args:
            q: Query tensor
            pos_ids: Position IDs tensor
            
        Returns:
            Rotated query tensor
        """
        output = Tensor(
            shape=q.shape(),
            dtype=q.dtype(),
            device=q.device_type(),
            device_id=q.device_id()
        )
        
        Ops.rope(output, q, pos_ids, self.theta)
        
        return output


class MultiHeadAttention:
    """Multi-head attention layer"""
    
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        q_weight: Tensor,
        q_bias: Tensor,
        k_weight: Tensor,
        k_bias: Tensor,
        v_weight: Tensor,
        v_bias: Tensor,
        out_weight: Tensor,
        out_bias: Tensor,
        rope_theta: float = 10000.0,
        eps: float = 1e-6,
    ):
        """Initialize multi-head attention
        
        Args:
            hidden_size: Hidden size of the model
            num_attention_heads: Number of query attention heads
            num_key_value_heads: Number of key-value heads
            q_weight: Weight for Q projection
            q_bias: Bias for Q projection
            k_weight: Weight for K projection
            k_bias: Bias for K projection
            v_weight: Weight for V projection
            v_bias: Bias for V projection
            out_weight: Weight for output projection
            out_bias: Bias for output projection
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
        
        self.rope = RotaryEmbedding(self.head_dim, rope_theta)
    
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
        
        # Project to Q, K, V (separate projections)
        q = Tensor(
            shape=(batch_size, seq_len, self.num_attention_heads * self.head_dim),
            dtype=x.dtype(),
            device=x.device_type(),
            device_id=x.device_id()
        )
        linear_nd(q, x, self.q_weight, self.q_bias)
        
        k = Tensor(
            shape=(batch_size, seq_len, self.num_key_value_heads * self.head_dim),
            dtype=x.dtype(),
            device=x.device_type(),
            device_id=x.device_id()
        )
        linear_nd(k, x, self.k_weight, self.k_bias)
        
        v = Tensor(
            shape=(batch_size, seq_len, self.num_key_value_heads * self.head_dim),
            dtype=x.dtype(),
            device=x.device_type(),
            device_id=x.device_id()
        )
        linear_nd(v, x, self.v_weight, self.v_bias)
        
        # Apply RoPE to Q and K
        q = self.rope.forward(q, pos_ids)
        k = self.rope.forward(k, pos_ids)
        
        # Reshape for attention computation
        # (batch, seq_len, num_heads, head_dim) -> (batch, num_heads, seq_len, head_dim)
        q_shape = (batch_size, seq_len, self.num_attention_heads, self.head_dim)
        q = q.view(*q_shape).permute(0, 2, 1, 3)
        
        k_shape = (batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        k = k.view(*k_shape).permute(0, 2, 1, 3)
        
        v_shape = (batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        v = v.view(*v_shape).permute(0, 2, 1, 3)
        
        # Compute attention scores
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_output = Tensor(
            shape=(batch_size, self.num_attention_heads, seq_len, self.head_dim),
            dtype=x.dtype(),
            device=x.device_type(),
            device_id=x.device_id()
        )
        
        Ops.self_attention(attn_output, q, k, v, scale)
        
        # Reshape output back to (batch, seq_len, hidden_size)
        attn_output = attn_output.permute(0, 2, 1, 3)  # (batch, seq_len, num_heads, head_dim)
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        
        # Output projection
        output = Tensor(
            shape=(batch_size, seq_len, self.hidden_size),
            dtype=x.dtype(),
            device=x.device_type(),
            device_id=x.device_id()
        )
        linear_nd(output, attn_output, self.out_weight, self.out_bias)
        
        return output
