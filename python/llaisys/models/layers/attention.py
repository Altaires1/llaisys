from typing import Dict
import ctypes
import math

from ...tensor import Tensor
from ...ops import Ops
from ...kv_cache import KVCache
from ...libllaisys import DeviceType, DataType

# Create ops instance
ops = Ops()


class SelfAttentionLayer:
    """
    Self-attention layer with RoPE and KV cache support.
    """
    
    def __init__(self, config: Dict, layer_weights: Dict, device: DeviceType):
        """
        Initialize the self-attention layer.
        
        Args:
            config: Model configuration dictionary
            layer_weights: Layer-specific weights dictionary
            device: Device to run the layer on
        """
        self.hidden_size = config["hidden_size"]
        self.num_heads = config["num_attention_heads"]
        self.num_kv_heads = config["num_key_value_heads"]
        self.head_dim = self.hidden_size // self.num_heads
        self.kv_head_dim = self.hidden_size // self.num_kv_heads
        self.rope_theta = config["rope_theta"]
        self.device = device
        
        # Load weights
        self.q_proj = layer_weights["attn_q_proj"]
        self.k_proj = layer_weights["attn_k_proj"]
        self.v_proj = layer_weights["attn_v_proj"]
        self.o_proj = layer_weights["attn_o_proj"]
        self.q_bias = layer_weights["attn_q_bias"]
        self.k_bias = layer_weights["attn_k_bias"]
        self.v_bias = layer_weights["attn_v_bias"]
    
    def forward(self, hidden_states: Tensor, kv_cache: KVCache, layer_idx: int, seq_len: int):
        """
        Forward pass through the self-attention layer.
        
        Args:
            hidden_states: Input hidden states (batch_size, seq_len, hidden_size)
            kv_cache: KV cache for storing past key-value pairs
            layer_idx: Index of the current layer
            seq_len: Current sequence length
            
        Returns:
            Attention output (batch_size, seq_len, hidden_size)
        """
        # Query projection
        q = Tensor(
            shape=[1, seq_len, self.hidden_size],
            dtype=DataType.BFLOAT16,
            device=self.device,
            device_id=0
        )
        ops.linear(q, hidden_states, self.q_proj, self.q_bias)
        
        # Key projection
        k = Tensor(
            shape=[1, seq_len, self.hidden_size // (self.num_heads // self.num_kv_heads)],
            dtype=DataType.BFLOAT16,
            device=self.device,
            device_id=0
        )
        ops.linear(k, hidden_states, self.k_proj, self.k_bias)
        
        # Value projection
        v = Tensor(
            shape=[1, seq_len, self.hidden_size // (self.num_heads // self.num_kv_heads)],
            dtype=DataType.BFLOAT16,
            device=self.device,
            device_id=0
        )
        ops.linear(v, hidden_states, self.v_proj, self.v_bias)
        
        # Apply RoPE
        q_rope = Tensor(
            shape=q.shape(),
            dtype=DataType.BFLOAT16,
            device=self.device,
            device_id=0
        )
        k_rope = Tensor(
            shape=k.shape(),
            dtype=DataType.BFLOAT16,
            device=self.device,
            device_id=0
        )
        
        # Create position ids
        pos_tensor = Tensor(
            shape=[1, seq_len],
            dtype=DataType.INT64,
            device=self.device,
            device_id=0
        )
        
        # Create position ids directly with ctypes
        pos_ids_array = (ctypes.c_int64 * seq_len)(*range(seq_len))
        pos_ids_ptr = ctypes.cast(pos_ids_array, ctypes.c_void_p)
        pos_tensor.load(pos_ids_ptr)
        
        ops.rope(q_rope, q, pos_tensor, self.rope_theta)
        ops.rope(k_rope, k, pos_tensor, self.rope_theta)
        
        # Update KV cache
        kv_cache.update(k_rope, v, layer_idx, seq_len)
        
        # Get cached KV
        cached_k, cached_v = kv_cache.get(layer_idx)
        
        # Self attention
        attn_output = Tensor(
            shape=[1, seq_len, self.hidden_size],
            dtype=DataType.BFLOAT16,
            device=self.device,
            device_id=0
        )
        
        scale = 1.0 / math.sqrt(self.head_dim)
        ops.self_attention(attn_output, q_rope, cached_k, cached_v, scale)
        
        # Output projection
        output = Tensor(
            shape=[1, seq_len, self.hidden_size],
            dtype=DataType.BFLOAT16,
            device=self.device,
            device_id=0
        )
        ops.linear(output, attn_output, self.o_proj, None)
        
        return output