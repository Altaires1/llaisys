from typing import Dict

from ...tensor import Tensor
from ...ops import Ops
from ...libllaisys import DeviceType, DataType
from ...kv_cache import KVCache

# Create ops instance
ops = Ops()

from .attention import SelfAttentionLayer
from .mlp import MLP


class TransformerLayer:
    """
    Single transformer layer implementation for DeepSeek-R1-Distill-Qwen-1.5B.
    """
    
    def __init__(self, config: Dict, layer_weights: Dict, device: DeviceType):
        """
        Initialize a transformer layer.
        
        Args:
            config: Model configuration dictionary
            layer_weights: Layer-specific weights dictionary
            device: Device to run the layer on
        """
        self.hidden_size = config["hidden_size"]
        self.rms_norm_eps = config["rms_norm_eps"]
        self.device = device
        
        # Load layer normalization weights
        self.input_norm = layer_weights["input_norm"]
        self.post_attn_norm = layer_weights["post_attn_norm"]
        
        # Initialize sub-layers
        self.attention = SelfAttentionLayer(config, layer_weights, device)
        self.mlp = MLP(config, layer_weights, device)
    
    def forward(self, hidden_states: Tensor, kv_cache: KVCache, layer_idx: int, seq_len: int):
        """
        Forward pass through the transformer layer.
        
        Args:
            hidden_states: Input hidden states (batch_size, seq_len, hidden_size)
            kv_cache: KV cache for storing past key-value pairs
            layer_idx: Index of the current layer
            seq_len: Current sequence length
            
        Returns:
            Layer output (batch_size, seq_len, hidden_size)
        """
        # Input normalization
        norm_input = Tensor(
            shape=hidden_states.shape(),
            dtype=DataType.BFLOAT16,
            device=self.device,
            device_id=0
        )
        ops.rms_norm(norm_input, hidden_states, self.input_norm, self.rms_norm_eps)
        
        # Self attention
        attn_output = self.attention.forward(norm_input, kv_cache, layer_idx, seq_len)
        
        # Residual connection
        hidden_states_add = Tensor(
            shape=hidden_states.shape(),
            dtype=DataType.BFLOAT16,
            device=self.device,
            device_id=0
        )
        ops.add(hidden_states_add, hidden_states, attn_output)
        
        # Post attention normalization
        norm_post_attn = Tensor(
            shape=hidden_states_add.shape(),
            dtype=DataType.BFLOAT16,
            device=self.device,
            device_id=0
        )
        ops.rms_norm(norm_post_attn, hidden_states_add, self.post_attn_norm, self.rms_norm_eps)
        
        # MLP
        mlp_output = self.mlp.forward(norm_post_attn)
        
        # Final residual connection
        output = Tensor(
            shape=hidden_states_add.shape(),
            dtype=DataType.BFLOAT16,
            device=self.device,
            device_id=0
        )
        ops.add(output, hidden_states_add, mlp_output)
        
        return output