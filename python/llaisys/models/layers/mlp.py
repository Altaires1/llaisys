from typing import Dict

from ...tensor import Tensor
from ...ops import Ops
from ...libllaisys import DeviceType, DataType

# Create ops instance
ops = Ops()


class MLP:
    """
    MLP layer with SwiGLU activation for DeepSeek-R1-Distill-Qwen-1.5B.
    """
    
    def __init__(self, config: Dict, layer_weights: Dict, device: DeviceType):
        """
        Initialize the MLP layer.
        
        Args:
            config: Model configuration dictionary
            layer_weights: Layer-specific weights dictionary
            device: Device to run the layer on
        """
        self.hidden_size = config["hidden_size"]
        self.intermediate_size = config["intermediate_size"]
        self.device = device
        
        # Load weights
        self.gate_proj = layer_weights["mlp_gate_proj"]
        self.up_proj = layer_weights["mlp_up_proj"]
        self.down_proj = layer_weights["mlp_down_proj"]
    
    def forward(self, hidden_states: Tensor):
        """
        Forward pass through the MLP layer.
        
        Args:
            hidden_states: Input hidden states (batch_size, seq_len, hidden_size)
            
        Returns:
            MLP output (batch_size, seq_len, hidden_size)
        """
        # Gate projection
        gate = Tensor(
            shape=[1, hidden_states.shape()[1], self.intermediate_size],
            dtype=DataType.BFLOAT16,
            device=self.device,
            device_id=0
        )
        ops.linear(gate, hidden_states, self.gate_proj, None)
        
        # Up projection
        up = Tensor(
            shape=[1, hidden_states.shape()[1], self.intermediate_size],
            dtype=DataType.BFLOAT16,
            device=self.device,
            device_id=0
        )
        ops.linear(up, hidden_states, self.up_proj, None)
        
        # SwiGLU activation
        swiglu = Tensor(
            shape=gate.shape(),
            dtype=DataType.BFLOAT16,
            device=self.device,
            device_id=0
        )
        ops.swiglu(swiglu, gate, up)
        
        # Down projection
        output = Tensor(
            shape=[1, hidden_states.shape()[1], self.hidden_size],
            dtype=DataType.BFLOAT16,
            device=self.device,
            device_id=0
        )
        ops.linear(output, swiglu, self.down_proj, None)
        
        return output