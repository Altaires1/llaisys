from typing import Dict

from ...tensor import Tensor
from ...ops import Ops
from ...libllaisys import DeviceType, DataType

# Create ops instance
ops = Ops()


class LayerNorm:
    """
    Layer normalization implementation for Qwen2 model.
    """
    
    def __init__(self, config: Dict, weights: Dict, device: DeviceType):
        """
        Initialize the layer normalization layer.
        
        Args:
            config: Model configuration dictionary
            weights: Dictionary containing layer norm weights
            device: Device to run the layer on
        """
        self.hidden_size = config["hidden_size"]
        self.rms_norm_eps = config["rms_norm_eps"]
        self.device = device
        
        # Load weights
        # Try both possible weight keys
        if "final_norm.weight" in weights:
            self.weight = weights["final_norm.weight"]
        elif "final_norm" in weights:
            self.weight = weights["final_norm"]
        else:
            raise KeyError("Could not find final_norm weights")
    
    def forward(self, hidden_states: Tensor) -> Tensor:
        """
        Forward pass through the layer normalization.
        
        Args:
            hidden_states: Input hidden states (batch_size, seq_len, hidden_size)
            
        Returns:
            Normalized hidden states (batch_size, seq_len, hidden_size)
        """
        # Create output tensor
        output = Tensor(
            shape=hidden_states.shape(),
            dtype=DataType.BFLOAT16,
            device=self.device,
            device_id=0
        )
        
        # Apply RMS normalization
        ops.rms_norm(output, hidden_states, self.weight, self.rms_norm_eps)
        
        return output