from typing import Dict

from ...tensor import Tensor
from ...ops import Ops
from ...libllaisys import DeviceType, DataType

# Create ops instance
ops = Ops()


class EmbeddingLayer:
    """
    Embedding layer implementation for Qwen2 model.
    """
    
    def __init__(self, config: Dict, weights: Dict, device: DeviceType):
        """
        Initialize the embedding layer.
        
        Args:
            config: Model configuration dictionary
            weights: Dictionary containing embedding weights
            device: Device to run the layer on
        """
        self.vocab_size = config["vocab_size"]
        self.hidden_size = config["hidden_size"]
        self.device = device
        
        # Load weights
        self.weight = weights["embed_tokens"]
    
    def forward(self, input_ids: Tensor) -> Tensor:
        """
        Forward pass through the embedding layer.
        
        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            
        Returns:
            Embedded hidden states (batch_size, seq_len, hidden_size)
        """
        # Get input shape
        batch_size, seq_len = input_ids.shape()
        
        # Create output tensor
        hidden_states = Tensor(
            shape=[batch_size, seq_len, self.hidden_size],
            dtype=DataType.BFLOAT16,
            device=self.device,
            device_id=0
        )
        
        # Embedding lookup
        ops.embedding_lookup(hidden_states, input_ids, self.weight)
        
        return hidden_states