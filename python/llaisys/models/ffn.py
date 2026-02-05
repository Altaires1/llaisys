"""Feed-forward network layer for Qwen2 model"""
from ..tensor import Tensor
from ..ops import Ops
from .utils import linear_nd


class FeedForwardNetwork:
    """Feed-forward network (MLP) layer"""
    
    def __init__(
        self,
        gate_weight: Tensor,
        gate_bias: Tensor,
        up_weight: Tensor,
        up_bias: Tensor,
        down_weight: Tensor,
        down_bias: Tensor,
    ):
        """Initialize FFN layer
        
        Args:
            gate_weight: Weight for gate projection
            gate_bias: Bias for gate projection
            up_weight: Weight for up projection
            up_bias: Bias for up projection
            down_weight: Weight for down projection (output)
            down_bias: Bias for down projection (output)
        """
        self.gate_weight = gate_weight
        self.gate_bias = gate_bias
        self.up_weight = up_weight
        self.up_bias = up_bias
        self.down_weight = down_weight
        self.down_bias = down_bias
        
        # Verify dimensions
        assert gate_weight.shape()[0] == up_weight.shape()[0], \
            "Gate and up projections must have same input dimension"
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of FFN
        
        Args:
            x: Input tensor of shape (batch, seq_len, hidden_size)
            
        Returns:
            Output tensor of shape (batch, seq_len, hidden_size)
        """
        batch_size, seq_len, hidden_size = x.shape()
        
        # Get intermediate size from weights
        intermediate_size = self.gate_weight.shape()[1]
        
        # Gate projection: (batch, seq_len, intermediate_size)
        gate = Tensor(
            shape=(batch_size, seq_len, intermediate_size),
            dtype=x.dtype(),
            device=x.device_type(),
            device_id=x.device_id()
        )
        linear_nd(gate, x, self.gate_weight, self.gate_bias)
        
        # Up projection: (batch, seq_len, intermediate_size)
        up = Tensor(
            shape=(batch_size, seq_len, intermediate_size),
            dtype=x.dtype(),
            device=x.device_type(),
            device_id=x.device_id()
        )
        linear_nd(up, x, self.up_weight, self.up_bias)
        
        # SwiGLU activation: gate * up
        activated = Tensor(
            shape=(batch_size, seq_len, intermediate_size),
            dtype=x.dtype(),
            device=x.device_type(),
            device_id=x.device_id()
        )
        Ops.swiglu(activated, gate, up)
        
        # Down projection (output): (batch, seq_len, hidden_size)
        output = Tensor(
            shape=(batch_size, seq_len, hidden_size),
            dtype=x.dtype(),
            device=x.device_type(),
            device_id=x.device_id()
        )
        linear_nd(output, activated, self.down_weight, self.down_bias)
        
        return output
