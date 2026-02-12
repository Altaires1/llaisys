"""Utility functions for SwiGLU layers"""
from ...tensor import Tensor
from ...ops import Ops

def swiglu_nd(gate: Tensor, up: Tensor) -> Tensor:
    """Apply SwiGLU operation to N-dimensional tensors.
    
    Args:
        gate: Gate tensor of shape (..., intermediate_size)
        up: Up-projection tensor of shape (..., intermediate_size)
        
    Returns:
        Output tensor of the same shape as gate and up.
    """
    shape = gate.shape()
    assert shape == up.shape(), f"SwiGLU: gate and up must have the same shape, got {shape} and {up.shape()}"
    
    intermediate_size = shape[-1]
    total_batch = 1
    for dim in shape[:-1]:
        total_batch *= dim
        
    # Create output tensor with the same shape as input
    output = Tensor(
        shape=shape,
        dtype=gate.dtype(),
        device=gate.device_type(),
        device_id=gate.device_id()
    )
    
    # Use 2D views for Ops.swiglu which expects (seq_len, intermediate_size)
    gate_2d = gate.contiguous().view(total_batch, intermediate_size)
    up_2d = up.contiguous().view(total_batch, intermediate_size)
    output_2d = output.view(total_batch, intermediate_size)
    
    Ops.swiglu(output_2d, gate_2d, up_2d)
    
    return output
