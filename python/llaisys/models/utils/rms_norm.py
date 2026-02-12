"""Utility functions for RMS Norm layers"""
from ...tensor import Tensor
from ...ops import Ops

def rms_norm_nd(input: Tensor, weight: Tensor, eps: float) -> Tensor:
    """Apply RMS Norm operation to N-dimensional tensors.
    
    Args:
        input: Input tensor of shape (..., hidden_size)
        weight: Weight tensor of shape (hidden_size,)
        eps: Small value for numerical stability
        
    Returns:
        Normalized tensor of the same shape as input.
    """
    input_shape = input.shape()
    hidden_size = input_shape[-1]
    
    total_batch = 1
    for dim in input_shape[:-1]:
        total_batch *= dim
        
    # Create output tensor with the same shape as input
    output = Tensor(
        shape=input_shape,
        dtype=input.dtype(),
        device=input.device_type(),
        device_id=input.device_id()
    )
    
    # Use 2D views for Ops.rms_norm which expects (batch, hidden)
    # Ensuring input is contiguous first
    input_2d = input.contiguous().view(total_batch, hidden_size)
    output_2d = output.view(total_batch, hidden_size)
    
    # Weight must be contiguous as well
    weight_cont = weight.contiguous()
    
    Ops.rms_norm(output_2d, input_2d, weight_cont, eps)
    
    return output
