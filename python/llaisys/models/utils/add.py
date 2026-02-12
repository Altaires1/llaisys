"""Utility functions for add operations"""
from ...tensor import Tensor
from ...ops import Ops

def add_nd(a: Tensor, b: Tensor) -> Tensor:
    """Apply add operation to N-dimensional tensors.
    
    Args:
        a: First input tensor
        b: Second input tensor
        
    Returns:
        Output tensor containing the sum of a and b.
    """
    shape = a.shape()
    assert shape == b.shape(), f"Add: a and b must have the same shape, got {shape} and {b.shape()}"
    
    # Create output tensor with the same shape as inputs
    output = Tensor(
        shape=shape,
        dtype=a.dtype(),
        device=a.device_type(),
        device_id=a.device_id()
    )
    
    # llaisys.ops.add already supports any number of dimensions as long as shapes match and it is contiguous.
    # Ensuring inputs are contiguous.
    a_cont = a.contiguous()
    b_cont = b.contiguous()
    
    Ops.add(output, a_cont, b_cont)
    
    return output
