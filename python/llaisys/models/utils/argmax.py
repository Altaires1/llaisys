"""Utility functions for argmax operations"""
from typing import Tuple
from ...tensor import Tensor
from ...ops import Ops
from ...libllaisys import DataType

def argmax_nd(vals: Tensor) -> Tuple[Tensor, Tensor]:
    """Apply argmax operation to N-dimensional tensors along the last dimension.
    
    Args:
        vals: Input tensor of shape (..., last_dim)
        
    Returns:
        A tuple of (max_idx, max_val), both of shape (...)
    """
    input_shape = vals.shape()
    last_dim = input_shape[-1]
    output_shape = input_shape[:-1]
    
    total_batch = 1
    for dim in output_shape:
        total_batch *= dim
        
    # Create output tensors
    max_idx = Tensor(
        shape=output_shape,
        dtype=DataType.I64,
        device=vals.device_type(),
        device_id=vals.device_id()
    )
    max_val = Tensor(
        shape=output_shape,
        dtype=vals.dtype(),
        device=vals.device_type(),
        device_id=vals.device_id()
    )
    
    # 2D and 1D views for batch processing
    # Since we need to slice on dimension 0, ensuring vals is contiguous first
    # makes the resulting slices contiguous and viewable.
    vals_2d = vals.contiguous().view(total_batch, last_dim)
    max_idx_1d = max_idx.view(total_batch)
    max_val_1d = max_val.view(total_batch)
    
    for i in range(total_batch):
        # Slice on first dimension results in a contiguous tensor for contiguous inputs.
        # v: shape (1, last_dim) -> view(last_dim) to get (last_dim,)
        v = vals_2d.slice(0, i, i+1).view(last_dim)
        # mi/mv: shape (1,) which is already 1D and size 1 as required by C++ Ops.argmax
        mi = max_idx_1d.slice(0, i, i+1)
        mv = max_val_1d.slice(0, i, i+1)
        
        Ops.argmax(mi, mv, v)
        
    return max_idx, max_val
