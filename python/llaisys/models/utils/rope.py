"""Utility functions for RoPE layers"""
from ...tensor import Tensor
from ...ops import Ops

def rope_nd(inp: Tensor, pos_ids: Tensor, theta: float) -> Tensor:
    """Apply RoPE operation to N-dimensional tensors.
    
    Args:
        inp: Input tensor of shape (..., n_head, head_dim)
        pos_ids: Position IDs tensor of shape (...,) matching the prefix of inp
        theta: RoPE theta parameter
        
    Returns:
        Rotated tensor of the same shape as inp.
    """
    input_shape = inp.shape()
    pos_ids_shape = pos_ids.shape()
    
    # RoPE in llaisys expects 3D input: (seq_len, n_head, head_dim)
    # and 1D pos_ids: (seq_len,)
    # We combine all leading dimensions of inp that match pos_ids into a single seq_len dimension.
    
    n_head = input_shape[-2]
    head_dim = input_shape[-1]
    
    # Prefix dimensions that will be flattened
    prefix_shape = input_shape[:-2]
    assert prefix_shape == pos_ids_shape, f"RoPE: pos_ids shape {pos_ids_shape} must match input prefix {prefix_shape}"
    
    total_seq = 1
    for dim in prefix_shape:
        total_seq *= dim
        
    # Create output tensor with the same shape as input
    output = Tensor(
        shape=input_shape,
        dtype=inp.dtype(),
        device=inp.device_type(),
        device_id=inp.device_id()
    )
    
    # Use views to satisfy Ops.rope requirements
    # We ensure input and pos_ids are contiguous
    inp_cont = inp.contiguous()
    pos_ids_cont = pos_ids.contiguous()
    
    inp_3d = inp_cont.view(total_seq, n_head, head_dim)
    output_3d = output.view(total_seq, n_head, head_dim)
    pos_ids_1d = pos_ids_cont.view(total_seq)
    
    Ops.rope(output_3d, inp_3d, pos_ids_1d, theta)
    
    return output
