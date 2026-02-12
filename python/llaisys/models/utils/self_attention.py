"""Utility functions for self-attention operations"""
from ...tensor import Tensor
from ...ops import Ops

def self_attention_nd(q: Tensor, k: Tensor, v: Tensor, scale: float) -> Tensor:
    """Apply self-attention operation to N-dimensional tensors.
    
    Args:
        q: Query tensor of shape (..., q_len, n_head, head_dim)
        k: Key tensor of shape (..., kv_len, n_kv_head, head_dim)
        v: Value tensor of shape (..., kv_len, n_kv_head, value_dim)
        scale: Attention scaling factor
        
    Returns:
        Attention output tensor of shape (..., q_len, n_head, value_dim)
    """
    q_shape = q.shape()
    k_shape = k.shape()
    v_shape = v.shape()
    
    q_len, n_head, head_dim = q_shape[-3:]
    kv_len, n_kv_head, _ = k_shape[-3:]
    value_dim = v_shape[-1]
    
    # Calculate batch size from leading dimensions
    batch_dims = q_shape[:-3]
    total_batch = 1
    for dim in batch_dims:
        total_batch *= dim
        
    # Expected output shape: (..., q_len, n_head, value_dim)
    output_shape = batch_dims + (q_len, n_head, value_dim)
    output = Tensor(
        shape=output_shape,
        dtype=q.dtype(),
        device=q.device_type(),
        device_id=q.device_id()
    )
    
    # Underlying C++ op expects 3D: (len, heads, dim)
    # We iterate over batch dimensions.
    q_cont = q.contiguous()
    k_cont = k.contiguous()
    v_cont = v.contiguous()
    
    # View as 4D to simplify batch iteration
    q_4d = q_cont.view(total_batch, q_len, n_head, head_dim)
    k_4d = k_cont.view(total_batch, kv_len, n_kv_head, head_dim)
    v_4d = v_cont.view(total_batch, kv_len, n_kv_head, value_dim)
    output_4d = output.view(total_batch, q_len, n_head, value_dim)
    
    for i in range(total_batch):
        # Slice one batch item and reshape to 3D
        q_i = q_4d.slice(0, i, i+1).view(q_len, n_head, head_dim)
        k_i = k_4d.slice(0, i, i+1).view(kv_len, n_kv_head, head_dim)
        v_i = v_4d.slice(0, i, i+1).view(kv_len, n_kv_head, value_dim)
        out_i = output_4d.slice(0, i, i+1).view(q_len, n_head, value_dim)
        
        Ops.self_attention(out_i, q_i, k_i, v_i, scale)
        
    return output
