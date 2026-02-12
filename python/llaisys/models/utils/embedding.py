"""Utility functions for embedding layers"""
from ...tensor import Tensor
from ...ops import Ops

def embedding_nd(input_ids: Tensor, weight: Tensor) -> Tensor:
    """Apply embedding operation to N-dimensional input_ids.
    
    Args:
        input_ids: Input token IDs tensor of any shape (..., num_indices)
        weight: Embedding weight tensor of shape (vocab_size, embedding_dim)
        
    Returns:
        Embedded tokens tensor with embedding_dim appended to input_ids shape.
    """
    input_shape = input_ids.shape()
    embedding_dim = weight.shape()[1]

    total_elements = 1
    for dim in input_shape:
        total_elements *= dim

    # Create output tensor with the final ND shape directly.
    # It will be contiguous by default.
    output_shape = input_shape + (embedding_dim,)
    output = Tensor(
        shape=output_shape,
        dtype=weight.dtype(),
        device=weight.device_type(),
        device_id=weight.device_id()
    )

    # Use view to satisfy Ops.embedding's 1D/2D requirements.
    # Since newly created tensors are contiguous and view() on contiguous tensors 
    # returns contiguous views, we can directly call Ops.embedding.
    input_ids_flat = input_ids.contiguous().view(total_elements)
    output_2d = output.view(total_elements, embedding_dim)

    Ops.embedding(output_2d, input_ids_flat, weight)

    return output
