"""Utility functions for linear layers"""
from ...tensor import Tensor
from ...ops import Ops

def linear_nd(input: Tensor, weight: Tensor, bias: Tensor = None) -> Tensor:
    """Apply linear operation to N-dimensional tensors.
    
    Args:
        input: Input tensor of shape (..., in_features)
        weight: Weight tensor of shape (out_features, in_features)
        bias: Optional bias tensor of shape (out_features,)
    """ 
    input_shape = input.shape()
    out_features, in_features = weight.shape()
    
    total_elements = 1
    for dim in input_shape[:-1]:
        total_elements *= dim

    # Create output tensor with final ND shape
    output_shape = input_shape[:-1] + (out_features,)
    output = Tensor(
        shape=output_shape,
        dtype=input.dtype(),
        device=input.device_type(),
        device_id=input.device_id()
    )
    
    # View both input and output as 2D for Ops.linear
    input_2d = input.contiguous().view(total_elements, in_features)
    output_2d = output.view(total_elements, out_features)
    
    bias_cont = bias.contiguous() if bias is not None else None
    
    Ops.linear(output_2d, input_2d, weight, bias_cont)

    return output
