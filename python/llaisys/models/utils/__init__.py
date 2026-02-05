"""Utility functions for model layers"""
from ...tensor import Tensor
from ...ops import Ops
from ctypes import c_size_t
from ...libllaisys import LIB_LLAISYS


def linear_nd(output: Tensor, input: Tensor, weight: Tensor, bias: Tensor = None):
    """Apply linear operation to N-dimensional tensors by creating 2D views.
    
    The linear operation only supports 2D tensors, but this function handles
    arbitrary dimensional inputs by creating a 2D view, applying the operation,
    with the view automatically sharing the underlying data.
    
    For contiguous ND tensors with shape (..., D), this function treats them as
    (batch_size, D) where batch_size is the product of all dimensions except the last.
    
    Args:
        output: Output tensor of shape (..., out_features)
        input: Input tensor of shape (..., in_features)
        weight: Weight tensor of shape (out_features, in_features)
        bias: Optional bias tensor of shape (out_features,)
    """
    # Get shapes
    input_shape = input.shape()
    output_shape = output.shape()
    in_features = weight.shape()[1]
    out_features = weight.shape()[0]
    
    # Calculate batch size as product of all dimensions except last
    batch_size = 1
    for dim in input_shape[:-1]:
        batch_size *= dim
    
    # If input is already 2D, just call linear directly
    if input.ndim() == 2:
        Ops.linear(output, input, weight, bias)
        return
    
    # For higher dimensions, create 2D views that share underlying memory
    # Create 2D view of input data
    input_2d_shape = (c_size_t * 2)(batch_size, in_features)
    input_2d_tensor = LIB_LLAISYS.tensorView(
        input.lib_tensor(),
        input_2d_shape,
        c_size_t(2)
    )
    input_2d = Tensor(tensor=input_2d_tensor)
    
    # Create 2D view of output data  
    output_2d_shape = (c_size_t * 2)(batch_size, out_features)
    output_2d_tensor = LIB_LLAISYS.tensorView(
        output.lib_tensor(),
        output_2d_shape,
        c_size_t(2)
    )
    output_2d = Tensor(tensor=output_2d_tensor)
    
    # Call Ops.linear with 2D views
    Ops.linear(output_2d, input_2d, weight, bias)


__all__ = ["linear_nd"]
