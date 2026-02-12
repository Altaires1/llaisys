import sys
import os
import numpy as np
from ctypes import c_void_p

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import llaisys
from llaisys.tensor import Tensor
from llaisys.libllaisys import DataType, DeviceType, MemcpyKind
from llaisys.models.utils import linear_nd, embedding_nd

def to_numpy(tensor: Tensor) -> np.ndarray:
    shape = tensor.shape()
    if tensor.dtype() == DataType.F32:
        arr = np.zeros(shape, dtype=np.float32)
    elif tensor.dtype() == DataType.I64:
        arr = np.zeros(shape, dtype=np.int64)
    else:
        raise ValueError(f"Unsupported dtype for test: {tensor.dtype()}")
    
    # Copy from device to host
    # Assuming CPU for simplicity in this test, but using memcpy for generality
    arr_ptr = arr.ctypes.data_as(c_void_p)
    llaisys.RuntimeAPI(tensor.device_type()).memcpy_sync(
        arr_ptr,
        tensor.data_ptr(),
        arr.nbytes,
        MemcpyKind.D2H
    )
    return arr

def from_numpy(arr: np.ndarray, device=DeviceType.CPU) -> Tensor:
    dtype = DataType.F32 if arr.dtype == np.float32 else DataType.I64
    tensor = Tensor(shape=arr.shape, dtype=dtype, device=device)
    arr_ptr = arr.ctypes.data_as(c_void_p)
    llaisys.RuntimeAPI(device).memcpy_sync(
        tensor.data_ptr(),
        arr_ptr,
        arr.nbytes,
        MemcpyKind.H2D
    )
    return tensor

def test_linear_nd():
    print("Testing linear_nd...")
    # Shape: (2, 3, 4) @ (5, 4)^T -> (2, 3, 5)
    input_shape = (2, 3, 4)
    weight_shape = (5, 4)
    bias_shape = (5,)
    
    np_input = np.random.randn(*input_shape).astype(np.float32)
    np_weight = np.random.randn(*weight_shape).astype(np.float32)
    np_bias = np.random.randn(*bias_shape).astype(np.float32)
    
    input_t = from_numpy(np_input)
    weight_t = from_numpy(np_weight)
    bias_t = from_numpy(np_bias)
    
    output_t = linear_nd(input_t, weight_t, bias_t)
    
    # Expected output using numpy
    # np.matmul(np_input, np_weight.T) + np_bias
    expected = np.matmul(np_input, np_weight.T) + np_bias
    actual = to_numpy(output_t)
    
    assert np.allclose(actual, expected, atol=1e-5), f"linear_nd failed: max diff {np.abs(actual - expected).max()}"
    print("linear_nd test passed!")

def test_embedding_nd():
    print("Testing embedding_nd...")
    # Shape: (2, 3) with vocab 10, dim 8 -> (2, 3, 8)
    input_shape = (2, 3)
    vocab_size = 10
    embedding_dim = 8
    
    np_input = np.random.randint(0, vocab_size, size=input_shape).astype(np.int64)
    np_weight = np.random.randn(vocab_size, embedding_dim).astype(np.float32)
    
    input_t = from_numpy(np_input)
    weight_t = from_numpy(np_weight)
    
    output_t = embedding_nd(input_t, weight_t)
    
    # Expected output using numpy
    expected = np_weight[np_input]
    actual = to_numpy(output_t)
    
    assert np.allclose(actual, expected, atol=1e-5), "embedding_nd test failed!"
    print("embedding_nd test passed!")

if __name__ == "__main__":
    try:
        test_linear_nd()
        test_embedding_nd()
        print("\033[92mAll ND utility tests passed!\033[0m")
    except Exception as e:
        print(f"\033[91mTest failed: {e}\033[0m")
        import traceback
        traceback.print_exc()
        sys.exit(1)
