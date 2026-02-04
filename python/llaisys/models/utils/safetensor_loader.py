import os
import ctypes
import struct
from typing import Dict, BinaryIO
from safetensors import safe_open
from ...tensor import Tensor, DataType
from ...libllaisys import DeviceType


def load_safetensors(filename: str, device: str = "cpu", device_id: int = 0) -> Dict[str, Tensor]:
    """
    Load safetensors file into llaisys Tensor objects using pure Python implementation.
    
    Args:
        filename: Path to the safetensors file
        device: Device to load tensors onto, either "cpu" or "cuda"
        device_id: Device ID to load tensors onto
        
    Returns:
        Dictionary mapping tensor names to llaisys Tensor objects
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Safetensors file not found: {filename}")
    
    tensors = {}
    
    with safe_open(filename, framework="np") as f:
        for name in f.keys():
            # Get tensor data first to determine shape and dtype
            data = f.get_raw(name)
            
            # Get tensor metadata using safetensors' metadata API
            metadata = f.metadata()
            tensor_metadata = metadata[name]
            shape = tensor_metadata["shape"]
            dtype = tensor_metadata["dtype"]
            
            # Convert safetensors dtype to llaisys DataType
            llaisys_dtype = _safetensors_dtype_to_llaisys(dtype)
            
            # Create llaisys Tensor
            tensor = Tensor(
                shape=shape,
                dtype=llaisys_dtype,
                device=device.lower() == "cuda" and DeviceType.NVIDIA or DeviceType.CPU,
                device_id=device_id
            )
            
            # Get tensor data as raw bytes
            data = f.get_raw(name)
            
            # Load raw data into llaisys Tensor
            _load_raw_bytes_into_tensor(tensor, data)
            
            tensors[name] = tensor
    
    return tensors


def _safetensors_dtype_to_llaisys(dtype: str) -> DataType:
    """
    Convert safetensors dtype string to llaisys DataType.
    
    Safetensors dtype strings:
    - "F32": float32
    - "F16": float16
    - "BF16": bfloat16
    - "F64": float64
    - "I64": int64
    - "I32": int32
    - "I16": int16
    - "I8": int8
    - "U64": uint64
    - "U32": uint32
    - "U16": uint16
    - "U8": uint8
    - "BOOL": bool
    """
    dtype_mapping = {
        "F32": DataType.F32,
        "F16": DataType.F16,
        "BF16": DataType.BF16,
        "F64": DataType.F64,
        "I64": DataType.I64,
        "I32": DataType.I32,
        "I16": DataType.I16,
        "I8": DataType.I8,
        "U64": DataType.U64,
        "U32": DataType.U32,
        "U16": DataType.U16,
        "U8": DataType.U8,
        "BOOL": DataType.BOOL
    }
    
    if dtype not in dtype_mapping:
        raise ValueError(f"Unsupported dtype: {dtype}")
    
    return dtype_mapping[dtype]


def _load_raw_bytes_into_tensor(tensor: Tensor, raw_bytes: bytes) -> None:
    """
    Load raw bytes data into llaisys Tensor.
    
    Args:
        tensor: llaisys Tensor to load data into
        raw_bytes: Raw bytes data from safetensors
    """
    # Get data pointer from the llaisys Tensor
    tensor_ptr = tensor.data_ptr()
    
    # Ensure the data size matches
    expected_size = _calculate_tensor_size(tensor)
    if len(raw_bytes) != expected_size:
        raise ValueError(f"Data size mismatch: expected {expected_size} bytes, got {len(raw_bytes)} bytes")
    
    # Copy raw bytes into the tensor memory
    src_buffer = ctypes.create_string_buffer(raw_bytes)
    ctypes.memmove(tensor_ptr, src_buffer, expected_size)


def _calculate_tensor_size(tensor: Tensor) -> int:
    """
    Calculate the total size in bytes of a tensor based on its shape and dtype.
    """
    dtype = tensor.dtype()
    shape = tensor.shape()
    
    # Get the size of each element in bytes
    element_size = _get_element_size(dtype)
    
    # Calculate total elements
    total_elements = 1
    for dim in shape:
        total_elements *= dim
    
    return total_elements * element_size


def _get_element_size(dtype: DataType) -> int:
    """
    Get the size in bytes of a single element for a given dtype.
    """
    size_mapping = {
        DataType.BYTE: 1,
        DataType.BOOL: 1,
        DataType.I8: 1,
        DataType.I16: 2,
        DataType.I32: 4,
        DataType.I64: 8,
        DataType.U8: 1,
        DataType.U16: 2,
        DataType.U32: 4,
        DataType.U64: 8,
        DataType.F16: 2,
        DataType.F32: 4,
        DataType.F64: 8,
        DataType.BF16: 2
    }
    
    if dtype not in size_mapping:
        raise ValueError(f"Unsupported dtype for size calculation: {dtype}")
    
    return size_mapping[dtype]