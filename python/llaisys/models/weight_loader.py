"""Weight loader for Qwen2 model"""
import numpy as np
from pathlib import Path
from typing import Dict
from ..tensor import Tensor
from ..libllaisys import DeviceType, DataType
from ..runtime import RuntimeAPI
from ctypes import c_void_p
import struct
import json
import gc


class WeightLoader:
    """Load model weights from safetensors files"""
    
    def __init__(self, model_path: str, device: DeviceType, device_id: int = 0):
        """Initialize weight loader
        
        Args:
            model_path: Path to the model directory
            device: Target device (CPU or NVIDIA)
            device_id: Device ID
        """
        try:
            from tqdm import tqdm
            self.tqdm = tqdm
        except ImportError:
            self.tqdm = None
            
        self.model_path = Path(model_path)
        self.device = device
        self.device_id = device_id
        self.runtime = RuntimeAPI(device)
        self._weights: Dict[str, Tensor] = {}
        self._weight_arrays: Dict[str, np.ndarray] = {}
    
    def load_weights(self) -> Dict[str, Tensor]:
        """Load all weights from safetensors files
        
        Returns:
            Dictionary mapping weight names to Tensor objects
        """
        # Load all safetensors files
        weight_files = sorted(self.model_path.glob("*.safetensors"))
        
        if not weight_files:
            raise FileNotFoundError(f"No safetensors files found in {self.model_path}")
        
        print(f"Loading {len(weight_files)} weight file(s)")
        
        file_iterator = weight_files
        if self.tqdm:
            file_iterator = self.tqdm(weight_files, desc="Loading weights", unit="file")
        
        for weight_file in file_iterator:
            self._load_from_safetensors(weight_file)
        
        # Clean up and force garbage collection
        self._weight_arrays.clear()
        gc.collect()
        
        print(f"Loaded {len(self._weights)} weight tensors")
        return self._weights
    
    def _load_from_safetensors(self, file_path: Path):
        """Load weights from a single safetensors file using low-level parsing
        
        Args:
            file_path: Path to the safetensors file
        """
        try:
            # Try using safetensors library first for non-bfloat16 tensors
            from safetensors.safe_open import safe_open
            
            with safe_open(str(file_path), framework="np") as f:
                for key in f.keys():
                    try:
                        data = f.get_tensor(key)
                        self._weight_arrays[key] = data
                        self._create_tensor_from_array(key, data)
                        print(f"  Loaded {key}: shape={data.shape}, dtype={data.dtype}")
                    except Exception as e:
                        # If safetensors fails (e.g., for bfloat16), try manual parsing
                        print(f"  safetensors failed for {key}: {e}, trying manual parse...")
                        try:
                            self._load_tensor_manual(file_path, key)
                        except Exception as e2:
                            print(f"Warning: Failed to load weight {key}: {e2}")
        except ImportError:
            # Fallback: manual safetensors parsing
            print("Using manual safetensors parsing for all tensors...")
            self._load_from_safetensors_manual(file_path)
    
    def _load_from_safetensors_manual(self, file_path: Path):
        """Manually parse safetensors file (handles bfloat16)
        
        Args:
            file_path: Path to the safetensors file
        """
        with open(file_path, "rb") as f:
            # Read header size (8 bytes, little-endian)
            header_size_bytes = f.read(8)
            header_size = int.from_bytes(header_size_bytes, byteorder='little')
            
            # Read header JSON
            header_json = f.read(header_size).decode('utf-8')
            header = json.loads(header_json)
            
            # Load each tensor (skip __metadata__)
            for key, tensor_info in header.items():
                if key == "__metadata__":
                    continue
                    
                try:
                    dtype_str = tensor_info['dtype']
                    shape = tuple(tensor_info['shape'])
                    data_offsets = tensor_info['data_offsets']
                    offset_start, offset_end = data_offsets[0], data_offsets[1]
                    
                    # Calculate expected data size
                    data_size = offset_end - offset_start
                    
                    # Read tensor data
                    f.seek(8 + header_size + offset_start)
                    tensor_bytes = f.read(data_size)
                    
                    # Verify data size matches
                    if len(tensor_bytes) != data_size:
                        print(f"Warning: Expected {data_size} bytes for {key}, got {len(tensor_bytes)}")
                    
                    # Parse based on dtype
                    data = self._parse_tensor_bytes(tensor_bytes, dtype_str, shape)
                    
                    self._weight_arrays[key] = data
                    self._create_tensor_from_array(key, data, dtype_str)
                    print(f"  Loaded {key}: shape={shape}, dtype={dtype_str}, size={data_size} bytes")
                except Exception as e:
                    print(f"Warning: Failed to load weight {key}: {e}")
    
    def _load_tensor_manual(self, file_path: Path, key: str):
        """Manually load a single tensor from safetensors file
        
        Args:
            file_path: Path to the safetensors file
            key: Tensor key to load
        """
        with open(file_path, "rb") as f:
            # Read header size
            header_size_bytes = f.read(8)
            header_size = int.from_bytes(header_size_bytes, byteorder='little')
            
            # Read header
            header_json = f.read(header_size).decode('utf-8')
            header = json.loads(header_json)
            
            if key not in header:
                raise KeyError(f"Tensor {key} not found in safetensors file")
            
            tensor_info = header[key]
            dtype_str = tensor_info['dtype']
            shape = tuple(tensor_info['shape'])
            offset_start, offset_end = tensor_info['data_offsets']
            
            # Read tensor data
            f.seek(8 + header_size + offset_start)
            tensor_bytes = f.read(offset_end - offset_start)
            
            # Parse tensor
            data = self._parse_tensor_bytes(tensor_bytes, dtype_str, shape)
            self._weight_arrays[key] = data
            self._create_tensor_from_array(key, data, dtype_str)
    
    def _parse_tensor_bytes(self, tensor_bytes: bytes, dtype_str: str, shape: tuple) -> np.ndarray:
        """Parse raw bytes into numpy array based on dtype
        
        Args:
            tensor_bytes: Raw tensor data
            dtype_str: Data type string (e.g., 'F32', 'BF16', 'F16', 'I32', 'I64')
            shape: Tensor shape
            
        Returns:
            Numpy array
        """
        # Normalize dtype string to uppercase for comparison
        dtype_str_upper = dtype_str.upper()
        
        if dtype_str_upper == "F32" or dtype_str_upper == "FLOAT32":
            data = np.frombuffer(tensor_bytes, dtype=np.float32)
        elif dtype_str_upper == "F16" or dtype_str_upper == "FLOAT16":
            data = np.frombuffer(tensor_bytes, dtype=np.float16)
        elif dtype_str_upper == "BF16" or dtype_str_upper == "BFLOAT16":
            # bfloat16 is 16-bit, stored as uint16 binary representation
            data = np.frombuffer(tensor_bytes, dtype=np.uint16)
        elif dtype_str_upper == "I32" or dtype_str_upper == "INT32":
            data = np.frombuffer(tensor_bytes, dtype=np.int32)
        elif dtype_str_upper == "I64" or dtype_str_upper == "INT64":
            data = np.frombuffer(tensor_bytes, dtype=np.int64)
        else:
            # Default to float32
            print(f"Warning: Unknown dtype '{dtype_str}', treating as float32")
            data = np.frombuffer(tensor_bytes, dtype=np.float32)
        
        # Verify data size matches shape
        expected_numel = int(np.prod(shape))
        actual_numel = data.shape[0]
        
        if expected_numel != actual_numel:
            print(f"Warning: Expected {expected_numel} elements but got {actual_numel}")
            print(f"  Shape: {shape}, actual data shape: {data.shape}")
        
        # Reshape to match the tensor shape
        try:
            data = data.reshape(shape)
        except ValueError as e:
            print(f"Error reshaping tensor: {e}")
            print(f"  Trying to reshape {data.shape} to {shape}")
            raise
        
        return data
    
    def _create_tensor_from_array(self, name: str, array: np.ndarray, dtype_str: str = None):
        """Create a llaisys Tensor from a numpy array and copy data
        
        Args:
            name: Weight name
            array: Numpy array containing the weight data
            dtype_str: String representation of data type
        """
        # Determine llaisys data type - prioritize dtype_str if provided
        if dtype_str:
            dtype_str_upper = dtype_str.upper()
            if "BF16" in dtype_str_upper or "BFLOAT16" in dtype_str_upper:
                dtype = DataType.BF16
            elif "F32" in dtype_str_upper or "FLOAT32" in dtype_str_upper:
                dtype = DataType.F32
            elif "F16" in dtype_str_upper or "FLOAT16" in dtype_str_upper:
                dtype = DataType.F16
            elif "I32" in dtype_str_upper or "INT32" in dtype_str_upper:
                dtype = DataType.I32
            elif "I64" in dtype_str_upper or "INT64" in dtype_str_upper:
                dtype = DataType.I64
            else:
                print(f"Warning: Unknown dtype string '{dtype_str}', inferring from array")
                dtype = self._infer_dtype_from_array(array)
        else:
            dtype = self._infer_dtype_from_array(array)
        
        # Create tensor
        tensor = Tensor(
            shape=tuple(array.shape),
            dtype=dtype,
            device=self.device,
            device_id=self.device_id
        )
        
        # Copy data to tensor
        array_ptr = array.ctypes.data_as(c_void_p)
        data_size = array.nbytes
        self.runtime.memcpy_sync(
            tensor.data_ptr(),
            array_ptr,
            data_size,
            self._get_memcpy_kind()
        )
        
        self._weights[name] = tensor
        
        # Clean up the numpy array to free memory
        del array
        gc.collect()
    
    def _infer_dtype_from_array(self, array: np.ndarray) -> 'DataType':
        """Infer llaisys DataType from numpy array dtype
        
        Args:
            array: Numpy array
            
        Returns:
            llaisys DataType
        """
        if array.dtype == np.float32:
            return DataType.F32
        elif array.dtype == np.float16:
            return DataType.F16
        elif array.dtype == np.uint16:
            # Assume uint16 is bfloat16
            return DataType.BF16
        elif array.dtype == np.int32:
            return DataType.I32
        elif array.dtype == np.int64:
            return DataType.I64
        else:
            print(f"Warning: Unsupported array dtype {array.dtype}, using F32")
            return DataType.F32
    
    def _get_memcpy_kind(self):
        """Get memcpy kind based on device type"""
        from ..libllaisys import MemcpyKind
        if self.device == DeviceType.CPU:
            return MemcpyKind.H2D
        else:
            return MemcpyKind.H2D
    
    def get_weight(self, name: str) -> Tensor:
        """Get a specific weight tensor
        
        Args:
            name: Weight name
            
        Returns:
            Tensor object
        """
        if name not in self._weights:
            raise KeyError(f"Weight {name} not found")
        return self._weights[name]
    
    def get_all_weights(self) -> Dict[str, Tensor]:
        """Get all loaded weights
        
        Returns:
            Dictionary of all weights
        """
        return self._weights


