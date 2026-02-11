"""Weight loader for Qwen2 model"""
import numpy as np
from pathlib import Path
from typing import Dict
from ..tensor import Tensor
from ..libllaisys import DeviceType, DataType
from ctypes import c_void_p
import gc
import ml_dtypes # New import

# Assume safetensors library is always available now
from safetensors import safe_open


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
        """Load weights from a single safetensors file using the safetensors library
        
        Args:
            file_path: Path to the safetensors file
        """
        with safe_open(str(file_path), framework="np") as f:
            keys_iterator = f.keys()
            if self.tqdm:
                keys_iterator = self.tqdm(keys_iterator, desc=f"Loading tensors from {file_path.name}", unit="tensor")

            for key in keys_iterator:
                data = f.get_tensor(key)
                self._weight_arrays[key] = data
                self._create_tensor_from_array(key, data)
    
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
            elif "F64" in dtype_str_upper or "FLOAT64" in dtype_str_upper:
                dtype = DataType.F64
            elif "I8" in dtype_str_upper or "INT8" in dtype_str_upper:
                dtype = DataType.I8
            elif "I16" in dtype_str_upper or "INT16" in dtype_str_upper:
                dtype = DataType.I16
            elif "I32" in dtype_str_upper or "INT32" in dtype_str_upper:
                dtype = DataType.I32
            elif "I64" in dtype_str_upper or "INT64" in dtype_str_upper:
                dtype = DataType.I64
            elif "U8" in dtype_str_upper or "UINT8" in dtype_str_upper:
                dtype = DataType.U8
            elif "U16" in dtype_str_upper or "UINT16" in dtype_str_upper:
                dtype = DataType.U16
            elif "U32" in dtype_str_upper or "UINT32" in dtype_str_upper:
                dtype = DataType.U32
            elif "U64" in dtype_str_upper or "UINT64" in dtype_str_upper:
                dtype = DataType.U64
            elif "BOOL" in dtype_str_upper:
                dtype = DataType.BOOL
            elif "C64" in dtype_str_upper or "COMPLEX64" in dtype_str_upper:
                dtype = DataType.C64
            elif "C128" in dtype_str_upper or "COMPLEX128" in dtype_str_upper:
                dtype = DataType.C128
            elif "INVALID" in dtype_str_upper or \
                 "BYTE" in dtype_str_upper or \
                 "F8" in dtype_str_upper or \
                 "C16" in dtype_str_upper or \
                 "C32" in dtype_str_upper:
                raise ValueError(f"Unsupported DataType encountered in safetensors metadata: '{dtype_str}' for tensor '{name}'")
            else:
                try:
                    dtype = self._infer_dtype_from_array(array)
                except ValueError as e:
                    raise ValueError(f"Could not determine DataType for '{dtype_str}' and array dtype {array.dtype} for tensor '{name}': {e}")
        else:
            dtype = self._infer_dtype_from_array(array)
        
        # Create tensor
        tensor = Tensor(
            shape=tuple(array.shape),
            dtype=dtype,
            device=self.device,
            device_id=self.device_id
        )
        
        # Load data into tensor using its load method
        tensor.load(array.ctypes.data_as(c_void_p))
        
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
        elif array.dtype == ml_dtypes.bfloat16: # Updated to use ml_dtypes.bfloat16
            return DataType.BF16
        elif array.dtype == np.float64:
            return DataType.F64
        elif array.dtype == np.int8:
            return DataType.I8
        elif array.dtype == np.int16:
            return DataType.I16
        elif array.dtype == np.int32:
            return DataType.I32
        elif array.dtype == np.int64:
            return DataType.I64
        elif array.dtype == np.uint8:
            return DataType.U8
        elif array.dtype == np.uint16:
            return DataType.U16
        elif array.dtype == np.uint32:
            return DataType.U32
        elif array.dtype == np.uint64:
            return DataType.U64
        elif array.dtype == np.bool_:
            return DataType.BOOL
        elif array.dtype == np.complex64:
            return DataType.C64
        elif array.dtype == np.complex128:
            return DataType.C128
        else:
            raise ValueError(f"Unsupported numpy array dtype: {array.dtype}")
    
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
