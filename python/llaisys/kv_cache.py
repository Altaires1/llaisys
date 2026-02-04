from typing import Tuple

from .libllaisys import LIB_LLAISYS
from .libllaisys import DeviceType
from .libllaisys import DataType
from .libllaisys import llaisysKVCache_t
from .tensor import Tensor


class KVCache:
    """KV Cache wrapper for llaisys KV Cache functionality"""

    def __init__(
        self,
        dtype: DataType,
        device: DeviceType,
        device_id: int,
        num_layers: int,
        batch_size: int,
        num_heads: int,
        head_dim: int,
        max_seq_len: int
    ):
        """Initialize a new KV Cache

        Args:
            dtype: Data type of the cache
            device: Device type (CPU, GPU, etc.)
            device_id: Device ID
            num_layers: Number of transformer layers
            batch_size: Batch size
            num_heads: Number of attention heads
            head_dim: Dimension of each attention head
            max_seq_len: Maximum sequence length for the cache
        """
        self._cache = LIB_LLAISYS.llaisysKVCacheCreate(
            dtype,
            device,
            device_id,
            num_layers,
            batch_size,
            num_heads,
            head_dim,
            max_seq_len
        )
        self._dtype = dtype
        self._device = device
        self._device_id = device_id
        self._num_layers = num_layers
        self._batch_size = batch_size
        self._num_heads = num_heads
        self._head_dim = head_dim
        self._max_seq_len = max_seq_len

    def __del__(self):
        """Destroy the KV Cache and free all resources"""
        if hasattr(self, '_cache') and self._cache is not None:
            LIB_LLAISYS.llaisysKVCacheDestroy(self._cache)
            self._cache = None

    def update(self, k: Tensor, v: Tensor, layer_idx: int, seq_len: int):
        """Update the KV Cache with new K and V tensors for a specific layer

        Args:
            k: New K tensor to add to the cache
            v: New V tensor to add to the cache
            layer_idx: Index of the transformer layer
            seq_len: Length of the new sequence to add
        """
        LIB_LLAISYS.llaisysKVCacheUpdate(
            self._cache,
            k.lib_tensor(),
            v.lib_tensor(),
            layer_idx,
            seq_len
        )

        self._max_seq_len = LIB_LLAISYS.llaisysKVCacheGetMaxSeqLen(self._cache)

    def get(self, k_out: Tensor, v_out: Tensor, layer_idx: int):
        """Get the cached K and V tensors for a specific layer

        Args:
            k_out: Output tensor to store the cached K
            v_out: Output tensor to store the cached V
            layer_idx: Index of the transformer layer
        """
        LIB_LLAISYS.llaisysKVCacheGet(
            self._cache,
            k_out.lib_tensor(),
            v_out.lib_tensor(),
            layer_idx
        )

    def reset(self):
        """Reset the KV Cache to its initial state"""
        LIB_LLAISYS.llaisysKVCacheReset(self._cache)

    def get_size(self, layer_idx: int) -> int:
        """Get the current size (sequence length) of the cached KV tensors for a specific layer

        Args:
            layer_idx: Index of the transformer layer

        Returns:
            Current size (sequence length) of the cache
        """
        return LIB_LLAISYS.llaisysKVCacheGetSize(self._cache, layer_idx)

    def expand(self, new_max_seq_len: int):
        """Expand the KV Cache to accommodate a larger sequence length

        Args:
            new_max_seq_len: New maximum sequence length
        """
        LIB_LLAISYS.llaisysKVCacheExpand(self._cache, new_max_seq_len)
        self._max_seq_len = new_max_seq_len

    def lib_cache(self) -> llaisysKVCache_t:
        """Get the underlying llaisysKVCache_t pointer

        Returns:
            The underlying llaisysKVCache_t pointer
        """
        return self._cache

    @property
    def dtype(self) -> DataType:
        """Get the data type of the cache"""
        return self._dtype

    @property
    def device(self) -> DeviceType:
        """Get the device type of the cache"""
        return self._device

    @property
    def device_id(self) -> int:
        """Get the device ID of the cache"""
        return self._device_id

    @property
    def num_layers(self) -> int:
        """Get the number of transformer layers"""
        return self._num_layers

    @property
    def batch_size(self) -> int:
        """Get the batch size"""
        return self._batch_size

    @property
    def num_heads(self) -> int:
        """Get the number of attention heads"""
        return self._num_heads

    @property
    def head_dim(self) -> int:
        """Get the dimension of each attention head"""
        return self._head_dim

    @property
    def max_seq_len(self) -> int:
        """Get the maximum sequence length of the cache"""
        return self._max_seq_len