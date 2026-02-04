from ctypes import c_size_t, c_int, c_void_p
from .llaisys_types import llaisysDataType_t, llaisysDeviceType_t
from .tensor import llaisysTensor_t

# Handle type
llaisysKVCache_t = c_void_p


def load_kv_cache(LIB_LLAISYS):
    """Load KV Cache functions from the shared library"""
    
    # Define function signatures
    LIB_LLAISYS.llaisysKVCacheCreate.argtypes = [
        llaisysDataType_t,
        llaisysDeviceType_t,
        c_int,
        c_size_t,
        c_size_t,
        c_size_t,
        c_size_t,
        c_size_t
    ]
    LIB_LLAISYS.llaisysKVCacheCreate.restype = llaisysKVCache_t
    
    LIB_LLAISYS.llaisysKVCacheDestroy.argtypes = [llaisysKVCache_t]
    LIB_LLAISYS.llaisysKVCacheDestroy.restype = None
    
    LIB_LLAISYS.llaisysKVCacheUpdate.argtypes = [
        llaisysKVCache_t,
        llaisysTensor_t,
        llaisysTensor_t,
        c_size_t,
        c_size_t
    ]
    LIB_LLAISYS.llaisysKVCacheUpdate.restype = None
    
    LIB_LLAISYS.llaisysKVCacheGet.argtypes = [
        llaisysKVCache_t,
        llaisysTensor_t,
        llaisysTensor_t,
        c_size_t
    ]
    LIB_LLAISYS.llaisysKVCacheGet.restype = None
    
    LIB_LLAISYS.llaisysKVCacheReset.argtypes = [llaisysKVCache_t]
    LIB_LLAISYS.llaisysKVCacheReset.restype = None
    
    LIB_LLAISYS.llaisysKVCacheGetSize.argtypes = [llaisysKVCache_t, c_size_t]
    LIB_LLAISYS.llaisysKVCacheGetSize.restype = c_size_t
    
    LIB_LLAISYS.llaisysKVCacheExpand.argtypes = [llaisysKVCache_t, c_size_t]
    LIB_LLAISYS.llaisysKVCacheExpand.restype = None

    LIB_LLAISYS.llaisysKVCacheGetMaxSeqLen.argtypes = [llaisysKVCache_t]
    LIB_LLAISYS.llaisysKVCacheGetMaxSeqLen.restype = c_size_t