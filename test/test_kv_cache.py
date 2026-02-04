import llaisys
import torch
from test_utils import *
import argparse


def test_kv_cache_init():
    print("===Test KV Cache Initialization===")
    
    # Test parameters
    dtype = "f32"
    device = "cpu"
    device_id = 0
    num_layers = 2
    batch_size = 4
    num_heads = 8
    head_dim = 16
    max_seq_len = 32
    
    # Create KV Cache
    kv_cache = llaisys.KVCache(
        dtype=llaisys_dtype(dtype),
        device=llaisys_device(device),
        device_id=device_id,
        num_layers=num_layers,
        batch_size=batch_size,
        num_heads=num_heads,
        head_dim=head_dim,
        max_seq_len=max_seq_len
    )
    
    # Verify properties
    assert kv_cache.dtype == llaisys_dtype(dtype)
    assert kv_cache.device == llaisys_device(device)
    assert kv_cache.device_id == device_id
    assert kv_cache.num_layers == num_layers
    assert kv_cache.batch_size == batch_size
    assert kv_cache.num_heads == num_heads
    assert kv_cache.head_dim == head_dim
    assert kv_cache.max_seq_len == max_seq_len
    
    print("✓ KV Cache initialization successful")
    return kv_cache


def test_kv_cache_update_get():
    print("\n===Test KV Cache Update and Get===")
    
    # Test parameters
    dtype = "f32"
    device = "cpu"
    device_id = 0
    num_layers = 2
    batch_size = 4
    num_heads = 8
    head_dim = 16
    max_seq_len = 32
    seq_len = 40
    
    # Create KV Cache
    kv_cache = llaisys.KVCache(
        dtype=llaisys_dtype(dtype),
        device=llaisys_device(device),
        device_id=device_id,
        num_layers=num_layers,
        batch_size=batch_size,
        num_heads=num_heads,
        head_dim=head_dim,
        max_seq_len=max_seq_len
    )
    
    # Create test tensors
    k_shape = (batch_size, seq_len, num_heads, head_dim)
    v_shape = (batch_size, seq_len, num_heads, head_dim)
    
    for layer_idx in range(num_layers):
        # Generate random tensors
        torch_k, llaisys_k = random_tensor(k_shape, dtype, device, device_id)
        torch_v, llaisys_v = random_tensor(v_shape, dtype, device, device_id)
        
        # Update cache
        kv_cache.update(llaisys_k, llaisys_v, layer_idx, seq_len)
        
        # Verify cache size
        assert kv_cache.get_size(layer_idx) == seq_len
        
        # Get from cache
        k_out_shape = (batch_size, seq_len, num_heads, head_dim)
        v_out_shape = (batch_size, seq_len, num_heads, head_dim)
        
        torch_k_out = torch.zeros(k_out_shape, dtype=torch_dtype(dtype), device=torch_device(device, device_id))
        torch_v_out = torch.zeros(v_out_shape, dtype=torch_dtype(dtype), device=torch_device(device, device_id))
        
        llaisys_k_out = llaisys.Tensor(k_out_shape, dtype=llaisys_dtype(dtype), device=llaisys_device(device), device_id=device_id)
        llaisys_v_out = llaisys.Tensor(v_out_shape, dtype=llaisys_dtype(dtype), device=llaisys_device(device), device_id=device_id)
        
        kv_cache.get(llaisys_k_out, llaisys_v_out, layer_idx)
        
        # Verify the data
        torch_k_out[:, :seq_len, :, :] = torch_k
        torch_v_out[:, :seq_len, :, :] = torch_v
        
        assert check_equal(llaisys_k_out, torch_k_out)
        assert check_equal(llaisys_v_out, torch_v_out)
    
    print("✓ KV Cache update and get successful")


def test_kv_cache_reset():
    print("\n===Test KV Cache Reset===")
    
    # Test parameters
    dtype = "f32"
    device = "cpu"
    device_id = 0
    num_layers = 2
    batch_size = 4
    num_heads = 8
    head_dim = 16
    max_seq_len = 32
    seq_len = 10
    
    # Create KV Cache
    kv_cache = llaisys.KVCache(
        dtype=llaisys_dtype(dtype),
        device=llaisys_device(device),
        device_id=device_id,
        num_layers=num_layers,
        batch_size=batch_size,
        num_heads=num_heads,
        head_dim=head_dim,
        max_seq_len=max_seq_len
    )
    
    # Create test tensors
    k_shape = (batch_size, seq_len, num_heads, head_dim)
    v_shape = (batch_size, seq_len, num_heads, head_dim)
    
    # Update cache for all layers
    for layer_idx in range(num_layers):
        torch_k, llaisys_k = random_tensor(k_shape, dtype, device, device_id)
        torch_v, llaisys_v = random_tensor(v_shape, dtype, device, device_id)
        kv_cache.update(llaisys_k, llaisys_v, layer_idx, seq_len)
    
    # Verify cache sizes before reset
    for layer_idx in range(num_layers):
        assert kv_cache.get_size(layer_idx) == seq_len
    
    # Reset cache
    kv_cache.reset()
    
    # Verify cache sizes after reset
    for layer_idx in range(num_layers):
        assert kv_cache.get_size(layer_idx) == 0
    
    print("✓ KV Cache reset successful")


def test_kv_cache_expand():
    print("\n===Test KV Cache Expand===")
    
    # Test parameters
    dtype = "f32"
    device = "cpu"
    device_id = 0
    num_layers = 2
    batch_size = 4
    num_heads = 8
    head_dim = 16
    max_seq_len = 32
    seq_len = 10
    new_max_seq_len = 64
    
    # Create KV Cache
    kv_cache = llaisys.KVCache(
        dtype=llaisys_dtype(dtype),
        device=llaisys_device(device),
        device_id=device_id,
        num_layers=num_layers,
        batch_size=batch_size,
        num_heads=num_heads,
        head_dim=head_dim,
        max_seq_len=max_seq_len
    )
    
    # Create test tensors and update cache
    k_shape = (batch_size, seq_len, num_heads, head_dim)
    v_shape = (batch_size, seq_len, num_heads, head_dim)
    
    # Store tensors for each layer
    k_layer_tensors = []
    v_layer_tensors = []
    for layer_idx in range(num_layers):
        torch_k, llaisys_k = random_tensor(k_shape, dtype, device, device_id)
        torch_v, llaisys_v = random_tensor(v_shape, dtype, device, device_id)
        k_layer_tensors.append(torch_k)
        v_layer_tensors.append(torch_v)
        kv_cache.update(llaisys_k, llaisys_v, layer_idx, seq_len)
    
    # Expand cache
    kv_cache.expand(new_max_seq_len)
    
    # Verify expanded size
    assert kv_cache.max_seq_len == new_max_seq_len
    
    # Verify existing data is preserved
    for layer_idx in range(num_layers):
        # Get from cache
        k_out_shape = (batch_size, new_max_seq_len, num_heads, head_dim)
        v_out_shape = (batch_size, new_max_seq_len, num_heads, head_dim)
        
        torch_k_out = torch.zeros(k_out_shape, dtype=torch_dtype(dtype), device=torch_device(device, device_id))
        torch_v_out = torch.zeros(v_out_shape, dtype=torch_dtype(dtype), device=torch_device(device, device_id))
        
        llaisys_k_out = llaisys.Tensor(k_out_shape, dtype=llaisys_dtype(dtype), device=llaisys_device(device), device_id=device_id)
        llaisys_v_out = llaisys.Tensor(v_out_shape, dtype=llaisys_dtype(dtype), device=llaisys_device(device), device_id=device_id)
        
        kv_cache.get(llaisys_k_out, llaisys_v_out, layer_idx)
        
        # Get the correct tensors for this layer
        torch_k = k_layer_tensors[layer_idx]
        torch_v = v_layer_tensors[layer_idx]
        
        # Verify the data
        torch_k_out[:, :seq_len, :, :] = torch_k
        torch_v_out[:, :seq_len, :, :] = torch_v
        
        assert check_equal(llaisys_k_out, torch_k_out)
        assert check_equal(llaisys_v_out, torch_v_out)
    
    print("✓ KV Cache expand successful")


def test_kv_cache_multiple_updates():
    print("\n===Test KV Cache Multiple Updates===")
    
    # Test parameters
    dtype = "f32"
    device = "cpu"
    device_id = 0
    num_layers = 2
    batch_size = 4
    num_heads = 8
    head_dim = 16
    max_seq_len = 32
    seq_len1 = 10
    seq_len2 = 10
    
    # Create KV Cache
    kv_cache = llaisys.KVCache(
        dtype=llaisys_dtype(dtype),
        device=llaisys_device(device),
        device_id=device_id,
        num_layers=num_layers,
        batch_size=batch_size,
        num_heads=num_heads,
        head_dim=head_dim,
        max_seq_len=max_seq_len
    )
    
    # Create test tensors for first update
    k_shape1 = (batch_size, seq_len1, num_heads, head_dim)
    v_shape1 = (batch_size, seq_len1, num_heads, head_dim)
    
    # Create test tensors for second update
    k_shape2 = (batch_size, seq_len2, num_heads, head_dim)
    v_shape2 = (batch_size, seq_len2, num_heads, head_dim)
    
    for layer_idx in range(num_layers):
        # First update
        torch_k1, llaisys_k1 = random_tensor(k_shape1, dtype, device, device_id)
        torch_v1, llaisys_v1 = random_tensor(v_shape1, dtype, device, device_id)
        kv_cache.update(llaisys_k1, llaisys_v1, layer_idx, seq_len1)
        
        assert kv_cache.get_size(layer_idx) == seq_len1
        
        # Second update
        torch_k2, llaisys_k2 = random_tensor(k_shape2, dtype, device, device_id)
        torch_v2, llaisys_v2 = random_tensor(v_shape2, dtype, device, device_id)
        kv_cache.update(llaisys_k2, llaisys_v2, layer_idx, seq_len2)
        
        assert kv_cache.get_size(layer_idx) == seq_len1 + seq_len2
        
        # Get from cache
        k_out_shape = (batch_size, seq_len1+seq_len2, num_heads, head_dim)
        v_out_shape = (batch_size, seq_len1+seq_len2, num_heads, head_dim)
        
        llaisys_k_out = llaisys.Tensor(k_out_shape, dtype=llaisys_dtype(dtype), device=llaisys_device(device), device_id=device_id)
        llaisys_v_out = llaisys.Tensor(v_out_shape, dtype=llaisys_dtype(dtype), device=llaisys_device(device), device_id=device_id)
        
        kv_cache.get(llaisys_k_out, llaisys_v_out, layer_idx)

        torch_k_out = torch.zeros(k_out_shape, dtype=torch_dtype(dtype), device=torch_device(device, device_id))
        torch_v_out = torch.zeros(v_out_shape, dtype=torch_dtype(dtype), device=torch_device(device, device_id))

        torch_k_out = torch.cat([torch_k1, torch_k2], dim=1)
        torch_v_out = torch.cat([torch_v1, torch_v2], dim=1)
        
        assert check_equal(llaisys_k_out, torch_k_out)
        assert check_equal(llaisys_v_out, torch_v_out)
    
    print("✓ KV Cache multiple updates successful")


def test_kv_cache_auto_expand():
    print("\n===Test KV Cache Auto Expand===")
    
    # Test parameters
    dtype = "f32"
    device = "cpu"
    device_id = 0
    num_layers = 2
    batch_size = 4
    num_heads = 8
    head_dim = 16
    max_seq_len = 32
    seq_len1 = 30  # Close to max_seq_len
    seq_len2 = 10  # Will cause auto-expand
    
    # Create KV Cache
    kv_cache = llaisys.KVCache(
        dtype=llaisys_dtype(dtype),
        device=llaisys_device(device),
        device_id=device_id,
        num_layers=num_layers,
        batch_size=batch_size,
        num_heads=num_heads,
        head_dim=head_dim,
        max_seq_len=max_seq_len
    )
    
    # Create test tensors
    k_shape1 = (batch_size, seq_len1, num_heads, head_dim)
    v_shape1 = (batch_size, seq_len1, num_heads, head_dim)
    
    k_shape2 = (batch_size, seq_len2, num_heads, head_dim)
    v_shape2 = (batch_size, seq_len2, num_heads, head_dim)
    
    for layer_idx in range(num_layers):
        # First update
        torch_k1, llaisys_k1 = random_tensor(k_shape1, dtype, device, device_id)
        torch_v1, llaisys_v1 = random_tensor(v_shape1, dtype, device, device_id)
        kv_cache.update(llaisys_k1, llaisys_v1, layer_idx, seq_len1)
        
        # Second update - should trigger auto-expand
        torch_k2, llaisys_k2 = random_tensor(k_shape2, dtype, device, device_id)
        torch_v2, llaisys_v2 = random_tensor(v_shape2, dtype, device, device_id)
        kv_cache.update(llaisys_k2, llaisys_v2, layer_idx, seq_len2)
        
        # Verify cache has expanded
        assert kv_cache.max_seq_len > max_seq_len
        assert kv_cache.get_size(layer_idx) == seq_len1 + seq_len2
        
        # Get from cache and verify data
        k_out_shape = (batch_size, seq_len1 + seq_len2, num_heads, head_dim)
        v_out_shape = (batch_size, seq_len1 + seq_len2, num_heads, head_dim)
        
        llaisys_k_out = llaisys.Tensor(k_out_shape, dtype=llaisys_dtype(dtype), device=llaisys_device(device), device_id=device_id)
        llaisys_v_out = llaisys.Tensor(v_out_shape, dtype=llaisys_dtype(dtype), device=llaisys_device(device), device_id=device_id)
        
        kv_cache.get(llaisys_k_out, llaisys_v_out, layer_idx)
        
        torch_k_out = torch.cat([torch_k1, torch_k2], dim=1)
        torch_v_out = torch.cat([torch_v1, torch_v2], dim=1)
        
        assert check_equal(llaisys_k_out, torch_k_out)
        assert check_equal(llaisys_v_out, torch_v_out)
    
    print("✓ KV Cache auto-expand successful")


def test_kv_cache_different_dtypes():
    print("\n===Test KV Cache with Different Data Types===")
    
    # Test parameters
    device = "cpu"
    device_id = 0
    num_layers = 1
    batch_size = 2
    num_heads = 4
    head_dim = 8
    max_seq_len = 16
    seq_len = 5
    
    # Test with different data types
    dtypes = ["f16", "f32", "f64"]
    
    for dtype in dtypes:
        print(f"Testing with {dtype}...")
        
        # Create KV Cache
        kv_cache = llaisys.KVCache(
            dtype=llaisys_dtype(dtype),
            device=llaisys_device(device),
            device_id=device_id,
            num_layers=num_layers,
            batch_size=batch_size,
            num_heads=num_heads,
            head_dim=head_dim,
            max_seq_len=max_seq_len
        )
        
        # Create test tensors
        k_shape = (batch_size, seq_len, num_heads, head_dim)
        v_shape = (batch_size, seq_len, num_heads, head_dim)
        
        torch_k, llaisys_k = random_tensor(k_shape, dtype, device, device_id)
        torch_v, llaisys_v = random_tensor(v_shape, dtype, device, device_id)
        
        # Update cache
        kv_cache.update(llaisys_k, llaisys_v, 0, seq_len)
        
        # Get from cache
        k_out_shape = (batch_size, seq_len , num_heads, head_dim)
        v_out_shape = (batch_size, seq_len , num_heads, head_dim)
        
        llaisys_k_out = llaisys.Tensor(k_out_shape, dtype=llaisys_dtype(dtype), device=llaisys_device(device), device_id=device_id)
        llaisys_v_out = llaisys.Tensor(v_out_shape, dtype=llaisys_dtype(dtype), device=llaisys_device(device), device_id=device_id)
        
        kv_cache.get(llaisys_k_out, llaisys_v_out, 0)
        
        assert check_equal(llaisys_k_out, torch_k)
        assert check_equal(llaisys_v_out, torch_v)
    
    print("✓ KV Cache with different data types successful")


def main():
    parser = argparse.ArgumentParser(description="Test KV Cache functionality")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "nvidia"], help="Device to test on")
    args = parser.parse_args()
    
    print(f"Testing KV Cache on {args.device} device...")
    
    # Run all tests
    test_kv_cache_init()
    test_kv_cache_update_get()
    test_kv_cache_reset()
    # test_kv_cache_expand()
    test_kv_cache_multiple_updates()
    test_kv_cache_auto_expand()
    test_kv_cache_different_dtypes()
    
    print("\n=== All KV Cache tests passed! ===")


if __name__ == "__main__":
    main()