import sys
import os
import torch

# Add project root to path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

import llaisys
from llaisys.models.utils import linear_nd, embedding_nd, argmax_nd, rms_norm_nd, rope_nd, self_attention_nd, swiglu_nd, add_nd
from test_utils import random_tensor, random_int_tensor, check_equal, benchmark

# Import torch reference implementations from existing op tests
from ops.linear import torch_linear
from ops.embedding import torch_embedding
from ops.argmax import torch_argmax
from ops.rms_norm import torch_rms_norm
from ops.rope import torch_rope
from ops.self_attention import torch_self_attention
from ops.swiglu import torch_swiglu
from ops.add import torch_add

def test_linear_nd(device_name="cpu", profile=False):
    print("Testing linear_nd...")
    # Shape: (2, 3, 4) @ (5, 4)^T -> (2, 3, 5)
    x_shape = (2, 3, 4)
    w_shape = (5, 4)
    out_shape = (2, 3, 5)
    dtype_name = "f32"
    
    x, x_ = random_tensor(x_shape, dtype_name, device_name)
    w, w_ = random_tensor(w_shape, dtype_name, device_name)
    b, b_ = random_tensor((w_shape[0],), dtype_name, device_name)
    out, out_ = random_tensor(out_shape, dtype_name, device_name)
    
    # ND utility
    res_ = linear_nd(x_, w_, b_)
    
    # Torch reference
    torch_linear(out, x, w, b)
    
    assert check_equal(res_, out)
    print("linear_nd test passed!")

def test_embedding_nd(device_name="cpu", profile=False):
    print("Testing embedding_nd...")
    idx_shape = (2, 3)
    embd_shape = (10, 8)
    dtype_name = "f32"
    
    idx, idx_ = random_int_tensor(idx_shape, device_name, high=embd_shape[0])
    embd, embd_ = random_tensor(embd_shape, dtype_name, device_name)
    out, out_ = random_tensor((idx_shape[0], idx_shape[1], embd_shape[1]), dtype_name, device_name)
    
    res_ = embedding_nd(idx_, embd_)
    
    # Torch reference (using view to match existing torch_embedding's 2D expectation if necessary)
    # The existing torch_embedding expects (idx_shape[0], embd_shape[1])
    # For ND, we can just use torch indexing
    expected = embd[idx]
    
    assert check_equal(res_, expected)
    print("embedding_nd test passed!")

def test_argmax_nd(device_name="cpu", profile=False):
    print("Testing argmax_nd...")
    shape = (2, 3, 10)
    dtype_name = "f32"
    
    x, x_ = random_tensor(shape, dtype_name, device_name)
    
    max_idx_, max_val_ = argmax_nd(x_)
    
    # Torch reference
    max_val, max_idx = torch.max(x, dim=-1)
    
    assert check_equal(max_idx_, max_idx)
    assert check_equal(max_val_, max_val)
    print("argmax_nd test passed!")

def test_rms_norm_nd(device_name="cpu", profile=False):
    print("Testing rms_norm_nd...")
    shape = (2, 3, 16)
    dtype_name = "f32"
    eps = 1e-5
    
    x, x_ = random_tensor(shape, dtype_name, device_name)
    w, w_ = random_tensor((shape[-1],), dtype_name, device_name)
    
    res_ = rms_norm_nd(x_, w_, eps)
    
    # Torch reference
    # torch_rms_norm expects (shape, x, w, eps) and modifies ans
    ans = torch.zeros_like(x)
    torch_rms_norm(ans, x, w, eps)
    
    assert check_equal(res_, ans)
    print("rms_norm_nd test passed!")

def test_rope_nd(device_name="cpu", profile=False):
    print("Testing rope_nd...")
    # ND Shape: (batch, seq, n_head, head_dim)
    shape = (2, 3, 4, 8)
    dtype_name = "f32"
    theta = 10000.0
    
    x, x_ = random_tensor(shape, dtype_name, device_name)
    pos_ids, pos_ids_ = random_int_tensor(shape[:-2], device_name, high=100)
    
    res_ = rope_nd(x_, pos_ids_, theta)
    
    # Torch reference
    # torch_rope expects (y, x, pos_ids, theta) where y, x are 3D (seq, heads, dim)
    # and pos_ids is 1D (seq)
    
    # Flatten ND to 3D to reuse torch_rope
    x_flat = x.view(-1, shape[-2], shape[-1])
    pos_ids_flat = pos_ids.view(-1)
    y_flat = torch.zeros_like(x_flat)
    
    torch_rope(y_flat, x_flat, pos_ids_flat, theta)
    expected = y_flat.view(shape)
    
    assert check_equal(res_, expected)
    print("rope_nd test passed!")

def test_self_attention_nd(device_name="cpu", profile=False):
    print("Testing self_attention_nd...")
    # ND Shape: (batch, qlen, nh, hd)
    q_shape = (2, 3, 4, 8)
    k_shape = (2, 5, 2, 8) # MQA/GQA: nkvh=2
    v_shape = (2, 5, 2, 8)
    dtype_name = "f32"
    scale = 0.5
    
    q, q_ = random_tensor(q_shape, dtype_name, device_name)
    k, k_ = random_tensor(k_shape, dtype_name, device_name)
    v, v_ = random_tensor(v_shape, dtype_name, device_name)
    
    res_ = self_attention_nd(q_, k_, v_, scale)
    
    # Torch reference
    # torch_self_attention expects (attn_val, query, key, value, scale)
    # It handles broadcasting and transposition internally.
    # We need to process batch by batch for the reference if we want to be sure,
    # or just use torch_self_attention if it supports ND.
    # Looking at torch_self_attention in self_attention.py:
    # it uses .transpose(-2, -3) and @ which works for ND in torch.
    
    expected = torch.zeros((2, 3, 4, 8), dtype=q.dtype, device=q.device)
    torch_self_attention(expected, q, k, v, scale)
    
    assert check_equal(res_, expected)
    print("self_attention_nd test passed!")

def test_swiglu_nd(device_name="cpu", profile=False):
    print("Testing swiglu_nd...")
    shape = (2, 3, 16)
    dtype_name = "f32"
    
    gate, gate_ = random_tensor(shape, dtype_name, device_name)
    up, up_ = random_tensor(shape, dtype_name, device_name)
    
    res_ = swiglu_nd(gate_, up_)
    
    # Torch reference
    expected = torch.zeros_like(gate)
    torch_swiglu(expected, gate, up)
    
    assert check_equal(res_, expected)
    print("swiglu_nd test passed!")

def test_add_nd(device_name="cpu", profile=False):
    print("Testing add_nd...")
    shape = (2, 3, 4, 16)
    dtype_name = "f32"
    
    a, a_ = random_tensor(shape, dtype_name, device_name)
    b, b_ = random_tensor(shape, dtype_name, device_name)
    
    res_ = add_nd(a_, b_)
    
    # Torch reference
    expected = torch.zeros_like(a)
    torch_add(expected, a, b)
    
    assert check_equal(res_, expected)
    print("add_nd test passed!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia"], type=str)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    
    print(f"Testing ND utilities on {args.device}")
    test_linear_nd(args.device, args.profile)
    test_embedding_nd(args.device, args.profile)
    test_argmax_nd(args.device, args.profile)
    test_rms_norm_nd(args.device, args.profile)
    test_rope_nd(args.device, args.profile)
    test_self_attention_nd(args.device, args.profile)
    test_swiglu_nd(args.device, args.profile)
    test_add_nd(args.device, args.profile)
    
    print("\033[92mAll ND utility tests passed!\033[0m")
