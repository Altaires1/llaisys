import unittest
import numpy as np
import ctypes
from llaisys.tensor import Tensor
from llaisys.libllaisys import DataType, DeviceType
from llaisys.models.transformer import TransformerBlock
from llaisys.kv_cache import KVCache

class TestTransformerBlock(unittest.TestCase):
    def setUp(self):
        # 模拟 Qwen2-1.5B 左右的参数
        self.hidden_size = 1536
        self.num_heads = 12
        self.num_kv_heads = 2
        self.head_dim = self.hidden_size // self.num_heads
        self.intermediate_size = 8960
        self.layer_idx = 0
        self.device = DeviceType.CPU
        self.rms_norm_eps = 1e-6
        self.rope_theta = 10000.0
        
        # 构造模拟权重字典
        self.weights = {}
        
        # 辅助函数：创建并初始化权重
        def add_weight(name, shape, is_bias=False):
            # llaisys 要求 bias 通常是 1D 的
            dtype = DataType.F32
            t = Tensor(shape=shape, dtype=dtype, device=self.device)
            data = np.random.randn(*shape).astype(np.float32) * 0.01
            t.load(data.ctypes.data_as(ctypes.c_void_p))
            self.weights[name] = t

        # Attention weights
        # input_layernorm
        add_weight(f"model.layers.{self.layer_idx}.input_layernorm.weight", (self.hidden_size,))
        
        # Q projection: (num_heads * head_dim, hidden_size)
        add_weight(f"model.layers.{self.layer_idx}.self_attn.q_proj.weight", (self.num_heads * self.head_dim, self.hidden_size))
        add_weight(f"model.layers.{self.layer_idx}.self_attn.q_proj.bias", (self.num_heads * self.head_dim,))
        
        # K projection: (num_kv_heads * head_dim, hidden_size)
        add_weight(f"model.layers.{self.layer_idx}.self_attn.k_proj.weight", (self.num_kv_heads * self.head_dim, self.hidden_size))
        add_weight(f"model.layers.{self.layer_idx}.self_attn.k_proj.bias", (self.num_kv_heads * self.head_dim,))
        
        # V projection: (num_kv_heads * head_dim, hidden_size)
        add_weight(f"model.layers.{self.layer_idx}.self_attn.v_proj.weight", (self.num_kv_heads * self.head_dim, self.hidden_size))
        add_weight(f"model.layers.{self.layer_idx}.self_attn.v_proj.bias", (self.num_kv_heads * self.head_dim,))
        
        # O projection: (hidden_size, num_heads * head_dim)
        add_weight(f"model.layers.{self.layer_idx}.self_attn.o_proj.weight", (self.hidden_size, self.num_heads * self.head_dim))
        # O projection usually has no bias in Qwen2
        
        # MLP weights
        # post_attention_layernorm
        add_weight(f"model.layers.{self.layer_idx}.post_attention_layernorm.weight", (self.hidden_size,))
        
        # MLP: Gate and Up are (intermediate_size, hidden_size), Down is (hidden_size, intermediate_size)
        add_weight(f"model.layers.{self.layer_idx}.mlp.gate_proj.weight", (self.intermediate_size, self.hidden_size))
        add_weight(f"model.layers.{self.layer_idx}.mlp.up_proj.weight", (self.intermediate_size, self.hidden_size))
        add_weight(f"model.layers.{self.layer_idx}.mlp.down_proj.weight", (self.hidden_size, self.intermediate_size))

        self.block = TransformerBlock(
            layer_idx=self.layer_idx,
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_heads,
            num_key_value_heads=self.num_kv_heads,
            intermediate_size=self.intermediate_size,
            rope_theta=self.rope_theta,
            rms_norm_eps=self.rms_norm_eps,
            weights=self.weights
        )

    def test_forward_prefill(self):
        """测试 Prefill 阶段（不带 KV Cache，输入多个 Token）"""
        batch_size = 1
        seq_len = 4
        
        x = Tensor(shape=(batch_size, seq_len, self.hidden_size), dtype=DataType.F32, device=self.device)
        x_data = np.random.randn(batch_size, seq_len, self.hidden_size).astype(np.float32)
        x.load(x_data.ctypes.data_as(ctypes.c_void_p))
        
        pos_ids = Tensor(shape=(batch_size, seq_len), dtype=DataType.I64, device=self.device)
        pos_data = np.arange(seq_len, dtype=np.int64).reshape(batch_size, seq_len)
        pos_ids.load(pos_data.ctypes.data_as(ctypes.c_void_p))
        
        print("\n[Test] Running TransformerBlock forward (prefill)...")
        output = self.block.forward(x, pos_ids)
        
        self.assertEqual(output.shape(), (batch_size, seq_len, self.hidden_size))
        print("[Test] Prefill forward successful.")

    def test_forward_decoding_with_cache(self):
        """测试 Decoding 阶段（带 KV Cache，输入单个 Token）"""
        batch_size = 1
        max_seq_len = 32
        
        # 初始化 KV Cache
        cache = KVCache(
            dtype=DataType.F32,
            device=self.device,
            device_id=0,
            num_layers=1,
            batch_size=batch_size,
            num_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            max_seq_len=max_seq_len
        )
        
        # 1. 模拟第一步 (seq_len=1)
        seq_len = 1
        x = Tensor(shape=(batch_size, seq_len, self.hidden_size), dtype=DataType.F32, device=self.device)
        x.load(np.random.randn(batch_size, seq_len, self.hidden_size).astype(np.float32).ctypes.data_as(ctypes.c_void_p))
        
        pos_ids = Tensor(shape=(batch_size, seq_len), dtype=DataType.I64, device=self.device)
        pos_ids.load(np.array([[0]], dtype=np.int64).ctypes.data_as(ctypes.c_void_p))
        
        print("\n[Test] Running TransformerBlock forward (decoding step 1)...")
        output1 = self.block.forward(x, pos_ids, kv_cache=cache)
        self.assertEqual(cache.get_size(self.layer_idx), 1)
        
        # 2. 模拟第二步 (seq_len=1, 但是在位置 1)
        pos_ids.load(np.array([[1]], dtype=np.int64).ctypes.data_as(ctypes.c_void_p))
        
        print("[Test] Running TransformerBlock forward (decoding step 2)...")
        output2 = self.block.forward(x, pos_ids, kv_cache=cache)
        self.assertEqual(cache.get_size(self.layer_idx), 2)
        
        self.assertEqual(output2.shape(), (batch_size, seq_len, self.hidden_size))
        print("[Test] Decoding forward with cache successful.")

if __name__ == '__main__':
    unittest.main()

