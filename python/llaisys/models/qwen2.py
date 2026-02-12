from typing import Sequence, Dict, Optional
from ctypes import c_void_p
import numpy as np
from pathlib import Path
import gc

from ..libllaisys import DeviceType, DataType, MemcpyKind
from ..tensor import Tensor
from ..ops import Ops
from ..runtime import RuntimeAPI
from ..kv_cache import KVCache

from .config import Qwen2Config, Qwen2GenerationConfig
from .weight_loader import WeightLoader
from .embedding import Embedding
from .attention import RMSNorm
from .transformer import TransformerBlock
from .utils import linear_nd


class Qwen2:
    """Qwen2 model for causal language modeling"""

    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        """Initialize Qwen2 model
        
        Args:
            model_path: Path to the model directory
            device: Device type (CPU or NVIDIA)
        """
        try:
            from tqdm import tqdm
            self.tqdm = tqdm
        except ImportError:
            # Fallback if tqdm not available
            self.tqdm = None
        
        model_path = Path(model_path)
        
        # Load configuration
        config_path = model_path / "config.json"
        self.config = Qwen2Config.from_json_file(str(config_path))
        
        # Load generation configuration
        gen_config_path = model_path / "generation_config.json"
        self.generation_config = Qwen2GenerationConfig.from_json_file(str(gen_config_path))
        
        print(f"[Qwen2] Config loaded: {self.config.hidden_size}D, "
              f"{self.config.num_hidden_layers}L, vocab={self.config.vocab_size}")
        
        self.device = device
        self.runtime = RuntimeAPI(device)
        
        # Load weights
        print(f"[Qwen2] Loading model weights...")
        self.weight_loader = WeightLoader(str(model_path), device, device_id=0)
        self.weights = self.weight_loader.load_weights()
        print(f"[Qwen2] Loaded {len(self.weights)} weight tensors")
        
        # Build model components
        print("[Qwen2] Building model components...")
        self._build_model()
        
        # Clean up after model building
        gc.collect()
    
    def _build_model(self):
        """Build model components from weights"""
        # Token embedding layer
        embed_weight = self.weights["model.embed_tokens.weight"]
        self.embed_tokens = Embedding(embed_weight)
        gc.collect()
        
        # Transformer layers
        self.layers = []
        layer_range = range(self.config.num_hidden_layers)
        if self.tqdm:
            layer_range = self.tqdm(layer_range, desc="Building layers", unit="layer")
        
        for layer_idx in layer_range:
            try:
                layer = TransformerBlock(
                    layer_idx=layer_idx,
                    hidden_size=self.config.hidden_size,
                    num_attention_heads=self.config.num_attention_heads,
                    num_key_value_heads=self.config.num_key_value_heads,
                    intermediate_size=self.config.intermediate_size,
                    rope_theta=self.config.rope_theta,
                    rms_norm_eps=self.config.rms_norm_eps,
                    weights=self.weights,
                )
                self.layers.append(layer)
                
                # Periodic garbage collection
                if layer_idx % 5 == 4:
                    gc.collect()
                    
            except KeyError as e:
                print(f"[Qwen2] ERROR: Failed to build layer {layer_idx}: {e}")
                print(f"[Qwen2] Available weight keys sample: {list(self.weights.keys())[:5]}")
                raise
        
        # Final layer norm
        self.norm = RMSNorm(
            self.weights["model.norm.weight"],
            self.config.rms_norm_eps
        )
        
        # Language model head (lm_head)
        self.lm_head_weight = self.weights["lm_head.weight"]
        self.lm_head_bias = self.weights.get("lm_head.bias", None)
        
        print(f"[Qwen2] Model built: {len(self.layers)} layers")
        gc.collect()
    
    def forward(
        self,
        input_ids: Tensor,
        position_ids: Optional[Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tensor:
        """Forward pass of the model
        
        Args:
            input_ids: Input token IDs tensor of shape (batch, seq_len)
            position_ids: Position IDs tensor
            kv_cache: Optional KV cache for inference
            
        Returns:
            Logits tensor of shape (batch, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape()
        
        # Embed tokens
        x = self.embed_tokens.forward(input_ids)  # (batch, seq_len, hidden_size)
        
        # Create position IDs if not provided
        if position_ids is None:
            position_ids = self._create_position_ids(batch_size, seq_len)
        
        # Pass through transformer layers
        layer_range = self.layers
        if self.tqdm:
            layer_range = self.tqdm(self.layers, desc="Forward", unit="layer", leave=False)
        
        for i, layer in enumerate(layer_range):
            x = layer.forward(x, position_ids, kv_cache)
        
        # Final layer norm
        x = self.norm.forward(x)  # (batch, seq_len, hidden_size)
        
        # Language model head
        logits = Tensor(
            shape=(batch_size, seq_len, self.config.vocab_size),
            dtype=x.dtype(),
            device=x.device_type(),
            device_id=x.device_id()
        )
        linear_nd(logits, x, self.lm_head_weight, self.lm_head_bias)
        
        return logits
    
    def _create_position_ids(self, batch_size: int, seq_len: int) -> Tensor:
        """Create position IDs tensor
        
        Args:
            batch_size: Batch size
            seq_len: Sequence length
            
        Returns:
            Position IDs tensor
        """
        position_ids = Tensor(
            shape=(batch_size, seq_len),
            dtype=DataType.I64,
            device=self.device,
            device_id=0
        )
        
        # Create positions: [0, 1, 2, ..., seq_len-1] repeated for batch
        positions = np.arange(seq_len, dtype=np.int64)
        positions = np.tile(positions, (batch_size, 1))
        
        # Load data to tensor using the load() method
        # This handles the memory transfer appropriately based on device type
        position_ids.load(positions.ctypes.data_as(c_void_p))
        
        return position_ids
    
    def _copy_tensor_to_cpu(self, tensor: Tensor) -> np.ndarray:
        """Copy a tensor from device to CPU
        
        Args:
            tensor: Tensor to copy
            
        Returns:
            Numpy array on CPU
        """
        shape = tensor.shape()
        
        # Create numpy array - handle bfloat16 as uint16
        if tensor.dtype() == DataType.I64:
            arr = np.zeros(shape, dtype=np.int64)
        elif tensor.dtype() == DataType.I32:
            arr = np.zeros(shape, dtype=np.int32)
        elif tensor.dtype() == DataType.F32:
            arr = np.zeros(shape, dtype=np.float32)
        elif tensor.dtype() == DataType.F16:
            arr = np.zeros(shape, dtype=np.float16)
        elif tensor.dtype() == DataType.BF16:
            # bfloat16 is stored as uint16 (16-bit format)
            arr = np.zeros(shape, dtype=np.uint16)
        else:
            arr = np.zeros(shape, dtype=np.float32)
        
        # Copy from device
        arr_ptr = arr.ctypes.data_as(c_void_p)
        self.runtime.memcpy_sync(
            arr_ptr,
            tensor.data_ptr(),
            arr.nbytes,
            MemcpyKind.D2H
        )
        
        return arr
    
    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
        max_cache_seq_len: int = None,
    ):
        """Generate text given input token IDs
        
        Args:
            inputs: Input token IDs
            max_new_tokens: Maximum number of new tokens to generate
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            temperature: Sampling temperature
            max_cache_seq_len: Maximum sequence length for KV cache (for memory optimization).
                             If None, will auto-calculate based on input length.
            
        Returns:
            Generated token IDs as a list
        """
        if max_new_tokens is None:
            max_new_tokens = 128
        
        print(f"[Qwen2] Starting generation with {len(inputs)} input tokens, "
              f"max_new_tokens={max_new_tokens}")
        
        # Convert input to tensor
        input_ids_np = np.array([inputs], dtype=np.int64)  # (1, seq_len)
        input_ids = Tensor(
            shape=tuple(input_ids_np.shape),
            dtype=DataType.I64,
            device=self.device,
            device_id=0
        )
        
        # Load input IDs to device using Tensor API
        input_ids.load(input_ids_np.ctypes.data_as(c_void_p))
        
        # Initialize KV cache with dynamic sizing based on actual input length
        # For small inputs (like in testing), use minimal cache to save memory
        if max_cache_seq_len is None:
            actual_seq_len = len(inputs)
            max_new_tokens_total = actual_seq_len + max_new_tokens
            
            # Use a reasonable upper bound for KV cache, but cap it for memory efficiency
            # For 8GB systems: use 512 as max practical seq_len in KV cache
            # For larger systems: can use higher values like 2048 or 4096
            max_cache_seq_len = min(max_new_tokens_total, 512)
        
        print(f"[Qwen2] Input seq_len: {len(inputs)}, total seq_len: {len(inputs) + max_new_tokens}, "
              f"KV cache max_seq_len: {max_cache_seq_len}")
        
        kv_cache = KVCache(
            dtype=DataType.BF16,
            device=self.device,
            device_id=0,
            num_layers=self.config.num_hidden_layers,
            batch_size=1,
            num_heads=self.config.num_key_value_heads,
            head_dim=self.config.head_dim,
            max_seq_len=max_cache_seq_len
        )
        
        # Generate tokens
        generated = list(inputs)
        
        step_range = range(max_new_tokens)
        if self.tqdm:
            step_range = self.tqdm(step_range, desc="Generating", unit="token")
        
        for step in step_range:
            
            # Create input for this step
            if step == 0:
                # Use full input sequence for first step
                current_input = input_ids
            else:
                # For efficiency, only use last token
                last_token = np.array([[generated[-1]]], dtype=np.int64)
                current_input = Tensor(
                    shape=(1, 1),
                    dtype=DataType.I64,
                    device=self.device,
                    device_id=0
                )
                # Load last token using Tensor API
                current_input.load(last_token.ctypes.data_as(c_void_p))
            
            # Forward pass
            logits = self.forward(current_input, kv_cache=kv_cache)
            
            # Get last token logits (batch=1, seq_len=?, vocab_size)
            batch, seq, vocab = logits.shape()
            
            # Extract last position logits
            logits_last = logits.slice(1, seq - 1, seq)  # (1, 1, vocab_size)
            
            # For now, use greedy decoding (argmax)
            max_idx = Tensor(
                shape=(1, 1, 1),
                dtype=DataType.I64,
                device=self.device,
                device_id=0
            )
            max_val = Tensor(
                shape=(1, 1, 1),
                dtype=logits.dtype(),
                device=self.device,
                device_id=0
            )
            
            # Find argmax along vocabulary dimension
            Ops.argmax(max_idx, max_val, logits_last)
            
            # Copy result back to CPU
            max_idx_np = self._copy_tensor_to_cpu(max_idx)
            next_token_id = int(max_idx_np.item())
            
            generated.append(next_token_id)
            
            # Check for EOS token
            if next_token_id == self.config.eos_token_id:
                print(f"[Qwen2] Reached EOS token at step {step + 1}")
                break
        
        print(f"[Qwen2] Generation completed. Generated {len(generated) - len(inputs)} new tokens")
        return generated
