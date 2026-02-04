from typing import Sequence, Dict, List
import ctypes
from ..libllaisys import LIB_LLAISYS
from ..libllaisys import DeviceType, DataType
from ..tensor import Tensor
from ..ops import Ops
from ..kv_cache import KVCache
from .layers import TransformerLayer, EmbeddingLayer, LayerNorm
from .utils import load_safetensors

from pathlib import Path

# Create ops instance
ops = Ops()


class Qwen2:
    """
    Qwen2 model implementation with causal language modeling capability.
    """

    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        """
        Initialize the Qwen2 model.
        
        Args:
            model_path: Path to the model weights directory
            device: Device to run the model on
        """
        self.device = device
        self.model_path = Path(model_path)
        
        # Load all weights
        self.weights = self._load_weights()
        
        # Extract config from weights
        self.config = self._extract_config()
        
        # Initialize model components
        self._initialize_components()
    
    def _load_weights(self) -> Dict[str, Tensor]:
        """
        Load model weights from the first safetensors file found.
        
        Returns:
            Dictionary mapping weight names to llaisys Tensor objects
        """
        device_str = "cuda" if self.device == DeviceType.NVIDIA else "cpu"
        
        # Get the first safetensors file
        files = sorted(self.model_path.glob("*.safetensors"))
        if not files:
            raise FileNotFoundError(f"No safetensors files found in {self.model_path}")
        
        file = files[0]
        print(f"Loading weights from {file}")
        weights = load_safetensors(str(file), device=device_str)
        
        return weights
    
    def _extract_config(self) -> Dict:
        """
        Extract model configuration from config.json and generation_config.json files in the model path.
        If files don't exist, use default values.
        
        Returns:
            Model configuration dictionary
        """
        import json
        
        # Default configuration as fallback
        default_config = {}
        
        # Load config.json if exists
        config_file = self.model_path / "config.json"
        if config_file.exists():
            with open(config_file, "r", encoding="utf-8") as f:
                default_config.update(json.load(f))
        else:
            print(f"Warning: config.json not found at {config_file}, using default values")
        
        # Load generation_config.json if exists
        generation_config_file = self.model_path / "generation_config.json"
        if generation_config_file.exists():
            with open(generation_config_file, "r", encoding="utf-8") as f:
                generation_config = json.load(f)
                default_config.update(generation_config)
        else:
            print(f"Warning: generation_config.json not found at {generation_config_file}, using default values")
        
        return default_config
    
    def _initialize_components(self):
        """
        Initialize all model components.
        """
        # Initialize embedding layer
        self.embedding_layer = EmbeddingLayer(self.config, self.weights, self.device)
        
        # Initialize final layer norm
        self.final_norm = LayerNorm(self.config, self.weights, self.device)
        
        # Initialize transformer layers
        self.layers = []
        for layer_idx in range(self.config["num_hidden_layers"]):
            # Extract layer-specific weights
            layer_weights = {
                "input_norm": self.weights[f"layers.{layer_idx}.input_norm.weight"],
                "post_attn_norm": self.weights[f"layers.{layer_idx}.post_attn_norm.weight"],
                "attn_q_proj": self.weights[f"layers.{layer_idx}.attn.q_proj.weight"],
                "attn_k_proj": self.weights[f"layers.{layer_idx}.attn.k_proj.weight"],
                "attn_v_proj": self.weights[f"layers.{layer_idx}.attn.v_proj.weight"],
                "attn_o_proj": self.weights[f"layers.{layer_idx}.attn.o_proj.weight"],
                "attn_q_bias": self.weights[f"layers.{layer_idx}.attn.q_proj.bias"],
                "attn_k_bias": self.weights[f"layers.{layer_idx}.attn.k_proj.bias"],
                "attn_v_bias": self.weights[f"layers.{layer_idx}.attn.v_proj.bias"],
                "mlp_gate_proj": self.weights[f"layers.{layer_idx}.mlp.gate_proj.weight"],
                "mlp_up_proj": self.weights[f"layers.{layer_idx}.mlp.up_proj.weight"],
                "mlp_down_proj": self.weights[f"layers.{layer_idx}.mlp.down_proj.weight"]
            }
            
            # Create transformer layer
            layer = TransformerLayer(self.config, layer_weights, self.device)
            self.layers.append(layer)
    
    def _embed_input(self, input_ids: List[int]) -> Tensor:
        """
        Embed input token IDs into hidden states.
        
        Args:
            input_ids: List of input token IDs
            
        Returns:
            Embedded hidden states (batch_size, seq_len, hidden_size)
        """
        seq_len = len(input_ids)
        
        # Create input tensor
        input_tensor = Tensor(
            shape=[1, seq_len],  # batch_size=1
            dtype=DataType.INT64,
            device=self.device,
            device_id=0
        )
        
        # Create a ctypes array directly from input_ids
        input_ids_array = (ctypes.c_int64 * seq_len)(*input_ids)
        input_ids_ptr = ctypes.cast(input_ids_array, ctypes.c_void_p)
        
        # Load data into tensor
        input_tensor.load(input_ids_ptr)
        
        # Use embedding layer
        hidden_states = self.embedding_layer.forward(input_tensor)
        
        return hidden_states
    
    def _forward(self, hidden_states: Tensor, kv_cache: KVCache, seq_len: int) -> Tensor:
        """
        Forward pass through the model.
        
        Args:
            hidden_states: Input hidden states
            kv_cache: KV cache for storing past key-value pairs
            seq_len: Current sequence length
            
        Returns:
            Final hidden states
        """
        # Pass through all transformer layers
        for layer_idx, layer in enumerate(self.layers):
            hidden_states = layer.forward(hidden_states, kv_cache, layer_idx, seq_len)
        
        # Final layer norm
        final_norm_output = self.final_norm.forward(hidden_states)
        
        return final_norm_output
    
    def _sample_token(self, logits: Tensor, top_k: int, top_p: float, temperature: float) -> int:
        """
        Sample a token from the logits using top-k and top-p sampling (Qwen2ForCausalLM implementation).
        
        Args:
            logits: Logits tensor (batch_size, seq_len, vocab_size)
            top_k: Number of highest probability tokens to consider
            top_p: Cumulative probability threshold for nucleus sampling
            temperature: Temperature for scaling logits
            
        Returns:
            Sampled token ID
        """
        import random
        import math
        
        # Get the last token's logits (batch_size=1, so index 0)
        # In a real implementation, we would extract the last token's logits directly
        # from the tensor without converting to list, but this is simplified for demonstration
        
        # Step 1: Get logits data as a list of scores
        logits_ptr = logits.data_ptr()
        logits_shape = logits.shape()
        vocab_size = logits_shape[-1]
        
        # Read logits data from memory
        # This is a simplified approach for demonstration
        logits_list = []
        if logits.dtype() == DataType.FLOAT32:
            # Read as float32
            float_ptr = ctypes.cast(logits_ptr, ctypes.POINTER(ctypes.c_float))
            for i in range(vocab_size):
                logits_list.append(float_ptr[i])
        elif logits.dtype() == DataType.BFLOAT16:
            # Read as uint16 (bfloat16)
            uint16_ptr = ctypes.cast(logits_ptr, ctypes.POINTER(ctypes.c_uint16))
            for i in range(vocab_size):
                # Convert bfloat16 to float32 for processing
                bf16_val = uint16_ptr[i]
                # Simplified bfloat16 to float32 conversion
                sign = (bf16_val >> 15) & 1
                exponent = (bf16_val >> 7) & 0xff
                mantissa = bf16_val & 0x7f
                
                if exponent == 0 and mantissa == 0:
                    float_val = 0.0
                elif exponent == 0xff:
                    float_val = float('inf') if mantissa == 0 else float('nan')
                else:
                    float_val = ((-1) ** sign) * (1 + mantissa / 128.0) * (2 ** (exponent - 127))
                
                logits_list.append(float_val)
        else:
            # For other dtypes, use a default approach
            raise ValueError(f"Unsupported logits dtype: {logits.dtype()}")
        
        # Step 2: Apply temperature scaling
        if temperature != 1.0:
            logits_list = [logit / temperature for logit in logits_list]
        
        # Step 3: Compute softmax probabilities
        # Subtract max for numerical stability
        max_logit = max(logits_list)
        exp_logits = [math.exp(logit - max_logit) for logit in logits_list]
        sum_exp = sum(exp_logits)
        probabilities = [exp / sum_exp for exp in exp_logits]
        
        # Step 4: Create token-probability pairs
        token_probs = list(enumerate(probabilities))
        
        # Step 5: Apply top-k filtering
        if top_k > 0:
            token_probs = sorted(token_probs, key=lambda x: x[1], reverse=True)[:top_k]
        
        # Step 6: Apply top-p filtering
        if top_p < 1.0:
            # Sort by probability descending
            token_probs = sorted(token_probs, key=lambda x: x[1], reverse=True)
            
            # Calculate cumulative probabilities
            cumulative_probs = []
            current_sum = 0.0
            for _, prob in token_probs:
                current_sum += prob
                cumulative_probs.append(current_sum)
            
            # Find cutoff point where cumulative probability exceeds top_p
            cutoff_idx = len(cumulative_probs)
            for i, cum_prob in enumerate(cumulative_probs):
                if cum_prob >= top_p:
                    cutoff_idx = i + 1
                    break
            
            # Filter tokens
            token_probs = token_probs[:cutoff_idx]
        
        # Step 7: Normalize probabilities
        total_prob = sum(prob for _, prob in token_probs)
        if total_prob > 0:
            token_probs = [(token, prob / total_prob) for token, prob in token_probs]
        else:
            # Fallback if all probabilities are zero
            token_probs = [(token, 1.0 / len(token_probs)) for token, _ in token_probs]
        
        # Step 8: Sample from the filtered distribution
        tokens, probs = zip(*token_probs)
        sampled_token = random.choices(tokens, weights=probs, k=1)[0]
        
        return int(sampled_token)
    
    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = 128,
        top_k: int = 50,
        top_p: float = 0.8,
        temperature: float = 1.0,
    ) -> List[int]:
        """
        Generate tokens from the model.
        
        Args:
            inputs: List of input token IDs
            max_new_tokens: Maximum number of tokens to generate
            top_k: Number of highest probability tokens to consider for sampling
            top_p: Cumulative probability threshold for nucleus sampling
            temperature: Temperature for scaling logits
            
        Returns:
            List of generated token IDs (including input tokens)
        """
        # Initialize KV cache
        kv_cache = KVCache(self.config["num_hidden_layers"], self.device)
        
        # Start with input tokens
        generated_tokens = list(inputs)
        
        # Embed input tokens
        hidden_states = self._embed_input(generated_tokens)
        
        # Forward pass on input tokens
        seq_len = len(generated_tokens)
        hidden_states = self._forward(hidden_states, kv_cache, seq_len)
        
        # Generate new tokens
        for _ in range(max_new_tokens):
            # Note: This model doesn't involve vocabulary processing, 
            # so we'll return hidden states directly without LM head projection
            logits = hidden_states[:, -1:, :]
            
            # Sample next token
            next_token = self._sample_token(logits, top_k, top_p, temperature)
            
            # Add to generated tokens
            generated_tokens.append(next_token)
            
            # Check if end of sequence
            if next_token == self.config["eos_token_id"]:
                break
            
            # Prepare for next iteration
            seq_len += 1
            
            # Embed new token
            next_token_embed = self._embed_input([next_token])
            
            # Forward pass with new token
            hidden_states = self._forward(next_token_embed, kv_cache, seq_len)
        
        return generated_tokens