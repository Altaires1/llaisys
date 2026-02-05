"""Qwen2 model configuration"""
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class Qwen2Config:
    """Qwen2 model configuration"""
    
    # Model architecture
    hidden_size: int = 1536
    num_hidden_layers: int = 28
    num_attention_heads: int = 12
    num_key_value_heads: int = 2
    intermediate_size: int = 8960
    hidden_act: str = "silu"
    
    # Model parameters
    vocab_size: int = 151936
    max_position_embeddings: int = 131072
    rope_theta: float = 10000.0
    
    # Normalization
    rms_norm_eps: float = 1e-6
    
    # Attention
    attention_dropout: float = 0.0
    
    # Tokens
    bos_token_id: int = 151643
    eos_token_id: int = 151643
    
    # Cache
    use_cache: bool = True
    
    # Sliding window
    sliding_window: Optional[int] = 4096
    max_window_layers: Optional[int] = 21
    use_sliding_window: bool = False
    
    # Other
    tie_word_embeddings: bool = False
    torch_dtype: str = "bfloat16"
    
    @classmethod
    def from_json_file(cls, config_path: str) -> "Qwen2Config":
        """Load configuration from JSON file
        
        Args:
            config_path: Path to the config.json file
            
        Returns:
            Qwen2Config instance
        """
        config_path = Path(config_path)
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        
        # Filter out keys not in dataclass
        filtered_dict = {
            k: v for k, v in config_dict.items() 
            if k in cls.__dataclass_fields__
        }
        
        return cls(**filtered_dict)
    
    @property
    def head_dim(self) -> int:
        """Calculate attention head dimension"""
        return self.hidden_size // self.num_attention_heads
    
    @property
    def num_key_value_groups(self) -> int:
        """Calculate number of key-value groups (for GQA)"""
        return self.num_attention_heads // self.num_key_value_heads
