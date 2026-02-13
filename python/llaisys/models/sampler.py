import numpy as np
from typing import Optional
from ..tensor import Tensor
from ..libllaisys import DataType, MemcpyKind, DeviceType
from .utils import argmax_nd

class Sampler:
    """Sampler for token generation"""
    
    def __init__(self, model):
        """Initialize sampler
        
        Args:
            model: The model instance (to access runtime and device)
        """
        self.model = model
        self.runtime = model.runtime
        self.device = model.device
        
    def sample(
        self,
        logits: Tensor,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        do_sample: bool = True
    ) -> int:
        """Sample a token from logits
        
        Args:
            logits: Logits tensor of shape (batch=1, seq=1, vocab_size)
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            do_sample: Whether to use sampling or greedy decoding
            
        Returns:
            The sampled token ID
        """
        if not do_sample:
            # Greedy decoding
            max_idx, _ = argmax_nd(logits)
            max_idx_np = self.model._copy_tensor_to_cpu(max_idx)
            return int(max_idx_np.item())
            
        # Sampling logic
        # 1. Copy logits to CPU
        logits_np = self.model._copy_tensor_to_cpu(logits)
        # Reshape to (vocab_size,)
        logits_np = logits_np.flatten().astype(np.float32)
        
        # 2. Apply temperature
        if temperature > 0 and temperature != 1.0:
            logits_np = logits_np / temperature
            
        # 3. Top-k filtering
        if top_k > 0:
            indices_to_remove = logits_np < np.partition(logits_np, -top_k)[-top_k]
            logits_np[indices_to_remove] = -float('Inf')
            
        # 4. Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_indices = np.argsort(logits_np)[::-1]
            sorted_logits = logits_np[sorted_indices]
            
            # Use softmax for probabilities
            exp_logits = np.exp(sorted_logits - np.max(sorted_logits))
            probs = exp_logits / np.sum(exp_logits)
            
            cumulative_probs = np.cumsum(probs)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].copy()
            sorted_indices_to_remove[0] = False
            
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits_np[indices_to_remove] = -float('Inf')
            
        # 5. Softmax and sample
        exp_logits = np.exp(logits_np - np.max(logits_np))
        probs = exp_logits / np.sum(exp_logits)
        
        token_id = np.random.choice(len(probs), p=probs)
        return int(token_id)
