"""Embedding layer for Qwen2 model"""
from ..tensor import Tensor
from .utils import embedding_nd


class Embedding:
    """Token embedding layer"""
    
    def __init__(self, weight: Tensor):
        """Initialize embedding layer
        
        Args:
            weight: Embedding weight tensor of shape (vocab_size, hidden_size)
        """
        self.weight = weight
        assert weight.ndim() == 2, "Embedding weight must be 2D"
        self.vocab_size = weight.shape()[0]
        self.embedding_dim = weight.shape()[1]
    
    def forward(self, input_ids: Tensor) -> Tensor:
        """Embed input token IDs

        Args:
            input_ids: Input token IDs tensor of any shape

        Returns:
            Embedded tokens tensor with embedding_dim appended to input shape
        """
        return embedding_nd(input_ids, self.weight)
