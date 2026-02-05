"""Embedding layer for Qwen2 model"""
from ..tensor import Tensor
from ..ops import Ops
from ..libllaisys import DeviceType, DataType


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
        # Get original input shape
        input_shape = input_ids.shape()

        # Calculate total number of elements
        total_elements = 1
        for dim in input_shape:
            total_elements *= dim

        # Flatten input to 1D for C++ operation
        input_ids_flat = input_ids.view(total_elements)

        # Create flattened output tensor for C++ operation
        output_flat = Tensor(
            shape=(total_elements, self.embedding_dim),
            dtype=self.weight.dtype(),
            device=self.weight.device_type(),
            device_id=self.weight.device_id()
        )

        # Call C++ embedding operation with 1D index and 2D output
        Ops.embedding(output_flat, input_ids_flat, self.weight)

        # Reshape output to final multi-dimensional shape
        output_final_shape = input_shape + (self.embedding_dim,)
        output = output_flat.view(*output_final_shape)

        return output
