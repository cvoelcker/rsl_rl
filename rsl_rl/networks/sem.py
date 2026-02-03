import torch
import torch.nn as nn


class SimplicalEmbeddingModule(nn.Module):
    """Module for simplicial embedding of inputs.

    This module takes an input tensor and maps it to a higher-dimensional space using a simplicial embedding.
    The embedding is performed by creating a set of basis vectors that form a simplex in the target space.
    """

    def __init__(self, embed_dim: int, chunk_size: int) -> None:
        """Initialize the SimplicalEmbeddingModule.

        Args:
            input_dim: Dimension of the input tensor.
            embed_dim: Dimension of the embedding space.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.chunk_size = chunk_size
        assert embed_dim % chunk_size == 0, "embed_dim must be divisible by chunk_size"
        self.num_chunks = embed_dim // chunk_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the SimplicalEmbeddingModule.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Embedded tensor of shape (batch_size, embed_dim).
        """
        # Map vector in n chunks on simplex by softmax
        batch_size = x.shape[:-1]
        x = x.view(*batch_size, self.num_chunks, self.chunk_size)
        x = torch.softmax(x, dim=-1)
        x = x.view(*batch_size, self.embed_dim)
        return x
