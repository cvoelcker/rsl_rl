"""HL-Gauss transform for distributional value prediction.

This module implements the Histogram Loss with Gaussian smoothing (HL-Gauss),
which represents scalar values as categorical distributions over a discretized range.
See: https://arxiv.org/abs/2403.03950 for details.
"""

import torch
import torch.nn as nn
from torch import Tensor


class HLGaussTransform:
    """Utility class for HL-Gauss target embedding and value reconstruction.

    Converts scalar values to soft categorical targets using Gaussian smoothing,
    and reconstructs scalar values from categorical distributions.

    Args:
        min_value: Minimum value of the support range.
        max_value: Maximum value of the support range.
        num_bins: Number of bins in the discretization.
        sigma: Standard deviation for Gaussian smoothing (default: 0.75).
    """

    def __init__(
        self,
        min_value: float,
        max_value: float,
        num_bins: int,
        sigma: float = 0.75,
    ):
        self.min_value = min_value
        self.max_value = max_value
        self.num_bins = num_bins
        self.sigma = sigma

        # Compute bin centers (support)
        self.support = torch.linspace(min_value, max_value, num_bins)
        self.bin_width = (max_value - min_value) / (num_bins - 1)

    def to(self, device: torch.device) -> "HLGaussTransform":
        """Move support tensor to specified device."""
        self.support = self.support.to(device)
        return self

    def embed_targets(self, targets: Tensor) -> Tensor:
        """Convert scalar targets to soft categorical distributions.

        Uses Gaussian smoothing to create soft targets, which provides better
        gradients than hard one-hot encodings.

        Args:
            targets: Scalar target values of shape (...,).

        Returns:
            Soft categorical distributions of shape (..., num_bins).
        """
        # Clamp targets to valid range
        targets = targets.clamp(self.min_value, self.max_value)

        # Compute distances to each bin center
        # targets: (...,) -> (..., 1), support: (num_bins,)
        support = self.support.to(targets.device)
        distances = targets.unsqueeze(-1) - support

        # Apply Gaussian kernel
        log_probs = -0.5 * (distances / self.sigma) ** 2

        # Normalize to get valid probability distribution
        probs = torch.softmax(log_probs, dim=-1)

        return probs

    def decode(self, logits: Tensor) -> Tensor:
        """Decode categorical logits to scalar values.

        Computes the expected value under the categorical distribution.

        Args:
            logits: Categorical logits of shape (..., num_bins).

        Returns:
            Scalar values of shape (...,).
        """
        support = self.support.to(logits.device)
        probs = torch.softmax(logits, dim=-1)
        return (probs * support).sum(dim=-1)


class HLGaussLayer(nn.Module):
    """Neural network layer for HL-Gauss distributional output.

    A linear layer that outputs categorical logits over a discretized value range.
    Can return both raw logits (for loss computation) and decoded scalar values.

    Args:
        in_features: Size of input features.
        min_value: Minimum value of the support range.
        max_value: Maximum value of the support range.
        num_bins: Number of bins in the discretization.
        sigma: Standard deviation for Gaussian smoothing (default: 0.75).
    """

    def __init__(
        self,
        in_features: int,
        min_value: float,
        max_value: float,
        num_bins: int,
        sigma: float = 0.75,
    ):
        super().__init__()

        self.transform = HLGaussTransform(min_value, max_value, num_bins, sigma)
        self.linear = nn.Linear(in_features, num_bins)

    @property
    def num_bins(self) -> int:
        return self.transform.num_bins

    @property
    def support(self) -> Tensor:
        return self.transform.support

    def forward(self, x: Tensor, return_logits: bool = False) -> Tensor | tuple[Tensor, Tensor]:
        """Forward pass through the HL-Gauss layer.

        Args:
            x: Input features of shape (..., in_features).
            return_logits: If True, return both decoded values and logits.

        Returns:
            If return_logits is False: Decoded scalar values of shape (...,).
            If return_logits is True: Tuple of (values, logits) where
                values has shape (...,) and logits has shape (..., num_bins).
        """
        logits = self.linear(x)
        values = self.transform.decode(logits)

        if return_logits:
            return values, torch.log_softmax(logits, dim=-1)
        return values

    def embed_targets(self, targets: Tensor) -> Tensor:
        """Convert scalar targets to soft categorical distributions.

        Convenience method that delegates to the internal transform.

        Args:
            targets: Scalar target values of shape (...,).

        Returns:
            Soft categorical distributions of shape (..., num_bins).
        """
        return self.transform.embed_targets(targets)

    def compute_loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Compute cross-entropy loss between logits and soft targets.

        Args:
            logits: Categorical logits of shape (..., num_bins).
            targets: Scalar target values of shape (...,).

        Returns:
            Cross-entropy loss (scalar).
        """
        soft_targets = self.embed_targets(targets)
        log_probs = torch.log_softmax(logits, dim=-1)
        return -(soft_targets * log_probs).sum(dim=-1).mean()


def embed_targets(
    targets: Tensor,
    min_value: float,
    max_value: float,
    num_bins: int,
    sigma: float = 0.75,
) -> Tensor:
    """Functional interface for embedding scalar targets as soft categorical distributions.

    Args:
        targets: Scalar target values of shape (...,).
        min_value: Minimum value of the support range.
        max_value: Maximum value of the support range.
        num_bins: Number of bins in the discretization.
        sigma: Standard deviation for Gaussian smoothing (default: 0.75).

    Returns:
        Soft categorical distributions of shape (..., num_bins).
    """
    transform = HLGaussTransform(min_value, max_value, num_bins, sigma)
    return transform.embed_targets(targets)
