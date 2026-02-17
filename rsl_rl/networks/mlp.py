# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from functools import reduce

from rsl_rl.networks.sem import SimplicalEmbeddingModule
from rsl_rl.utils import resolve_nn_activation


class NormedActivation(nn.Module):
    """A layer that applies RMSNorm followed by an activation function."""

    def __init__(self, dim: int, activation: str = "elu") -> None:
        """Initialize the NormedActivationLayer.

        Args:
            dim: Dimension of the input tensor.
            activation: Activation function.
        """
        super().__init__()
        self.norm = nn.RMSNorm(dim)
        self.activation = resolve_nn_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the NormedActivationLayer.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after applying RMSNorm and activation.
        """
        x = self.activation(x)
        x = self.norm(x)
        return x
    

class ActivationLayer(nn.Module):
    """A layer that applies an activation function."""

    def __init__(self, activation: str = "elu") -> None:
        """Initialize the ActivationLayer.

        Args:
            activation: Activation function.
        """
        super().__init__()
        self.activation = resolve_nn_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the ActivationLayer.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after applying activation.
        """
        x = self.activation(x)
        return x


class MLP(nn.Sequential):
    """Multi-layer perceptron.

    The MLP network is a sequence of linear layers and activation functions. The last layer is a linear layer that
    outputs the desired dimension unless the last activation function is specified.

    It provides additional conveniences:
    - If the hidden dimensions have a value of ``-1``, the dimension is inferred from the input dimension.
    - If the output dimension is a tuple, the output is reshaped to the desired shape.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int | tuple[int] | list[int],
        hidden_dims: tuple[int] | list[int],
        activation: str = "elu",
        last_activation: str | None = None,
        use_layer_norm: bool = False,
        add_sem: bool = False,
    ) -> None:
        """Initialize the MLP.

        Args:
            input_dim: Dimension of the input.
            output_dim: Dimension of the output.
            hidden_dims: Dimensions of the hidden layers. A value of ``-1`` indicates that the dimension should be
                inferred from the input dimension.
            activation: Activation function.
            last_activation: Activation function of the last layer. None results in a linear last layer.
        """
        super().__init__()

        if add_sem and use_layer_norm:
            raise ValueError("Cannot use both SEM and RMSNorm in the MLP.")

        # Resolve activation functions
        activation_mod = resolve_nn_activation(activation)
        last_activation_mod = resolve_nn_activation(last_activation) if last_activation is not None else None
        # Resolve number of hidden dims if they are -1
        hidden_dims_processed = [input_dim if dim == -1 else dim for dim in hidden_dims]

        # Create layers sequentially
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dims_processed[0]))
        if use_layer_norm:
            layers.append(nn.RMSNorm(hidden_dims_processed[0]))
        layers.append(activation_mod())

        for layer_index in range(len(hidden_dims_processed) - 1):
            layers.append(nn.Linear(hidden_dims_processed[layer_index], hidden_dims_processed[layer_index + 1]))
            if use_layer_norm:
                layers.append(nn.RMSNorm(hidden_dims_processed[layer_index + 1]))
            layers.append(activation_mod())

        if add_sem:
            layers.append(SimplicalEmbeddingModule(embed_dim=hidden_dims_processed[-1], chunk_size=16))

        # Add last layer
        total_out_dim = output_dim if isinstance(output_dim, int) else reduce(lambda x, y: x * y, output_dim)
        if isinstance(output_dim, int):
            layers.append(nn.Linear(hidden_dims_processed[-1], total_out_dim))
        else:
            last_layer = nn.Linear(hidden_dims_processed[-1], total_out_dim)
            # set small random orthogonal initialization
            nn.init.orthogonal_(last_layer.weight, gain=0.01)
            nn.init.zeros_(last_layer.bias)

            layers.append(last_layer)
            layers.append(nn.Unflatten(dim=-1, unflattened_size=output_dim))

        # Add last activation function if specified
        if last_activation_mod is not None:
            layers.append(last_activation_mod())

        # Register the layers
        for idx, layer in enumerate(layers):
            self.add_module(f"{idx}", layer)

    # def init_weights(self, scales: float | tuple[float]) -> None:
    #     """Initialize the weights of the MLP.

    #     Args:
    #         scales: Scale factor for the weights.
    #     """
    #     for idx, module in enumerate(self):
    #         if isinstance(module, nn.Linear):
    #             nn.init.orthogonal_(module.weight, gain=get_param(scales, idx))
    #             nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the MLP."""
        for layer in self:
            x = layer(x)
        return x
