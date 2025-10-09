"""
MLP Networks for DeepONet - From Reference Implementation

This module implements the Multi-Layer Perceptron (MLP) networks used as
branch and trunk networks in DeepONet. The implementation is preserved exactly
from the reference code with only documentation additions.

Reference: DeepONet_Pytorch_Demo-main/DeepONet.ipynb cell 3
"""

import torch
import torch.nn as nn
from typing import Tuple


class MLP(nn.Module):
    """
    Multi-layer perceptron with tanh activations.

    Used as both branch and trunk networks in DeepONet. The tanh activation
    is chosen for its proven effectiveness in operator learning tasks.

    Args:
        in_features: Input dimension
        out_features: Output dimension
        hidden_features: Hidden layer dimension
        num_hidden_layers: Number of hidden layers
    """

    def __init__(self, in_features: int, out_features: int,
                 hidden_features: int, num_hidden_layers: int) -> None:
        super().__init__()

        self.linear_in = nn.Linear(in_features, hidden_features)
        self.linear_out = nn.Linear(hidden_features, out_features)

        self.activation = torch.tanh
        self.layers = nn.ModuleList([self.linear_in] +
            [nn.Linear(hidden_features, hidden_features) for _ in range(num_hidden_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through MLP.

        Args:
            x: Input tensor

        Returns:
            output: Transformed tensor
        """
        for layer in self.layers:
            x = self.activation(layer(x))
        return self.linear_out(x)


class DeepONet(nn.Module):
    """
    DeepONet: Deep Operator Network using branch-trunk architecture.

    The DeepONet learns operator mappings by combining outputs from:
    - Branch network: Processes discretized input function
    - Trunk network: Processes query coordinates

    The outputs are combined via element-wise multiplication followed by
    a final linear layer to produce the operator output.

    Mathematical formulation:
        G(u)(y) = sum_i b_i(u) * t_i(y)

    where b_i are branch network outputs and t_i are trunk network outputs.

    Args:
        latent_features: Dimension of the latent space (hidden dimension)
        out_features: Output dimension
        branch: MLP network for branch
        trunk: MLP network for trunk

    Input shapes:
        y: Trunk input [B, N_query, coord_dim]
        u: Branch input [B, 1, sensor_dim]

    Output shape:
        [B, N_query, out_features]
    """

    def __init__(self, latent_features: int, out_features: int,
                 branch: nn.Module, trunk: nn.Module) -> None:
        super().__init__()
        self.branch = branch
        self.trunk = trunk
        self.fc = nn.Linear(latent_features, out_features, bias=False)

    def forward(self, y: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through DeepONet.

        Args:
            y: Trunk input [B, N_query, coord_dim] - query coordinates
            u: Branch input [B, 1, sensor_dim] - discretized function

        Returns:
            output: Operator output [B, N_query, out_features]
        """
        # Branch and trunk outputs are element-wise multiplied
        # This is the key operation that combines function and coordinate information
        return self.fc(self.trunk(y) * self.branch(u))
