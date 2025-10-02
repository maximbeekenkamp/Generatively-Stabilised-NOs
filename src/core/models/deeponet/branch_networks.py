"""
Branch Network Implementations for DeepONet

Branch networks encode input functions sampled at sensor locations.
Different variants handle different types of input functions and sampling strategies.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class BranchNetwork(nn.Module, ABC):
    """Abstract base class for DeepONet branch networks."""

    @abstractmethod
    def forward(self, u_sensors: torch.Tensor) -> torch.Tensor:
        """
        Encode input function sampled at sensors.

        Args:
            u_sensors: Function values at sensor locations [B, n_sensors]

        Returns:
            branch_output: Encoded function representation [B, latent_dim]
        """
        pass

    @abstractmethod
    def get_input_dim(self) -> int:
        """Get expected input dimension (number of sensors)."""
        pass

    @abstractmethod
    def get_output_dim(self) -> int:
        """Get output dimension (latent dimension)."""
        pass


class DenseBranchNetwork(BranchNetwork):
    """
    Dense (fully connected) branch network.

    Standard implementation using multilayer perceptron to encode
    function values sampled at sensor locations.
    """

    def __init__(self,
                 n_sensors: int,
                 latent_dim: int,
                 hidden_layers: List[int] = None,
                 activation: str = "gelu",
                 dropout: float = 0.1,
                 batch_norm: bool = True):
        super().__init__()

        self.n_sensors = n_sensors
        self.latent_dim = latent_dim
        self.activation_name = activation
        self.dropout_rate = dropout
        self.use_batch_norm = batch_norm

        # Default hidden layers
        if hidden_layers is None:
            hidden_layers = [128, 256]

        # Build layer dimensions
        layer_dims = [n_sensors] + hidden_layers + [latent_dim]

        # Activation function
        if activation.lower() == "gelu":
            self.activation = nn.GELU()
        elif activation.lower() == "relu":
            self.activation = nn.ReLU()
        elif activation.lower() == "silu":
            self.activation = nn.SiLU()
        elif activation.lower() == "tanh":
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Build layers
        layers = []
        for i in range(len(layer_dims) - 1):
            # Linear layer
            layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))

            # Skip activation and normalization for last layer
            if i < len(layer_dims) - 2:
                # Batch normalization
                if batch_norm:
                    layers.append(nn.BatchNorm1d(layer_dims[i + 1]))

                # Activation
                layers.append(self.activation)

                # Dropout
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

        self.network = nn.Sequential(*layers)

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, u_sensors: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through dense branch network.

        Args:
            u_sensors: Function values at sensors [B, n_sensors]

        Returns:
            branch_output: Encoded representation [B, latent_dim]
        """
        if u_sensors.shape[-1] != self.n_sensors:
            raise ValueError(f"Expected {self.n_sensors} sensors, got {u_sensors.shape[-1]}")

        return self.network(u_sensors)

    def get_input_dim(self) -> int:
        return self.n_sensors

    def get_output_dim(self) -> int:
        return self.latent_dim


class ConvolutionalBranchNetwork(BranchNetwork):
    """
    Convolutional branch network for spatially structured sensor data.

    Uses 1D convolutions to capture local patterns in function values
    before global aggregation.
    """

    def __init__(self,
                 n_sensors: int,
                 latent_dim: int,
                 conv_channels: List[int] = None,
                 kernel_sizes: List[int] = None,
                 hidden_layers: List[int] = None,
                 activation: str = "gelu",
                 dropout: float = 0.1,
                 batch_norm: bool = True):
        super().__init__()

        self.n_sensors = n_sensors
        self.latent_dim = latent_dim

        # Default convolutional parameters
        if conv_channels is None:
            conv_channels = [32, 64, 128]
        if kernel_sizes is None:
            kernel_sizes = [3, 3, 3]
        if hidden_layers is None:
            hidden_layers = [256]

        # Activation function
        if activation.lower() == "gelu":
            self.activation = nn.GELU()
        elif activation.lower() == "relu":
            self.activation = nn.ReLU()
        else:
            self.activation = nn.GELU()  # Default

        # Build convolutional layers
        conv_layers = []
        in_channels = 1  # Start with single channel (function values)

        for i, (out_channels, kernel_size) in enumerate(zip(conv_channels, kernel_sizes)):
            # 1D convolution
            conv_layers.append(nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2))

            if batch_norm:
                conv_layers.append(nn.BatchNorm1d(out_channels))

            conv_layers.append(self.activation)

            if dropout > 0:
                conv_layers.append(nn.Dropout(dropout))

            # Max pooling to reduce spatial dimension
            if i < len(conv_channels) - 1:  # Don't pool on last layer
                conv_layers.append(nn.MaxPool1d(2))

            in_channels = out_channels

        self.conv_layers = nn.Sequential(*conv_layers)

        # Calculate size after convolutions for dense layers
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, n_sensors)
            conv_output = self.conv_layers(dummy_input)
            conv_output_size = conv_output.numel()

        # Dense layers for final encoding
        dense_dims = [conv_output_size] + hidden_layers + [latent_dim]
        dense_layers = []

        for i in range(len(dense_dims) - 1):
            dense_layers.append(nn.Linear(dense_dims[i], dense_dims[i + 1]))

            if i < len(dense_dims) - 2:  # Skip activation on last layer
                if batch_norm:
                    dense_layers.append(nn.BatchNorm1d(dense_dims[i + 1]))
                dense_layers.append(self.activation)
                if dropout > 0:
                    dense_layers.append(nn.Dropout(dropout))

        self.dense_layers = nn.Sequential(*dense_layers)

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, u_sensors: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through convolutional branch network.

        Args:
            u_sensors: Function values at sensors [B, n_sensors]

        Returns:
            branch_output: Encoded representation [B, latent_dim]
        """
        B = u_sensors.shape[0]

        # Add channel dimension: [B, n_sensors] -> [B, 1, n_sensors]
        x = u_sensors.unsqueeze(1)

        # Convolutional feature extraction
        x = self.conv_layers(x)  # [B, channels, spatial]

        # Flatten for dense layers
        x = x.view(B, -1)

        # Final encoding
        x = self.dense_layers(x)

        return x

    def get_input_dim(self) -> int:
        return self.n_sensors

    def get_output_dim(self) -> int:
        return self.latent_dim


class FourierBranchNetwork(BranchNetwork):
    """
    Fourier-enhanced branch network for periodic functions.

    Incorporates Fourier features to better capture periodic patterns
    in input functions.
    """

    def __init__(self,
                 n_sensors: int,
                 latent_dim: int,
                 fourier_modes: int = 32,
                 hidden_layers: List[int] = None,
                 activation: str = "gelu",
                 dropout: float = 0.1,
                 batch_norm: bool = True):
        super().__init__()

        self.n_sensors = n_sensors
        self.latent_dim = latent_dim
        self.fourier_modes = fourier_modes

        # Default hidden layers
        if hidden_layers is None:
            hidden_layers = [256, 512]

        # Fourier feature dimension (cos + sin for each mode)
        fourier_dim = 2 * fourier_modes

        # Combined input dimension (original + Fourier features)
        input_dim = n_sensors + fourier_dim

        # Activation function
        if activation.lower() == "gelu":
            self.activation = nn.GELU()
        elif activation.lower() == "relu":
            self.activation = nn.ReLU()
        else:
            self.activation = nn.GELU()

        # Fourier frequency parameters (learnable)
        self.fourier_freqs = nn.Parameter(torch.randn(fourier_modes) * 2 * math.pi)

        # Build dense network
        layer_dims = [input_dim] + hidden_layers + [latent_dim]
        layers = []

        for i in range(len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))

            if i < len(layer_dims) - 2:
                if batch_norm:
                    layers.append(nn.BatchNorm1d(layer_dims[i + 1]))
                layers.append(self.activation)
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

        self.network = nn.Sequential(*layers)

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def compute_fourier_features(self, u_sensors: torch.Tensor) -> torch.Tensor:
        """
        Compute Fourier features from input function values.

        Args:
            u_sensors: Function values [B, n_sensors]

        Returns:
            fourier_features: Fourier features [B, 2*fourier_modes]
        """
        B = u_sensors.shape[0]

        # Compute global Fourier transform-like features
        # Simple approach: use mean activation for each frequency
        fourier_features = []

        for freq in self.fourier_freqs:
            # Create spatial coordinates (normalized)
            coords = torch.linspace(0, 2*math.pi, self.n_sensors, device=u_sensors.device)

            # Compute weighted averages with cosine and sine basis
            cos_weights = torch.cos(freq * coords)
            sin_weights = torch.sin(freq * coords)

            cos_feature = (u_sensors * cos_weights).mean(dim=1, keepdim=True)  # [B, 1]
            sin_feature = (u_sensors * sin_weights).mean(dim=1, keepdim=True)  # [B, 1]

            fourier_features.extend([cos_feature, sin_feature])

        return torch.cat(fourier_features, dim=1)  # [B, 2*fourier_modes]

    def forward(self, u_sensors: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Fourier-enhanced branch network.

        Args:
            u_sensors: Function values at sensors [B, n_sensors]

        Returns:
            branch_output: Encoded representation [B, latent_dim]
        """
        # Compute Fourier features
        fourier_features = self.compute_fourier_features(u_sensors)

        # Concatenate original values with Fourier features
        combined_features = torch.cat([u_sensors, fourier_features], dim=1)

        # Forward through network
        return self.network(combined_features)

    def get_input_dim(self) -> int:
        return self.n_sensors

    def get_output_dim(self) -> int:
        return self.latent_dim