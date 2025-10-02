"""
Abstract Base Classes for Model-Agnostic Generative Operators

This module defines the abstract interfaces for pluggable neural operator priors
and generative correctors in the Gen Stabilised framework. These interfaces enable
seamless swapping of different model types (FNO/TNO/U-Net priors with Diffusion/GAN/VAE correctors).

Key Design Principles:
- Minimal abstraction overhead - prefer composition over inheritance
- Gen Stabilised [B,T,C,H,W] format preservation throughout
- Framework-native parameter system integration
- Extensibility for future model types without framework changes
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Any, Tuple
import torch
import torch.nn as nn


class NeuralOperatorPrior(nn.Module, ABC):
    """
    Abstract base class for neural operator priors in generative operator models.

    Any neural operator (FNO, TNO, U-Net, etc.) can serve as a prior by implementing
    this interface. The prior provides initial predictions that are then refined by
    a generative corrector.

    All implementations must preserve the Gen Stabilised [B,T,C,H,W] tensor format.
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate prior prediction from input sequence.

        Args:
            x: Input tensor in Gen Stabilised format [B, T, C, H, W]

        Returns:
            prior_pred: Prior prediction tensor [B, T, C, H, W]
        """
        pass

    @abstractmethod
    def get_prior_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract conditioning features for generative model.

        Args:
            x: Input tensor [B, T, C, H, W]

        Returns:
            features: Dictionary of feature tensors for conditioning
                     e.g., {'intermediate_features': [B, T, F, H, W],
                            'global_features': [B, T, F]}
        """
        pass

    @abstractmethod
    def validate_input_shape(self, x: torch.Tensor) -> bool:
        """
        Validate input tensor shape compatibility.

        Args:
            x: Input tensor to validate

        Returns:
            valid: True if input shape is compatible with this prior
        """
        pass

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information for logging and debugging.

        Returns:
            info: Dictionary containing model metadata
        """
        return {
            'type': 'neural_operator_prior',
            'model_class': self.__class__.__name__,
            'parameter_count': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }


class GenerativeCorrector(nn.Module, ABC):
    """
    Abstract base class for generative correctors in generative operator models.

    Any generative model (Diffusion, GAN, VAE, etc.) can serve as a corrector by
    implementing this interface. The corrector refines prior predictions using
    generative modeling techniques.

    All implementations must preserve the Gen Stabilised [B,T,C,H,W] tensor format.
    """

    @abstractmethod
    def correct_prediction(self,
                          prior_pred: torch.Tensor,
                          prior_features: Optional[Dict[str, torch.Tensor]] = None,
                          conditioning: Optional[Dict[str, torch.Tensor]] = None,
                          **kwargs) -> torch.Tensor:
        """
        Apply generative correction to prior prediction.

        Args:
            prior_pred: Prior prediction tensor [B, T, C, H, W]
            prior_features: Optional features from prior model for conditioning
            conditioning: Optional additional conditioning information
            **kwargs: Model-specific parameters (e.g., correction_strength, num_steps)

        Returns:
            corrected_pred: Corrected prediction tensor [B, T, C, H, W]
        """
        pass

    @abstractmethod
    def train_step(self,
                   batch: Dict[str, torch.Tensor],
                   prior_pred: torch.Tensor,
                   optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        """
        Training step compatible with Gen Stabilised Trainer.

        Args:
            batch: Training batch with 'data' and 'target' keys
            prior_pred: Prior predictions for this batch
            optimizer: Optimizer for training step

        Returns:
            losses: Dictionary of loss values for logging
        """
        pass

    @abstractmethod
    def set_correction_strength(self, strength: float) -> None:
        """
        Control the intensity of generative correction.

        Args:
            strength: Correction strength (0.0 = no correction, 1.0 = full correction)
        """
        pass

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information for logging and debugging.

        Returns:
            info: Dictionary containing model metadata
        """
        return {
            'type': 'generative_corrector',
            'model_class': self.__class__.__name__,
            'parameter_count': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }


class DataFormatHandler:
    """
    Utility class for handling data format conversions in generative operators.

    This class provides static methods for converting between Gen Stabilised
    format and model-specific formats, ensuring shape consistency throughout
    the data pipeline.
    """

    @staticmethod
    def validate_gen_stabilised_format(x: torch.Tensor) -> bool:
        """
        Validate tensor is in Gen Stabilised [B, T, C, H, W] format.

        Args:
            x: Tensor to validate

        Returns:
            valid: True if tensor has correct 5D shape
        """
        return len(x.shape) == 5 and x.shape[1] > 0  # [B, T, C, H, W] with T > 0

    @staticmethod
    def convert_for_prior(x: torch.Tensor, prior_type: str) -> torch.Tensor:
        """
        Convert from Gen Stabilised format to prior-specific format if needed.

        Args:
            x: Input tensor [B, T, C, H, W]
            prior_type: Type of prior model ('fno', 'tno', 'unet', etc.)

        Returns:
            converted: Tensor in format expected by prior model
        """
        # Most priors can handle [B, T, C, H, W] directly
        # Special handling can be added here for specific prior types
        if prior_type == 'tno':
            # TNO might need special temporal dimension handling
            return x
        elif prior_type == 'fno':
            # FNO expects standard format
            return x
        elif prior_type == 'unet':
            # U-Net typically works on individual frames
            return x
        else:
            return x

    @staticmethod
    def convert_for_corrector(x: torch.Tensor, corrector_type: str) -> torch.Tensor:
        """
        Convert tensor format for corrector model if needed.

        Args:
            x: Input tensor [B, T, C, H, W]
            corrector_type: Type of corrector ('diffusion', 'gan', 'vae', etc.)

        Returns:
            converted: Tensor in format expected by corrector
        """
        if corrector_type == 'diffusion':
            # Diffusion models typically work on individual frames [B*T, C, H, W]
            B, T, C, H, W = x.shape
            return x.reshape(B * T, C, H, W)
        elif corrector_type == 'gan':
            # GANs might work on full sequences or individual frames
            return x
        elif corrector_type == 'vae':
            # VAEs typically work on individual frames
            B, T, C, H, W = x.shape
            return x.reshape(B * T, C, H, W)
        else:
            return x

    @staticmethod
    def convert_from_corrector(x: torch.Tensor,
                              corrector_type: str,
                              target_shape: Tuple[int, ...]) -> torch.Tensor:
        """
        Convert corrector output back to Gen Stabilised format.

        Args:
            x: Corrector output tensor
            corrector_type: Type of corrector that produced the output
            target_shape: Target shape [B, T, C, H, W]

        Returns:
            converted: Tensor in Gen Stabilised format [B, T, C, H, W]
        """
        if corrector_type == 'diffusion':
            # Reshape from [B*T, C, H, W] back to [B, T, C, H, W]
            B, T, C, H, W = target_shape
            return x.reshape(B, T, C, H, W)
        elif corrector_type == 'vae':
            # Reshape from [B*T, C, H, W] back to [B, T, C, H, W]
            B, T, C, H, W = target_shape
            return x.reshape(B, T, C, H, W)
        else:
            # Assume output is already in correct format
            return x

    @staticmethod
    def extract_shape_info(x: torch.Tensor) -> Dict[str, int]:
        """
        Extract shape information from Gen Stabilised tensor.

        Args:
            x: Input tensor [B, T, C, H, W]

        Returns:
            shape_info: Dictionary with shape dimensions
        """
        if not DataFormatHandler.validate_gen_stabilised_format(x):
            raise ValueError(f"Expected 5D tensor [B,T,C,H,W], got shape {x.shape}")

        B, T, C, H, W = x.shape
        return {
            'batch_size': B,
            'sequence_length': T,
            'channels': C,
            'height': H,
            'width': W,
            'spatial_size': (H, W),
            'total_elements': x.numel()
        }