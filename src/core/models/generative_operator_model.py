"""
Generative Operator Model

This module implements the main GenerativeOperatorModel class that combines
neural operator priors with generative correctors in a unified interface
compatible with the Gen Stabilised framework.

The model supports different training phases (prior-only, corrector training,
full inference) and provides DCAR (Diffusion-Corrected AutoRegressive) rollout
capabilities.
"""

from typing import Dict, Optional, Any, Tuple, Union
import torch
import torch.nn as nn
import logging

from .base_classes import NeuralOperatorPrior, GenerativeCorrector, DataFormatHandler
from .model_registry import ModelRegistry
from src.core.utils.params import ModelParamsDecoder, DataParams


class GenerativeOperatorModel(nn.Module):
    """
    Universal model-agnostic wrapper for neural operator + generative model combinations.

    This class provides a unified interface for any combination of neural operator
    priors (FNO, TNO, U-Net) with generative correctors (Diffusion, GAN, VAE),
    while maintaining full compatibility with the Gen Stabilised framework.

    Key Features:
    - Mode switching: prior-only, corrector-training, full-inference
    - DCAR rollout for improved long-term predictions
    - Two-stage training support
    - Framework-native parameter integration
    - Memory-efficient inference options
    """

    def __init__(self,
                 prior_model: NeuralOperatorPrior,
                 corrector_model: GenerativeCorrector,
                 p_md: ModelParamsDecoder,
                 p_d: DataParams,
                 **kwargs):
        super().__init__()

        self.prior_model = prior_model
        self.corrector_model = corrector_model
        self.p_md = p_md
        self.p_d = p_d
        self.data_handler = DataFormatHandler()

        # Training phase management
        self.training_mode = 'prior_only'  # 'prior_only', 'corrector_training', 'full_inference'
        self.correction_strength = getattr(p_md, 'correction_strength', 1.0)

        # DCAR configuration
        self.enable_dcar = kwargs.get('enable_dcar', True)
        self.dcar_correction_frequency = kwargs.get('dcar_correction_frequency', 1)  # Correct every N steps

        # Memory optimization
        self.memory_efficient = kwargs.get('memory_efficient', True)
        self.gradient_checkpointing = kwargs.get('gradient_checkpointing', False)

        logging.info(f"Created GenerativeOperatorModel: {type(prior_model).__name__} + {type(corrector_model).__name__}")

    def forward(self, x: torch.Tensor, unused=None) -> torch.Tensor:
        """
        Forward pass through generative operator model.

        Args:
            x: Input tensor [B, T, C, H, W] or [B, C, H, W]
            unused: Unused parameter for compatibility with PredictionModel interface

        Returns:
            output: Predicted tensor matching input shape
        """
        # Handle both [B, T, C, H, W] and [B, C, H, W] formats
        if len(x.shape) == 4:
            # [B, C, H, W] - add dummy time dimension
            x = x.unsqueeze(1)  # [B, 1, C, H, W]
            squeeze_output = True
        elif len(x.shape) == 5:
            squeeze_output = False
        else:
            raise ValueError(f"Expected 4D [B,C,H,W] or 5D [B,T,C,H,W] input, got shape {x.shape}")

        # Validate input format
        if not self.data_handler.validate_gen_stabilised_format(x):
            raise ValueError(f"Expected [B,T,C,H,W] format, got shape {x.shape}")

        if self.training_mode == 'prior_only':
            output = self._forward_prior_only(x)
        elif self.training_mode == 'corrector_training':
            output = self._forward_corrector_training(x)
        elif self.training_mode == 'full_inference':
            output = self._forward_full_inference(x)
        else:
            raise ValueError(f"Unknown training mode: {self.training_mode}")

        # Remove time dimension if input was 4D
        if squeeze_output:
            output = output.squeeze(1)

        return output

    def _forward_prior_only(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using only the neural operator prior."""
        return self.prior_model(x)

    def _forward_corrector_training(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for training the corrector (prior frozen)."""
        # Get prior prediction with no gradients
        with torch.no_grad():
            prior_pred = self.prior_model(x)

        # Extract features for conditioning
        prior_features = self.prior_model.get_prior_features(x)

        # Apply correction
        corrected_pred = self.corrector_model.correct_prediction(
            prior_pred,
            prior_features=prior_features,
            correction_strength=self.correction_strength
        )

        return corrected_pred

    def _forward_full_inference(self, x: torch.Tensor) -> torch.Tensor:
        """Full inference with both prior and corrector."""
        # Get prior prediction
        prior_pred = self.prior_model(x)

        # Extract features for conditioning
        prior_features = self.prior_model.get_prior_features(x)

        # Apply correction
        corrected_pred = self.corrector_model.correct_prediction(
            prior_pred,
            prior_features=prior_features,
            correction_strength=self.correction_strength
        )

        return corrected_pred

    def dcar_rollout(self,
                    initial_states: torch.Tensor,
                    num_steps: int,
                    correction_frequency: Optional[int] = None) -> torch.Tensor:
        """
        DCAR (Diffusion-Corrected AutoRegressive) rollout for long-term prediction.

        This method implements the DCAR pattern where the neural operator provides
        predictions that are corrected by the generative model at specified intervals.

        Args:
            initial_states: Initial condition [B, T_init, C, H, W]
            num_steps: Number of steps to predict
            correction_frequency: How often to apply correction (None = every step)

        Returns:
            trajectory: Complete trajectory [B, T_init + num_steps, C, H, W]
        """
        if not self.enable_dcar:
            raise ValueError("DCAR rollout not enabled. Set enable_dcar=True")

        correction_freq = correction_frequency or self.dcar_correction_frequency
        device = initial_states.device
        B, T_init, C, H, W = initial_states.shape

        # Initialize trajectory with initial states
        trajectory = [initial_states]
        current_input = initial_states

        # Determine input window size based on prior model
        if hasattr(self.prior_model, 'prev_steps'):
            window_size = self.prior_model.prev_steps
        elif hasattr(self.prior_model, 'L'):
            window_size = self.prior_model.L
        else:
            window_size = 1

        logging.info(f"Starting DCAR rollout: {num_steps} steps, correction every {correction_freq} steps")

        for step in range(num_steps):
            # Extract input window
            if len(trajectory) == 1:
                # First step: use initial states
                input_window = trajectory[0][:, -window_size:]
            else:
                # Use last window_size frames from trajectory
                all_frames = torch.cat(trajectory, dim=1)
                input_window = all_frames[:, -window_size:]

            # Ensure input window has correct shape
            if input_window.shape[1] < window_size:
                # Pad with last frame if needed
                last_frame = input_window[:, -1:].repeat(1, window_size - input_window.shape[1], 1, 1, 1)
                input_window = torch.cat([last_frame, input_window], dim=1)

            # Generate next prediction with prior
            with torch.no_grad():
                next_pred = self.prior_model(input_window)
                next_frame = next_pred[:, -1:]  # Take last predicted frame

            # Apply correction at specified frequency
            if (step + 1) % correction_freq == 0 and self.correction_strength > 0:
                # Extract features for conditioning
                prior_features = self.prior_model.get_prior_features(input_window)

                # Apply generative correction
                corrected_frame = self.corrector_model.correct_prediction(
                    next_frame,
                    prior_features=prior_features,
                    correction_strength=self.correction_strength
                )
                next_frame = corrected_frame

            trajectory.append(next_frame)

            # Memory management for very long rollouts
            if self.memory_efficient and len(trajectory) > 50:
                # Keep only recent frames in memory
                trajectory = [torch.cat(trajectory[-25:], dim=1)]

        # Concatenate all frames
        full_trajectory = torch.cat(trajectory, dim=1)

        logging.info(f"DCAR rollout completed: {full_trajectory.shape}")
        return full_trajectory

    def set_training_mode(self, mode: str) -> None:
        """
        Set training mode for the generative operator.

        Args:
            mode: Training mode ('prior_only', 'corrector_training', 'full_inference')
        """
        valid_modes = ['prior_only', 'corrector_training', 'full_inference']
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode: {mode}. Valid modes: {valid_modes}")

        self.training_mode = mode

        # Set component training states
        if mode == 'prior_only':
            self.prior_model.train()
            self.corrector_model.eval()
        elif mode == 'corrector_training':
            self.prior_model.eval()
            self.corrector_model.train()
        elif mode == 'full_inference':
            self.prior_model.eval()
            self.corrector_model.eval()

        logging.info(f"Set training mode to: {mode}")

    def freeze_prior(self) -> None:
        """Freeze neural operator prior for stage 2 training."""
        for param in self.prior_model.parameters():
            param.requires_grad = False
        logging.info("Frozen neural operator prior")

    def unfreeze_prior(self) -> None:
        """Unfreeze neural operator prior."""
        for param in self.prior_model.parameters():
            param.requires_grad = True
        logging.info("Unfrozen neural operator prior")

    def set_correction_strength(self, strength: float) -> None:
        """
        Set correction strength for the generative corrector.

        Args:
            strength: Correction strength (0.0 = no correction, 1.0 = full correction)
        """
        self.correction_strength = max(0.0, min(2.0, strength))
        self.corrector_model.set_correction_strength(strength)
        logging.info(f"Set correction strength to: {strength}")

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information.

        Returns:
            info: Dictionary with model information
        """
        prior_info = self.prior_model.get_model_info()
        corrector_info = self.corrector_model.get_model_info()

        total_params = prior_info['parameter_count'] + corrector_info['parameter_count']
        total_trainable = prior_info['trainable_parameters'] + corrector_info['trainable_parameters']

        return {
            'model_type': 'generative_operator',
            'training_mode': self.training_mode,
            'correction_strength': self.correction_strength,
            'enable_dcar': self.enable_dcar,
            'memory_efficient': self.memory_efficient,
            'prior_model': prior_info,
            'corrector_model': corrector_info,
            'total_parameters': total_params,
            'total_trainable_parameters': total_trainable,
            'parameter_breakdown': {
                'prior_params': prior_info['parameter_count'],
                'corrector_params': corrector_info['parameter_count'],
                'prior_trainable': prior_info['trainable_parameters'],
                'corrector_trainable': corrector_info['trainable_parameters']
            }
        }

    def get_training_config(self) -> Dict[str, Any]:
        """
        Get training configuration for logging.

        Returns:
            config: Training configuration dictionary
        """
        return {
            'training_mode': self.training_mode,
            'correction_strength': self.correction_strength,
            'dcar_enabled': self.enable_dcar,
            'dcar_frequency': self.dcar_correction_frequency,
            'memory_efficient': self.memory_efficient,
            'gradient_checkpointing': self.gradient_checkpointing
        }

    def save_state_dict(self) -> Dict[str, Any]:
        """
        Save complete model state including both components.

        Returns:
            state_dict: Complete state dictionary
        """
        return {
            'prior_model': self.prior_model.state_dict(),
            'corrector_model': self.corrector_model.state_dict(),
            'training_mode': self.training_mode,
            'correction_strength': self.correction_strength,
            'model_config': self.get_training_config()
        }

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True) -> None:
        """
        Load complete model state.

        Args:
            state_dict: State dictionary from save_state_dict
            strict: Whether to strictly enforce state dict keys
        """
        self.prior_model.load_state_dict(state_dict['prior_model'], strict=strict)
        self.corrector_model.load_state_dict(state_dict['corrector_model'], strict=strict)

        # Restore configuration
        self.training_mode = state_dict.get('training_mode', 'full_inference')
        self.correction_strength = state_dict.get('correction_strength', 1.0)

        if 'model_config' in state_dict:
            config = state_dict['model_config']
            self.enable_dcar = config.get('dcar_enabled', True)
            self.dcar_correction_frequency = config.get('dcar_frequency', 1)
            self.memory_efficient = config.get('memory_efficient', True)
            self.gradient_checkpointing = config.get('gradient_checkpointing', False)

        logging.info("Loaded GenerativeOperatorModel state")


def create_generative_operator_model(prior_name: str,
                                   corrector_name: str,
                                   p_md: ModelParamsDecoder,
                                   p_d: DataParams,
                                   **kwargs) -> GenerativeOperatorModel:
    """
    Factory function to create generative operator models using the registry.

    Args:
        prior_name: Name of registered neural operator prior
        corrector_name: Name of registered generative corrector
        p_md: Model parameters
        p_d: Data parameters
        **kwargs: Additional configuration

    Returns:
        model: Configured GenerativeOperatorModel

    Raises:
        ValueError: If models are not registered or incompatible
    """
    # Validate combination
    if not ModelRegistry.validate_combination(prior_name, corrector_name):
        raise ValueError(f"Invalid combination: {prior_name} + {corrector_name}")

    # Create models
    prior_model = ModelRegistry.create_prior(prior_name, p_md, p_d, **kwargs)
    corrector_model = ModelRegistry.create_corrector(corrector_name, p_md, p_d, **kwargs)

    # Create wrapper
    genop_model = GenerativeOperatorModel(
        prior_model=prior_model,
        corrector_model=corrector_model,
        p_md=p_md,
        p_d=p_d,
        **kwargs
    )

    logging.info(f"Created generative operator: {prior_name} + {corrector_name}")
    return genop_model