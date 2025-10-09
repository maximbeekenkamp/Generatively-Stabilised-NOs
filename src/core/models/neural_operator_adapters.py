"""
Neural Operator Prior Adapters

This module implements adapter classes that wrap existing neural operator models
(FNO, TNO, U-Net) to serve as priors in generative operator architectures.

The adapters follow the NeuralOperatorPrior interface while maintaining
compatibility with existing Gen Stabilised models and their parameter systems.
"""

from typing import Dict, Optional, Any
import torch
import torch.nn as nn
from neuralop.models import FNO

from .base_classes import NeuralOperatorPrior, DataFormatHandler
from .model_tno import TNOModel
from .model_diffusion_blocks import Unet
from src.core.utils.params import ModelParamsDecoder, DataParams
from src.core.utils.model_utils import get_prev_steps_from_arch, calculate_input_channels, calculate_output_channels


class FNOPriorAdapter(NeuralOperatorPrior):
    """
    Adapter for Fourier Neural Operator models to serve as priors.

    This adapter wraps the existing FNO implementation from neuralop,
    making it compatible with the generative operator interface while
    preserving all existing functionality.
    """

    def __init__(self, p_md: ModelParamsDecoder, p_d: DataParams, **kwargs):
        super().__init__()

        self.p_md = p_md
        self.p_d = p_d
        self.data_handler = DataFormatHandler()

        # Determine input/output channels based on data parameters
        prevSteps = get_prev_steps_from_arch(p_md)
        inChannels = calculate_input_channels(p_d, prevSteps)
        outChannels = calculate_output_channels(p_d)

        # Create FNO model using existing patterns
        self.fno_model = FNO(
            n_modes=(p_md.fnoModes[0], p_md.fnoModes[1]),
            hidden_channels=p_md.decWidth,
            in_channels=inChannels,
            out_channels=outChannels,
            n_layers=4
        )

        self.prev_steps = prevSteps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through FNO with Gen Stabilised format preservation.

        Args:
            x: Input tensor [B, T, C, H, W]

        Returns:
            prediction: FNO prediction [B, T, C, H, W]
        """
        # Validate input format
        if not self.data_handler.validate_gen_stabilised_format(x):
            raise ValueError(f"Expected [B,T,C,H,W] format, got shape {x.shape}")

        sizeBatch, sizeSeq = x.shape[0], x.shape[1]

        # Initialize prediction with input frames based on prev_steps
        prediction = []
        for i in range(self.prev_steps):
            if i < sizeSeq:
                prediction.append(x[:, i])
            else:
                # Pad with last available frame if needed
                prediction.append(x[:, -1])

        # Process remaining timesteps
        for i in range(self.prev_steps, sizeSeq):
            # Prepare input: concatenate previous frames
            uIn = torch.cat(prediction[i-self.prev_steps:i], dim=1)  # [B, prev_steps*C, H, W]

            # FNO forward pass
            result = self.fno_model(uIn)  # [B, C, H, W]

            # Preserve simulation parameters if they exist
            if self.p_d.simParams and len(self.p_d.simParams) > 0:
                param_channels = len(self.p_d.simParams)
                result[:, -param_channels:] = x[:, i, -param_channels:]

            prediction.append(result)

        # Stack predictions to [B, T, C, H, W]
        prediction = torch.stack(prediction, dim=1)

        return prediction

    def get_prior_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract FNO intermediate features for conditioning.

        Args:
            x: Input tensor [B, T, C, H, W]

        Returns:
            features: Dictionary of conditioning features
        """
        # For now, return empty features
        # TODO: Modify FNO to extract intermediate features if needed
        return {}

    def validate_input_shape(self, x: torch.Tensor) -> bool:
        """
        Validate input shape compatibility with FNO.

        Args:
            x: Input tensor to validate

        Returns:
            valid: True if compatible
        """
        if not self.data_handler.validate_gen_stabilised_format(x):
            return False

        # Check spatial dimensions match expected size
        expected_h, expected_w = self.p_d.dataSize
        _, _, _, h, w = x.shape

        return h == expected_h and w == expected_w


class TNOPriorAdapter(NeuralOperatorPrior):
    """
    Adapter for Transformer Neural Operator models to serve as priors.

    This adapter wraps the existing TNO implementation, providing
    enhanced integration with the generative operator framework.
    """

    def __init__(self, p_md: ModelParamsDecoder, p_d: DataParams, **kwargs):
        super().__init__()

        self.p_md = p_md
        self.p_d = p_d
        self.data_handler = DataFormatHandler()

        # Extract TNO parameters from architecture string
        tno_L = self._get_tno_L_from_arch()
        tno_K = 1 if tno_L == 1 else 4  # Phase 0: K=1, Phase 1+: K=4
        tno_width = p_md.decWidth or 360
        dataset_type = self._infer_dataset_type()

        # Create TNO model using existing implementation
        self.tno_model = TNOModel(
            width=tno_width,
            L=tno_L,
            K=tno_K,
            dataset_type=dataset_type
        )

        self.L = tno_L
        self.K = tno_K

    def _get_tno_L_from_arch(self) -> int:
        """Extract L parameter from architecture string (TNO-specific wrapper)."""
        return get_prev_steps_from_arch(self.p_md)

    def _infer_dataset_type(self) -> str:
        """Infer dataset type from data parameters."""
        if hasattr(self.p_d, 'simParams') and self.p_d.simParams:
            if "rey" in self.p_d.simParams and len(self.p_d.simParams) == 1:
                return "inc"
            elif "mach" in self.p_d.simParams and "rey" in self.p_d.simParams:
                return "tra"

        if hasattr(self.p_d, 'simFields') and self.p_d.simFields:
            if "velZ" in self.p_d.simFields:
                return "iso"

        return "inc"  # Default

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through TNO with enhanced temporal bundling.

        Args:
            x: Input tensor [B, T, C, H, W]

        Returns:
            prediction: TNO prediction [B, T, C, H, W]
        """
        # Validate input format
        if not self.data_handler.validate_gen_stabilised_format(x):
            raise ValueError(f"Expected [B,T,C,H,W] format, got shape {x.shape}")

        # Use existing TNO forward method from PredictionModel
        return self._forward_tno_adapted(x)

    def _forward_tno_adapted(self, d: torch.Tensor) -> torch.Tensor:
        """
        Adapted TNO forward pass from PredictionModel.forwardTNO.

        Args:
            d: Input tensor [B, T, C, H, W]

        Returns:
            prediction: Output tensor [B, T, C, H, W]
        """
        sizeBatch, sizeSeq = d.shape[0], d.shape[1]
        device = d.device
        dtype = d.dtype

        # Initialize prediction with first timestep
        prediction = [d[:, 0]]

        # Enhanced temporal bundling loop
        i = 1
        while i < sizeSeq:
            # Dynamic K adaptation based on remaining steps
            remaining_steps = sizeSeq - i
            effective_K = min(self.K, remaining_steps)

            # Prepare TNO input with proper history
            input_start = max(0, i - self.L)
            input_end = i

            # Extract input sequence for TNO
            if input_end > input_start:
                tno_input = d[:, input_start:input_end + 1]
            else:
                # Edge case: not enough history, pad with first timestep
                tno_input = d[:, 0:1].expand(-1, self.L + 1, -1, -1, -1)

            # TNO forward pass
            tno_output = self.tno_model(tno_input)  # [B, K, target_channels, H, W]

            # Extract only needed predictions
            tno_predictions = tno_output[:, :effective_K]

            # Handle different field reconstruction strategies
            num_input_fields = d.shape[2]
            target_channels = tno_predictions.shape[2]

            # Create full field predictions
            for step_idx in range(effective_K):
                step_pred = tno_predictions[:, step_idx]  # [B, target_channels, H, W]

                # Create full field tensor
                full_pred = torch.zeros(
                    sizeBatch, num_input_fields, d.shape[3], d.shape[4],
                    device=device, dtype=dtype
                )

                # Map channels appropriately
                if target_channels == 1:
                    full_pred[:, 0:1] = step_pred
                    # Preserve other fields
                    if num_input_fields > 1 and i + step_idx < sizeSeq:
                        full_pred[:, 1:] = d[:, i + step_idx, 1:]
                elif target_channels > 1:
                    channels_to_copy = min(target_channels, num_input_fields)
                    full_pred[:, :channels_to_copy] = step_pred[:, :channels_to_copy]
                    # Preserve remaining fields
                    if num_input_fields > target_channels and i + step_idx < sizeSeq:
                        full_pred[:, target_channels:] = d[:, i + step_idx, target_channels:]

                prediction.append(full_pred)

            i += effective_K

        # Stack all predictions
        prediction = torch.stack(prediction, dim=1)

        return prediction

    def get_prior_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract TNO features for conditioning.

        Args:
            x: Input tensor [B, T, C, H, W]

        Returns:
            features: Dictionary of TNO features
        """
        # Return TNO status and configuration for conditioning
        status = self.tno_model.get_status()
        return {
            'tno_config': {
                'L': status['L'],
                'K': status['K'],
                'dataset_type': status['dataset_type'],
                'width': status['width']
            }
        }

    def validate_input_shape(self, x: torch.Tensor) -> bool:
        """
        Validate input shape compatibility with TNO.

        Args:
            x: Input tensor to validate

        Returns:
            valid: True if compatible
        """
        if not self.data_handler.validate_gen_stabilised_format(x):
            return False

        # Check minimum sequence length
        _, T, _, _, _ = x.shape
        return T >= self.L

    def set_training_phase(self, phase: str):
        """Set TNO training phase."""
        if hasattr(self.tno_model, 'set_training_phase'):
            self.tno_model.set_training_phase(phase)

    def update_epoch(self, epoch: int):
        """Update TNO epoch."""
        if hasattr(self.tno_model, 'update_epoch'):
            self.tno_model.update_epoch(epoch)


class UNetPriorAdapter(NeuralOperatorPrior):
    """
    Adapter for U-Net models to serve as priors.

    This adapter wraps the existing U-Net implementation, making it
    compatible with the generative operator interface.
    """

    def __init__(self, p_md: ModelParamsDecoder, p_d: DataParams, **kwargs):
        super().__init__()

        self.p_md = p_md
        self.p_d = p_d
        self.data_handler = DataFormatHandler()

        # Determine input/output channels
        prevSteps = get_prev_steps_from_arch(p_md)
        inChannels = calculate_input_channels(p_d, prevSteps)
        outChannels = calculate_output_channels(p_d)

        # Create U-Net model using existing implementation
        self.unet_model = Unet(
            dim=p_d.dataSize[0],
            out_dim=outChannels,
            channels=inChannels,
            dim_mults=(1, 1, 1),
            use_convnext=True,
            convnext_mult=1,
            with_time_emb=False
        )

        self.prev_steps = prevSteps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through U-Net with Gen Stabilised format preservation.

        Args:
            x: Input tensor [B, T, C, H, W]

        Returns:
            prediction: U-Net prediction [B, T, C, H, W]
        """
        # Validate input format
        if not self.data_handler.validate_gen_stabilised_format(x):
            raise ValueError(f"Expected [B,T,C,H,W] format, got shape {x.shape}")

        sizeBatch, sizeSeq = x.shape[0], x.shape[1]

        # Initialize prediction with input frames
        prediction = []
        for i in range(self.prev_steps):
            if i < sizeSeq:
                prediction.append(x[:, i])
            else:
                prediction.append(x[:, -1])

        # Process remaining timesteps
        for i in range(self.prev_steps, sizeSeq):
            # Prepare input: concatenate previous frames
            uIn = torch.cat(prediction[i-self.prev_steps:i], dim=1)

            # U-Net forward pass
            result = self.unet_model(uIn, None)  # Second parameter is time embedding (None)

            # Preserve simulation parameters
            if self.p_d.simParams and len(self.p_d.simParams) > 0:
                param_channels = len(self.p_d.simParams)
                result[:, -param_channels:] = x[:, i, -param_channels:]

            prediction.append(result)

        # Stack predictions
        prediction = torch.stack(prediction, dim=1)

        return prediction

    def get_prior_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract U-Net features for conditioning.

        Args:
            x: Input tensor [B, T, C, H, W]

        Returns:
            features: Dictionary of U-Net features
        """
        # Return empty features for now
        # TODO: Modify U-Net to extract intermediate features if needed
        return {}

    def validate_input_shape(self, x: torch.Tensor) -> bool:
        """
        Validate input shape compatibility with U-Net.

        Args:
            x: Input tensor to validate

        Returns:
            valid: True if compatible
        """
        if not self.data_handler.validate_gen_stabilised_format(x):
            return False

        # Check spatial dimensions
        expected_h, expected_w = self.p_d.dataSize
        _, _, _, h, w = x.shape

        return h == expected_h and w == expected_w


class DeepONetPriorAdapter(NeuralOperatorPrior):
    """
    DeepONet Prior Adapter for Gen Stabilised Framework.

    This adapter integrates DeepONet (Deep Operator Network) as a neural operator
    prior in the generative operator framework. DeepONet uses a branch-trunk
    architecture for operator learning.

    Key architectural features:
    - Branch network: Processes discretized input fields
    - Trunk network: Processes query coordinates (batched)
    - Per-channel processing: Each field processed independently
    - Autoregressive rollout: Processes one timestep at a time

    Difference from TNO:
    - No temporal branch (t-branch) - processes single timesteps
    - Standard MLP networks instead of U-Nets
    - Element-wise multiplication for branch-trunk combination

    Args:
        p_md: Model decoder parameters
        p_d: Data parameters
        **kwargs: Additional configuration overrides
    """

    def __init__(self, p_md: ModelParamsDecoder, p_d: DataParams, **kwargs):
        super().__init__()

        from .deeponet.mlp_networks import MLP, DeepONet
        from .deeponet.deeponet_config import DeepONetConfig
        from .deeponet.deeponet_adapter import DeepONetFormatAdapter

        self.p_md = p_md
        self.p_d = p_d
        self.data_handler = DataFormatHandler()

        # Create configuration
        config = DeepONetConfig.from_params(p_md, p_d)
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        # Spatial dimensions
        self.H, self.W = p_d.dataSize[-2], p_d.dataSize[-1]

        # Create branch and trunk MLPs
        branch_config = config.get_branch_config()
        trunk_config = config.get_trunk_config()

        branch_mlp = MLP(*branch_config)
        trunk_mlp = MLP(*trunk_config)

        # Create DeepONet
        base_deeponet = DeepONet(
            latent_features=config.latent_features,
            out_features=1,  # Single output per query point
            branch=branch_mlp,
            trunk=trunk_mlp
        )

        # Wrap with format adapter
        self.deeponet_wrapper = DeepONetFormatAdapter(
            deeponet_model=base_deeponet,
            spatial_dims=(self.H, self.W),
            num_channels=config.num_channels
        )

        self.prev_steps = get_prev_steps_from_arch(p_md)

        print(f"[DeepONet] Initialized: spatial={self.H}×{self.W}, "
              f"channels={config.num_channels}, latent={config.latent_features}, "
              f"prev_steps={self.prev_steps}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Autoregressive rollout through DeepONet.

        Processes sequence autoregressively, one timestep at a time. Unlike TNO
        which uses temporal bundling, DeepONet processes each frame independently.

        Args:
            x: Input tensor [B, T, C, H, W]

        Returns:
            prediction: Predicted sequence [B, T, C, H, W]
        """
        if not self.data_handler.validate_gen_stabilised_format(x):
            raise ValueError(f"Expected [B,T,C,H,W], got {x.shape}")

        B, T, C, H, W = x.shape

        # Initialize with first frames
        prediction = [x[:, i] for i in range(min(self.prev_steps, T))]

        # Autoregressive rollout
        for i in range(self.prev_steps, T):
            # Use most recent prediction as input
            current_input = prediction[-1]  # [B, C, H, W]

            # DeepONet forward (adapter handles per-channel processing)
            pred = self.deeponet_wrapper(current_input)

            # Preserve simulation parameters
            if self.p_d.simParams and len(self.p_d.simParams) > 0:
                param_channels = len(self.p_d.simParams)
                pred[:, -param_channels:] = x[:, i, -param_channels:]

            prediction.append(pred)

        return torch.stack(prediction, dim=1)

    def get_prior_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract DeepONet configuration features."""
        return {
            'deeponet_config': {
                'spatial_dims': (self.H, self.W),
                'prev_steps': self.prev_steps,
                'has_t_branch': False,  # Key difference from TNO
                'processing_mode': 'per_channel'
            }
        }

    def validate_input_shape(self, x: torch.Tensor) -> bool:
        """Validate input shape compatibility."""
        if not self.data_handler.validate_gen_stabilised_format(x):
            return False
        _, _, _, h, w = x.shape
        return h == self.H and w == self.W


class DeepOKANPriorAdapter(NeuralOperatorPrior):
    """
    DeepOKAN Prior Adapter - DeepONet with KAN layers.

    This adapter integrates DeepOKAN (DeepONet with Kolmogorov-Arnold Networks)
    as a neural operator prior. DeepOKAN replaces standard MLPs with RBF-KAN
    networks, providing improved expressiveness for operator learning.

    Key architectural differences from DeepONet:
    - RBF-KAN networks instead of MLPs
    - Shared trunk: [H*W, 2] not [B, H*W, 2]
    - Einsum combination: 'bik,nk->bni'
    - Higher precision: float64 for KAN stability

    Args:
        p_md: Model decoder parameters
        p_d: Data parameters
        **kwargs: Additional configuration overrides
    """

    def __init__(self, p_md: ModelParamsDecoder, p_d: DataParams, **kwargs):
        super().__init__()

        from .deepokan.deepokan_base import DeepOKAN
        from .deepokan.deepokan_config import DeepOKANConfig
        from .deepokan.deepokan_adapter import DeepOKANFormatAdapter

        self.p_md = p_md
        self.p_d = p_d
        self.data_handler = DataFormatHandler()

        # Create configuration
        config = DeepOKANConfig.from_params(p_md, p_d)
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        # Spatial dimensions
        self.H, self.W = p_d.dataSize[-2], p_d.dataSize[-1]

        # Create DeepOKAN
        base_deepokan = DeepOKAN(config)

        # Wrap with format adapter
        self.deepokan_wrapper = DeepOKANFormatAdapter(
            deepokan_model=base_deepokan,
            spatial_dims=(self.H, self.W),
            num_channels=len(p_d.simFields) if p_d.simFields else 2
        )

        self.prev_steps = get_prev_steps_from_arch(p_md)

        print(f"[DeepOKAN] Initialized: spatial={self.H}×{self.W}, "
              f"HD={config.HD}, grid_count={config.grid_count}, "
              f"prev_steps={self.prev_steps}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Autoregressive rollout through DeepOKAN.

        Same autoregressive pattern as DeepONet, but using KAN networks
        and shared trunk architecture.

        Args:
            x: Input tensor [B, T, C, H, W]

        Returns:
            prediction: Predicted sequence [B, T, C, H, W]
        """
        if not self.data_handler.validate_gen_stabilised_format(x):
            raise ValueError(f"Expected [B,T,C,H,W], got {x.shape}")

        B, T, C, H, W = x.shape

        # Initialize with first frames
        prediction = [x[:, i] for i in range(min(self.prev_steps, T))]

        # Autoregressive rollout
        for i in range(self.prev_steps, T):
            current_input = prediction[-1]  # [B, C, H, W]

            # DeepOKAN forward (adapter handles shared trunk and per-channel processing)
            pred = self.deepokan_wrapper(current_input)

            # Preserve simulation parameters
            if self.p_d.simParams and len(self.p_d.simParams) > 0:
                param_channels = len(self.p_d.simParams)
                pred[:, -param_channels:] = x[:, i, -param_channels:]

            prediction.append(pred)

        return torch.stack(prediction, dim=1)

    def get_prior_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract DeepOKAN configuration features."""
        return {
            'deepokan_config': {
                'spatial_dims': (self.H, self.W),
                'prev_steps': self.prev_steps,
                'uses_kan': True,
                'trunk_mode': 'shared'  # Key difference: trunk is shared, not batched
            }
        }

    def validate_input_shape(self, x: torch.Tensor) -> bool:
        """Validate input shape compatibility."""
        if not self.data_handler.validate_gen_stabilised_format(x):
            return False
        _, _, _, h, w = x.shape
        return h == self.H and w == self.W