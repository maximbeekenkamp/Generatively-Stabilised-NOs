"""
Generative Corrector Implementations

This module implements generative corrector models that refine neural operator
predictions using generative modeling techniques. Currently includes diffusion
models with placeholders for future GAN and VAE implementations.

All correctors maintain compatibility with the Gen Stabilised [B,T,C,H,W] format
and integrate with the existing parameter system.
"""

from typing import Dict, Optional, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from random import random
import logging

from .base_classes import GenerativeCorrector, DataFormatHandler
from .model_diffusion_blocks import Unet
from src.core.utils.params import ModelParamsDecoder, DataParams


class ElucidatedDiffusion(nn.Module):
    """
    Adapted ElucidatedDiffusion for Gen Stabilised format.

    This is based on the original NO+DM implementation but modified to work
    with the Gen Stabilised [B,T,C,H,W] tensor format and parameter system.
    """

    def __init__(self,
                 net,
                 *,
                 image_size_h: int,
                 image_size_w: int,
                 channels: int = 3,
                 num_sample_steps: int = 32,
                 sigma_min: float = 0.002,
                 sigma_max: float = 80,
                 sigma_data: float = 0.5,
                 rho: float = 7,
                 P_mean: float = -1.2,
                 P_std: float = 1.2,
                 S_churn: float = 80,
                 S_tmin: float = 0.05,
                 S_tmax: float = 50,
                 S_noise: float = 1.003):
        super().__init__()

        self.self_condition = getattr(net, 'self_condition', False)
        self.net = net

        # Image dimensions - support non-square images
        self.channels = channels
        self.image_size_h = image_size_h
        self.image_size_w = image_size_w

        # Diffusion parameters
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.rho = rho
        self.P_mean = P_mean
        self.P_std = P_std
        self.num_sample_steps = num_sample_steps
        self.S_churn = S_churn
        self.S_tmin = S_tmin
        self.S_tmax = S_tmax
        self.S_noise = S_noise

    @property
    def device(self):
        return next(self.net.parameters()).device

    # Preconditioning functions from original implementation
    def c_skip(self, sigma):
        return (self.sigma_data ** 2) / (sigma ** 2 + self.sigma_data ** 2)

    def c_out(self, sigma):
        return sigma * self.sigma_data * (self.sigma_data ** 2 + sigma ** 2) ** -0.5

    def c_in(self, sigma):
        return 1 * (sigma ** 2 + self.sigma_data ** 2) ** -0.5

    def c_noise(self, sigma):
        return torch.log(sigma) * 0.25

    def preconditioned_network_forward(self, noised_images, sigma, self_cond=None, clamp=False):
        batch, device = noised_images.shape[0], noised_images.device

        if isinstance(sigma, float):
            sigma = torch.full((batch,), sigma, device=device)

        # Handle different spatial dimensions
        if len(noised_images.shape) == 4:  # [B, C, H, W]
            padded_sigma = sigma.view(batch, 1, 1, 1)
        else:
            raise ValueError(f"Unexpected tensor shape: {noised_images.shape}")

        # Apply sigma-dependent input scaling
        scaled_input = self.c_in(padded_sigma) * noised_images

        # Concatenate with conditioning (following DiffusionModel pattern)
        # This allows U-Net to learn: p(u_true | û_NO_prediction)
        if self_cond is not None:
            # Concatenate: [B, C, H, W] + [B, C, H, W] → [B, 2*C, H, W]
            unet_input = torch.cat([scaled_input, self_cond], dim=1)
        else:
            # No conditioning provided - pad with zeros to maintain consistent input size
            # This should rarely happen in NO+DM inference/training
            import warnings
            if self.training:
                warnings.warn("ElucidatedDiffusion received None conditioning during training. "
                             "NO+DM corrector should always receive prior predictions as conditioning.")
            zeros_cond = torch.zeros_like(scaled_input)
            unet_input = torch.cat([scaled_input, zeros_cond], dim=1)

        # Forward through U-Net with doubled input channels
        net_out = self.net(unet_input, self.c_noise(sigma))

        out = self.c_skip(padded_sigma) * noised_images + self.c_out(padded_sigma) * net_out

        if clamp:
            out = out.clamp(-1., 1.)

        return out

    def sample_schedule(self, num_sample_steps=None):
        num_sample_steps = num_sample_steps or self.num_sample_steps
        N = num_sample_steps
        inv_rho = 1 / self.rho

        steps = torch.arange(num_sample_steps, device=self.device, dtype=torch.float32)
        sigmas = (self.sigma_max ** inv_rho + steps / (N - 1) *
                 (self.sigma_min ** inv_rho - self.sigma_max ** inv_rho)) ** self.rho

        sigmas = F.pad(sigmas, (0, 1), value=0.)
        return sigmas

    @torch.no_grad()
    def sample(self, self_cond, batch_size=None, num_sample_steps=None, clamp=True):
        batch_size = self_cond.shape[0] if self_cond is not None else batch_size
        num_sample_steps = num_sample_steps or self.num_sample_steps

        shape = (batch_size, self.channels, self.image_size_h, self.image_size_w)

        sigmas = self.sample_schedule(num_sample_steps)
        gammas = torch.where(
            (sigmas >= self.S_tmin) & (sigmas <= self.S_tmax),
            min(self.S_churn / num_sample_steps, sqrt(2) - 1),
            0.
        )

        sigmas_and_gammas = list(zip(sigmas[:-1], sigmas[1:], gammas[:-1]))

        # Initialize with noise
        init_sigma = sigmas[0]
        images = init_sigma * torch.randn(shape, device=self.device)

        # Denoising loop
        for sigma, sigma_next, gamma in sigmas_and_gammas:
            sigma, sigma_next, gamma = map(lambda t: t.item(), (sigma, sigma_next, gamma))

            eps = self.S_noise * torch.randn(shape, device=self.device)
            sigma_hat = sigma + gamma * sigma
            images_hat = images + sqrt(sigma_hat ** 2 - sigma ** 2) * eps

            model_output = self.preconditioned_network_forward(images_hat, sigma_hat, self_cond, clamp=clamp)
            denoised_over_sigma = (images_hat - model_output) / sigma_hat
            images_next = images_hat + (sigma_next - sigma_hat) * denoised_over_sigma

            # Second order correction
            if sigma_next != 0:
                model_output_next = self.preconditioned_network_forward(images_next, sigma_next, self_cond, clamp=clamp)
                denoised_prime_over_sigma = (images_next - model_output_next) / sigma_next
                images_next = images_hat + 0.5 * (sigma_next - sigma_hat) * (denoised_over_sigma + denoised_prime_over_sigma)

            images = images_next

        images = images.clamp(-1., 1.)
        return self._unnormalize_to_zero_to_one(images)

    def loss_weight(self, sigma):
        return (sigma ** 2 + self.sigma_data ** 2) * (sigma * self.sigma_data) ** -2

    def noise_distribution(self, batch_size):
        return (self.P_mean + self.P_std * torch.randn((batch_size,), device=self.device)).exp()

    def _normalize_to_neg_one_to_one(self, img):
        return img * 2 - 1

    def _unnormalize_to_zero_to_one(self, t):
        return (t + 1) * 0.5

    def forward(self, images, self_cond=None):
        batch_size, c, h, w, device = *images.shape, images.device

        assert h == self.image_size_h and w == self.image_size_w, \
            f'height and width must be {self.image_size_h}x{self.image_size_w}'
        assert c == self.channels, f'channels must be {self.channels}'

        images = self._normalize_to_neg_one_to_one(images)
        sigmas = self.noise_distribution(batch_size)
        padded_sigmas = sigmas.view(batch_size, 1, 1, 1)

        noise = torch.randn_like(images)
        noised_images = images + padded_sigmas * noise

        denoised = self.preconditioned_network_forward(noised_images, sigmas, self_cond)
        losses = F.mse_loss(denoised, images, reduction='none')
        losses = losses.view(batch_size, -1).mean(dim=1)
        losses = losses * self.loss_weight(sigmas)

        return losses.mean()


class DiffusionCorrector(GenerativeCorrector):
    """
    Diffusion-based generative corrector for neural operator predictions.

    This corrector uses an adapted ElucidatedDiffusion model to refine neural
    operator predictions, maintaining compatibility with Gen Stabilised format.
    """

    def __init__(self, p_md: ModelParamsDecoder, p_d: DataParams, **kwargs):
        super().__init__()

        self.p_md = p_md
        self.p_d = p_d
        self.data_handler = DataFormatHandler()

        # Determine channels and spatial dimensions
        self.channels = p_d.dimension + len(p_d.simFields) + len(p_d.simParams)
        self.image_size_h, self.image_size_w = p_d.dataSize

        # Diffusion model configuration
        diffusion_config = kwargs.get('diffusion_config', {})
        self.sigma_data = diffusion_config.get('sigma_data', 0.5)
        self.num_sample_steps = diffusion_config.get('num_sample_steps', 32)
        self.correction_strength = 1.0

        # Create U-Net for diffusion model
        # Input channels doubled for conditioning: [noisy_image | prior_prediction]
        self.unet = Unet(
            dim=max(16, p_md.decWidth // 4),  # Scale down for efficiency
            out_dim=self.channels,
            channels=self.channels * 2,  # Double channels for concatenated conditioning
            dim_mults=(1, 2, 4, 8),
            use_convnext=True,
            convnext_mult=1,
            with_time_emb=True
        )

        # Create ElucidatedDiffusion model
        self.diffusion_model = ElucidatedDiffusion(
            self.unet,
            image_size_h=self.image_size_h,
            image_size_w=self.image_size_w,
            channels=self.channels,
            num_sample_steps=self.num_sample_steps,
            sigma_data=self.sigma_data,
            **diffusion_config
        )

        logging.info(f"Created DiffusionCorrector: {self.channels}ch, {self.image_size_h}x{self.image_size_w}")

    def correct_prediction(self,
                          prior_pred: torch.Tensor,
                          prior_features: Optional[Dict[str, torch.Tensor]] = None,
                          conditioning: Optional[Dict[str, torch.Tensor]] = None,
                          **kwargs) -> torch.Tensor:
        """
        Apply diffusion correction to prior prediction.

        Args:
            prior_pred: Prior prediction [B, T, C, H, W]
            prior_features: Optional features from prior model
            conditioning: Optional additional conditioning
            **kwargs: Additional parameters (correction_strength, num_steps)

        Returns:
            corrected_pred: Diffusion-corrected prediction [B, T, C, H, W]
        """
        # Validate input format
        if not self.data_handler.validate_gen_stabilised_format(prior_pred):
            raise ValueError(f"Expected [B,T,C,H,W] format, got shape {prior_pred.shape}")

        B, T, C, H, W = prior_pred.shape

        # Extract correction parameters
        correction_strength = kwargs.get('correction_strength', self.correction_strength)
        num_steps = kwargs.get('num_steps', self.num_sample_steps)

        # Convert to diffusion format [B*T, C, H, W]
        prior_frames = self.data_handler.convert_for_corrector(prior_pred, 'diffusion')

        # Apply diffusion correction frame by frame
        corrected_frames = []

        # Process in batches to manage memory
        batch_size = 8  # Process 8 frames at a time
        for i in range(0, B * T, batch_size):
            end_idx = min(i + batch_size, B * T)
            batch_frames = prior_frames[i:end_idx]

            # Use prior prediction as conditioning for diffusion
            with torch.no_grad():
                if correction_strength > 0:
                    corrected_batch = self.diffusion_model.sample(
                        self_cond=batch_frames,
                        num_sample_steps=max(1, int(num_steps * correction_strength))
                    )
                else:
                    # No correction - return prior prediction
                    corrected_batch = batch_frames

            corrected_frames.append(corrected_batch)

        # Concatenate all corrected frames
        corrected_output = torch.cat(corrected_frames, dim=0)

        # Convert back to Gen Stabilised format
        corrected_pred = self.data_handler.convert_from_corrector(
            corrected_output, 'diffusion', (B, T, C, H, W)
        )

        return corrected_pred

    def train_step(self,
                   batch: Dict[str, torch.Tensor],
                   prior_pred: torch.Tensor,
                   optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        """
        Training step for diffusion corrector.

        Args:
            batch: Training batch with 'data' and 'target' keys
            prior_pred: Prior predictions for this batch
            optimizer: Optimizer for training

        Returns:
            losses: Dictionary of loss values
        """
        target = batch['target']  # [B, T, C, H, W]

        # Validate shapes
        if not self.data_handler.validate_gen_stabilised_format(target):
            raise ValueError(f"Invalid target shape: {target.shape}")

        B, T, C, H, W = target.shape

        # Convert to diffusion format
        target_frames = self.data_handler.convert_for_corrector(target, 'diffusion')
        prior_frames = self.data_handler.convert_for_corrector(prior_pred, 'diffusion')

        # Forward pass through diffusion model
        diffusion_loss = self.diffusion_model(target_frames, self_cond=prior_frames)

        # Compute consistency loss (prior vs target)
        consistency_loss = F.mse_loss(prior_pred, target)

        # Combined loss
        total_loss = diffusion_loss + 0.1 * consistency_loss

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        return {
            'diffusion_loss': diffusion_loss.item(),
            'consistency_loss': consistency_loss.item(),
            'total_loss': total_loss.item()
        }

    def set_correction_strength(self, strength: float) -> None:
        """
        Set correction strength for inference.

        Args:
            strength: Correction strength (0.0-2.0)
        """
        self.correction_strength = max(0.0, min(2.0, strength))

    def get_model_info(self) -> Dict[str, Any]:
        """Get diffusion corrector information."""
        base_info = super().get_model_info()
        base_info.update({
            'corrector_type': 'diffusion',
            'channels': self.channels,
            'image_size': (self.image_size_h, self.image_size_w),
            'num_sample_steps': self.num_sample_steps,
            'sigma_data': self.sigma_data,
            'correction_strength': self.correction_strength
        })
        return base_info


class GANCorrector(GenerativeCorrector):
    """
    GAN-based generative corrector (placeholder for future implementation).

    This is a placeholder class for future GAN-based correction models.
    """

    def __init__(self, p_md: ModelParamsDecoder, p_d: DataParams, **kwargs):
        super().__init__()
        raise NotImplementedError("GAN corrector not yet implemented")

    def correct_prediction(self, prior_pred: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError("GAN corrector not yet implemented")

    def train_step(self, batch: Dict[str, torch.Tensor], prior_pred: torch.Tensor,
                   optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        raise NotImplementedError("GAN corrector not yet implemented")

    def set_correction_strength(self, strength: float) -> None:
        raise NotImplementedError("GAN corrector not yet implemented")


class VAECorrector(GenerativeCorrector):
    """
    VAE-based generative corrector (placeholder for future implementation).

    This is a placeholder class for future VAE-based correction models.
    """

    def __init__(self, p_md: ModelParamsDecoder, p_d: DataParams, **kwargs):
        super().__init__()
        raise NotImplementedError("VAE corrector not yet implemented")

    def correct_prediction(self, prior_pred: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError("VAE corrector not yet implemented")

    def train_step(self, batch: Dict[str, torch.Tensor], prior_pred: torch.Tensor,
                   optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        raise NotImplementedError("VAE corrector not yet implemented")

    def set_correction_strength(self, strength: float) -> None:
        raise NotImplementedError("VAE corrector not yet implemented")