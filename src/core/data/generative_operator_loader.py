"""
Data Loading Pipeline for Generative Operator Training

This module extends the existing TurbulenceDataset to support generative operator
training patterns, including specialized data preparation for two-stage training
and enhanced data augmentation strategies.

Author: Phase 2 Implementation
"""

import os
import copy
import logging
from typing import Dict, List, Tuple, Optional, Union
import random

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler

from src.core.utils.params import DataParams, TrainingParams
from src.core.data_processing.turbulence_dataset import TurbulenceDataset
from src.core.data_processing.data_transformations import Transforms


class GenerativeOperatorDataset(Dataset):
    """
    Enhanced dataset wrapper for generative operator training.

    Provides specialized data loading patterns for two-stage training:
    - Prior training: Standard sequence prediction
    - Corrector training: Prior predictions + targets for correction

    Features:
    - Dynamic batch composition for different training stages
    - Enhanced augmentation strategies
    - Memory-efficient data loading
    - Support for multi-scale training
    """

    def __init__(self,
                 base_dataset: TurbulenceDataset,
                 training_stage: str = 'prior_only',
                 prior_model: Optional[torch.nn.Module] = None,
                 augmentation_strategy: str = 'standard',
                 memory_efficient: bool = True):
        """
        Initialize GenerativeOperatorDataset.

        Args:
            base_dataset: Base TurbulenceDataset instance
            training_stage: 'prior_only', 'corrector_training', or 'full_inference'
            prior_model: Prior model for generating predictions during corrector training
            augmentation_strategy: 'standard', 'conservative', 'aggressive'
            memory_efficient: Enable memory optimizations
        """
        self.base_dataset = base_dataset
        self.training_stage = training_stage
        self.prior_model = prior_model
        self.augmentation_strategy = augmentation_strategy
        self.memory_efficient = memory_efficient

        # Enhanced augmentation probabilities based on strategy
        self.augmentation_probs = {
            'conservative': {
                'flip': 0.3,
                'rotate': 0.2,
                'noise': 0.1,
                'scale': 0.15
            },
            'standard': {
                'flip': 0.5,
                'rotate': 0.3,
                'noise': 0.2,
                'scale': 0.25
            },
            'aggressive': {
                'flip': 0.7,
                'rotate': 0.5,
                'noise': 0.3,
                'scale': 0.4
            }
        }

        # Cache for prior predictions (for corrector training)
        self.prior_cache = {} if training_stage == 'corrector_training' else None

        logging.info(f"Created GenerativeOperatorDataset:")
        logging.info(f"  Training stage: {training_stage}")
        logging.info(f"  Augmentation strategy: {augmentation_strategy}")
        logging.info(f"  Memory efficient: {memory_efficient}")
        logging.info(f"  Base dataset length: {len(base_dataset)}")

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get item with stage-aware data preparation.

        Returns:
            sample: Dictionary containing data prepared for the current training stage
        """
        # Get base sample
        sample = self.base_dataset[idx]
        data = sample['data']  # [T, C, H, W] format

        # Convert to tensor and ensure [B, T, C, H, W] format for batch processing
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()

        # Add batch dimension if not present
        if data.dim() == 4:  # [T, C, H, W] -> [1, T, C, H, W]
            data = data.unsqueeze(0)

        # Apply stage-specific processing
        if self.training_stage == 'prior_only':
            return self._prepare_prior_training_sample(data, sample, idx)
        elif self.training_stage == 'corrector_training':
            return self._prepare_corrector_training_sample(data, sample, idx)
        else:  # full_inference
            return self._prepare_inference_sample(data, sample, idx)

    def _prepare_prior_training_sample(self, data: torch.Tensor,
                                     original_sample: dict, idx: int) -> Dict[str, torch.Tensor]:
        """Prepare sample for prior model training."""
        B, T, C, H, W = data.shape

        # Standard sequence prediction: input -> target
        input_frames = data[:, :-1]  # [B, T-1, C, H, W]
        target_frames = data[:, 1:]  # [B, T-1, C, H, W]

        # Apply augmentations
        if self.base_dataset.transform:
            input_frames, target_frames = self._apply_augmentations(input_frames, target_frames)

        sample = {
            'input': input_frames,
            'target': target_frames,
            'stage': 'prior_only',
            'path': original_sample.get('path', ''),
            'idx': idx
        }

        # Add simulation parameters if available
        if 'simParameters' in original_sample:
            sample['sim_params'] = torch.from_numpy(
                original_sample['simParameters']
            ).float() if isinstance(original_sample['simParameters'], np.ndarray) else original_sample['simParameters']

        return sample

    def _prepare_corrector_training_sample(self, data: torch.Tensor,
                                         original_sample: dict, idx: int) -> Dict[str, torch.Tensor]:
        """Prepare sample for corrector model training."""
        B, T, C, H, W = data.shape

        input_frames = data[:, :-1]  # [B, T-1, C, H, W]
        target_frames = data[:, 1:]  # [B, T-1, C, H, W]

        # Get or generate prior predictions
        if idx in self.prior_cache:
            prior_predictions = self.prior_cache[idx]
        else:
            prior_predictions = self._generate_prior_predictions(input_frames, idx)

        # Apply augmentations to both prior predictions and targets
        if self.base_dataset.transform:
            prior_predictions, target_frames = self._apply_augmentations(
                prior_predictions, target_frames
            )

        sample = {
            'input': input_frames,
            'prior_prediction': prior_predictions,
            'target': target_frames,
            'stage': 'corrector_training',
            'path': original_sample.get('path', ''),
            'idx': idx
        }

        # Add simulation parameters if available
        if 'simParameters' in original_sample:
            sample['sim_params'] = torch.from_numpy(
                original_sample['simParameters']
            ).float() if isinstance(original_sample['simParameters'], np.ndarray) else original_sample['simParameters']

        return sample

    def _prepare_inference_sample(self, data: torch.Tensor,
                                original_sample: dict, idx: int) -> Dict[str, torch.Tensor]:
        """Prepare sample for full model inference."""
        B, T, C, H, W = data.shape

        input_frames = data[:, :-1]  # [B, T-1, C, H, W]
        target_frames = data[:, 1:]  # [B, T-1, C, H, W]

        # No augmentation during inference
        sample = {
            'input': input_frames,
            'target': target_frames,
            'stage': 'full_inference',
            'path': original_sample.get('path', ''),
            'idx': idx
        }

        # Add simulation parameters if available
        if 'simParameters' in original_sample:
            sample['sim_params'] = torch.from_numpy(
                original_sample['simParameters']
            ).float() if isinstance(original_sample['simParameters'], np.ndarray) else original_sample['simParameters']

        return sample

    def _generate_prior_predictions(self, input_frames: torch.Tensor, idx: int) -> torch.Tensor:
        """Generate prior predictions for corrector training."""
        if self.prior_model is None:
            # Use input as prediction (identity) if no prior model available
            logging.warning("No prior model available for corrector training, using identity mapping")
            return input_frames

        try:
            with torch.no_grad():
                self.prior_model.eval()
                device = next(self.prior_model.parameters()).device
                input_frames_gpu = input_frames.to(device)

                # Generate prior predictions
                predictions = self.prior_model(input_frames_gpu)
                predictions_cpu = predictions.cpu()

                # Cache for future use if memory allows
                if self.memory_efficient and len(self.prior_cache) < 1000:
                    self.prior_cache[idx] = predictions_cpu

                return predictions_cpu

        except Exception as e:
            logging.warning(f"Failed to generate prior predictions: {e}")
            return input_frames  # Fallback to identity

    def _apply_augmentations(self, input_data: torch.Tensor,
                           target_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply enhanced augmentations based on strategy."""
        if self.augmentation_strategy == 'conservative':
            return self._apply_conservative_augmentations(input_data, target_data)
        elif self.augmentation_strategy == 'aggressive':
            return self._apply_aggressive_augmentations(input_data, target_data)
        else:
            return self._apply_standard_augmentations(input_data, target_data)

    def _apply_conservative_augmentations(self, input_data: torch.Tensor,
                                        target_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply conservative augmentations (good for shock/turbulence preservation)."""
        probs = self.augmentation_probs['conservative']

        # Horizontal flip (physics preserving)
        if random.random() < probs['flip']:
            input_data = torch.flip(input_data, [-1])  # Flip width dimension
            target_data = torch.flip(target_data, [-1])

        # Small rotation (very conservative)
        if random.random() < probs['rotate']:
            angle = random.uniform(-5, 5)  # Very small angles
            input_data = self._rotate_tensor(input_data, angle)
            target_data = self._rotate_tensor(target_data, angle)

        # Minimal noise
        if random.random() < probs['noise']:
            noise_std = 0.001
            input_data += torch.randn_like(input_data) * noise_std

        return input_data, target_data

    def _apply_standard_augmentations(self, input_data: torch.Tensor,
                                    target_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply standard augmentations."""
        probs = self.augmentation_probs['standard']

        # Horizontal flip
        if random.random() < probs['flip']:
            input_data = torch.flip(input_data, [-1])
            target_data = torch.flip(target_data, [-1])

        # Rotation
        if random.random() < probs['rotate']:
            angle = random.uniform(-10, 10)
            input_data = self._rotate_tensor(input_data, angle)
            target_data = self._rotate_tensor(target_data, angle)

        # Noise injection
        if random.random() < probs['noise']:
            noise_std = 0.005
            input_data += torch.randn_like(input_data) * noise_std

        # Scale perturbation
        if random.random() < probs['scale']:
            scale_factor = random.uniform(0.95, 1.05)
            input_data *= scale_factor
            target_data *= scale_factor

        return input_data, target_data

    def _apply_aggressive_augmentations(self, input_data: torch.Tensor,
                                      target_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply aggressive augmentations (for robust training)."""
        probs = self.augmentation_probs['aggressive']

        # Horizontal flip
        if random.random() < probs['flip']:
            input_data = torch.flip(input_data, [-1])
            target_data = torch.flip(target_data, [-1])

        # Vertical flip (for isotropic data)
        if random.random() < 0.3:
            input_data = torch.flip(input_data, [-2])  # Flip height dimension
            target_data = torch.flip(target_data, [-2])

        # Rotation
        if random.random() < probs['rotate']:
            angle = random.uniform(-15, 15)
            input_data = self._rotate_tensor(input_data, angle)
            target_data = self._rotate_tensor(target_data, angle)

        # Noise injection
        if random.random() < probs['noise']:
            noise_std = 0.01
            input_data += torch.randn_like(input_data) * noise_std

        # Scale perturbation
        if random.random() < probs['scale']:
            scale_factor = random.uniform(0.9, 1.1)
            input_data *= scale_factor
            target_data *= scale_factor

        return input_data, target_data

    def _rotate_tensor(self, tensor: torch.Tensor, angle: float) -> torch.Tensor:
        """Rotate tensor by given angle (simplified implementation)."""
        # This is a placeholder - would need proper rotation implementation
        # For now, return original tensor
        return tensor

    def update_training_stage(self, stage: str, prior_model: Optional[torch.nn.Module] = None):
        """Update training stage and optionally the prior model."""
        self.training_stage = stage
        if prior_model is not None:
            self.prior_model = prior_model

        # Clear cache when changing stages
        if self.prior_cache is not None:
            self.prior_cache.clear()

        logging.info(f"Updated dataset to training stage: {stage}")


def get_data_loader(p_d: DataParams,
                   split: str = 'train',
                   training_stage: str = 'prior_only',
                   prior_model: Optional[torch.nn.Module] = None,
                   **kwargs) -> DataLoader:
    """
    Enhanced data loader factory for generative operator training.

    Args:
        p_d: Data parameters
        split: 'train', 'val', or 'test'
        training_stage: 'prior_only', 'corrector_training', or 'full_inference'
        prior_model: Prior model for corrector training
        **kwargs: Additional arguments

    Returns:
        DataLoader configured for generative operator training
    """

    # Create base turbulence dataset based on dataset name
    if p_d.dataset_name == 'inc':
        base_dataset = create_inc_dataset(p_d, split)
    elif p_d.dataset_name == 'tra':
        base_dataset = create_tra_dataset(p_d, split)
    elif p_d.dataset_name == 'iso':
        base_dataset = create_iso_dataset(p_d, split)
    else:
        raise ValueError(f"Unknown dataset: {p_d.dataset_name}")

    # Determine augmentation strategy based on dataset and split
    if split == 'train':
        if p_d.dataset_name == 'tra':
            augmentation_strategy = 'conservative'  # Shock preservation
        elif p_d.dataset_name == 'iso':
            augmentation_strategy = 'aggressive'    # Turbulence robustness
        else:
            augmentation_strategy = 'standard'      # Inc dataset
    else:
        augmentation_strategy = 'conservative'  # Conservative for validation/test

    # Create enhanced dataset
    enhanced_dataset = GenerativeOperatorDataset(
        base_dataset=base_dataset,
        training_stage=training_stage,
        prior_model=prior_model,
        augmentation_strategy=augmentation_strategy,
        memory_efficient=kwargs.get('memory_efficient', True)
    )

    # Configure sampler
    if split == 'train':
        sampler = RandomSampler(enhanced_dataset)
    else:
        sampler = SequentialSampler(enhanced_dataset)

    # Create data loader
    loader = DataLoader(
        enhanced_dataset,
        sampler=sampler,
        batch_size=p_d.batch_size,
        drop_last=(split == 'train'),
        num_workers=p_d.num_workers,
        pin_memory=p_d.pin_memory,
        prefetch_factor=getattr(p_d, 'prefetch_factor', 2),
        persistent_workers=getattr(p_d, 'persistent_workers', True) if p_d.num_workers > 0 else False
    )

    logging.info(f"Created data loader for {split} split:")
    logging.info(f"  Dataset: {p_d.dataset_name}")
    logging.info(f"  Training stage: {training_stage}")
    logging.info(f"  Batch size: {p_d.batch_size}")
    logging.info(f"  Length: {len(enhanced_dataset)}")
    logging.info(f"  Augmentation strategy: {augmentation_strategy}")

    return loader


def create_inc_dataset(p_d: DataParams, split: str) -> TurbulenceDataset:
    """Create TurbulenceDataset for Inc (incompressible) data."""

    if split == 'train':
        # Training data: broader range
        filter_sim = [(10, 81)]
        filter_frame = [(800, 1300)]
    elif split == 'val':
        # Validation data: specific simulations
        filter_sim = [[82, 84, 86, 88, 90]]
        filter_frame = [(1000, 1150)]
    else:  # test
        # Test data: different range
        filter_sim = [[0, 2, 4, 6, 8]]
        filter_frame = [(1000, 1150)]

    dataset = TurbulenceDataset(
        name=f"Inc_{split}",
        dataDirs=[p_d.data_dir],
        filterTop=["128_inc"],
        filterSim=filter_sim,
        filterFrame=filter_frame,
        sequenceLength=[p_d.input_length, 1],
        randSeqOffset=(split == 'train'),
        simFields=getattr(p_d, 'sim_fields', ["pres"]),
        simParams=getattr(p_d, 'sim_params', ["rey"]),
        printLevel="sim"
    )

    # Apply transforms
    transforms = Transforms(p_d)
    dataset.transform = transforms

    return dataset


def create_tra_dataset(p_d: DataParams, split: str) -> TurbulenceDataset:
    """Create TurbulenceDataset for Tra (transonic) data."""

    if split == 'train':
        # Training data
        filter_sim = [(5, 71)]
        filter_frame = [(500, 1000)]
    elif split == 'val':
        # Validation data
        filter_sim = [[72, 74, 76, 78, 80]]
        filter_frame = [(800, 950)]
    else:  # test
        # Test data
        filter_sim = [[0, 1, 2, 3, 4]]
        filter_frame = [(800, 950)]

    dataset = TurbulenceDataset(
        name=f"Tra_{split}",
        dataDirs=[p_d.data_dir],
        filterTop=["128_tra"],
        filterSim=filter_sim,
        filterFrame=filter_frame,
        sequenceLength=[p_d.input_length, 1],
        randSeqOffset=(split == 'train'),
        simFields=getattr(p_d, 'sim_fields', ["dens", "pres"]),
        simParams=getattr(p_d, 'sim_params', ["rey", "mach"]),
        printLevel="sim"
    )

    # Apply transforms
    transforms = Transforms(p_d)
    dataset.transform = transforms

    return dataset


def create_iso_dataset(p_d: DataParams, split: str) -> TurbulenceDataset:
    """Create TurbulenceDataset for Iso (isotropic turbulence) data."""

    if split == 'train':
        # Training data: large range for turbulence statistics
        filter_sim = [(10, 151)]
        filter_frame = [(200, 800)]
    elif split == 'val':
        # Validation data
        filter_sim = [[152, 154, 156, 158, 160]]
        filter_frame = [(400, 600)]
    else:  # test
        # Test data
        filter_sim = [[0, 2, 4, 6, 8]]
        filter_frame = [(400, 600)]

    dataset = TurbulenceDataset(
        name=f"Iso_{split}",
        dataDirs=[p_d.data_dir],
        filterTop=["64_iso"],
        filterSim=filter_sim,
        filterFrame=filter_frame,
        sequenceLength=[p_d.input_length, 1],
        randSeqOffset=(split == 'train'),
        simFields=getattr(p_d, 'sim_fields', ["velZ"]),  # 3D velocity
        simParams=getattr(p_d, 'sim_params', ["rey"]),
        printLevel="sim"
    )

    # Apply transforms
    transforms = Transforms(p_d)
    dataset.transform = transforms

    return dataset


def create_data_loaders_for_stage(p_d: DataParams,
                                 training_stage: str,
                                 prior_model: Optional[torch.nn.Module] = None) -> Dict[str, DataLoader]:
    """
    Create train/val data loaders for a specific training stage.

    Args:
        p_d: Data parameters
        training_stage: Current training stage
        prior_model: Prior model for corrector training

    Returns:
        Dictionary with 'train' and 'val' data loaders
    """

    train_loader = get_data_loader(
        p_d=p_d,
        split='train',
        training_stage=training_stage,
        prior_model=prior_model
    )

    val_loader = get_data_loader(
        p_d=p_d,
        split='val',
        training_stage=training_stage,
        prior_model=prior_model
    )

    logging.info(f"Created data loaders for {training_stage} stage")
    logging.info(f"  Training batches: {len(train_loader)}")
    logging.info(f"  Validation batches: {len(val_loader)}")

    return {
        'train': train_loader,
        'val': val_loader
    }


def update_data_loaders_stage(data_loaders: Dict[str, DataLoader],
                             new_stage: str,
                             prior_model: Optional[torch.nn.Module] = None):
    """
    Update existing data loaders to a new training stage.

    Args:
        data_loaders: Dictionary of existing data loaders
        new_stage: New training stage
        prior_model: Prior model for corrector training
    """

    for split, loader in data_loaders.items():
        if hasattr(loader.dataset, 'update_training_stage'):
            loader.dataset.update_training_stage(new_stage, prior_model)

    logging.info(f"Updated data loaders to stage: {new_stage}")