"""
DeepONet Data Processing and Loading Utilities

Utilities for preprocessing data for operator learning, including sensor sampling,
query point generation, and batch processing for DeepONet training.
"""

from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader, Sampler
import logging


@dataclass
class DeepONetDataConfig:
    """Configuration for DeepONet data processing."""

    # Sensor configuration
    n_sensors: int = 100
    sensor_strategy: str = "uniform"  # "uniform", "random", "adaptive", "grid"
    sensor_locations: Optional[torch.Tensor] = None

    # Query point configuration
    n_query_train: int = 1000
    n_query_eval: int = None  # Use full grid if None
    query_strategy: str = "random"  # "random", "uniform", "grid"

    # Data processing
    normalize_inputs: bool = True
    normalize_outputs: bool = True
    subsample_timesteps: Optional[int] = None

    # Augmentation
    add_noise: bool = False
    noise_level: float = 0.01
    random_sensor_dropout: bool = False
    sensor_dropout_rate: float = 0.1


class OperatorDataset(Dataset):
    """
    Dataset class for operator learning with DeepONet.

    Handles conversion of sequential/spatial data to function-operator pairs
    suitable for DeepONet training.
    """

    def __init__(self,
                 input_functions: torch.Tensor,
                 output_functions: torch.Tensor,
                 config: DeepONetDataConfig,
                 spatial_coords: Optional[torch.Tensor] = None):
        """
        Initialize operator dataset.

        Args:
            input_functions: Input function data [N, T, C, H, W]
            output_functions: Target output data [N, T, C, H, W]
            config: Data configuration
            spatial_coords: Optional spatial coordinates [H, W, 2]
        """
        self.input_functions = input_functions
        self.output_functions = output_functions
        self.config = config

        N, T, C, H, W = input_functions.shape
        self.N, self.T, self.C, self.H, self.W = N, T, C, H, W

        # Generate spatial coordinates if not provided
        if spatial_coords is None:
            h_coords = torch.linspace(0, 1, H)
            w_coords = torch.linspace(0, 1, W)
            hh, ww = torch.meshgrid(h_coords, w_coords, indexing='ij')
            self.spatial_coords = torch.stack([hh, ww], dim=-1)  # [H, W, 2]
        else:
            self.spatial_coords = spatial_coords

        # Initialize sensors
        self.sensor_locations = self._initialize_sensors()

        # Compute normalization statistics
        if config.normalize_inputs or config.normalize_outputs:
            self._compute_normalization_stats()

        logging.info(f"OperatorDataset initialized: {N} samples, {T} timesteps, "
                    f"{config.n_sensors} sensors, {H}×{W} spatial resolution")

    def _initialize_sensors(self) -> torch.Tensor:
        """Initialize sensor locations based on strategy."""
        if self.config.sensor_locations is not None:
            return self.config.sensor_locations.clone()

        if self.config.sensor_strategy == "uniform":
            return self._uniform_sensors()
        elif self.config.sensor_strategy == "random":
            return self._random_sensors()
        elif self.config.sensor_strategy == "grid":
            return self._grid_sensors()
        elif self.config.sensor_strategy == "adaptive":
            return self._adaptive_sensors()
        else:
            raise ValueError(f"Unknown sensor strategy: {self.config.sensor_strategy}")

    def _uniform_sensors(self) -> torch.Tensor:
        """Generate uniformly distributed sensor locations."""
        n_h = int(np.sqrt(self.config.n_sensors * self.H / self.W))
        n_w = int(np.sqrt(self.config.n_sensors * self.W / self.H))

        h_coords = torch.linspace(0, 1, n_h)
        w_coords = torch.linspace(0, 1, n_w)

        hh, ww = torch.meshgrid(h_coords, w_coords, indexing='ij')
        sensors = torch.stack([hh.flatten(), ww.flatten()], dim=1)

        # Trim to exact number
        return sensors[:self.config.n_sensors]

    def _random_sensors(self) -> torch.Tensor:
        """Generate random sensor locations."""
        return torch.rand(self.config.n_sensors, 2)

    def _grid_sensors(self) -> torch.Tensor:
        """Generate regular grid sensor locations."""
        n_side = int(np.ceil(np.sqrt(self.config.n_sensors)))
        coords = torch.linspace(0, 1, n_side)
        hh, ww = torch.meshgrid(coords, coords, indexing='ij')
        sensors = torch.stack([hh.flatten(), ww.flatten()], dim=1)
        return sensors[:self.config.n_sensors]

    def _adaptive_sensors(self) -> torch.Tensor:
        """Generate adaptive sensor locations based on data variance."""
        # Compute spatial variance across all samples and timesteps
        variance_map = torch.var(self.input_functions, dim=(0, 1, 2))  # [H, W]

        # Flatten and get indices of highest variance locations
        flat_variance = variance_map.flatten()
        _, top_indices = torch.topk(flat_variance, self.config.n_sensors)

        # Convert flat indices to 2D coordinates
        h_indices = top_indices // self.W
        w_indices = top_indices % self.W

        # Convert to normalized coordinates [0, 1]
        h_coords = h_indices.float() / (self.H - 1)
        w_coords = w_indices.float() / (self.W - 1)

        return torch.stack([h_coords, w_coords], dim=1)

    def _compute_normalization_stats(self):
        """Compute normalization statistics for inputs and outputs."""
        if self.config.normalize_inputs:
            self.input_mean = torch.mean(self.input_functions)
            self.input_std = torch.std(self.input_functions)

        if self.config.normalize_outputs:
            self.output_mean = torch.mean(self.output_functions)
            self.output_std = torch.std(self.output_functions)

    def _sample_at_sensors(self, functions: torch.Tensor) -> torch.Tensor:
        """
        Sample functions at sensor locations.

        Args:
            functions: Function data [T, C, H, W]

        Returns:
            sensor_values: Values at sensors [T, n_sensors, C]
        """
        T, C, H, W = functions.shape

        # Convert normalized coordinates to pixel indices
        h_indices = (self.sensor_locations[:, 0] * (H - 1)).long()
        w_indices = (self.sensor_locations[:, 1] * (W - 1)).long()

        # Clamp to valid range
        h_indices = torch.clamp(h_indices, 0, H - 1)
        w_indices = torch.clamp(w_indices, 0, W - 1)

        # Sample at sensor locations
        sensor_values = functions[:, :, h_indices, w_indices]  # [T, C, n_sensors]
        sensor_values = sensor_values.permute(0, 2, 1)  # [T, n_sensors, C]

        return sensor_values

    def _generate_query_points(self, n_query: int) -> torch.Tensor:
        """
        Generate query points for training/evaluation.

        Args:
            n_query: Number of query points to generate

        Returns:
            query_coords: Query coordinates [n_query, 2]
        """
        if self.config.query_strategy == "random":
            return torch.rand(n_query, 2)
        elif self.config.query_strategy == "uniform":
            n_side = int(np.ceil(np.sqrt(n_query)))
            coords = torch.linspace(0, 1, n_side)
            hh, ww = torch.meshgrid(coords, coords, indexing='ij')
            query_coords = torch.stack([hh.flatten(), ww.flatten()], dim=1)
            return query_coords[:n_query]
        elif self.config.query_strategy == "grid":
            # Use full spatial grid
            coords = self.spatial_coords.view(-1, 2)  # [H*W, 2]
            if n_query >= coords.shape[0]:
                return coords
            else:
                # Subsample grid points
                indices = torch.randperm(coords.shape[0])[:n_query]
                return coords[indices]
        else:
            raise ValueError(f"Unknown query strategy: {self.config.query_strategy}")

    def _interpolate_at_queries(self,
                               functions: torch.Tensor,
                               query_coords: torch.Tensor) -> torch.Tensor:
        """
        Interpolate function values at query coordinates.

        Args:
            functions: Function data [T, C, H, W]
            query_coords: Query coordinates [n_query, 2]

        Returns:
            query_values: Interpolated values [T, n_query, C]
        """
        T, C, H, W = functions.shape
        n_query = query_coords.shape[0]

        # Convert normalized coordinates to grid coordinates
        # PyTorch grid_sample expects coordinates in [-1, 1]
        grid_coords = query_coords * 2.0 - 1.0  # [0, 1] -> [-1, 1]

        # Reshape for grid_sample: [1, n_query, 1, 2]
        grid = grid_coords.view(1, n_query, 1, 2)

        # Interpolate for each timestep
        query_values = []
        for t in range(T):
            # grid_sample expects [N, C, H, W] input
            func_t = functions[t:t+1]  # [1, C, H, W]

            # Interpolate: [1, C, n_query, 1]
            interp_t = F.grid_sample(func_t, grid,
                                   mode='bilinear',
                                   padding_mode='border',
                                   align_corners=True)

            # Reshape to [C, n_query] then to [n_query, C]
            interp_t = interp_t.squeeze(0).squeeze(-1).transpose(0, 1)
            query_values.append(interp_t)

        return torch.stack(query_values, dim=0)  # [T, n_query, C]

    def __len__(self) -> int:
        return self.N

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single operator learning sample.

        Args:
            idx: Sample index

        Returns:
            sample: Dictionary containing:
                - input_sensors: Input function at sensors [T, n_sensors, C]
                - query_coords: Query coordinates [n_query, 2]
                - target_values: Target values at queries [T, n_query, C]
                - metadata: Additional information
        """
        # Get input and output functions
        input_func = self.input_functions[idx]  # [T, C, H, W]
        output_func = self.output_functions[idx]  # [T, C, H, W]

        # Apply normalization
        if self.config.normalize_inputs:
            input_func = (input_func - self.input_mean) / (self.input_std + 1e-8)
        if self.config.normalize_outputs:
            output_func = (output_func - self.output_mean) / (self.output_std + 1e-8)

        # Add noise if configured
        if self.config.add_noise and self.training:
            noise = torch.randn_like(input_func) * self.config.noise_level
            input_func = input_func + noise

        # Sample input function at sensors
        input_sensors = self._sample_at_sensors(input_func)  # [T, n_sensors, C]

        # Apply sensor dropout if configured
        if (self.config.random_sensor_dropout and
            self.training and
            torch.rand(1).item() < self.config.sensor_dropout_rate):
            n_keep = int(self.config.n_sensors * (1 - self.config.sensor_dropout_rate))
            keep_indices = torch.randperm(self.config.n_sensors)[:n_keep]
            input_sensors = input_sensors[:, keep_indices]

        # Generate query points
        n_query = (self.config.n_query_train if self.training
                  else (self.config.n_query_eval or self.H * self.W))
        query_coords = self._generate_query_points(n_query)

        # Get target values at query points
        target_values = self._interpolate_at_queries(output_func, query_coords)

        # Subsample timesteps if configured
        if self.config.subsample_timesteps is not None and self.config.subsample_timesteps < self.T:
            t_indices = torch.randperm(self.T)[:self.config.subsample_timesteps]
            t_indices = t_indices.sort()[0]  # Keep temporal order
            input_sensors = input_sensors[t_indices]
            target_values = target_values[t_indices]

        return {
            'input_sensors': input_sensors,
            'query_coords': query_coords,
            'target_values': target_values,
            'sensor_locations': self.sensor_locations,
            'sample_idx': idx,
            'metadata': {
                'input_shape': input_func.shape,
                'n_sensors': input_sensors.shape[1],
                'n_queries': query_coords.shape[0]
            }
        }


class OperatorDataLoader:
    """
    Specialized data loader for DeepONet training with operator data.

    Handles batching of variable-size sensor and query data.
    """

    def __init__(self,
                 dataset: OperatorDataset,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 num_workers: int = 0,
                 pin_memory: bool = True,
                 collate_fn: Optional[Callable] = None):

        self.dataset = dataset
        self.batch_size = batch_size

        # Use custom collate function if not provided
        if collate_fn is None:
            collate_fn = self._collate_operator_batch

        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn
        )

    def _collate_operator_batch(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate operator learning samples into batch.

        Args:
            batch: List of samples from dataset

        Returns:
            batched_data: Batched tensors
        """
        # Stack tensors that have consistent shapes within batch
        input_sensors = torch.stack([sample['input_sensors'] for sample in batch])
        query_coords = torch.stack([sample['query_coords'] for sample in batch])
        target_values = torch.stack([sample['target_values'] for sample in batch])
        sensor_locations = torch.stack([sample['sensor_locations'] for sample in batch])
        sample_indices = torch.tensor([sample['sample_idx'] for sample in batch])

        # Collect metadata
        metadata = {
            'input_shapes': [sample['metadata']['input_shape'] for sample in batch],
            'n_sensors': [sample['metadata']['n_sensors'] for sample in batch],
            'n_queries': [sample['metadata']['n_queries'] for sample in batch]
        }

        return {
            'input_sensors': input_sensors,      # [B, T, n_sensors, C]
            'query_coords': query_coords,        # [B, n_query, 2]
            'target_values': target_values,      # [B, T, n_query, C]
            'sensor_locations': sensor_locations, # [B, n_sensors, 2]
            'sample_indices': sample_indices,     # [B]
            'metadata': metadata
        }

    def __iter__(self):
        return iter(self.dataloader)

    def __len__(self):
        return len(self.dataloader)


def create_operator_dataloaders(
    train_input: torch.Tensor,
    train_output: torch.Tensor,
    val_input: Optional[torch.Tensor] = None,
    val_output: Optional[torch.Tensor] = None,
    test_input: Optional[torch.Tensor] = None,
    test_output: Optional[torch.Tensor] = None,
    config: Optional[DeepONetDataConfig] = None,
    batch_size: int = 32,
    num_workers: int = 0
) -> Dict[str, OperatorDataLoader]:
    """
    Create train/validation/test dataloaders for operator learning.

    Args:
        train_input: Training input functions [N, T, C, H, W]
        train_output: Training output functions [N, T, C, H, W]
        val_input: Optional validation input functions
        val_output: Optional validation output functions
        test_input: Optional test input functions
        test_output: Optional test output functions
        config: Data configuration
        batch_size: Batch size for training
        num_workers: Number of data loading workers

    Returns:
        dataloaders: Dictionary of dataloaders
    """
    if config is None:
        config = DeepONetDataConfig()

    dataloaders = {}

    # Training dataset
    train_dataset = OperatorDataset(train_input, train_output, config)
    train_dataset.training = True
    dataloaders['train'] = OperatorDataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    # Validation dataset
    if val_input is not None and val_output is not None:
        val_config = DeepONetDataConfig(**config.__dict__)
        val_config.n_query_train = config.n_query_eval or (train_input.shape[-2] * train_input.shape[-1])

        val_dataset = OperatorDataset(val_input, val_output, val_config)
        val_dataset.training = False
        dataloaders['val'] = OperatorDataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )

    # Test dataset
    if test_input is not None and test_output is not None:
        test_config = DeepONetDataConfig(**config.__dict__)
        test_config.n_query_train = config.n_query_eval or (train_input.shape[-2] * train_input.shape[-1])

        test_dataset = OperatorDataset(test_input, test_output, test_config)
        test_dataset.training = False
        dataloaders['test'] = OperatorDataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )

    logging.info(f"Created dataloaders: {list(dataloaders.keys())}")
    return dataloaders


class AdaptiveSensorSampler:
    """
    Adaptive sensor placement based on information theory and data variance.

    Updates sensor locations during training to maximize information gain.
    """

    def __init__(self,
                 initial_sensors: torch.Tensor,
                 update_frequency: int = 100,
                 learning_rate: float = 0.01):
        """
        Initialize adaptive sensor sampler.

        Args:
            initial_sensors: Initial sensor locations [n_sensors, 2]
            update_frequency: How often to update sensors (in batches)
            learning_rate: Learning rate for sensor position updates
        """
        self.sensors = initial_sensors.clone().requires_grad_(True)
        self.update_frequency = update_frequency
        self.learning_rate = learning_rate
        self.batch_count = 0
        self.information_history = []

    def compute_information_gain(self,
                                model: torch.nn.Module,
                                input_functions: torch.Tensor,
                                query_coords: torch.Tensor) -> torch.Tensor:
        """
        Compute information gain from current sensor placement.

        Args:
            model: DeepONet model
            input_functions: Input function data [B, T, C, H, W]
            query_coords: Query coordinates [B, n_query, 2]

        Returns:
            information_gain: Scalar information gain measure
        """
        # Sample functions at current sensor locations
        B, T, C, H, W = input_functions.shape

        # Convert sensor coordinates to indices
        h_indices = (self.sensors[:, 0] * (H - 1)).long()
        w_indices = (self.sensors[:, 1] * (W - 1)).long()

        h_indices = torch.clamp(h_indices, 0, H - 1)
        w_indices = torch.clamp(w_indices, 0, W - 1)

        # Extract sensor values
        sensor_values = []
        for b in range(B):
            for t in range(T):
                for c in range(C):
                    vals = input_functions[b, t, c, h_indices, w_indices]
                    sensor_values.append(vals)

        sensor_tensor = torch.stack(sensor_values).view(B, T, C, -1)
        sensor_tensor = sensor_tensor.permute(0, 1, 3, 2)  # [B, T, n_sensors, C]

        # Get model predictions
        with torch.no_grad():
            predictions = model(input_functions, query_coords)

        # Compute information-theoretic measure (mutual information approximation)
        # Use variance of predictions as proxy for information content
        info_gain = torch.var(predictions)

        return info_gain

    def update_sensors(self,
                      model: torch.nn.Module,
                      input_functions: torch.Tensor,
                      query_coords: torch.Tensor):
        """
        Update sensor locations to maximize information gain.

        Args:
            model: DeepONet model
            input_functions: Input function data
            query_coords: Query coordinates
        """
        self.batch_count += 1

        if self.batch_count % self.update_frequency != 0:
            return

        # Compute information gain
        info_gain = self.compute_information_gain(model, input_functions, query_coords)

        # Compute gradients w.r.t. sensor positions
        info_gain.backward()

        # Update sensor positions
        with torch.no_grad():
            if self.sensors.grad is not None:
                self.sensors += self.learning_rate * self.sensors.grad

                # Clamp to valid domain [0, 1]
                self.sensors.clamp_(0, 1)

                # Clear gradients
                self.sensors.grad.zero_()

        # Store information gain
        self.information_history.append(info_gain.item())

        logging.debug(f"Updated sensors at batch {self.batch_count}, "
                     f"info gain: {info_gain.item():.6f}")

    def get_current_sensors(self) -> torch.Tensor:
        """Get current sensor locations."""
        return self.sensors.detach().clone()

    def get_information_history(self) -> List[float]:
        """Get history of information gains."""
        return self.information_history.copy()