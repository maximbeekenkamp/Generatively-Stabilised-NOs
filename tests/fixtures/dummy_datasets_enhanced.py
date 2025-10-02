"""
Enhanced Dummy datasets for testing all model variants

Creates lightweight test data that mimics the structure of the real datasets
but with small dimensions for fast testing. Includes sophisticated patterns,
physical properties, and comprehensive dataset validation.
"""

import torch
import numpy as np
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any
from torch.utils.data import Dataset
from dataclasses import dataclass
from enum import Enum


class FieldType(Enum):
    """Types of physical fields"""
    VELOCITY = "velocity"
    PRESSURE = "pressure"
    DENSITY = "density"
    TEMPERATURE = "temperature"
    VORTICITY = "vorticity"


@dataclass
class DatasetConfig:
    """Configuration for dummy dataset generation"""
    dataset_type: str
    num_samples: int = 5
    spatial_size: Union[int, Tuple[int, int]] = 16
    sequence_length: int = 10
    num_channels: int = 3
    field_types: List[str] = None
    reynolds_number: float = 1000.0
    mach_number: float = 0.3
    add_noise: bool = True
    noise_level: float = 0.1
    temporal_evolution: bool = True
    physical_constraints: bool = True

    def __post_init__(self):
        if self.field_types is None:
            self.field_types = ["velocity", "pressure", "density"][:self.num_channels]

        # Ensure spatial_size is tuple
        if isinstance(self.spatial_size, int):
            self.spatial_size = (self.spatial_size, self.spatial_size)


class PhysicalPatternGenerator:
    """Generates physically realistic patterns for different flow types"""

    @staticmethod
    def generate_incompressible_flow(H: int, W: int, re_number: float = 1000.0) -> Dict[str, torch.Tensor]:
        """Generate incompressible flow patterns"""
        x = torch.linspace(-2, 2, W)
        y = torch.linspace(-2, 2, H)
        X, Y = torch.meshgrid(x, y, indexing='ij')

        # Kelvin-Helmholtz like instability
        u_velocity = torch.tanh(Y) + 0.1 * torch.sin(2 * np.pi * X) * torch.exp(-0.5 * Y**2)
        v_velocity = 0.1 * torch.cos(2 * np.pi * X) * torch.exp(-0.5 * Y**2)

        # Pressure from incompressibility (simplified)
        pressure = -0.5 * (u_velocity**2 + v_velocity**2)

        return {
            "velocity_u": u_velocity,
            "velocity_v": v_velocity,
            "pressure": pressure
        }

    @staticmethod
    def generate_transonic_flow(H: int, W: int, mach: float = 0.8) -> Dict[str, torch.Tensor]:
        """Generate transonic flow with shock-like features"""
        x = torch.linspace(0, 4, W)
        y = torch.linspace(-1, 1, H)
        X, Y = torch.meshgrid(x, y, indexing='ij')

        # Shock wave at x = 2
        shock_pos = W // 2
        u_velocity = torch.ones_like(X) * mach
        u_velocity[:, shock_pos:] *= 0.6  # Post-shock velocity drop

        # Add expansion fan
        expansion_region = (X > 1) & (X < 2) & (torch.abs(Y) < 0.5)
        u_velocity[expansion_region] *= 1.2

        v_velocity = 0.1 * torch.sin(np.pi * Y) * torch.exp(-0.5 * (X - 2)**2)

        # Pressure jump across shock
        pressure = torch.ones_like(X)
        pressure[:, shock_pos:] *= 1.8

        return {
            "velocity_u": u_velocity,
            "velocity_v": v_velocity,
            "pressure": pressure
        }

    @staticmethod
    def generate_isotropic_turbulence(H: int, W: int, energy_spectrum: str = "kolmogorov") -> Dict[str, torch.Tensor]:
        """Generate isotropic turbulence using spectral methods"""
        # Generate random phases
        kx = torch.fft.fftfreq(W, d=1.0).unsqueeze(0).repeat(H, 1)
        ky = torch.fft.fftfreq(H, d=1.0).unsqueeze(1).repeat(1, W)
        k_mag = torch.sqrt(kx**2 + ky**2) + 1e-8

        # Kolmogorov spectrum: E(k) ~ k^(-5/3)
        if energy_spectrum == "kolmogorov":
            energy_spectrum_2d = k_mag**(-5/3)
        else:
            energy_spectrum_2d = torch.exp(-0.5 * k_mag**2)

        # Generate velocity components
        phase_u = torch.rand(H, W) * 2 * np.pi
        phase_v = torch.rand(H, W) * 2 * np.pi

        u_hat = torch.sqrt(energy_spectrum_2d) * torch.exp(1j * phase_u)
        v_hat = torch.sqrt(energy_spectrum_2d) * torch.exp(1j * phase_v)

        # Ensure incompressibility: k · u = 0
        dot_product = kx * u_hat + ky * v_hat
        u_hat -= kx * dot_product / (k_mag**2 + 1e-8)
        v_hat -= ky * dot_product / (k_mag**2 + 1e-8)

        u_velocity = torch.real(torch.fft.ifft2(u_hat))
        v_velocity = torch.real(torch.fft.ifft2(v_hat))

        # Pressure from Poisson equation (simplified)
        pressure = -0.5 * (u_velocity**2 + v_velocity**2)

        return {
            "velocity_u": u_velocity,
            "velocity_v": v_velocity,
            "pressure": pressure
        }


class DummyTurbulenceDataset(Dataset):
    """
    Enhanced lightweight dataset for testing purposes.

    Mimics the real TurbulenceDataset but with small dimensions and
    sophisticated physical patterns for comprehensive testing.
    """

    def __init__(self, config: DatasetConfig):
        self.config = config
        self.pattern_generator = PhysicalPatternGenerator()

        # Validate configuration
        self._validate_config()

        # Generate dummy data with different characteristics per dataset
        self.data = self._generate_dummy_data()

        # Apply physical constraints if requested
        if self.config.physical_constraints:
            self._apply_physical_constraints()

    def _validate_config(self):
        """Validate dataset configuration"""
        if self.config.num_samples <= 0:
            raise ValueError("num_samples must be positive")

        if self.config.sequence_length <= 0:
            raise ValueError("sequence_length must be positive")

        if any(s <= 0 for s in self.config.spatial_size):
            raise ValueError("spatial_size must be positive")

        if self.config.num_channels != len(self.config.field_types):
            warnings.warn(f"num_channels ({self.config.num_channels}) != len(field_types) ({len(self.config.field_types)})")

    def _apply_physical_constraints(self):
        """Apply physical constraints like mass conservation"""
        # Ensure reasonable bounds
        self.data = torch.clamp(self.data, -10.0, 10.0)

        # Apply conservation constraints for incompressible flows
        if "inc" in self.config.dataset_type and self.config.num_channels >= 2:
            # Smooth velocity fields to satisfy incompressibility approximately
            self.data[:, :, :2] = self._apply_smoothing(self.data[:, :, :2])

    def _apply_smoothing(self, fields: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
        """Apply smoothing to enforce physical constraints"""
        import torch.nn.functional as F

        # Simple 2D smoothing
        kernel = torch.ones(1, 1, kernel_size, kernel_size) / (kernel_size**2)

        # fields shape: [num_samples, sequence_length, num_channels, H, W]
        N, T, C, H, W = fields.shape

        # Reshape to process all samples and timesteps together
        # Reshape to [N*T*C, 1, H, W] for conv2d
        fields_reshaped = fields.reshape(N * T * C, 1, H, W)

        # Apply smoothing
        smoothed = F.conv2d(fields_reshaped, kernel, padding=kernel_size//2)

        # Reshape back to original shape
        smoothed = smoothed.reshape(N, T, C, H, W)

        return smoothed

    def _generate_dummy_data(self) -> torch.Tensor:
        """Generate dummy data with sophisticated dataset-specific characteristics"""
        H, W = self.config.spatial_size

        # Initialize data tensor
        data = torch.zeros(
            self.config.num_samples,
            self.config.sequence_length,
            self.config.num_channels,
            H, W
        )

        # Generate base patterns for each sample
        for i in range(self.config.num_samples):
            base_patterns = self._generate_base_pattern(H, W)

            # Apply temporal evolution
            for t in range(self.config.sequence_length):
                time_factor = t / max(1, self.config.sequence_length - 1)

                # Evolve patterns over time
                if self.config.temporal_evolution:
                    evolved_patterns = self._evolve_pattern(base_patterns, time_factor)
                else:
                    evolved_patterns = base_patterns

                # Assign to channels based on field types
                for c, field_type in enumerate(self.config.field_types[:self.config.num_channels]):
                    if field_type in ["velocity", "velocity_u"]:
                        data[i, t, c] = evolved_patterns.get("velocity_u", evolved_patterns.get("velocity", torch.zeros(H, W)))
                    elif field_type == "velocity_v":
                        data[i, t, c] = evolved_patterns.get("velocity_v", torch.zeros(H, W))
                    elif field_type == "pressure":
                        data[i, t, c] = evolved_patterns.get("pressure", torch.zeros(H, W))
                    elif field_type == "density":
                        data[i, t, c] = evolved_patterns.get("density", torch.ones(H, W))
                    elif field_type == "temperature":
                        data[i, t, c] = evolved_patterns.get("temperature", torch.ones(H, W))
                    else:
                        # Default to velocity-like field
                        data[i, t, c] = evolved_patterns.get("velocity_u", torch.randn(H, W))

        # Add noise if requested
        if self.config.add_noise:
            noise = torch.randn_like(data) * self.config.noise_level
            data += noise

        return data.float()

    def _generate_base_pattern(self, H: int, W: int) -> Dict[str, torch.Tensor]:
        """Generate base physical patterns based on dataset type"""
        if "inc" in self.config.dataset_type:
            # Incompressible flow patterns
            patterns = self.pattern_generator.generate_incompressible_flow(
                H, W, self.config.reynolds_number
            )
        elif "tra" in self.config.dataset_type:
            # Transonic flow patterns
            patterns = self.pattern_generator.generate_transonic_flow(
                H, W, self.config.mach_number
            )
        elif "iso" in self.config.dataset_type:
            # Isotropic turbulence patterns
            patterns = self.pattern_generator.generate_isotropic_turbulence(H, W)
        else:
            # Default: simple flow patterns
            x = torch.linspace(-1, 1, W)
            y = torch.linspace(-1, 1, H)
            X, Y = torch.meshgrid(x, y, indexing='ij')
            patterns = {
                "velocity_u": torch.sin(X * np.pi) * torch.cos(Y * np.pi),
                "velocity_v": torch.cos(X * np.pi) * torch.sin(Y * np.pi),
                "pressure": torch.sin(2 * X * np.pi) * torch.sin(2 * Y * np.pi)
            }

        # Add density and temperature fields
        if "density" not in patterns:
            patterns["density"] = torch.ones(H, W) + 0.1 * torch.randn(H, W)

        if "temperature" not in patterns:
            patterns["temperature"] = torch.ones(H, W) + 0.1 * torch.randn(H, W)

        return patterns

    def _evolve_pattern(self, base_patterns: Dict[str, torch.Tensor], time_factor: float) -> Dict[str, torch.Tensor]:
        """Evolve patterns over time with simple physics"""
        evolved = {}

        for key, pattern in base_patterns.items():
            if "velocity" in key:
                # Add temporal oscillation and advection
                phase = 2 * np.pi * time_factor
                evolved[key] = pattern * (0.8 + 0.2 * torch.cos(torch.tensor(phase)))

                # Add simple advection (shift pattern)
                if "u" in key:
                    shift = int(time_factor * 2)  # Shift by up to 2 pixels
                    evolved[key] = torch.roll(pattern, shift, dims=1)
                elif "v" in key:
                    shift = int(time_factor * 1)  # Shift by up to 1 pixel
                    evolved[key] = torch.roll(pattern, shift, dims=0)
            else:
                # Other fields evolve more slowly
                phase = np.pi * time_factor
                evolved[key] = pattern * (0.9 + 0.1 * torch.sin(torch.tensor(phase)))

        return evolved

    def __len__(self) -> int:
        return self.config.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns (input_sequence, target_sequence) pair
        Input: first frames, Target: remaining frames (can be None for some models)
        """
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")

        sequence = self.data[idx]

        # Adaptive split based on sequence length
        if self.config.sequence_length <= 2:
            # Very short sequences: use all as input, duplicate as target
            return sequence, sequence
        elif self.config.sequence_length <= 4:
            # Short sequences: split in half
            mid = self.config.sequence_length // 2
            return sequence[:mid], sequence[mid:]
        else:
            # Longer sequences: use 80% as input, 20% as target
            split_point = int(0.8 * self.config.sequence_length)
            return sequence[:split_point], sequence[split_point:]

    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        """Get detailed information about a sample"""
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range")

        input_seq, target_seq = self[idx]

        return {
            "dataset_type": self.config.dataset_type,
            "sample_index": idx,
            "input_shape": input_seq.shape,
            "target_shape": target_seq.shape if target_seq is not None else None,
            "field_types": self.config.field_types,
            "reynolds_number": self.config.reynolds_number,
            "mach_number": self.config.mach_number,
            "spatial_resolution": self.config.spatial_size,
            "temporal_length": self.config.sequence_length,
            "data_statistics": {
                "input_mean": torch.mean(input_seq).item(),
                "input_std": torch.std(input_seq).item(),
                "input_min": torch.min(input_seq).item(),
                "input_max": torch.max(input_seq).item(),
            }
        }

    def validate_physics(self) -> Dict[str, bool]:
        """Validate physical properties of generated data"""
        results = {}

        # Check for NaN or Inf values
        results["finite_values"] = torch.all(torch.isfinite(self.data))

        # Check reasonable bounds
        data_max = torch.max(torch.abs(self.data))
        results["reasonable_bounds"] = data_max < 100.0

        # Check for temporal continuity
        if self.config.sequence_length > 1:
            temporal_diff = torch.abs(self.data[:, 1:] - self.data[:, :-1])
            max_temporal_change = torch.max(temporal_diff)
            results["temporal_continuity"] = max_temporal_change < 10.0
        else:
            results["temporal_continuity"] = True

        # Check spatial smoothness
        spatial_grad_x = torch.abs(self.data[:, :, :, 1:, :] - self.data[:, :, :, :-1, :])
        spatial_grad_y = torch.abs(self.data[:, :, :, :, 1:] - self.data[:, :, :, :, :-1])
        max_spatial_grad = max(torch.max(spatial_grad_x), torch.max(spatial_grad_y))
        results["spatial_smoothness"] = max_spatial_grad < 10.0

        return results


class DummyDatasetFactory:
    """Enhanced factory for creating dummy datasets for testing"""

    DEFAULT_CONFIGS = {
        'inc_low': DatasetConfig(
            dataset_type='inc_low',
            num_samples=5,
            spatial_size=(16, 16),
            sequence_length=10,
            num_channels=3,
            field_types=["velocity_u", "velocity_v", "pressure"],
            reynolds_number=100.0,
            add_noise=True,
            noise_level=0.05
        ),
        'inc_high': DatasetConfig(
            dataset_type='inc_high',
            num_samples=5,
            spatial_size=(16, 16),
            sequence_length=10,
            num_channels=3,
            field_types=["velocity_u", "velocity_v", "pressure"],
            reynolds_number=10000.0,
            add_noise=True,
            noise_level=0.1
        ),
        'tra_ext': DatasetConfig(
            dataset_type='tra_ext',
            num_samples=5,
            spatial_size=(16, 16),
            sequence_length=10,
            num_channels=3,
            field_types=["velocity_u", "velocity_v", "pressure"],
            mach_number=0.8,
            add_noise=True,
            noise_level=0.08
        ),
        'tra_inc': DatasetConfig(
            dataset_type='tra_inc',
            num_samples=5,
            spatial_size=(16, 16),
            sequence_length=10,
            num_channels=3,
            field_types=["velocity_u", "velocity_v", "pressure"],
            mach_number=0.3,
            add_noise=True,
            noise_level=0.06
        ),
        'iso': DatasetConfig(
            dataset_type='iso',
            num_samples=5,
            spatial_size=(16, 16),
            sequence_length=10,
            num_channels=3,
            field_types=["velocity_u", "velocity_v", "pressure"],
            add_noise=True,
            noise_level=0.15,
            temporal_evolution=True
        )
    }

    @classmethod
    def create_dataset(cls, dataset_name: str, **overrides) -> DummyTurbulenceDataset:
        """Create a dummy dataset by name with optional parameter overrides"""
        if dataset_name not in cls.DEFAULT_CONFIGS:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(cls.DEFAULT_CONFIGS.keys())}")

        config = cls.DEFAULT_CONFIGS[dataset_name]

        # Apply overrides
        if overrides:
            # Create new config with overrides
            config_dict = {
                "dataset_type": config.dataset_type,
                "num_samples": config.num_samples,
                "spatial_size": config.spatial_size,
                "sequence_length": config.sequence_length,
                "num_channels": config.num_channels,
                "field_types": config.field_types.copy(),
                "reynolds_number": config.reynolds_number,
                "mach_number": config.mach_number,
                "add_noise": config.add_noise,
                "noise_level": config.noise_level,
                "temporal_evolution": config.temporal_evolution,
                "physical_constraints": config.physical_constraints
            }
            config_dict.update(overrides)
            config = DatasetConfig(**config_dict)

        return DummyTurbulenceDataset(config)

    @classmethod
    def create_custom_dataset(cls, config: DatasetConfig) -> DummyTurbulenceDataset:
        """Create a dataset with custom configuration"""
        return DummyTurbulenceDataset(config)

    @classmethod
    def create_for_model_testing(cls,
                               model_type: str,
                               dataset_name: str = "inc_low",
                               **kwargs) -> DummyTurbulenceDataset:
        """Create dataset optimized for specific model testing"""

        # Model-specific optimizations
        model_configs = {
            "fno": {"spatial_size": (32, 32), "temporal_evolution": True},
            "tno": {"sequence_length": 16, "temporal_evolution": True},
            "unet": {"spatial_size": (32, 32), "num_channels": 4},
            "resnet": {"spatial_size": (32, 32), "sequence_length": 8},
            "transformer": {"sequence_length": 12, "num_channels": 4},
            "diffusion": {"add_noise": True, "noise_level": 0.2},
            "refiner": {"physical_constraints": True, "temporal_evolution": True}
        }

        overrides = model_configs.get(model_type.lower(), {})
        overrides.update(kwargs)

        return cls.create_dataset(dataset_name, **overrides)

    @classmethod
    def create_all_datasets(cls, **overrides) -> Dict[str, DummyTurbulenceDataset]:
        """Create all dummy datasets with optional overrides"""
        return {name: cls.create_dataset(name, **overrides) for name in cls.DEFAULT_CONFIGS.keys()}

    @classmethod
    def get_available_datasets(cls) -> List[str]:
        """Get list of available dataset names"""
        return list(cls.DEFAULT_CONFIGS.keys())

    @classmethod
    def get_dataset_info(cls, dataset_name: str) -> Dict[str, Any]:
        """Get information about a dataset configuration"""
        if dataset_name not in cls.DEFAULT_CONFIGS:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        config = cls.DEFAULT_CONFIGS[dataset_name]
        return {
            "dataset_type": config.dataset_type,
            "num_samples": config.num_samples,
            "spatial_size": config.spatial_size,
            "sequence_length": config.sequence_length,
            "num_channels": config.num_channels,
            "field_types": config.field_types,
            "reynolds_number": config.reynolds_number,
            "mach_number": config.mach_number,
            "has_noise": config.add_noise,
            "noise_level": config.noise_level if config.add_noise else 0.0
        }

    @classmethod
    def validate_all_datasets(cls) -> Dict[str, Dict[str, bool]]:
        """Validate all default datasets"""
        results = {}
        for dataset_name in cls.DEFAULT_CONFIGS.keys():
            try:
                dataset = cls.create_dataset(dataset_name)
                results[dataset_name] = dataset.validate_physics()
            except Exception as e:
                results[dataset_name] = {"creation_error": str(e)}
        return results


def get_dummy_batch(dataset_name: str = "inc_low",
                   batch_size: int = 2,
                   **dataset_overrides) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Get a single dummy batch for quick testing with enhanced options

    Args:
        dataset_name: Name of the dataset to use
        batch_size: Number of samples in the batch
        **dataset_overrides: Override any dataset configuration parameters

    Returns:
        input_batch: [B, T_in, C, H, W] - Input sequence
        target_batch: [B, T_out, C, H, W] - Target sequence (can be None)
    """
    dataset = DummyDatasetFactory.create_dataset(dataset_name, **dataset_overrides)

    # Get samples
    inputs = []
    targets = []
    actual_batch_size = min(batch_size, len(dataset))

    for i in range(actual_batch_size):
        inp, tgt = dataset[i]
        inputs.append(inp)
        if tgt is not None:
            targets.append(tgt)

    input_batch = torch.stack(inputs)

    if targets:
        target_batch = torch.stack(targets)
    else:
        target_batch = None

    return input_batch, target_batch


def get_dummy_batch_for_model(model_type: str,
                             dataset_name: str = "inc_low",
                             batch_size: int = 2,
                             **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Get a dummy batch optimized for specific model testing

    Args:
        model_type: Type of model (fno, tno, unet, etc.)
        dataset_name: Dataset to use
        batch_size: Batch size
        **kwargs: Additional overrides

    Returns:
        Tuple of (input_batch, target_batch)
    """
    dataset = DummyDatasetFactory.create_for_model_testing(
        model_type=model_type,
        dataset_name=dataset_name,
        **kwargs
    )

    inputs = []
    targets = []
    actual_batch_size = min(batch_size, len(dataset))

    for i in range(actual_batch_size):
        inp, tgt = dataset[i]
        inputs.append(inp)
        if tgt is not None:
            targets.append(tgt)

    input_batch = torch.stack(inputs)
    target_batch = torch.stack(targets) if targets else None

    return input_batch, target_batch


def create_test_dataloader(dataset_name: str = "inc_low",
                          batch_size: int = 2,
                          shuffle: bool = True,
                          **dataset_overrides):
    """
    Create a DataLoader for testing purposes

    Args:
        dataset_name: Name of the dataset
        batch_size: Batch size for the DataLoader
        shuffle: Whether to shuffle the data
        **dataset_overrides: Override dataset parameters

    Returns:
        torch.utils.data.DataLoader
    """
    from torch.utils.data import DataLoader

    dataset = DummyDatasetFactory.create_dataset(dataset_name, **dataset_overrides)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def validate_dummy_data_quality() -> Dict[str, Any]:
    """
    Comprehensive validation of dummy data quality across all datasets

    Returns:
        Dict containing validation results
    """
    results = {
        "datasets_tested": [],
        "all_passed": True,
        "detailed_results": {},
        "summary": {}
    }

    factory_results = DummyDatasetFactory.validate_all_datasets()

    for dataset_name, validation_results in factory_results.items():
        results["datasets_tested"].append(dataset_name)
        results["detailed_results"][dataset_name] = validation_results

        # Check if all validations passed
        if "creation_error" in validation_results:
            results["all_passed"] = False
        else:
            dataset_passed = all(validation_results.values())
            if not dataset_passed:
                results["all_passed"] = False

    # Generate summary
    total_datasets = len(results["datasets_tested"])
    passed_datasets = sum(1 for name in results["datasets_tested"]
                         if "creation_error" not in results["detailed_results"][name]
                         and all(results["detailed_results"][name].values()))

    results["summary"] = {
        "total_datasets": total_datasets,
        "passed_datasets": passed_datasets,
        "failed_datasets": total_datasets - passed_datasets,
        "success_rate": passed_datasets / total_datasets if total_datasets > 0 else 0.0
    }

    return results


if __name__ == "__main__":
    # Test the enhanced dummy datasets
    print("Testing enhanced dummy datasets...")
    print("=" * 50)

    # Test all datasets
    for dataset_name in DummyDatasetFactory.get_available_datasets():
        print(f"\nTesting {dataset_name}:")
        print("-" * 30)

        try:
            dataset = DummyDatasetFactory.create_dataset(dataset_name)
            print(f"  Samples: {len(dataset)}")

            # Test a sample
            inp, tgt = dataset[0]
            target_shape = tgt.shape if tgt is not None else "None"
            print(f"  Input shape: {inp.shape}")
            print(f"  Target shape: {target_shape}")

            # Get sample info
            info = dataset.get_sample_info(0)
            print(f"  Field types: {info['field_types']}")
            print(f"  Reynolds: {info['reynolds_number']:.1f}")
            print(f"  Mach: {info['mach_number']:.2f}")

            # Validate physics
            validation = dataset.validate_physics()
            print(f"  Physics validation: {validation}")

        except Exception as e:
            print(f"  ERROR: {e}")

    # Test batch creation
    print("\n" + "=" * 50)
    print("Testing batch creation:")
    input_batch, target_batch = get_dummy_batch("inc_low", batch_size=3)
    target_shape = target_batch.shape if target_batch is not None else "None"
    print(f"  Input batch: {input_batch.shape}")
    print(f"  Target batch: {target_shape}")

    # Test model-specific datasets
    print("\nTesting model-specific datasets:")
    for model_type in ["fno", "tno", "unet", "diffusion"]:
        try:
            inp, tgt = get_dummy_batch_for_model(model_type, "inc_low", batch_size=2)
            target_shape = tgt.shape if tgt is not None else "None"
            print(f"  {model_type}: Input {inp.shape}, Target {target_shape}")
        except Exception as e:
            print(f"  {model_type}: ERROR - {e}")

    # Comprehensive validation
    print("\n" + "=" * 50)
    print("Comprehensive data quality validation:")
    validation_results = validate_dummy_data_quality()
    print(f"  Total datasets: {validation_results['summary']['total_datasets']}")
    print(f"  Passed: {validation_results['summary']['passed_datasets']}")
    print(f"  Failed: {validation_results['summary']['failed_datasets']}")
    print(f"  Success rate: {validation_results['summary']['success_rate']:.2%}")

    if validation_results["all_passed"]:
        print("\n✅ All dummy datasets are working correctly!")
    else:
        print("\n❌ Some datasets have issues. Check detailed results.")
        for dataset_name, results in validation_results["detailed_results"].items():
            if "creation_error" in results or not all(results.values()):
                print(f"    {dataset_name}: {results}")