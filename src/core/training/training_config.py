"""
Training Configuration Management

This module provides configuration classes for different dataset types,
replacing the hardcoded configurations in separate training_*.py files.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Union
from src.core.utils.params import DataParams, TrainingParams, LossParams, ModelParamsEncoder, ModelParamsDecoder, ModelParamsLatent


@dataclass
class DatasetConfig:
    """Configuration for dataset-specific parameters."""
    name: str
    model_name: str
    data_params: DataParams
    training_params: TrainingParams
    loss_params: LossParams
    encoder_params: Optional[ModelParamsEncoder]
    decoder_params: ModelParamsDecoder
    latent_params: Optional[ModelParamsLatent]
    pretrain_path: str = ""

    # Dataset specific filters
    filter_top: List[str] = None
    filter_sim: List[Union[Tuple[int, int], List[int]]] = None
    filter_frame: List[Tuple[int, int]] = None

    # Test set configurations
    test_sets: Dict[str, Dict] = None


class TrainingConfigs:
    """Predefined training configurations for all datasets."""

    @staticmethod
    def get_inc_config() -> DatasetConfig:
        """Get Incompressible Wake Flow training configuration."""
        return DatasetConfig(
            name="inc",
            model_name="2D_Inc/128_unet-m2",
            data_params=DataParams(
                batch=64,
                augmentations=["normalize"],
                sequenceLength=[2,2],
                randSeqOffset=True,
                dataSize=[128,64],
                dimension=2,
                simFields=["pres"],
                simParams=["rey"],
                normalizeMode="incMixed"
            ),
            training_params=TrainingParams(epochs=1000, lr=0.0001),
            loss_params=LossParams(recMSE=0.0, predMSE=1.0),
            encoder_params=None,
            decoder_params=ModelParamsDecoder(arch="unet", pretrained=False, trainingNoise=0.0),
            latent_params=None,
            filter_top=["128_inc"],
            filter_sim=[(10,81)],
            filter_frame=[(800,1300)],
            test_sets={
                "lowRey": {
                    "name": "Test Low Reynolds 100-200",
                    "filter_top": ["128_inc"],
                    "filter_sim": [[82,84,86,88,90]],
                    "filter_frame": [(1000,1150)],
                    "sequence_length": [[60,2]]
                },
                "highRey": {
                    "name": "Test High Reynolds 900-1000",
                    "filter_top": ["128_inc"],
                    "filter_sim": [[0,2,4,6,8]],
                    "filter_frame": [(1000,1150)],
                    "sequence_length": [[60,2]]
                },
                "varReyIn": {
                    "name": "Test Varying Reynolds Number (200-900)",
                    "filter_top": ["128_reyVar"],
                    "filter_sim": [[0]],
                    "filter_frame": [(300,800)],
                    "sequence_length": [[250,2]]
                }
            }
        )

    @staticmethod
    def get_iso_config() -> DatasetConfig:
        """Get Isotropic Turbulence training configuration."""
        return DatasetConfig(
            name="iso",
            model_name="2D_Iso/128_unet-m2",
            data_params=DataParams(
                batch=64,
                augmentations=["normalize"],
                sequenceLength=[2,1],
                randSeqOffset=True,
                dataSize=[128,64],
                dimension=2,
                simFields=["velZ", "pres"],
                simParams=[],
                normalizeMode="isoSingle"
            ),
            training_params=TrainingParams(epochs=100, lr=0.0001),
            loss_params=LossParams(recMSE=0.0, predMSE=1.0),
            encoder_params=None,
            decoder_params=ModelParamsDecoder(arch="unet", pretrained=False, trainingNoise=0.0),
            latent_params=None,
            filter_top=["128_iso"],
            filter_sim=[(200,351)],
            filter_frame=[(0,1000)],
            test_sets={
                "testIso": {
                    "name": "Test Isotropic Turbulence",
                    "filter_top": ["128_iso"],
                    "filter_sim": [[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]],
                    "filter_frame": [(400,500)],
                    "sequence_length": [[60,1]]
                }
            }
        )

    @staticmethod
    def get_tra_config() -> DatasetConfig:
        """Get Transonic Cylinder Flow training configuration."""
        return DatasetConfig(
            name="tra",
            model_name="2D_Tra/128_unet-m2",
            data_params=DataParams(
                batch=64,
                augmentations=["normalize"],
                sequenceLength=[2,2],
                randSeqOffset=True,
                dataSize=[128,64],
                dimension=2,
                simFields=["pres", "density"],
                simParams=["mach"],
                normalizeMode="traMixed"
            ),
            training_params=TrainingParams(epochs=1000, lr=0.0001),
            loss_params=LossParams(recMSE=0.0, predMSE=1.0),
            encoder_params=None,
            decoder_params=ModelParamsDecoder(arch="unet", pretrained=False, trainingNoise=0.0),
            latent_params=None,
            filter_top=["128_tra"],
            filter_sim=[[0,1,2,14,15,16,17,18]],
            filter_frame=[(0,1000)],
            test_sets={
                "testTra": {
                    "name": "Test Transonic Cylinder Flow",
                    "filter_top": ["128_tra"],
                    "filter_sim": [[19,20,21,22,23,24,25,26,27]],
                    "filter_frame": [(400,500)],
                    "sequence_length": [[60,2]]
                }
            }
        )

    @staticmethod
    def get_config(dataset_name: str) -> DatasetConfig:
        """Get configuration for specified dataset."""
        # Try to load from YAML first, fall back to hardcoded configs
        try:
            from src.core.utils.config_loader import load_training_config
            return load_training_config(dataset_name)
        except (ImportError, FileNotFoundError):
            # Fall back to hardcoded configurations
            configs = {
                "inc": TrainingConfigs.get_inc_config,
                "iso": TrainingConfigs.get_iso_config,
                "tra": TrainingConfigs.get_tra_config
            }

            if dataset_name not in configs:
                raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(configs.keys())}")

            return configs[dataset_name]()


# Configuration variants for different training modes
class ConfigVariants:
    """Variants of base configurations for different training modes."""

    @staticmethod
    def get_unrolled_training_config(base_config: DatasetConfig, unroll_length: int = 8) -> DatasetConfig:
        """Modify config for unrolled training."""
        config = base_config
        config.data_params.sequenceLength = [unroll_length, 2]
        config.data_params.batch = max(16, config.data_params.batch // 4)  # Reduce batch size
        config.model_name = config.model_name.replace("-m2", f"-m{unroll_length}")
        return config

    @staticmethod
    def get_noise_training_config(base_config: DatasetConfig, noise_level: float = 0.01) -> DatasetConfig:
        """Modify config for training with noise."""
        config = base_config
        config.decoder_params.trainingNoise = noise_level
        config.model_name = config.model_name + f"-noise{noise_level}"
        return config