"""
Configuration Loader Module

Loads YAML configuration files and converts them to the Python parameter classes
used throughout the codebase. This replaces hardcoded configurations and provides
a unified configuration system.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional

from src.core.utils.common_imports import *
from src.core.utils.params import DataParams, TrainingParams, LossParams, ModelParamsDecoder
from src.core.training.training_config import DatasetConfig


class ConfigLoader:
    """Loads and parses YAML configuration files."""

    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize the config loader.

        Args:
            config_dir: Path to configuration directory. If None, uses project default.
        """
        if config_dir is None:
            # Find project root and configs directory
            current_file = Path(__file__)
            project_root = current_file.parent
            while project_root.parent != project_root:
                if (project_root / 'configs').exists():
                    break
                project_root = project_root.parent
            config_dir = project_root / 'configs'

        self.config_dir = Path(config_dir)
        if not self.config_dir.exists():
            raise FileNotFoundError(f"Configuration directory not found: {config_dir}")

    def load_yaml(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """Load a YAML configuration file."""
        config_path = Path(config_path)
        if not config_path.is_absolute():
            config_path = self.config_dir / config_path

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def load_training_config(self, dataset_name: str, variant: Optional[str] = None) -> DatasetConfig:
        """
        Load training configuration for a specific dataset.

        Args:
            dataset_name: Name of the dataset ('inc', 'iso', 'tra')
            variant: Optional variant name ('unrolled', 'noise_training', etc.)

        Returns:
            DatasetConfig object with loaded parameters
        """
        # Load the base configuration
        config_file = f"training/{dataset_name}_config.yaml"
        config = self.load_yaml(config_file)

        # Apply variant if specified
        if variant and variant in config.get('variants', {}):
            variant_config = config['variants'][variant]
            self._apply_variant_config(config, variant_config)

        # Convert to parameter objects
        return self._yaml_to_dataset_config(config)

    def _apply_variant_config(self, base_config: Dict[str, Any], variant_config: Dict[str, Any]):
        """Apply variant configuration overrides to base configuration."""
        # Update model name with suffix if specified
        if 'model_suffix' in variant_config:
            base_config['model']['name'] += variant_config['model_suffix']

        # Override data parameters
        data_params = base_config.get('data_params', {})
        if 'sequence_length' in variant_config:
            data_params['sequence_length'] = variant_config['sequence_length']
        if 'batch_size' in variant_config:
            data_params['batch_size'] = variant_config['batch_size']

        # Override model parameters
        model_params = base_config.get('model', {})
        if 'training_noise' in variant_config:
            model_params['training_noise'] = variant_config['training_noise']

    def _yaml_to_dataset_config(self, config: Dict[str, Any]) -> DatasetConfig:
        """Convert YAML configuration to DatasetConfig object."""
        # Extract sections
        dataset_info = config['dataset']
        model_info = config['model']
        data_params_yaml = config['data_params']
        training_params_yaml = config['training_params']
        loss_params_yaml = config['loss_params']
        data_filters = config.get('data_filters', {})
        test_sets_yaml = config.get('test_sets', {})

        # Create parameter objects
        data_params = DataParams(
            batch=data_params_yaml['batch_size'],
            augmentations=data_params_yaml['augmentations'],
            sequenceLength=data_params_yaml['sequence_length'],
            randSeqOffset=data_params_yaml['random_sequence_offset'],
            dataSize=data_params_yaml['data_size'],
            dimension=data_params_yaml['dimension'],
            simFields=data_params_yaml['sim_fields'],
            simParams=data_params_yaml['sim_params'],
            normalizeMode=data_params_yaml['normalize_mode']
        )

        training_params = TrainingParams(
            epochs=training_params_yaml['epochs'],
            lr=training_params_yaml['learning_rate'],
            weightDecay=training_params_yaml.get('weight_decay', 0.0),
            expLrGamma=training_params_yaml.get('exp_lr_gamma', 1.0),
            testInterval=training_params_yaml.get('test_interval', 10),
            saveInterval=training_params_yaml.get('save_interval', 100)
        )

        loss_params = LossParams(
            recFieldError=loss_params_yaml.get('reconstruction_field_error', loss_params_yaml.get('reconstruction_mse', 1.0)),
            predFieldError=loss_params_yaml.get('prediction_field_error', loss_params_yaml.get('prediction_mse', 1.0)),
            recLSIM=loss_params_yaml.get('reconstruction_lsim', 0.0),
            predLSIM=loss_params_yaml.get('prediction_lsim', 0.0)
        )

        decoder_params = ModelParamsDecoder(
            arch=model_info.get('architecture', 'unet'),
            pretrained=model_info.get('pretrained', False),
            trainingNoise=model_info.get('training_noise', 0.0)
        )

        # Convert test sets configuration
        test_sets = {}
        for test_name, test_config in test_sets_yaml.items():
            test_sets[test_name] = {
                'name': test_config['name'],
                'filter_top': test_config.get('filter_top', data_filters.get('filter_top', [])),
                'filter_sim': test_config['filter_sim'],
                'filter_frame': test_config['filter_frame'],
                'sequence_length': test_config['sequence_length']
            }

        # Create and return DatasetConfig
        return DatasetConfig(
            name=dataset_info['name'],
            model_name=model_info['name'],
            data_params=data_params,
            training_params=training_params,
            loss_params=loss_params,
            encoder_params=None,
            decoder_params=decoder_params,
            latent_params=None,
            filter_top=data_filters.get('filter_top', []),
            filter_sim=data_filters.get('filter_sim', []),
            filter_frame=data_filters.get('filter_frame', []),
            test_sets=test_sets
        )

    def list_available_configs(self) -> Dict[str, List[str]]:
        """List all available configuration files by category."""
        configs = {}
        for category_dir in self.config_dir.iterdir():
            if category_dir.is_dir():
                category_configs = []
                for config_file in category_dir.glob('*.yaml'):
                    category_configs.append(config_file.stem)
                if category_configs:
                    configs[category_dir.name] = sorted(category_configs)
        return configs

    def validate_config(self, config_path: Union[str, Path]) -> bool:
        """
        Validate a configuration file.

        Args:
            config_path: Path to configuration file

        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            config = self.load_yaml(config_path)
            # Basic validation - check for required sections
            required_sections = ['dataset', 'model', 'data_params', 'training_params', 'loss_params']
            for section in required_sections:
                if section not in config:
                    print(f"Missing required section: {section}")
                    return False
            return True
        except Exception as e:
            print(f"Configuration validation failed: {e}")
            return False


# Global config loader instance for convenience
_global_loader = None


def get_config_loader() -> ConfigLoader:
    """Get global configuration loader instance."""
    global _global_loader
    if _global_loader is None:
        _global_loader = ConfigLoader()
    return _global_loader


def load_training_config(dataset_name: str, variant: Optional[str] = None) -> DatasetConfig:
    """Convenience function to load training configuration."""
    return get_config_loader().load_training_config(dataset_name, variant)