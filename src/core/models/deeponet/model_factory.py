"""
DeepONet Model Factory

Provides factory methods for creating different DeepONet variants
with standardized configuration and parameter handling.
"""

from typing import Dict, Any, Optional, Type, Union
import logging

from .deeponet_base import DeepONet, DeepONetConfig
from .deeponet_variants import (
    StandardDeepONet,
    FourierDeepONet,
    PhysicsInformedDeepONet,
    MultiScaleDeepONet
)
from src.core.utils.params import ModelParamsDecoder, DataParams


class DeepONetFactory:
    """Factory for creating DeepONet models with various configurations."""

    # Registry of available DeepONet variants
    _variants = {
        'standard': StandardDeepONet,
        'fourier': FourierDeepONet,
        'physics': PhysicsInformedDeepONet,
        'physics_informed': PhysicsInformedDeepONet,  # Alias
        'multiscale': MultiScaleDeepONet,
        'multi_scale': MultiScaleDeepONet  # Alias
    }

    @classmethod
    def create(cls,
               variant: str = 'standard',
               config: Optional[Union[Dict[str, Any], DeepONetConfig]] = None,
               p_md: Optional[ModelParamsDecoder] = None,
               p_d: Optional[DataParams] = None,
               **kwargs) -> DeepONet:
        """
        Create a DeepONet model of the specified variant.

        Args:
            variant: Type of DeepONet variant to create
            config: Configuration dictionary or DeepONetConfig object
            p_md: Model parameter decoder (legacy compatibility)
            p_d: Data parameters (legacy compatibility)
            **kwargs: Additional configuration parameters

        Returns:
            DeepONet model instance

        Raises:
            ValueError: If variant is not supported
        """
        variant = variant.lower()
        if variant not in cls._variants:
            available = ', '.join(cls._variants.keys())
            raise ValueError(f"Unknown DeepONet variant '{variant}'. Available: {available}")

        # Handle configuration
        if config is None:
            config = {}
        elif isinstance(config, DeepONetConfig):
            # Convert to dict for merging with kwargs
            config_dict = {
                'latent_dim': config.latent_dim,
                'bias': config.bias,
                'normalize_inputs': config.normalize_inputs,
                'branch_type': config.branch_type,
                'branch_layers': config.branch_layers,
                'branch_activation': config.branch_activation,
                'branch_dropout': config.branch_dropout,
                'branch_batch_norm': config.branch_batch_norm,
                'trunk_type': config.trunk_type,
                'trunk_layers': config.trunk_layers,
                'trunk_activation': config.trunk_activation,
                'trunk_dropout': config.trunk_dropout,
                'trunk_batch_norm': config.trunk_batch_norm,
                'positional_encoding': config.positional_encoding,
                'sensor_strategy': config.sensor_strategy,
                'n_sensors': config.n_sensors,
                'n_query_train': config.n_query_train,
                'query_sampling': config.query_sampling
            }
            config = config_dict

        # Merge with kwargs
        config.update(kwargs)

        # Create DeepONetConfig from merged parameters
        if p_md is not None and p_d is not None:
            # Use legacy parameter system
            deeponet_config = DeepONetConfig.from_params(p_md, p_d)
            # Override with provided config
            for key, value in config.items():
                if hasattr(deeponet_config, key):
                    setattr(deeponet_config, key, value)
        else:
            deeponet_config = DeepONetConfig(**config)

        # Get the model class
        model_class = cls._variants[variant]

        # Create model with variant-specific parameters
        try:
            if variant in ['fourier']:
                # Fourier variants need additional parameters
                branch_modes = config.get('branch_fourier_modes', 32)
                trunk_modes = config.get('trunk_fourier_modes', 64)
                model = model_class(deeponet_config, branch_modes, trunk_modes, p_md, p_d)

            elif variant in ['physics', 'physics_informed']:
                # Physics-informed variants need physics type
                physics_type = config.get('physics_type', 'general')
                use_physics_loss = config.get('use_physics_loss', True)
                model = model_class(deeponet_config, physics_type, use_physics_loss, p_md, p_d)

            elif variant in ['multiscale', 'multi_scale']:
                # Multi-scale variants need scale parameters
                n_scales = config.get('n_scales', 3)
                scale_factors = config.get('scale_factors', None)
                model = model_class(deeponet_config, n_scales, scale_factors, p_md, p_d)

            else:
                # Standard variant
                model = model_class(deeponet_config, p_md, p_d)

            logging.info(f"Created {variant} DeepONet with {model.get_parameter_count()} parameters")
            return model

        except Exception as e:
            logging.error(f"Failed to create {variant} DeepONet: {str(e)}")
            raise

    @classmethod
    def create_from_config_file(cls,
                              config_path: str,
                              variant: Optional[str] = None) -> DeepONet:
        """
        Create DeepONet from configuration file.

        Args:
            config_path: Path to configuration file (YAML/JSON)
            variant: Override variant specified in config

        Returns:
            DeepONet model instance
        """
        import yaml
        import json
        import os

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        # Load configuration
        with open(config_path, 'r') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                config = yaml.safe_load(f)
            elif config_path.endswith('.json'):
                config = json.load(f)
            else:
                raise ValueError("Configuration file must be YAML or JSON")

        # Extract DeepONet configuration
        deeponet_config = config.get('deeponet', config)  # Support both formats

        # Override variant if specified
        if variant is not None:
            deeponet_config['variant'] = variant

        variant = deeponet_config.pop('variant', 'standard')

        return cls.create(variant=variant, config=deeponet_config)

    @classmethod
    def get_default_config(cls, variant: str = 'standard') -> Dict[str, Any]:
        """
        Get default configuration for a DeepONet variant.

        Args:
            variant: DeepONet variant name

        Returns:
            Default configuration dictionary
        """
        base_config = {
            'latent_dim': 256,
            'bias': True,
            'normalize_inputs': True,
            'branch_layers': [128, 256],
            'branch_activation': 'gelu',
            'branch_dropout': 0.1,
            'branch_batch_norm': True,
            'trunk_layers': [64, 128],
            'trunk_activation': 'gelu',
            'trunk_dropout': 0.1,
            'trunk_batch_norm': True,
            'positional_encoding': True,
            'sensor_strategy': 'uniform',
            'n_sensors': 100,
            'n_query_train': 1000,
            'query_sampling': 'random'
        }

        variant = variant.lower()

        if variant == 'fourier':
            base_config.update({
                'branch_fourier_modes': 32,
                'trunk_fourier_modes': 64,
                'trunk_layers': [128, 256]  # Larger trunk for Fourier features
            })

        elif variant in ['physics', 'physics_informed']:
            base_config.update({
                'physics_type': 'general',
                'use_physics_loss': True,
                'trunk_layers': [64, 128, 256],  # Larger trunk for physics features
                'latent_dim': 512  # Larger latent dimension for physics encoding
            })

        elif variant in ['multiscale', 'multi_scale']:
            base_config.update({
                'n_scales': 3,
                'scale_factors': [1.0, 2.0, 4.0],
                'latent_dim': 384,  # Divisible by 3 scales
                'trunk_layers': [32, 64]  # Smaller individual trunks
            })

        return base_config

    @classmethod
    def list_variants(cls) -> list:
        """Get list of available DeepONet variants."""
        return list(cls._variants.keys())

    @classmethod
    def register_variant(cls, name: str, model_class: Type[DeepONet]):
        """
        Register a new DeepONet variant.

        Args:
            name: Name for the variant
            model_class: DeepONet class to register
        """
        if not issubclass(model_class, DeepONet):
            raise ValueError(f"Model class must inherit from DeepONet")

        cls._variants[name.lower()] = model_class
        logging.info(f"Registered DeepONet variant: {name}")

    @classmethod
    def create_for_physics_domain(cls,
                                physics_domain: str,
                                spatial_dim: int = 2,
                                n_sensors: int = None,
                                **kwargs) -> DeepONet:
        """
        Create DeepONet optimized for specific physics domain.

        Args:
            physics_domain: Physics domain ('fluid', 'heat', 'wave', 'general')
            spatial_dim: Spatial dimension (1, 2, or 3)
            n_sensors: Number of sensors (auto-determined if None)
            **kwargs: Additional configuration

        Returns:
            Optimized DeepONet for the physics domain
        """
        # Domain-specific optimizations
        domain_configs = {
            'fluid': {
                'variant': 'physics',
                'physics_type': 'fluid',
                'latent_dim': 512,
                'trunk_layers': [64, 128, 256, 512],
                'branch_layers': [128, 256, 512],
                'n_sensors': 200,
                'use_physics_loss': True
            },
            'heat': {
                'variant': 'physics',
                'physics_type': 'heat',
                'latent_dim': 256,
                'trunk_layers': [32, 64, 128, 256],
                'branch_layers': [64, 128, 256],
                'n_sensors': 100,
                'use_physics_loss': True
            },
            'wave': {
                'variant': 'fourier',  # Waves often periodic
                'physics_type': 'wave',
                'latent_dim': 384,
                'trunk_fourier_modes': 128,
                'branch_fourier_modes': 64,
                'n_sensors': 150
            },
            'periodic': {
                'variant': 'fourier',
                'latent_dim': 512,
                'trunk_fourier_modes': 96,
                'branch_fourier_modes': 48,
                'n_sensors': 120
            }
        }

        if physics_domain not in domain_configs:
            available = ', '.join(domain_configs.keys())
            raise ValueError(f"Unknown physics domain '{physics_domain}'. Available: {available}")

        config = domain_configs[physics_domain].copy()

        # Override with user parameters
        if n_sensors is not None:
            config['n_sensors'] = n_sensors
        config.update(kwargs)

        variant = config.pop('variant')

        logging.info(f"Creating {variant} DeepONet optimized for {physics_domain} domain")
        return cls.create(variant=variant, config=config)