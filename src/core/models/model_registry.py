"""
Model Registry System for Generative Operators

This module provides a registry system for managing pluggable neural operator priors
and generative correctors. It enables dynamic model loading and creation while
maintaining type safety and framework compatibility.

The registry follows Gen Stabilised conventions and integrates with the existing
parameter system.
"""

from typing import Dict, Type, Any, Optional, Callable
import torch
import logging
from .base_classes import NeuralOperatorPrior, GenerativeCorrector
from src.core.utils.params import ModelParamsDecoder, DataParams


class ModelRegistry:
    """
    Registry for neural operator priors and generative correctors.

    This class manages the registration and creation of pluggable model components,
    following the existing Gen Stabilised patterns for model management.
    """

    # Class-level registries
    _prior_models: Dict[str, Type[NeuralOperatorPrior]] = {}
    _corrector_models: Dict[str, Type[GenerativeCorrector]] = {}
    _prior_factories: Dict[str, Callable] = {}
    _corrector_factories: Dict[str, Callable] = {}

    @classmethod
    def register_prior(cls,
                      name: str,
                      adapter_class: Type[NeuralOperatorPrior],
                      factory_func: Optional[Callable] = None) -> None:
        """
        Register a neural operator prior adapter.

        Args:
            name: Unique identifier for the prior (e.g., 'fno', 'tno', 'unet')
            adapter_class: Class implementing NeuralOperatorPrior interface
            factory_func: Optional factory function for complex initialization
        """
        if name in cls._prior_models:
            logging.warning(f"Overwriting existing prior model registration: {name}")

        cls._prior_models[name] = adapter_class
        if factory_func:
            cls._prior_factories[name] = factory_func

        logging.info(f"Registered neural operator prior: {name} -> {adapter_class.__name__}")

    @classmethod
    def register_corrector(cls,
                          name: str,
                          corrector_class: Type[GenerativeCorrector],
                          factory_func: Optional[Callable] = None) -> None:
        """
        Register a generative corrector implementation.

        Args:
            name: Unique identifier for the corrector (e.g., 'diffusion', 'gan', 'vae')
            corrector_class: Class implementing GenerativeCorrector interface
            factory_func: Optional factory function for complex initialization
        """
        if name in cls._corrector_models:
            logging.warning(f"Overwriting existing corrector model registration: {name}")

        cls._corrector_models[name] = corrector_class
        if factory_func:
            cls._corrector_factories[name] = factory_func

        logging.info(f"Registered generative corrector: {name} -> {corrector_class.__name__}")

    @classmethod
    def create_prior(cls,
                    name: str,
                    p_md: ModelParamsDecoder,
                    p_d: DataParams,
                    **kwargs) -> NeuralOperatorPrior:
        """
        Create a neural operator prior using the existing parameter system.

        Args:
            name: Name of the registered prior
            p_md: Model parameters following Gen Stabilised conventions
            p_d: Data parameters following Gen Stabilised conventions
            **kwargs: Additional model-specific parameters

        Returns:
            prior: Instantiated neural operator prior

        Raises:
            ValueError: If prior name is not registered
        """
        if name not in cls._prior_models:
            available = list(cls._prior_models.keys())
            raise ValueError(f"Unknown prior type: {name}. Available: {available}")

        adapter_class = cls._prior_models[name]

        # Use factory function if available
        if name in cls._prior_factories:
            factory_func = cls._prior_factories[name]
            return factory_func(p_md, p_d, **kwargs)

        # Standard instantiation using Gen Stabilised parameter pattern
        try:
            return adapter_class(p_md, p_d, **kwargs)
        except Exception as e:
            logging.error(f"Failed to create prior {name}: {e}")
            raise

    @classmethod
    def create_corrector(cls,
                        name: str,
                        p_md: ModelParamsDecoder,
                        p_d: DataParams,
                        **kwargs) -> GenerativeCorrector:
        """
        Create a generative corrector using the existing parameter system.

        Args:
            name: Name of the registered corrector
            p_md: Model parameters following Gen Stabilised conventions
            p_d: Data parameters following Gen Stabilised conventions
            **kwargs: Additional model-specific parameters

        Returns:
            corrector: Instantiated generative corrector

        Raises:
            ValueError: If corrector name is not registered
        """
        if name not in cls._corrector_models:
            available = list(cls._corrector_models.keys())
            raise ValueError(f"Unknown corrector type: {name}. Available: {available}")

        corrector_class = cls._corrector_models[name]

        # Use factory function if available
        if name in cls._corrector_factories:
            factory_func = cls._corrector_factories[name]
            return factory_func(p_md, p_d, **kwargs)

        # Standard instantiation using Gen Stabilised parameter pattern
        try:
            return corrector_class(p_md, p_d, **kwargs)
        except Exception as e:
            logging.error(f"Failed to create corrector {name}: {e}")
            raise

    @classmethod
    def list_priors(cls) -> Dict[str, str]:
        """
        List all registered neural operator priors.

        Returns:
            priors: Dictionary mapping names to class names
        """
        return {name: cls_type.__name__ for name, cls_type in cls._prior_models.items()}

    @classmethod
    def list_correctors(cls) -> Dict[str, str]:
        """
        List all registered generative correctors.

        Returns:
            correctors: Dictionary mapping names to class names
        """
        return {name: cls_type.__name__ for name, cls_type in cls._corrector_models.items()}

    @classmethod
    def is_prior_registered(cls, name: str) -> bool:
        """Check if a prior is registered."""
        return name in cls._prior_models

    @classmethod
    def is_corrector_registered(cls, name: str) -> bool:
        """Check if a corrector is registered."""
        return name in cls._corrector_models

    @classmethod
    def validate_combination(cls, prior_name: str, corrector_name: str) -> bool:
        """
        Validate that a prior-corrector combination is supported.

        Args:
            prior_name: Name of the prior
            corrector_name: Name of the corrector

        Returns:
            valid: True if combination is supported
        """
        # Check if both models are registered
        if not (cls.is_prior_registered(prior_name) and cls.is_corrector_registered(corrector_name)):
            return False

        # Add specific compatibility checks here if needed
        # For now, assume all combinations are valid
        return True

    @classmethod
    def get_model_info(cls, name: str, model_type: str) -> Dict[str, Any]:
        """
        Get information about a registered model.

        Args:
            name: Model name
            model_type: Either 'prior' or 'corrector'

        Returns:
            info: Model information dictionary
        """
        if model_type == 'prior':
            if name not in cls._prior_models:
                raise ValueError(f"Prior {name} not registered")
            model_class = cls._prior_models[name]
        elif model_type == 'corrector':
            if name not in cls._corrector_models:
                raise ValueError(f"Corrector {name} not registered")
            model_class = cls._corrector_models[name]
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        return {
            'name': name,
            'type': model_type,
            'class_name': model_class.__name__,
            'module': model_class.__module__,
            'has_factory': name in (cls._prior_factories if model_type == 'prior' else cls._corrector_factories)
        }

    @classmethod
    def clear_registry(cls) -> None:
        """Clear all registered models (mainly for testing)."""
        cls._prior_models.clear()
        cls._corrector_models.clear()
        cls._prior_factories.clear()
        cls._corrector_factories.clear()
        logging.info("Cleared model registry")

    @classmethod
    def get_registry_status(cls) -> Dict[str, Any]:
        """
        Get overall registry status for debugging.

        Returns:
            status: Dictionary with registry information
        """
        return {
            'num_priors': len(cls._prior_models),
            'num_correctors': len(cls._corrector_models),
            'registered_priors': list(cls._prior_models.keys()),
            'registered_correctors': list(cls._corrector_models.keys()),
            'prior_factories': list(cls._prior_factories.keys()),
            'corrector_factories': list(cls._corrector_factories.keys())
        }


def parse_generative_operator_architecture(arch: str) -> tuple[str, str]:
    """
    Parse generative operator architecture string.

    Follows Gen Stabilised conventions like "tno+Prev" and extends to
    "genop-prior-corrector" format.

    Args:
        arch: Architecture string (e.g., "genop-fno-diffusion", "genop-tno-gan+Prev")

    Returns:
        prior_type: Type of neural operator prior
        corrector_type: Type of generative corrector

    Raises:
        ValueError: If architecture string is invalid
    """
    # Remove +Prev, +2Prev, etc. for parsing
    base_arch = arch.split('+')[0]

    # Handle genop-prior-corrector format
    if base_arch.startswith('genop-'):
        parts = base_arch.split('-')
        if len(parts) != 3:
            raise ValueError(f"Invalid genop architecture: {arch}. Expected format: genop-prior-corrector")

        _, prior_type, corrector_type = parts
        return prior_type, corrector_type

    # Handle legacy formats
    elif base_arch in ['nodm', 'no_dm']:
        # Default NO+DM configuration
        return 'fno', 'diffusion'

    else:
        raise ValueError(f"Unknown generative operator architecture: {arch}")


def validate_architecture_string(arch: str) -> bool:
    """
    Validate architecture string format.

    Args:
        arch: Architecture string to validate

    Returns:
        valid: True if string is valid
    """
    try:
        prior_type, corrector_type = parse_generative_operator_architecture(arch)
        return ModelRegistry.validate_combination(prior_type, corrector_type)
    except ValueError:
        return False