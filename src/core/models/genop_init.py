"""
Generative Operator Model Registration and Initialization

This module handles the automatic registration of built-in neural operator priors
and generative correctors with the model registry system.
"""

import logging
from .model_registry import ModelRegistry
from .neural_operator_adapters import (
    FNOPriorAdapter,
    TNOPriorAdapter,
    UNetPriorAdapter,
    DeepONetPriorAdapter,
    DeepOKANPriorAdapter
)
from .generative_correctors import DiffusionCorrector, GANCorrector, VAECorrector


def register_builtin_models():
    """
    Register all built-in neural operator priors and generative correctors.

    This function should be called during framework initialization to ensure
    all standard model types are available through the registry.
    """
    logging.info("Registering built-in generative operator models...")

    # Register neural operator priors
    ModelRegistry.register_prior('fno', FNOPriorAdapter)
    ModelRegistry.register_prior('tno', TNOPriorAdapter)
    ModelRegistry.register_prior('unet', UNetPriorAdapter)
    ModelRegistry.register_prior('deeponet', DeepONetPriorAdapter)
    ModelRegistry.register_prior('deepokan', DeepOKANPriorAdapter)

    # Register generative correctors
    ModelRegistry.register_corrector('diffusion', DiffusionCorrector)

    # Future models (not yet implemented)
    # ModelRegistry.register_corrector('gan', GANCorrector)
    # ModelRegistry.register_corrector('vae', VAECorrector)

    status = ModelRegistry.get_registry_status()
    logging.info(f"Registration complete: {status['num_priors']} priors, {status['num_correctors']} correctors")
    logging.info(f"Available priors: {status['registered_priors']}")
    logging.info(f"Available correctors: {status['registered_correctors']}")


def initialize_generative_operators():
    """
    Initialize the generative operator framework.

    This function performs all necessary setup for using generative operators
    in the Gen Stabilised framework.
    """
    try:
        register_builtin_models()
        logging.info("Generative operator framework initialized successfully")
        return True
    except Exception as e:
        logging.error(f"Failed to initialize generative operator framework: {e}")
        return False


# Auto-initialize when module is imported
if not ModelRegistry.get_registry_status()['num_priors']:
    initialize_generative_operators()