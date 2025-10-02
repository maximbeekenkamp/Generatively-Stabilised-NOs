#!/usr/bin/env python3
"""
Generative Operator Evaluation Configuration System

This module provides a comprehensive configuration system for evaluating
generative operators across different datasets, architectures, and experimental
settings. It supports standardized evaluation protocols, ablation studies,
and comparison configurations.

Author: Phase 3 Implementation
"""

import os
import json
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import logging


@dataclass
class GenerativeOperatorEvalConfig:
    """Configuration for generative operator evaluation."""

    # Model identification
    model_name: str
    model_type: str  # "genop-fno-diffusion", "genop-tno-diffusion", etc.
    model_paths: List[str]

    # Dataset configuration
    dataset: str  # "inc", "tra", "iso"
    test_sets: Dict[str, int]  # test_set_name -> batch_size

    # Evaluation parameters
    num_evals: int = 3
    sequential_eval_runs: Dict[str, bool] = None

    # Generative operator specific settings
    generative_operator_mode: str = "full_inference"
    correction_strength: float = 1.0
    enable_dcar: bool = True
    dcar_correction_frequency: int = 1
    diffusion_steps: int = 32
    memory_efficient: bool = True

    # Dataset-specific options
    dataset_specific_options: Dict[str, Any] = None

    # Ablation study parameters
    ablation_type: Optional[str] = None  # "correction_strength", "diffusion_steps", etc.
    ablation_values: Optional[List[Any]] = None

    def __post_init__(self):
        """Initialize default values based on dataset and model type."""
        if self.sequential_eval_runs is None:
            self.sequential_eval_runs = self._get_default_sequential_runs()

        if self.dataset_specific_options is None:
            self.dataset_specific_options = self._get_default_dataset_options()

    def _get_default_sequential_runs(self) -> Dict[str, bool]:
        """Get default sequential run configuration based on dataset."""
        if self.dataset == "inc":
            return {"lowRey": True, "highRey": True, "varReyIn": True}
        elif self.dataset == "tra":
            return {"extrap": True, "interp": True, "longer": True}
        elif self.dataset == "iso":
            return {"zInterp": True}
        else:
            return {}

    def _get_default_dataset_options(self) -> Dict[str, Any]:
        """Get default dataset-specific options."""
        base_options = {}

        if self.dataset == "inc":
            # Incompressible flow options
            base_options.update({
                "physicsConservation": True,
                "divergenceFreeConstraint": True,
            })

        elif self.dataset == "tra":
            # Transonic flow options
            base_options.update({
                "shockPreservation": True,
                "gradientClipping": 1.0,
                "shockDetectionThreshold": 0.1,
            })

        elif self.dataset == "iso":
            # Isotropic turbulence options
            base_options.update({
                "energySpectrumPreservation": True,
                "kolmogorovScaling": True,
                "turbulenceIntensityMonitoring": True,
            })

        # Add TNO-specific options if model uses TNO
        if "tno" in self.model_type:
            base_options.update({
                "tnoPhase": "fine_tuning",
                "tnoMemoryLength": 2,
                "tnoBundleSize": 4,
            })

        return base_options

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format compatible with sampling scripts."""
        return asdict(self)

    def to_sampling_script_format(self) -> tuple:
        """Convert to the format expected by sampling scripts."""
        eval_options = {
            "numEvals": self.num_evals,
            "sequentialEvalRuns": self.sequential_eval_runs,
            "generativeOperatorMode": self.generative_operator_mode,
            "correctionStrength": self.correction_strength,
            "enableDCAR": self.enable_dcar,
            "dcarCorrectionFrequency": self.dcar_correction_frequency,
            "diffusionSteps": self.diffusion_steps,
            "memoryEfficient": self.memory_efficient,
        }

        # Add dataset-specific options
        eval_options.update(self.dataset_specific_options)

        return (self.model_paths, self.test_sets, eval_options)


class GenerativeOperatorEvaluationConfigGenerator:
    """Generator for comprehensive evaluation configurations."""

    def __init__(self, base_config_dir: str = "configs/evaluation"):
        self.base_config_dir = Path(base_config_dir)
        self.base_config_dir.mkdir(parents=True, exist_ok=True)

        # Standard model configurations
        self.model_types = {
            "genop-fno-diffusion": {
                "description": "FNO + Diffusion generative operator",
                "default_batch_sizes": {"inc": 2, "tra": 1, "iso": 2},
                "default_diffusion_steps": 32,
            },
            "genop-tno-diffusion": {
                "description": "TNO + Diffusion generative operator",
                "default_batch_sizes": {"inc": 2, "tra": 1, "iso": 1},
                "default_diffusion_steps": 32,
            },
            "genop-unet-diffusion": {
                "description": "U-Net + Diffusion generative operator",
                "default_batch_sizes": {"inc": 4, "tra": 2, "iso": 4},
                "default_diffusion_steps": 32,
            },
        }

        # Dataset test set configurations
        self.dataset_test_sets = {
            "inc": ["lowRey", "highRey", "varReyIn"],
            "tra": ["extrap", "interp", "longer"],
            "iso": ["zInterp"],
        }

    def generate_standard_configs(self) -> Dict[str, GenerativeOperatorEvalConfig]:
        """Generate standard evaluation configurations for all model-dataset combinations."""
        configs = {}

        for model_type, model_info in self.model_types.items():
            for dataset in ["inc", "tra", "iso"]:
                config_name = f"{model_type}_{dataset}_standard"

                # Default model paths
                model_paths = [
                    f"128_{model_type}_{dataset}_{i:02d}/Model.pth"
                    for i in range(3)
                ]

                # Default batch sizes
                batch_size = model_info["default_batch_sizes"][dataset]
                test_sets = {
                    test_set: batch_size
                    for test_set in self.dataset_test_sets[dataset]
                }

                config = GenerativeOperatorEvalConfig(
                    model_name=config_name,
                    model_type=model_type,
                    model_paths=model_paths,
                    dataset=dataset,
                    test_sets=test_sets,
                    num_evals=3,
                    diffusion_steps=model_info["default_diffusion_steps"],
                )

                configs[config_name] = config

        return configs

    def generate_ablation_configs(self) -> Dict[str, GenerativeOperatorEvalConfig]:
        """Generate ablation study configurations."""
        configs = {}

        # Correction strength ablation
        for model_type in self.model_types.keys():
            for dataset in ["inc", "tra", "iso"]:
                base_config_name = f"{model_type}_{dataset}_standard"

                if base_config_name in self.generate_standard_configs():
                    base_config = self.generate_standard_configs()[base_config_name]

                    # Correction strength ablation
                    for strength in [0.3, 0.6, 1.0, 1.3, 1.6]:
                        config_name = f"{model_type}_{dataset}_strength_{strength:.1f}"
                        config = GenerativeOperatorEvalConfig(
                            model_name=config_name,
                            model_type=base_config.model_type,
                            model_paths=base_config.model_paths,
                            dataset=base_config.dataset,
                            test_sets=base_config.test_sets,
                            num_evals=base_config.num_evals,
                            correction_strength=strength,
                            ablation_type="correction_strength",
                            ablation_values=[strength],
                        )
                        configs[config_name] = config

                    # Diffusion steps ablation
                    for steps in [16, 32, 48, 64]:
                        config_name = f"{model_type}_{dataset}_steps_{steps}"
                        config = GenerativeOperatorEvalConfig(
                            model_name=config_name,
                            model_type=base_config.model_type,
                            model_paths=base_config.model_paths,
                            dataset=base_config.dataset,
                            test_sets=base_config.test_sets,
                            num_evals=base_config.num_evals,
                            diffusion_steps=steps,
                            ablation_type="diffusion_steps",
                            ablation_values=[steps],
                        )
                        configs[config_name] = config

        return configs

    def generate_comparison_configs(self) -> Dict[str, GenerativeOperatorEvalConfig]:
        """Generate standardized configurations for fair model comparison."""
        configs = {}

        for dataset in ["inc", "tra", "iso"]:
            # Standardized settings for comparison
            standard_batch_size = 2  # Conservative batch size for all models
            standard_diffusion_steps = 32
            standard_num_evals = 3

            test_sets = {
                test_set: standard_batch_size
                for test_set in self.dataset_test_sets[dataset]
            }

            for model_type in self.model_types.keys():
                config_name = f"{model_type}_{dataset}_comparison"

                model_paths = [
                    f"128_{model_type}_{dataset}_{i:02d}/Model.pth"
                    for i in range(3)
                ]

                config = GenerativeOperatorEvalConfig(
                    model_name=config_name,
                    model_type=model_type,
                    model_paths=model_paths,
                    dataset=dataset,
                    test_sets=test_sets,
                    num_evals=standard_num_evals,
                    correction_strength=1.0,
                    diffusion_steps=standard_diffusion_steps,
                    memory_efficient=True,
                )

                configs[config_name] = config

        return configs

    def generate_prior_only_configs(self) -> Dict[str, GenerativeOperatorEvalConfig]:
        """Generate configurations for prior-only evaluation (no diffusion correction)."""
        configs = {}

        for model_type in self.model_types.keys():
            for dataset in ["inc", "tra", "iso"]:
                config_name = f"{model_type}_{dataset}_prior_only"

                # Larger batch sizes since no diffusion correction
                batch_size = self.model_types[model_type]["default_batch_sizes"][dataset] * 2
                test_sets = {
                    test_set: batch_size
                    for test_set in self.dataset_test_sets[dataset]
                }

                model_paths = [
                    f"128_{model_type}_{dataset}_{i:02d}/Model.pth"
                    for i in range(3)
                ]

                config = GenerativeOperatorEvalConfig(
                    model_name=config_name,
                    model_type=model_type,
                    model_paths=model_paths,
                    dataset=dataset,
                    test_sets=test_sets,
                    num_evals=1,  # Deterministic, so only one evaluation needed
                    generative_operator_mode="prior_only",
                    correction_strength=0.0,
                    enable_dcar=False,
                    diffusion_steps=0,
                )

                configs[config_name] = config

        return configs

    def save_configs_to_json(self, configs: Dict[str, GenerativeOperatorEvalConfig],
                           output_dir: Optional[str] = None):
        """Save configurations to JSON files."""
        if output_dir is None:
            output_dir = self.base_config_dir
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        for config_name, config in configs.items():
            config_path = output_dir / f"{config_name}.json"

            with open(config_path, 'w') as f:
                json.dump(config.to_dict(), f, indent=2)

            logging.info(f"Saved configuration: {config_path}")

    def load_config_from_json(self, config_path: str) -> GenerativeOperatorEvalConfig:
        """Load configuration from JSON file."""
        with open(config_path, 'r') as f:
            config_dict = json.load(f)

        return GenerativeOperatorEvalConfig(**config_dict)

    def generate_all_configs(self) -> Dict[str, Dict[str, GenerativeOperatorEvalConfig]]:
        """Generate all configuration types."""
        all_configs = {
            "standard": self.generate_standard_configs(),
            "ablation": self.generate_ablation_configs(),
            "comparison": self.generate_comparison_configs(),
            "prior_only": self.generate_prior_only_configs(),
        }

        return all_configs

    def save_all_configs(self):
        """Generate and save all configuration types."""
        all_configs = self.generate_all_configs()

        for config_type, configs in all_configs.items():
            type_dir = self.base_config_dir / config_type
            self.save_configs_to_json(configs, type_dir)
            logging.info(f"Saved {len(configs)} {config_type} configurations to {type_dir}")


class GenerativeOperatorEvaluationRunner:
    """Runner for executing evaluations with generated configurations."""

    def __init__(self, config_dir: str = "configs/evaluation"):
        self.config_dir = Path(config_dir)
        self.generator = GenerativeOperatorEvaluationConfigGenerator(config_dir)

    def run_evaluation(self, config: GenerativeOperatorEvalConfig,
                      output_dir: str = "results/genop_evaluation"):
        """Run evaluation with a specific configuration."""
        from src.analysis.sample import sample_models_inc, sample_models_tra, sample_models_iso

        # Map dataset to appropriate sampling module
        dataset_modules = {
            "inc": sample_models_inc,
            "tra": sample_models_tra,
            "iso": sample_models_iso,
        }

        if config.dataset not in dataset_modules:
            raise ValueError(f"Unknown dataset: {config.dataset}")

        module = dataset_modules[config.dataset]

        # Convert config to sampling script format
        model_paths, batch_dict, eval_options = config.to_sampling_script_format()

        # Create temporary model entry
        temp_model_entry = {
            config.model_name: (model_paths, batch_dict, eval_options)
        }

        # Update the module's models dictionary
        original_models = getattr(module, 'models', {})
        module.models = {**original_models, **temp_model_entry}

        # Update output folder
        original_out_folder = getattr(module, 'outFolder', 'results/sampling')
        module.outFolder = output_dir

        logging.info(f"Running evaluation for {config.model_name} on {config.dataset} dataset")

        # Note: In practice, you would call the module's main execution here
        # For this implementation, we're setting up the configuration structure

        return {
            "config_name": config.model_name,
            "dataset": config.dataset,
            "model_paths": model_paths,
            "eval_options": eval_options,
            "status": "configured"
        }

    def run_batch_evaluation(self, config_names: List[str],
                           config_type: str = "standard"):
        """Run evaluation for multiple configurations."""
        results = []

        config_dir = self.config_dir / config_type

        for config_name in config_names:
            config_path = config_dir / f"{config_name}.json"

            if not config_path.exists():
                logging.error(f"Configuration not found: {config_path}")
                continue

            config = self.generator.load_config_from_json(config_path)
            result = self.run_evaluation(config)
            results.append(result)

        return results


# Utility functions for easy configuration management
def create_standard_evaluation_configs():
    """Create and save standard evaluation configurations."""
    generator = GenerativeOperatorEvaluationConfigGenerator()
    generator.save_all_configs()
    logging.info("Created all standard evaluation configurations")


def load_evaluation_config(config_name: str, config_type: str = "standard") -> GenerativeOperatorEvalConfig:
    """Load a specific evaluation configuration."""
    generator = GenerativeOperatorEvaluationConfigGenerator()
    config_path = generator.base_config_dir / config_type / f"{config_name}.json"
    return generator.load_config_from_json(config_path)


def run_generative_operator_evaluation(config_name: str, config_type: str = "standard"):
    """Run evaluation for a specific configuration."""
    runner = GenerativeOperatorEvaluationRunner()
    config = load_evaluation_config(config_name, config_type)
    return runner.run_evaluation(config)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Create all configurations
    create_standard_evaluation_configs()

    # Example: Load and inspect a configuration
    config = load_evaluation_config("genop-fno-diffusion_inc_standard")
    print(f"Loaded config: {config.model_name}")
    print(f"Correction strength: {config.correction_strength}")
    print(f"Dataset-specific options: {config.dataset_specific_options}")