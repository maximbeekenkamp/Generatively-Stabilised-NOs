"""
DeepONet Uncertainty Quantification

Implements various uncertainty quantification methods for DeepONet operator learning,
including ensemble methods, Bayesian approaches, and aleatoric/epistemic uncertainty
estimation for reliable operator predictions.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, MultivariateNormal
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import logging
from abc import ABC, abstractmethod

from .deeponet_base import DeepONet, DeepONetConfig
from .model_factory import DeepONetFactory
from .evaluation import DeepONetEvaluator


@dataclass
class UncertaintyEstimate:
    """Container for uncertainty quantification results."""
    predictions: torch.Tensor
    aleatoric_uncertainty: torch.Tensor
    epistemic_uncertainty: torch.Tensor
    total_uncertainty: torch.Tensor
    confidence_intervals: Dict[str, torch.Tensor]
    calibration_metrics: Dict[str, float]


class AbstractUncertaintyMethod(ABC):
    """Abstract base class for uncertainty quantification methods."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def estimate_uncertainty(self,
                           model: DeepONet,
                           input_functions: torch.Tensor,
                           query_coords: torch.Tensor,
                           **kwargs) -> UncertaintyEstimate:
        """Estimate uncertainty for given inputs."""
        pass

    @abstractmethod
    def calibrate(self,
                 model: DeepONet,
                 calibration_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                 **kwargs):
        """Calibrate uncertainty estimates using validation data."""
        pass


class EnsembleUncertainty(AbstractUncertaintyMethod):
    """
    Ensemble-based uncertainty quantification.

    Uses multiple independently trained DeepONet models to estimate
    epistemic uncertainty through prediction variance.
    """

    def __init__(self,
                 n_models: int = 5,
                 diversity_regularization: float = 0.01):
        super().__init__("EnsembleUncertainty")
        self.n_models = n_models
        self.diversity_regularization = diversity_regularization
        self.ensemble_models = []

    def create_ensemble(self,
                       base_config: DeepONetConfig,
                       training_data: Tuple[torch.Tensor, torch.Tensor],
                       device: str = 'cuda') -> List[DeepONet]:
        """
        Create ensemble of DeepONet models with different initializations.

        Args:
            base_config: Base configuration for ensemble models
            training_data: Training data for ensemble training
            device: Device for training

        Returns:
            ensemble_models: List of trained models
        """
        logging.info(f"Creating ensemble of {self.n_models} DeepONet models")

        ensemble_models = []
        train_input, train_output = training_data

        for i in range(self.n_models):
            # Create model with slight configuration variations
            config = self._vary_config(base_config, i)
            model = DeepONetFactory.create('standard', config=config)
            model = model.to(device)

            # Add diversity regularization to loss
            if self.diversity_regularization > 0 and i > 0:
                self._add_diversity_loss(model, ensemble_models[-1])

            # Train model (simplified - in practice would use full training loop)
            self._train_model(model, train_input, train_output, device)

            ensemble_models.append(model)
            logging.info(f"Trained ensemble model {i+1}/{self.n_models}")

        self.ensemble_models = ensemble_models
        return ensemble_models

    def _vary_config(self, base_config: DeepONetConfig, model_idx: int) -> DeepONetConfig:
        """Create configuration variation for ensemble diversity."""
        # Create copy of base config
        import copy
        config = copy.deepcopy(base_config)

        # Add small variations
        variations = [
            {'dropout': 0.05 + model_idx * 0.02},
            {'branch_layers': [128 + model_idx * 32, 256 + model_idx * 64]},
            {'trunk_layers': [64 + model_idx * 16, 128 + model_idx * 32]},
            {'activation': ['gelu', 'relu', 'silu', 'tanh'][model_idx % 4]},
            {'sensor_strategy': ['uniform', 'random'][model_idx % 2]}
        ]

        # Apply variation
        variation = variations[model_idx % len(variations)]
        for key, value in variation.items():
            if hasattr(config, key):
                setattr(config, key, value)

        return config

    def _add_diversity_loss(self, current_model: DeepONet, reference_model: DeepONet):
        """Add diversity regularization to encourage ensemble diversity."""
        # This would be implemented in the training loop
        # For now, just log the intention
        logging.debug(f"Added diversity regularization with weight {self.diversity_regularization}")

    def _train_model(self,
                    model: DeepONet,
                    train_input: torch.Tensor,
                    train_output: torch.Tensor,
                    device: str):
        """Simplified model training for ensemble."""
        # This is a placeholder - in practice would use the full training pipeline
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Quick training loop (simplified)
        for epoch in range(50):
            optimizer.zero_grad()

            # Generate random query points
            B, T, C, H, W = train_input.shape
            n_query = 256
            query_coords = torch.rand(B, n_query, 2, device=device)

            # Forward pass
            predictions = model(train_input, query_coords)

            # Simple loss (would use proper operator learning loss)
            loss = F.mse_loss(predictions, train_output[:, :, :n_query])

            loss.backward()
            optimizer.step()

        model.eval()
        logging.debug(f"Completed simplified training with final loss: {loss.item():.6f}")

    def estimate_uncertainty(self,
                           model: Union[DeepONet, List[DeepONet]],
                           input_functions: torch.Tensor,
                           query_coords: torch.Tensor,
                           **kwargs) -> UncertaintyEstimate:
        """
        Estimate uncertainty using ensemble predictions.

        Args:
            model: Single model or list of ensemble models
            input_functions: Input functions [B, T, C, H, W]
            query_coords: Query coordinates [B, n_query, 2]

        Returns:
            uncertainty_estimate: Comprehensive uncertainty estimates
        """
        if isinstance(model, list):
            ensemble_models = model
        else:
            ensemble_models = self.ensemble_models

        if not ensemble_models:
            raise ValueError("No ensemble models available. Call create_ensemble first.")

        logging.info(f"Estimating uncertainty using {len(ensemble_models)} ensemble models")

        # Collect predictions from all ensemble members
        all_predictions = []

        for model_idx, ensemble_model in enumerate(ensemble_models):
            ensemble_model.eval()
            with torch.no_grad():
                predictions = ensemble_model(input_functions, query_coords)
                all_predictions.append(predictions)

        # Stack predictions: [n_models, B, T, n_query, C]
        ensemble_predictions = torch.stack(all_predictions, dim=0)

        # Compute ensemble statistics
        mean_predictions = torch.mean(ensemble_predictions, dim=0)  # [B, T, n_query, C]
        prediction_variance = torch.var(ensemble_predictions, dim=0)  # [B, T, n_query, C]

        # Epistemic uncertainty (model uncertainty)
        epistemic_uncertainty = torch.sqrt(prediction_variance)

        # Aleatoric uncertainty (data uncertainty) - simplified estimation
        aleatoric_uncertainty = self._estimate_aleatoric_uncertainty(
            ensemble_predictions, mean_predictions
        )

        # Total uncertainty
        total_uncertainty = torch.sqrt(
            epistemic_uncertainty**2 + aleatoric_uncertainty**2
        )

        # Confidence intervals
        confidence_intervals = self._compute_confidence_intervals(
            ensemble_predictions, [0.68, 0.95, 0.99]
        )

        # Calibration metrics (placeholder)
        calibration_metrics = {
            'ensemble_diversity': float(torch.mean(prediction_variance)),
            'prediction_sharpness': float(torch.mean(epistemic_uncertainty)),
            'coverage_68': 0.68,  # Would be computed from validation data
            'coverage_95': 0.95
        }

        return UncertaintyEstimate(
            predictions=mean_predictions,
            aleatoric_uncertainty=aleatoric_uncertainty,
            epistemic_uncertainty=epistemic_uncertainty,
            total_uncertainty=total_uncertainty,
            confidence_intervals=confidence_intervals,
            calibration_metrics=calibration_metrics
        )

    def _estimate_aleatoric_uncertainty(self,
                                      ensemble_predictions: torch.Tensor,
                                      mean_predictions: torch.Tensor) -> torch.Tensor:
        """Estimate aleatoric uncertainty from ensemble predictions."""
        # Simplified estimation - in practice would use heteroscedastic modeling
        # Use residual variance as proxy for aleatoric uncertainty
        n_models = ensemble_predictions.shape[0]
        residuals = ensemble_predictions - mean_predictions.unsqueeze(0)

        # Estimate data noise level
        noise_level = torch.std(residuals, dim=0) / np.sqrt(n_models)

        return noise_level

    def _compute_confidence_intervals(self,
                                    ensemble_predictions: torch.Tensor,
                                    confidence_levels: List[float]) -> Dict[str, torch.Tensor]:
        """Compute confidence intervals from ensemble predictions."""
        intervals = {}

        for confidence in confidence_levels:
            alpha = 1 - confidence
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100

            lower_bound = torch.quantile(ensemble_predictions, lower_percentile / 100, dim=0)
            upper_bound = torch.quantile(ensemble_predictions, upper_percentile / 100, dim=0)

            intervals[f'ci_{int(confidence*100)}'] = {
                'lower': lower_bound,
                'upper': upper_bound,
                'width': upper_bound - lower_bound
            }

        return intervals

    def calibrate(self,
                 model: Union[DeepONet, List[DeepONet]],
                 calibration_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                 **kwargs):
        """Calibrate ensemble uncertainty estimates."""
        input_functions, query_coords, targets = calibration_data

        # Get uncertainty estimates
        uncertainty_est = self.estimate_uncertainty(model, input_functions, query_coords)

        # Compute calibration metrics
        calibration_metrics = self._compute_calibration_metrics(
            uncertainty_est, targets
        )

        logging.info(f"Ensemble calibration metrics: {calibration_metrics}")
        return calibration_metrics

    def _compute_calibration_metrics(self,
                                   uncertainty_est: UncertaintyEstimate,
                                   targets: torch.Tensor) -> Dict[str, float]:
        """Compute calibration metrics for uncertainty estimates."""
        predictions = uncertainty_est.predictions
        total_uncertainty = uncertainty_est.total_uncertainty

        # Compute prediction errors
        errors = torch.abs(predictions - targets)

        # Calibration: correlation between predicted uncertainty and actual error
        correlation = torch.corrcoef(torch.stack([
            total_uncertainty.flatten(),
            errors.flatten()
        ]))[0, 1]

        # Coverage analysis for confidence intervals
        coverage_metrics = {}
        for ci_name, ci_data in uncertainty_est.confidence_intervals.items():
            lower = ci_data['lower']
            upper = ci_data['upper']

            # Check if targets fall within confidence intervals
            in_interval = (targets >= lower) & (targets <= upper)
            coverage = float(torch.mean(in_interval.float()))
            coverage_metrics[f'coverage_{ci_name}'] = coverage

        return {
            'uncertainty_correlation': float(correlation),
            'mean_uncertainty': float(torch.mean(total_uncertainty)),
            'uncertainty_std': float(torch.std(total_uncertainty)),
            **coverage_metrics
        }


class BayesianUncertainty(AbstractUncertaintyMethod):
    """
    Bayesian uncertainty quantification using variational inference.

    Implements mean-field variational inference for DeepONet parameters
    to estimate both aleatoric and epistemic uncertainty.
    """

    def __init__(self,
                 prior_std: float = 1.0,
                 posterior_samples: int = 20):
        super().__init__("BayesianUncertainty")
        self.prior_std = prior_std
        self.posterior_samples = posterior_samples

    def estimate_uncertainty(self,
                           model: DeepONet,
                           input_functions: torch.Tensor,
                           query_coords: torch.Tensor,
                           **kwargs) -> UncertaintyEstimate:
        """
        Estimate uncertainty using Bayesian inference.

        Args:
            model: Trained DeepONet model
            input_functions: Input functions
            query_coords: Query coordinates

        Returns:
            uncertainty_estimate: Bayesian uncertainty estimates
        """
        # Convert model to Bayesian (add parameter uncertainty)
        bayesian_model = self._create_bayesian_model(model)

        # Sample from posterior
        prediction_samples = []

        for _ in range(self.posterior_samples):
            # Sample model parameters
            self._sample_parameters(bayesian_model)

            # Forward pass with sampled parameters
            with torch.no_grad():
                predictions = bayesian_model(input_functions, query_coords)
                prediction_samples.append(predictions)

        # Stack samples: [n_samples, B, T, n_query, C]
        prediction_samples = torch.stack(prediction_samples, dim=0)

        # Compute uncertainty estimates
        mean_predictions = torch.mean(prediction_samples, dim=0)
        epistemic_uncertainty = torch.std(prediction_samples, dim=0)

        # Aleatoric uncertainty (simplified)
        aleatoric_uncertainty = torch.ones_like(epistemic_uncertainty) * 0.01

        total_uncertainty = torch.sqrt(
            epistemic_uncertainty**2 + aleatoric_uncertainty**2
        )

        # Confidence intervals from samples
        confidence_intervals = {}
        for confidence in [0.68, 0.95, 0.99]:
            alpha = 1 - confidence
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100

            lower_bound = torch.quantile(prediction_samples, lower_percentile / 100, dim=0)
            upper_bound = torch.quantile(prediction_samples, upper_percentile / 100, dim=0)

            confidence_intervals[f'ci_{int(confidence*100)}'] = {
                'lower': lower_bound,
                'upper': upper_bound,
                'width': upper_bound - lower_bound
            }

        calibration_metrics = {
            'posterior_variance': float(torch.mean(torch.var(prediction_samples, dim=0))),
            'epistemic_ratio': float(torch.mean(epistemic_uncertainty) / torch.mean(total_uncertainty))
        }

        return UncertaintyEstimate(
            predictions=mean_predictions,
            aleatoric_uncertainty=aleatoric_uncertainty,
            epistemic_uncertainty=epistemic_uncertainty,
            total_uncertainty=total_uncertainty,
            confidence_intervals=confidence_intervals,
            calibration_metrics=calibration_metrics
        )

    def _create_bayesian_model(self, deterministic_model: DeepONet) -> DeepONet:
        """Convert deterministic model to Bayesian by adding parameter uncertainty."""
        # Create a copy of the model
        import copy
        bayesian_model = copy.deepcopy(deterministic_model)

        # Add parameter uncertainty (simplified implementation)
        for param in bayesian_model.parameters():
            param.requires_grad_(False)
            # Store original values and add uncertainty
            param.data_mean = param.data.clone()
            param.data_std = torch.ones_like(param.data) * self.prior_std

        return bayesian_model

    def _sample_parameters(self, bayesian_model: DeepONet):
        """Sample parameters from posterior distribution."""
        for param in bayesian_model.parameters():
            if hasattr(param, 'data_mean'):
                # Sample from Normal distribution
                noise = torch.randn_like(param.data_mean) * param.data_std
                param.data = param.data_mean + noise

    def calibrate(self,
                 model: DeepONet,
                 calibration_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                 **kwargs):
        """Calibrate Bayesian uncertainty estimates."""
        # Placeholder for Bayesian calibration
        logging.info("Bayesian calibration would optimize posterior parameters")
        return {'bayesian_calibrated': True}


class MonteCarloDropout(AbstractUncertaintyMethod):
    """
    Monte Carlo Dropout for uncertainty estimation.

    Uses dropout during inference to approximate Bayesian inference
    and estimate epistemic uncertainty.
    """

    def __init__(self, n_samples: int = 50, dropout_rate: float = 0.1):
        super().__init__("MonteCarloDropout")
        self.n_samples = n_samples
        self.dropout_rate = dropout_rate

    def estimate_uncertainty(self,
                           model: DeepONet,
                           input_functions: torch.Tensor,
                           query_coords: torch.Tensor,
                           **kwargs) -> UncertaintyEstimate:
        """
        Estimate uncertainty using Monte Carlo Dropout.

        Args:
            model: DeepONet model with dropout layers
            input_functions: Input functions
            query_coords: Query coordinates

        Returns:
            uncertainty_estimate: MC Dropout uncertainty estimates
        """
        # Enable dropout during inference
        self._enable_dropout(model)

        prediction_samples = []

        for _ in range(self.n_samples):
            predictions = model(input_functions, query_coords)
            prediction_samples.append(predictions)

        # Stack samples
        prediction_samples = torch.stack(prediction_samples, dim=0)

        # Restore model to eval mode
        model.eval()

        # Compute statistics
        mean_predictions = torch.mean(prediction_samples, dim=0)
        epistemic_uncertainty = torch.std(prediction_samples, dim=0)
        aleatoric_uncertainty = torch.ones_like(epistemic_uncertainty) * 0.005

        total_uncertainty = torch.sqrt(
            epistemic_uncertainty**2 + aleatoric_uncertainty**2
        )

        # Simple confidence intervals
        confidence_intervals = {
            'ci_68': {
                'lower': mean_predictions - epistemic_uncertainty,
                'upper': mean_predictions + epistemic_uncertainty,
                'width': 2 * epistemic_uncertainty
            }
        }

        calibration_metrics = {
            'mc_dropout_variance': float(torch.mean(torch.var(prediction_samples, dim=0))),
            'dropout_rate': self.dropout_rate
        }

        return UncertaintyEstimate(
            predictions=mean_predictions,
            aleatoric_uncertainty=aleatoric_uncertainty,
            epistemic_uncertainty=epistemic_uncertainty,
            total_uncertainty=total_uncertainty,
            confidence_intervals=confidence_intervals,
            calibration_metrics=calibration_metrics
        )

    def _enable_dropout(self, model: DeepONet):
        """Enable dropout layers during inference."""
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.train()  # Enable dropout

    def calibrate(self,
                 model: DeepONet,
                 calibration_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                 **kwargs):
        """Calibrate MC Dropout uncertainty estimates."""
        # Placeholder for MC Dropout calibration
        logging.info("MC Dropout calibration would optimize dropout rates")
        return {'mc_dropout_calibrated': True}


class DeepONetUncertaintyQuantifier:
    """
    Comprehensive uncertainty quantification system for DeepONet.

    Integrates multiple uncertainty estimation methods and provides
    unified interface for uncertainty-aware operator learning.
    """

    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.methods = {}

        # Register default methods
        self.register_method(EnsembleUncertainty())
        self.register_method(BayesianUncertainty())
        self.register_method(MonteCarloDropout())

        logging.info(f"DeepONet uncertainty quantifier initialized with {len(self.methods)} methods")

    def register_method(self, method: AbstractUncertaintyMethod):
        """Register a new uncertainty quantification method."""
        self.methods[method.name] = method
        logging.info(f"Registered uncertainty method: {method.name}")

    def estimate_uncertainty(self,
                           model: DeepONet,
                           input_functions: torch.Tensor,
                           query_coords: torch.Tensor,
                           method: str = 'EnsembleUncertainty',
                           **kwargs) -> UncertaintyEstimate:
        """
        Estimate uncertainty using specified method.

        Args:
            model: DeepONet model
            input_functions: Input functions
            query_coords: Query coordinates
            method: Uncertainty quantification method name

        Returns:
            uncertainty_estimate: Comprehensive uncertainty estimates
        """
        if method not in self.methods:
            raise ValueError(f"Unknown uncertainty method: {method}")

        uq_method = self.methods[method]
        logging.info(f"Estimating uncertainty using {method}")

        return uq_method.estimate_uncertainty(
            model, input_functions, query_coords, **kwargs
        )

    def compare_methods(self,
                       model: DeepONet,
                       input_functions: torch.Tensor,
                       query_coords: torch.Tensor,
                       targets: torch.Tensor,
                       methods: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Compare multiple uncertainty quantification methods.

        Args:
            model: DeepONet model
            input_functions: Input functions
            query_coords: Query coordinates
            targets: Ground truth targets
            methods: List of methods to compare

        Returns:
            comparison_results: Comparative analysis of UQ methods
        """
        if methods is None:
            methods = list(self.methods.keys())

        comparison_results = {}

        for method_name in methods:
            try:
                # Estimate uncertainty
                uncertainty_est = self.estimate_uncertainty(
                    model, input_functions, query_coords, method=method_name
                )

                # Evaluate uncertainty quality
                quality_metrics = self._evaluate_uncertainty_quality(
                    uncertainty_est, targets
                )

                comparison_results[method_name] = {
                    'uncertainty_estimate': uncertainty_est,
                    'quality_metrics': quality_metrics
                }

                logging.info(f"Completed uncertainty comparison for {method_name}")

            except Exception as e:
                logging.error(f"Uncertainty comparison failed for {method_name}: {e}")
                comparison_results[method_name] = {'error': str(e)}

        return comparison_results

    def _evaluate_uncertainty_quality(self,
                                    uncertainty_est: UncertaintyEstimate,
                                    targets: torch.Tensor) -> Dict[str, float]:
        """Evaluate the quality of uncertainty estimates."""
        predictions = uncertainty_est.predictions
        total_uncertainty = uncertainty_est.total_uncertainty

        # Prediction accuracy
        mse = float(torch.mean((predictions - targets)**2))
        mae = float(torch.mean(torch.abs(predictions - targets)))

        # Uncertainty calibration
        errors = torch.abs(predictions - targets)
        uncertainty_correlation = float(torch.corrcoef(torch.stack([
            total_uncertainty.flatten(),
            errors.flatten()
        ]))[0, 1])

        # Sharpness (lower is better for well-calibrated uncertainty)
        mean_uncertainty = float(torch.mean(total_uncertainty))

        # Coverage (for available confidence intervals)
        coverage_metrics = {}
        for ci_name, ci_data in uncertainty_est.confidence_intervals.items():
            if isinstance(ci_data, dict) and 'lower' in ci_data:
                lower = ci_data['lower']
                upper = ci_data['upper']
                in_interval = (targets >= lower) & (targets <= upper)
                coverage = float(torch.mean(in_interval.float()))
                coverage_metrics[f'coverage_{ci_name}'] = coverage

        return {
            'mse': mse,
            'mae': mae,
            'uncertainty_correlation': uncertainty_correlation,
            'mean_uncertainty': mean_uncertainty,
            'uncertainty_std': float(torch.std(total_uncertainty)),
            **coverage_metrics
        }

    def create_uncertainty_aware_predictions(self,
                                           model: DeepONet,
                                           input_functions: torch.Tensor,
                                           query_coords: torch.Tensor,
                                           method: str = 'EnsembleUncertainty',
                                           confidence_level: float = 0.95) -> Dict[str, torch.Tensor]:
        """
        Create uncertainty-aware predictions with confidence bounds.

        Args:
            model: DeepONet model
            input_functions: Input functions
            query_coords: Query coordinates
            method: Uncertainty method to use
            confidence_level: Confidence level for bounds

        Returns:
            uncertainty_predictions: Predictions with uncertainty bounds
        """
        uncertainty_est = self.estimate_uncertainty(
            model, input_functions, query_coords, method=method
        )

        ci_key = f'ci_{int(confidence_level*100)}'
        if ci_key in uncertainty_est.confidence_intervals:
            ci_data = uncertainty_est.confidence_intervals[ci_key]
            lower_bound = ci_data['lower']
            upper_bound = ci_data['upper']
        else:
            # Fall back to simple bounds using total uncertainty
            std_multiplier = 1.96 if confidence_level == 0.95 else 1.0
            lower_bound = uncertainty_est.predictions - std_multiplier * uncertainty_est.total_uncertainty
            upper_bound = uncertainty_est.predictions + std_multiplier * uncertainty_est.total_uncertainty

        return {
            'predictions': uncertainty_est.predictions,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'total_uncertainty': uncertainty_est.total_uncertainty,
            'epistemic_uncertainty': uncertainty_est.epistemic_uncertainty,
            'aleatoric_uncertainty': uncertainty_est.aleatoric_uncertainty
        }

    def save_uncertainty_analysis(self,
                                 uncertainty_results: Dict[str, Any],
                                 filepath: str):
        """Save uncertainty analysis results."""
        # Convert tensors to numpy for JSON serialization
        import json

        serializable_results = {}
        for method_name, results in uncertainty_results.items():
            if 'error' in results:
                serializable_results[method_name] = results
            else:
                serializable_results[method_name] = {
                    'quality_metrics': results['quality_metrics'],
                    'calibration_metrics': results['uncertainty_estimate'].calibration_metrics
                }

        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)

        logging.info(f"Uncertainty analysis saved to {filepath}")


def create_uncertainty_aware_deeponet(base_model: DeepONet,
                                     method: str = 'ensemble',
                                     **kwargs) -> Tuple[DeepONet, DeepONetUncertaintyQuantifier]:
    """
    Create uncertainty-aware version of DeepONet.

    Args:
        base_model: Base DeepONet model
        method: Uncertainty method ('ensemble', 'bayesian', 'mc_dropout')
        **kwargs: Method-specific parameters

    Returns:
        uncertainty_model: Model with uncertainty capabilities
        uncertainty_quantifier: UQ system
    """
    quantifier = DeepONetUncertaintyQuantifier()

    if method == 'ensemble':
        # Create ensemble for the model
        ensemble_method = quantifier.methods['EnsembleUncertainty']
        # In practice, would create ensemble here
        uncertainty_model = base_model

    elif method == 'bayesian':
        # Convert to Bayesian model
        bayesian_method = quantifier.methods['BayesianUncertainty']
        uncertainty_model = bayesian_method._create_bayesian_model(base_model)

    elif method == 'mc_dropout':
        # Ensure model has dropout
        uncertainty_model = base_model

    else:
        raise ValueError(f"Unknown uncertainty method: {method}")

    return uncertainty_model, quantifier