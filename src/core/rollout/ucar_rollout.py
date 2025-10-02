"""
UCAR (Universal Corrected Autoregressive) Rollout Implementation

This module implements the UCAR rollout algorithm for generative operators,
adapted from the DCAR methodology for Gen Stabilised framework compatibility.

UCAR enables long-term stable predictions by:
1. Autoregressive prediction using neural operator prior
2. Periodic correction using generative corrector model
3. Adaptive correction strength based on prediction stability
4. Memory-efficient processing for long sequences

Author: Phase 3 Implementation
"""

import logging
import time
import math
from typing import Dict, Optional, Any, Tuple, List, Union
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.core.models.generative_operator_model import GenerativeOperatorModel
from src.core.models.base_classes import DataFormatHandler
from src.core.utils.params import DataParams
from src.core.models.memory_optimization import MemoryOptimizer


class StabilityMonitor:
    """
    Monitors rollout stability and provides adaptive correction recommendations.

    Tracks various stability metrics to detect divergence, instability, or
    degradation in prediction quality during long rollouts.
    """

    def __init__(self, window_size: int = 10, stability_threshold: float = 1.5):
        """
        Initialize stability monitor.

        Args:
            window_size: Number of recent steps to consider for stability
            stability_threshold: Multiplier for detecting instability
        """
        self.window_size = window_size
        self.stability_threshold = stability_threshold

        # Stability tracking
        self.energy_history = []
        self.gradient_norm_history = []
        self.divergence_history = []
        self.prediction_variance_history = []

        # Physics constraints tracking
        self.mass_conservation_errors = []
        self.momentum_conservation_errors = []
        self.energy_conservation_errors = []

        self.step_count = 0

        logging.info(f"Initialized StabilityMonitor with window_size={window_size}")

    def update(self, prediction: torch.Tensor, previous_state: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Update stability metrics with new prediction.

        Args:
            prediction: Current prediction [B, T, C, H, W]
            previous_state: Previous state for temporal analysis

        Returns:
            stability_metrics: Dictionary of current stability metrics
        """
        metrics = {}

        # Energy analysis
        energy = self._compute_energy(prediction)
        self.energy_history.append(energy)
        metrics['energy'] = energy

        # Gradient analysis
        grad_norm = self._compute_gradient_norm(prediction)
        self.gradient_norm_history.append(grad_norm)
        metrics['gradient_norm'] = grad_norm

        # Divergence analysis (for incompressible flows)
        divergence = self._compute_divergence(prediction)
        self.divergence_history.append(divergence)
        metrics['divergence'] = divergence

        # Prediction variance
        variance = torch.var(prediction).item()
        self.prediction_variance_history.append(variance)
        metrics['variance'] = variance

        # Conservation law analysis
        if previous_state is not None:
            conservation_errors = self._check_conservation_laws(prediction, previous_state)
            metrics.update(conservation_errors)

        self.step_count += 1

        # Trim histories to maintain window size
        self._trim_histories()

        return metrics

    def get_stability_score(self) -> float:
        """
        Compute overall stability score.

        Returns:
            stability_score: 0-1 score, 1 = perfectly stable
        """
        if len(self.energy_history) < 2:
            return 1.0

        scores = []

        # Energy stability
        if len(self.energy_history) >= self.window_size:
            recent_energy = self.energy_history[-self.window_size:]
            energy_std = np.std(recent_energy)
            energy_mean = np.mean(recent_energy)
            energy_stability = 1.0 / (1.0 + energy_std / (energy_mean + 1e-8))
            scores.append(energy_stability)

        # Gradient stability
        if len(self.gradient_norm_history) >= self.window_size:
            recent_grads = self.gradient_norm_history[-self.window_size:]
            grad_growth = recent_grads[-1] / (recent_grads[0] + 1e-8)
            grad_stability = 1.0 / (1.0 + max(0, grad_growth - 1.0))
            scores.append(grad_stability)

        # Divergence stability (for fluid dynamics)
        if len(self.divergence_history) >= self.window_size:
            recent_div = self.divergence_history[-self.window_size:]
            div_score = 1.0 / (1.0 + np.mean(recent_div))
            scores.append(div_score)

        return np.mean(scores) if scores else 1.0

    def should_apply_correction(self) -> bool:
        """
        Determine if correction should be applied based on stability.

        Returns:
            apply_correction: True if correction is recommended
        """
        stability_score = self.get_stability_score()
        return stability_score < (1.0 / self.stability_threshold)

    def get_recommended_correction_strength(self) -> float:
        """
        Get recommended correction strength based on stability.

        Returns:
            correction_strength: 0-1 strength recommendation
        """
        stability_score = self.get_stability_score()

        # Higher correction strength for lower stability
        # Maps stability score 0-1 to correction strength 1-0.1
        return max(0.1, 1.1 - stability_score)

    def _compute_energy(self, prediction: torch.Tensor) -> float:
        """Compute total energy of the prediction."""
        return torch.sum(prediction ** 2).item()

    def _compute_gradient_norm(self, prediction: torch.Tensor) -> float:
        """Compute spatial gradient norm."""
        # Simple finite difference gradients
        grad_x = torch.diff(prediction, dim=-1)  # Width gradient
        grad_y = torch.diff(prediction, dim=-2)  # Height gradient

        grad_norm = torch.sqrt(torch.sum(grad_x ** 2) + torch.sum(grad_y ** 2))
        return grad_norm.item()

    def _compute_divergence(self, prediction: torch.Tensor) -> float:
        """Compute velocity divergence for incompressible flow constraint."""
        if prediction.shape[2] < 2:  # Need at least 2 velocity components
            return 0.0

        # Assume first 2 channels are velocity components
        u = prediction[:, :, 0]  # u velocity
        v = prediction[:, :, 1]  # v velocity

        # Compute divergence using finite differences
        du_dx = torch.diff(u, dim=-1)  # ∂u/∂x
        dv_dy = torch.diff(v, dim=-2)  # ∂v/∂y

        # Align shapes for divergence computation
        min_h = min(du_dx.shape[-2], dv_dy.shape[-2])
        min_w = min(du_dx.shape[-1], dv_dy.shape[-1])

        du_dx = du_dx[..., :min_h, :min_w]
        dv_dy = dv_dy[..., :min_h, :min_w]

        divergence = du_dx + dv_dy
        return torch.abs(divergence).mean().item()

    def _check_conservation_laws(self, current: torch.Tensor, previous: torch.Tensor) -> Dict[str, float]:
        """Check conservation law violations."""
        conservation_errors = {}

        # Mass conservation (for density-based flows)
        if current.shape[2] >= 1:  # Assume first channel is density or similar
            mass_current = torch.sum(current[:, :, 0])
            mass_previous = torch.sum(previous[:, :, 0])
            mass_error = torch.abs(mass_current - mass_previous).item()
            self.mass_conservation_errors.append(mass_error)
            conservation_errors['mass_conservation_error'] = mass_error

        # Energy conservation
        energy_current = self._compute_energy(current)
        energy_previous = self._compute_energy(previous)
        energy_error = abs(energy_current - energy_previous)
        self.energy_conservation_errors.append(energy_error)
        conservation_errors['energy_conservation_error'] = energy_error

        return conservation_errors

    def _trim_histories(self):
        """Trim history lists to maintain window size."""
        max_history = self.window_size * 2  # Keep extra for analysis

        if len(self.energy_history) > max_history:
            self.energy_history = self.energy_history[-max_history:]
        if len(self.gradient_norm_history) > max_history:
            self.gradient_norm_history = self.gradient_norm_history[-max_history:]
        if len(self.divergence_history) > max_history:
            self.divergence_history = self.divergence_history[-max_history:]
        if len(self.prediction_variance_history) > max_history:
            self.prediction_variance_history = self.prediction_variance_history[-max_history:]


class UCARRollout:
    """
    Universal Corrected Autoregressive Rollout for Generative Operators.

    Implements stable long-term rollout using:
    1. Neural operator prior for base predictions
    2. Generative corrector for refinement
    3. Adaptive correction based on stability monitoring
    4. Memory-efficient processing for long sequences
    """

    def __init__(self,
                 model: GenerativeOperatorModel,
                 data_params: Optional[DataParams] = None,
                 memory_efficient: bool = True,
                 stability_monitoring: bool = True):
        """
        Initialize UCAR rollout.

        Args:
            model: Trained generative operator model
            data_params: Data parameters for format validation
            memory_efficient: Enable memory optimizations
            stability_monitoring: Enable stability monitoring
        """
        self.model = model
        self.data_params = data_params
        self.memory_efficient = memory_efficient
        self.stability_monitoring = stability_monitoring

        # Components
        self.data_handler = DataFormatHandler()
        self.memory_optimizer = MemoryOptimizer() if memory_efficient else None
        self.stability_monitor = StabilityMonitor() if stability_monitoring else None

        # Rollout state
        self.device = next(model.parameters()).device
        self.rollout_history = []

        # Configuration
        self.default_correction_interval = 1
        self.default_correction_strength = 1.0
        self.max_sequence_length = 1000  # Safety limit

        logging.info(f"Initialized UCARRollout:")
        logging.info(f"  Device: {self.device}")
        logging.info(f"  Memory efficient: {memory_efficient}")
        logging.info(f"  Stability monitoring: {stability_monitoring}")

    def rollout_sequence(self,
                        initial_conditions: torch.Tensor,
                        num_steps: int,
                        correction_interval: int = 1,
                        correction_strength: float = 1.0,
                        correction_schedule: str = "constant",
                        save_intermediate: bool = False,
                        batch_processing: bool = True) -> Dict[str, torch.Tensor]:
        """
        Perform UCAR rollout for specified number of steps.

        Args:
            initial_conditions: Initial state [B, T_init, C, H, W]
            num_steps: Number of future steps to predict
            correction_interval: Apply correction every N steps (1 = every step)
            correction_strength: Base strength for corrections (0-1)
            correction_schedule: "constant", "decay", "adaptive"
            save_intermediate: Save intermediate predictions
            batch_processing: Process multiple sequences in parallel

        Returns:
            rollout_results: Dictionary containing predictions and metadata
        """
        start_time = time.time()

        # Validate inputs
        self._validate_inputs(initial_conditions, num_steps)

        # Prepare for rollout
        self.model.eval()
        predictions = []
        correction_strengths_used = []
        stability_scores = []

        # Initialize stability monitor
        if self.stability_monitor:
            self.stability_monitor = StabilityMonitor()  # Reset for new rollout

        # Get window size for autoregressive prediction
        window_size = getattr(self.model.prior_model, 'prev_steps', 1)
        window_size = getattr(self.model.prior_model, 'L', window_size)

        current_state = initial_conditions.to(self.device)

        logging.info(f"Starting UCAR rollout:")
        logging.info(f"  Steps: {num_steps}")
        logging.info(f"  Correction interval: {correction_interval}")
        logging.info(f"  Base correction strength: {correction_strength}")
        logging.info(f"  Window size: {window_size}")

        with torch.no_grad():
            for step in range(num_steps):
                step_start_time = time.time()

                # Memory management
                if self.memory_optimizer and step % 10 == 0:
                    self.memory_optimizer.clear_cache()

                # Prepare input window
                input_window = self._prepare_input_window(current_state, window_size)

                # Prior prediction
                self.model.set_training_mode('prior_only')
                prior_prediction = self.model.prior_model(input_window)
                next_frame = prior_prediction[:, -1:]  # Take last predicted frame

                # Determine if correction should be applied
                should_correct = (step % correction_interval == 0)
                actual_correction_strength = correction_strength

                # Adaptive correction based on stability
                if self.stability_monitoring and self.stability_monitor:
                    # Update stability metrics
                    if step > 0:
                        stability_metrics = self.stability_monitor.update(
                            next_frame,
                            predictions[-1] if predictions else initial_conditions[:, -1:]
                        )
                        stability_score = self.stability_monitor.get_stability_score()
                        stability_scores.append(stability_score)

                        # Adjust correction based on stability
                        if correction_schedule == "adaptive":
                            if self.stability_monitor.should_apply_correction():
                                should_correct = True
                                actual_correction_strength = self.stability_monitor.get_recommended_correction_strength()
                    else:
                        stability_scores.append(1.0)
                else:
                    stability_scores.append(1.0)

                # Apply correction schedule
                if correction_schedule == "decay":
                    decay_factor = max(0.1, 1.0 - step / num_steps)
                    actual_correction_strength *= decay_factor

                # Apply correction if needed
                if should_correct and self.model.correction_strength > 0:
                    corrected_frame = self._apply_correction(
                        next_frame, input_window, actual_correction_strength
                    )
                    final_frame = corrected_frame
                else:
                    final_frame = next_frame

                # Store results
                predictions.append(final_frame.cpu())
                correction_strengths_used.append(actual_correction_strength if should_correct else 0.0)

                # Update state for next iteration
                current_state = self._update_state(current_state, final_frame)

                # Progress logging
                step_time = time.time() - step_start_time
                if step % 20 == 0 or step == num_steps - 1:
                    stability_info = f", Stability: {stability_scores[-1]:.3f}" if stability_scores else ""
                    correction_info = f", Correction: {actual_correction_strength:.3f}" if should_correct else ""
                    logging.info(f"  Step {step+1}/{num_steps} ({step_time:.3f}s){stability_info}{correction_info}")

        # Compile results
        total_time = time.time() - start_time

        # Concatenate predictions
        full_predictions = torch.cat(predictions, dim=1)  # [B, num_steps, C, H, W]

        results = {
            'predictions': full_predictions,
            'initial_conditions': initial_conditions.cpu(),
            'correction_strengths_used': correction_strengths_used,
            'stability_scores': stability_scores,
            'num_steps': num_steps,
            'total_time': total_time,
            'steps_per_second': num_steps / total_time,
            'rollout_metadata': {
                'correction_interval': correction_interval,
                'base_correction_strength': correction_strength,
                'correction_schedule': correction_schedule,
                'stability_monitoring': self.stability_monitoring,
                'window_size': window_size
            }
        }

        # Add intermediate results if requested
        if save_intermediate:
            results['intermediate_predictions'] = [p.cpu() for p in predictions]

        logging.info(f"UCAR rollout completed:")
        logging.info(f"  Total time: {total_time:.2f}s")
        logging.info(f"  Speed: {results['steps_per_second']:.1f} steps/sec")
        logging.info(f"  Output shape: {full_predictions.shape}")

        return results

    def _apply_correction(self,
                         prediction: torch.Tensor,
                         context: torch.Tensor,
                         strength: float) -> torch.Tensor:
        """
        Apply generative correction to prediction.

        Args:
            prediction: Prior prediction to correct [B, 1, C, H, W]
            context: Context frames for conditioning
            strength: Correction strength (0-1)

        Returns:
            corrected_prediction: Corrected prediction
        """
        if strength <= 0:
            return prediction

        # Set model to full inference mode
        self.model.set_training_mode('full_inference')

        # Apply correction through the full model
        # The corrector will use the context to refine the prediction
        corrected = self.model.corrector_model.correct_prediction(
            prediction,
            prior_features=self.model.prior_model.get_prior_features(context) if hasattr(self.model.prior_model, 'get_prior_features') else None,
            correction_strength=strength * self.model.correction_strength
        )

        return corrected

    def _prepare_input_window(self, current_state: torch.Tensor, window_size: int) -> torch.Tensor:
        """
        Prepare input window for autoregressive prediction.

        Args:
            current_state: Current state [B, T, C, H, W]
            window_size: Required window size

        Returns:
            input_window: Window for prediction [B, window_size, C, H, W]
        """
        if current_state.shape[1] >= window_size:
            return current_state[:, -window_size:]
        else:
            # Pad with repeated frames if needed
            B, T, C, H, W = current_state.shape
            padding_needed = window_size - T

            # Repeat the last frame
            last_frame = current_state[:, -1:].repeat(1, padding_needed, 1, 1, 1)
            return torch.cat([current_state, last_frame], dim=1)

    def _update_state(self, current_state: torch.Tensor, new_prediction: torch.Tensor) -> torch.Tensor:
        """
        Update state with new prediction, maintaining temporal window.

        Args:
            current_state: Current state [B, T, C, H, W]
            new_prediction: New prediction [B, 1, C, H, W]

        Returns:
            updated_state: Updated state [B, T, C, H, W]
        """
        # Append new prediction and maintain window size
        updated = torch.cat([current_state, new_prediction], dim=1)

        # Keep only the most recent frames (maintain reasonable window)
        max_window = 20  # Reasonable maximum for memory
        if updated.shape[1] > max_window:
            updated = updated[:, -max_window:]

        return updated

    def _validate_inputs(self, initial_conditions: torch.Tensor, num_steps: int):
        """Validate rollout inputs."""
        # Shape validation
        assert len(initial_conditions.shape) == 5, f"Expected 5D tensor [B,T,C,H,W], got {initial_conditions.shape}"

        # Data format validation
        if not self.data_handler.validate_gen_stabilised_format(initial_conditions):
            warnings.warn("Input may not be in expected Gen Stabilised format")

        # Range validation
        assert 1 <= num_steps <= self.max_sequence_length, f"num_steps {num_steps} out of range [1, {self.max_sequence_length}]"

        logging.debug(f"Input validation passed: {initial_conditions.shape} for {num_steps} steps")

    def rollout_with_multiple_corrections(self,
                                        initial_conditions: torch.Tensor,
                                        num_steps: int,
                                        correction_configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Perform rollout with multiple correction configurations for comparison.

        Args:
            initial_conditions: Initial state
            num_steps: Number of steps
            correction_configs: List of correction configurations to try

        Returns:
            comparison_results: Results for each configuration
        """
        results = {}

        for i, config in enumerate(correction_configs):
            config_name = config.get('name', f'config_{i}')
            logging.info(f"Running rollout with configuration: {config_name}")

            rollout_result = self.rollout_sequence(
                initial_conditions=initial_conditions.clone(),
                num_steps=num_steps,
                correction_interval=config.get('correction_interval', 1),
                correction_strength=config.get('correction_strength', 1.0),
                correction_schedule=config.get('correction_schedule', 'constant')
            )

            results[config_name] = rollout_result

        return results

    def get_rollout_statistics(self) -> Dict[str, Any]:
        """Get statistics about recent rollouts."""
        return {
            'device': str(self.device),
            'memory_efficient': self.memory_efficient,
            'stability_monitoring': self.stability_monitoring,
            'rollout_history_length': len(self.rollout_history)
        }