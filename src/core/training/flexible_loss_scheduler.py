"""
Flexible Loss Scheduler - Adaptive transition between any two losses.

Supports scheduling between:
- field_error (relative MSE in real space)
- lsim (learned perceptual similarity)
- spectrum_error (relative MSE in log power spectrum)

Safety features:
- Weight caps (max_weight_target)
- Warmup periods (warmup_epochs)
- Adaptive step sizing (smaller steps near final weights)
- Stability monitoring (max_consecutive_adaptations)
- EMA smoothing for validation metrics

Author: GenStabilisation-Proj
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, List
import torch
import numpy as np
import math
import logging


@dataclass
class FlexibleLossSchedulerConfig:
    """
    Configuration for flexible loss scheduling.

    Example (Field Error → Spectrum Error):
    ```python
    config = FlexibleLossSchedulerConfig(
        enabled=True,
        source_loss="field_error",
        target_loss="spectrum_error",
        source_components={"recFieldError": True, "predFieldError": True},
        target_components={"spectrumError": True},
        final_weight_source=0.3,
        final_weight_target=0.7
    )
    ```
    """

    # Basic control
    enabled: bool = False  # Default: OFF (backward compatible)

    # Loss specification
    source_loss: str = "field_error"      # "field_error", "lsim", "spectrum_error"
    target_loss: str = "spectrum_error"   # "field_error", "lsim", "spectrum_error"

    # Component mapping (which loss components use which schedule)
    source_components: Dict[str, bool] = field(default_factory=lambda: {
        'recFieldError': True,
        'predFieldError': True
    })
    target_components: Dict[str, bool] = field(default_factory=lambda: {
        'spectrumError': True
    })
    static_components: Dict[str, float] = field(default_factory=dict)

    # Weight schedule - CONSERVATIVE DEFAULTS
    initial_weight_source: float = 1.0
    initial_weight_target: float = 0.0
    final_weight_source: float = 0.3       # Keep 30% source (stable)
    final_weight_target: float = 0.7       # Max 70% target
    max_weight_target: float = 0.8         # Safety cap (prevents overfitting to target)

    # Adaptation parameters
    adaptation_trigger: str = "plateau"    # "plateau", "increase", "epoch"
    patience: int = 5                      # Epochs to wait before adapting
    min_delta: float = 0.001               # Minimum improvement threshold
    transition_step: float = 0.1           # Base step size (10%)
    adaptive_step: bool = True             # Scale step by progress
    min_epochs_before_adapt: int = 10      # Don't adapt too early
    warmup_epochs: int = 5                 # Additional warmup before adaptation
    max_consecutive_adaptations: int = 10  # Stop if unstable

    # Monitoring
    monitor_metric: str = "spectrum_error"  # What to monitor: "target_loss", "source_loss", "val_loss"
    use_ema: bool = True                    # Use EMA smoothing
    ema_beta: float = 0.9                   # EMA smoothing factor

    # Transition schedule type
    transition_schedule: str = "linear"     # "linear", "cosine", "exponential"


class FlexibleLossScheduler:
    """
    Adaptive scheduler for transitioning between any two losses.

    Key features:
    - Any-to-any scheduling (field_error ↔ lsim ↔ spectrum_error)
    - Component-level control (rec/pred separate, static weights)
    - Safety features (caps, warmup, stability monitoring)
    - Flexible adaptation triggers (plateau, increase, epoch-based)
    - EMA smoothing for stable decisions

    Example usage:
    ```python
    # Create scheduler
    scheduler = FlexibleLossScheduler(config)

    # During validation
    validation_metrics = {
        'field_error': 0.05,
        'spectrum_error': 0.12,
        'val_loss': 0.08
    }
    adapted = scheduler.step(epoch, validation_metrics)

    # Get current weights for training
    weights = scheduler.get_loss_weights()
    # {'recFieldError': 0.7, 'predFieldError': 0.7, 'spectrumError': 0.3}
    ```
    """

    def __init__(self, config: FlexibleLossSchedulerConfig):
        """
        Initialize flexible loss scheduler.

        Args:
            config: FlexibleLossSchedulerConfig instance
        """
        self.config = config
        self.enabled = config.enabled

        if not self.enabled:
            logging.info("[FlexibleLossScheduler] Disabled - using static weights")
            return

        # Validate loss types
        valid_losses = {"field_error", "lsim", "spectrum_error"}
        if config.source_loss not in valid_losses:
            raise ValueError(f"Invalid source_loss: {config.source_loss}. Must be one of {valid_losses}")
        if config.target_loss not in valid_losses:
            raise ValueError(f"Invalid target_loss: {config.target_loss}. Must be one of {valid_losses}")
        if config.source_loss == config.target_loss:
            raise ValueError(f"source_loss and target_loss must be different")

        # Current weights
        self.weight_source = config.initial_weight_source
        self.weight_target = config.initial_weight_target

        # Tracking
        self.monitored_metric_history = []
        self.best_monitored_metric = float('inf')
        self.epochs_since_improvement = 0
        self.monitored_metric_ema = None

        self.total_adaptations = 0
        self.consecutive_adaptations = 0
        self.adaptation_history = []

        self.warmup_complete = False
        self.current_epoch = 0

        logging.info(f"[FlexibleLossScheduler] Initialized")
        logging.info(f"  Schedule: {config.source_loss} → {config.target_loss}")
        logging.info(f"  Initial weights: source={config.initial_weight_source}, target={config.initial_weight_target}")
        logging.info(f"  Final weights: source={config.final_weight_source}, target={config.final_weight_target}")
        logging.info(f"  Monitor: {config.monitor_metric}, Patience: {config.patience}")
        logging.info(f"  Warmup: {config.warmup_epochs} epochs, Min epochs: {config.min_epochs_before_adapt}")

    def step(self, epoch: int, validation_metrics: Dict[str, float]) -> bool:
        """
        Update scheduler based on validation metrics.

        Args:
            epoch: Current epoch number
            validation_metrics: Dict containing metrics:
                - 'field_error': float
                - 'lsim': float (if 2D)
                - 'spectrum_error': float
                - 'val_loss': float

        Returns:
            adapted: Whether weights were adapted this step
        """
        if not self.enabled:
            return False

        self.current_epoch = epoch

        # Check warmup
        total_min_epochs = self.config.min_epochs_before_adapt + self.config.warmup_epochs
        if epoch < total_min_epochs:
            if epoch == total_min_epochs - 1:
                self.warmup_complete = True
                logging.info(f"[FlexibleLossScheduler] Warmup complete at epoch {epoch}. Adaptation enabled from epoch {epoch+1}")
            return False

        # Get monitored metric
        monitored_metric = self._get_monitored_metric(validation_metrics)
        self.monitored_metric_history.append(monitored_metric)

        # Update EMA
        if self.config.use_ema:
            if self.monitored_metric_ema is None:
                self.monitored_metric_ema = monitored_metric
            else:
                self.monitored_metric_ema = (
                    self.config.ema_beta * self.monitored_metric_ema +
                    (1 - self.config.ema_beta) * monitored_metric
                )
            tracked_metric = self.monitored_metric_ema
        else:
            tracked_metric = monitored_metric

        # Stability check
        if self.consecutive_adaptations >= self.config.max_consecutive_adaptations:
            logging.warning(
                f"[FlexibleLossScheduler] {self.consecutive_adaptations} consecutive adaptations detected. "
                f"Pausing adaptation to prevent instability."
            )
            self.consecutive_adaptations = 0
            return False

        # Check for adaptation based on trigger type
        adapted = False
        if self.config.adaptation_trigger == "plateau":
            adapted = self._check_plateau_and_adapt(epoch, tracked_metric)
        elif self.config.adaptation_trigger == "increase":
            adapted = self._check_increase_and_adapt(epoch, tracked_metric)
        elif self.config.adaptation_trigger == "epoch":
            # Fixed epoch schedule (not yet implemented)
            pass

        # Update consecutive counter
        if adapted:
            self.consecutive_adaptations += 1
        else:
            self.consecutive_adaptations = 0

        return adapted

    def _get_monitored_metric(self, validation_metrics: Dict[str, float]) -> float:
        """Extract the metric we're monitoring from validation results"""
        # Map monitor_metric config to actual metric name
        if self.config.monitor_metric == "target_loss":
            metric_name = self.config.target_loss
        elif self.config.monitor_metric == "source_loss":
            metric_name = self.config.source_loss
        elif self.config.monitor_metric == "val_loss":
            metric_name = "val_loss"
        else:
            raise ValueError(f"Unknown monitor_metric: {self.config.monitor_metric}")

        if metric_name not in validation_metrics:
            available = list(validation_metrics.keys())
            raise ValueError(
                f"Monitored metric '{metric_name}' not found in validation_metrics. "
                f"Available: {available}"
            )

        return validation_metrics[metric_name]

    def _check_plateau_and_adapt(self, epoch: int, tracked_metric: float) -> bool:
        """Check for plateau in monitored metric and adapt if needed"""
        # Check for improvement
        if tracked_metric < (self.best_monitored_metric - self.config.min_delta):
            # Improvement detected
            self.best_monitored_metric = tracked_metric
            self.epochs_since_improvement = 0
            return False
        else:
            # No improvement or got worse
            self.epochs_since_improvement += 1

            # Trigger adaptation?
            if self.epochs_since_improvement >= self.config.patience:
                adapted = self._adapt_weights(epoch, tracked_metric)
                if adapted:
                    self.epochs_since_improvement = 0
                    self.best_monitored_metric = tracked_metric
                return adapted

        return False

    def _check_increase_and_adapt(self, epoch: int, tracked_metric: float) -> bool:
        """Check for metric increase (got worse) and adapt immediately"""
        if len(self.monitored_metric_history) < 2:
            return False

        # Compare to previous epoch
        if tracked_metric > self.monitored_metric_history[-2]:
            # Metric increased (got worse) - adapt immediately
            return self._adapt_weights(epoch, tracked_metric)

        return False

    def _adapt_weights(self, epoch: int, current_metric: float) -> bool:
        """
        Adapt weights from source to target.

        Uses adaptive step sizing: smaller steps as we approach final weights.

        Args:
            epoch: Current epoch
            current_metric: Current value of monitored metric

        Returns:
            True if adaptation occurred, False if already at final weights
        """
        # Check if already at final weights
        if (abs(self.weight_source - self.config.final_weight_source) < 0.01 and
            abs(self.weight_target - self.config.final_weight_target) < 0.01):
            logging.info(f"[FlexibleLossScheduler] Already at final weights, no adaptation needed")
            return False

        # Compute step size
        if self.config.adaptive_step:
            # Adaptive: smaller steps as we approach final weights
            progress = self._compute_progress()
            step = self.config.transition_step * (1 - progress)
            step = max(step, 0.05)  # Minimum step size
        else:
            # Fixed step size
            step = self.config.transition_step

        # Move weights toward final values
        if self.weight_source > self.config.final_weight_source:
            self.weight_source = max(self.config.final_weight_source,
                                    self.weight_source - step)
        else:
            self.weight_source = min(self.config.final_weight_source,
                                    self.weight_source + step)

        if self.weight_target < self.config.final_weight_target:
            self.weight_target = min(self.config.final_weight_target,
                                    self.weight_target + step)
        else:
            self.weight_target = max(self.config.final_weight_target,
                                    self.weight_target - step)

        # Apply max_weight_target safety cap
        if self.weight_target > self.config.max_weight_target:
            excess = self.weight_target - self.config.max_weight_target
            self.weight_target = self.config.max_weight_target
            self.weight_source += excess  # Redistribute excess to source
            logging.info(f"[FlexibleLossScheduler] Applied max_weight_target cap: {self.config.max_weight_target}")

        # Normalize to maintain total weight
        target_total = self.config.initial_weight_source + self.config.initial_weight_target
        current_total = self.weight_source + self.weight_target

        if current_total > 0:
            self.weight_source = self.weight_source * target_total / current_total
            self.weight_target = self.weight_target * target_total / current_total

        # Record adaptation
        self.total_adaptations += 1
        self.adaptation_history.append({
            'epoch': epoch,
            'weight_source': self.weight_source,
            'weight_target': self.weight_target,
            'monitored_metric': current_metric,
            'step_size': step,
            'progress': self._compute_progress()
        })

        logging.info(
            f"[FlexibleLossScheduler] Adaptation #{self.total_adaptations} at epoch {epoch}: "
            f"source={self.weight_source:.3f}, target={self.weight_target:.3f} (step={step:.3f})"
        )

        return True

    def _compute_progress(self) -> float:
        """
        Compute progress toward final weights (0.0 to 1.0).

        Used for adaptive step sizing.
        """
        initial_diff = abs(self.config.initial_weight_target - self.config.final_weight_target)
        current_diff = abs(self.weight_target - self.config.final_weight_target)

        if initial_diff == 0:
            return 1.0

        progress = 1.0 - (current_diff / initial_diff)
        return max(0.0, min(1.0, progress))

    def get_loss_weights(self) -> Dict[str, float]:
        """
        Get current weights for all loss components.

        Returns:
            Dict mapping component names to weights, e.g.:
            {
                'recFieldError': 0.7,
                'predFieldError': 0.7,
                'spectrumError': 0.3,
                'recLSIM': 0.0,
                'predLSIM': 0.0
            }
        """
        if not self.enabled:
            # Scheduler disabled - return static components or defaults
            return self.config.static_components if self.config.static_components else {}

        weights = {}

        # Apply source weight to source components
        for component, use in self.config.source_components.items():
            if use:
                weights[component] = self.weight_source

        # Apply target weight to target components
        for component, use in self.config.target_components.items():
            if use:
                weights[component] = self.weight_target

        # Apply static weights (always at specified value)
        for component, weight in self.config.static_components.items():
            weights[component] = weight

        return weights

    def get_scheduler_state(self) -> Dict:
        """
        Get current scheduler state for logging/monitoring.

        Returns:
            Dict with current state information
        """
        return {
            'weight_source': self.weight_source,
            'weight_target': self.weight_target,
            'epochs_since_improvement': self.epochs_since_improvement,
            'total_adaptations': self.total_adaptations,
            'consecutive_adaptations': self.consecutive_adaptations,
            'progress': self._compute_progress() if self.enabled else 0.0,
            'warmup_complete': self.warmup_complete,
            'best_metric': self.best_monitored_metric,
            'current_metric_ema': self.monitored_metric_ema
        }

    def state_dict(self) -> dict:
        """
        Get complete state for checkpointing.

        Returns:
            Dict containing all state needed to resume training
        """
        return {
            'weight_source': self.weight_source,
            'weight_target': self.weight_target,
            'best_monitored_metric': self.best_monitored_metric,
            'epochs_since_improvement': self.epochs_since_improvement,
            'monitored_metric_ema': self.monitored_metric_ema,
            'total_adaptations': self.total_adaptations,
            'consecutive_adaptations': self.consecutive_adaptations,
            'adaptation_history': self.adaptation_history,
            'monitored_metric_history': self.monitored_metric_history,
            'warmup_complete': self.warmup_complete,
            'current_epoch': self.current_epoch,
            'config': self.config
        }

    def load_state_dict(self, state_dict: dict):
        """
        Resume scheduler from checkpoint.

        Args:
            state_dict: State dict from previous training
        """
        self.weight_source = state_dict['weight_source']
        self.weight_target = state_dict['weight_target']
        self.best_monitored_metric = state_dict['best_monitored_metric']
        self.epochs_since_improvement = state_dict['epochs_since_improvement']
        self.monitored_metric_ema = state_dict.get('monitored_metric_ema')
        self.total_adaptations = state_dict.get('total_adaptations', 0)
        self.consecutive_adaptations = state_dict.get('consecutive_adaptations', 0)
        self.adaptation_history = state_dict.get('adaptation_history', [])
        self.monitored_metric_history = state_dict.get('monitored_metric_history', [])
        self.warmup_complete = state_dict.get('warmup_complete', False)
        self.current_epoch = state_dict.get('current_epoch', 0)

        logging.info(f"[FlexibleLossScheduler] Loaded state from checkpoint at epoch {self.current_epoch}")
        logging.info(f"  Weights: source={self.weight_source:.3f}, target={self.weight_target:.3f}")
        logging.info(f"  Total adaptations: {self.total_adaptations}")


if __name__ == "__main__":
    # Quick test
    print("Testing FlexibleLossScheduler...")

    # Create test configuration
    config = FlexibleLossSchedulerConfig(
        enabled=True,
        source_loss="field_error",
        target_loss="spectrum_error",
        initial_weight_source=1.0,
        initial_weight_target=0.0,
        final_weight_source=0.3,
        final_weight_target=0.7,
        patience=3,
        warmup_epochs=2,
        min_epochs_before_adapt=5
    )

    scheduler = FlexibleLossScheduler(config)

    # Simulate training
    print("\nSimulating training with plateau at epoch 10:")
    for epoch in range(20):
        # Simulate metrics
        if epoch < 10:
            spectrum_error = 0.5 - epoch * 0.03  # Improving
        else:
            spectrum_error = 0.2  # Plateau

        validation_metrics = {
            'field_error': 0.1,
            'spectrum_error': spectrum_error,
            'val_loss': 0.15
        }

        adapted = scheduler.step(epoch, validation_metrics)

        if adapted or epoch % 5 == 0:
            state = scheduler.get_scheduler_state()
            print(f"Epoch {epoch}: spectrum_error={spectrum_error:.4f}, "
                  f"source_weight={state['weight_source']:.3f}, "
                  f"target_weight={state['weight_target']:.3f}, "
                  f"adapted={adapted}")

    print("\n✓ Test completed successfully!")
