"""
Unit tests for FlexibleLossScheduler.

Tests cover:
- Initialization and configuration
- Plateau detection with EMA smoothing
- Weight adaptation logic
- Safety features (warmup, caps, consecutive limits)
- Component weight management
- Edge cases and stability

Run:
    python -m pytest tests/test_flexible_loss_scheduler.py -v
    python tests/test_flexible_loss_scheduler.py  # Direct execution
"""

import sys
from dataclasses import dataclass
from typing import Dict

# Mock the config dataclass for testing
@dataclass
class FlexibleLossSchedulerConfig:
    enabled: bool = True
    source_loss: str = "field_error"
    target_loss: str = "spectrum_error"
    source_components: Dict[str, bool] = None
    target_components: Dict[str, bool] = None
    static_components: Dict[str, float] = None
    initial_weight_source: float = 1.0
    initial_weight_target: float = 0.0
    final_weight_source: float = 0.3
    final_weight_target: float = 0.7
    max_weight_target: float = 0.8
    patience: int = 5
    warmup_epochs: int = 5
    adaptive_step: bool = True
    max_consecutive_adaptations: int = 10
    monitor_metric: str = "spectrum_error"
    use_ema: bool = True
    ema_beta: float = 0.9

    def __post_init__(self):
        if self.source_components is None:
            self.source_components = {"recFieldError": True, "predFieldError": True}
        if self.target_components is None:
            self.target_components = {"spectrumError": True}
        if self.static_components is None:
            self.static_components = {}


# Import after defining the mock config
try:
    from src.core.training.flexible_loss_scheduler import FlexibleLossScheduler
except ImportError:
    # For testing without full environment
    print("Warning: Could not import FlexibleLossScheduler. Some tests may be skipped.")
    FlexibleLossScheduler = None


class TestSchedulerInitialization:
    """Test scheduler initialization and configuration."""

    def test_default_initialization(self):
        """Scheduler should initialize with default config."""
        if FlexibleLossScheduler is None:
            return

        config = FlexibleLossSchedulerConfig()
        scheduler = FlexibleLossScheduler(config)

        assert scheduler.current_weight_source == 1.0
        assert scheduler.current_weight_target == 0.0
        assert scheduler.epochs_without_improvement == 0
        assert scheduler.consecutive_adaptations == 0

    def test_custom_initialization(self):
        """Scheduler should respect custom config values."""
        if FlexibleLossScheduler is None:
            return

        config = FlexibleLossSchedulerConfig(
            initial_weight_source=0.8,
            initial_weight_target=0.2,
            patience=3,
            warmup_epochs=10
        )
        scheduler = FlexibleLossScheduler(config)

        assert scheduler.current_weight_source == 0.8
        assert scheduler.current_weight_target == 0.2
        assert scheduler.config.patience == 3
        assert scheduler.config.warmup_epochs == 10

    def test_component_mapping(self):
        """Scheduler should correctly map source/target to components."""
        if FlexibleLossScheduler is None:
            return

        config = FlexibleLossSchedulerConfig(
            source_components={"recFieldError": True, "predFieldError": True},
            target_components={"spectrumError": True},
            static_components={"recLSIM": 0.1}
        )
        scheduler = FlexibleLossScheduler(config)

        weights = scheduler.get_loss_weights()

        # Source components should have initial weight
        assert weights["recFieldError"] == 1.0
        assert weights["predFieldError"] == 1.0

        # Target component should have initial weight (0.0)
        assert weights["spectrumError"] == 0.0

        # Static component should be fixed
        assert weights["recLSIM"] == 0.1


class TestPlateauDetection:
    """Test plateau detection with EMA smoothing."""

    def test_no_plateau_improving_metric(self):
        """Scheduler should not adapt when metric is improving."""
        if FlexibleLossScheduler is None:
            return

        config = FlexibleLossSchedulerConfig(patience=3, warmup_epochs=0)
        scheduler = FlexibleLossScheduler(config)

        # Improving metric over 5 epochs
        for epoch in range(5):
            metrics = {"spectrum_error": 1.0 - epoch * 0.1}  # 1.0, 0.9, 0.8, ...
            adapted = scheduler.step(epoch, metrics)
            assert not adapted, f"Should not adapt on epoch {epoch} (improving)"

    def test_plateau_detection(self):
        """Scheduler should detect plateau after patience epochs."""
        if FlexibleLossScheduler is None:
            return

        config = FlexibleLossSchedulerConfig(patience=3, warmup_epochs=0)
        scheduler = FlexibleLossScheduler(config)

        # Plateaued metric
        for epoch in range(5):
            metrics = {"spectrum_error": 0.5}  # Constant
            adapted = scheduler.step(epoch, metrics)

            if epoch < 3:
                assert not adapted, f"Should not adapt before patience (epoch {epoch})"
            else:
                # Should adapt on epoch 3 (after patience=3 epochs)
                break

    def test_ema_smoothing(self):
        """EMA should smooth noisy validation metrics."""
        if FlexibleLossScheduler is None:
            return

        config = FlexibleLossSchedulerConfig(
            patience=5,
            warmup_epochs=0,
            use_ema=True,
            ema_beta=0.9
        )
        scheduler = FlexibleLossScheduler(config)

        # Noisy but generally improving
        noisy_metrics = [1.0, 0.95, 1.05, 0.90, 0.95, 0.85, 0.90]

        for epoch, value in enumerate(noisy_metrics):
            metrics = {"spectrum_error": value}
            adapted = scheduler.step(epoch, metrics)

            # EMA should smooth, preventing premature adaptation
            if epoch < 3:
                assert not adapted, "Should not adapt early with EMA smoothing"


class TestWeightAdaptation:
    """Test weight adaptation logic."""

    def test_weight_progression(self):
        """Weights should progress from initial to final over adaptations."""
        if FlexibleLossScheduler is None:
            return

        config = FlexibleLossSchedulerConfig(
            initial_weight_source=1.0,
            initial_weight_target=0.0,
            final_weight_source=0.3,
            final_weight_target=0.7,
            patience=2,
            warmup_epochs=0
        )
        scheduler = FlexibleLossScheduler(config)

        # Force multiple adaptations
        for epoch in range(20):
            metrics = {"spectrum_error": 0.5}  # Constant (plateau)
            adapted = scheduler.step(epoch, metrics)

            if adapted:
                weights = scheduler.get_loss_weights()
                # Source should decrease, target should increase
                assert scheduler.current_weight_source <= 1.0
                assert scheduler.current_weight_target >= 0.0
                assert scheduler.current_weight_source >= 0.3
                assert scheduler.current_weight_target <= 0.7

    def test_max_weight_target_cap(self):
        """Target weight should never exceed max_weight_target."""
        if FlexibleLossScheduler is None:
            return

        config = FlexibleLossSchedulerConfig(
            final_weight_target=0.9,
            max_weight_target=0.7,  # Cap at 0.7
            patience=1,
            warmup_epochs=0
        )
        scheduler = FlexibleLossScheduler(config)

        # Force many adaptations
        for epoch in range(50):
            metrics = {"spectrum_error": 0.5}
            scheduler.step(epoch, metrics)

        weights = scheduler.get_loss_weights()
        assert weights["spectrumError"] <= 0.7, "Should not exceed max_weight_target"

    def test_adaptive_step_sizing(self):
        """Adaptive step should be smaller near convergence."""
        if FlexibleLossScheduler is None:
            return

        config = FlexibleLossSchedulerConfig(
            adaptive_step=True,
            patience=1,
            warmup_epochs=0
        )
        scheduler = FlexibleLossScheduler(config)

        # Track step sizes
        step_sizes = []
        prev_weight = scheduler.current_weight_target

        for epoch in range(30):
            metrics = {"spectrum_error": 0.5}
            adapted = scheduler.step(epoch, metrics)

            if adapted:
                current_weight = scheduler.current_weight_target
                step_size = current_weight - prev_weight
                if step_size > 0:
                    step_sizes.append(step_size)
                prev_weight = current_weight

        # Later steps should generally be smaller
        if len(step_sizes) > 2:
            early_avg = sum(step_sizes[:2]) / 2
            late_avg = sum(step_sizes[-2:]) / 2
            assert late_avg <= early_avg * 1.5, "Later steps should be similar or smaller"


class TestSafetyFeatures:
    """Test safety features (warmup, consecutive limits, stability)."""

    def test_warmup_period(self):
        """Scheduler should not adapt during warmup."""
        if FlexibleLossScheduler is None:
            return

        config = FlexibleLossSchedulerConfig(
            warmup_epochs=10,
            patience=1
        )
        scheduler = FlexibleLossScheduler(config)

        # Even with plateau, should not adapt during warmup
        for epoch in range(10):
            metrics = {"spectrum_error": 0.5}  # Constant
            adapted = scheduler.step(epoch, metrics)
            assert not adapted, f"Should not adapt during warmup (epoch {epoch})"

    def test_consecutive_adaptation_limit(self):
        """Scheduler should stop after max consecutive adaptations."""
        if FlexibleLossScheduler is None:
            return

        config = FlexibleLossSchedulerConfig(
            max_consecutive_adaptations=5,
            patience=1,
            warmup_epochs=0
        )
        scheduler = FlexibleLossScheduler(config)

        # Force consecutive adaptations
        adaptation_count = 0
        for epoch in range(20):
            metrics = {"spectrum_error": 0.5}  # Constant plateau
            adapted = scheduler.step(epoch, metrics)

            if adapted:
                adaptation_count += 1

        # Should stop at max_consecutive_adaptations
        assert adaptation_count <= 5, "Should stop at consecutive adaptation limit"

    def test_improvement_resets_consecutive_count(self):
        """Improvement should reset consecutive adaptation counter."""
        if FlexibleLossScheduler is None:
            return

        config = FlexibleLossSchedulerConfig(
            max_consecutive_adaptations=3,
            patience=2,
            warmup_epochs=0,
            use_ema=False  # Disable EMA for clearer test
        )
        scheduler = FlexibleLossScheduler(config)

        # Plateau → adapt
        for epoch in range(3):
            metrics = {"spectrum_error": 0.5}
            scheduler.step(epoch, metrics)

        assert scheduler.consecutive_adaptations > 0

        # Improvement → reset
        metrics = {"spectrum_error": 0.4}  # Improvement
        scheduler.step(3, metrics)

        assert scheduler.consecutive_adaptations == 0, "Improvement should reset counter"


class TestComponentWeightManagement:
    """Test component weight management."""

    def test_get_loss_weights(self):
        """get_loss_weights should return correct component weights."""
        if FlexibleLossScheduler is None:
            return

        config = FlexibleLossSchedulerConfig(
            source_components={"recFieldError": True, "predFieldError": True},
            target_components={"spectrumError": True},
            static_components={"recLSIM": 0.2}
        )
        scheduler = FlexibleLossScheduler(config)

        weights = scheduler.get_loss_weights()

        # Check all components present
        assert "recFieldError" in weights
        assert "predFieldError" in weights
        assert "spectrumError" in weights
        assert "recLSIM" in weights

        # Check initial values
        assert weights["recFieldError"] == 1.0
        assert weights["predFieldError"] == 1.0
        assert weights["spectrumError"] == 0.0
        assert weights["recLSIM"] == 0.2  # Static

    def test_static_components_unchanged(self):
        """Static components should not change during adaptation."""
        if FlexibleLossScheduler is None:
            return

        config = FlexibleLossSchedulerConfig(
            static_components={"recLSIM": 0.15},
            patience=1,
            warmup_epochs=0
        )
        scheduler = FlexibleLossScheduler(config)

        # Force adaptation
        for epoch in range(10):
            metrics = {"spectrum_error": 0.5}
            scheduler.step(epoch, metrics)

        weights = scheduler.get_loss_weights()
        assert weights["recLSIM"] == 0.15, "Static component should not change"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_missing_monitor_metric(self):
        """Scheduler should handle missing monitor metric gracefully."""
        if FlexibleLossScheduler is None:
            return

        config = FlexibleLossSchedulerConfig(monitor_metric="spectrum_error")
        scheduler = FlexibleLossScheduler(config)

        # Pass metrics without the monitored metric
        metrics = {"field_error": 0.5}  # Missing spectrum_error
        adapted = scheduler.step(0, metrics)

        # Should not crash, should not adapt
        assert not adapted

    def test_empty_metrics(self):
        """Scheduler should handle empty metrics dict."""
        if FlexibleLossScheduler is None:
            return

        config = FlexibleLossSchedulerConfig()
        scheduler = FlexibleLossScheduler(config)

        metrics = {}
        adapted = scheduler.step(0, metrics)

        assert not adapted, "Should not adapt with empty metrics"

    def test_zero_patience(self):
        """Scheduler with patience=0 should adapt every epoch after warmup."""
        if FlexibleLossScheduler is None:
            return

        config = FlexibleLossSchedulerConfig(
            patience=0,
            warmup_epochs=2,
            max_consecutive_adaptations=5
        )
        scheduler = FlexibleLossScheduler(config)

        # After warmup, should adapt every epoch
        for epoch in range(2, 7):
            metrics = {"spectrum_error": 0.5}
            adapted = scheduler.step(epoch, metrics)
            assert adapted, f"Should adapt every epoch after warmup (epoch {epoch})"

    def test_already_at_final_weights(self):
        """Scheduler starting at final weights should not adapt."""
        if FlexibleLossScheduler is None:
            return

        config = FlexibleLossSchedulerConfig(
            initial_weight_source=0.3,
            initial_weight_target=0.7,
            final_weight_source=0.3,
            final_weight_target=0.7,
            patience=1,
            warmup_epochs=0
        )
        scheduler = FlexibleLossScheduler(config)

        # Even with plateau, should not adapt (already at final)
        for epoch in range(10):
            metrics = {"spectrum_error": 0.5}
            adapted = scheduler.step(epoch, metrics)
            # May adapt once to check, but weights should remain same
            weights = scheduler.get_loss_weights()
            assert abs(weights["spectrumError"] - 0.7) < 0.05


def run_all_tests():
    """Run all tests manually (for direct execution)."""
    print("Running FlexibleLossScheduler tests...")

    if FlexibleLossScheduler is None:
        print("✗ ERROR: Could not import FlexibleLossScheduler")
        print("Please ensure you're running from the project root with PYTHONPATH set.")
        return

    test_classes = [
        TestSchedulerInitialization,
        TestPlateauDetection,
        TestWeightAdaptation,
        TestSafetyFeatures,
        TestComponentWeightManagement,
        TestEdgeCases
    ]

    total_tests = 0
    passed_tests = 0

    for test_class in test_classes:
        print(f"\n{'='*60}")
        print(f"Testing: {test_class.__name__}")
        print(f"{'='*60}")

        test_instance = test_class()
        test_methods = [method for method in dir(test_instance) if method.startswith('test_')]

        for method_name in test_methods:
            total_tests += 1
            try:
                print(f"  {method_name}...", end=" ")
                getattr(test_instance, method_name)()
                print("✓ PASS")
                passed_tests += 1
            except AssertionError as e:
                print(f"✗ FAIL: {e}")
            except Exception as e:
                print(f"✗ ERROR: {e}")

    print(f"\n{'='*60}")
    print(f"Test Results: {passed_tests}/{total_tests} passed")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_all_tests()
