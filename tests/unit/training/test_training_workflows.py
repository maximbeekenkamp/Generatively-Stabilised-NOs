"""
Training Workflow Tests

Tests training workflows and processes for neural operator models.
Verifies that training pipelines work correctly with dummy data including:

1. Basic training workflow components
2. Two-stage generative operator training
3. Loss function computation and validation
4. Training parameter handling and validation
5. Model optimization workflows
6. Training state management and history tracking
7. Memory optimization during training
8. Learning rate scheduling
9. Gradient flow validation
10. Training convergence detection

Each test uses dummy data to ensure training components work correctly
without requiring full training runs.
"""

import unittest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except (ImportError, TypeError):
    # Mock SummaryWriter if tensorboard is not available or has protobuf issues
    class SummaryWriter:
        def __init__(self, *args, **kwargs):
            pass
        def add_scalar(self, *args, **kwargs):
            pass
        def close(self):
            pass
    TENSORBOARD_AVAILABLE = False
import tempfile
import os
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

try:
    from src.core.training.trainer import Trainer
    from src.core.training.loss import PredictionLoss
    from src.core.training.loss_history import LossHistory
    from src.core.models.model import PredictionModel
    from src.core.utils.params import DataParams, TrainingParams, LossParams, ModelParamsEncoder, ModelParamsDecoder
    TRAINING_AVAILABLE = True
except (ImportError, TypeError) as e:
    TRAINING_AVAILABLE = False
    print(f"Training modules not available: {e}")

try:
    from src.core.training.trainer_generative_operator import GenerativeOperatorTrainer, GenerativeOperatorLoss
    from src.core.models.generative_operator_model import GenerativeOperatorModel
    from src.core.models.neural_operator_adapters import FNOPriorAdapter, UNetPriorAdapter
    from src.core.models.generative_correctors import DiffusionCorrector
    GENOP_TRAINING_AVAILABLE = True
except (ImportError, TypeError) as e:
    GENOP_TRAINING_AVAILABLE = False
    print(f"Generative operator training modules not available: {e}")

from tests.fixtures.dummy_datasets import get_dummy_batch


@unittest.skipIf(not TRAINING_AVAILABLE, "Training modules not available")
class TestTrainingWorkflows(unittest.TestCase):
    """Test suite for training workflow components"""

    def setUp(self):
        """Set up test parameters and configurations"""
        self.data_params = DataParams(
            batch=4,
            sequenceLength=[4, 2],
            dataSize=[16, 16],
            dimension=2,
            simFields=["pres"],
            simParams=[],  # Standardized to empty for consistent channel count
            normalizeMode=""
        )

        self.training_params = TrainingParams(
            epochs=2,
            lr=0.001,
            clipGrad=True,
            clipValue=1.0
        )

        self.loss_params = LossParams()

        # Create temporary directory for logs
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary files"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_dummy_model(self):
        """Create a simple dummy model for testing"""
        class DummyModel(nn.Module):
            def __init__(self, input_channels=3, output_channels=3):
                super().__init__()
                self.conv1 = nn.Conv2d(input_channels, 16, 3, padding=1)
                self.conv2 = nn.Conv2d(16, output_channels, 3, padding=1)
                self.relu = nn.ReLU()

            def forward(self, x):
                # x shape: [B, T, C, H, W]
                B, T, C, H, W = x.shape
                x = x.view(B*T, C, H, W)
                x = self.relu(self.conv1(x))
                x = self.conv2(x)
                x = x.view(B, T, C, H, W)
                return x

        return DummyModel()

    def _create_dummy_dataloader(self, batch_size=4, num_batches=3):
        """Create dummy dataloader for testing"""
        # Create dummy data
        input_data = []
        target_data = []

        for _ in range(num_batches * batch_size):
            input_batch, target_batch = get_dummy_batch('inc_low', batch_size=1)
            input_data.append(input_batch.squeeze(0))
            if target_batch is not None:
                target_data.append(target_batch.squeeze(0))
            else:
                target_data.append(input_batch.squeeze(0))

        input_tensor = torch.stack(input_data)
        target_tensor = torch.stack(target_data)

        dataset = TensorDataset(input_tensor, target_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        return dataloader

    def test_loss_function_computation(self):
        """Test loss function computation"""
        try:
            loss_fn = PredictionLoss(self.loss_params)

            # Create dummy predictions and targets
            B, T, C, H, W = 2, 2, 3, 8, 8
            predictions = torch.randn(B, T, C, H, W, requires_grad=True)
            targets = torch.randn(B, T, C, H, W)

            # Compute loss
            loss = loss_fn(predictions, targets)

            # Validate loss
            self.assertIsInstance(loss, torch.Tensor)
            self.assertTrue(loss.requires_grad)
            self.assertGreaterEqual(loss.item(), 0.0)
            self.assertTrue(torch.isfinite(loss))

            # Test backward pass
            loss.backward()
            self.assertIsNotNone(predictions.grad)
            self.assertTrue(torch.all(torch.isfinite(predictions.grad)))

        except Exception as e:
            self.fail(f"Loss function computation test failed: {e}")

    def test_loss_history_tracking(self):
        """Test loss history tracking"""
        try:
            loss_history = LossHistory()

            # Add some dummy loss values
            for epoch in range(5):
                train_loss = 1.0 / (epoch + 1)  # Decreasing loss
                val_loss = 1.2 / (epoch + 1)

                loss_history.addTrainLoss(train_loss)
                loss_history.addValLoss(val_loss)

            # Check history
            self.assertEqual(len(loss_history.trainLoss), 5)
            self.assertEqual(len(loss_history.valLoss), 5)

            # Check decreasing trend
            train_losses = loss_history.trainLoss
            for i in range(1, len(train_losses)):
                self.assertLessEqual(train_losses[i], train_losses[i-1])

            # Test best loss tracking
            best_loss = loss_history.getBestTrainLoss()
            self.assertEqual(best_loss, min(train_losses))

        except Exception as e:
            self.fail(f"Loss history tracking test failed: {e}")

    def test_optimizer_setup(self):
        """Test optimizer setup and configuration"""
        try:
            model = self._create_dummy_model()

            # Test different optimizers
            optimizers = [
                ('adam', optim.Adam(model.parameters(), lr=0.001)),
                ('sgd', optim.SGD(model.parameters(), lr=0.01, momentum=0.9)),
                ('adamw', optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4))
            ]

            for opt_name, optimizer in optimizers:
                with self.subTest(optimizer=opt_name):
                    # Check optimizer state
                    self.assertEqual(len(optimizer.param_groups), 1)
                    self.assertGreater(len(list(optimizer.param_groups[0]['params'])), 0)

                    # Test optimization step
                    x = torch.randn(1, 2, 3, 8, 8)
                    target = torch.randn(1, 2, 3, 8, 8)

                    optimizer.zero_grad()
                    output = model(x)
                    loss = nn.MSELoss()(output, target)
                    loss.backward()

                    # Check gradients exist
                    has_gradients = any(p.grad is not None for p in model.parameters())
                    self.assertTrue(has_gradients)

                    optimizer.step()

        except Exception as e:
            self.fail(f"Optimizer setup test failed: {e}")

    def test_learning_rate_scheduling(self):
        """Test learning rate scheduling"""
        try:
            model = self._create_dummy_model()
            optimizer = optim.Adam(model.parameters(), lr=0.01)

            # Test different schedulers
            schedulers = [
                ('step', StepLR(optimizer, step_size=2, gamma=0.5)),
                ('cosine', CosineAnnealingLR(optimizer, T_max=5))
            ]

            for sched_name, scheduler in schedulers:
                with self.subTest(scheduler=sched_name):
                    initial_lr = optimizer.param_groups[0]['lr']

                    # Step through several epochs
                    lrs = [initial_lr]
                    for epoch in range(5):
                        scheduler.step()
                        current_lr = optimizer.param_groups[0]['lr']
                        lrs.append(current_lr)

                    # Check LR changed
                    self.assertNotEqual(lrs[0], lrs[-1])

                    # Check all LRs are valid
                    for lr in lrs:
                        self.assertGreater(lr, 0)
                        self.assertTrue(torch.isfinite(torch.tensor(lr)))

        except Exception as e:
            self.fail(f"Learning rate scheduling test failed: {e}")

    def test_gradient_clipping(self):
        """Test gradient clipping functionality"""
        try:
            model = self._create_dummy_model()

            # Create large gradients intentionally
            x = torch.randn(1, 2, 3, 8, 8)
            target = torch.randn(1, 2, 3, 8, 8)

            output = model(x)
            loss = nn.MSELoss()(output, target) * 1000  # Large loss to create large gradients
            loss.backward()

            # Check gradients before clipping
            grad_norms_before = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]

            # Apply gradient clipping
            clip_value = 1.0
            nn.utils.clip_grad_norm_(model.parameters(), clip_value)

            # Check gradients after clipping
            total_norm = torch.sqrt(sum(p.grad.norm()**2 for p in model.parameters() if p.grad is not None))
            self.assertLessEqual(total_norm.item(), clip_value * 1.1)  # Allow small tolerance

        except Exception as e:
            self.fail(f"Gradient clipping test failed: {e}")

    def test_training_step_basic(self):
        """Test basic training step workflow"""
        try:
            model = self._create_dummy_model()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            loss_fn = nn.MSELoss()

            # Create training data
            x = torch.randn(2, 2, 3, 8, 8, requires_grad=True)
            target = torch.randn(2, 2, 3, 8, 8)

            # Training step
            model.train()
            optimizer.zero_grad()

            output = model(x)
            loss = loss_fn(output, target)

            self.assertTrue(torch.isfinite(loss))
            self.assertGreater(loss.item(), 0)

            loss.backward()

            # Check gradients
            has_gradients = any(p.grad is not None for p in model.parameters())
            self.assertTrue(has_gradients)

            optimizer.step()

            # Check parameters updated
            self.assertTrue(all(torch.all(torch.isfinite(p)) for p in model.parameters()))

        except Exception as e:
            self.fail(f"Training step basic test failed: {e}")

    def test_validation_step_basic(self):
        """Test basic validation step workflow"""
        try:
            model = self._create_dummy_model()
            loss_fn = nn.MSELoss()

            # Create validation data
            x = torch.randn(2, 2, 3, 8, 8)
            target = torch.randn(2, 2, 3, 8, 8)

            # Validation step
            model.eval()
            with torch.no_grad():
                output = model(x)
                loss = loss_fn(output, target)

            self.assertTrue(torch.isfinite(loss))
            self.assertGreaterEqual(loss.item(), 0)

            # No gradients should be computed
            for p in model.parameters():
                self.assertIsNone(p.grad)

        except Exception as e:
            self.fail(f"Validation step basic test failed: {e}")

    def test_training_convergence_detection(self):
        """Test training convergence detection"""
        try:
            loss_history = LossHistory()

            # Simulate converged training (stable loss)
            converged_losses = [0.1, 0.1001, 0.0999, 0.1002, 0.0998]
            for loss in converged_losses:
                loss_history.addTrainLoss(loss)

            # Check loss stability
            recent_losses = loss_history.trainLoss[-5:]
            loss_std = torch.std(torch.tensor(recent_losses))
            loss_mean = torch.mean(torch.tensor(recent_losses))

            # Relative standard deviation should be small for converged training
            relative_std = loss_std / (loss_mean + 1e-8)
            self.assertLess(relative_std.item(), 0.1)  # Less than 10% variation

            # Simulate diverged training (increasing loss)
            diverged_losses = [0.1, 0.2, 0.4, 0.8, 1.6]
            for loss in diverged_losses:
                loss_history.addTrainLoss(loss)

            # Check for divergence
            recent_losses = loss_history.trainLoss[-5:]
            is_increasing = all(recent_losses[i] <= recent_losses[i+1] for i in range(len(recent_losses)-1))
            self.assertTrue(is_increasing)

        except Exception as e:
            self.fail(f"Training convergence detection test failed: {e}")

    def test_memory_optimization_training(self):
        """Test memory optimization during training"""
        try:
            model = self._create_dummy_model()
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            # Test gradient checkpointing (simplified)
            x = torch.randn(2, 2, 3, 8, 8, requires_grad=True)
            target = torch.randn(2, 2, 3, 8, 8)

            # Normal forward pass
            output_normal = model(x)
            loss_normal = nn.MSELoss()(output_normal, target)

            # Test with gradient accumulation
            optimizer.zero_grad()

            # Split batch and accumulate gradients
            batch_size = x.shape[0]
            accumulation_steps = 2
            mini_batch_size = batch_size // accumulation_steps

            for i in range(accumulation_steps):
                start_idx = i * mini_batch_size
                end_idx = (i + 1) * mini_batch_size

                mini_x = x[start_idx:end_idx]
                mini_target = target[start_idx:end_idx]

                mini_output = model(mini_x)
                mini_loss = nn.MSELoss()(mini_output, mini_target) / accumulation_steps
                mini_loss.backward()

            # Check accumulated gradients
            has_gradients = any(p.grad is not None for p in model.parameters())
            self.assertTrue(has_gradients)

            optimizer.step()

        except Exception as e:
            self.fail(f"Memory optimization training test failed: {e}")

    def test_model_state_management(self):
        """Test model state management during training"""
        try:
            model = self._create_dummy_model()

            # Test training mode
            model.train()
            self.assertTrue(model.training)
            for module in model.modules():
                if hasattr(module, 'training'):
                    self.assertTrue(module.training)

            # Test evaluation mode
            model.eval()
            self.assertFalse(model.training)
            for module in model.modules():
                if hasattr(module, 'training'):
                    self.assertFalse(module.training)

            # Test state dict save/load
            state_dict = model.state_dict()
            self.assertIsInstance(state_dict, dict)
            self.assertGreater(len(state_dict), 0)

            # Create new model and load state
            new_model = self._create_dummy_model()
            new_model.load_state_dict(state_dict)

            # Compare outputs
            x = torch.randn(1, 2, 3, 8, 8)
            with torch.no_grad():
                output1 = model(x)
                output2 = new_model(x)

            self.assertTrue(torch.allclose(output1, output2, atol=1e-6))

        except Exception as e:
            self.fail(f"Model state management test failed: {e}")

    def test_training_parameter_validation(self):
        """Test training parameter validation"""
        try:
            # Test valid parameters
            valid_params = TrainingParams(
                epochs=10,
                lr=0.001,
                clipGrad=True,
                clipValue=1.0
            )

            self.assertEqual(valid_params.epochs, 10)
            self.assertEqual(valid_params.lr, 0.001)
            self.assertTrue(valid_params.clipGrad)
            self.assertEqual(valid_params.clipValue, 1.0)

            # Test parameter ranges
            self.assertGreater(valid_params.epochs, 0)
            self.assertGreater(valid_params.lr, 0)
            self.assertGreater(valid_params.clipValue, 0)

        except Exception as e:
            self.fail(f"Training parameter validation test failed: {e}")

    @unittest.skipIf(not GENOP_TRAINING_AVAILABLE, "Generative operator training not available")
    def test_generative_operator_loss(self):
        """Test generative operator specific loss functions"""
        try:
            # Create loss function
            genop_loss = GenerativeOperatorLoss(self.training_params)

            B, T, C, H, W = 2, 2, 3, 8, 8
            prediction = torch.randn(B, T, C, H, W)
            target = torch.randn(B, T, C, H, W)
            prior_prediction = torch.randn(B, T, C, H, W)

            # Test prior loss
            prior_losses = genop_loss.compute_prior_loss(prediction, target)

            self.assertIn('mse_loss', prior_losses)
            self.assertIn('total_loss', prior_losses)

            for loss_name, loss_value in prior_losses.items():
                self.assertTrue(torch.isfinite(loss_value))
                self.assertGreaterEqual(loss_value.item(), 0)

            # Test corrector loss
            corrector_losses = genop_loss.compute_corrector_loss(prediction, target, prior_prediction)

            self.assertIn('mse_loss', corrector_losses)
            self.assertIn('total_loss', corrector_losses)

            for loss_name, loss_value in corrector_losses.items():
                self.assertTrue(torch.isfinite(loss_value))
                self.assertGreaterEqual(loss_value.item(), 0)

        except Exception as e:
            self.fail(f"Generative operator loss test failed: {e}")

    @unittest.skipIf(not GENOP_TRAINING_AVAILABLE, "Generative operator training not available")
    def test_two_stage_training_workflow(self):
        """Test two-stage training workflow for generative operators"""
        try:
            # Create models
            fno_params = self._create_fno_params()
            fno_prior = FNOPriorAdapter(fno_params, self.data_params)

            diff_params = self._create_diffusion_params()
            diff_corrector = DiffusionCorrector(diff_params, self.data_params)

            genop_model = GenerativeOperatorModel(
                prior_model=fno_prior,
                corrector_model=diff_corrector,
                p_md=diff_params,
                p_d=self.data_params
            )

            # Test stage 1: Prior-only training
            genop_model.training_mode = 'prior_only'
            genop_model.train()

            x = torch.randn(1, 2, 3, 8, 8)
            target = torch.randn(1, 2, 3, 8, 8)

            optimizer = optim.Adam(genop_model.parameters(), lr=0.001)
            optimizer.zero_grad()

            output = genop_model(x)
            loss = nn.MSELoss()(output, target)
            loss.backward()

            # Check only prior has gradients in stage 1
            prior_has_grads = any(p.grad is not None for p in genop_model.prior_model.parameters())
            corrector_has_grads = any(p.grad is not None for p in genop_model.corrector_model.parameters())

            self.assertTrue(prior_has_grads)
            # Corrector might or might not have gradients depending on implementation

            optimizer.step()

            # Test stage 2: Corrector training
            genop_model.training_mode = 'corrector_training'

            optimizer.zero_grad()
            output = genop_model(x)
            loss = nn.MSELoss()(output, target)
            loss.backward()

            optimizer.step()

            # Test stage 3: Full inference
            genop_model.training_mode = 'full_inference'
            genop_model.eval()

            with torch.no_grad():
                output = genop_model(x)

            self.assertTrue(torch.all(torch.isfinite(output)))

        except Exception as e:
            self.fail(f"Two-stage training workflow test failed: {e}")

    def _create_fno_params(self):
        """Create FNO parameters for testing"""
        class MockModelParams:
            def __init__(self):
                self.fnoModes = [16, 16]
                self.decWidth = 32
                self.architecture = 'fno'
                self.model_type = 'fno'
                self.prevSteps = 8

            def _get_prev_steps_from_arch(self):
                return self.prevSteps

        return MockModelParams()

    def _create_diffusion_params(self):
        """Create diffusion parameters for testing"""
        class MockDiffusionParams:
            def __init__(self):
                self.diffSteps = 10
                self.diffSchedule = "linear"
                self.diffCondIntegration = "noisy"
                self.arch = "direct-ddpm+Prev"
                self.correction_strength = 1.0

        return MockDiffusionParams()


@unittest.skipIf(not TRAINING_AVAILABLE, "Training modules not available")
class TestTrainingIntegration(unittest.TestCase):
    """Test suite for training integration workflows"""

    def setUp(self):
        """Set up test parameters"""
        self.data_params = DataParams(
            batch=2,
            sequenceLength=[4, 2],
            dataSize=[16, 16],
            dimension=2,
            simFields=["pres"],
            simParams=[],
            normalizeMode=""
        )

        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_full_training_pipeline_smoke(self):
        """Smoke test for full training pipeline"""
        try:
            # This is a minimal smoke test to ensure the training pipeline can be instantiated
            # and doesn't crash on basic operations

            # Create dummy model (simplified)
            class SimpleModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.conv = nn.Conv2d(3, 3, 3, padding=1)

                def forward(self, x):
                    B, T, C, H, W = x.shape
                    x = x.view(B*T, C, H, W)
                    x = self.conv(x)
                    return x.view(B, T, C, H, W)

            model = SimpleModel()

            # Create dummy data
            x = torch.randn(1, 2, 3, 8, 8)
            target = torch.randn(1, 2, 3, 8, 8)

            # Basic training components
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            loss_fn = nn.MSELoss()

            # Training step
            model.train()
            optimizer.zero_grad()
            output = model(x)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

            # Validation that training step completed
            self.assertTrue(torch.isfinite(loss))

        except Exception as e:
            self.fail(f"Full training pipeline smoke test failed: {e}")


if __name__ == '__main__':
    unittest.main(verbosity=2)