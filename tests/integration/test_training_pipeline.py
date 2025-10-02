"""
Integration tests for training pipeline

Tests that models can integrate with the training infrastructure
and work end-to-end with real training workflows.
"""

import unittest
import torch
import torch.nn as nn
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

try:
    from src.core.models.neural_operator_adapters import FNOPriorAdapter, UNetPriorAdapter, TNOPriorAdapter
    from src.core.utils.params import DataParams
    MODELS_AVAILABLE = True
except ImportError as e:
    MODELS_AVAILABLE = False
    print(f"Model modules not available: {e}")

from tests.fixtures.dummy_datasets import DummyDatasetFactory, get_dummy_batch


@unittest.skipIf(not MODELS_AVAILABLE, "Model modules not available")
class TestTrainingPipeline(unittest.TestCase):
    """Integration tests for training pipeline"""

    def setUp(self):
        """Set up test parameters"""
        self.data_params = DataParams(
            batch=4,
            sequenceLength=[8, 2],
            dataSize=[16, 16],
            dimension=2,
            simFields=["pres"],
            simParams=[],
            normalizeMode=""
        )

    def _create_mock_model_params(self, model_type):
        """Create mock model parameters"""
        class MockModelParams:
            def __init__(self, model_type):
                self.model_type = model_type
                self.architecture = model_type
                self.arch = model_type
                self.fnoModes = [8, 8] if model_type == 'fno' else None
                self.decWidth = 32
                self.prevSteps = 2

        return MockModelParams(model_type)

    def test_fno_training_integration(self):
        """Test FNO model can integrate with training pipeline"""
        try:
            model_params = self._create_mock_model_params('fno')
            model = FNOPriorAdapter(model_params, self.data_params)

            # Create dummy training batch
            input_batch, target_batch = get_dummy_batch("inc_low", batch_size=4)

            # Forward pass
            output = model(input_batch)

            # Compute loss (using last 2 frames to match target)
            loss = nn.MSELoss()(output[:, -2:], target_batch)

            # Backward pass
            loss.backward()

            # Check that loss is computed
            self.assertTrue(loss.item() >= 0, "Training loss should be non-negative")

        except Exception as e:
            self.fail(f"FNO training integration failed: {e}")

    def test_unet_training_integration(self):
        """Test UNet model can integrate with training pipeline"""
        try:
            model_params = self._create_mock_model_params('unet')
            model = UNetPriorAdapter(model_params, self.data_params)

            input_batch, target_batch = get_dummy_batch("tra_ext", batch_size=4)
            output = model(input_batch)
            loss = nn.MSELoss()(output[:, -2:], target_batch)
            loss.backward()

            self.assertTrue(loss.item() >= 0)

        except Exception as e:
            self.fail(f"UNet training integration failed: {e}")

    def test_tno_training_integration(self):
        """Test TNO model can integrate with training pipeline"""
        try:
            model_params = self._create_mock_model_params('tno')
            model = TNOPriorAdapter(model_params, self.data_params)

            input_batch, target_batch = get_dummy_batch("iso", batch_size=4)
            output = model(input_batch)
            loss = nn.MSELoss()(output[:, -2:], target_batch)
            loss.backward()

            self.assertTrue(loss.item() >= 0)

        except Exception as e:
            self.fail(f"TNO training integration failed: {e}")

    def test_multi_dataset_training(self):
        """Test training integration across multiple datasets"""
        model_params = self._create_mock_model_params('fno')
        model = FNOPriorAdapter(model_params, self.data_params)

        dataset_names = ['inc_low', 'inc_high', 'tra_ext', 'tra_inc', 'iso']

        for dataset_name in dataset_names:
            with self.subTest(dataset=dataset_name):
                try:
                    input_batch, target_batch = get_dummy_batch(dataset_name, batch_size=2)

                    # Zero gradients
                    model.zero_grad()

                    # Forward and backward
                    output = model(input_batch)
                    loss = nn.MSELoss()(output[:, -2:], target_batch)
                    loss.backward()

                    self.assertTrue(loss.item() >= 0)

                except Exception as e:
                    self.fail(f"Multi-dataset training failed on {dataset_name}: {e}")

    def test_optimizer_integration(self):
        """Test model integration with optimizers"""
        try:
            model_params = self._create_mock_model_params('fno')
            model = FNOPriorAdapter(model_params, self.data_params)

            # Create optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            input_batch, target_batch = get_dummy_batch("inc_low", batch_size=2)

            # Training step
            optimizer.zero_grad()
            output = model(input_batch)
            loss = nn.MSELoss()(output[:, -2:], target_batch)
            loss.backward()
            optimizer.step()

            # Should complete without errors
            self.assertTrue(True, "Optimizer integration successful")

        except Exception as e:
            self.fail(f"Optimizer integration failed: {e}")


if __name__ == '__main__':
    unittest.main(verbosity=2)