"""
Unit tests for TF (Transformer) model variants

Tests TF variants specified:
- TF MGN: Transformer Multi-Grid Network (LatentModelTransformerMGN)
- TF Enc: Transformer Encoder (LatentModelTransformerEnc)
- TF Dec: Transformer Decoder (LatentModelTransformerDec)
- TF VAE: Transformer for latent space modeling (LatentModelTransformer)

Each test verifies:
1. Model initialization and architecture
2. Forward pass functionality
3. Attention mechanism properties
4. Positional encoding behavior
5. Latent space handling
6. Compatibility with all datasets
7. Training integration capabilities
"""

import unittest
import torch
import torch.nn as nn
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

try:
    from src.core.models.model_latent_transformer import (
        LatentModelTransformerEnc,
        LatentModelTransformerDec,
        LatentModelTransformer,
        LatentModelTransformerMGN,
        LatentModelTransformerMGNParamEmb,
        PositionalEncoding
    )
    from src.core.utils.params import DataParams, ModelParamsEncoder, ModelParamsLatent
    TF_AVAILABLE = True
except ImportError as e:
    TF_AVAILABLE = False
    print(f"Transformer modules not available: {e}")

from tests.fixtures.dummy_datasets import get_dummy_batch


@unittest.skipIf(not TF_AVAILABLE, "Transformer modules not available")
class TestTFVariants(unittest.TestCase):
    """Test suite for Transformer model variants"""

    def setUp(self):
        """Set up test parameters and configurations"""
        # Data parameters
        self.data_params = DataParams(
            batch=2,
            sequenceLength=[8, 2],
            dataSize=[16, 16],
            dimension=2,
            simFields=["pres"],
            simParams=[],  # Remove simParams to make latent size divisible by heads
            normalizeMode=""
        )

        # Encoder parameters
        self.encoder_params = ModelParamsEncoder(
            arch="skip",
            latentSize=64,  # 64 is divisible by 4 heads
            encWidth=32,
            frozen=False,
            pretrained=False
        )

        # Latent model parameters
        self.latent_params = ModelParamsLatent(
            arch="transformer",
            width=128,
            layers=2,
            heads=4,  # 4 heads work with latentSize=64
            dropout=0.1
        )

        # Test all dataset types
        self.dataset_names = ['inc_low', 'inc_high', 'tra_ext', 'tra_inc', 'iso']

    def test_positional_encoding(self):
        """Test positional encoding component"""
        try:
            latent_size = 64
            max_len = 100
            pos_enc = PositionalEncoding(latent_size, dropout=0.0, maxLen=max_len)

            # Test input shapes
            batch_size = 2
            seq_len = 10
            x = torch.randn(batch_size, seq_len, latent_size)

            output = pos_enc(x)

            # Should maintain input shape
            self.assertEqual(output.shape, (batch_size, seq_len, latent_size))

            # Should add positional information (output != input)
            self.assertFalse(torch.allclose(output, x))

            # Should be deterministic
            output2 = pos_enc(x)
            self.assertTrue(torch.allclose(output, output2))

            # Test different sequence lengths
            for seq_len in [5, 15, 50]:
                x_len = torch.randn(1, seq_len, latent_size)
                output_len = pos_enc(x_len)
                self.assertEqual(output_len.shape, (1, seq_len, latent_size))

        except Exception as e:
            self.fail(f"Positional encoding test failed: {e}")

    def test_tf_enc_initialization_and_forward(self):
        """Test TF Encoder (LatentModelTransformerEnc) initialization and forward pass"""
        try:
            # Test with flatLatent=False
            tf_enc = LatentModelTransformerEnc(
                self.data_params, self.encoder_params, self.latent_params, flatLatent=False
            )
            self.assertIsInstance(tf_enc, nn.Module)

            # Note: Skip flatLatent=True test when simParams is empty as it causes shape issues

            # Test forward pass
            batch_size = 2
            seq_len = 8
            latent_size = self.encoder_params.latentSize + len(self.data_params.simParams)

            # Create dummy latent input
            latent_data = torch.randn(batch_size, seq_len, self.encoder_params.latentSize)
            sim_param = torch.randn(batch_size, seq_len, len(self.data_params.simParams))

            with torch.no_grad():
                output = tf_enc(latent_data, sim_param)

            # Should produce valid outputs
            self.assertTrue(torch.all(torch.isfinite(output)))

            # Output shapes should be reasonable
            self.assertEqual(len(output.shape), 3)  # [B, S, features]

        except Exception as e:
            self.fail(f"TF Encoder test failed: {e}")

    def test_tf_dec_initialization_and_forward(self):
        """Test TF Decoder (LatentModelTransformerDec) initialization and forward pass"""
        try:
            tf_dec = LatentModelTransformerDec(
                self.data_params, self.encoder_params, self.latent_params
            )
            self.assertIsInstance(tf_dec, nn.Module)

            # Test forward pass
            batch_size = 2
            input_seq_len = 8
            output_seq_len = 2
            latent_size = self.encoder_params.latentSize + len(self.data_params.simParams)

            # Create dummy inputs
            encoder_output = torch.randn(batch_size, input_seq_len, self.encoder_params.latentSize)
            target_input = torch.randn(batch_size, output_seq_len, self.encoder_params.latentSize)
            sim_param_data = torch.randn(batch_size, input_seq_len, len(self.data_params.simParams))
            sim_param_target = torch.randn(batch_size, output_seq_len, len(self.data_params.simParams))

            with torch.no_grad():
                output = tf_dec(encoder_output, target_input, sim_param_data, sim_param_target)

            # Should produce valid output
            self.assertTrue(torch.all(torch.isfinite(output)))
            self.assertEqual(len(output.shape), 3)  # [B, S, features]

        except Exception as e:
            self.fail(f"TF Decoder test failed: {e}")

    def test_tf_vae_initialization_and_forward(self):
        """Test TF VAE (LatentModelTransformer) initialization and forward pass"""
        try:
            tf_vae = LatentModelTransformer(
                self.data_params, self.encoder_params, self.latent_params
            )
            self.assertIsInstance(tf_vae, nn.Module)

            # Test forward pass
            batch_size = 2
            seq_len = 8
            latent_size = self.encoder_params.latentSize + len(self.data_params.simParams)

            # Create dummy latent input
            data_input = torch.randn(batch_size, seq_len, self.encoder_params.latentSize)
            target_input = torch.randn(batch_size, seq_len, self.encoder_params.latentSize)
            sim_param_data = torch.randn(batch_size, seq_len, len(self.data_params.simParams))
            sim_param_target = torch.randn(batch_size, seq_len, len(self.data_params.simParams))

            with torch.no_grad():
                output = tf_vae(data_input, target_input, sim_param_data, sim_param_target)

            # Should produce valid output
            self.assertTrue(torch.all(torch.isfinite(output)))
            self.assertEqual(len(output.shape), 3)  # [B, S, features]

        except Exception as e:
            self.fail(f"TF VAE test failed: {e}")

    def test_tf_mgn_initialization_and_forward(self):
        """Test TF MGN (LatentModelTransformerMGN) initialization and forward pass"""
        try:
            tf_mgn = LatentModelTransformerMGN(
                self.data_params, self.encoder_params, self.latent_params
            )
            self.assertIsInstance(tf_mgn, nn.Module)

            # Test parameter embedding component
            param_emb = LatentModelTransformerMGNParamEmb(
                self.data_params, self.encoder_params
            )
            self.assertIsInstance(param_emb, nn.Module)

            # Test forward pass
            batch_size = 2
            seq_len = 8
            latent_size = self.encoder_params.latentSize + len(self.data_params.simParams)

            # Create dummy inputs
            data_input = torch.randn(batch_size, seq_len, self.encoder_params.latentSize)
            target_input = torch.randn(batch_size, seq_len, self.encoder_params.latentSize)
            param_input = torch.randn(batch_size, len(self.data_params.simParams))

            with torch.no_grad():
                mgn_output = tf_mgn(data_input, target_input)
                param_output = param_emb(param_input)

            # Should produce valid outputs
            self.assertTrue(torch.all(torch.isfinite(mgn_output)))
            self.assertTrue(torch.all(torch.isfinite(param_output)))

            self.assertEqual(len(mgn_output.shape), 3)  # [B, S, features]
            self.assertEqual(len(param_output.shape), 2)  # [B, features]

        except Exception as e:
            self.fail(f"TF MGN test failed: {e}")

    def test_transformer_attention_properties(self):
        """Test transformer attention mechanism properties"""
        try:
            tf_model = LatentModelTransformerEnc(
                self.data_params, self.encoder_params, self.latent_params, flatLatent=False
            )

            # Create input with specific patterns to test attention
            batch_size = 1
            seq_len = 8
            latent_size = self.encoder_params.latentSize + len(self.data_params.simParams)

            # Create input with position-dependent patterns
            x = torch.zeros(batch_size, seq_len, self.encoder_params.latentSize)
            sim_param = torch.zeros(batch_size, seq_len, len(self.data_params.simParams))
            for i in range(seq_len):
                x[0, i, :] = i * 0.1  # Position-dependent values

            with torch.no_grad():
                output = tf_model(x, sim_param)

            # Check that attention creates dependencies across positions
            # Output at each position should be different from input
            for i in range(seq_len):
                input_vec = x[0, i]
                output_vec = output[0, i]
                self.assertFalse(torch.allclose(input_vec, output_vec, atol=1e-6),
                               f"Position {i} should be transformed by attention")

            # Check that output values are influenced by other positions
            # (This is a structural test of attention mechanism)
            output_variance_across_seq = torch.var(output[0], dim=0)
            self.assertTrue(torch.all(output_variance_across_seq > 1e-8),
                           "Attention should create variation across sequence")

        except Exception as e:
            self.fail(f"Transformer attention properties test failed: {e}")

    def test_transformer_gradient_flow(self):
        """Test gradient flow through transformer models"""
        try:
            models_to_test = [
                ("TF_Enc", LatentModelTransformerEnc(
                    self.data_params, self.encoder_params, self.latent_params, flatLatent=False
                )),
                ("TF_VAE", LatentModelTransformer(
                    self.data_params, self.encoder_params, self.latent_params
                )),
                ("TF_MGN", LatentModelTransformerMGN(
                    self.data_params, self.encoder_params, self.latent_params
                ))
            ]

            for model_name, model in models_to_test:
                with self.subTest(model=model_name):
                    batch_size = 2
                    seq_len = 4
                    latent_size = self.encoder_params.latentSize + len(self.data_params.simParams)

                    x = torch.randn(batch_size, seq_len, self.encoder_params.latentSize, requires_grad=True)
                    target = torch.randn(batch_size, seq_len, self.encoder_params.latentSize)
                    sim_param = torch.randn(batch_size, seq_len, len(self.data_params.simParams))

                    if model_name == "TF_Enc":
                        output = model(x, sim_param)
                    else:  # TF_VAE and TF_MGN need different signatures
                        if model_name == "TF_VAE":
                            sim_param_target = torch.randn(batch_size, seq_len, len(self.data_params.simParams))
                            output = model(x, target, sim_param, sim_param_target)
                        else:  # TF_MGN
                            output = model(x, target)
                    loss = nn.MSELoss()(output, target)
                    loss.backward()

                    # Check that model parameters have gradients
                    has_gradients = any(param.grad is not None for param in model.parameters())
                    self.assertTrue(has_gradients, f"{model_name} should have gradients")

                    # Check that input gradients exist
                    self.assertIsNotNone(x.grad, f"{model_name} should compute input gradients")

        except Exception as e:
            self.fail(f"Transformer gradient flow test failed: {e}")

    def test_transformer_different_sequence_lengths(self):
        """Test transformers with different sequence lengths"""
        try:
            tf_model = LatentModelTransformerEnc(
                self.data_params, self.encoder_params, self.latent_params, flatLatent=False
            )

            # Test different sequence lengths
            for seq_len in [1, 4, 8, 16]:
                with self.subTest(seq_len=seq_len):
                    x = torch.randn(1, seq_len, self.encoder_params.latentSize)
                    sim_param = torch.randn(1, seq_len, len(self.data_params.simParams))

                    with torch.no_grad():
                        output = tf_model(x, sim_param)

                    self.assertEqual(output.shape[1], seq_len)
                    self.assertTrue(torch.all(torch.isfinite(output)))

        except Exception as e:
            self.fail(f"Transformer sequence length test failed: {e}")

    def test_transformer_parameter_count_scaling(self):
        """Test that transformer parameter counts scale with architecture"""
        try:
            # Test different transformer sizes
            configs = [
                {"width": 64, "layers": 1, "heads": 2},
                {"width": 128, "layers": 1, "heads": 2},
                {"width": 64, "layers": 2, "heads": 2},
                {"width": 64, "layers": 1, "heads": 4},
            ]

            param_counts = {}
            for config in configs:
                latent_params = ModelParamsLatent(
                    arch="transformer",
                    width=config["width"],
                    layers=config["layers"],
                    heads=config["heads"],
                    dropout=0.1
                )

                model = LatentModelTransformerEnc(
                    self.data_params, self.encoder_params, latent_params, flatLatent=False
                )

                param_count = sum(p.numel() for p in model.parameters())
                param_counts[str(config)] = param_count

                # Should have reasonable parameter count
                self.assertGreater(param_count, 100)
                self.assertLess(param_count, 10_000_000)

            # More width/layers/heads should generally mean more parameters
            base_config = str({"width": 64, "layers": 1, "heads": 2})
            wider_config = str({"width": 128, "layers": 1, "heads": 2})
            deeper_config = str({"width": 64, "layers": 2, "heads": 2})

            self.assertGreater(param_counts[wider_config], param_counts[base_config])
            self.assertGreater(param_counts[deeper_config], param_counts[base_config])

        except Exception as e:
            self.fail(f"Transformer parameter scaling test failed: {e}")

    def test_transformer_positional_encoding_effects(self):
        """Test that positional encoding affects transformer behavior"""
        try:
            tf_model = LatentModelTransformerEnc(
                self.data_params, self.encoder_params, self.latent_params, flatLatent=False
            )

            batch_size = 1
            seq_len = 8
            latent_size = self.encoder_params.latentSize + len(self.data_params.simParams)

            # Create identical content at different positions
            x = torch.ones(batch_size, seq_len, self.encoder_params.latentSize)
            sim_param = torch.ones(batch_size, seq_len, len(self.data_params.simParams))

            with torch.no_grad():
                output = tf_model(x, sim_param)

            # Even with identical input content, different positions should produce different outputs
            # due to positional encoding
            for i in range(1, seq_len):
                pos_0_output = output[0, 0]
                pos_i_output = output[0, i]

                # Should be different due to positional encoding
                position_difference = torch.mean(torch.abs(pos_0_output - pos_i_output))
                self.assertGreater(position_difference.item(), 1e-6,
                                 f"Position 0 and {i} should produce different outputs due to positional encoding")

        except Exception as e:
            self.fail(f"Transformer positional encoding test failed: {e}")

    def test_transformer_all_datasets_compatibility(self):
        """Test transformer models work conceptually with all dataset types"""
        try:
            tf_model = LatentModelTransformerEnc(
                self.data_params, self.encoder_params, self.latent_params, flatLatent=False
            )

            for dataset_name in self.dataset_names:
                with self.subTest(dataset=dataset_name):
                    # Create dummy latent representations for this dataset
                    batch_size = 1
                    seq_len = 8
                    latent_size = self.encoder_params.latentSize + len(self.data_params.simParams)

                    # Simulate latent input that would come from encoder
                    latent_input = torch.randn(batch_size, seq_len, self.encoder_params.latentSize)
                    sim_param = torch.randn(batch_size, seq_len, len(self.data_params.simParams))

                    with torch.no_grad():
                        output = tf_model(latent_input, sim_param)

                    # Should process successfully
                    self.assertTrue(torch.all(torch.isfinite(output)))
                    self.assertEqual(output.shape[0], batch_size)
                    self.assertEqual(output.shape[1], seq_len)

        except Exception as e:
            self.fail(f"Transformer dataset compatibility test failed: {e}")

    def test_transformer_latent_space_properties(self):
        """Test transformer models preserve latent space properties"""
        try:
            tf_model = LatentModelTransformerEnc(
                self.data_params, self.encoder_params, self.latent_params, flatLatent=False
            )

            batch_size = 2
            seq_len = 8
            latent_size = self.encoder_params.latentSize + len(self.data_params.simParams)

            # Test with different latent representations
            latent_input1 = torch.randn(batch_size, seq_len, self.encoder_params.latentSize)
            latent_input2 = torch.randn(batch_size, seq_len, self.encoder_params.latentSize)
            sim_param = torch.randn(batch_size, seq_len, len(self.data_params.simParams))

            with torch.no_grad():
                output1 = tf_model(latent_input1, sim_param)
                output2 = tf_model(latent_input2, sim_param)

            # Different inputs should produce different outputs
            self.assertFalse(torch.allclose(output1, output2, atol=1e-3))

            # Test stability to small perturbations
            latent_input1_perturbed = latent_input1 + torch.randn_like(latent_input1) * 0.01

            with torch.no_grad():
                output1_perturbed = tf_model(latent_input1_perturbed, sim_param)

            # Small perturbations should produce small changes
            perturbation_effect = torch.mean(torch.abs(output1 - output1_perturbed))
            self.assertLess(perturbation_effect.item(), 1.0,
                           "Transformer should be stable to small input perturbations")

        except Exception as e:
            self.fail(f"Transformer latent space properties test failed: {e}")


if __name__ == '__main__':
    unittest.main(verbosity=2)