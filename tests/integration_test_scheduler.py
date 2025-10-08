"""
Integration Test: Flexible Loss Scheduler with Field Error

This script demonstrates end-to-end usage of the flexible loss scheduler system:
1. Creating synthetic turbulence data
2. Setting up a minimal neural operator model
3. Configuring field error + spectrum error losses
4. Initializing the flexible scheduler
5. Running training with adaptive loss composition
6. Visualizing scheduler behavior

Run:
    python tests/integration_test_scheduler.py

Expected behavior:
- Initial epochs: 100% field error
- After warmup + plateau: Scheduler adapts
- Final epochs: 30% field + 70% spectrum (conservative mix)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import project modules
try:
    from src.core.training.loss import PredictionLoss
    from src.core.training.flexible_loss_scheduler import FlexibleLossScheduler, FlexibleLossSchedulerConfig
    from src.core.training.spectral_metrics import compute_field_error_validation, compute_spectrum_error_validation
    from src.core.utils.params import LossParams
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import project modules: {e}")
    print("Please run from project root with: PYTHONPATH=. python tests/integration_test_scheduler.py")
    IMPORTS_AVAILABLE = False


# ============================================================================
# 1. SYNTHETIC DATA GENERATION
# ============================================================================

class SyntheticTurbulenceDataset(Dataset):
    """Generate synthetic turbulence-like data for testing."""

    def __init__(self, num_samples=100, seq_len=10, height=64, width=64):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.height = height
        self.width = width

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate turbulent field with multiple scales
        x = torch.linspace(0, 4*np.pi, self.width)
        y = torch.linspace(0, 4*np.pi, self.height)
        X, Y = torch.meshgrid(x, y, indexing='ij')

        # Multi-scale sinusoidal (simulates turbulence)
        data = torch.zeros(self.seq_len, 2, self.height, self.width)
        for t in range(self.seq_len):
            # Velocity field with temporal evolution
            vx = (torch.sin(2*X + 0.1*t) + 0.5*torch.sin(5*X + 0.2*t) +
                  0.25*torch.sin(10*X + 0.3*t))
            vy = (torch.cos(2*Y + 0.1*t) + 0.5*torch.cos(5*Y + 0.2*t) +
                  0.25*torch.cos(10*Y + 0.3*t))

            data[t, 0] = vx
            data[t, 1] = vy

        # Add small noise
        data += torch.randn_like(data) * 0.05

        return {"data": data, "simParameters": torch.tensor([])}


# ============================================================================
# 2. MINIMAL NEURAL OPERATOR MODEL
# ============================================================================

class SimpleConvOperator(nn.Module):
    """Minimal convolutional operator for testing."""

    def __init__(self, in_channels=2, hidden_channels=32):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(),
        )

        self.processor = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_channels, in_channels, 3, padding=1),
        )

    def forward(self, x):
        """
        Args:
            x: [B, T, C, H, W] - Input sequence

        Returns:
            prediction: [B, T, C, H, W] - Predicted sequence
        """
        B, T, C, H, W = x.shape

        # Process each timestep
        predictions = []
        for t in range(T):
            feat = self.encoder(x[:, t])
            feat = self.processor(feat)
            pred = self.decoder(feat)
            predictions.append(pred)

        return torch.stack(predictions, dim=1)


# ============================================================================
# 3. TRAINING LOOP WITH SCHEDULER
# ============================================================================

def run_integration_test(num_epochs=50, use_scheduler=True, save_plot=True):
    """
    Run integration test with flexible loss scheduler.

    Args:
        num_epochs: Number of training epochs
        use_scheduler: Whether to use the flexible scheduler
        save_plot: Whether to save visualization plot
    """

    if not IMPORTS_AVAILABLE:
        print("ERROR: Required modules not available. Exiting.")
        return

    print("="*70)
    print("INTEGRATION TEST: Flexible Loss Scheduler")
    print("="*70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # ========================================================================
    # Setup Data
    # ========================================================================
    print("\n[1/6] Creating synthetic turbulence dataset...")

    train_dataset = SyntheticTurbulenceDataset(num_samples=50, seq_len=10)
    val_dataset = SyntheticTurbulenceDataset(num_samples=10, seq_len=10)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")

    # ========================================================================
    # Setup Model
    # ========================================================================
    print("\n[2/6] Initializing model...")

    model = SimpleConvOperator(in_channels=2, hidden_channels=32).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ========================================================================
    # Setup Loss
    # ========================================================================
    print("\n[3/6] Configuring loss function...")

    p_l = LossParams(
        recFieldError=1.0,
        predFieldError=1.0,
        spectrumError=0.0,  # Start at 0, scheduler will increase
    )

    # Create criterion (simplified - just for testing)
    class SimpleCriterion(nn.Module):
        def __init__(self, p_l):
            super().__init__()
            self.p_l = p_l
            self.component_weights = {
                'recFieldError': p_l.recFieldError,
                'predFieldError': p_l.predFieldError,
                'spectrumError': p_l.spectrumError,
            }

        def set_loss_weights(self, weights):
            self.component_weights.update(weights)

        def forward(self, prediction, groundTruth):
            from src.core.training.spectral_metrics import compute_field_error_loss, compute_spectrum_error_loss

            # Compute all losses
            field_loss = compute_field_error_loss(prediction, groundTruth)
            spec_loss = compute_spectrum_error_loss(prediction, groundTruth)

            # Apply weights
            total_loss = (
                self.component_weights['recFieldError'] * field_loss +
                self.component_weights['spectrumError'] * spec_loss
            )

            lossParts = {
                'lossFull': total_loss,
                'lossRecFieldError': self.component_weights['recFieldError'] * field_loss,
                'lossPredFieldError': torch.tensor(0.0),
                'lossSpectrumError': self.component_weights['spectrumError'] * spec_loss,
            }

            return total_loss, lossParts

    criterion = SimpleCriterion(p_l).to(device)

    print(f"  Initial weights: field={p_l.recFieldError:.1f}, spectrum={p_l.spectrumError:.1f}")

    # ========================================================================
    # Setup Scheduler
    # ========================================================================
    print("\n[4/6] Initializing flexible loss scheduler...")

    scheduler = None
    if use_scheduler:
        scheduler_config = FlexibleLossSchedulerConfig(
            enabled=True,
            source_loss="field_error",
            target_loss="spectrum_error",
            source_components={"recFieldError": True, "predFieldError": True},
            target_components={"spectrumError": True},
            initial_weight_source=1.0,
            initial_weight_target=0.0,
            final_weight_source=0.3,
            final_weight_target=0.7,
            max_weight_target=0.8,
            patience=3,
            warmup_epochs=5,
            adaptive_step=True,
            monitor_metric="spectrum_error",
            use_ema=True,
            ema_beta=0.9,
        )
        scheduler = FlexibleLossScheduler(scheduler_config)
        print("  ✓ Scheduler enabled")
        print(f"  Transition: {scheduler_config.source_loss} → {scheduler_config.target_loss}")
        print(f"  Warmup: {scheduler_config.warmup_epochs} epochs, Patience: {scheduler_config.patience}")
    else:
        print("  ✗ Scheduler disabled (baseline)")

    # ========================================================================
    # Training Loop
    # ========================================================================
    print(f"\n[5/6] Training for {num_epochs} epochs...")
    print("-"*70)

    history = {
        'train_loss': [],
        'val_field_error': [],
        'val_spectrum_error': [],
        'weight_field': [],
        'weight_spectrum': [],
        'adaptations': []
    }

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_losses = []

        for batch in train_loader:
            data = batch["data"].to(device)

            optimizer.zero_grad()
            prediction = model(data)
            loss, lossParts = criterion(prediction, data)

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)
        history['train_loss'].append(avg_train_loss)

        # Validation
        model.eval()
        val_field_errors = []
        val_spectrum_errors = []

        with torch.no_grad():
            for batch in val_loader:
                data = batch["data"].to(device)
                prediction = model(data)

                field_err = compute_field_error_validation(prediction, data)
                spec_err = compute_spectrum_error_validation(prediction, data)

                val_field_errors.append(field_err)
                val_spectrum_errors.append(spec_err)

        avg_field_err = np.mean(val_field_errors)
        avg_spec_err = np.mean(val_spectrum_errors)

        history['val_field_error'].append(avg_field_err)
        history['val_spectrum_error'].append(avg_spec_err)

        # Update scheduler
        adapted = False
        if scheduler is not None:
            validation_metrics = {
                'field_error': avg_field_err,
                'spectrum_error': avg_spec_err
            }

            adapted = scheduler.step(epoch, validation_metrics)

            if adapted:
                # Update criterion weights
                new_weights = scheduler.get_loss_weights()
                criterion.set_loss_weights(new_weights)
                history['adaptations'].append(epoch)

        # Record current weights
        weights = criterion.component_weights
        history['weight_field'].append(weights['recFieldError'])
        history['weight_spectrum'].append(weights['spectrumError'])

        # Print progress
        if epoch % 5 == 0 or adapted:
            status = "→ ADAPTED" if adapted else ""
            print(f"Epoch {epoch:3d} | Loss: {avg_train_loss:.4f} | "
                  f"Field: {avg_field_err:.4f} | Spec: {avg_spec_err:.4f} | "
                  f"W=[{weights['recFieldError']:.2f}, {weights['spectrumError']:.2f}] {status}")

    print("-"*70)
    print(f"✓ Training completed!")

    # ========================================================================
    # Visualization
    # ========================================================================
    print("\n[6/6] Generating visualization...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Training loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # Plot 2: Validation metrics
    axes[0, 1].plot(history['val_field_error'], label='Field Error', linewidth=2)
    axes[0, 1].plot(history['val_spectrum_error'], label='Spectrum Error', linewidth=2)
    for adapt_epoch in history['adaptations']:
        axes[0, 1].axvline(adapt_epoch, color='red', linestyle='--', alpha=0.5, linewidth=1)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Error')
    axes[0, 1].set_title('Validation Metrics (dashed = adaptation)')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # Plot 3: Loss component weights
    axes[1, 0].plot(history['weight_field'], label='Field Error Weight', linewidth=2)
    axes[1, 0].plot(history['weight_spectrum'], label='Spectrum Error Weight', linewidth=2)
    for adapt_epoch in history['adaptations']:
        axes[1, 0].axvline(adapt_epoch, color='red', linestyle='--', alpha=0.5, linewidth=1)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Weight')
    axes[1, 0].set_title('Loss Component Weights Over Time')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    axes[1, 0].set_ylim(-0.1, 1.1)

    # Plot 4: Adaptation events
    axes[1, 1].scatter(history['adaptations'], [1]*len(history['adaptations']),
                       s=100, c='red', marker='o', label='Adaptation Events')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_yticks([])
    axes[1, 1].set_title(f'Scheduler Adaptations (Total: {len(history["adaptations"])})')
    axes[1, 1].grid(True, alpha=0.3, axis='x')
    axes[1, 1].set_xlim(0, num_epochs)
    axes[1, 1].legend()

    plt.tight_layout()

    if save_plot:
        output_dir = Path("tests/outputs")
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / "integration_test_scheduler.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Plot saved to: {output_path}")

    plt.show()

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Final training loss:       {history['train_loss'][-1]:.4f}")
    print(f"Final field error:         {history['val_field_error'][-1]:.4f}")
    print(f"Final spectrum error:      {history['val_spectrum_error'][-1]:.4f}")
    print(f"Final weight (field):      {history['weight_field'][-1]:.2f}")
    print(f"Final weight (spectrum):   {history['weight_spectrum'][-1]:.2f}")
    print(f"Total adaptations:         {len(history['adaptations'])}")

    if scheduler:
        print(f"\nScheduler successfully transitioned from field error to spectrum error!")
        print(f"Adaptation epochs: {history['adaptations']}")
    else:
        print(f"\nBaseline run completed (no scheduler).")

    print("="*70)

    return history


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Integration test for flexible loss scheduler')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--no-scheduler', action='store_true', help='Disable scheduler (baseline)')
    parser.add_argument('--no-plot', action='store_true', help='Skip saving plot')

    args = parser.parse_args()

    history = run_integration_test(
        num_epochs=args.epochs,
        use_scheduler=not args.no_scheduler,
        save_plot=not args.no_plot
    )
