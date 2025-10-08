"""
Example Configuration: Field Error with Flexible Loss Scheduler

This example demonstrates how to use the new field error loss (replaces MSE)
with the flexible loss scheduler for adaptive spectral bias mitigation.

Key Features:
- Field error as primary training loss (per-frame relative MSE)
- Spectrum error for high-frequency detail preservation
- Flexible scheduler: field_error → spectrum_error when plateau detected
- Conservative defaults (70% spectrum max) prevent overfitting

Usage:
    from configs.examples.field_error_with_scheduler_example import create_example_config
    p_d, p_t, p_l, p_s, ... = create_example_config()
"""

from src.core.utils.params import DataParams, TrainingParams, LossParams, SchedulerParams
from src.core.utils.params import ModelParamsEncoder, ModelParamsDecoder, ModelParamsLatent


def create_example_config(dataset="iso"):
    """
    Create example configuration with field error and flexible scheduler.

    Args:
        dataset: Dataset type ("iso", "tra", or "inc")

    Returns:
        Tuple of parameter objects (p_d, p_t, p_l, p_s, p_me, p_md, p_ml)
    """

    # ===========================================================================
    # DATA PARAMETERS
    # ===========================================================================
    p_d = DataParams(
        batch=4,
        sequenceLength=[20, 20],
        dataSize=[128, 128],
        dimension=2,
        simFields=["pres"],  # Velocity + pressure
        normalizeMode=dataset
    )

    # ===========================================================================
    # TRAINING PARAMETERS
    # ===========================================================================
    p_t = TrainingParams(
        epochs=200,
        lr=0.001,
        expLrGamma=0.995,
        weightDecay=1e-5,
        fadeInPredLoss=[10, 30],
        fadeInSeqLen=[-1, 0],  # No sequence length curriculum
    )

    # ===========================================================================
    # LOSS PARAMETERS (NEW - Field Error Architecture)
    # ===========================================================================
    p_l = LossParams(
        # Primary training losses (field error replaces MSE)
        recFieldError=1.0,      # Reconstruction: per-frame relative MSE
        predFieldError=1.0,     # Prediction: per-frame relative MSE
        spectrumError=0.0,      # Start at 0, scheduler will increase this

        # Perceptual loss (optional, disabled by default)
        recLSIM=0.0,
        predLSIM=0.0,

        # Regularization (optional)
        regMeanStd=0.05,        # Mean/std matching
        regDiv=0.0,             # Divergence-free constraint
    )

    # ===========================================================================
    # FLEXIBLE LOSS SCHEDULER (NEW - Adaptive Spectral Bias Mitigation)
    # ===========================================================================
    p_s = SchedulerParams(
        # Enable scheduler
        enabled=True,

        # Loss transition: field_error → spectrum_error
        source_loss="field_error",
        target_loss="spectrum_error",

        # Component mapping (which loss parts to adjust)
        source_components={"recFieldError": True, "predFieldError": True},
        target_components={"spectrumError": True},
        static_components={},  # No fixed components

        # Weight progression (CONSERVATIVE - prevents overfitting to spectrum)
        initial_weight_source=1.0,    # Start: 100% field error
        initial_weight_target=0.0,    # Start: 0% spectrum error
        final_weight_source=0.3,      # End: 30% field error
        final_weight_target=0.7,      # End: 70% spectrum error
        max_weight_target=0.8,        # Safety cap: never exceed 80% spectrum

        # Adaptation parameters
        patience=5,                    # Wait 5 epochs before adapting
        warmup_epochs=5,               # No adaptation in first 5 epochs
        adaptive_step=True,            # Smaller steps near convergence
        max_consecutive_adaptations=10,  # Stop if unstable (10 adaptations in a row)

        # Monitoring (what metric to watch)
        monitor_metric="spectrum_error",  # Adapt when spectrum_error plateaus
        use_ema=True,                     # Smooth validation metrics
        ema_beta=0.9,                     # EMA smoothing factor
    )

    # ===========================================================================
    # MODEL PARAMETERS (Standard)
    # ===========================================================================
    p_me = ModelParamsEncoder(
        arch="skip",
        encWidth=16,
        latentSize=128,
    )

    p_md = ModelParamsDecoder(
        arch="unet",
        decWidth=32,
        decDepth=4,
    )

    p_ml = ModelParamsLatent(
        arch="gru",
        latWidth=64,
        latDepth=2,
    )

    return p_d, p_t, p_l, p_s, p_me, p_md, p_ml


# ==============================================================================
# ALTERNATIVE CONFIGURATIONS
# ==============================================================================

def create_lsim_to_field_error_config():
    """
    Alternative: Transition from LSIM (perceptual) to field error (physics).

    Use case: Start with perceptual quality, then refine physics accuracy.
    """
    p_d, p_t, p_l, p_s, p_me, p_md, p_ml = create_example_config()

    # Modify loss to start with LSIM
    p_l.recFieldError = 0.0
    p_l.predFieldError = 0.0
    p_l.recLSIM = 1.0
    p_l.predLSIM = 1.0

    # Modify scheduler to transition LSIM → field_error
    p_s.source_loss = "lsim"
    p_s.target_loss = "field_error"
    p_s.source_components = {"recLSIM": True, "predLSIM": True}
    p_s.target_components = {"recFieldError": True, "predFieldError": True}
    p_s.monitor_metric = "field_error"

    return p_d, p_t, p_l, p_s, p_me, p_md, p_ml


def create_static_spectrum_config():
    """
    Static configuration: Fixed 50/50 field + spectrum (no scheduler).

    Use case: When you want spectral bias mitigation without adaptation.
    """
    p_d, p_t, p_l, p_s, p_me, p_md, p_ml = create_example_config()

    # Set static weights
    p_l.recFieldError = 0.5
    p_l.predFieldError = 0.5
    p_l.spectrumError = 0.5

    # Disable scheduler
    p_s.enabled = False

    return p_d, p_t, p_l, p_s, p_me, p_md, p_ml


# ==============================================================================
# DATASET-SPECIFIC RECOMMENDATIONS
# ==============================================================================

def get_recommended_config_for_dataset(dataset: str):
    """
    Get recommended scheduler config for specific dataset.

    Dataset characteristics:
    - ISO: Isotropic turbulence - moderate spectral bias
    - TRA: Transitional flow - high spectral bias (needs aggressive scheduling)
    - INC: Incompressible flow - low spectral bias (gentle scheduling)
    """
    p_d, p_t, p_l, p_s, p_me, p_md, p_ml = create_example_config(dataset)

    if dataset == "tra":
        # Transitional: Aggressive spectrum scheduling
        p_s.final_weight_target = 0.8  # Higher spectrum weight
        p_s.max_weight_target = 0.9
        p_s.patience = 3  # Faster adaptation

    elif dataset == "inc":
        # Incompressible: Gentle spectrum scheduling
        p_s.final_weight_target = 0.5  # Moderate spectrum weight
        p_s.max_weight_target = 0.6
        p_s.patience = 8  # Slower adaptation

    else:  # iso
        # Isotropic: Balanced (use defaults)
        pass

    return p_d, p_t, p_l, p_s, p_me, p_md, p_ml


if __name__ == "__main__":
    # Example usage
    print("Creating example configuration with field error + scheduler...")
    p_d, p_t, p_l, p_s, p_me, p_md, p_ml = create_example_config()

    print(f"\nLoss Configuration:")
    print(f"  recFieldError:   {p_l.recFieldError}")
    print(f"  predFieldError:  {p_l.predFieldError}")
    print(f"  spectrumError:   {p_l.spectrumError}")

    print(f"\nScheduler Configuration:")
    print(f"  Enabled:         {p_s.enabled}")
    print(f"  Transition:      {p_s.source_loss} → {p_s.target_loss}")
    print(f"  Final weights:   {p_s.final_weight_source:.1f} / {p_s.final_weight_target:.1f}")
    print(f"  Monitor metric:  {p_s.monitor_metric}")
    print(f"  Patience:        {p_s.patience} epochs")

    print("\n✓ Configuration created successfully!")
