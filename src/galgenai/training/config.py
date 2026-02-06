"""Training configuration dataclasses."""

from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass
class BaseTrainingConfig:
    """Base training configuration with shared parameters."""

    # Optimization
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0

    # Scheduler (callable that takes optimizer and returns scheduler)
    scheduler_factory: Optional[Callable[[Any], Any]] = None

    # Logging & checkpointing
    log_every: int = 100
    save_every: int = 1000

    # Paths
    output_dir: str = "./output"
    checkpoint_path: Optional[str] = None

    # Device (auto-detect if None)
    device: Optional[str] = None


@dataclass
class VAETrainingConfig(BaseTrainingConfig):
    """VAE-specific training configuration."""

    # VAE-specific parameters
    reconstruction_loss_fn: str = "mse"
    beta: float = 1.0

    # Epoch-based training
    num_epochs: int = 10

    # Validation
    validate_every: int = 1

    # Override defaults for epoch-based training
    log_every: int = 1
    save_every: int = 10


@dataclass
class LCFMTrainingConfig(BaseTrainingConfig):
    """LCFM-specific training configuration."""

    # LCFM-specific loss
    beta: float = 0.001

    # Step-based training
    num_steps: int = 100_000
    warmup_steps: int = 1000

    # Sampling during training
    sample_every: int = 5000
    num_sample_images: int = 16

    # Validation (should be a multiple of log_every)
    validate_every: int = 500

    # Override defaults for step-based training
    log_every: int = 100
    save_every: int = 10_000
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
