"""Training configuration dataclasses."""

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class BaseTrainingConfig:
    """Base training configuration with shared parameters."""

    # Optimization
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0

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

    # Scheduler
    scheduler_type: Optional[str] = None

    # Override defaults for epoch-based training
    log_every: int = 1
    save_every: int = 10


@dataclass
class LCFMTrainingConfig(BaseTrainingConfig):
    """LCFM-specific training configuration."""

    # Model architecture
    latent_dim: int = 32
    in_channels: int = 5
    base_channels: int = 64
    channel_mult: Tuple[int, ...] = field(default=(1, 2, 4, 4))
    num_res_blocks: int = 2
    attention_resolutions: Tuple[int, ...] = field(default=(16, 8))
    dropout: float = 0.1
    num_heads: int = 4

    # LCFM-specific loss
    beta: float = 0.001

    # Step-based training
    num_steps: int = 100_000
    warmup_steps: int = 1000
    batch_size: int = 128

    # Sampling during training
    sample_every: int = 5000
    num_sample_images: int = 16

    # Override defaults for step-based training
    log_every: int = 100
    save_every: int = 10_000
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
