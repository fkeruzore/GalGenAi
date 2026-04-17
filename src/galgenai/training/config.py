"""Training configuration dataclasses."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Optional

from galgenai.config import load_config


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


@dataclass
class CNFTrainingConfig(BaseTrainingConfig):
    """CNF training config."""

    # Step-based training
    num_steps: int = 50_000
    warmup_steps: int = 1000

    # Sampling during training
    sample_every: int = 5000
    num_sample_latents: int = 64

    # Validation
    validate_every: int = 500


@dataclass
class DLCFMTrainingConfig(LCFMTrainingConfig):
    """DL-CFM training configuration (LCFM + disentanglement)."""

    # DL-CFM disentanglement hyperparameters
    dlcfm_beta: float = 8e-5
    lambda1: float = 8e-2
    lambda2: float = 1e-2
    K: int = 2
    tau_sq: float = 1.0
    n_aux: int = 6
    condition_cols: List[str] = field(default_factory=list)


# ============================================================================
# Config loading functions
# ============================================================================


def load_vae_training_config(
    config_path: Optional[str] = None,
) -> VAETrainingConfig:
    """
    Load VAE training config from YAML file.

    Parameters
    ----------
    config_path : str, optional
        Path to config file. If None, uses default galgenai_config.yaml

    Returns
    -------
    VAETrainingConfig
        Configuration instance loaded from file
    """
    config = load_config(config_path)
    training_config = config["training"]
    vae_config = training_config["vae"]

    # Automatically append run_name/vae to output directory
    run_name = config.get("run_name", "")
    output_dir = Path(training_config["output_dir"]) / run_name / "vae"

    return VAETrainingConfig(
        # VAE-specific
        reconstruction_loss_fn=vae_config["reconstruction_loss_fn"],
        beta=vae_config["beta"],
        num_epochs=vae_config["epochs"],
        validate_every=vae_config["validate_every"],
        # Base config
        learning_rate=vae_config["lr"],
        weight_decay=vae_config["weight_decay"],
        max_grad_norm=vae_config["max_grad_norm"],
        log_every=vae_config["log_every"],
        save_every=vae_config["save_every"],
        output_dir=str(output_dir),
        checkpoint_path=None,
        device=None,
        scheduler_factory=None,
    )


def _load_lcfm_base_fields(section: dict, output_dir: Path) -> dict:
    """Extract shared LCFM/DL-CFM fields from a config section."""
    return dict(
        num_steps=section["steps"],
        warmup_steps=section["warmup"],
        sample_every=section["sample_every"],
        num_sample_images=section["num_sample_images"],
        validate_every=section["validate_every"],
        learning_rate=section["lr"],
        weight_decay=section["weight_decay"],
        max_grad_norm=section["max_grad_norm"],
        log_every=section["log_every"],
        save_every=section["save_every"],
        output_dir=str(output_dir),
        checkpoint_path=None,
        device=None,
        scheduler_factory=None,
    )


def load_lcfm_training_config(
    config_path: Optional[str] = None,
) -> LCFMTrainingConfig:
    """Load LCFM training config from YAML file."""
    config = load_config(config_path)
    training_config = config["training"]
    lcfm_config = training_config["lcfm"]
    run_name = config.get("run_name", "")
    output_dir = Path(training_config["output_dir"]) / run_name / "lcfm"

    fields = _load_lcfm_base_fields(lcfm_config, output_dir)
    return LCFMTrainingConfig(beta=lcfm_config["beta"], **fields)


def load_cnf_training_config(
    config_path: Optional[str] = None,
) -> CNFTrainingConfig:
    """
    Load CNF training config from YAML file.

    Parameters
    ----------
    config_path : str, optional
        Path to config file. If None, uses default galgenai_config.yaml

    Returns
    -------
    CNFTrainingConfig
        Configuration instance loaded from file
    """
    config = load_config(config_path)
    training_config = config["training"]
    cnf_config = training_config["cnf"]

    # Automatically append run_name/cnf to output directory
    run_name = config.get("run_name", "")
    output_dir = Path(training_config["output_dir"]) / run_name / "cnf"

    return CNFTrainingConfig(
        # CNF-specific
        num_steps=cnf_config["steps"],
        warmup_steps=cnf_config["warmup"],
        sample_every=cnf_config["sample_every"],
        num_sample_latents=cnf_config["num_sample_latents"],
        validate_every=cnf_config["validate_every"],
        # Base config
        learning_rate=cnf_config["lr"],
        weight_decay=cnf_config["weight_decay"],
        max_grad_norm=cnf_config["max_grad_norm"],
        log_every=cnf_config["log_every"],
        save_every=cnf_config["save_every"],
        output_dir=str(output_dir),
        checkpoint_path=None,
        device=None,
        scheduler_factory=None,
    )


def load_dlcfm_training_config(
    config_path: Optional[str] = None,
) -> DLCFMTrainingConfig:
    """Load DL-CFM training config from YAML file."""
    config = load_config(config_path)
    training_config = config["training"]
    dlcfm_config = training_config["dlcfm"]
    run_name = config.get("run_name", "")
    output_dir = Path(training_config["output_dir"]) / run_name / "dlcfm"

    condition_cols = dlcfm_config.get("condition_cols", [])
    fields = _load_lcfm_base_fields(dlcfm_config, output_dir)
    return DLCFMTrainingConfig(
        beta=0.0,
        dlcfm_beta=dlcfm_config["dlcfm_beta"],
        lambda1=dlcfm_config["lambda1"],
        lambda2=dlcfm_config["lambda2"],
        K=dlcfm_config["K"],
        tau_sq=dlcfm_config.get("tau_sq", 1.0),
        n_aux=len(condition_cols),
        condition_cols=condition_cols,
        **fields,
    )
