"""
LCFM Training Loop

This script trains the Latent Conditional Flow Matching model.

Assumptions about your codebase:
1. You have a VAEEncoder class with forward(x) -> (mu, logvar)
2. You have a PyTorch Dataset that returns galaxy images as tensors of shape (5, 64, 64)
3. Images are normalized (we'll assume roughly zero-mean, unit variance per channel)

Hyperparameter choices explained:
- Learning rate: 2e-4 (standard for flow matching, from Samaddar et al.)
- Batch size: 128 (from paper; reduce if GPU memory limited)
- Beta (KL weight): 0.001 (from Darcy flow experiments - scientific data benefits from
  lower beta to allow more information through the latent bottleneck)
- Base channels: 64 (64x64 images don't need as many channels as 256x256)
- Channel multipliers: [1, 2, 4, 4] -> [64, 128, 256, 256] channels at each resolution
- Dropout: 0.1 (standard regularization)
- Training steps: 100K-600K depending on dataset size (paper uses 100K for 10K samples,
  600K for 50K samples)
"""

import time
from pathlib import Path
from typing import Optional, Dict
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import the LCFM model
from ..models.lcfm import LCFM, count_parameters


@dataclass
class TrainingConfig:
    """
    Training configuration with sensible defaults for galaxy generation.

    All hyperparameters are explained in comments.
    """

    # === Model architecture ===
    latent_dim: int = 32  # Must match your VAE
    in_channels: int = 5  # 5-band galaxy images
    base_channels: int = 64  # Base channel count for U-Net
    channel_mult: tuple = (
        1,
        2,
        4,
        4,
    )  # Channel multipliers at each resolution
    num_res_blocks: int = 2  # Residual blocks per resolution
    attention_resolutions: tuple = (
        16,
        8,
    )  # Apply attention at these spatial sizes
    dropout: float = 0.1  # Dropout for regularization
    num_heads: int = 4  # Attention heads

    # === Loss weights ===
    beta: float = 0.001  # KL divergence weight
    # Low value (0.001) lets more info through latent
    # Higher value (0.01) regularizes latent more strongly

    # === Optimization ===
    learning_rate: float = (
        2e-4  # Adam learning rate (standard for flow matching)
    )
    weight_decay: float = 0.01  # AdamW weight decay
    batch_size: int = 128  # Batch size (reduce if OOM)
    num_steps: int = 100_000  # Total training steps
    # Rule of thumb: ~2 epochs per 1K samples minimum
    warmup_steps: int = 1000  # Linear LR warmup steps

    # === Logging & checkpointing ===
    log_every: int = 100  # Log metrics every N steps
    sample_every: int = 5000  # Generate samples every N steps
    save_every: int = 10_000  # Save checkpoint every N steps
    num_sample_images: int = 16  # Number of images to sample for visualization

    # === Paths ===
    output_dir: str = "./lcfm_output"
    checkpoint_path: Optional[str] = None  # Resume from checkpoint


class LCFMTrainer:
    """
    Trainer class for LCFM.

    Handles:
    - Training loop with gradient accumulation
    - Learning rate scheduling (cosine with warmup)
    - Logging and visualization
    - Checkpointing
    """

    def __init__(
        self,
        model: LCFM,
        train_loader: DataLoader,
        config: TrainingConfig,
        val_loader: Optional[DataLoader] = None,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        # Setup output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "samples").mkdir(exist_ok=True)
        (self.output_dir / "checkpoints").mkdir(exist_ok=True)

        # Optimizer: AdamW with weight decay
        # Only optimize trainable parameters (latent layer + velocity net)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = AdamW(
            trainable_params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999),  # Standard Adam betas
        )

        # Learning rate scheduler: cosine annealing with warmup
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_steps - config.warmup_steps,
            eta_min=config.learning_rate * 0.01,  # Minimum LR = 1% of initial
        )

        # Training state
        self.global_step = 0
        self.best_loss = float("inf")

        # Logging
        self.loss_history = []

        # Resume from checkpoint if specified
        if config.checkpoint_path is not None:
            self.load_checkpoint(config.checkpoint_path)

        # Print model info
        num_params = count_parameters(model)
        print("LCFM Model initialized:")
        print(f"  Trainable parameters: {num_params:,}")
        print(f"  Latent dimension: {config.latent_dim}")
        print(f"  Beta (KL weight): {config.beta}")
        print(f"  Learning rate: {config.learning_rate}")
        print(f"  Total training steps: {config.num_steps:,}")

    def get_lr(self) -> float:
        """Get current learning rate with warmup."""
        if self.global_step < self.config.warmup_steps:
            # Linear warmup
            return self.config.learning_rate * (
                self.global_step / self.config.warmup_steps
            )
        else:
            # Use scheduler LR
            return self.scheduler.get_last_lr()[0]

    def set_lr(self, lr: float):
        """Set learning rate for all parameter groups."""
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def train_step(self, batch: torch.Tensor) -> Dict[str, float]:
        """
        Execute single training step.

        Args:
            batch: (batch_size, channels, H, W) images or tuple (flux, ivar, mask)

        Returns:
            Dictionary of loss components
        """
        self.model.train()

        # Handle tuple format from HSCDataset
        if isinstance(batch, (tuple, list)):
            x = batch[0].to(self.device)  # Extract flux only
        else:
            x = batch.to(self.device)

        # Compute loss
        loss, loss_dict = self.model.compute_loss(x, return_components=True)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        # Optimizer step
        self.optimizer.step()

        # Update learning rate
        current_lr = self.get_lr()
        self.set_lr(current_lr)

        # Update scheduler after warmup
        if self.global_step >= self.config.warmup_steps:
            self.scheduler.step()

        loss_dict["lr"] = current_lr
        return loss_dict

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Compute validation loss."""
        if self.val_loader is None:
            return {}

        self.model.eval()

        total_flow_loss = 0.0
        total_kl_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            # Extract flux from tuple
            if isinstance(batch, (list, tuple)):
                x = batch[0].to(self.device)
            else:
                x = batch.to(self.device)
            _, loss_dict = self.model.compute_loss(x, return_components=True)

            total_flow_loss += loss_dict["flow_loss"]
            total_kl_loss += loss_dict["kl_loss"]
            num_batches += 1

        return {
            "val_flow_loss": total_flow_loss / num_batches,
            "val_kl_loss": total_kl_loss / num_batches,
        }

    @torch.no_grad()
    def generate_samples(self, num_samples: int = 16) -> torch.Tensor:
        """
        Generate samples for visualization.

        We need training images to get latents from. This is a key property
        of LCFM: samples are conditioned on latents from real data.
        """
        self.model.eval()

        # Get training images and extract flux
        train_batch = next(iter(self.train_loader))
        if isinstance(train_batch, (list, tuple)):
            train_batch = train_batch[0][:num_samples].to(self.device)
        else:
            train_batch = train_batch[:num_samples].to(self.device)

        # Generate samples
        samples = self.model.sample(train_batch, num_steps=50)

        return samples, train_batch

    def save_checkpoint(
        self, path: Optional[str] = None, is_best: bool = False
    ):
        """Save model checkpoint."""
        if path is None:
            path = (
                self.output_dir / "checkpoints" / f"step_{self.global_step}.pt"
            )

        checkpoint = {
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.config,
            "loss_history": self.loss_history,
            "best_loss": self.best_loss,
        }

        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")

        if is_best:
            best_path = self.output_dir / "checkpoints" / "best.pt"
            torch.save(checkpoint, best_path)
            print(f"Saved best checkpoint to {best_path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.loss_history = checkpoint.get("loss_history", [])
        self.best_loss = checkpoint.get("best_loss", float("inf"))

        print(f"Loaded checkpoint from {path} at step {self.global_step}")

    def train(self):
        """
        Main training loop.

        The training process:
        1. Sample batch of galaxy images x₁
        2. Encode x₁ -> latent f via frozen encoder + trainable stochastic layer
        3. Sample noise x₀ ~ N(0, I) and time t ~ U(0, 1)
        4. Interpolate x_t = (1-t)x₀ + tx₁
        5. Predict velocity v(x_t, f, t) and compute MSE with target u_t = x₁ - x₀
        6. Add KL regularization on latent distribution
        7. Backprop and update
        """
        print(f"\nStarting training from step {self.global_step}")
        print(
            f"Training for {self.config.num_steps - self.global_step} more steps"
        )
        print("-" * 60)

        # Create infinite data iterator
        def infinite_loader():
            while True:
                for batch in self.train_loader:
                    # Handle different batch formats
                    if isinstance(batch, (list, tuple)):
                        batch = batch[0]  # Assume images are first element
                    yield batch

        data_iter = iter(infinite_loader())

        # Training metrics
        running_flow_loss = 0.0
        running_kl_loss = 0.0
        running_total_loss = 0.0
        log_steps = 0

        start_time = time.time()

        while self.global_step < self.config.num_steps:
            # Get batch
            batch = next(data_iter)

            # Training step
            loss_dict = self.train_step(batch)

            # Accumulate metrics
            running_flow_loss += loss_dict["flow_loss"]
            running_kl_loss += loss_dict["kl_loss"]
            running_total_loss += loss_dict["total_loss"]
            log_steps += 1

            self.global_step += 1

            # === Logging ===
            if self.global_step % self.config.log_every == 0:
                avg_flow_loss = running_flow_loss / log_steps
                avg_kl_loss = running_kl_loss / log_steps
                avg_total_loss = running_total_loss / log_steps

                elapsed = time.time() - start_time
                steps_per_sec = self.config.log_every / elapsed

                print(
                    f"Step {self.global_step:6d} | "
                    f"Loss: {avg_total_loss:.4f} | "
                    f"Flow: {avg_flow_loss:.4f} | "
                    f"KL: {avg_kl_loss:.4f} | "
                    f"LR: {loss_dict['lr']:.2e} | "
                    f"{steps_per_sec:.1f} steps/s"
                )

                # Save to history
                self.loss_history.append(
                    {
                        "step": self.global_step,
                        "flow_loss": avg_flow_loss,
                        "kl_loss": avg_kl_loss,
                        "total_loss": avg_total_loss,
                        "lr": loss_dict["lr"],
                    }
                )

                # Reset metrics
                running_flow_loss = 0.0
                running_kl_loss = 0.0
                running_total_loss = 0.0
                log_steps = 0
                start_time = time.time()

                # Track best loss
                if avg_total_loss < self.best_loss:
                    self.best_loss = avg_total_loss

            # === Sample generation ===
            if self.global_step % self.config.sample_every == 0:
                print(f"Generating samples at step {self.global_step}...")
                samples, conditioning_images = self.generate_samples(
                    self.config.num_sample_images
                )

                # Save samples
                sample_path = (
                    self.output_dir
                    / "samples"
                    / f"samples_step_{self.global_step}.pt"
                )
                torch.save(
                    {
                        "samples": samples.cpu(),
                        "conditioning": conditioning_images.cpu(),
                    },
                    sample_path,
                )
                print(f"Saved samples to {sample_path}")

                # Validation
                val_metrics = self.validate()
                if val_metrics:
                    print(
                        f"Validation - Flow: {val_metrics['val_flow_loss']:.4f}, "
                        f"KL: {val_metrics['val_kl_loss']:.4f}"
                    )

            # === Checkpointing ===
            if self.global_step % self.config.save_every == 0:
                self.save_checkpoint()

        # Final checkpoint
        print("\nTraining complete!")
        self.save_checkpoint(
            is_best=(running_total_loss / max(log_steps, 1) <= self.best_loss)
        )


# =============================================================================
# Example usage
# =============================================================================


def create_model_and_trainer(
    vae_encoder: nn.Module,
    train_dataset: torch.utils.data.Dataset,
    val_dataset: Optional[torch.utils.data.Dataset] = None,
    config: Optional[TrainingConfig] = None,
    device: str = "cuda",
) -> LCFMTrainer:
    """
    Factory function to create LCFM model and trainer.

    Args:
        vae_encoder: Your trained VAE encoder module
        train_dataset: Training dataset returning (5, 64, 64) tensors
        val_dataset: Optional validation dataset
        config: Training configuration (uses defaults if None)
        device: Device to train on

    Returns:
        LCFMTrainer ready to call .train()
    """
    if config is None:
        config = TrainingConfig()

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,  # Drop incomplete batches for consistent batch size
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

    # Create LCFM model
    model = LCFM(
        vae_encoder=vae_encoder,
        latent_dim=config.latent_dim,
        in_channels=config.in_channels,
        base_channels=config.base_channels,
        beta=config.beta,
        channel_mult=config.channel_mult,
        num_res_blocks=config.num_res_blocks,
        attention_resolutions=config.attention_resolutions,
        dropout=config.dropout,
        num_heads=config.num_heads,
    )

    # Create trainer
    trainer = LCFMTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
    )

    return trainer


# =============================================================================
# Standalone script
# =============================================================================

if __name__ == "__main__":
    """
    Example of how to use the training script.
    
    You'll need to modify this to load your actual VAE encoder and dataset.
    """

    # === MODIFY THIS SECTION FOR YOUR CODEBASE ===

    # Example: Load your VAE encoder
    # from your_vae_module import VAEEncoder
    # vae_encoder = VAEEncoder(...)
    # vae_encoder.load_state_dict(torch.load("path/to/vae_encoder.pt"))

    # Example: Load your dataset
    # from your_data_module import GalaxyDataset
    # train_dataset = GalaxyDataset("path/to/data", split="train")
    # val_dataset = GalaxyDataset("path/to/data", split="val")

    # === PLACEHOLDER FOR TESTING ===
    # Remove this and use your actual data

    print("=" * 60)
    print("LCFM Training Script")
    print("=" * 60)
    print("\nTo use this script, modify the __main__ section to:")
    print("1. Import and load your trained VAE encoder")
    print("2. Import and create your galaxy dataset")
    print("3. Call create_model_and_trainer() and trainer.train()")
    print("\nExample:")
    print("""
    from your_vae import VAEEncoder
    from your_data import GalaxyDataset
    
    # Load encoder
    encoder = VAEEncoder(latent_dim=32)
    encoder.load_state_dict(torch.load("vae_encoder.pt"))
    
    # Load data
    train_data = GalaxyDataset("data/", split="train")
    
    # Create trainer
    config = TrainingConfig(
        num_steps=100_000,
        beta=0.001,
        output_dir="./lcfm_galaxies"
    )
    trainer = create_model_and_trainer(encoder, train_data, config=config)
    
    # Train!
    trainer.train()
    """)
