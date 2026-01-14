"""LCFM trainer implementation."""

import time
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from ..models.lcfm import LCFM, count_parameters
from .base_trainer import BaseTrainer
from .config import LCFMTrainingConfig
from .utils import extract_batch_data


class LCFMTrainer(BaseTrainer[LCFMTrainingConfig]):
    """
    Trainer for LCFM with step-based training.

    Features:
    - Warmup + cosine annealing scheduler
    - Sample generation for visualization
    - Infinite data loader pattern
    """

    def __init__(
        self,
        model: LCFM,
        train_loader: DataLoader,
        config: LCFMTrainingConfig,
        val_loader: Optional[DataLoader] = None,
    ):
        super().__init__(model, train_loader, config, val_loader)

        # Additional LCFM-specific directories
        (self.output_dir / "samples").mkdir(exist_ok=True)

        # Print model info
        num_params = count_parameters(model)
        print("LCFM Model initialized:")
        print(f"  Trainable parameters: {num_params:,}")
        print(f"  Latent dimension: {config.latent_dim}")
        print(f"  Beta (KL weight): {config.beta}")
        print(f"  Learning rate: {config.learning_rate}")
        print(f"  Total training steps: {config.num_steps:,}")

    def _setup_optimizer(self):
        """Set up AdamW with warmup + cosine scheduler."""
        trainable_params = [
            p for p in self.model.parameters() if p.requires_grad
        ]
        self.optimizer = AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.999),
        )

        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.num_steps - self.config.warmup_steps,
            eta_min=self.config.learning_rate * 0.01,
        )

    def _get_lr_with_warmup(self) -> float:
        """Get current LR accounting for warmup."""
        if self.global_step < self.config.warmup_steps:
            return self.config.learning_rate * (
                self.global_step / self.config.warmup_steps
            )
        return self.scheduler.get_last_lr()[0]

    def _set_lr(self, lr: float):
        """Set learning rate for all parameter groups."""
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _train_step(self, batch: Any) -> Dict[str, float]:
        """Execute single LCFM training step."""
        x, _, _ = extract_batch_data(batch, self.device)

        # Compute loss using model method
        loss, loss_dict = self.model.compute_loss(x, return_components=True)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self._clip_gradients()
        self.optimizer.step()

        # Update LR with warmup handling
        current_lr = self._get_lr_with_warmup()
        self._set_lr(current_lr)

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
            x, _, _ = extract_batch_data(batch, self.device)
            _, loss_dict = self.model.compute_loss(x, return_components=True)

            total_flow_loss += loss_dict["flow_loss"]
            total_kl_loss += loss_dict["kl_loss"]
            num_batches += 1

        return {
            "val_flow_loss": total_flow_loss / num_batches,
            "val_kl_loss": total_kl_loss / num_batches,
        }

    @torch.no_grad()
    def generate_samples(
        self, num_samples: int = 16
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate samples for visualization."""
        self.model.eval()

        train_batch = next(iter(self.train_loader))
        train_images, _, _ = extract_batch_data(train_batch, self.device)
        train_images = train_images[:num_samples]

        samples = self.model.sample(train_images, num_steps=50)
        return samples, train_images

    def train(self):
        """Main step-based training loop."""
        print(f"\nStarting training from step {self.global_step}")
        print(f"Training for {self.config.num_steps - self.global_step} steps")
        print("-" * 60)

        self.model.train()
        try:
            self.model = torch.compile(self.model)
            print("Model compiled with torch.compile()")
        except RuntimeError:
            print("torch.compile() not available, skipping")

        def infinite_loader():
            while True:
                for batch in self.train_loader:
                    yield batch

        data_iter = iter(infinite_loader())

        # Running averages for logging
        running_flow_loss = 0.0
        running_kl_loss = 0.0
        running_total_loss = 0.0
        log_steps = 0

        start_time = time.time()

        while self.global_step < self.config.num_steps:
            batch = next(data_iter)
            loss_dict = self._train_step(batch)

            running_flow_loss += loss_dict["flow_loss"]
            running_kl_loss += loss_dict["kl_loss"]
            running_total_loss += loss_dict["total_loss"]
            log_steps += 1

            self.global_step += 1

            # Logging
            if self.global_step % self.config.log_every == 0:
                avg_metrics = {
                    "flow_loss": running_flow_loss / log_steps,
                    "kl_loss": running_kl_loss / log_steps,
                    "total_loss": running_total_loss / log_steps,
                    "lr": loss_dict["lr"],
                }

                elapsed = time.time() - start_time
                steps_per_sec = self.config.log_every / elapsed

                print(
                    f"Step {self.global_step:6d} | "
                    f"Loss: {avg_metrics['total_loss']:.3e} | "
                    f"Flow: {avg_metrics['flow_loss']:.3e} | "
                    f"KL: {avg_metrics['kl_loss']:.3e} | "
                    f"LR: {avg_metrics['lr']:.3e} | "
                    f"{steps_per_sec:.1f} steps/s"
                )

                self._log_metrics(avg_metrics)

                if avg_metrics["total_loss"] < self.best_loss:
                    self.best_loss = avg_metrics["total_loss"]

                # Reset running stats
                running_flow_loss = 0.0
                running_kl_loss = 0.0
                running_total_loss = 0.0
                log_steps = 0
                start_time = time.time()

            # Sample generation
            if self.global_step % self.config.sample_every == 0:
                print(f"Generating samples at step {self.global_step}...")
                samples, conditioning = self.generate_samples(
                    self.config.num_sample_images
                )

                sample_path = (
                    self.output_dir
                    / "samples"
                    / f"samples_step_{self.global_step}.pt"
                )
                torch.save(
                    {
                        "samples": samples.cpu(),
                        "conditioning": conditioning.cpu(),
                    },
                    sample_path,
                )

                # Validation
                val_metrics = self.validate()
                if val_metrics:
                    print(
                        f"Validation - "
                        f"Flow: {val_metrics['val_flow_loss']:.3e}, "
                        f"KL: {val_metrics['val_kl_loss']:.3e}"
                    )

            # Checkpointing
            if self.global_step % self.config.save_every == 0:
                self.save_checkpoint()

        print("\nTraining complete!")
        final_loss = running_total_loss / max(log_steps, 1)
        self.save_checkpoint(is_best=(final_loss <= self.best_loss))


def create_lcfm_trainer(
    vae_encoder: nn.Module,
    train_dataset: torch.utils.data.Dataset,
    val_dataset: Optional[torch.utils.data.Dataset] = None,
    config: Optional[LCFMTrainingConfig] = None,
) -> LCFMTrainer:
    """
    Factory function to create LCFM model and trainer.

    Args:
        vae_encoder: Trained VAE encoder module.
        train_dataset: Training dataset.
        val_dataset: Optional validation dataset.
        config: Training configuration.

    Returns:
        LCFMTrainer ready to call .train()
    """
    if config is None:
        config = LCFMTrainingConfig()

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
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

    return LCFMTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
    )
