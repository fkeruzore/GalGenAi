"""LCFM trainer implementation."""

from typing import Any, Dict, Optional, Tuple

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

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
        print(f"  Beta (KL weight): {config.beta}")
        print(f"  Learning rate: {config.learning_rate}")
        print(f"  Total training steps: {config.num_steps:,}")

    def _setup_optimizer(self):
        """Set up AdamW with cosine annealing (default) or custom
        scheduler."""
        trainable_params = [
            p for p in self.model.parameters() if p.requires_grad
        ]
        self.optimizer = AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.999),
        )

        if self.config.scheduler_factory is not None:
            self.scheduler = self.config.scheduler_factory(self.optimizer)
        else:
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
        if self.scheduler is not None:
            return self.scheduler.get_last_lr()[0]
        return self.config.learning_rate

    def _set_lr(self, lr: float):
        """Set learning rate for all parameter groups."""
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _train_step(self, batch: Any) -> Dict[str, float]:
        """Execute single LCFM training step."""
        x, ivar, mask = extract_batch_data(batch, self.device)

        # Compute loss using model method
        loss, loss_dict = self.model.compute_loss(
            x, ivar=ivar, mask=mask, return_components=True
        )

        # Backward pass
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self._clip_gradients()
        self.optimizer.step()

        # Update LR with warmup handling
        current_lr = self._get_lr_with_warmup()
        self._set_lr(current_lr)

        if (
            self.scheduler is not None
            and self.global_step >= self.config.warmup_steps
        ):
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
        total_total_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            x, ivar, mask = extract_batch_data(batch, self.device)
            _, loss_dict = self.model.compute_loss(
                x, ivar=ivar, mask=mask, return_components=True
            )

            total_flow_loss += loss_dict["flow_loss"]
            total_kl_loss += loss_dict["kl_loss"]
            total_total_loss += loss_dict["total_loss"]
            num_batches += 1

        self.model.train()
        return {
            "val_flow_loss": total_flow_loss / num_batches,
            "val_kl_loss": total_kl_loss / num_batches,
            "val_total_loss": total_total_loss / num_batches,
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
        self.model.train()
        return samples, train_images

    def _pbar_postfix(self, loss_dict: Dict[str, float]) -> Dict[str, str]:
        """Progress bar postfix. Override for custom display."""
        return {
            "loss": f"{loss_dict['total_loss']:.3e}",
            "flow": f"{loss_dict['flow_loss']:.3e}",
            "kl": f"{loss_dict['kl_loss']:.3e}",
            "lr": f"{loss_dict['lr']:.3e}",
        }

    def _val_summary(self, val_metrics: Dict[str, float]) -> str:
        """One-line validation summary. Override for custom format."""
        return (
            f"  Step {self.global_step} Val"
            f" - Flow: {val_metrics['val_flow_loss']:.3e}"
            f", KL: {val_metrics['val_kl_loss']:.3e}"
            f", Total: {val_metrics['val_total_loss']:.3e}"
        )

    def train(self):
        """Main step-based training loop."""
        print(f"\nStarting training from step {self.global_step}")
        print(f"Training for {self.config.num_steps - self.global_step} steps")

        self.model.train()
        if self.device.type == "mps":
            print(
                "torch.compile() skipped on MPS (inductor Metal backend bug)"
            )
        else:
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

        # Discover metric keys from first step, then
        # track running averages dynamically.
        running: Dict[str, float] = {}
        metric_keys: Optional[list] = None
        log_steps = 0

        pbar = tqdm(
            total=self.config.num_steps,
            initial=self.global_step,
            desc="Training",
            unit="step",
        )

        while self.global_step < self.config.num_steps:
            batch = next(data_iter)
            loss_dict = self._train_step(batch)

            # Initialise running-average keys on first step
            if metric_keys is None:
                metric_keys = [k for k in loss_dict if k != "lr"]
                running = {k: 0.0 for k in metric_keys}

            for k in metric_keys:
                running[k] += loss_dict[k]
            log_steps += 1

            self.global_step += 1
            pbar.update(1)
            pbar.set_postfix(self._pbar_postfix(loss_dict))

            # Periodic logging
            if self.global_step % self.config.log_every == 0:
                avg_metrics = {k: running[k] / log_steps for k in metric_keys}
                avg_metrics["lr"] = loss_dict["lr"]

                # Validation
                val_metrics = {}
                if self.global_step % self.config.validate_every == 0:
                    val_metrics = self.validate()
                    if val_metrics:
                        pbar.write(self._val_summary(val_metrics))
                        avg_metrics.update(val_metrics)

                self._log_metrics(avg_metrics)

                # Track best loss
                if val_metrics:
                    current_loss = val_metrics["val_total_loss"]
                else:
                    current_loss = avg_metrics["total_loss"]

                if current_loss < self.best_loss:
                    self.best_loss = current_loss
                    loss_type = "val" if val_metrics else "train"
                    self.save_checkpoint(is_best=True)
                    pbar.write(
                        f"  New best {loss_type} loss "
                        f"{current_loss:.4f} at step "
                        f"{self.global_step} — saved best.pt"
                    )

                running = {k: 0.0 for k in metric_keys}
                log_steps = 0

            # Sample generation
            if self.global_step % self.config.sample_every == 0:
                pbar.write(f"Generating samples at step {self.global_step}...")
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

            # Checkpointing
            if self.global_step % self.config.save_every == 0:
                self.save_checkpoint()

        pbar.close()
        print("\nTraining complete!")

        self.save_checkpoint()
