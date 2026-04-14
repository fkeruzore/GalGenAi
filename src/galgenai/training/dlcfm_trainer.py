"""DL-CFM trainer implementation."""

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models.lcfm import LCFM
from .config import DLCFMTrainingConfig
from .lcfm_trainer import LCFMTrainer
from .utils import dlcfm_disentanglement_loss, extract_dlcfm_batch_data


class DLCFMTrainer(LCFMTrainer):
    """
    Trainer for DL-CFM (Disentangled Latent Conditional Flow Matching).

    Extends LCFMTrainer by replacing the standard KL loss with the
    DL-CFM disentanglement loss from Ganguli et al. (2025). Expects
    batches with auxiliary variables (flux, ivar, mask, aux_vars).
    """

    def __init__(
        self,
        model: LCFM,
        train_loader: DataLoader,
        config: DLCFMTrainingConfig,
        val_loader: Optional[DataLoader] = None,
    ):
        super().__init__(model, train_loader, config, val_loader)
        self.dlcfm_config = config

        print(f"  DL-CFM beta: {config.dlcfm_beta}")
        print(f"  lambda1: {config.lambda1}")
        print(f"  lambda2: {config.lambda2}")
        print(f"  K: {config.K}")
        print(f"  n_aux: {config.n_aux}")

    def _train_step(self, batch: Any) -> Dict[str, float]:
        """Execute single DL-CFM training step."""
        x1, ivar, mask, aux_vars = extract_dlcfm_batch_data(batch, self.device)
        batch_size = x1.shape[0]

        # 1. Encode to get latent features
        f, mu, logvar = self.model.encode(x1)

        # 2. Flow matching loss (replicated from LCFM.compute_loss)
        x0 = torch.randn_like(x1)
        t = torch.rand(batch_size, device=self.device)
        t_broadcast = t[:, None, None, None]
        x_t = (1 - t_broadcast) * x0 + t_broadcast * x1
        u_t = x1 - x0
        v_pred = self.model.velocity_net(x_t, f, t)

        if ivar is not None and mask is not None:
            squared_error = (v_pred - u_t).pow(2)
            mask_float = mask.float()
            weighted_error = squared_error * ivar * mask_float
            num_valid = mask_float.sum().clamp(min=1.0)
            flow_loss = weighted_error.sum() / num_valid
        else:
            flow_loss = F.mse_loss(v_pred, u_t)

        # 3. DL-CFM disentanglement loss
        cfg = self.dlcfm_config
        dis_loss, components = dlcfm_disentanglement_loss(
            mu,
            logvar,
            aux_vars,
            n_aux=cfg.n_aux,
            beta=cfg.dlcfm_beta,
            lambda1=cfg.lambda1,
            lambda2=cfg.lambda2,
            K=cfg.K,
            tau_sq=cfg.tau_sq,
        )

        # 4. Total loss
        total_loss = flow_loss + dis_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self._clip_gradients()
        self.optimizer.step()

        current_lr = self._get_lr_with_warmup()
        self._set_lr(current_lr)

        if (
            self.scheduler is not None
            and self.global_step >= self.config.warmup_steps
        ):
            self.scheduler.step()

        return {
            "flow_loss": flow_loss.item(),
            "kl": components["kl"],
            "align": components["align"],
            "intra_decorr": components["intra_decorr"],
            "inter_decorr": components["inter_decorr"],
            "disentanglement_loss": components["disentanglement_loss"],
            "total_loss": total_loss.item(),
            "lr": current_lr,
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Compute validation loss with DL-CFM components."""
        if self.val_loader is None:
            return {}

        self.model.eval()
        cfg = self.dlcfm_config
        totals = {
            "flow_loss": 0.0,
            "kl": 0.0,
            "align": 0.0,
            "intra_decorr": 0.0,
            "inter_decorr": 0.0,
            "disentanglement_loss": 0.0,
            "total_loss": 0.0,
        }
        num_batches = 0

        for batch in self.val_loader:
            x1, ivar, mask, aux_vars = extract_dlcfm_batch_data(
                batch, self.device
            )
            batch_size = x1.shape[0]

            f, mu, logvar = self.model.encode(x1)

            # Flow matching loss
            x0 = torch.randn_like(x1)
            t = torch.rand(batch_size, device=self.device)
            t_broadcast = t[:, None, None, None]
            x_t = (1 - t_broadcast) * x0 + t_broadcast * x1
            u_t = x1 - x0
            v_pred = self.model.velocity_net(x_t, f, t)

            if ivar is not None and mask is not None:
                squared_error = (v_pred - u_t).pow(2)
                mask_float = mask.float()
                weighted_error = squared_error * ivar * mask_float
                num_valid = mask_float.sum().clamp(min=1.0)
                flow_loss = weighted_error.sum() / num_valid
            else:
                flow_loss = F.mse_loss(v_pred, u_t)

            dis_loss, components = dlcfm_disentanglement_loss(
                mu,
                logvar,
                aux_vars,
                n_aux=cfg.n_aux,
                beta=cfg.dlcfm_beta,
                lambda1=cfg.lambda1,
                lambda2=cfg.lambda2,
                K=cfg.K,
                tau_sq=cfg.tau_sq,
            )

            total_loss = flow_loss + dis_loss

            totals["flow_loss"] += flow_loss.item()
            totals["kl"] += components["kl"]
            totals["align"] += components["align"]
            totals["intra_decorr"] += components["intra_decorr"]
            totals["inter_decorr"] += components["inter_decorr"]
            totals["disentanglement_loss"] += components[
                "disentanglement_loss"
            ]
            totals["total_loss"] += total_loss.item()
            num_batches += 1

        self.model.train()
        return {f"val_{k}": v / num_batches for k, v in totals.items()}

    def train(self):
        """Main step-based training loop with DL-CFM metrics."""
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

        # Running averages for all DL-CFM components
        component_keys = [
            "flow_loss",
            "kl",
            "align",
            "intra_decorr",
            "inter_decorr",
            "disentanglement_loss",
            "total_loss",
        ]
        running = {k: 0.0 for k in component_keys}
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

            for k in component_keys:
                running[k] += loss_dict[k]
            log_steps += 1

            self.global_step += 1
            pbar.update(1)

            pbar.set_postfix(
                {
                    "loss": f"{loss_dict['total_loss']:.3e}",
                    "flow": f"{loss_dict['flow_loss']:.3e}",
                    "dis": (f"{loss_dict['disentanglement_loss']:.3e}"),
                    "lr": f"{loss_dict['lr']:.3e}",
                }
            )

            if self.global_step % self.config.log_every == 0:
                avg_metrics = {
                    k: running[k] / log_steps for k in component_keys
                }
                avg_metrics["lr"] = loss_dict["lr"]

                # Validation
                val_metrics = {}
                if self.global_step % self.config.validate_every == 0:
                    val_metrics = self.validate()
                    if val_metrics:
                        pbar.write(
                            f"  Step {self.global_step} Val"
                            f" - Flow: "
                            f"{val_metrics['val_flow_loss']:.3e}"
                            f", Dis: "
                            f"{val_metrics['val_disentanglement_loss']:.3e}"
                            f", Total: "
                            f"{val_metrics['val_total_loss']:.3e}"
                        )
                        avg_metrics.update(val_metrics)

                self._log_metrics(avg_metrics)

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

                running = {k: 0.0 for k in component_keys}
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
