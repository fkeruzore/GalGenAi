"""DL-CFM trainer implementation."""

from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader

from ..models.lcfm import LCFM
from .config import DLCFMTrainingConfig
from .lcfm_trainer import LCFMTrainer
from .utils import (
    dlcfm_disentanglement_loss,
    extract_dlcfm_batch_data,
)


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

        # 1. Encode to get latent features
        f, mu, logvar = self.model.encode(x1)

        # 2. Flow matching loss
        flow_loss = self.model.compute_flow_loss(x1, f, ivar=ivar, mask=mask)

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

        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        self._clip_gradients()

        # Set LR before stepping so step 0 uses the warmup value.
        current_lr = self._get_lr_with_warmup()
        self._set_lr(current_lr)
        self.optimizer.step()

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

            f, mu, logvar = self.model.encode(x1)
            flow_loss = self.model.compute_flow_loss(
                x1, f, ivar=ivar, mask=mask
            )

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

    def _pbar_postfix(self, loss_dict: Dict[str, float]) -> Dict[str, str]:
        """Progress bar postfix with disentanglement loss."""
        return {
            "loss": f"{loss_dict['total_loss']:.3e}",
            "flow": f"{loss_dict['flow_loss']:.3e}",
            "dis": (f"{loss_dict['disentanglement_loss']:.3e}"),
            "lr": f"{loss_dict['lr']:.3e}",
        }

    def _val_summary(self, val_metrics: Dict[str, float]) -> str:
        """Validation summary with disentanglement loss."""
        return (
            f"  Step {self.global_step} Val"
            f" - Flow: "
            f"{val_metrics['val_flow_loss']:.3e}"
            f", Dis: "
            f"{val_metrics['val_disentanglement_loss']:.3e}"
            f", Total: "
            f"{val_metrics['val_total_loss']:.3e}"
        )
