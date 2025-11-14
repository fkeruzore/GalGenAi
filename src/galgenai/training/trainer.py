"""Training utilities for VAE models."""

import math
import torch
import torch.nn as nn
from typing import Tuple, Optional
from tqdm import tqdm


def vae_loss(
    reconstruction: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    reconstruction_loss_fn: str = "mse",
    beta: float = 1.0,
    ivar: torch.Tensor = None,
    mask: torch.Tensor = None,
    logvar_clamp: Optional[Tuple[float, float]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculate VAE loss = reconstruction_loss + beta * KL_divergence.

    Args:
        reconstruction: Reconstructed images.
        x: Original images.
        mu: Mean of latent distribution.
        logvar: Log variance of latent distribution.
        reconstruction_loss_fn: Type of reconstruction loss
            ('mse', 'bce', or 'masked_weighted_mse').
        beta: Weight for KL divergence term (beta-VAE).
        ivar: Inverse variance weights for each pixel (optional, required
            for 'masked_weighted_mse').
        mask: Boolean mask indicating valid pixels (optional, required
            for 'masked_weighted_mse').
        logvar_clamp: Optional tuple (min, max) to clamp logvar before
            exp() operation. Prevents numerical overflow. Default: None
            (no clamping). Recommended: (-10.0, 10.0).

    Returns:
        Tuple of (total_loss, reconstruction_loss, kl_divergence).
    """
    # Reconstruction loss
    if reconstruction_loss_fn == "mse":
        recon_loss = nn.functional.mse_loss(reconstruction, x, reduction="sum")
    elif reconstruction_loss_fn == "bce":
        recon_loss = nn.functional.binary_cross_entropy(
            reconstruction, x, reduction="sum"
        )
    elif reconstruction_loss_fn == "masked_weighted_mse":
        if ivar is None or mask is None:
            raise ValueError(
                "masked_weighted_mse requires both ivar and mask arguments"
            )
        # Compute squared error: (reconstruction - x)^2
        squared_error = (reconstruction - x).pow(2)

        # Weight by inverse variance and mask
        # ivar has units of 1/flux^2, so weighted error is dimensionless
        weighted_error = squared_error * ivar * mask.float()

        # Sum over all pixels and normalize by number of valid pixels
        # This makes the loss scale-invariant w.r.t. number of valid pixels
        num_valid_pixels = mask.float().sum()
        recon_loss = weighted_error.sum() / num_valid_pixels.clamp(min=1.0)

        # Scale by total number of pixels to match MSE magnitude
        # This ensures KL term has appropriate relative weight
        total_pixels = torch.tensor(
            reconstruction.numel(),
            dtype=torch.float32,
            device=reconstruction.device,
        )
        recon_loss = recon_loss * total_pixels
    else:
        raise ValueError(
            f"Unknown reconstruction loss: {reconstruction_loss_fn}"
        )

    # Clamp logvar for numerical stability if requested
    if logvar_clamp is not None:
        logvar = torch.clamp(logvar, min=logvar_clamp[0], max=logvar_clamp[1])

    # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Total loss
    total_loss = recon_loss + beta * kl_div

    return total_loss, recon_loss, kl_div


def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    reconstruction_loss_fn: str = "mse",
    beta: float = 1.0,
    scheduler: torch.optim.lr_scheduler._LRScheduler = None,
    max_grad_norm: Optional[float] = None,
    logvar_clamp: Optional[Tuple[float, float]] = None,
    detect_nan_per_batch: bool = False,
    checkpoint_path: Optional[str] = None,
) -> Tuple[float, float, float]:
    """
    Train the VAE for one epoch.

    Args:
        model: VAE model.
        dataloader: Training data loader.
        optimizer: Optimizer.
        device: Device to train on.
        reconstruction_loss_fn: Type of reconstruction loss.
        beta: Weight for KL divergence.
        scheduler: Optional learning rate scheduler.
        max_grad_norm: If provided, clip gradients to this max norm.
            Recommended: 1.0 for stability.
        logvar_clamp: Optional tuple (min, max) to clamp logvar.
        detect_nan_per_batch: If True, check for NaN after each batch
            and stop training immediately if detected.
        checkpoint_path: Path to restore model from if NaN detected.

    Returns:
        Tuple of (avg_total_loss, avg_recon_loss, avg_kl_loss).
    """
    model.train()
    total_loss_sum = 0.0
    recon_loss_sum = 0.0
    kl_loss_sum = 0.0
    num_batches = 0

    progress_bar = tqdm(dataloader, desc="Training")

    for _batch_idx, batch in enumerate(progress_bar):
        # Handle both single tensor and tuple formats
        if isinstance(batch, (tuple, list)):
            # New format: (flux, ivar, mask)
            data = batch[0].to(device)
            ivar = batch[1].to(device) if len(batch) > 1 else None
            mask = batch[2].to(device) if len(batch) > 2 else None
        else:
            # Old format: single tensor
            data = batch.to(device)
            ivar = None
            mask = None

        # Forward pass
        reconstruction, mu, logvar = model(data)

        # Calculate loss
        total_loss, recon_loss, kl_loss = vae_loss(
            reconstruction,
            data,
            mu,
            logvar,
            reconstruction_loss_fn,
            beta,
            ivar=ivar,
            mask=mask,
            logvar_clamp=logvar_clamp,
        )

        # Normalize by batch size
        batch_size = data.size(0)
        total_loss = total_loss / batch_size
        recon_loss = recon_loss / batch_size
        kl_loss = kl_loss / batch_size

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping for stability
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_grad_norm
            )

        optimizer.step()

        # Batch-level NaN detection (optional, for early stopping)
        if detect_nan_per_batch:
            if not math.isfinite(total_loss.item()):
                print("\n" + "=" * 60)
                print(f"[ERROR] NaN detected at batch {_batch_idx + 1}!")
                print(f"  Total Loss: {total_loss.item()}")
                print(f"  Recon Loss: {recon_loss.item()}")
                print(f"  KL Loss: {kl_loss.item()}")
                if checkpoint_path is not None:
                    print(f"Restoring from checkpoint: {checkpoint_path}")
                    try:
                        model.load_state_dict(
                            torch.load(checkpoint_path, map_location=device)
                        )
                        print("Model restored to last good state.")
                    except FileNotFoundError:
                        print("[WARNING] No checkpoint found.")
                else:
                    print("[WARNING] No checkpoint to restore from.")
                print("Stopping training early.")
                print("=" * 60)
                # Return NaN to signal caller to stop
                return float('nan'), float('nan'), float('nan')

        # Accumulate losses
        total_loss_sum += total_loss.item()
        recon_loss_sum += recon_loss.item()
        kl_loss_sum += kl_loss.item()
        num_batches += 1

        # Update progress bar
        progress_bar.set_postfix(
            {
                "loss": f"{total_loss.item():.4f}",
                "recon": f"{recon_loss.item():.4f}",
                "kl": f"{kl_loss.item():.4f}",
            }
        )

    # Return average losses
    avg_total_loss = total_loss_sum / num_batches
    avg_recon_loss = recon_loss_sum / num_batches
    avg_kl_loss = kl_loss_sum / num_batches

    return avg_total_loss, avg_recon_loss, avg_kl_loss


def train(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    reconstruction_loss_fn: str = "mse",
    beta: float = 1.0,
    scheduler: torch.optim.lr_scheduler._LRScheduler = None,
    checkpoint_path: str = None,
    max_grad_norm: Optional[float] = None,
    logvar_clamp: Optional[Tuple[float, float]] = None,
    warmup_epochs: int = 0,
    detect_nan_per_batch: bool = False,
) -> None:
    """
    Train the VAE for multiple epochs.

    Args:
        model: VAE model.
        train_loader: Training data loader.
        optimizer: Optimizer.
        device: Device to train on.
        num_epochs: Number of epochs to train.
        reconstruction_loss_fn: Type of reconstruction loss.
        beta: Weight for KL divergence.
        scheduler: Optional learning rate scheduler (applied after warmup).
        checkpoint_path: Optional path to save checkpoints. If provided,
            model state will be saved after each successful epoch and
            restored if NaN is detected.
        max_grad_norm: If provided, clip gradients to this max norm for
            stability. Recommended: 1.0. Default: None (no clipping).
        logvar_clamp: Optional tuple (min, max) to clamp log variance
            before exp() to prevent overflow. Recommended: (-10.0, 10.0).
            Default: None (no clamping).
        warmup_epochs: Number of epochs to linearly warm up learning rate
            from 0 to the base LR. Default: 0 (no warmup).
        detect_nan_per_batch: If True, check for NaN after every batch
            and stop immediately. Default: False (check per epoch only).
    """
    print(f"Training on device: {device}")
    print(f"Number of epochs: {num_epochs}")
    print(f"Reconstruction loss: {reconstruction_loss_fn}")
    print(f"Beta: {beta}")
    if scheduler is not None:
        print(f"Learning rate scheduler: {scheduler.__class__.__name__}")
    if checkpoint_path is not None:
        print(f"Checkpointing enabled: {checkpoint_path}")
    if max_grad_norm is not None:
        print(f"Gradient clipping: max_norm={max_grad_norm}")
    if logvar_clamp is not None:
        print(f"Log variance clamping: {logvar_clamp}")
    if warmup_epochs > 0:
        print(f"LR warmup: {warmup_epochs} epochs")
    if detect_nan_per_batch:
        print("Batch-level NaN detection: enabled")
    print("-" * 60)

    # Setup learning rate warmup
    base_lr = optimizer.param_groups[0]["lr"]
    if warmup_epochs > 0:
        # Start with very small LR for warmup
        for param_group in optimizer.param_groups:
            param_group["lr"] = 0.0

    for epoch in range(1, num_epochs + 1):
        # Apply warmup schedule if in warmup period
        if warmup_epochs > 0 and epoch <= warmup_epochs:
            # Linear warmup: lr = base_lr * (epoch / warmup_epochs)
            warmup_lr = base_lr * (epoch / warmup_epochs)
            for param_group in optimizer.param_groups:
                param_group["lr"] = warmup_lr

        # Get current learning rate
        current_lr = optimizer.param_groups[0]["lr"]
        warmup_suffix = " [warmup]" if (warmup_epochs > 0 and epoch <= warmup_epochs) else ""
        print(f"\nEpoch {epoch}/{num_epochs} (lr: {current_lr:.6f}){warmup_suffix}")

        avg_total_loss, avg_recon_loss, avg_kl_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            reconstruction_loss_fn,
            beta,
            max_grad_norm=max_grad_norm,
            logvar_clamp=logvar_clamp,
            detect_nan_per_batch=detect_nan_per_batch,
            checkpoint_path=checkpoint_path,
        )

        # Check if loss is NaN or Inf (epoch-level check)
        if not math.isfinite(avg_total_loss):
            print("\n" + "=" * 60)
            print("[ERROR] Non-finite loss detected during training!")
            print(f"  Total Loss: {avg_total_loss}")
            print(f"  Recon Loss: {avg_recon_loss}")
            print(f"  KL Loss: {avg_kl_loss}")
            if checkpoint_path is not None:
                print(f"Restoring model from last checkpoint: {checkpoint_path}")
                try:
                    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
                    print("Model successfully restored to last good state.")
                except FileNotFoundError:
                    print("[WARNING] No checkpoint found. Model state not restored.")
            else:
                print("[WARNING] No checkpoint path provided. Cannot restore model.")
            print("Stopping training early.")
            print("=" * 60)
            break

        print(
            f"Epoch {epoch} Summary - "
            f"Total Loss: {avg_total_loss:.4f}, "
            f"Recon Loss: {avg_recon_loss:.4f}, "
            f"KL Loss: {avg_kl_loss:.4f}"
        )

        # Save checkpoint after successful epoch
        if checkpoint_path is not None:
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  Checkpoint saved: {checkpoint_path}")

        # Step the scheduler after warmup period
        if scheduler is not None and epoch > warmup_epochs:
            scheduler.step()
