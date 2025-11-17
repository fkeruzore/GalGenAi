"""Training utilities for VAE models."""

import math
import torch
import torch.nn as nn
from typing import Tuple
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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculate VAE loss = reconstruction_loss + beta * KL_divergence.

    Args:
        reconstruction: Reconstructed images.
        x: Original images.
        mu: Mean of latent distribution.
        logvar: Log variance of latent distribution (already clamped by
            encoder).
        reconstruction_loss_fn: Type of reconstruction loss ('mse' or
            'masked_weighted_mse').
        beta: Weight for KL divergence term (beta-VAE).
        ivar: Inverse variance weights for each pixel (optional, required
            for 'masked_weighted_mse').
        mask: Boolean mask indicating valid pixels (optional, required
            for 'masked_weighted_mse').

    Returns:
        Tuple of (total_loss, reconstruction_loss, kl_divergence).
    """
    # Reconstruction loss
    if reconstruction_loss_fn == "mse":
        recon_loss = nn.functional.mse_loss(reconstruction, x, reduction="sum")
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

    # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # Note: logvar is already clamped by the encoder for numerical stability
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
    max_grad_norm: float = 1.0,
) -> Tuple[float, float, float]:
    """
    Train the VAE for one epoch.

    Args:
        model: VAE model.
        dataloader: Training data loader.
        optimizer: Optimizer.
        device: Device to train on.
        reconstruction_loss_fn: Type of reconstruction loss ('mse' or
            'masked_weighted_mse').
        beta: Weight for KL divergence.
        scheduler: Optional learning rate scheduler (stepped per batch).
        max_grad_norm: Maximum gradient norm for gradient clipping.
            Default: 1.0.

    Returns:
        Tuple of (avg_total_loss, avg_recon_loss, avg_kl_loss).
    """
    model.train()
    total_loss_sum = 0.0
    recon_loss_sum = 0.0
    kl_loss_sum = 0.0
    num_batches = 0

    progress_bar = tqdm(dataloader, desc="Training")

    for batch in progress_bar:
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()

        # Step the scheduler after each batch if configured
        if scheduler is not None:
            scheduler.step()

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
    max_grad_norm: float = 1.0,
) -> None:
    """
    Train the VAE for multiple epochs.

    Args:
        model: VAE model.
        train_loader: Training data loader.
        optimizer: Optimizer.
        device: Device to train on.
        num_epochs: Number of epochs to train.
        reconstruction_loss_fn: Type of reconstruction loss ('mse' or
            'masked_weighted_mse').
        beta: Weight for KL divergence.
        scheduler: Optional learning rate scheduler (stepped per batch).
        max_grad_norm: Maximum gradient norm for gradient clipping.
            Default: 1.0.
    """
    print(f"Training on device: {device}")
    print(f"Number of epochs: {num_epochs}")
    print(f"Reconstruction loss: {reconstruction_loss_fn}")
    print(f"Beta: {beta}")
    print(f"Gradient clipping: max_norm={max_grad_norm}")
    if scheduler is not None:
        print(f"Learning rate scheduler: {scheduler.__class__.__name__}")
    print("-" * 60)

    for epoch in range(1, num_epochs + 1):
        # Get current learning rate
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch {epoch}/{num_epochs} (lr: {current_lr:.6f})")

        avg_total_loss, avg_recon_loss, avg_kl_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            reconstruction_loss_fn,
            beta,
            scheduler,
            max_grad_norm,
        )

        # Check if loss is NaN or Inf
        if not math.isfinite(avg_total_loss):
            print("\n" + "=" * 60)
            print("[ERROR] Non-finite loss detected during training!")
            print("Stopping training early.")
            print("=" * 60)
            break

        print(
            f"Epoch {epoch} Summary - "
            f"Total Loss: {avg_total_loss:.4f}, "
            f"Recon Loss: {avg_recon_loss:.4f}, "
            f"KL Loss: {avg_kl_loss:.4f}"
        )
