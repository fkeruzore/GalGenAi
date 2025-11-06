"""Training utilities for VAE models."""

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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculate VAE loss = reconstruction_loss + beta * KL_divergence.

    Args:
        reconstruction: Reconstructed images.
        x: Original images.
        mu: Mean of latent distribution.
        logvar: Log variance of latent distribution.
        reconstruction_loss_fn: Type of reconstruction loss ('mse' or 'bce').
        beta: Weight for KL divergence term (beta-VAE).

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
    else:
        raise ValueError(
            f"Unknown reconstruction loss: {reconstruction_loss_fn}"
        )

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

    Returns:
        Tuple of (avg_total_loss, avg_recon_loss, avg_kl_loss).
    """
    model.train()
    total_loss_sum = 0.0
    recon_loss_sum = 0.0
    kl_loss_sum = 0.0
    num_batches = 0

    progress_bar = tqdm(dataloader, desc="Training")

    for batch_idx, data in enumerate(progress_bar):
        data = data.to(device)

        # Forward pass
        reconstruction, mu, logvar = model(data)

        # Calculate loss
        total_loss, recon_loss, kl_loss = vae_loss(
            reconstruction, data, mu, logvar, reconstruction_loss_fn, beta
        )

        # Normalize by batch size
        batch_size = data.size(0)
        total_loss = total_loss / batch_size
        recon_loss = recon_loss / batch_size
        kl_loss = kl_loss / batch_size

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

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
        scheduler: Optional learning rate scheduler.
    """
    print(f"Training on device: {device}")
    print(f"Number of epochs: {num_epochs}")
    print(f"Reconstruction loss: {reconstruction_loss_fn}")
    print(f"Beta: {beta}")
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
        )

        print(
            f"Epoch {epoch} Summary - "
            f"Total Loss: {avg_total_loss:.4f}, "
            f"Recon Loss: {avg_recon_loss:.4f}, "
            f"KL Loss: {avg_kl_loss:.4f}"
        )

        # Step the scheduler after each epoch
        if scheduler is not None:
            scheduler.step()
