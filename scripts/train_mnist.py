#!/usr/bin/env python3
"""Train VAE on MNIST dataset with 32x32 padded images."""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

from galgenai.models import VAE
from galgenai.training import train
from galgenai.utils import get_device, get_device_name


def get_mnist_dataloaders(batch_size: int = 128, num_workers: int = 4):
    """
    Load MNIST dataset and create dataloaders with 32x32 padding.

    Args:
        batch_size: Batch size for training.
        num_workers: Number of worker processes for data loading.

    Returns:
        Train and test dataloaders.
    """
    # Transform: convert to tensor and pad to 32x32
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Pad(2),  # Pad 28x28 to 32x32
        ]
    )

    # Download and load MNIST
    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, test_loader


def visualize_reconstructions(model, test_loader, device, num_images=8):
    """
    Visualize original and reconstructed images.

    Args:
        model: Trained VAE model.
        test_loader: Test data loader.
        device: Device to run inference on.
        num_images: Number of images to visualize.
    """
    model.eval()

    # Get a batch of test images
    data, _ = next(iter(test_loader))
    data = data[:num_images].to(device)

    # Get reconstructions
    with torch.no_grad():
        reconstructions, _, _ = model(data)

    # Move to CPU for visualization
    data = data.cpu()
    reconstructions = reconstructions.cpu()

    # Create figure
    fig, axes = plt.subplots(2, num_images, figsize=(num_images * 2, 4))

    for i in range(num_images):
        # Original images
        axes[0, i].imshow(data[i, 0], cmap="gray")
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_title("Original", fontsize=10)

        # Reconstructions
        axes[1, i].imshow(reconstructions[i, 0], cmap="gray")
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_title("Reconstructed", fontsize=10)

    plt.tight_layout()
    plt.savefig("mnist_reconstructions.png", dpi=150, bbox_inches="tight")
    print("Saved reconstructions to mnist_reconstructions.png")
    plt.close()


def visualize_samples(model, device, num_samples=16):
    """
    Generate and visualize samples from the prior.

    Args:
        model: Trained VAE model.
        device: Device to run inference on.
        num_samples: Number of samples to generate.
    """
    model.eval()

    with torch.no_grad():
        samples = model.generate(num_samples, device)

    samples = samples.cpu()

    # Create grid
    grid_size = int(np.sqrt(num_samples))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(8, 8))

    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            axes[i, j].imshow(samples[idx, 0], cmap="gray")
            axes[i, j].axis("off")

    plt.tight_layout()
    plt.savefig("mnist_samples.png", dpi=150, bbox_inches="tight")
    print("Saved generated samples to mnist_samples.png")
    plt.close()


def main():
    """Main training script."""
    # Hyperparameters
    BATCH_SIZE = 128
    LATENT_DIM = 16
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 10
    INPUT_SIZE = 32  # Padded MNIST size
    BETA = 1.0  # Beta-VAE parameter

    print("=" * 60)
    print("VAE Training on MNIST")
    print("=" * 60)

    # Get device
    device = get_device()
    device_name = get_device_name()
    print(f"\nUsing device: {device_name}")

    # Load data
    print(f"\nLoading MNIST dataset (padded to {INPUT_SIZE}x{INPUT_SIZE})...")
    train_loader, test_loader = get_mnist_dataloaders(batch_size=BATCH_SIZE)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    # Create model
    print(f"\nInitializing VAE model (latent_dim={LATENT_DIM})...")
    model = VAE(in_channels=1, latent_dim=LATENT_DIM, input_size=INPUT_SIZE)
    model = model.to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train
    print("\nStarting training...")
    print("=" * 60)
    train(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=NUM_EPOCHS,
        reconstruction_loss_fn="mse",
        beta=BETA,
    )

    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)

    # Visualize results
    print("\nGenerating visualizations...")
    visualize_reconstructions(model, test_loader, device, num_images=8)
    visualize_samples(model, device, num_samples=16)

    # Save model
    model_path = "vae_mnist.pt"
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")


if __name__ == "__main__":
    main()
