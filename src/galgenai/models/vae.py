"""Variational Autoencoder with ResNet architecture."""

import torch
import torch.nn as nn
from typing import Tuple

from .layers import DownsampleBlock, UpsampleBlock, ResidualBlock


class VAEEncoder(nn.Module):
    """
    VAE Encoder with 7 stages of downsampling using ResNet blocks.

    Architecture:
    - Initial dense embedding: 1 -> 16 channels
    - 7 downsampling stages with depths: 16 -> 32 -> 64 -> 128 -> 256 -> 512 -> 512
    - Each stage has 2 residual blocks
    - Final dense layer compresses to latent_dim (outputs mean and log_var)

    Args:
        in_channels: Number of input channels (e.g., 1 for grayscale).
        latent_dim: Dimension of the latent space.
        input_size: Spatial size of input images (assumes square images).
    """

    def __init__(self, in_channels: int = 1, latent_dim: int = 16, input_size: int = 32):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_size = input_size

        # Calculate the spatial size after 7 downsampling operations
        # Each downsample reduces size by factor of 2
        self.final_spatial_size = input_size // (2**7)  # 32 -> 0.25, need to handle carefully

        # Initial channel-wise dense embedding
        self.initial_conv = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)

        # 7 downsampling stages with increasing depth
        # Depths: 16 -> 32 -> 64 -> 128 -> 256 -> 512 -> 512
        self.stage1 = DownsampleBlock(16, 32)
        self.stage2 = DownsampleBlock(32, 64)
        self.stage3 = DownsampleBlock(64, 128)
        self.stage4 = DownsampleBlock(128, 256)
        self.stage5 = DownsampleBlock(256, 512)
        self.stage6 = DownsampleBlock(512, 512)
        self.stage7 = DownsampleBlock(512, 512)

        # Calculate flattened size after convolutions
        # After 7 downsamples: 32 -> 16 -> 8 -> 4 -> 2 -> 1 -> 0.5 (not valid!)
        # We need at least 128x128 for 7 downsamples, or fewer stages for 32x32
        # For 32x32: can do 5 downsamples (32->16->8->4->2->1)
        # Let's adjust: use 5 stages for 32x32 input

        # Recalculating for 5 stages to fit 32x32 input
        # Will modify architecture to be adaptive

        self.flatten_size = 512 * (input_size // (2**7)) ** 2 if input_size >= 128 else 512

        # Dense layers for mean and log variance
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through encoder.

        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            Tuple of (mu, log_var) for the latent distribution.
        """
        # Initial embedding
        x = self.initial_conv(x)

        # Downsampling stages
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)

        # Only use stages 6 and 7 if input is large enough
        if self.input_size >= 64:
            x = self.stage6(x)
        if self.input_size >= 128:
            x = self.stage7(x)

        # Flatten
        x = torch.flatten(x, start_dim=1)

        # Latent distribution parameters
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar


class VAEDecoder(nn.Module):
    """
    VAE Decoder that mirrors the encoder architecture.

    Args:
        latent_dim: Dimension of the latent space.
        out_channels: Number of output channels (e.g., 1 for grayscale).
        input_size: Spatial size of target output images.
    """

    def __init__(self, latent_dim: int = 16, out_channels: int = 1, input_size: int = 32):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_size = input_size

        # Determine number of stages based on input size
        self.num_stages = 5 if input_size < 64 else (6 if input_size < 128 else 7)
        self.initial_spatial_size = input_size // (2**self.num_stages)

        self.flatten_size = 512 * self.initial_spatial_size**2 if self.initial_spatial_size > 0 else 512

        # Dense layer to expand from latent space
        self.fc = nn.Linear(latent_dim, self.flatten_size)

        # Unflatten parameters
        self.unflatten_channels = 512
        self.unflatten_size = self.initial_spatial_size if self.initial_spatial_size > 0 else 1

        # Upsampling stages (mirror of encoder)
        # Depths: 512 -> 512 -> 256 -> 128 -> 64 -> 32 -> 16
        if self.num_stages >= 7:
            self.stage7 = UpsampleBlock(512, 512)
            self.stage6 = UpsampleBlock(512, 256)
        elif self.num_stages >= 6:
            self.stage6 = UpsampleBlock(512, 256)
        else:
            # For 5 stages, start directly
            pass

        self.stage5 = UpsampleBlock(512 if self.num_stages < 6 else 256, 256 if self.num_stages < 6 else 128)
        self.stage4 = UpsampleBlock(256 if self.num_stages < 6 else 128, 128 if self.num_stages < 6 else 64)
        self.stage3 = UpsampleBlock(128 if self.num_stages < 6 else 64, 64 if self.num_stages < 6 else 32)
        self.stage2 = UpsampleBlock(64 if self.num_stages < 6 else 32, 32 if self.num_stages < 6 else 16)
        self.stage1 = UpsampleBlock(32 if self.num_stages < 6 else 16, 16)

        # Final convolution to output channels
        self.final_conv = nn.Conv2d(16, out_channels, kernel_size=3, padding=1)

        # Softplus activation for positivity
        self.softplus = nn.Softplus()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through decoder.

        Args:
            z: Latent vector of shape (batch_size, latent_dim).

        Returns:
            Reconstructed image of shape (batch_size, out_channels, height, width).
        """
        # Expand from latent space
        x = self.fc(z)
        x = x.view(-1, self.unflatten_channels, self.unflatten_size, self.unflatten_size)

        # Upsampling stages
        if self.num_stages >= 7:
            x = self.stage7(x)
            x = self.stage6(x)
        elif self.num_stages >= 6:
            x = self.stage6(x)

        x = self.stage5(x)
        x = self.stage4(x)
        x = self.stage3(x)
        x = self.stage2(x)
        x = self.stage1(x)

        # Final output
        x = self.final_conv(x)
        x = self.softplus(x)

        return x


class VAE(nn.Module):
    """
    Complete Variational Autoencoder with ResNet architecture.

    Args:
        in_channels: Number of input channels.
        latent_dim: Dimension of the latent space.
        input_size: Spatial size of input images (assumes square).
    """

    def __init__(self, in_channels: int = 1, latent_dim: int = 16, input_size: int = 32):
        super().__init__()
        self.encoder = VAEEncoder(in_channels, latent_dim, input_size)
        self.decoder = VAEDecoder(latent_dim, in_channels, input_size)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = mu + sigma * epsilon.

        Args:
            mu: Mean of the latent distribution.
            logvar: Log variance of the latent distribution.

        Returns:
            Sampled latent vector.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the VAE.

        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            Tuple of (reconstruction, mu, logvar).
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decoder(z)
        return reconstruction, mu, logvar

    def generate(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """
        Generate new samples from the prior distribution.

        Args:
            num_samples: Number of samples to generate.
            device: Device to generate samples on.

        Returns:
            Generated samples.
        """
        z = torch.randn(num_samples, self.encoder.latent_dim, device=device)
        samples = self.decoder(z)
        return samples
