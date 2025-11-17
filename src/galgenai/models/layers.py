"""ResNet blocks and custom layers for VAE architecture."""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    Residual block with two convolutional layers and a skip connection.

    Args:
        channels: Number of input and output channels.
    """

    def __init__(self, channels: int):
        super().__init__()

        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with skip connection."""
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.relu(out)

        return out


class DownsampleBlock(nn.Module):
    """
    Downsampling block with two residual blocks and spatial downsampling.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        # Downsampling with stride 2
        self.downsample = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=2, padding=1
        )

        # Two residual blocks at the new resolution
        self.res_block1 = ResidualBlock(out_channels)
        self.res_block2 = ResidualBlock(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with downsampling and residual blocks."""
        x = self.downsample(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        return x


class UpsampleBlock(nn.Module):
    """
    Upsampling block with two residual blocks and spatial upsampling.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        # Two residual blocks at current resolution
        self.res_block1 = ResidualBlock(in_channels)
        self.res_block2 = ResidualBlock(in_channels)

        # Upsampling with transposed convolution
        self.upsample = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=4, stride=2, padding=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual blocks and upsampling."""
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.upsample(x)
        return x
