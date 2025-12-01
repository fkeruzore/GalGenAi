"""GalGenAI - Generative AI models for galaxy images."""

from .models import (
    VAE,
    VAEEncoder,
    VAEDecoder,
    ResidualBlock,
    DownsampleBlock,
    UpsampleBlock,
    SqueezeExcitationBlock,
    LCFM,
    VelocityUNet,
    LatentStochasticLayer,
)
from .training import vae_loss, train_epoch, train
from .utils import get_device, get_device_name

__version__ = "0.1.0"

__all__ = [
    "VAE",
    "VAEEncoder",
    "VAEDecoder",
    "ResidualBlock",
    "DownsampleBlock",
    "UpsampleBlock",
    "SqueezeExcitationBlock",
    "LCFM",
    "VelocityUNet",
    "LatentStochasticLayer",
    "vae_loss",
    "train_epoch",
    "train",
    "get_device",
    "get_device_name",
]


def main() -> None:
    """CLI entry point for galgenai."""
    print("GalGenAI - Generative AI models for galaxy images")
    print("Version:", __version__)
