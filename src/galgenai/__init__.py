"""GalGenAI - Generative AI models for galaxy images."""

from . import models
from . import training
from . import data
from .utils import get_device, get_device_name

__version__ = "0.1.0"

__all__ = [
    "models",
    "training",
    "data",
    "get_device",
    "get_device_name",
    "config",
]
