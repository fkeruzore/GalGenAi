"""Device selection utilities with priority: cuda > mps > cpu."""

import torch


def get_device() -> torch.device:
    """
    Get the best available device with priority order: cuda > mps > cpu.

    Returns:
        torch.device: The selected device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_device_name() -> str:
    """
    Get the name of the best available device.

    Returns:
        str: Name of the selected device ('cuda', 'mps', or 'cpu').
    """
    device = get_device()
    return device.type
