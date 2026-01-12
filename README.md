# GalGenAI

Generative AI models for galaxy images.

## Overview

GalGenAI is a Python library for building and training generative models for galaxy images. The library provides modular implementations of deep learning architectures optimized for image generation tasks.

## Features

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```sh
# Clone the repository
git clone git@github.com:fkeruzore/GalGenAi.git
cd galgenai

# Install dependencies
uv sync

# Activate the environment (if needed)
source .venv/bin/activate
```

## Quick Start

### Using the Library

```py
import torch
from galgenai.models import VAE
from galgenai.utils import get_device

# Initialize model
device = get_device()
model = VAE(in_channels=1, latent_dim=16, input_size=32).to(device)

# Forward pass
x = torch.randn(4, 1, 32, 32).to(device)
reconstruction, mu, logvar = model(x)

# Generate samples
samples = model.generate(num_samples=16, device=device)
```

## Training Custom Models

```py
from galgenai.models import VAE
from galgenai.training import VAETrainer, VAETrainingConfig
from torch.utils.data import DataLoader

# Prepare your dataloader
train_loader = DataLoader(your_dataset, batch_size=128, shuffle=True)

# Create model and config
model = VAE(in_channels=1, latent_dim=16, input_size=32)
config = VAETrainingConfig(
    num_epochs=10,
    learning_rate=1e-3,
    reconstruction_loss_fn="mse",
    beta=1.0,  # Beta-VAE parameter
    output_dir="./vae_output",
)

# Train
trainer = VAETrainer(model=model, train_loader=train_loader, config=config)
trainer.train()
```
