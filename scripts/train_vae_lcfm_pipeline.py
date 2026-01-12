"""
VAE + LCFM Training Pipeline

Demonstrates the complete workflow for training generative
models on galaxy images:
1. Train a VAE on HSC galaxy data
2. Extract and save the frozen encoder
3. Train an LCFM model using the encoder

Run with:
    uv run python scripts/train_vae_lcfm_pipeline.py
"""

import torch
from pathlib import Path
from datasets import load_from_disk

from galgenai.models import VAE, VAEEncoder, LCFM

from galgenai.training import (
    VAETrainer,
    VAETrainingConfig,
    LCFMTrainer,
    LCFMTrainingConfig,
)

from galgenai.data.hsc import get_dataset_and_loaders

from galgenai import get_device


# Get the best available device
device = get_device()
print(f"Using device: {device}")

# Create output directory structure
output_dir = Path("./pipeline_output")
output_dir.mkdir(exist_ok=True)
print(f"Output directory: {output_dir.absolute()}")


# =================================================================
# ----  DATA LOADING  --------------------------------------------
# =================================================================

print("\n" + "=" * 60)
print("LOADING DATA")
print("=" * 60)

# Path to the HSC mini dataset (HuggingFace format)
data_path = "./data/hsc_mmu_mini/"

# Load the raw HuggingFace dataset
dataset_raw = load_from_disk(data_path)
print(f"Loaded dataset from: {data_path}")

# Create PyTorch dataset and data loaders
# - nx=64: crop images to 64x64 pixels
# - batch_size=32: small batch for demo
# - Returns (flux, ivar, mask) tuples per batch
dataset, train_loader, val_loader = get_dataset_and_loaders(
    dataset_raw,
    nx=64,  # Crop size (64x64 pixels)
    batch_size=32,  # Batch size for training
)
print(f"Dataset size: {len(dataset)} samples")
print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")


# =================================================================
# ----  VAE TRAINING  ---------------------------------------------
# =================================================================

print("\n" + "=" * 60)
print("TRAINING VAE")
print("=" * 60)

# Create VAE model
# - in_channels=5: HSC has 5 photometric bands (g, r, i, z, y)
# - latent_dim=32: dimension of the latent space
# - input_size=64: matches our crop size
vae = VAE(
    in_channels=5,
    latent_dim=32,
    input_size=64,
)
print(f"VAE parameters: {sum(p.numel() for p in vae.parameters()):,}")

# Configure VAE training
# - num_epochs=5: quick demo training
# - reconstruction_loss_fn="masked_weighted_mse": uses inverse
#   variance weights and masks for bad pixels/noise
# - beta=1.0: standard VAE
vae_config = VAETrainingConfig(
    num_epochs=5,
    learning_rate=1e-3,
    reconstruction_loss_fn="masked_weighted_mse",
    beta=1.0,
    output_dir=str(output_dir / "vae"),
    save_every=5,  # Save checkpoint every 5 epochs
    device=str(device),
)

# Create trainer and run training
vae_trainer = VAETrainer(
    model=vae,
    train_loader=train_loader,
    config=vae_config,
    val_loader=val_loader,
)
vae_trainer.train()

print("VAE training complete!")
print(f"Checkpoints saved to: {output_dir / 'vae' / 'checkpoints'}")


# =================================================================
# ----  EXTRACT & SAVE ENCODER  -----------------------------------
# =================================================================

print("\n" + "=" * 60)
print("SAVING FROZEN ENCODER")
print("=" * 60)

# The VAE has separate encoder and decoder components
# We save just the encoder for use with LCFM
encoder_path = output_dir / "encoder.pt"
torch.save(vae.encoder.state_dict(), encoder_path)
print(f"Encoder saved to: {encoder_path}")

# Clean up VAE to free memory before LCFM training
del vae, vae_trainer
if torch.cuda.is_available():
    torch.cuda.empty_cache()


# =================================================================
# ----  LCFM TRAINING  --------------------------------------------
# =================================================================

print("\n" + "=" * 60)
print("TRAINING LCFM")
print("=" * 60)

# Create a fresh encoder and load the trained weights
# Must match the VAE encoder architecture
encoder = VAEEncoder(
    in_channels=5,
    latent_dim=32,
    input_size=64,
)
encoder.load_state_dict(torch.load(encoder_path, map_location=device))
encoder.to(device).eval()
print(f"Loaded encoder from: {encoder_path}")

# Create LCFM model
# - Encoder is automatically frozen (requires_grad=False)
# - base_channels=64: U-Net base channel count
# - beta=0.001: KL weight (low for scientific data)
lcfm = LCFM(
    vae_encoder=encoder,
    latent_dim=32,
    in_channels=5,
    base_channels=64,
    beta=0.001,
).to(device)
print(f"LCFM parameters: {sum(p.numel() for p in lcfm.parameters()):,}")
print("  (encoder frozen, only flow network is trained)")

# Configure LCFM training
# - num_steps=5000: step-based training (not epochs)
# - learning_rate=2e-4: standard for flow matching
# - warmup_steps=500: linear LR warmup
# - sample_every=2500: generate samples periodically
lcfm_config = LCFMTrainingConfig(
    latent_dim=32,
    in_channels=5,
    base_channels=64,
    num_steps=5000,
    batch_size=32,
    learning_rate=2e-4,
    warmup_steps=500,
    beta=0.001,
    sample_every=2500,  # Generate samples every 2500 steps
    save_every=5000,  # Save checkpoint at the end
    log_every=100,  # Log every 100 steps
    output_dir=str(output_dir / "lcfm"),
    device=str(device),
)

# Create trainer and run training
lcfm_trainer = LCFMTrainer(
    model=lcfm,
    train_loader=train_loader,
    config=lcfm_config,
    val_loader=val_loader,
)
lcfm_trainer.train()

print("LCFM training complete!")
print(f"Checkpoints saved to: {output_dir / 'lcfm' / 'checkpoints'}")


# =================================================================
# ----  SUMMARY  --------------------------------------------------
# =================================================================

print("\n" + "=" * 60)
print("TRAINING PIPELINE COMPLETE")
print("=" * 60)
vae_ckpt = output_dir / "vae" / "checkpoints"
lcfm_ckpt = output_dir / "lcfm" / "checkpoints"
best_model = lcfm_ckpt / "best.pt"

print(f"""
Output files:
  - VAE checkpoints:  {vae_ckpt}
  - Frozen encoder:   {encoder_path}
  - LCFM checkpoints: {lcfm_ckpt}
  - LCFM samples:     {output_dir / "lcfm" / "samples"}

To load the trained LCFM model:
    encoder = VAEEncoder(in_channels=5, latent_dim=32,
                         input_size=64)
    encoder.load_state_dict(torch.load("{encoder_path}"))
    lcfm = LCFM(encoder, latent_dim=32, in_channels=5)
    lcfm.load_state_dict(
        torch.load("{best_model}")["model_state_dict"]
    )
""")
