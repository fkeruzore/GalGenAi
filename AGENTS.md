# Repository Guidelines

## Project Overview & Research Basis

GalGenAI is a Python library for training generative models on galaxy
images. It currently supports a ResNet-based VAE for reconstruction and
Latent Conditional Flow Matching (LCFM) for higher-quality conditional
generation. The LCFM implementation follows *Samaddar et al. (2025),
“Efficient Flow Matching using Latent Variables”* by freezing the encoder
backbone while finetuning the latent heads (`fc_mu`, `fc_logvar`) and
training a U‑Net vector field conditioned on encoder latents. The VAE
decoder uses `Softplus` outputs to keep flux values non‑negative; masked
weighted MSE supports inverse-variance weighting for astronomical data.
Key architecture notes:
- **VAE**: ResNet-style down/upsampling with residual blocks; encoder
  outputs `mu`/`logvar`, decoder mirrors the encoder and applies
  `Softplus`.
- **LCFM**: U‑Net vector field with time conditioning via AdaGN; latent
  features are linearly projected and added to the final velocity
  output. LCFM trains with straight-line flow matching loss plus KL.

## Project Structure & Module Organization

`src/galgenai/` contains the library code. Core areas:
- `src/galgenai/models/` for model architectures (`vae.py`, `lcfm.py`, `layers.py`).
- `src/galgenai/training/` for trainers and configs (`base_trainer.py`, `vae_trainer.py`, `lcfm_trainer.py`, `config.py`).
- `src/galgenai/data/` for datasets and loaders.
- `src/galgenai/utils/` for utilities like device selection.

Supporting paths:
- `tests/` for pytest-based tests and verification scripts.
- `scripts/` for runnable workflows (e.g., `train_vae_lcfm_pipeline.py`).
- `notebooks/` for exploratory work and rendered artifacts.
- `data/` and `pipeline_output/` for datasets and generated outputs.

## Build, Test, and Development Commands

This repo uses `uv` for environment management.
- `uv sync` to install dependencies.
- `uv run python scripts/train_vae_lcfm_pipeline.py` to run the training pipeline.
- `uv run pytest` to run the test suite.
- `uv run ruff format` to format code.
- `uv run ruff check` to lint code.

## Coding Style & Naming Conventions

Python 3.12+ is required (`pyproject.toml`). Use Ruff for formatting and linting.
- Line length: 79.
- Lint rules: `E`, `F`, `B`, `W`.
- Keep module names `snake_case` and classes in `PascalCase`.
- Prefer explicit, descriptive names for training configs and model components (e.g., `VAETrainingConfig`).

## Testing Guidelines

Tests live in `tests/` and follow `test_*.py` naming. Use pytest:
- `uv run pytest` to run all tests.
- `uv run pytest tests/test_vae.py` to target a specific module.

Keep tests fast and focused on model construction, shapes, and training
utilities. If you add new modules under `src/galgenai/`, add
corresponding tests under `tests/`. LCFM tests typically validate input
size handling, encoder freezing, and gradient flow.

## Commit & Pull Request Guidelines

Recent commits use short, lowercase, imperative messages (e.g., “add rendered lcfm pipeline notebook”). Keep commits small and focused.

For PRs:
- Provide a concise summary of changes and rationale.
- Link related issues if applicable.
- Include command output or screenshots when changes affect notebooks or generated outputs.

## Configuration & Data Tips

Use `get_device()` for device selection when running training. Large artifacts (models, outputs) should live under `pipeline_output/` or `data/` rather than `src/`.
