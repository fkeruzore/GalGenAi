"""Tests for LCFM integration"""

import torch
import pytest
from galgenai import VAEEncoder
from galgenai.models.lcfm import (
    LCFM,
    VelocityUNet,
    count_parameters,
)


def test_lcfm_imports():
    """Test that LCFM classes can be imported"""
    from galgenai import LCFM, VelocityUNet, LatentStochasticLayer

    assert LCFM is not None
    assert VelocityUNet is not None
    assert LatentStochasticLayer is not None


def test_lcfm_initialization():
    """Test LCFM model initialization"""
    encoder = VAEEncoder(in_channels=5, latent_dim=32, input_size=64)
    encoder.eval()  # Set to eval mode to avoid batch norm issues
    lcfm = LCFM(
        vae_encoder=encoder, latent_dim=32, in_channels=5, base_channels=64
    )
    assert isinstance(lcfm, LCFM)

    # Verify encoder is frozen
    for param in lcfm.vae_encoder.parameters():
        assert not param.requires_grad, "Encoder should be frozen"


def test_lcfm_forward_pass():
    """Test forward pass and loss computation"""
    encoder = VAEEncoder(in_channels=5, latent_dim=32, input_size=64)
    encoder.eval()  # Set to eval mode to avoid batch norm issues
    lcfm = LCFM(encoder, latent_dim=32, in_channels=5)

    # Create dummy batch
    x = torch.randn(4, 5, 64, 64)

    # Compute loss
    loss, loss_dict = lcfm.compute_loss(x, return_components=True)

    assert loss.item() > 0
    assert "flow_loss" in loss_dict
    assert "kl_loss" in loss_dict
    assert "total_loss" in loss_dict


def test_lcfm_sampling():
    """Test sample generation"""
    encoder = VAEEncoder(in_channels=5, latent_dim=32, input_size=64)
    encoder.eval()  # Set to eval mode to avoid batch norm issues
    lcfm = LCFM(encoder, latent_dim=32, in_channels=5)

    # Training images to condition on
    x_train = torch.randn(4, 5, 64, 64)

    # Generate samples
    samples = lcfm.sample(x_train, num_steps=10)

    assert samples.shape == (4, 5, 64, 64)


def test_encoder_wrapper():
    """Test that encoder returns correct format"""
    encoder = VAEEncoder(in_channels=5, latent_dim=32, input_size=64)
    encoder.eval()  # Set to eval mode to avoid batch norm issues
    x = torch.randn(2, 5, 64, 64)

    mu, logvar = encoder(x)

    assert mu.shape == (2, 32)
    assert logvar.shape == (2, 32)


def test_velocity_unet():
    """Test VelocityUNet independently"""
    unet = VelocityUNet(in_channels=5, latent_dim=32, base_channels=64)

    x = torch.randn(4, 5, 64, 64)
    f = torch.randn(4, 32)
    t = torch.rand(4)

    v = unet(x, f, t)

    assert v.shape == (4, 5, 64, 64)


def test_parameter_counting():
    """Test parameter counting utility"""
    encoder = VAEEncoder(in_channels=5, latent_dim=32, input_size=64)
    encoder.eval()  # Set to eval mode to avoid batch norm issues
    lcfm = LCFM(encoder, latent_dim=32, in_channels=5)

    num_params = count_parameters(lcfm)

    assert num_params > 0
    print(f"LCFM trainable parameters: {num_params:,}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
