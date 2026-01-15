"""Tests for VAE model and building blocks."""

import torch
import pytest
from galgenai.models import (
    VAE,
    VAEEncoder,
    VAEDecoder,
    ResidualBlock,
    DownsampleBlock,
    UpsampleBlock,
    SqueezeExcitationBlock,
)


# --- Fixtures ---


@pytest.fixture
def vae_32():
    """VAE model for 32x32 grayscale images."""
    model = VAE(in_channels=1, latent_dim=16, input_size=32)
    model.eval()
    return model


@pytest.fixture
def vae_64():
    """VAE model for 64x64 grayscale images."""
    model = VAE(in_channels=1, latent_dim=16, input_size=64)
    model.eval()
    return model


@pytest.fixture
def vae_128():
    """VAE model for 128x128 grayscale images."""
    model = VAE(in_channels=1, latent_dim=16, input_size=128)
    model.eval()
    return model


@pytest.fixture
def vae_multichannel():
    """VAE model for 5-channel 64x64 galaxy images."""
    model = VAE(in_channels=5, latent_dim=32, input_size=64)
    model.eval()
    return model


# --- Layer Tests ---


def test_squeeze_excitation_block():
    """Test SE block preserves shape and produces valid output."""
    se = SqueezeExcitationBlock(channels=64, reduction=8)
    x = torch.randn(4, 64, 16, 16)

    out = se(x)

    assert out.shape == x.shape, "SE block should preserve input shape"
    # SE block applies channel attention via sigmoid
    assert not torch.isnan(out).any(), "Output should not contain NaN"


def test_residual_block_with_se():
    """Test ResidualBlock with SE attention preserves shape."""
    block = ResidualBlock(channels=64, use_se=True, se_reduction=8)
    block.eval()
    x = torch.randn(4, 64, 16, 16)

    with torch.no_grad():
        out = block(x)

    assert out.shape == x.shape, "ResidualBlock should preserve input shape"
    assert block.se is not None, "SE block should be present when use_se=True"


def test_residual_block_without_se():
    """Test ResidualBlock without SE attention works correctly."""
    block = ResidualBlock(channels=64, use_se=False)
    block.eval()
    x = torch.randn(4, 64, 16, 16)

    with torch.no_grad():
        out = block(x)

    assert out.shape == x.shape, "ResidualBlock should preserve input shape"
    assert block.se is None, "SE block should be None when use_se=False"


def test_downsample_block():
    """Test DownsampleBlock halves spatial dims and changes channels."""
    block = DownsampleBlock(in_channels=32, out_channels=64)
    block.eval()
    x = torch.randn(4, 32, 16, 16)

    with torch.no_grad():
        out = block(x)

    assert out.shape == (4, 64, 8, 8), (
        f"Expected (4, 64, 8, 8), got {out.shape}"
    )


def test_upsample_block():
    """Test UpsampleBlock doubles spatial dims and changes channels."""
    block = UpsampleBlock(in_channels=64, out_channels=32)
    block.eval()
    x = torch.randn(4, 64, 8, 8)

    with torch.no_grad():
        out = block(x)

    assert out.shape == (4, 32, 16, 16), (
        f"Expected (4, 32, 16, 16), got {out.shape}"
    )


# --- VAE Encoder Tests ---

# Channel depths for 32x32 input (5 stages = 4 downsamples)
CHANNEL_DEPTHS_32 = [16, 32, 64, 128, 256]


def test_encoder_output_shapes():
    """Test encoder produces mu and logvar with correct shapes."""
    encoder = VAEEncoder(
        in_channels=1,
        latent_dim=16,
        input_size=32,
        channel_depths=CHANNEL_DEPTHS_32,
    )
    encoder.eval()
    x = torch.randn(4, 1, 32, 32)

    with torch.no_grad():
        mu, logvar = encoder(x)

    assert mu.shape == (4, 16), f"Expected mu shape (4, 16), got {mu.shape}"
    assert logvar.shape == (4, 16), (
        f"Expected logvar shape (4, 16), got {logvar.shape}"
    )


def test_encoder_logvar_clamping():
    """Test encoder clamps logvar to specified bounds."""
    clamp_min, clamp_max = -30.0, 20.0
    encoder = VAEEncoder(
        in_channels=1,
        latent_dim=16,
        input_size=32,
        channel_depths=CHANNEL_DEPTHS_32,
        logvar_clamp=(clamp_min, clamp_max),
    )
    encoder.eval()

    # Use extreme input to potentially trigger extreme logvar values
    x = torch.randn(4, 1, 32, 32) * 100

    with torch.no_grad():
        _, logvar = encoder(x)

    assert logvar.min() >= clamp_min, (
        f"logvar min {logvar.min()} below clamp_min {clamp_min}"
    )
    assert logvar.max() <= clamp_max, (
        f"logvar max {logvar.max()} above clamp_max {clamp_max}"
    )


def test_encoder_forward_conv():
    """Test encoder conv layers produce expected feature map shape."""
    encoder = VAEEncoder(
        in_channels=1,
        latent_dim=16,
        input_size=32,
        channel_depths=CHANNEL_DEPTHS_32,
    )
    encoder.eval()
    x = torch.randn(4, 1, 32, 32)

    with torch.no_grad():
        features = encoder.forward_conv(x)

    # 32x32 with 4 downsamples -> 32/16 = 2x2 spatial, 256 channels
    assert features.shape == (4, 256, 2, 2), (
        f"Expected (4, 256, 2, 2), got {features.shape}"
    )


# --- VAE Decoder Tests ---

# Channel depths for 32x32 decoder (reversed from encoder)
DECODER_CHANNEL_DEPTHS_32 = [256, 128, 64, 32, 16]


def test_decoder_output_shape():
    """Test decoder produces output matching input_size."""
    decoder = VAEDecoder(
        latent_dim=16,
        out_channels=1,
        input_size=32,
        channel_depths=DECODER_CHANNEL_DEPTHS_32,
    )
    decoder.eval()
    z = torch.randn(4, 16)

    with torch.no_grad():
        out = decoder(z)

    assert out.shape == (4, 1, 32, 32), (
        f"Expected (4, 1, 32, 32), got {out.shape}"
    )


def test_decoder_softplus_activation():
    """Test decoder output is non-negative (softplus activation)."""
    decoder = VAEDecoder(
        latent_dim=16,
        out_channels=1,
        input_size=32,
        channel_depths=DECODER_CHANNEL_DEPTHS_32,
    )
    decoder.eval()
    z = torch.randn(4, 16)

    with torch.no_grad():
        out = decoder(z)

    assert out.min() >= 0, (
        f"Decoder output min {out.min()} should be >= 0 (softplus)"
    )


# --- VAE Complete Model Tests ---


def test_vae_forward_pass_32x32(vae_32):
    """Test VAE forward pass with 32x32 input."""
    x = torch.randn(4, 1, 32, 32)

    with torch.no_grad():
        recon, mu, logvar = vae_32(x)

    assert recon.shape == x.shape, (
        f"Reconstruction shape {recon.shape} should match input {x.shape}"
    )
    assert mu.shape == (4, 16), f"Expected mu shape (4, 16), got {mu.shape}"
    assert logvar.shape == (4, 16), (
        f"Expected logvar shape (4, 16), got {logvar.shape}"
    )


def test_vae_forward_pass_64x64(vae_64):
    """Test VAE forward pass with 64x64 input."""
    x = torch.randn(4, 1, 64, 64)

    with torch.no_grad():
        recon, mu, logvar = vae_64(x)

    assert recon.shape == x.shape, (
        f"Reconstruction shape {recon.shape} should match input {x.shape}"
    )
    assert mu.shape == (4, 16), f"Expected mu shape (4, 16), got {mu.shape}"


def test_vae_forward_pass_128x128(vae_128):
    """Test VAE forward pass with 128x128 input."""
    x = torch.randn(4, 1, 128, 128)

    with torch.no_grad():
        recon, mu, logvar = vae_128(x)

    assert recon.shape == x.shape, (
        f"Reconstruction shape {recon.shape} should match input {x.shape}"
    )
    assert mu.shape == (4, 16), f"Expected mu shape (4, 16), got {mu.shape}"


def test_vae_reparameterize(vae_32):
    """Test reparameterization trick produces correct shape."""
    mu = torch.zeros(4, 16)
    logvar = torch.zeros(4, 16)

    z = vae_32.reparameterize(mu, logvar)

    assert z.shape == (4, 16), f"Expected z shape (4, 16), got {z.shape}"
    # With mu=0 and logvar=0 (std=1), z should be ~ N(0, 1)
    # Just check it's not identical to mu (stochastic sampling)
    assert not torch.allclose(z, mu), (
        "z should differ from mu due to reparameterization"
    )


def test_vae_generate(vae_32):
    """Test generation from prior distribution."""
    with torch.no_grad():
        samples = vae_32.generate(num_samples=8, device=torch.device("cpu"))

    assert samples.shape == (8, 1, 32, 32), (
        f"Expected (8, 1, 32, 32), got {samples.shape}"
    )
    assert samples.min() >= 0, "Generated samples should be non-negative"


def test_vae_multichannel(vae_multichannel):
    """Test VAE with 5-channel input (galaxy images)."""
    x = torch.randn(4, 5, 64, 64)

    with torch.no_grad():
        recon, mu, logvar = vae_multichannel(x)

    assert recon.shape == x.shape, (
        f"Reconstruction shape {recon.shape} should match input {x.shape}"
    )
    assert mu.shape == (4, 32), f"Expected mu shape (4, 32), got {mu.shape}"


# --- Edge Case Tests ---


def test_vae_batch_size_one(vae_32):
    """Test VAE works with batch_size=1 in eval mode."""
    x = torch.randn(1, 1, 32, 32)

    with torch.no_grad():
        recon, mu, logvar = vae_32(x)

    assert recon.shape == (1, 1, 32, 32), (
        f"Expected (1, 1, 32, 32), got {recon.shape}"
    )


def test_encoder_invalid_input_size():
    """Test encoder raises error for too small input size."""
    with pytest.raises(ValueError, match="too small"):
        # 8x8 with 7 channel depths means 6 downsamples: 8 / 64 < 1
        VAEEncoder(
            in_channels=1,
            latent_dim=16,
            input_size=8,
            channel_depths=[16, 32, 64, 128, 256, 512, 512],
        )


# --- Parameter Count Tests ---


def test_parameter_count_32x32():
    """Test exact parameter counts for 32x32 model."""
    model = VAE(in_channels=1, latent_dim=16, input_size=32)

    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    decoder_params = sum(p.numel() for p in model.decoder.parameters())
    total_params = sum(p.numel() for p in model.parameters())

    assert encoder_params == 3608920, (
        f"Encoder params: expected 3608920, got {encoder_params}"
    )
    assert decoder_params == 3897913, (
        f"Decoder params: expected 3897913, got {decoder_params}"
    )
    assert total_params == 7506833, (
        f"Total params: expected 7506833, got {total_params}"
    )


def test_parameter_count_128x128():
    """Test exact parameter counts for 128x128 model."""
    model = VAE(in_channels=1, latent_dim=16, input_size=128)

    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    decoder_params = sum(p.numel() for p in model.decoder.parameters())
    total_params = sum(p.numel() for p in model.parameters())

    assert encoder_params == 26332760, (
        f"Encoder params: expected 26332760, got {encoder_params}"
    )
    assert decoder_params == 29358649, (
        f"Decoder params: expected 29358649, got {decoder_params}"
    )
    assert total_params == 55691409, (
        f"Total params: expected 55691409, got {total_params}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
