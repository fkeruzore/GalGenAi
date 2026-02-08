"""Tests for LCFM integration"""

import torch
import pytest
from torch.utils.data import DataLoader, TensorDataset
from galgenai.models import VAEEncoder
from galgenai.models.lcfm import (
    LCFM,
    VelocityUNet,
    AttentionBlock,
    count_parameters,
)
from galgenai.training import LCFMTrainer, LCFMTrainingConfig


def test_lcfm_imports():
    """Test that LCFM classes can be imported"""
    from galgenai.models.lcfm import LCFM, VelocityUNet

    assert LCFM is not None
    assert VelocityUNet is not None


def test_lcfm_initialization():
    """Test LCFM model initialization"""
    encoder = VAEEncoder(in_channels=5, latent_dim=32, input_size=64)
    encoder.eval()  # Set to eval mode to avoid batch norm issues
    lcfm = LCFM(
        vae_encoder=encoder,
        latent_dim=32,
        in_channels=5,
        input_size=64,
        base_channels=64,
    )
    assert isinstance(lcfm, LCFM)

    # Verify encoder backbone is frozen, fc_mu/fc_logvar trainable
    for name, param in lcfm.vae_encoder.named_parameters():
        if "fc_mu" in name or "fc_logvar" in name:
            assert param.requires_grad, f"{name} should be trainable"
        else:
            assert not param.requires_grad, f"{name} should be frozen"


def test_lcfm_forward_pass():
    """Test forward pass and loss computation"""
    encoder = VAEEncoder(in_channels=5, latent_dim=32, input_size=64)
    encoder.eval()  # Set to eval mode to avoid batch norm issues
    lcfm = LCFM(encoder, latent_dim=32, in_channels=5, input_size=64)

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
    lcfm = LCFM(encoder, latent_dim=32, in_channels=5, input_size=64)

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
    unet = VelocityUNet(
        in_channels=5, latent_dim=32, input_size=64, base_channels=64
    )

    x = torch.randn(4, 5, 64, 64)
    f = torch.randn(4, 32)
    t = torch.rand(4)

    v = unet(x, f, t)

    assert v.shape == (4, 5, 64, 64)


def test_parameter_counting():
    """Test parameter counting utility"""
    encoder = VAEEncoder(in_channels=5, latent_dim=32, input_size=64)
    encoder.eval()  # Set to eval mode to avoid batch norm issues
    lcfm = LCFM(encoder, latent_dim=32, in_channels=5, input_size=64)

    num_params = count_parameters(lcfm)

    assert num_params > 0
    print(f"LCFM trainable parameters: {num_params:,}")


@pytest.mark.parametrize("input_size", [32, 64, 128])
def test_lcfm_multi_resolution(input_size):
    """Test LCFM with different input sizes"""
    encoder = VAEEncoder(in_channels=3, latent_dim=16, input_size=input_size)
    encoder.eval()

    lcfm = LCFM(
        vae_encoder=encoder,
        latent_dim=16,
        in_channels=3,
        input_size=input_size,
        base_channels=32,  # Smaller for faster tests
    )

    # Test forward pass
    x = torch.randn(2, 3, input_size, input_size)
    loss = lcfm.compute_loss(x)
    assert loss.item() > 0

    # Test sampling
    samples = lcfm.sample(x, num_steps=5)
    assert samples.shape == (2, 3, input_size, input_size)


def test_gradient_flow():
    """Test that gradients flow correctly through trainable parts"""
    encoder = VAEEncoder(in_channels=5, latent_dim=32, input_size=64)
    encoder.eval()
    lcfm = LCFM(encoder, latent_dim=32, in_channels=5, input_size=64)

    x = torch.randn(2, 5, 64, 64)
    loss = lcfm.compute_loss(x)
    loss.backward()

    # Check encoder fc_mu/fc_logvar have gradients
    assert lcfm.vae_encoder.fc_mu.weight.grad is not None
    assert lcfm.vae_encoder.fc_logvar.weight.grad is not None

    # Check encoder backbone has no gradients
    assert lcfm.vae_encoder.initial_conv.weight.grad is None

    # Check velocity network has gradients
    assert lcfm.velocity_net.conv_in.weight.grad is not None


def test_encoder_stays_in_eval_mode():
    """Test that encoder stays in eval mode when model is trained"""
    encoder = VAEEncoder(in_channels=5, latent_dim=32, input_size=64)
    lcfm = LCFM(encoder, latent_dim=32, in_channels=5, input_size=64)

    # Encoder should be in eval mode after init
    assert not lcfm.vae_encoder.training, "Encoder should be in eval mode"

    # Set model to train mode
    lcfm.train()

    # Encoder should still be in eval mode
    assert not lcfm.vae_encoder.training, (
        "Encoder should stay in eval mode after train()"
    )

    # Velocity net should be in train mode
    assert lcfm.velocity_net.training, "Velocity net should be in train mode"


def test_invalid_input_size():
    """Test that invalid input_size raises ValueError"""
    # input_size=30 is not divisible by 8 (3 downsampling stages)
    with pytest.raises(ValueError, match="must be divisible by"):
        VelocityUNet(
            in_channels=5,
            latent_dim=32,
            input_size=30,  # Invalid: not divisible by 8
        )

    # input_size=50 is also invalid
    with pytest.raises(ValueError, match="must be divisible by"):
        VelocityUNet(
            in_channels=5,
            latent_dim=32,
            input_size=50,  # Invalid: not divisible by 8
        )


def test_valid_input_sizes():
    """Test that valid input_sizes work"""
    for input_size in [32, 64, 128, 256]:
        unet = VelocityUNet(
            in_channels=5,
            latent_dim=32,
            input_size=input_size,
            base_channels=32,
        )
        assert unet.input_size == input_size


def test_invalid_attention_heads():
    """Test that invalid num_heads raises ValueError"""
    # channels=64 is not divisible by num_heads=5
    with pytest.raises(ValueError, match="must be divisible by num_heads"):
        AttentionBlock(channels=64, num_heads=5)

    # channels=100 is not divisible by num_heads=8
    with pytest.raises(ValueError, match="must be divisible by num_heads"):
        AttentionBlock(channels=100, num_heads=8)


def test_valid_attention_heads():
    """Test that valid num_heads work"""
    # 64 / 4 = 16 (valid)
    attn = AttentionBlock(channels=64, num_heads=4)
    assert attn.num_heads == 4
    assert attn.head_dim == 16

    # 128 / 8 = 16 (valid)
    attn = AttentionBlock(channels=128, num_heads=8)
    assert attn.num_heads == 8
    assert attn.head_dim == 16


def test_encoder_missing_fc_mu():
    """Test that encoder without fc_mu raises ValueError"""

    class BadEncoder(torch.nn.Module):
        """Encoder missing fc_mu layer"""

        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(5, 32, 3, padding=1)
            self.fc_logvar = torch.nn.Linear(32, 32)  # Has fc_logvar
            # Missing fc_mu!

        def forward(self, x):
            h = self.conv(x)
            h = h.mean(dim=(2, 3))
            return h, self.fc_logvar(h)

    encoder = BadEncoder()
    with pytest.raises(ValueError, match="fc_mu"):
        LCFM(encoder, latent_dim=32, in_channels=5, input_size=64)


def test_encoder_missing_fc_logvar():
    """Test that encoder without fc_logvar raises ValueError"""

    class BadEncoder(torch.nn.Module):
        """Encoder missing fc_logvar layer"""

        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(5, 32, 3, padding=1)
            self.fc_mu = torch.nn.Linear(32, 32)  # Has fc_mu
            # Missing fc_logvar!

        def forward(self, x):
            h = self.conv(x)
            h = h.mean(dim=(2, 3))
            return self.fc_mu(h), h

    encoder = BadEncoder()
    with pytest.raises(ValueError, match="fc_logvar"):
        LCFM(encoder, latent_dim=32, in_channels=5, input_size=64)


def test_runtime_input_size_mismatch():
    """Test that mismatched input size at runtime raises ValueError"""
    unet = VelocityUNet(
        in_channels=5, latent_dim=32, input_size=64, base_channels=32
    )

    # Correct size works
    x_correct = torch.randn(2, 5, 64, 64)
    f = torch.randn(2, 32)
    t = torch.rand(2)
    v = unet(x_correct, f, t)
    assert v.shape == (2, 5, 64, 64)

    # Wrong size raises error
    x_wrong = torch.randn(2, 5, 32, 32)
    with pytest.raises(
        ValueError, match="Expected input size 64x64, got 32x32"
    ):
        unet(x_wrong, f, t)

    # Non-square wrong size
    x_nonsquare = torch.randn(2, 5, 64, 32)
    with pytest.raises(
        ValueError, match="Expected input size 64x64, got 64x32"
    ):
        unet(x_nonsquare, f, t)


def test_lcfm_trainer_validate(tmp_path):
    """Test that validate() returns expected keys and restores
    train mode."""
    encoder = VAEEncoder(in_channels=5, latent_dim=32, input_size=64)
    encoder.eval()
    lcfm = LCFM(
        vae_encoder=encoder,
        latent_dim=32,
        in_channels=5,
        input_size=64,
        base_channels=32,
    )

    # Dummy data loaders
    dummy_data = torch.randn(8, 5, 64, 64)
    dummy_dataset = TensorDataset(dummy_data)
    train_loader = DataLoader(dummy_dataset, batch_size=4)
    val_loader = DataLoader(dummy_dataset, batch_size=4)

    config = LCFMTrainingConfig(
        num_steps=10,
        log_every=5,
        save_every=10,
        validate_every=5,
        output_dir=str(tmp_path),
        device="cpu",
    )

    trainer = LCFMTrainer(
        model=lcfm,
        train_loader=train_loader,
        config=config,
        val_loader=val_loader,
    )

    # Put model in train mode before calling validate
    lcfm.train()
    val_metrics = trainer.validate()

    assert "val_flow_loss" in val_metrics
    assert "val_kl_loss" in val_metrics
    assert "val_total_loss" in val_metrics
    assert lcfm.training, "Model should be back in train mode"


def test_lcfm_weighted_loss():
    """Test compute_loss with ivar and mask tensors."""
    encoder = VAEEncoder(in_channels=5, latent_dim=32, input_size=64)
    encoder.eval()
    lcfm = LCFM(
        encoder,
        latent_dim=32,
        in_channels=5,
        input_size=64,
        base_channels=32,
    )

    x = torch.randn(2, 5, 64, 64)
    ivar = torch.ones(2, 5, 64, 64)
    mask = torch.ones(2, 5, 64, 64)

    loss, loss_dict = lcfm.compute_loss(
        x, ivar=ivar, mask=mask, return_components=True
    )

    assert loss.item() > 0
    assert "flow_loss" in loss_dict
    assert "kl_loss" in loss_dict
    assert "total_loss" in loss_dict


def test_lcfm_mask_affects_loss():
    """Test that partial masking changes the loss value."""
    torch.manual_seed(42)
    encoder = VAEEncoder(in_channels=5, latent_dim=32, input_size=64)
    encoder.eval()
    lcfm = LCFM(
        encoder,
        latent_dim=32,
        in_channels=5,
        input_size=64,
        base_channels=32,
    )

    x = torch.randn(2, 5, 64, 64)
    ivar = torch.ones(2, 5, 64, 64)

    # Full mask
    mask_full = torch.ones(2, 5, 64, 64)
    # Partial mask: zero out half the pixels
    mask_partial = torch.ones(2, 5, 64, 64)
    mask_partial[:, :, :32, :] = 0.0

    torch.manual_seed(0)
    _, dict_full = lcfm.compute_loss(
        x, ivar=ivar, mask=mask_full, return_components=True
    )
    torch.manual_seed(0)
    _, dict_partial = lcfm.compute_loss(
        x, ivar=ivar, mask=mask_partial, return_components=True
    )

    # Flow losses should differ (different masked regions)
    assert dict_full["flow_loss"] != pytest.approx(
        dict_partial["flow_loss"], abs=1e-6
    )


def test_lcfm_trainer_validate_with_weights(tmp_path):
    """Test validate() with (flux, ivar, mask) dataset tuples."""
    encoder = VAEEncoder(in_channels=5, latent_dim=32, input_size=64)
    encoder.eval()
    lcfm = LCFM(
        vae_encoder=encoder,
        latent_dim=32,
        in_channels=5,
        input_size=64,
        base_channels=32,
    )

    flux = torch.randn(8, 5, 64, 64)
    ivar = torch.ones(8, 5, 64, 64)
    mask = torch.ones(8, 5, 64, 64)
    dataset = TensorDataset(flux, ivar, mask)
    train_loader = DataLoader(dataset, batch_size=4)
    val_loader = DataLoader(dataset, batch_size=4)

    config = LCFMTrainingConfig(
        num_steps=10,
        log_every=5,
        save_every=10,
        validate_every=5,
        output_dir=str(tmp_path),
        device="cpu",
    )

    trainer = LCFMTrainer(
        model=lcfm,
        train_loader=train_loader,
        config=config,
        val_loader=val_loader,
    )

    lcfm.train()
    val_metrics = trainer.validate()

    assert "val_flow_loss" in val_metrics
    assert "val_kl_loss" in val_metrics
    assert "val_total_loss" in val_metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
