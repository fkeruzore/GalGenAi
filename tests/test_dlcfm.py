"""Tests for DL-CFM: disentanglement losses and trainer."""

import torch
import pytest
from torch.utils.data import DataLoader, TensorDataset

from galgenai.models import VAEEncoder
from galgenai.models.lcfm import LCFM
from galgenai.training import DLCFMTrainer, DLCFMTrainingConfig
from galgenai.training.utils import (
    align_loss,
    decorr_loss,
    disentangled_kl,
    dlcfm_disentanglement_loss,
    extract_dlcfm_batch_data,
)

NX = 64
IN_CH = 5
LATENT_DIM = 32
BASE_CH = 32
N_AUX = 6
BATCH = 8


# ---- fixtures ----


@pytest.fixture
def lcfm_model():
    encoder = VAEEncoder(
        in_channels=IN_CH, latent_dim=LATENT_DIM, input_size=NX
    )
    encoder.eval()
    return LCFM(
        vae_encoder=encoder,
        latent_dim=LATENT_DIM,
        in_channels=IN_CH,
        input_size=NX,
        base_channels=BASE_CH,
        beta=0.0,
    )


@pytest.fixture
def aux_dataset():
    flux = torch.randn(BATCH, IN_CH, NX, NX)
    ivar = torch.ones(BATCH, IN_CH, NX, NX)
    mask = torch.ones(BATCH, IN_CH, NX, NX)
    aux = torch.rand(BATCH, N_AUX)
    return TensorDataset(flux, ivar, mask, aux)


@pytest.fixture
def dlcfm_config(tmp_path):
    return DLCFMTrainingConfig(
        num_steps=10,
        log_every=5,
        save_every=10,
        validate_every=5,
        output_dir=str(tmp_path),
        device="cpu",
        n_aux=N_AUX,
    )


@pytest.fixture
def dlcfm_trainer(lcfm_model, aux_dataset, dlcfm_config):
    loader = DataLoader(aux_dataset, batch_size=4)
    return DLCFMTrainer(
        model=lcfm_model,
        train_loader=loader,
        config=dlcfm_config,
        val_loader=loader,
    )


# ---- align_loss ----


def test_align_loss():
    """Aligned pair -> low loss; random pair -> high loss."""
    torch.manual_seed(42)
    u = torch.linspace(0, 1, 128)
    mu_aligned = 2.0 * u + 1.0
    assert align_loss(u, mu_aligned, K=1).item() < 0.05

    mu_random = torch.randn(128)
    assert align_loss(u, mu_random, K=1).item() > 0.5


# ---- decorr_loss ----


def test_decorr_uncorrelated_low():
    torch.manual_seed(42)
    a = torch.randn(2000, 3)
    b = torch.randn(2000, 4)
    assert decorr_loss(a, b, K=1).item() < 0.1


def test_decorr_correlated_high():
    torch.manual_seed(42)
    x = torch.randn(256, 1)
    a = x.expand(-1, 3)
    b = x.expand(-1, 2) + 0.01 * torch.randn(256, 2)
    assert decorr_loss(a, b, K=1).item() > 0.5


# ---- disentangled_kl ----


def test_disentangled_kl_matches_standard_when_no_aux():
    torch.manual_seed(42)
    mu = torch.randn(64, 16)
    logvar = torch.randn(64, 16)
    aux = torch.zeros(64, 0)

    kl_disent = disentangled_kl(mu, logvar, aux, n_aux=0, tau_sq=1.0)
    kl_std = -0.5 * torch.mean(
        torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    )
    assert torch.allclose(kl_disent, kl_std, atol=1e-5)


def test_disentangled_kl_zero_at_prior():
    torch.manual_seed(42)
    n_aux, d_z, tau_sq = 2, 8, 0.01
    aux = torch.rand(128, n_aux)
    mu = torch.zeros(128, d_z)
    mu[:, :n_aux] = aux
    logvar = torch.zeros(128, d_z)
    logvar[:, :n_aux] = torch.log(torch.tensor(tau_sq))

    assert disentangled_kl(mu, logvar, aux, n_aux, tau_sq).item() < 1e-5


# ---- dlcfm_disentanglement_loss ----


def test_combined_loss_output_format():
    torch.manual_seed(42)
    mu = torch.randn(64, 16)
    logvar = torch.randn(64, 16)
    aux = torch.rand(64, 2)

    total, components = dlcfm_disentanglement_loss(
        mu, logvar, aux, n_aux=2, tau_sq=1.0
    )
    assert total.dim() == 0
    assert set(components.keys()) == {
        "kl",
        "align",
        "intra_decorr",
        "inter_decorr",
        "disentanglement_loss",
    }


@pytest.mark.parametrize(
    "beta,lam1,lam2,n_aux,check",
    [
        # zero lambdas -> only KL contributes
        (1.0, 0.0, 0.0, 2, "kl_only"),
        # zero beta -> KL excluded from total
        (0.0, 1.0, 1.0, 2, "no_kl"),
        # single aux -> intra_decorr must be 0
        (8e-5, 8e-2, 1e-2, 1, "no_intra"),
    ],
    ids=["zero-lambdas", "zero-beta", "single-aux"],
)
def test_combined_loss_lambda_routing(beta, lam1, lam2, n_aux, check):
    torch.manual_seed(42)
    mu = torch.randn(64, 16)
    logvar = torch.randn(64, 16)
    aux = torch.rand(64, n_aux)

    total, comp = dlcfm_disentanglement_loss(
        mu,
        logvar,
        aux,
        n_aux=n_aux,
        beta=beta,
        lambda1=lam1,
        lambda2=lam2,
        tau_sq=1.0,
    )

    if check == "kl_only":
        kl = disentangled_kl(mu, logvar, aux, n_aux=n_aux, tau_sq=1.0)
        assert torch.allclose(total, kl, atol=1e-5)
    elif check == "no_kl":
        reg = (
            lam1 * (comp["align"] + comp["intra_decorr"])
            + lam2 * comp["inter_decorr"]
        )
        assert abs(total.item() - reg) < 1e-4
    elif check == "no_intra":
        assert comp["intra_decorr"] == 0.0


def test_combined_loss_k2_aligned_variables():
    """K>=2 with n_aux>1: perfectly aligned per-variable mapping
    should give a small align component. Regression test for a
    layout bug where polynomial_lift emits degree-first columns
    but the K>=2 branch sliced assuming variable-first layout,
    which made align measure cross-variable same-degree
    correlations instead of same-variable cross-degree."""
    torch.manual_seed(0)
    n_aux = 3
    aux = torch.rand(512, n_aux)
    mu_aux = 2.0 * aux + 0.1
    mu = torch.cat([mu_aux, torch.randn(512, 4)], dim=1)
    logvar = torch.zeros_like(mu)

    _, comp = dlcfm_disentanglement_loss(
        mu,
        logvar,
        aux,
        n_aux=n_aux,
        K=2,
        beta=0.0,
        lambda1=1.0,
        lambda2=0.0,
        tau_sq=1.0,
    )
    # With correct indexing, aligned variables -> align ~ 0.
    # With the old bug, align ~ n_aux (=3).
    assert comp["align"] < 0.2, (
        f"align={comp['align']:.3f} too high; layout bug?"
    )


# ---- extract_dlcfm_batch_data ----


def test_extract_batch_4tuple():
    batch = (
        torch.randn(4, IN_CH, NX, NX),
        torch.ones(4, IN_CH, NX, NX),
        torch.ones(4, IN_CH, NX, NX),
        torch.rand(4, N_AUX),
    )
    data, iv, mk, av = extract_dlcfm_batch_data(batch, torch.device("cpu"))
    assert data.shape == batch[0].shape
    assert av.shape == batch[3].shape


def test_extract_batch_3tuple():
    batch = (
        torch.randn(4, IN_CH, NX, NX),
        torch.ones(4, IN_CH, NX, NX),
        torch.ones(4, IN_CH, NX, NX),
    )
    _, iv, mk, av = extract_dlcfm_batch_data(batch, torch.device("cpu"))
    assert iv is not None
    assert mk is not None
    assert av is None


def test_extract_batch_tensor():
    flux = torch.randn(4, IN_CH, NX, NX)
    _, iv, mk, av = extract_dlcfm_batch_data(flux, torch.device("cpu"))
    assert iv is None
    assert mk is None
    assert av is None


# ---- DLCFMTrainingConfig ----


def test_dlcfm_config_defaults():
    cfg = DLCFMTrainingConfig()
    assert cfg.dlcfm_beta == 8e-5
    assert cfg.lambda1 == 8e-2
    assert cfg.lambda2 == 1e-2
    assert cfg.K == 2
    assert cfg.tau_sq == 1.0
    assert cfg.n_aux == 6


# ---- DLCFMTrainer ----


def test_train_step_keys_and_gradient_flow(dlcfm_trainer):
    batch = next(iter(dlcfm_trainer.train_loader))
    loss_dict = dlcfm_trainer._train_step(batch)

    assert set(loss_dict.keys()) == {
        "flow_loss",
        "kl",
        "align",
        "intra_decorr",
        "inter_decorr",
        "disentanglement_loss",
        "total_loss",
        "lr",
    }

    lcfm = dlcfm_trainer.model
    assert lcfm.vae_encoder.fc_mu.weight.grad is not None
    assert lcfm.vae_encoder.fc_logvar.weight.grad is not None
    assert lcfm.vae_encoder.initial_conv.weight.grad is None
    assert lcfm.velocity_net.conv_in.weight.grad is not None


def test_validate(lcfm_model, aux_dataset, dlcfm_config):
    loader = DataLoader(aux_dataset, batch_size=4)

    # With val_loader -> returns expected keys, restores train mode
    trainer = DLCFMTrainer(
        model=lcfm_model,
        train_loader=loader,
        config=dlcfm_config,
        val_loader=loader,
    )
    trainer.model.train()
    val_metrics = trainer.validate()
    assert set(val_metrics.keys()) == {
        "val_flow_loss",
        "val_kl",
        "val_align",
        "val_intra_decorr",
        "val_inter_decorr",
        "val_disentanglement_loss",
        "val_total_loss",
    }
    assert trainer.model.training

    # Without val_loader -> empty dict
    trainer_no_val = DLCFMTrainer(
        model=lcfm_model,
        train_loader=loader,
        config=dlcfm_config,
    )
    assert trainer_no_val.validate() == {}
