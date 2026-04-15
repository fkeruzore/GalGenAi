"""Shared training utilities."""

from typing import Optional, Tuple

import torch
import torch.nn as nn


def extract_batch_data(
    batch, device: torch.device
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Extract data from batch, handling both tuple and tensor formats.

    Args:
        batch: Either a tensor or tuple of (flux, ivar, mask).
        device: Device to move tensors to.

    Returns:
        Tuple of (data, ivar, mask). ivar and mask may be None.
    """
    if isinstance(batch, (tuple, list)):
        data = batch[0].to(device)
        ivar = batch[1].to(device) if len(batch) > 1 else None
        mask = batch[2].to(device) if len(batch) > 2 else None
    else:
        data = batch.to(device)
        ivar = None
        mask = None
    return data, ivar, mask


def extract_dlcfm_batch_data(
    batch, device: torch.device
) -> Tuple[
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    """
    Extract data from batch for DL-CFM training.

    Delegates to :func:`extract_batch_data` for the first three
    elements and adds the optional fourth (aux_vars).

    Args:
        batch: Either a tensor, tuple of (flux, ivar, mask), or
            tuple of (flux, ivar, mask, aux_vars).
        device: Device to move tensors to.

    Returns:
        Tuple of (data, ivar, mask, aux_vars). ivar, mask, and
        aux_vars may be None.
    """
    data, ivar, mask = extract_batch_data(batch, device)
    aux_vars = None
    if isinstance(batch, (tuple, list)) and len(batch) > 3:
        aux_vars = batch[3].to(device)
    return data, ivar, mask, aux_vars


def vae_loss(
    reconstruction: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    reconstruction_loss_fn: str = "mse",
    beta: float = 1.0,
    ivar: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculate VAE loss = reconstruction_loss + beta * KL_divergence.

    Args:
        reconstruction: Reconstructed images.
        x: Original images.
        mu: Mean of latent distribution.
        logvar: Log variance of latent distribution.
        reconstruction_loss_fn: Type of reconstruction loss
            ('mse' or 'masked_weighted_mse').
        beta: Weight for KL divergence term (beta-VAE).
        ivar: Inverse variance weights for each pixel (optional).
        mask: Boolean mask indicating valid pixels (optional).

    Returns:
        Tuple of (total_loss, reconstruction_loss, kl_divergence).
    """
    if reconstruction_loss_fn == "mse":
        recon_loss = nn.functional.mse_loss(reconstruction, x, reduction="sum")
    elif reconstruction_loss_fn == "masked_weighted_mse":
        if ivar is None or mask is None:
            raise ValueError(
                "masked_weighted_mse requires both ivar and mask arguments"
            )
        squared_error = (reconstruction - x).pow(2)
        weighted_error = squared_error * ivar * mask.float()
        num_valid_pixels = mask.float().sum()
        recon_loss = weighted_error.sum() / num_valid_pixels.clamp(min=1.0)
        total_pixels = torch.tensor(
            reconstruction.numel(),
            dtype=torch.float32,
            device=reconstruction.device,
        )
        recon_loss = recon_loss * total_pixels
    else:
        raise ValueError(
            f"Unknown reconstruction loss: {reconstruction_loss_fn}"
        )

    # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    total_loss = recon_loss + beta * kl_div
    return total_loss, recon_loss, kl_div


# -------------------------------------------------------------------
# DL-CFM disentanglement losses (Ganguli et al. 2025)
# -------------------------------------------------------------------


def _batch_corr_matrix(v: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Pearson correlation matrix between columns of v and w.

    Args:
        v: (N, m_v) batch of variables.
        w: (N, m_w) batch of variables.

    Returns:
        Correlation matrix of shape (m_v, m_w).
    """
    eps = 1e-8
    v_centered = v - v.mean(dim=0, keepdim=True)
    w_centered = w - w.mean(dim=0, keepdim=True)
    v_std = v_centered.std(dim=0, correction=1).clamp(min=eps)
    w_std = w_centered.std(dim=0, correction=1).clamp(min=eps)
    v_normed = v_centered / v_std
    w_normed = w_centered / w_std
    n = v.shape[0]
    return (v_normed.T @ w_normed) / (n - 1)


def _polynomial_lift(x: torch.Tensor, K: int) -> torch.Tensor:
    """Elementwise polynomial lift up to degree K.

    Args:
        x: (N, m) input tensor.
        K: Maximum polynomial degree.

    Returns:
        (N, K * m) tensor: [x, x^2, ..., x^K] concatenated.
    """
    if K == 1:
        return x
    return torch.cat([x.pow(k) for k in range(1, K + 1)], dim=1)


def align_loss(
    u_j: torch.Tensor,
    mu_j: torch.Tensor,
    K: int = 2,
) -> torch.Tensor:
    """Explicitness penalty (Eq. 9 / App. B.5).

    Encourages one-to-one alignment between auxiliary variable
    u_j and guided latent coordinate mu_j.

    Align(u_j, mu_j) = 1 - R_1^K(u_j, mu_j)

    For K=1, uses standard Pearson correlation.
    For K>=2, adds cross-degree nonlinear terms.

    Args:
        u_j: (N,) j-th auxiliary variable (normalized to [0,1]).
        mu_j: (N,) j-th guided coordinate of encoder mean.
        K: Polynomial degree for capturing nonlinear alignment.

    Returns:
        Scalar alignment loss (0 when perfectly correlated).
    """
    # Reshape to (N, 1) for matrix operations
    u = u_j.unsqueeze(1)
    mu = mu_j.unsqueeze(1)

    if K == 1:
        corr = _batch_corr_matrix(u, mu)  # (1, 1)
        return 1.0 - corr.abs().squeeze()

    # K >= 2: cross-degree terms only (k != k')
    u_lifted = _polynomial_lift(u, K)  # (N, K)
    mu_lifted = _polynomial_lift(mu, K)  # (N, K)
    # Full correlation matrix between all lifted features
    corr = _batch_corr_matrix(u_lifted, mu_lifted)  # (K, K)

    # R_1: average diagonal |correlation| for cross-degree pairs
    # Pairs (k, k') with k != k'
    total = torch.tensor(0.0, device=u_j.device)
    n_pairs = 0
    for k in range(K):
        for kp in range(K):
            if k != kp:
                total = total + (1.0 - corr[k, kp].abs())
                n_pairs += 1

    r1 = total / max(n_pairs, 1)
    return r1


def decorr_loss(
    a: torch.Tensor,
    b: torch.Tensor,
    K: int = 2,
) -> torch.Tensor:
    """Decorrelation penalty (Eq. 8 / App. B.5).

    Penalizes cross-correlation between two sets of variables.

    Decorr(a, b) = R_0^K(a, b)

    For K=1, uses standard Pearson correlations.
    For K>=2, adds cross-degree nonlinear terms.

    Args:
        a: (N, m_a) first set of variables.
        b: (N, m_b) second set of variables.
        K: Polynomial degree for capturing nonlinear dependencies.

    Returns:
        Scalar decorrelation loss (0 when fully uncorrelated).
    """
    if a.dim() == 1:
        a = a.unsqueeze(1)
    if b.dim() == 1:
        b = b.unsqueeze(1)

    m_a = a.shape[1]
    m_b = b.shape[1]

    if K == 1:
        corr = _batch_corr_matrix(a, b)  # (m_a, m_b)
        return corr.abs().mean()

    # K >= 2: cross-degree terms
    a_lifted = _polynomial_lift(a, K)  # (N, K * m_a)
    b_lifted = _polynomial_lift(b, K)  # (N, K * m_b)
    corr = _batch_corr_matrix(a_lifted, b_lifted)  # (K*m_a, K*m_b)

    # Sum |corr_ij| for cross-degree blocks only (k != k')
    total = torch.tensor(0.0, device=a.device)
    n_pairs = 0
    for k in range(K):
        for kp in range(K):
            if k != kp:
                block = corr[
                    k * m_a : (k + 1) * m_a,
                    kp * m_b : (kp + 1) * m_b,
                ]
                total = total + block.abs().sum()
                n_pairs += 1

    denom = max(n_pairs, 1) * m_a * m_b
    return total / denom


def disentangled_kl(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    aux_vars: torch.Tensor,
    n_aux: int,
    tau_sq: Optional[float] = None,
) -> torch.Tensor:
    """KL divergence against auxiliary-informed prior (Eq. 1 & 6).

    Prior: p(z | u) = N(mu_0(u), Sigma_0) where
        mu_0 = [u_1, ..., u_d, 0, ..., 0]
        Sigma_0 = diag(tau^2 * I_d, I_{d_Z - d})

    Args:
        mu: (N, d_Z) encoder mean.
        logvar: (N, d_Z) encoder log-variance.
        aux_vars: (N, d) auxiliary variables normalized to [0,1].
        n_aux: Number of auxiliary dimensions d.
        tau_sq: Variance for guided dims (default: 1/N).

    Returns:
        Scalar KL divergence averaged over the batch.
    """
    batch_size, d_z = mu.shape
    if tau_sq is None:
        tau_sq = 1.0 / batch_size

    # Build prior mean: [aux_vars, zeros]
    mu_0 = torch.zeros_like(mu)
    mu_0[:, :n_aux] = aux_vars

    # Build prior log-variance: log(tau^2) for guided, 0 for residual
    log_sigma0_sq = torch.zeros(d_z, device=mu.device)
    log_sigma0_sq[:n_aux] = torch.log(torch.tensor(tau_sq, device=mu.device))
    # Sigma_0 diagonal: tau^2 for guided, 1 for residual
    sigma0_sq = log_sigma0_sq.exp()

    # Closed-form KL (Eq. 6):
    # KL = 0.5 * [log|Sigma_0|/|Sigma_phi|
    #       - d_Z + (mu_phi - mu_0)^T Sigma_0^{-1} (mu_phi - mu_0)
    #       + tr(Sigma_0^{-1} Sigma_phi)]
    sigma_phi_sq = logvar.exp()  # (N, d_Z)
    diff = mu - mu_0  # (N, d_Z)

    kl_per_sample = 0.5 * (
        (log_sigma0_sq - logvar).sum(dim=1)  # log|Sigma_0|/|Sigma_phi|
        - d_z
        + (diff.pow(2) / sigma0_sq).sum(dim=1)
        + (sigma_phi_sq / sigma0_sq).sum(dim=1)
    )

    return kl_per_sample.mean()


def dlcfm_disentanglement_loss(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    aux_vars: torch.Tensor,
    n_aux: int,
    beta: float = 8e-5,
    lambda1: float = 8e-2,
    lambda2: float = 1e-2,
    K: int = 2,
    tau_sq: Optional[float] = None,
) -> Tuple[torch.Tensor, dict]:
    """DL-CFM disentanglement loss terms (Eq. 2, excluding flow loss).

    Computes: beta * KL(q_phi(z|x) || p(z|u))
              + lambda1 * sum_j [Align(u_j, mu_aux_j)
                                 + Decorr(u_j, mu_aux_{-j})]
              + lambda2 * Decorr(u, mu_rec)

    The flow matching MSE loss is NOT included here; it should be
    computed separately and added by the caller.

    Args:
        mu: (N, d_Z) encoder mean. First n_aux dims are
            auxiliary-guided, rest are reconstruction-focused.
        logvar: (N, d_Z) encoder log-variance.
        aux_vars: (N, n_aux) auxiliary variables normalized to [0,1].
        n_aux: Number of auxiliary-guided latent dimensions.
        beta: Weight for KL divergence term.
        lambda1: Weight for explicitness + intra-independence.
        lambda2: Weight for inter-independence.
        K: Polynomial degree for correlation penalties.
        tau_sq: Variance for guided prior dims (default: 1/N).

    Returns:
        total_loss: Scalar disentanglement loss.
        components: Dict with individual loss terms for logging.
    """
    mu_aux = mu[:, :n_aux]  # (N, n_aux)
    mu_rec = mu[:, n_aux:]  # (N, d_Z - n_aux)

    # 1. KL divergence with conditional prior
    kl = disentangled_kl(mu, logvar, aux_vars, n_aux, tau_sq)

    # 2. Explicitness + intra-independence (per guided dimension)
    align_total = torch.tensor(0.0, device=mu.device)
    intra_decorr_total = torch.tensor(0.0, device=mu.device)

    for j in range(n_aux):
        # Explicitness: guided dim j tracks u_j
        align_total = align_total + align_loss(
            aux_vars[:, j], mu_aux[:, j], K=K
        )

        # Intra-independence: u_j decorrelated from other guided dims
        if n_aux > 1:
            other_idx = [i for i in range(n_aux) if i != j]
            mu_aux_other = mu_aux[:, other_idx]  # (N, n_aux - 1)
            intra_decorr_total = intra_decorr_total + decorr_loss(
                aux_vars[:, j], mu_aux_other, K=K
            )

    # 3. Inter-independence: aux vars decorrelated from residual latents
    if mu_rec.shape[1] > 0:
        inter_decorr = decorr_loss(aux_vars, mu_rec, K=K)
    else:
        inter_decorr = torch.tensor(0.0, device=mu.device)

    # Combine
    total_loss = (
        beta * kl
        + lambda1 * (align_total + intra_decorr_total)
        + lambda2 * inter_decorr
    )

    components = {
        "kl": kl.item(),
        "align": align_total.item(),
        "intra_decorr": intra_decorr_total.item(),
        "inter_decorr": inter_decorr.item(),
        "disentanglement_loss": total_loss.item(),
    }

    return total_loss, components
