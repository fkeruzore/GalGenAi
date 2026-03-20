"""Conditional Normalizing Flow for VAE latent space."""

import math
from typing import Tuple

import torch
import torch.nn as nn

from .cnf_layers import AffineCoupling, create_checkerboard_mask


class ConditionalNormalizingFlow(nn.Module):
    """
    Conditional Normalizing Flow for modeling VAE latent distribution.

    - Density estimation: compute exact log p(z|c)
    - Conditional sampling: sample z ~ p(z|c) given conditioning
    """

    def __init__(
        self,
        latent_dim: int = 32,
        condition_dim: int = 6,
        num_blocks: int = 8,
        hidden_dim: int = 256,
        num_layers: int = 3,
    ):
        """Initialize Conditional Normalizing Flow.

        Parameters:
        -----------
        latent_dim : int
            Dimension of VAE latent space (default: 16)
        condition_dim : int
            Dimension of conditioning (default: 6 for redshift + 5 photometry)
        num_blocks : int
            Number of coupling blocks (default: 8)
        hidden_dim : int
            Hidden dimension for transformation networks (default: 256)
        num_layers : int
            Number of layers per transformation network (default: 3)
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.num_blocks = num_blocks

        # Create coupling blocks with alternating masks
        self.coupling_blocks = nn.ModuleList()
        for i in range(num_blocks):
            # Alternate mask direction
            mask = create_checkerboard_mask(latent_dim, invert=(i % 2 == 1))

            coupling = AffineCoupling(
                latent_dim=latent_dim,
                condition_dim=condition_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                mask=mask,
            )
            self.coupling_blocks.append(coupling)

    def forward(
        self, z: torch.Tensor, condition: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: latent z → base distribution u.

        Used for training and density estimation.

        Parameters:
        -----------
        z : torch.Tensor
            Latent codes (batch, latent_dim)
        condition : torch.Tensor
            Conditioning variables (batch, condition_dim)

        Returns:
        --------
        z_base : torch.Tensor
            Transformed to base distribution (batch, latent_dim)
        log_det_jacobian : torch.Tensor
            Log determinant of Jacobian (batch,)
        """
        log_det_total = torch.zeros(z.size(0), device=z.device)

        z_current = z
        for coupling in self.coupling_blocks:
            z_current, log_det = coupling.forward(z_current, condition)
            log_det_total += log_det

        return z_current, log_det_total

    def inverse(
        self, z_base: torch.Tensor, condition: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inverse pass: base distribution u → latent z.

        Used for sampling z ~ p(z|c).

        Parameters:
        -----------
        z_base : torch.Tensor
            Samples from base distribution N(0, I) (batch, latent_dim)
        condition : torch.Tensor
            Conditioning variables (batch, condition_dim)

        Returns:
        --------
        z : torch.Tensor
            Latent codes (batch, latent_dim)
        log_det_jacobian : torch.Tensor
            Log determinant of Jacobian (batch,)
        """
        log_det_total = torch.zeros(z_base.size(0), device=z_base.device)

        z_current = z_base
        # Apply inverse transformations in reverse order
        for coupling in reversed(self.coupling_blocks):
            z_current, log_det = coupling.inverse(z_current, condition)
            log_det_total += log_det

        return z_current, log_det_total

    def log_prob(self, z: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Compute log probability log p(z|c).

        Uses the change of variables formula:
            log p(z|c) = log p_base(f(z|c)) + log |det J_f|

        where p_base = N(0, I) and f is the forward transformation.

        Parameters:
        -----------
        z : torch.Tensor
            Latent codes (batch, latent_dim)
        condition : torch.Tensor
            Conditioning variables (batch, condition_dim)

        Returns:
        --------
        log_prob : torch.Tensor
            Log probability of z given c (batch,)
        """
        # Transform to base distribution
        z_base, log_det = self.forward(z, condition)

        # Compute log probability under base distribution N(0, I)
        # log p(u) = -0.5 * (u^2 + log(2π))
        log_p_base = -0.5 * (
            z_base.pow(2).sum(dim=1) + self.latent_dim * math.log(2 * math.pi)
        )

        # Apply change of variables
        log_prob = log_p_base + log_det

        return log_prob

    def sample(
        self, condition: torch.Tensor, num_samples: int = 1
    ) -> torch.Tensor:
        """Sample latent codes z ~ p(z|c).

        Parameters:
        -----------
        condition : torch.Tensor
            Conditioning variables (batch, condition_dim)
        num_samples : int
            Number of samples per conditioning (default: 1)

        Returns:
        --------
        z : torch.Tensor
            Sampled latent codes (batch * num_samples, latent_dim)
        """
        batch_size = condition.size(0)
        device = condition.device

        # Expand conditioning for multiple samples
        if num_samples > 1:
            condition = condition.repeat_interleave(num_samples, dim=0)

        z_base = torch.randn(
            batch_size * num_samples, self.latent_dim, device=device
        )
        z, _ = self.inverse(z_base, condition)

        return z