"""Conditional Normalizing Flow layers and components."""

from typing import Tuple

import torch
import torch.nn as nn


def create_checkerboard_mask(dim: int, invert: bool = False) -> torch.Tensor:
    """Create binary mask for splitting dimensions in coupling layers.

    Parameters:
    -----------
    dim : int
        Total dimension of latent space
    invert : bool
        If True, flip the mask (0s become 1s and vice versa)

    Returns:
    --------
    torch.Tensor
        Binary mask of shape (dim,) with first half 1s, second half 0s
        (or inverted if invert=True)
    """
    mask = torch.zeros(dim)
    mask[: dim // 2] = 1

    if invert:
        mask = 1 - mask

    return mask


class ConditionNetwork(nn.Module):
    """
    Network that maps conditioning variables to FiLM parameters.

    Takes conditioning input (redshift, photometry) and produces
    scale (gamma) and shift (beta) parameters for FiLM layers.
    """

    def __init__(
        self,
        condition_dim: int,
        num_film_params: int,
        hidden_dim: int = 128,
    ):
        """Initialize condition network.

        Parameters:
        -----------
        condition_dim : int
            Dimension of conditioning (e.g., 6 for redshift + 5 photometry)
        num_film_params : int
            Total number of FiLM parameters to generate
        hidden_dim : int
            Hidden layer dimension
        """
        super().__init__()
        self.condition_dim = condition_dim
        self.num_film_params = num_film_params

        self.net = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_film_params),
        )

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        """Map conditioning to FiLM parameters.

        Parameters:
        -----------
        condition : torch.Tensor
            Conditioning tensor (batch, condition_dim)

        Returns:
        --------
        torch.Tensor
            FiLM parameters (batch, num_film_params)
        """
        return self.net(condition)


class TransformationNetwork(nn.Module):
    """
    Network that predicts scale and shift for affine coupling.

    Takes masked latent dimensions and FiLM parameters, outputs
    scale (s) and shift (t) for transforming the other half.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
    ):
        """Initialize transformation network.

        Parameters:
        -----------
        input_dim : int
            Dimension of masked input (latent_dim // 2)
        output_dim : int
            Dimension of output scale/shift (latent_dim // 2)
        hidden_dim : int
            Hidden layer dimension
        num_layers : int
            Number of hidden layers
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.layers = nn.ModuleList(layers)

        # Output layer: produces both scale and shift
        self.output_layer = nn.Linear(hidden_dim, output_dim * 2)

        # Initialize output layer to near-identity transformation
        nn.init.zeros_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(
        self, x: torch.Tensor, film_params: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute scale and shift for coupling transformation.

        Parameters:
        -----------
        x : torch.Tensor
            Masked latent input (batch, input_dim)
        film_params : torch.Tensor
            FiLM parameters from condition network
            (batch, num_layers * hidden_dim * 2)

        Returns:
        --------
        scale : torch.Tensor
            (batch, output_dim)
        shift : torch.Tensor
            (batch, output_dim)
        """
        h = x

        # Split FiLM parameters for each layer
        film_param_size = self.hidden_dim * 2
        film_params_split = film_params.view(
            film_params.size(0), self.num_layers, film_param_size
        )

        # Forward through layers with FiLM modulation
        for i, layer in enumerate(self.layers):
            h = layer(h)

            # Extract gamma and beta
            gamma_beta = film_params_split[:, i, :]
            gamma = gamma_beta[:, : self.hidden_dim]
            beta = gamma_beta[:, self.hidden_dim :]

            # Apply FiLM modulation: gamma * h + beta
            h = gamma * h + beta

            h = torch.nn.functional.silu(h)

        # Output scale and shift
        output = self.output_layer(h)
        scale, shift = output.chunk(2, dim=1)

        scale = torch.clamp(scale, -10.0, 10.0)

        return scale, shift


class AffineCoupling(nn.Module):
    """
    Affine coupling layer with FiLM conditioning.

    Splits input into two parts using a binary mask, then applies:
        z2' = z2 * exp(s(z1, c)) + t(z1, c)

    where s and t are scale and shift predicted by transformation network,
    and c is conditioning via FiLM parameters.
    """

    def __init__(
        self,
        latent_dim: int,
        condition_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        mask: torch.Tensor = None,
    ):
        """Initialize affine coupling layer.

        Parameters:
        -----------
        latent_dim : int
            Dimension of latent space
        condition_dim : int
            Dimension of conditioning
        hidden_dim : int
            Hidden dimension for transformation network
        num_layers : int
            Number of layers in transformation network
        mask : torch.Tensor
            Binary mask for splitting dimensions (created if None)
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim

        if mask is None:
            mask = create_checkerboard_mask(latent_dim, invert=False)
        self.register_buffer("mask", mask)

        # Dimensions for transformation network
        masked_dim = int(mask.sum().item())
        transform_dim = latent_dim - masked_dim

        # FiLM parameters
        num_film_params = num_layers * hidden_dim * 2

        self.condition_net = ConditionNetwork(
            condition_dim, num_film_params, hidden_dim=hidden_dim
        )

        # Transformation network: z_masked + FiLM → (scale, shift)
        self.transform_net = TransformationNetwork(
            input_dim=masked_dim,
            output_dim=transform_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )

    def forward(
        self, z: torch.Tensor, condition: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: z → z_base (for training/density estimation).

        Parameters:
        -----------
        z : torch.Tensor
            Latent input (batch, latent_dim)
        condition : torch.Tensor
            Conditioning (batch, condition_dim)

        Returns:
        --------
        z_transformed : torch.Tensor
            Transformed latent (batch, latent_dim)
        log_det_jacobian : torch.Tensor
            Log determinant of Jacobian (batch,)
        """
        mask_bool = self.mask.bool()

        # Extract compact views: masked dims (input to net) and transform dims (to be changed)
        z_masked_compact = z[:, mask_bool]      # (batch, masked_dim)
        z_transform_compact = z[:, ~mask_bool]  # (batch, transform_dim)

        film_params = self.condition_net(condition)

        scale, shift = self.transform_net(z_masked_compact, film_params)

        # Apply affine transformation to the non-masked dims
        z_transform_new = z_transform_compact * torch.exp(scale) + shift

        z_out = z.clone()
        z_out[:, ~mask_bool] = z_transform_new

        # Log determinant is sum of scale
        log_det = scale.sum(dim=1)

        return z_out, log_det

    def inverse(
        self, z: torch.Tensor, condition: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inverse pass: z_base → z (for sampling).

        Parameters:
        -----------
        z : torch.Tensor
            Latent from base distribution (batch, latent_dim)
        condition : torch.Tensor
            Conditioning (batch, condition_dim)

        Returns:
        --------
        z_original : torch.Tensor
            Inverse transformed latent (batch, latent_dim)
        log_det_jacobian : torch.Tensor
            Log determinant of Jacobian (batch,)
        """
        mask_bool = self.mask.bool()

        # Extract compact views
        z_masked_compact = z[:, mask_bool]      # (batch, masked_dim)
        z_transform_compact = z[:, ~mask_bool]  # (batch, transform_dim)

        film_params = self.condition_net(condition)
        scale, shift = self.transform_net(z_masked_compact, film_params)

        # Apply inverse affine transformation
        z_transform_orig = (z_transform_compact - shift) * torch.exp(-scale)

        z_out = z.clone()
        z_out[:, ~mask_bool] = z_transform_orig

        # Log determinant (negative of forward)
        log_det = -scale.sum(dim=1)

        return z_out, log_det
