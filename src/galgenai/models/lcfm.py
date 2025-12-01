"""
Latent Conditional Flow Matching (LCFM) Implementation

Based on Samaddar et al. (2025) "Efficient Flow Matching using Latent Variables"

Architecture:
- Frozen VAE encoder extracts features from images
- Trainable stochastic layer outputs latent distribution q(f|x)
- U-Net predicts velocity field v(x_t, f, t) for flow matching
- Flow transports noise x_0 ~ N(0,I) to data x_1 conditioned on latent f
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


# =============================================================================
# Building Blocks
# =============================================================================


class SinusoidalTimeEmbedding(nn.Module):
    """
    Sinusoidal positional embedding for time t ∈ [0, 1].

    Maps scalar time to a high-dimensional vector using sin/cos at different
    frequencies, similar to transformer positional encodings. This gives the
    network a rich representation of where we are along the flow trajectory.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (batch,) tensor of times in [0, 1]
        Returns:
            (batch, dim) time embeddings
        """
        device = t.device
        half_dim = self.dim // 2

        # Frequencies span from 1 to 10000 on log scale
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half_dim, device=device) / half_dim
        )

        # Outer product: (batch, 1) * (half_dim,) -> (batch, half_dim)
        args = (
            t[:, None] * freqs[None, :] * 1000
        )  # scale up for better gradients

        # Concatenate sin and cos
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ResBlock(nn.Module):
    """
    Residual block with time/latent conditioning via adaptive group normalization.

    The conditioning signal (time + latent embedding) modulates the features
    through scale and shift parameters after group normalization. This is the
    standard approach in diffusion/flow models (e.g., ADM, DiT).

    Architecture:
        x -> GroupNorm -> scale/shift by cond -> SiLU -> Conv
          -> GroupNorm -> scale/shift by cond -> SiLU -> Dropout -> Conv -> + x
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        dropout: float = 0.1,
        num_groups: int = 8,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # First conv block
        self.norm1 = nn.GroupNorm(num_groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        # Second conv block
        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.dropout = nn.Dropout(dropout)

        # Conditioning: project to scale and shift for both norms
        # Output: 4 * out_channels (scale1, shift1, scale2, shift2)
        self.cond_proj = nn.Sequential(
            nn.SiLU(), nn.Linear(cond_dim, out_channels * 4)
        )

        # Skip connection (identity if channels match, else 1x1 conv)
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, in_channels, H, W) input features
            cond: (batch, cond_dim) conditioning vector (time + latent)
        Returns:
            (batch, out_channels, H, W) output features
        """
        # Get conditioning parameters
        cond_params = self.cond_proj(cond)  # (batch, out_channels * 4)
        scale1, shift1, scale2, shift2 = cond_params.chunk(4, dim=1)

        # Reshape for broadcasting: (batch, channels) -> (batch, channels, 1, 1)
        scale1 = scale1[:, :, None, None]
        shift1 = shift1[:, :, None, None]
        scale2 = scale2[:, :, None, None]
        shift2 = shift2[:, :, None, None]

        # First block with adaptive normalization
        h = self.norm1(x)
        # For first norm, we need to handle channel mismatch
        if self.in_channels != self.out_channels:
            # Project x first, then apply scale/shift
            h = F.silu(h)
            h = self.conv1(h)
            h = h * (1 + scale1) + shift1
        else:
            h = h * (1 + scale1) + shift1
            h = F.silu(h)
            h = self.conv1(h)

        # Second block
        h = self.norm2(h)
        h = h * (1 + scale2) + shift2
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.skip(x)


class AttentionBlock(nn.Module):
    """
    Self-attention block for capturing long-range spatial dependencies.

    Applied at lower resolutions (e.g., 8x8, 16x16) where the computational
    cost is manageable. Uses multi-head attention with QKV projection.
    """

    def __init__(self, channels: int, num_heads: int = 4, num_groups: int = 8):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim**-0.5

        self.norm = nn.GroupNorm(num_groups, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, H, W)
        Returns:
            (batch, channels, H, W)
        """
        B, C, H, W = x.shape

        h = self.norm(x)
        qkv = self.qkv(h)  # (B, 3*C, H, W)
        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, H * W)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # (3, B, heads, H*W, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention: softmax(QK^T / sqrt(d)) V
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)  # (B, heads, H*W, head_dim)
        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)

        return x + self.proj(out)


class Downsample(nn.Module):
    """Spatial downsampling by factor of 2 using strided convolution."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """Spatial upsampling by factor of 2 using nearest neighbor + conv."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


# =============================================================================
# U-Net Velocity Field Network
# =============================================================================


class VelocityUNet(nn.Module):
    """
    U-Net architecture for predicting velocity field v(x_t, f, t).

    The network takes:
    - x_t: noisy image at time t (interpolation between noise and data)
    - f: latent features from VAE encoder
    - t: scalar time in [0, 1]

    And outputs the predicted velocity field (same shape as x_t).

    Architecture follows standard U-Net with:
    - Encoder: progressively downsamples while increasing channels
    - Middle: self-attention at lowest resolution
    - Decoder: progressively upsamples with skip connections from encoder
    - Conditioning: time and latent embeddings added and injected via AdaGN

    Hyperparameter choices (following Samaddar et al. for scientific data):
    - Base channels: 64 (reduced from 128 since our images are 64x64 not 256x256)
    - Channel multipliers: [1, 2, 4, 4] gives [64, 128, 256, 256] channels
    - Attention at 16x16 and 8x8 resolutions
    - 2 residual blocks per resolution level
    """

    def __init__(
        self,
        in_channels: int = 5,
        latent_dim: int = 32,
        base_channels: int = 64,
        channel_mult: Tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int, ...] = (16, 8),
        dropout: float = 0.1,
        num_heads: int = 4,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.base_channels = base_channels

        # Compute channel counts at each level
        channels = [base_channels * m for m in channel_mult]

        # Time embedding: scalar -> vector
        time_dim = base_channels * 4
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Latent embedding: project to same dimension as time
        self.latent_embed = nn.Sequential(
            nn.Linear(latent_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Combined conditioning dimension
        cond_dim = time_dim

        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # Encoder (downsampling path)
        self.encoder_blocks = nn.ModuleList()
        self.downsamplers = nn.ModuleList()

        current_res = 64  # Starting resolution
        prev_channels = base_channels

        for level, ch in enumerate(channels):
            level_blocks = nn.ModuleList()

            for _ in range(num_res_blocks):
                level_blocks.append(
                    ResBlock(prev_channels, ch, cond_dim, dropout)
                )
                prev_channels = ch

                # Add attention at specified resolutions
                if current_res in attention_resolutions:
                    level_blocks.append(AttentionBlock(ch, num_heads))

            self.encoder_blocks.append(level_blocks)

            # Downsample except at last level
            if level < len(channels) - 1:
                self.downsamplers.append(Downsample(ch))
                current_res //= 2
            else:
                self.downsamplers.append(nn.Identity())

        # Middle block (at lowest resolution)
        self.middle = nn.ModuleList(
            [
                ResBlock(channels[-1], channels[-1], cond_dim, dropout),
                AttentionBlock(channels[-1], num_heads),
                ResBlock(channels[-1], channels[-1], cond_dim, dropout),
            ]
        )

        # Decoder (upsampling path)
        self.decoder_blocks = nn.ModuleList()
        self.upsamplers = nn.ModuleList()

        reversed_channels = list(reversed(channels))

        for level, ch in enumerate(reversed_channels):
            level_blocks = nn.ModuleList()

            # Output channels for this level
            out_ch = reversed_channels[
                min(level + 1, len(reversed_channels) - 1)
            ]
            if level == len(reversed_channels) - 1:
                out_ch = base_channels

            for i in range(
                num_res_blocks + 1
            ):  # +1 for skip connection processing
                # Input channels include skip connection
                block_in = prev_channels + ch if i == 0 else out_ch
                block_out = out_ch

                level_blocks.append(
                    ResBlock(block_in, block_out, cond_dim, dropout)
                )
                prev_channels = block_out

                if current_res in attention_resolutions:
                    level_blocks.append(AttentionBlock(block_out, num_heads))

            self.decoder_blocks.append(level_blocks)

            # Upsample except at last level
            if level < len(reversed_channels) - 1:
                self.upsamplers.append(Upsample(out_ch))
                current_res *= 2
            else:
                self.upsamplers.append(nn.Identity())

        # Final output
        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, in_channels, 3, padding=1),
        )

        # Initialize output conv to zero for stable training start
        nn.init.zeros_(self.conv_out[-1].weight)
        nn.init.zeros_(self.conv_out[-1].bias)

    def forward(
        self, x: torch.Tensor, f: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict velocity field at time t.

        Args:
            x: (batch, in_channels, 64, 64) noisy image x_t
            f: (batch, latent_dim) latent features
            t: (batch,) time values in [0, 1]

        Returns:
            (batch, in_channels, 64, 64) predicted velocity
        """
        # Compute conditioning: time + latent embeddings (added together)
        t_emb = self.time_embed(t)
        f_emb = self.latent_embed(f)
        cond = t_emb + f_emb  # Simple addition, as in Samaddar et al.

        # Initial conv
        h = self.conv_in(x)

        # Encoder - store outputs for skip connections
        skips = []
        for level_blocks, downsample in zip(
            self.encoder_blocks, self.downsamplers
        ):
            for block in level_blocks:
                if isinstance(block, ResBlock):
                    h = block(h, cond)
                else:
                    h = block(h)
            skips.append(h)
            h = downsample(h)

        # Middle
        for block in self.middle:
            if isinstance(block, ResBlock):
                h = block(h, cond)
            else:
                h = block(h)

        # Decoder with skip connections
        for level_blocks, upsample in zip(
            self.decoder_blocks, self.upsamplers
        ):
            # Concatenate skip connection
            skip = skips.pop()
            h = torch.cat([h, skip], dim=1)

            for block in level_blocks:
                if isinstance(block, ResBlock):
                    h = block(h, cond)
                else:
                    h = block(h)

            h = upsample(h)

        return self.conv_out(h)


# =============================================================================
# Latent Stochastic Layer
# =============================================================================


class LatentStochasticLayer(nn.Module):
    """
    Trainable stochastic layer on top of frozen VAE encoder.

    Takes the (μ, logvar) from the frozen encoder and learns to refine them
    for the flow matching task. This is a key component from Samaddar et al.:
    rather than using the encoder's latent directly, we add a learnable layer
    that can adapt the latent distribution specifically for conditioning the flow.

    The KL divergence term in training encourages this distribution to stay
    close to N(0, I), which regularizes the latent space.
    """

    def __init__(self, latent_dim: int = 32, hidden_dim: int = 64):
        super().__init__()

        self.latent_dim = latent_dim

        # Refine μ and logvar from encoder
        # Input: concatenation of μ and logvar (2 * latent_dim)
        self.refine = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

        # Initialize to approximately identity mapping
        nn.init.zeros_(self.mu_head.weight)
        nn.init.zeros_(self.mu_head.bias)
        nn.init.zeros_(self.logvar_head.weight)
        nn.init.zeros_(self.logvar_head.bias)

    def forward(
        self, mu_enc: torch.Tensor, logvar_enc: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            mu_enc: (batch, latent_dim) mean from frozen encoder
            logvar_enc: (batch, latent_dim) log-variance from frozen encoder

        Returns:
            f: (batch, latent_dim) sampled latent (reparameterized)
            mu: (batch, latent_dim) refined mean
            logvar: (batch, latent_dim) refined log-variance
        """
        # Concatenate encoder outputs
        h = torch.cat([mu_enc, logvar_enc], dim=-1)
        h = self.refine(h)

        # Predict refined parameters (as residuals to encoder output)
        mu = mu_enc + self.mu_head(h)
        logvar = logvar_enc + self.logvar_head(h)

        # Clamp logvar for numerical stability
        logvar = torch.clamp(logvar, min=-10, max=10)

        # Reparameterization trick: f = μ + σ * ε, where ε ~ N(0, I)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        f = mu + std * eps

        return f, mu, logvar


# =============================================================================
# Full LCFM Model
# =============================================================================


class LCFM(nn.Module):
    """
    Latent Conditional Flow Matching model.

    Combines:
    1. Frozen VAE encoder (provided externally)
    2. Trainable stochastic layer for latent refinement
    3. U-Net velocity field network

    Training objective (Eq. 11 from Samaddar et al.):
        L = E[||v_θ(x_t, f, t) - u_t||²] + β * KL(q(f|x₁) || N(0,I))

    where:
        - x_t = (1-t)*x₀ + t*x₁ (linear interpolation)
        - u_t = x₁ - x₀ (target velocity, constant along path)
        - f ~ q(f|x₁) (latent from stochastic layer)
    """

    def __init__(
        self,
        vae_encoder: nn.Module,
        latent_dim: int = 32,
        in_channels: int = 5,
        base_channels: int = 64,
        beta: float = 0.001,
        **unet_kwargs,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.beta = beta

        # Frozen VAE encoder
        self.vae_encoder = vae_encoder
        for param in self.vae_encoder.parameters():
            param.requires_grad = False
        self.vae_encoder.eval()

        # Trainable components
        self.latent_layer = LatentStochasticLayer(latent_dim)
        self.velocity_net = VelocityUNet(
            in_channels=in_channels,
            latent_dim=latent_dim,
            base_channels=base_channels,
            **unet_kwargs,
        )

    def encode(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode image to latent space.

        Args:
            x: (batch, channels, H, W) input image

        Returns:
            f: sampled latent
            mu: mean of latent distribution
            logvar: log-variance of latent distribution
        """
        with torch.no_grad():
            mu_enc, logvar_enc = self.vae_encoder(x)

        return self.latent_layer(mu_enc, logvar_enc)

    def compute_loss(
        self, x1: torch.Tensor, return_components: bool = False
    ) -> torch.Tensor:
        """
        Compute LCFM training loss.

        Args:
            x1: (batch, channels, H, W) real images from dataset
            return_components: if True, also return individual loss terms

        Returns:
            loss: scalar loss value
            (optional) dict with 'flow_loss' and 'kl_loss'
        """
        batch_size = x1.shape[0]
        device = x1.device

        # 1. Encode to get latent features
        f, mu, logvar = self.encode(x1)

        # 2. Sample noise (source distribution)
        x0 = torch.randn_like(x1)

        # 3. Sample time uniformly
        t = torch.rand(batch_size, device=device)

        # 4. Interpolate: x_t = (1-t)*x_0 + t*x_1
        # Reshape t for broadcasting: (batch,) -> (batch, 1, 1, 1)
        t_broadcast = t[:, None, None, None]
        x_t = (1 - t_broadcast) * x0 + t_broadcast * x1

        # 5. Target velocity (constant along straight path)
        u_t = x1 - x0

        # 6. Predict velocity
        v_pred = self.velocity_net(x_t, f, t)

        # 7. Flow matching loss (MSE)
        flow_loss = F.mse_loss(v_pred, u_t)

        # 8. KL divergence loss: KL(N(μ, σ²) || N(0, I))
        # = 0.5 * sum(μ² + σ² - log(σ²) - 1)
        kl_loss = -0.5 * torch.mean(
            torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        )

        # 9. Total loss
        loss = flow_loss + self.beta * kl_loss

        if return_components:
            return loss, {
                "flow_loss": flow_loss.item(),
                "kl_loss": kl_loss.item(),
                "total_loss": loss.item(),
            }
        return loss

    @torch.no_grad()
    def sample(
        self,
        x_train: torch.Tensor,
        num_steps: int = 50,
        return_trajectory: bool = False,
    ) -> torch.Tensor:
        """
        Generate samples using the trained model.

        Following Samaddar et al., we sample latents from the aggregate posterior
        by encoding training images. This is different from sampling from N(0,I)!

        Args:
            x_train: (batch, channels, H, W) training images to condition on
            num_steps: number of ODE integration steps (Euler method)
            return_trajectory: if True, return intermediate states

        Returns:
            x1: (batch, channels, H, W) generated samples
            (optional) trajectory: list of intermediate states
        """
        self.eval()

        batch_size = x_train.shape[0]
        device = x_train.device

        # Get latent features from training images
        f, _, _ = self.encode(x_train)

        # Start from noise
        x = torch.randn(batch_size, self.in_channels, 64, 64, device=device)

        trajectory = [x.clone()] if return_trajectory else None

        # Euler integration from t=0 to t=1
        dt = 1.0 / num_steps
        for step in range(num_steps):
            t = torch.full((batch_size,), step * dt, device=device)

            # Predict velocity at current state
            v = self.velocity_net(x, f, t)

            # Euler step
            x = x + v * dt

            if return_trajectory:
                trajectory.append(x.clone())

        if return_trajectory:
            return x, trajectory
        return x

    @torch.no_grad()
    def sample_with_ode_solver(
        self,
        x_train: torch.Tensor,
        solver: str = "dopri5",
        rtol: float = 1e-5,
        atol: float = 1e-5,
    ) -> torch.Tensor:
        """
        Generate samples using adaptive ODE solver (requires torchdiffeq).

        This is more accurate than fixed-step Euler but slower.

        Args:
            x_train: training images to get latents from
            solver: ODE solver ('dopri5', 'rk4', etc.)
            rtol, atol: tolerances for adaptive solver

        Returns:
            Generated samples
        """
        try:
            from torchdiffeq import odeint
        except ImportError:
            raise ImportError("Install torchdiffeq: pip install torchdiffeq")

        self.eval()

        batch_size = x_train.shape[0]
        device = x_train.device

        # Get latent features
        f, _, _ = self.encode(x_train)

        # Define ODE function
        def ode_fn(t, x):
            t_batch = torch.full((batch_size,), t.item(), device=device)
            return self.velocity_net(x, f, t_batch)

        # Initial condition
        x0 = torch.randn(batch_size, self.in_channels, 64, 64, device=device)

        # Solve ODE
        t_span = torch.tensor([0.0, 1.0], device=device)
        solution = odeint(
            ode_fn, x0, t_span, method=solver, rtol=rtol, atol=atol
        )

        return solution[-1]  # Return final state at t=1


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
