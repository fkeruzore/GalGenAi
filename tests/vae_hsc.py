# VAE on HSC images
import numpy as np
import matplotlib.pyplot as plt
import random

import torch
from torch.utils.data import DataLoader, random_split
from datasets import load_from_disk

from galgenai import VAE, get_device, get_device_name
from galgenai.training import train, vae_loss


device = get_device()


class FluxDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        hf_dataset,
        nx: int,
        mins: torch.TensorType,
        maxs: torch.TensorType,
    ):
        self.dataset = hf_dataset
        self.nx = nx

        # crop
        self.og_nx2 = self.dataset[0]["image"]["flux"].shape[1] // 2
        self.og_ny2 = self.dataset[0]["image"]["flux"].shape[2] // 2
        self.nx2 = nx // 2

        # norms
        self.mins = mins[:, None, None]
        self.maxs = maxs[:, None, None]

        # bands
        self.bands = self.dataset[0]["image"]["band"]
        self.n_bands = self.dataset[0]["image"]["flux"].shape[0]

    def __len__(self):
        return len(self.dataset)

    def normalize(self, img):
        return (img - self.mins) / (self.maxs - self.mins)

    def crop(self, img):
        return img[
            :,
            self.og_nx2 - self.nx2 : self.og_nx2 + self.nx2,
            self.og_ny2 - self.nx2 : self.og_ny2 + self.nx2,
        ]

    def __getitem__(self, idx):
        image_data = self.dataset[idx]["image"]

        # Extract and crop flux
        flux = self.crop(image_data["flux"])
        # flux_normalized = self.normalize(flux)
        flux_normalized = flux

        # Extract and crop inverse variance
        ivar = self.crop(image_data["ivar"])
        # ivar_normalized = self.normalize(ivar**(-0.5)) ** (-2)
        ivar_normalized = ivar

        # Extract and crop mask
        mask = self.crop(image_data["mask"])

        return flux_normalized, ivar_normalized, mask


def get_and_validate_data(path="./data/hsc_mmu_mini/", nx=64, minmax=(-2, 99)):
    dataset_raw = load_from_disk(path)
    dataset_raw = dataset_raw.select_columns(["image"]).with_format("torch")

    n_gals = len(dataset_raw)
    bands = dataset_raw[0]["image"]["band"]

    # min/max
    mins = torch.Tensor(minmax[0] * np.ones(len(bands)))
    maxs = torch.Tensor(minmax[1] * np.ones(len(bands)))

    dataset = FluxDataset(dataset_raw, nx=nx, mins=mins, maxs=maxs)
    n_bands, n_x, n_y = dataset[0][0].shape  # First element of tuple is flux
    assert n_x == n_y
    assert len(bands) == n_bands
    print(f"Images dimension: {n_bands}*{n_x}*{n_y} ({n_gals} galaxies)")

    return dataset


def get_and_validate_model(n_bands, latent_dim, nx):
    model = VAE(in_channels=n_bands, latent_dim=latent_dim, input_size=nx).to(
        device
    )

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    x = torch.randn(4, n_bands, nx, nx).to(device)  # batch
    print(f"Input shape: {x.shape}")

    reconstruction, mu, logvar = model(x)
    print(f"Reconstruction shape: {reconstruction.shape}")
    print(f"Latent mean shape: {mu.shape}")
    print(f"Latent log variance shape: {logvar.shape}")

    return model


def mini_train(
    model,
    dataset,
    batch_size=128,
    num_epochs=1,
    lr=1e-3,
    num_workers=8,
    seed=42,
):
    print(
        f"BATCH DIMENSION: {batch_size * dataset.n_bands * dataset.nx**2:.2e};",
        f"# NITS: {batch_size * model.encoder.latent_dim:.2e}",
    )

    torch.random.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    dataset_train, dataset_test = random_split(dataset, [0.8, 0.2])

    train_loader = DataLoader(
        dataset_train,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
    )
    test_loader = DataLoader(
        dataset_test,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )

    # Get first batch for sanity check
    first_batch = next(iter(train_loader))
    flux, ivar, mask = [x.to(device) for x in first_batch]

    # Compute initial loss before training
    model.eval()
    with torch.no_grad():
        recon, mu, logvar = model(flux)
        L_initial, _, _ = vae_loss(
            recon,
            flux,
            mu,
            logvar,
            reconstruction_loss_fn="masked_weighted_mse",
            ivar=ivar,
            mask=mask,
            beta=1.0,
        )

    # Warm up
    train(
        model=model,
        train_loader=train_loader,
        optimizer=torch.optim.Adam(model.parameters(), lr=lr),
        device=device,
        num_epochs=num_epochs,
        reconstruction_loss_fn="masked_weighted_mse",
        beta=1.0,
        max_grad_norm=2.0,
    )

    # Compute final loss after training
    model.eval()
    with torch.no_grad():
        recon, mu, logvar = model(flux)
        L_final, _, _ = vae_loss(
            recon,
            flux,
            mu,
            logvar,
            reconstruction_loss_fn="masked_weighted_mse",
            ivar=ivar,
            mask=mask,
            beta=1.0,
        )
    print(f"Initial loss on first batch: {L_initial.item():.3e}")
    print(f"Final loss on first batch: {L_final.item():.3e}")

    # Sanity check: loss should decrease
    assert L_final < L_initial, (
        f"Loss didn't decrease! {L_initial.item():.3e} -> {L_final.item():.3e}"
    )


def plot(data_loader, model=None, plot_type="reconst", n_plot=3):
    fig, axs = plt.subplots(2 * n_plot, dataset.n_bands, figsize=(12, 10))
    cmap = plt.get_cmap("magma")
    cmap.set_bad("0.5")

    imgs_a = torch.stack([torch.stack(dataset[i]) for i in range(n_plot)]).to(
        device
    )
    if (model is not None) and (plot_type == "reconst"):
        print("Plotting images & reconstructions")
        model.eval()
        with torch.no_grad():
            imgs_b, _, _ = model(imgs_a[:, 0, :, :, :].to(device))
    else:
        print("Plotting images & stdevs")
        imgs_b = imgs_a[:, 1] ** -0.5

    for i in range(n_plot):
        img_a, ivr, msk = imgs_a[i].cpu()
        img_b = imgs_b[i].cpu()

        for j, band in enumerate(dataset.bands):
            vmin, vmax = img_a[j].min(), img_a[j].max()
            axs[2 * i + 0, j].imshow(
                img_a[j] / msk[j],
                origin="lower",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
            axs[2 * i + 1, j].imshow(
                img_b[j] / msk[j],
                origin="lower",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
            axs[2 * i + 0, j].set_title(band)
            axs[2 * i + 0, j].set_ylabel("image")
            axs[2 * i + 1, j].set_ylabel(plot_type)

    for ax in axs.flatten():
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
    fig.tight_layout()
    return fig, axs


if __name__ == "__main__":
    print(f"Using device: {get_device_name()}")

    dataset = get_and_validate_data()

    fig1 = plot(dataset, plot_type="stdev")

    model = get_and_validate_model(dataset.n_bands, 32, dataset.nx)

    fig2 = plot(dataset, model=model, plot_type="reconst")

    mini_train(model, dataset, num_epochs=3)

    fig3 = plot(dataset, model=model, plot_type="reconst")

    plt.show()
