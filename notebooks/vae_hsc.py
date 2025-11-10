# %% [markdown]
# # VAE on HSC images

# %%
import numpy as np
import matplotlib.pyplot as plt

import torch
from datasets import load_from_disk

from galgenai import VAE, get_device, get_device_name


device = get_device()
print(f"Using device: {get_device_name()}")

# %% [markdown]
# ## Load and inspect data

# %%
dataset_raw = load_from_disk("../data/hsc_mmu_mini/")
dataset_raw = dataset_raw.select_columns(["image"]).with_format("torch")

n_gals = len(dataset_raw)
bands = dataset_raw[0]["image"]["band"]

# dataset.set_transform(lambda data: {"flux": data["image"]["flux"]})
# dataset = dataset.map(lambda x: {"flux": x["image"]["flux"]}, remove_columns=["image"])

# min/max
mins = torch.Tensor(-2 * np.ones(len(bands)))
maxs = torch.Tensor(99 * np.ones(len(bands)))


class FluxDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        hf_dataset,
        nx: int,
        mins: torch.TensorType,
        maxs: torch.TensorType,
    ):
        self.dataset = hf_dataset

        # crop to 128
        self.og_nx2 = self.dataset[0]["image"]["flux"].shape[1] // 2
        self.og_ny2 = self.dataset[0]["image"]["flux"].shape[2] // 2
        self.mins = mins[:, None, None]
        self.maxs = maxs[:, None, None]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image_data = self.dataset[idx]["image"]

        # Extract and crop flux
        flux = image_data["flux"][
            :,
            self.og_nx2 - 64 : self.og_nx2 + 64,
            self.og_ny2 - 64 : self.og_ny2 + 64,
        ]
        flux_normalized = (flux - self.mins) / (self.maxs - self.mins)

        # Extract and crop inverse variance
        ivar = image_data["ivar"][
            :,
            self.og_nx2 - 64 : self.og_nx2 + 64,
            self.og_ny2 - 64 : self.og_ny2 + 64,
        ]

        # Extract and crop mask
        mask = image_data["mask"][
            :,
            self.og_nx2 - 64 : self.og_nx2 + 64,
            self.og_ny2 - 64 : self.og_ny2 + 64,
        ]

        return flux_normalized, ivar, mask


dataset = FluxDataset(dataset_raw, nx=128, mins=mins, maxs=maxs)
n_bands, n_x, n_y = dataset[0][0].shape  # First element of tuple is flux
assert n_x == n_y
assert len(bands) == n_bands
print(f"Images dimension: {n_bands}*{n_x}*{n_y} ({n_gals} galaxies)")

# %%
_n_plot = 4
fig, axs = plt.subplots(_n_plot, n_bands, figsize=(12, 10))

for i in range(_n_plot):
    axs_row = axs[i]
    flux, ivar, mask = dataset[i]
    for im, band, ax in zip(flux, bands, axs_row, strict=True):
        ax.imshow(im, origin="lower", cmap="gray")
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
        if i == 0:
            ax.set_title(band)

# %% [markdown]
# ## Define model

# %%
model = VAE(in_channels=n_bands, latent_dim=32, input_size=n_x).to(device)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model parameters: {num_params:,}")

# %%
x = torch.randn(4, n_bands, n_x, n_y).to(device)  # batch
print(f"Input shape: {x.shape}")

reconstruction, mu, logvar = model(x)
print(f"Reconstruction shape: {reconstruction.shape}")
print(f"Latent mean shape: {mu.shape}")
print(f"Latent log variance shape: {logvar.shape}")

# %% [markdown]
# ## Check pre-training outputs

# %%
model.eval()
_n_plot = 3

# Extract only flux (first element of tuple) for visualization
imgs = torch.stack([dataset[i][0] for i in range(4)]).to(device)
with torch.no_grad():
    recs, _, _ = model(imgs)

fig, axs = plt.subplots(_n_plot * 2, n_bands, figsize=(12, 16))
for i in range(_n_plot):
    axs_row_true = axs[2 * i]
    axs_row_reco = axs[2 * i + 1]

    axs_row_true[0].set_ylabel("True")
    axs_row_reco[0].set_ylabel("Reconstructed")

    for j, band in enumerate(bands):
        minmax = {
            "vmin": imgs[i, j].cpu().min(),
            "vmax": imgs[i, j].cpu().max(),
        }
        axs_row_true[j].imshow(
            imgs[i, j].cpu(), origin="lower", cmap="gray", **minmax
        )
        axs_row_reco[j].imshow(
            recs[i, j].cpu(), origin="lower", cmap="gray", **minmax
        )

        axs_row_true[j].xaxis.set_ticks([])
        axs_row_true[j].yaxis.set_ticks([])
        axs_row_reco[j].xaxis.set_ticks([])
        axs_row_reco[j].yaxis.set_ticks([])
        axs_row_true[j].set_title(band)

# %% [markdown]
# ## Train

# %%
if __name__ == "__main__":
    from torch.utils.data import DataLoader, random_split
    from galgenai.training import train

    BATCH_SIZE = 256
    NUM_WORKERS = 4
    NUM_EPOCHS = 10

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.75)

    dataset_train, dataset_test = random_split(dataset, [0.8, 0.2])

    train_loader = DataLoader(
        dataset_train,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=True,
    )
    test_loader = DataLoader(
        dataset_test,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=False,
    )

    train(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=NUM_EPOCHS,
        reconstruction_loss_fn="masked_weighted_mse",
        beta=1.0,
    )

    # %% [markdown]
    # ## Check post-training outputs

    # %%
    model.eval()
    _n_plot = 3

    # Extract only flux (first element of tuple) for visualization
    imgs = torch.stack([dataset_test[i][0] for i in range(4)]).to(device)
    with torch.no_grad():
        recs, _, _ = model(imgs)

    fig, axs = plt.subplots(_n_plot * 2, n_bands, figsize=(12, 16))
    for i in range(_n_plot):
        axs_row_true = axs[2 * i]
        axs_row_reco = axs[2 * i + 1]

        axs_row_true[0].set_ylabel("True")
        axs_row_reco[0].set_ylabel("Reconstructed")

        for j, band in enumerate(bands):
            minmax = {
                "vmin": imgs[i, j].cpu().min(),
                "vmax": imgs[i, j].cpu().max(),
            }
            axs_row_true[j].imshow(
                imgs[i, j].cpu(), origin="lower", cmap="gray", **minmax
            )
            axs_row_reco[j].imshow(
                recs[i, j].cpu(), origin="lower", cmap="gray", **minmax
            )

            axs_row_true[j].xaxis.set_ticks([])
            axs_row_true[j].yaxis.set_ticks([])
            axs_row_reco[j].xaxis.set_ticks([])
            axs_row_reco[j].yaxis.set_ticks([])
            axs_row_true[j].set_title(band)

# %%
# Debug: check max values
# flux, ivar, mask = dataset[0]
# flux.max()

# %%
