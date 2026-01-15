import numpy as np
from typing import Tuple

import torch
from torch.utils.data import DataLoader, random_split
from datasets import Dataset


class HSCDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        hf_dataset,
        nx: int,
        mins: torch.Tensor,
        maxs: torch.Tensor,
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


def get_dataset_and_loaders(
    dataset_raw: Dataset,
    nx: int = 64,
    minmax: Tuple[float, float] = (-2.0, 99.0),
    split: float = 0.8,
    batch_size: int = 128,
    num_workers: int = 8,
) -> Tuple[HSCDataset, DataLoader, DataLoader]:
    dataset_raw = dataset_raw.select_columns(["image"]).with_format("torch")

    n_gals = len(dataset_raw)
    bands = dataset_raw[0]["image"]["band"]

    # min/max
    mins = torch.Tensor(minmax[0] * np.ones(len(bands)))
    maxs = torch.Tensor(minmax[1] * np.ones(len(bands)))

    dataset = HSCDataset(dataset_raw, nx=nx, mins=mins, maxs=maxs)
    n_bands, n_x, n_y = dataset[0][0].shape  # First element of tuple is flux
    print(f"Images dimension: {n_bands}*{n_x}*{n_y} ({n_gals} galaxies)")

    dataset_train, dataset_test = random_split(dataset, [split, 1 - split])

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

    return dataset, train_loader, test_loader
