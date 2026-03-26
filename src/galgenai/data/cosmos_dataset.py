from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
from astropy.io import fits
from datasets import Dataset
from torch.utils.data import DataLoader, random_split

from galgenai.data.hsc import HSCDataset
from galgenai.data.latent import precompute_latents, LatentDataset

#TODO: CHECK catalog what the sentinel values are
def load_fits_dataset(
    data_dir,
    metadata_file="metadata.csv",
    format="torch",
    filter_invalid_mags=True,
    mag_sentinel=999.0,
    mag_cols=None,
    filter_invalid_redshift=True,
    redshift_sentinel=-99.0,
    redshift_col=None,
):
    """
    Load a FITS galaxy dataset produced by generate_fits_dataset.py.

    All galaxies are stored together under ``data_dir/images/`` with
    metadata CSV file(s). The returned dataset is then split in make_loaders().

    The returned dataset has one column image (nested dict with keys: 
    flux, ivar, mask, band) plus all metadata columns from
    the CSV. This layout matches the HSC dataset format so that
    hsc.HSCDataset can be used directly.

    If IVAR is absent from a FITS file, a ones array is used (uniform
    weighting). If MASK is absent, a zeros array is used (no masking).


    Parameters:
    -----------
    data_dir : str or Path
        Root directory of the dataset (contains ``images/`` and metadata CSV).
    metadata_file : str or dict
        If str: Name of a single metadata CSV file. Default "metadata.csv".
        If dict: Dictionary mapping split names to metadata filenames.
                 Returns a dictionary of datasets.
    format : str
        Output format for arrays. Options: "torch" (default), "numpy", "tensorflow",
        or None (Python lists). Default "torch".
    filter_invalid_mags : bool
        If True, filter out galaxies where any magnitude column equals the sentinel
        value. Default True.
    mag_sentinel : float
        Sentinel value indicating missing magnitude (default: 999.0). Rows with any
        magnitude exactly equal to this value will be filtered out.
    mag_cols : list of str or None
        List of magnitude column names to check (e.g., ['mag_g', 'mag_r', 'mag_i']).
        If None, auto-detects all columns starting with 'mag_'. Default None.
    filter_invalid_redshift : bool
        If True, filter out galaxies where redshift equals the sentinel value. Default True.
    redshift_sentinel : float
        Sentinel value indicating missing redshift (default: -99.0).
    redshift_col : str
        Name of the redshift column in metadata. Required if filter_invalid_redshift is True.

    Returns:
    --------
    datasets.Dataset or dict
        HuggingFace Dataset with PyTorch tensors (default format="torch").
        If metadata_file is str: Single dataset loading from the specified file.
        If metadata_file is dict: Dictionary mapping split names to datasets.
    """
    data_dir = Path(data_dir)

    # Handle dictionary of metadata files: if they are splited into train, test, val
    if isinstance(metadata_file, dict):
        result = {}
        for split_name, meta_file in metadata_file.items():
            result[split_name] = load_fits_dataset(
                data_dir,
                metadata_file=meta_file,
                format=format,
                filter_invalid_mags=filter_invalid_mags,
                mag_sentinel=mag_sentinel,
                mag_cols=mag_cols,
            )
        return result

    # Single metadata file
    images_path = data_dir / "images"
    metadata = pd.read_csv(data_dir / metadata_file)

    # Filter out galaxies with invalid magnitudes
    if filter_invalid_mags:
        initial_count = len(metadata)

        # Determine which columns to check
        if mag_cols is None:
            # Auto-detect magnitude columns
            raise ValueError("Mag col names should be provided to apply cuts")

        if mag_cols:
            # Create mask for valid magnitudes
            mask = np.ones(len(metadata), dtype=bool)
            for mag_col in mag_cols:
                if mag_col in metadata.columns:
                    # Filter out rows where magnitude equals sentinel value
                    col_mask = metadata[mag_col] <= mag_sentinel
                    mask &= col_mask

            metadata = metadata[mask].reset_index(drop=True)
            n_removed = initial_count - len(metadata)
            if n_removed > 0:
                print(f"Filtered {n_removed} galaxies with invalid magnitudes (sentinel={mag_sentinel})")
                print(f"Remaining: {len(metadata)} galaxies")

    if filter_invalid_redshift:

        if redshift_col is None:
            raise ValueError("Redshift col name should be provided to apply cuts")

        if redshift_col in metadata.columns:
            col_mask = metadata[redshift_col] != redshift_sentinel
            metadata = metadata[col_mask].reset_index(drop=True)
            n_removed = len(col_mask) - len(metadata)
            if n_removed > 0:
                print(f"Filtered {n_removed} galaxies with invalid redshift (sentinel={redshift_sentinel})")
                print(f"Remaining: {len(metadata)} galaxies")
        else:
            raise ValueError(f"Warning: Redshift column '{redshift_col}' not found in metadata. Skipping redshift filtering.")

    def make_generator(rows, images_path):
        def generator():
            for _, row in rows.iterrows():
                with fits.open(images_path / row["filename"]) as hdul:
                    image = hdul["IMAGE"].data.astype("float32")
                    n_bands = image.shape[0]
                    bands = [
                        hdul["IMAGE"].header.get(f"BAND{i}", f"band{i}")
                        for i in range(n_bands)
                    ]

                    if "IVAR" in hdul:
                        ivar = hdul["IVAR"].data.astype("float32")
                    else:
                        ivar = np.ones_like(image)

                    if "MASK" in hdul:
                        mask = hdul["MASK"].data.astype(np.int32)
                    else:
                        mask = np.zeros(image.shape, dtype=np.int32)

                result = {
                    "image": {
                        "flux": image,
                        "ivar": ivar,
                        "mask": mask,
                        "band": bands,
                    },
                    **row.to_dict(),
                }
                yield result
        return generator

    gen = make_generator(metadata, images_path)
    dataset = Dataset.from_generator(gen)

    # Apply format if specified
    if format is not None:
        dataset = dataset.with_format(format)

    return dataset


def make_loaders(
    dataset_raw: Dataset,
    nx: int,
    batch_size: int,
    num_workers: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    random_seed: int = 42,
    image_norm_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    return_aux_data: bool = True,
    condition_cols: Optional[list] = None,
    conditional_norm_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    shuffle: bool = True,
    split_datasets: Optional[tuple] = None,
    return_splits: bool = False,
):
    """Build train/val/test DataLoaders from a raw dataset.

    Takes a raw dataset and splits it into train/val/test, wraps each split in HSCDataset,
    and creates dataloaders. Similar to get_dataset_and_loaders in hsc.py.

    Supports multiple data modes:
    1. VAE training: return_aux_data=True, no conditioning, shuffle=True
       Returns (flux, ivar, mask) tuples for training.

    2. Latent precomputation: return_aux_data=False, with conditioning
       Returns (flux, condition) tuples for encoding.

    3. CNF training on raw images: return_aux_data=False, with conditioning, shuffle=True
       Returns (flux, condition) tuples for direct CNF training (rare).

    Parameters:
    -----------
    dataset_raw: Raw HuggingFace Dataset to be split
    nx: Side length of center-cropped output patch
    batch_size: Batch size for DataLoaders
    num_workers: Number of DataLoader worker processes
    train_ratio: Fraction of data for training split. Default 0.8.
    val_ratio: Fraction of data for validation split. Default 0.1.
    random_seed: Random seed for reproducible splits. Default 42.
    image_norm_fn: Optional image normalization function. Create externally
        using get_image_norm_fn() [see normalization.py] and pass here.
    return_aux_data: If True, return (flux, ivar, mask). If False with conditioning,
        return (flux, condition). Default True.
    condition_cols: Optional list of column names for conditioning variables.
        If provided, enables conditioning mode.
    conditional_norm_fn: Optional function to normalize conditioning variables.
        Create externally using get_conditional_norm_fn() and pass here.
        Required if condition_cols is provided. [see normalization.py]
    shuffle: Whether to shuffle training data. Default True.
    split_datasets: Optional tuple of (train_ds, val_ds, test_ds) from a previous call.
        If provided, uses these splits instead of creating new ones.
    return_splits: If True, return the split datasets tuple for reuse. Default False.

    Returns:
    --------
    If return_splits=False: (train_loader, val_loader, test_loader)
        where test_loader is None if train_ratio + val_ratio == 1.0
    If return_splits=True: (train_loader, val_loader, test_loader, split_datasets)
        where split_datasets is a tuple (train_ds, val_ds, test_ds) that can be
        passed to subsequent calls to reuse the same split with different normalization.
    """
    # Validate conditioning parameters
    if condition_cols is not None and conditional_norm_fn is None:
        raise ValueError(
            "conditional_norm_fn must be provided when using condition_cols. "
            "Use get_conditional_norm_fn() to create it."
        )

    # Create HSCDataset wrapper for the full dataset
    full_dataset = HSCDataset(
        dataset_raw,
        nx=nx,
        image_norm_fn=image_norm_fn,
        return_aux_data=return_aux_data,
        condition_cols=condition_cols or [],
        conditional_norm_fn=conditional_norm_fn,
    )

    # Split dataset using random_split or reuse existing split
    if split_datasets is not None:
        # Reuse existing split by extracting indices and creating new Subsets
        old_train, old_val, old_test = split_datasets
        train_ds = torch.utils.data.Subset(full_dataset, old_train.indices)
        val_ds = torch.utils.data.Subset(full_dataset, old_val.indices)
        test_ds = torch.utils.data.Subset(full_dataset, old_test.indices)
    else:
        # Create new split
        test_ratio = 1.0 - train_ratio - val_ratio
        train_ds, val_ds, test_ds = random_split(
            full_dataset,
            [train_ratio, val_ratio, test_ratio],
            generator=torch.Generator().manual_seed(random_seed)
        )

    # Create dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,  # Reuse worker processes across epochs
        prefetch_factor=num_workers * 4 if num_workers > 0 else None,  # Prefetch batches
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,  # Validation is never shuffled
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=num_workers * 4 if num_workers > 0 else None,
    )

    # Create test loader if test set exists
    test_ratio = 1.0 - train_ratio - val_ratio
    if test_ratio > 0:
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,  # Test is never shuffled
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=num_workers * 4 if num_workers > 0 else None,
        )
    else:
        test_loader = None

    if return_splits:
        return train_loader, val_loader, test_loader, (train_ds, val_ds, test_ds)
    else:
        return train_loader, val_loader, test_loader
