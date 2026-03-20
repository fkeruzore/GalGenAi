"""Normalization and denormalization utilities for COSMOS galaxy images and magnitudes."""
from functools import partial

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import yaml
from torch.utils.data import Subset



# ---------------------------------------------------------------------------
# Image normalization
# ---------------------------------------------------------------------------


@dataclass
class ASinhNormStats:
    """Per-channel statistics used to normalise COSMOS images.

    The normalization pipeline is:
        1. arcsinh stretch : arcsinh(x / scale)
        2. min-max norm    : (stretched - min) / (max - min)
    """

    min: torch.Tensor  # shape (C,) # one entry per-band / col
    max: torch.Tensor  # shape (C,)
    scale: torch.Tensor  # shape (C,)

    def to(self, device: torch.device | str) -> "ASinhNormStats":
        """Move all tensors to the specified device."""
        return ASinhNormStats(
            min=self.min.to(device),
            max=self.max.to(device),
            scale=self.scale.to(device),
        )


@dataclass
class LinearNormStats:
    """Per-channel min/max statistics for min-max normalization of images.

    The normalization is: (x - min) / (max - min)
    """

    min: torch.Tensor  # shape (C,)
    max: torch.Tensor  # shape (C,)

    def to(self, device: torch.device | str) -> "LinearNormStats":
        """Move all tensors to the specified device."""
        return LinearNormStats(
            min=self.min.to(device),
            max=self.max.to(device),
        )


def compute_linear_norm_stats(
    dataset: torch.utils.data.Dataset,
    n_samples: int | None = 2000,
    seed: int = 0,
) -> LinearNormStats:
    """Estimate per-channel min and max from a subset of the dataset.

    Parameters:
    -----------
    dataset: A torch Dataset whose items are image tensors shape: (C, H, W)
        or ``(image, ...)`` tuples where the first element is the image.
    n_samples: Number of images to sample.
        Default is set to None which uses the full dataset.
    seed: Random seed for reproducible sub-sampling

    Returns:
    --------
    LinearNormStats with min and max tensors of shape (C,)
    """
    n = len(dataset)
    rng = np.random.default_rng(seed)
    if n_samples is None or n_samples >= n:
        indices = np.arange(n)
    else:
        indices = rng.choice(n, size=n_samples, replace=False)

    subset = Subset(dataset, indices.tolist())

    first = subset[0][0] if isinstance(subset[0], tuple) else subset[0]
    c = first.shape[0]
    running_min = torch.full((c,), float("inf"))
    running_max = torch.full((c,), float("-inf"))

    for idx in range(len(subset)):
        item = subset[idx]
        img = item[0] if isinstance(item, tuple) else item  # (C, H, W)
        img_min = img.flatten(1).min(dim=1).values           # (C,)
        img_max = img.flatten(1).max(dim=1).values           # (C,)
        running_min = torch.minimum(running_min, img_min)
        running_max = torch.maximum(running_max, img_max)

    # Round to 2 decimal places
    running_min = running_min.round(decimals=2)
    running_max = running_max.round(decimals=2)

    return LinearNormStats(min=running_min, max=running_max)


def arcsinh_stretch(x: torch.Tensor, scale: torch.Tensor | float) -> torch.Tensor:
    """Apply an arcsinh stretch to compress the dynamic range.

    Parameters:
    -----------
    x: Input tensor of shape (C, H, W)
    scale: Scale parameter, either a scalar or tensor of shape (C,)

    Returns:
    --------
    Stretched tensor of same shape as input
    """
    if isinstance(scale, torch.Tensor):
        # Per-channel scale: (C,) -> (C, 1, 1)
        scale = scale[:, None, None]
    return torch.arcsinh(x / scale)


def compute_arcsinh_norm_stats(
    dataset: torch.utils.data.Dataset,
    n_samples: int | None = 2000,
    seed: int = 0,
    scale_quantile=.95
) -> ASinhNormStats:
    """Estimate per-channel min, max, and scale from a subset of the dataset.

    The scale parameter is computed per-band as the mean of the top `scale_quantile` 
    percentile flux values for that band to reduce the influence of background noise. 

    Min and max are computed after the arcsinh stretch for min-max normalization
    """
    n = len(dataset)
    rng = np.random.default_rng(seed)
    if n_samples is None or n_samples >= n:
        indices = np.arange(n)
    else:
        indices = rng.choice(n, size=n_samples, replace=False)

    subset = Subset(dataset, indices.tolist())

    # 1. compute scale for arcsinh
    # Collect all flux values per band to compute percentile
    first = subset[0][0] if isinstance(subset[0], tuple) else subset[0]
    c = first.shape[0]
    flux_values = [[] for _ in range(c)]

    for idx in range(len(subset)):
        item = subset[idx]
        img = item[0] if isinstance(item, tuple) else item  # (C, H, W)
        # Collect per-channel flux values
        for ch in range(c):
            flux_values[ch].append(img[ch].flatten())

    # Compute scale as mean of top 90 percentile values per channel
    scale = torch.zeros(c)
    for ch in range(c):
        ch_values = torch.cat(flux_values[ch])
        percentile_10 = torch.quantile(ch_values, scale_quantile)
        top_90_values = ch_values[ch_values >= percentile_10]
        scale[ch] = top_90_values.mean()

    scale = scale.round(decimals=2) # TODO: should we remove decimals here?

    # 2. compute min and max of arcsinh-stretched values
    running_min = torch.full((c,), float("inf"))
    running_max = torch.full((c,), float("-inf"))

    for idx in range(len(subset)):
        item = subset[idx]
        img = item[0] if isinstance(item, tuple) else item
        img = arcsinh_stretch(img, scale)           # (C, H, W)
        img_min = img.flatten(1).min(dim=1).values  # (C,)
        img_max = img.flatten(1).max(dim=1).values  # (C,)
        running_min = torch.minimum(running_min, img_min)
        running_max = torch.maximum(running_max, img_max)

    min_val = running_min
    max_val = running_max

    # Round to 2 decimal places
    min_val = min_val.round(decimals=2)
    max_val = max_val.round(decimals=2)

    return ASinhNormStats(min=min_val, max=max_val, scale=scale)


def arcsinh_norm(x: torch.Tensor, stats: ASinhNormStats) -> torch.Tensor:
    """Normalise raw flux using arcsinh stretch then min-max normalization"""
    stats = stats.to(x.device)
    x = arcsinh_stretch(x, stats.scale)
    min_val = stats.min[:, None, None]
    max_val = stats.max[:, None, None]
    return (x - min_val) / (max_val - min_val)


def arcsinh_denorm(x: torch.Tensor, stats: ASinhNormStats) -> torch.Tensor:
    """Invert ``arcsinh_norm`` to recover approximate raw flux values"""
    stats = stats.to(x.device)
    min_val = stats.min[:, None, None]
    max_val = stats.max[:, None, None]
    scale = stats.scale[:, None, None]
    x = x * (max_val - min_val) + min_val
    return torch.sinh(x) * scale


def linear_norm(x: torch.Tensor, stats: LinearNormStats) -> torch.Tensor:
    """Normalise raw flux using min-max scaling: (x - min) / (max - min)."""
    stats = stats.to(x.device)
    min_ = stats.min[:, None, None]
    max_ = stats.max[:, None, None]
    return (x - min_) / (max_ - min_)


def linear_denorm(x: torch.Tensor, stats: LinearNormStats) -> torch.Tensor:
    """Invert ``linear_normalize`` to recover raw flux values."""
    stats = stats.to(x.device)
    min_ = stats.min[:, None, None]
    max_ = stats.max[:, None, None]
    return x * (max_ - min_) + min_


def load_stats_from_config(
    norm_config: dict,
    image: bool = True,
    norm_type: str = "arcsinh"
) -> LinearNormStats | ASinhNormStats | ConditionalStats:
    """Load normalization stats from config dictionary.

    Parameters:
    -----------
    norm_config: Config dictionary containing normalization stats.
        For image stats: expects keys `arcsinh` or `linear` with nested stats.
        For conditional stats: expects keys `cols`, `min`, `max` directly.
    image: If True, load image normalization stats. If False, load conditional stats.
    norm_type: Type of image normalization ("linear" or "arcsinh").
        Only used when image=True.

    Returns:
    --------
    LinearNormStats, ASinhNormStats, or ConditionalStats depending on parameters.
    """
    if image:
        # Load image normalization stats
        if norm_type == "arcsinh":
            if "arcsinh" not in norm_config:
                raise ValueError("norm_config must contain 'arcsinh' section")
            stats_dict = norm_config["arcsinh"]
            if not all(k in stats_dict for k in ["min", "max", "scale"]):
                raise ValueError(
                    "arcsinh normalization must contain 'min', 'max', and 'scale' fields."
                )
            return ASinhNormStats(
                min=torch.tensor(stats_dict["min"], dtype=torch.float32),
                max=torch.tensor(stats_dict["max"], dtype=torch.float32),
                scale=torch.tensor(stats_dict["scale"], dtype=torch.float32),
            )
        elif norm_type == "linear":
            if "linear" not in norm_config:
                raise ValueError("norm_config must contain 'linear' section")
            stats_dict = norm_config["linear"]
            if not all(k in stats_dict for k in ["min", "max"]):
                raise ValueError(
                    "linear normalization must contain 'min' and 'max' fields."
                )
            return LinearNormStats(
                min=torch.tensor(stats_dict["min"], dtype=torch.float32),
                max=torch.tensor(stats_dict["max"], dtype=torch.float32),
            )
        else:
            raise ValueError(
                f"Unknown norm_type: {norm_type}. Must be 'linear' or 'arcsinh'."
            )
    else:
        # Load conditional normalization stats
        if not all(k in norm_config for k in ["cols", "min", "max"]):
            raise ValueError(
                "Conditional normalization must contain 'cols', 'min', and 'max' fields."
            )
        return ConditionalStats(
            cols=norm_config["cols"],
            min=torch.tensor(norm_config["min"], dtype=torch.float32),
            max=torch.tensor(norm_config["max"], dtype=torch.float32),
        )


def save_image_norm_stats(
    stats: LinearNormStats | ASinhNormStats,
    save_path: Path | str,
) -> None:
    """Save image normalization statistics to disk as YAML.

    Parameters:
    -----------
    stats: The normalization statistics to save (LinearNormStats or ASinhNormStats).
    save_path: Path where statistics will be saved as a .yaml file.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(stats, LinearNormStats):
        data = {
            "type": "linear",
            "min": stats.min.tolist(),
            "max": stats.max.tolist(),
        }
    elif isinstance(stats, ASinhNormStats):
        data = {
            "type": "arcsinh",
            "min": stats.min.tolist(),
            "max": stats.max.tolist(),
            "scale": stats.scale.tolist() if isinstance(stats.scale, torch.Tensor) else float(stats.scale),
        }
    else:
        raise TypeError(
            f"stats must be LinearNormStats or ASinhNormStats, got {type(stats).__name__}"
        )

    with open(save_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def get_image_norm_fn(
    img_norm_type: str = "linear",
    config: Optional[dict] = None,
    stats: Optional[LinearNormStats | ASinhNormStats] = None,
    stats_path: Optional[Path | str] = None,
    compute_stats: Optional[torch.utils.data.Dataset] = None,
    return_denorm: bool = False,
) -> tuple[Callable[[torch.Tensor], torch.Tensor], LinearNormStats | ASinhNormStats] | tuple[Callable[[torch.Tensor], torch.Tensor], Callable[[torch.Tensor], torch.Tensor], LinearNormStats | ASinhNormStats]:
    """Create a normalization function based on the specified type.

    Priority order:
        1. config: Load statistics from config dictionary
        2. stats: Use pre-loaded statistics object
        3. stats_path: Load statistics from file
        4. compute_stats: Compute statistics from dataset

    At least one of the four options must be provided.

    Parameters:
    -----------
    img_norm_type: Type of normalization. Options:
        - "linear": Min-max normalization (x - min) / (max - min)
        - "arcsinh": Arcsinh stretch + min-max normalization
    config: Optional config dictionary containing normalization stats.
        For arcsinh: expects {"arcsinh": {"min": [...], "max": [...], "scale": [...]}}
        For linear: expects {"linear": {"min": [...], "max": [...]}}
    stats: Optional pre-loaded normalization statistics object.
        - For "linear": expects a LinearNormStats instance
        - For "arcsinh": expects a ASinhNormStats instance
    stats_path: Optional path to load pre-computed normalization statistics (YAML file).
        - For "linear": expects a YAML file with keys "min" and "max"
        - For "arcsinh": expects a YAML file with keys "min", "max", and "scale"
    compute_stats: Optional dataset to compute statistics from.
        Stats will be computed on-the-fly from this dataset.
    return_denorm: If True, also return the denormalization function.

    Returns:
    --------
    If return_denorm is False:
        A tuple of (normalize_fn, norm_stats) where:
            - normalize_fn: Callable that normalizes images (C, H, W) -> (C, H, W)
            - norm_stats: The normalization stats object (LinearNormStats or ASinhNormStats)
    If return_denorm is True:
        A tuple of (normalize_fn, denormalize_fn, norm_stats) where:
            - normalize_fn: Callable that normalizes images (C, H, W) -> (C, H, W)
            - denormalize_fn: Callable that denormalizes images (C, H, W) -> (C, H, W)
            - norm_stats: The normalization stats object (LinearNormStats or ASinhNormStats)
    """
    # Check that at least one option is provided
    if config is None and stats is None and stats_path is None and compute_stats is None:
        raise ValueError(
            "Must provide at least one of: config, stats, stats_path, or compute_stats. "
            "Cannot create normalization function without normalization statistics."
        )

    if img_norm_type == "linear":
        # Priority 1: Load from config
        if config is not None:
            norm_stats = load_stats_from_config(config, image=True, norm_type="linear")
        # Priority 2: Use provided stats directly
        elif stats is not None:
            if not isinstance(stats, LinearNormStats):
                raise TypeError(
                    f"For norm_type='linear', stats must be a LinearNormStats instance, "
                    f"got {type(stats).__name__}"
                )
            norm_stats = stats
        # Priority 3: Load from path
        elif stats_path is not None:
            stats_path = Path(stats_path)
            if not stats_path.exists():
                raise FileNotFoundError(f"Stats file not found: {stats_path}")

            with open(stats_path, "r") as f:
                state = yaml.safe_load(f)
            norm_stats = LinearNormStats(
                min=torch.tensor(state["min"], dtype=torch.float32),
                max=torch.tensor(state["max"], dtype=torch.float32)
            )
        # Priority 3: Compute from dataset
        else:  # compute_stats is not None
            norm_stats = compute_linear_norm_stats(compute_stats)

        if return_denorm:
            return (
                partial(linear_norm, stats=norm_stats),
                partial(linear_denorm, stats=norm_stats),
                norm_stats
            )
        else:
            return partial(linear_norm, stats=norm_stats), norm_stats

    elif img_norm_type == "arcsinh":
        # Priority 1: Load from config
        if config is not None:
            norm_stats = load_stats_from_config(config, image=True, norm_type="arcsinh")
        # Priority 2: Use provided stats directly
        elif stats is not None:
            if not isinstance(stats, ASinhNormStats):
                raise TypeError(
                    f"For norm_type='arcsinh', stats must be a ASinhNormStats instance, "
                    f"got {type(stats).__name__}"
                )
            norm_stats = stats
        # Priority 3: Load from path
        elif stats_path is not None:
            stats_path = Path(stats_path)
            if not stats_path.exists():
                raise FileNotFoundError(f"Stats file not found: {stats_path}")

            with open(stats_path, "r") as f:
                state = yaml.safe_load(f)

            # Handle scale as either per-band (list) or global (float)
            scale_val = state.get("scale", 100.0)
            if isinstance(scale_val, list):
                scale = torch.tensor(scale_val, dtype=torch.float32)
            else:
                scale = torch.tensor([float(scale_val)], dtype=torch.float32)

            norm_stats = ASinhNormStats(
                min=torch.tensor(state["min"], dtype=torch.float32),
                max=torch.tensor(state["max"], dtype=torch.float32),
                scale=scale,
            )
        # Priority 3: Compute from dataset
        else:  # compute_stats is not None
            norm_stats = compute_arcsinh_norm_stats(compute_stats)

        if return_denorm:
            return (
                partial(arcsinh_norm, stats=norm_stats),
                partial(arcsinh_denorm, stats=norm_stats),
                norm_stats
            )
        else:
            return partial(arcsinh_norm, stats=norm_stats), norm_stats

    else:
        raise ValueError(
            f"Unknown norm_type: {img_norm_type}. Must be 'linear' or 'arcsinh'."
        )



# ---------------------------------------------------------------------------
# Conditional variable normalization
# ---------------------------------------------------------------------------


@dataclass
class ConditionalStats:
    """Per-column statistics for min-max normalising CNF conditioning variables.

    ``cols`` lists the HF dataset column names in order.
    ``min`` and ``max`` are 1-D tensors of shape ``(len(cols),)``.
    """

    cols: list
    min: torch.Tensor
    max: torch.Tensor

    def to(self, device: torch.device | str) -> "ConditionalStats":
        """Move all tensors to the specified device."""
        return ConditionalStats(
            cols=self.cols,
            min=self.min.to(device),
            max=self.max.to(device),
        )


def normalize_conditionals(x: torch.Tensor, stats: ConditionalStats) -> torch.Tensor:
    """Min-max normalise a conditioning tensor."""
    stats = stats.to(x.device)
    return (x - stats.min) / (stats.max - stats.min)


def denormalize_conditionals(x: torch.Tensor, stats: ConditionalStats) -> torch.Tensor:
    """Invert ``normalize_conditionals``."""
    stats = stats.to(x.device)
    return x * (stats.max - stats.min) + stats.min


def compute_conditional_stats(
    hf_dataset,
    cols: list,
    n_samples: int | None = None,
    seed: int = 0,
    filter_value: float | None = 999.0,
) -> ConditionalStats:
    """Estimate per-column min and max from a HF Dataset split.

    Missing values (NaN) and rows where any column equals filter_value are
    excluded from the statistics calculation.

    Parameters:
    -----------
    hf_dataset: A single HuggingFace Dataset split (e.g. dataset["train"]).
    cols: Column names to compute statistics for.
    n_samples: Number of galaxies to sample. ``None`` uses the full split.
    seed: Random seed for reproducible sub-sampling.
    filter_value: Sentinel value to filter out (e.g., 999.0 for missing magnitudes).
        If None, no filtering is applied. Default: 999.0.

    Returns:
    --------
    ``ConditionalStats`` with ``min`` and ``max`` tensors of shape ``(len(cols),)``.
    """
    n = len(hf_dataset)
    rng = np.random.default_rng(seed)
    if n_samples is None or n_samples >= n:
        indices = list(range(n))
    else:
        indices = rng.choice(n, size=n_samples, replace=False).tolist()

    rows = []
    for idx in indices:
        row = hf_dataset[int(idx)]
        values = np.array([float(row[c]) for c in cols], dtype=np.float32)
        # Filter out rows where any value equals sentinel value
        if filter_value is not None and np.any(values == filter_value):
            continue
        rows.append(values)

    if len(rows) == 0:
        raise ValueError(
            f"No valid rows found after filtering. Check filter_value={filter_value} "
            f"and ensure data contains valid values."
        )

    arr = np.stack(rows, axis=0)  # (N, len(cols))
    min_val = np.nanmin(arr, axis=0)
    max_val = np.nanmax(arr, axis=0)

    # Round to 2 decimal places
    min_val = min_val.round(decimals=2)
    max_val = max_val.round(decimals=2)

    return ConditionalStats(
        cols=list(cols),
        min=torch.as_tensor(min_val, dtype=torch.float32),
        max=torch.as_tensor(max_val, dtype=torch.float32),
    )

def save_conditional_stats(
    stats: ConditionalStats,
    save_path: Path | str,
) -> None:
    """Save conditional normalization statistics to disk as YAML.

    Parameters:
    -----------
    stats: The ConditionalStats object to save.
    save_path: Path where statistics will be saved as a .yaml file.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "cols": stats.cols,
        "min": stats.min.tolist(),
        "max": stats.max.tolist(),
    }

    with open(save_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def get_conditional_norm_fn(
    config: Optional[dict] = None,
    stats: Optional[ConditionalStats] = None,
    stats_path: Optional[Path | str] = None,
    compute_stats: Optional[tuple] = None,  # (dataset, cols, filter_value=999.0)
    return_denorm: bool = False,
) -> tuple[Callable[[torch.Tensor], torch.Tensor], ConditionalStats] | tuple[Callable[[torch.Tensor], torch.Tensor], Callable[[torch.Tensor], torch.Tensor], ConditionalStats]:
    """Create a conditional variable normalization function.

    Priority order:
        1. config: Load ConditionalStats from config dictionary
        2. stats: Use pre-loaded ConditionalStats object
        3. stats_path: Load ConditionalStats from file
        4. compute_stats: Compute ConditionalStats from dataset

    At least one of the four options must be provided.

    Parameters:
    -----------
    config: Optional config dictionary containing conditional stats.
        Expects keys "cols", "min", "max".
    stats: Optional pre-loaded ConditionalStats object.
    stats_path: Optional path to load ConditionalStats from YAML file.
        Expects a YAML file with keys "cols", "min", "max".
    compute_stats: Optional tuple of (dataset, cols) or (dataset, cols, filter_value).
        dataset is a HuggingFace Dataset, cols is a list of column names.
        filter_value is optional sentinel value to filter out (default: 999.0).
    return_denorm: If True, also return the denormalization function.

    Returns:
    --------
    If return_denorm is False:
        A tuple of (normalize_fn, cond_stats) where:
            - normalize_fn: Callable that normalizes conditioning variables
            - cond_stats: The ConditionalStats object used for normalization
    If return_denorm is True:
        A tuple of (normalize_fn, denormalize_fn, cond_stats) where:
            - normalize_fn: Callable that normalizes conditioning variables
            - denormalize_fn: Callable that denormalizes conditioning variables
            - cond_stats: The ConditionalStats object used for normalization

    Raises:
    -------
    ValueError: If none of stats, stats_path, or compute_stats are provided.
    FileNotFoundError: If stats_path is provided but file doesn't exist.
    """
    # Check that at least one option is provided
    if config is None and stats is None and stats_path is None and compute_stats is None:
        raise ValueError(
            "Must provide at least one of: config, stats, stats_path, or compute_stats. "
            "Cannot create conditional normalization function without statistics."
        )

    # Priority 1: Load from config
    if config is not None:
        cond_stats = load_stats_from_config(config, image=False)
    # Priority 2: Use provided stats directly
    elif stats is not None:
        cond_stats = stats
    # Priority 3: Load from path
    elif stats_path is not None:
        stats_path = Path(stats_path)
        if not stats_path.exists():
            raise FileNotFoundError(f"Stats file not found: {stats_path}")

        with open(stats_path, "r") as f:
            state = yaml.safe_load(f)
        cond_stats = ConditionalStats(
            cols=state["cols"],
            min=torch.tensor(state["min"], dtype=torch.float32),
            max=torch.tensor(state["max"], dtype=torch.float32),
        )
    # Priority 3: Compute from dataset
    else:  # compute_stats is not None
        dataset, cols = compute_stats
        cond_stats = compute_conditional_stats(dataset, cols=cols)

    if return_denorm:
        return (
            partial(normalize_conditionals, stats=cond_stats),
            partial(denormalize_conditionals, stats=cond_stats),
            cond_stats
        )
    else:
        return partial(normalize_conditionals, stats=cond_stats), cond_stats
