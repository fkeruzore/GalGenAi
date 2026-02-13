"""to load config"""

import os
import yaml
from pathlib import Path
from typing import Dict, Optional

def load_config(config_path: Optional[str] = None) -> Dict:
    """
    Load configuration from YAML file.

    Parameters
    ----------
    config_path : str, optional
        Explicit path to config file. If None, searches default locations.

    Returns
    -------
    dict
        Configuration dictionary. Returns empty dict if no config found.
    """
    if config_path  is None:
        config_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "galgenai_config.yaml",
        )
        print(config_path)
    config_path = Path(config_path)

    if config_path.exists() and config_path.is_file():
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                if config is None:
                    config = {}
                return config
        except Exception as e:
            raise ValueError(f"Failed to load config from {config_path}: {e}")

    return {}
