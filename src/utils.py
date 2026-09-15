"""Shared helpers used across the project."""
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_images_per_class(split_dir: str, classes: list) -> dict:
    """Count how many crop images sit in each class folder of a split."""
    counts = {}
    for cls in classes:
        cls_dir = Path(split_dir) / cls
        counts[cls] = len(list(cls_dir.glob("*.png"))) if cls_dir.exists() else 0
    return counts


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
