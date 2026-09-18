"""
PyTorch Dataset + transforms + the weighted sampler that handles class
imbalance at the batch-sampling level (used together with the loss
weighting in losses.py — see README).
"""
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms


class CellCropDataset(Dataset):
    def __init__(self, root_dir: str, classes: list, transform=None):
        self.root_dir = Path(root_dir)
        self.classes = classes
        self.transform = transform
        self.samples = []  # list of (path, class_idx)

        for idx, cls in enumerate(classes):
            cls_dir = self.root_dir / cls
            if not cls_dir.exists():
                continue
            for img_path in cls_dir.glob("*.png"):
                self.samples.append((img_path, idx))

        if not self.samples:
            raise RuntimeError(f"No images found under {root_dir} — run crop_dataset.py first.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

    def class_counts(self) -> list:
        counts = [0] * len(self.classes)
        for _, label in self.samples:
            counts[label] += 1
        return counts


def build_transforms(cfg: dict, train: bool):
    mean, std = cfg["imagenet_mean"], cfg["imagenet_std"]
    size = cfg["image_size"]

    if not train:
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    aug = cfg["augment"]
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomRotation(aug["rotation_degrees"]),
        transforms.RandomHorizontalFlip(p=aug["horizontal_flip_prob"]),
        transforms.ColorJitter(
            brightness=aug["brightness_jitter"],
            contrast=aug["contrast_jitter"],
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def build_weighted_sampler(dataset: CellCropDataset) -> WeightedRandomSampler:
    """
    Gives each sample a weight inversely proportional to its class frequency,
    so minority classes (WBC, Platelet) are drawn roughly as often as the
    majority class (RBC) within each training batch. Use this ALONGSIDE the
    class-weighted / focal loss in losses.py, not instead of it.
    """
    counts = dataset.class_counts()
    class_weights = [1.0 / c if c > 0 else 0.0 for c in counts]
    sample_weights = [class_weights[label] for _, label in dataset.samples]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


def build_dataloaders(cfg: dict):
    classes = cfg["classes"]
    batch_size = cfg["batch_size"]
    processed_dir = cfg["processed_dir"]

    train_ds = CellCropDataset(f"{processed_dir}/train", classes, build_transforms(cfg, train=True))
    val_ds = CellCropDataset(f"{processed_dir}/val", classes, build_transforms(cfg, train=False))
    test_ds = CellCropDataset(f"{processed_dir}/test", classes, build_transforms(cfg, train=False))

    sampler = build_weighted_sampler(train_ds)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader, train_ds.class_counts()
