"""Dataset construction helpers.

Provides:
    * ``build_transforms`` - standard train / eval transforms at 224x224
      with ImageNet normalization.
    * ``build_datasets`` - wraps ``data/labeled/`` into ImageFolder datasets
      and returns stratified train/val/test splits.
    * ``build_loaders`` - thin wrapper around ``torch.utils.data.DataLoader``.
    * ``denormalize`` - undo ImageNet normalization for visualization.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMG_SIZE = 224


def build_transforms(train: bool = True) -> transforms.Compose:
    """Return a torchvision transform pipeline.

    Train pipeline: horizontal flip, small rotation, colour jitter.
    Eval pipeline: deterministic resize + center crop + normalize.
    """
    if train:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
            transforms.RandomCrop(IMG_SIZE),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def _stratified_indices(
    targets: List[int],
    ratios: Tuple[float, float, float],
    seed: int,
) -> Tuple[List[int], List[int], List[int]]:
    """Return train/val/test indices, preserving class proportions."""
    assert abs(sum(ratios) - 1.0) < 1e-6, "ratios must sum to 1.0"
    rng = torch.Generator().manual_seed(seed)

    by_class: Dict[int, List[int]] = defaultdict(list)
    for idx, cls in enumerate(targets):
        by_class[cls].append(idx)

    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []

    for cls, idxs in by_class.items():
        idxs_t = torch.tensor(idxs)
        perm = idxs_t[torch.randperm(len(idxs_t), generator=rng)].tolist()
        n = len(perm)
        n_train = int(round(n * ratios[0]))
        n_val = int(round(n * ratios[1]))
        train_idx.extend(perm[:n_train])
        val_idx.extend(perm[n_train:n_train + n_val])
        test_idx.extend(perm[n_train + n_val:])

    return train_idx, val_idx, test_idx


def build_datasets(
    root: str | Path = "data/labeled",
    ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
) -> Tuple[Subset, Subset, Subset, List[str]]:
    """Build stratified train/val/test datasets from ``root``.

    Train subset uses augmentation transforms, val/test use deterministic
    eval transforms.

    Returns
    -------
    train_ds, val_ds, test_ds, class_names
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(
            f"Dataset root not found: {root.resolve()}\n"
        )

    base_train = datasets.ImageFolder(root=str(root), transform=build_transforms(train=True))
    base_eval = datasets.ImageFolder(root=str(root), transform=build_transforms(train=False))
    class_names = base_train.classes

    train_idx, val_idx, test_idx = _stratified_indices(base_train.targets, ratios, seed)

    train_ds = Subset(base_train, train_idx)
    val_ds = Subset(base_eval, val_idx)
    test_ds = Subset(base_eval, test_idx)

    return train_ds, val_ds, test_ds, class_names


def build_loaders(
    train_ds,
    val_ds,
    test_ds,
    batch_size: int = 32,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader, test_loader


def denormalize(t: torch.Tensor) -> torch.Tensor:
    """Invert the ImageNet normalization so images can be plotted.

    Accepts a tensor of shape ``(C, H, W)`` or ``(B, C, H, W)``.
    """
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    x = t.detach().cpu()
    if x.ndim == 3:
        x = x.unsqueeze(0)
    x = x * std + mean
    return x.clamp(0, 1)
