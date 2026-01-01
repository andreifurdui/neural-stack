"""Data loading utilities for common datasets.

This module provides lightweight wrappers for loading public datasets
with appropriate transforms. Designed for easy integration with the
Trainer class.

Example:
    from neural_stack.training import DataConfig, build_dataloaders

    config = DataConfig(dataset="cifar10", batch_size=128)
    train_loader, val_loader = build_dataloaders(config)
"""

from pathlib import Path
from typing import Tuple, Optional

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100

from neural_stack.training.config import DataConfig


# =============================================================================
# Normalization Statistics
# =============================================================================

# Pre-computed normalization statistics for common datasets
DATASET_STATS = {
    "cifar10": {
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2023, 0.1994, 0.2010),
    },
    "cifar100": {
        "mean": (0.5071, 0.4867, 0.4408),
        "std": (0.2675, 0.2565, 0.2761),
    }
}


# =============================================================================
# Transform Builders
# =============================================================================

def build_cifar_transforms(
    config: DataConfig,
    train: bool = True,
) -> transforms.Compose:
    """Build transforms for CIFAR datasets.

    Args:
        config: Data configuration.
        train: Whether this is for training (applies augmentation).

    Returns:
        Composed transform pipeline.
    """
    stats = DATASET_STATS.get(config.dataset, DATASET_STATS["cifar10"])
    mean, std = stats["mean"], stats["std"]

    if train:
        transform_list = []

        if config.random_horizontal_flip:
            transform_list.append(transforms.RandomHorizontalFlip())

        if config.random_crop:
            transform_list.append(transforms.RandomResizedCrop(
                (32, 32),
                scale=config.crop_scale,
                ratio=config.crop_ratio,
            ))
        
        if config.rand_aug:
            transform_list.append(transforms.RandAugment(
                num_ops=config.rand_aug_num_ops,
                magnitude=config.rand_aug_magnitude
            ))

        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        return transforms.Compose(transform_list)
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


# =============================================================================
# Dataset Factory
# =============================================================================

def build_datasets(
    config: DataConfig,
) -> Tuple[Dataset, Dataset]:
    """Build train and test datasets from configuration.

    Args:
        config: Data configuration specifying dataset and transforms.

    Returns:
        Tuple of (train_dataset, test_dataset).

    Raises:
        ValueError: If dataset name is not recognized.

    Supported datasets:
        - "cifar10": CIFAR-10 (10 classes, 32x32 RGB)
        - "cifar100": CIFAR-100 (100 classes, 32x32 RGB)
    """
    data_dir = Path(config.data_dir, config.dataset)
    data_dir.mkdir(parents=True, exist_ok=True)

    if config.dataset == "cifar10":
        train_transform = build_cifar_transforms(config, train=True)
        test_transform = build_cifar_transforms(config, train=False)

        train_dataset = CIFAR10(
            root=str(data_dir),
            train=True,
            transform=train_transform,
            download=True,
        )
        test_dataset = CIFAR10(
            root=str(data_dir),
            train=False,
            transform=test_transform,
            download=True,
        )

    elif config.dataset == "cifar100":
        train_transform = build_cifar_transforms(config, train=True)
        test_transform = build_cifar_transforms(config, train=False)

        train_dataset = CIFAR100(
            root=str(data_dir),
            train=True,
            transform=train_transform,
            download=True,
        )
        test_dataset = CIFAR100(
            root=str(data_dir),
            train=False,
            transform=test_transform,
            download=True,
        )

    else:
        raise ValueError(
            f"Unknown dataset: '{config.dataset}'. "
            f"Supported datasets: 'cifar10', 'cifar100'"
        )

    return train_dataset, test_dataset


def build_dataloaders(
    config: DataConfig,
    val_split: Optional[float] = None,
) -> Tuple[DataLoader, DataLoader]:
    """Build train and validation/test dataloaders from configuration.

    Args:
        config: Data configuration.
        val_split: If provided, split training set into train/val with this
                  ratio for validation (e.g., 0.1 = 10% validation).
                  If None, returns train and test loaders.

    Returns:
        Tuple of (train_loader, val_or_test_loader).
    """
    train_dataset, test_dataset = build_datasets(config)

    # Optionally split training data for validation
    if val_split is not None and 0 < val_split < 1:
        val_size = int(len(train_dataset) * val_split)
        train_size = len(train_dataset) - val_size

        train_dataset, val_dataset = random_split(
            train_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )
        eval_dataset = val_dataset
    else:
        eval_dataset = test_dataset

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True,  # Drop last incomplete batch for consistent batch sizes
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    return train_loader, eval_loader


# =============================================================================
# Dataset Info
# =============================================================================

def get_dataset_info(dataset_name: str) -> dict:
    """Get metadata about a dataset.

    Args:
        dataset_name: Name of the dataset.

    Returns:
        Dictionary with:
            - num_classes: Number of output classes
            - img_size: Image dimensions (H, W)
            - in_channels: Number of input channels
            - mean: Normalization mean
            - std: Normalization std
    """
    info = {
        "cifar10": {
            "num_classes": 10,
            "img_size": (32, 32),
            "in_channels": 3,
        },
        "cifar100": {
            "num_classes": 100,
            "img_size": (32, 32),
            "in_channels": 3,
        }
    }

    if dataset_name not in info:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    result = info[dataset_name]
    result.update(DATASET_STATS.get(dataset_name, {}))
    return result
