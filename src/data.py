from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(image_size: int = 224, train: bool = False):
    if train:
        return transforms.Compose([
            transforms.Resize(int(image_size * 1.15)),
            transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(8),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.15)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def _weighted_sampler(dataset: datasets.ImageFolder) -> WeightedRandomSampler:
    targets = torch.tensor(dataset.targets)
    class_counts = torch.bincount(targets)
    class_weights = 1.0 / class_counts.float()
    sample_weights = class_weights[targets]
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def build_loaders(data_dir: str, batch_size: int = 32, num_workers: int = 2, image_size: int = 224):
    root = Path(data_dir)
    train_ds = datasets.ImageFolder(root / "train", transform=build_transforms(image_size, train=True))
    val_ds = datasets.ImageFolder(root / "val", transform=build_transforms(image_size, train=False))
    test_ds = datasets.ImageFolder(root / "test", transform=build_transforms(image_size, train=False))

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=_weighted_sampler(train_ds),
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader, train_ds.classes
