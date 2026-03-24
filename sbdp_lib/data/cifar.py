import numpy as np
import torchvision
import torchvision.transforms as T
from pathlib import Path


def get_cifar10_transforms():
    train_transform = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    return train_transform, test_transform


def get_cifar10(data_dir: str = "./data"):
    train_transform, test_transform = get_cifar10_transforms()
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform
    )
    return train_dataset, test_dataset


def get_cifar10_notransform(data_dir: str = "./data"):
    """For scoring: no augmentation, only normalize."""
    _, test_transform = get_cifar10_transforms()
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=test_transform
    )
    return train_dataset


def apply_symmetric_noise(dataset, noise_rate: float, seed: int = 0):
    """Apply symmetric label noise to a dataset in-place.

    Each sample is independently flipped to a uniformly random *other* class
    with probability noise_rate. The same seed produces the same flip pattern,
    so train_dataset and score_dataset can be kept consistent.

    Returns the dataset with noisy targets (modifies in-place and returns).
    """
    if noise_rate <= 0.0:
        return dataset

    num_classes = 10
    rng = np.random.RandomState(seed)
    targets = list(dataset.targets)
    for i in range(len(targets)):
        if rng.rand() < noise_rate:
            orig = targets[i]
            other = list(range(num_classes))
            other.remove(orig)
            targets[i] = int(rng.choice(other))
    dataset.targets = targets
    return dataset
