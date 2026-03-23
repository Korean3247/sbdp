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
