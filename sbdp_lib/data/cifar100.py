import numpy as np
import torchvision
import torchvision.transforms as T


def get_cifar100_transforms():
    train_transform = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    return train_transform, test_transform


def get_cifar100(data_dir: str = "./data"):
    train_transform, test_transform = get_cifar100_transforms()
    train_dataset = torchvision.datasets.CIFAR100(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR100(
        root=data_dir, train=False, download=True, transform=test_transform
    )
    return train_dataset, test_dataset


def get_cifar100_notransform(data_dir: str = "./data"):
    """For scoring: no augmentation, only normalize."""
    _, test_transform = get_cifar100_transforms()
    train_dataset = torchvision.datasets.CIFAR100(
        root=data_dir, train=True, download=True, transform=test_transform
    )
    return train_dataset


def apply_symmetric_noise(dataset, noise_rate: float, seed: int = 0):
    if noise_rate <= 0.0:
        return dataset
    num_classes = 100
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
