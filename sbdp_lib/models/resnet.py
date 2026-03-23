import torch.nn as nn
import torchvision.models as models


def get_resnet18(num_classes: int = 10) -> nn.Module:
    """ResNet-18 adapted for CIFAR-10 (32x32 input)."""
    model = models.resnet18(weights=None)
    # Adapt first conv for 32x32 input (no aggressive downsampling)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
