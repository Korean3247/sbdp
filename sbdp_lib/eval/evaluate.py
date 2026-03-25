import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def _forward(model, data, device):
    """Forward pass handling both image tensors and text dicts."""
    if isinstance(data, dict):
        return model(data["input_ids"].to(device), data["attention_mask"].to(device))
    else:
        return model(data.to(device))


@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device) -> dict:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    for data, labels, *rest in dataloader:
        labels = labels.to(device)
        outputs = _forward(model, data, device)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * labels.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return {
        "loss": total_loss / total,
        "acc": correct / total,
        "correct": correct,
        "total": total,
    }
