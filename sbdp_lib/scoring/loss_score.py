import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def _forward(model, data, device):
    """Forward pass handling both image tensors and text dicts."""
    if isinstance(data, dict):
        return model(data["input_ids"].to(device), data["attention_mask"].to(device))
    else:
        return model(data.to(device))


class LossScorer:
    """Computes per-sample cross-entropy loss as importance score."""

    def __init__(self):
        self.criterion = nn.CrossEntropyLoss(reduction="none")

    @torch.no_grad()
    def compute_scores(
        self, model: nn.Module, dataloader: DataLoader, device: torch.device
    ) -> dict[int, float]:
        model.eval()
        scores = {}
        for data, labels, sample_ids in dataloader:
            labels = labels.to(device)
            outputs = _forward(model, data, device)
            losses = self.criterion(outputs, labels)
            for sid, loss_val in zip(sample_ids.tolist(), losses.tolist()):
                scores[sid] = loss_val
        model.train()
        return scores
