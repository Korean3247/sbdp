import torch
import torch.nn as nn
from torch.utils.data import DataLoader


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
        for images, labels, sample_ids in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            losses = self.criterion(outputs, labels)
            for sid, loss_val in zip(sample_ids.tolist(), losses.tolist()):
                scores[sid] = loss_val
        model.train()
        return scores
