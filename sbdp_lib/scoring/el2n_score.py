import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def _forward(model, data, device):
    if isinstance(data, dict):
        return model(data["input_ids"].to(device), data["attention_mask"].to(device))
    else:
        return model(data.to(device))


class EL2NScorer:
    """Computes per-sample EL2N score: ||softmax(logits) - one_hot(y)||_2."""

    @torch.no_grad()
    def compute_scores(
        self, model: nn.Module, dataloader: DataLoader, device: torch.device
    ) -> dict[int, float]:
        model.eval()
        scores = {}
        for data, labels, sample_ids in dataloader:
            labels = labels.to(device)
            outputs = _forward(model, data, device)
            probs = F.softmax(outputs, dim=1)
            one_hot = F.one_hot(labels, num_classes=outputs.size(1)).float()
            el2n = torch.norm(probs - one_hot, p=2, dim=1)
            for sid, score_val in zip(sample_ids.tolist(), el2n.tolist()):
                scores[sid] = score_val
        model.train()
        return scores
