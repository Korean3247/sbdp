import numpy as np
from .base import BasePruner


class EL2NPruner(BasePruner):
    """Select top-k samples by EL2N score (highest error = most important).

    EL2N score = ||p(x) - e_y||_2 where p(x) is softmax output, e_y is one-hot.
    Higher EL2N = harder sample = more important for training.
    This pruner uses pre-computed EL2N scores passed via the scores dict.
    """

    def select(
        self,
        scores: dict[int, float],
        retention_ratio: float,
        state: dict | None = None,
    ) -> list[int]:
        k = self._compute_k(len(scores), retention_ratio)
        sorted_ids = sorted(scores.keys(), key=lambda sid: scores[sid], reverse=True)
        selected = sorted_ids[:k]
        return sorted(selected)
