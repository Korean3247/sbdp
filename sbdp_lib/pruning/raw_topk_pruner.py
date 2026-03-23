from .base import BasePruner


class RawTopKPruner(BasePruner):
    """Select top-k samples by raw score (highest loss = most important)."""

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
