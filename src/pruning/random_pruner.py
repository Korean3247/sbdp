import random
from .base import BasePruner


class RandomPruner(BasePruner):
    def select(
        self,
        scores: dict[int, float],
        retention_ratio: float,
        state: dict | None = None,
    ) -> list[int]:
        all_ids = list(scores.keys())
        k = self._compute_k(len(all_ids), retention_ratio)
        selected = random.sample(all_ids, k)
        return sorted(selected)
