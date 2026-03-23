from abc import ABC, abstractmethod


class BasePruner(ABC):
    @abstractmethod
    def select(
        self,
        scores: dict[int, float],
        retention_ratio: float,
        state: dict | None = None,
    ) -> list[int]:
        """Select sample IDs to retain for next training interval.

        Args:
            scores: {sample_id: raw_score}
            retention_ratio: fraction of samples to keep (0, 1]
            state: optional state from previous pruning events

        Returns:
            List of selected sample_ids
        """
        ...

    def _compute_k(self, n: int, retention_ratio: float) -> int:
        k = max(1, int(n * retention_ratio))
        return k
