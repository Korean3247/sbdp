import numpy as np
from .base import BasePruner


class CalibratedTopKPruner(BasePruner):
    """Top-k pruner with local z-score calibration and EMA smoothing.

    Uses event-wise global z-score (Option B from spec):
    - Normalize current scores using current event's mean/std
    - Apply EMA smoothing across pruning events
    """

    def __init__(self, window_size: int = 2, ema_alpha: float = 0.8, epsilon: float = 1e-8):
        self.window_size = window_size
        self.ema_alpha = ema_alpha
        self.epsilon = epsilon

    def select(
        self,
        scores: dict[int, float],
        retention_ratio: float,
        state: dict | None = None,
    ) -> list[int]:
        if state is None:
            state = {}

        score_history = state.get("score_history", [])
        ema_scores = state.get("ema_scores", {})

        # Current event z-score calibration
        all_ids = list(scores.keys())
        raw_values = np.array([scores[sid] for sid in all_ids])
        mu = raw_values.mean()
        std = raw_values.std()
        calibrated = {
            sid: (scores[sid] - mu) / (std + self.epsilon) for sid in all_ids
        }

        # If we have history, use windowed calibration
        if len(score_history) >= 1:
            # Collect recent window scores for additional context
            recent = score_history[-self.window_size:]
            # Stack recent + current for window stats
            window_scores = []
            for hist in recent:
                for sid in all_ids:
                    if sid in hist:
                        window_scores.append(hist[sid])
            window_scores.extend(raw_values.tolist())
            w_arr = np.array(window_scores)
            w_mu = w_arr.mean()
            w_std = w_arr.std()
            calibrated = {
                sid: (scores[sid] - w_mu) / (w_std + self.epsilon) for sid in all_ids
            }

        # EMA smoothing
        alpha = self.ema_alpha
        new_ema = {}
        for sid in all_ids:
            if sid in ema_scores:
                new_ema[sid] = alpha * ema_scores[sid] + (1 - alpha) * calibrated[sid]
            else:
                new_ema[sid] = calibrated[sid]

        # Update state (caller must persist this)
        state["score_history"] = score_history + [{sid: scores[sid] for sid in all_ids}]
        state["ema_scores"] = new_ema

        # Select top-k by EMA score
        k = self._compute_k(len(all_ids), retention_ratio)
        sorted_ids = sorted(all_ids, key=lambda sid: new_ema[sid], reverse=True)
        selected = sorted_ids[:k]
        return sorted(selected)
