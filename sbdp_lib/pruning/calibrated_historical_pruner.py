import math
import numpy as np
from .base import BasePruner


class CalibratedHistoricalPruner(BasePruner):
    """Calibrated top-k pruner with historical correction.

    Extends CalibratedTopKPruner by adding a re-entry bonus for samples
    that have been excluded for many consecutive pruning steps, and a
    small penalty for samples that have been selected too frequently.

    final_score_i = ema_i + beta * u_i
    u_i = log(1 + age_i) - lambda_c * count_i

    - age_i: consecutive pruning steps excluded (reset to 0 on selection)
    - count_i: total times selected so far
    - beta: historical correction strength
    - lambda_c: count penalty strength
    """

    def __init__(
        self,
        window_size: int = 2,
        ema_alpha: float = 0.8,
        beta: float = 0.1,
        lambda_c: float = 0.01,
        epsilon: float = 1e-8,
    ):
        self.window_size = window_size
        self.ema_alpha = ema_alpha
        self.beta = beta
        self.lambda_c = lambda_c
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
        age = state.get("age", {})
        count = state.get("count", {})

        all_ids = list(scores.keys())

        # --- Calibration (same as CalibratedTopKPruner) ---
        raw_values = np.array([scores[sid] for sid in all_ids])
        mu = raw_values.mean()
        std = raw_values.std()
        calibrated = {
            sid: (scores[sid] - mu) / (std + self.epsilon) for sid in all_ids
        }

        if len(score_history) >= 1:
            recent = score_history[-self.window_size:]
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

        # --- EMA smoothing ---
        alpha = self.ema_alpha
        new_ema = {}
        for sid in all_ids:
            if sid in ema_scores:
                new_ema[sid] = alpha * ema_scores[sid] + (1 - alpha) * calibrated[sid]
            else:
                new_ema[sid] = calibrated[sid]

        # --- Historical correction ---
        final_scores = {}
        for sid in all_ids:
            a = age.get(sid, 0)
            c = count.get(sid, 0)
            u = math.log(1 + a) - self.lambda_c * c
            final_scores[sid] = new_ema[sid] + self.beta * u

        # --- Select top-k ---
        k = self._compute_k(len(all_ids), retention_ratio)
        sorted_ids = sorted(all_ids, key=lambda sid: final_scores[sid], reverse=True)
        selected = sorted_ids[:k]
        selected_set = set(selected)

        # --- Update state ---
        new_age = {}
        new_count = {}
        for sid in all_ids:
            if sid in selected_set:
                new_age[sid] = 0
                new_count[sid] = count.get(sid, 0) + 1
            else:
                new_age[sid] = age.get(sid, 0) + 1
                new_count[sid] = count.get(sid, 0)

        state["score_history"] = score_history + [{sid: scores[sid] for sid in all_ids}]
        state["ema_scores"] = new_ema
        state["age"] = new_age
        state["count"] = new_count

        return sorted(selected)
