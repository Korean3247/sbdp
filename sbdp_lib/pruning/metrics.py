import numpy as np


def score_drift_index(score_history: list[dict]) -> float:
    """Compute mean absolute score drift between adjacent pruning events.

    SDI = mean_{i,t} |s_i^{(t+1)} - s_i^{(t)}|
    Lower is more stable.
    """
    if len(score_history) < 2:
        return 0.0

    drifts = []
    for t in range(len(score_history) - 1):
        curr = score_history[t]["scores"]
        next_ = score_history[t + 1]["scores"]
        common_ids = set(curr.keys()) & set(next_.keys())
        for sid in common_ids:
            drifts.append(abs(next_[sid] - curr[sid]))

    return float(np.mean(drifts)) if drifts else 0.0


def selection_turnover(mask_history: list[dict]) -> list[float]:
    """Compute selection turnover between adjacent pruning events.

    turnover_t = 1 - |S_t ∩ S_{t+1}| / |S_t ∪ S_{t+1}|
    Lower is more stable.

    Returns list of turnover values.
    """
    turnovers = []
    for t in range(len(mask_history) - 1):
        s_t = set(mask_history[t]["selected_ids"])
        s_t1 = set(mask_history[t + 1]["selected_ids"])
        intersection = len(s_t & s_t1)
        union = len(s_t | s_t1)
        if union == 0:
            turnovers.append(0.0)
        else:
            turnovers.append(1.0 - intersection / union)
    return turnovers


def mean_turnover(mask_history: list[dict]) -> float:
    turnovers = selection_turnover(mask_history)
    return float(np.mean(turnovers)) if turnovers else 0.0
