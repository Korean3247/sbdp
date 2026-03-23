# SBDP: Stable Budgeted Data Pruning

Dynamic data pruning framework for comparing raw vs. locally-calibrated importance scoring.

## Purpose

Training-time data pruning reduces compute by selecting important samples. This project implements and compares:
- **Full training** (no pruning)
- **Random pruning** (baseline)
- **Raw top-k loss pruning** (standard dynamic pruning)
- **Calibrated top-k loss pruning** (local z-score normalization + EMA smoothing)

Key metrics: test accuracy, score drift index (SDI), selection turnover.

## Setup

```bash
pip install -r requirements.txt
```

Data (CIFAR-10) is auto-downloaded to `./data/` on first run.

## Usage

### Single run
```bash
python scripts/run_train.py --config configs/cifar10_resnet18_calibrated.yaml --seed 0
```

### All experiments (4 modes × 3 seeds = 12 runs)
```bash
bash scripts/run_all_seeds.sh
```

### Summarize results
```bash
python scripts/summarize_results.py outputs
```

## Config

| Key | Description |
|-----|-------------|
| `pruning.mode` | `full`, `random_pruning`, `raw_topk_loss`, `calibrated_topk_loss` |
| `pruning.warmup_epochs` | Epochs before first pruning |
| `pruning.interval_epochs` | Epochs between pruning steps |
| `pruning.retention_ratio` | Fraction of data to keep (0,1] |
| `pruning.local_window` | Number of past events for calibration |
| `pruning.ema_alpha` | EMA decay for score smoothing |

## Output Structure

Each run saves to `outputs/<run_name>/`:
- `config.yaml` — run configuration
- `metrics.csv` — per-epoch train/test metrics
- `score_history.pt` — per-sample scores at each pruning event
- `mask_history.pt` — selected sample IDs at each pruning event
- `final_checkpoint.pt` — model weights
- `summary.json` — final metrics + stability indices
- `train.log` — training log

## Modes

| Mode | Description |
|------|-------------|
| `full` | Standard training, all data every epoch |
| `random_pruning` | Random subset selection at each pruning step |
| `raw_topk_loss` | Keep highest-loss samples (raw scores) |
| `calibrated_topk_loss` | Keep highest-loss samples after local z-score calibration + EMA |

## Stability Metrics

- **Score Drift Index (SDI)**: Mean absolute score change between adjacent pruning events. Lower = more stable.
- **Selection Turnover**: 1 - Jaccard similarity of selected sets between adjacent events. Lower = more stable.

## Reproducibility

- Seeds are fixed for Python, NumPy, PyTorch, and CUDNN
- Same config + same seed → same result (on same hardware)
- Run name auto-generated from config to avoid path collisions
