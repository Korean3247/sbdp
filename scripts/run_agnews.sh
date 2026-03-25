#!/bin/bash
# Phase 3: AG News Cross-Domain Generalization
# DistilBERT fine-tuning with calibrated top-k pruning
#
# 5 configs x 3 seeds = 15 runs
# Estimated time: ~9 hours on single GPU

set -e

CONFIGS=(
    "configs/agnews_distilbert_full.yaml"
    "configs/agnews_distilbert_calibrated_ret0.3.yaml"
    "configs/agnews_distilbert_calibrated_ret0.5.yaml"
    "configs/agnews_distilbert_random_ret0.3.yaml"
    "configs/agnews_distilbert_random_ret0.5.yaml"
)
SEEDS=(0 1 2)

echo "=== Phase 3: AG News Cross-Domain Experiment ==="
echo "Configs: ${#CONFIGS[@]}, Seeds: ${#SEEDS[@]}, Total: $(( ${#CONFIGS[@]} * ${#SEEDS[@]} ))"

for config in "${CONFIGS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        echo ">>> Running: $config seed=$seed"
        python scripts/run_train.py --config "$config" --seed "$seed"
        echo ">>> Done"
    done
done

echo "=== AG News experiments complete ==="
python scripts/summarize_results.py outputs
