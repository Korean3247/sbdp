import yaml
import argparse
from pathlib import Path


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config


def parse_args():
    parser = argparse.ArgumentParser(description="SBDP Training")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--seed", type=int, default=None, help="Override seed from config")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output directory")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.seed is not None:
        config["seed"] = args.seed
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    return config


def generate_run_name(config: dict) -> str:
    dataset = config.get("dataset", "unknown")
    model = config.get("model", "unknown")
    mode = config.get("pruning", {}).get("mode", "full")
    seed = config.get("seed", 0)
    retention = config.get("pruning", {}).get("retention_ratio", 1.0)
    noise_rate = config.get("noise_rate", 0.0)
    if noise_rate > 0.0:
        return f"{dataset}_{model}_{mode}_noise{noise_rate}_ret{retention}_seed{seed}"
    return f"{dataset}_{model}_{mode}_ret{retention}_seed{seed}"
