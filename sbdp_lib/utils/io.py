import json
import csv
import torch
import yaml
from pathlib import Path


def ensure_dir(path: str | Path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_config(config: dict, path: str | Path):
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def save_json(data: dict, path: str | Path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(path: str | Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def save_scores(score_history: list[dict], path: str | Path):
    torch.save(score_history, path)


def load_scores(path: str | Path) -> list[dict]:
    return torch.load(path, weights_only=False)


def save_masks(mask_history: list[dict], path: str | Path):
    torch.save(mask_history, path)


def load_masks(path: str | Path) -> list[dict]:
    return torch.load(path, weights_only=False)


class MetricsLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.rows = []
        self.fieldnames = [
            "epoch", "train_loss", "train_acc", "test_loss", "test_acc",
            "current_subset_size", "mode", "seed"
        ]

    def log(self, row: dict):
        self.rows.append(row)

    def save(self):
        with open(self.path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()
            for row in self.rows:
                writer.writerow(row)
