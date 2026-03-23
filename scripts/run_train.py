#!/usr/bin/env python3
"""Run a single SBDP training job."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import parse_args
from src.train.trainer import train


def main():
    config = parse_args()
    summary = train(config)
    print("\n=== Training Complete ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
