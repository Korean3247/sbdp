#!/usr/bin/env python3
"""Run a single SBDP training job."""

import os
import sys

# Add project root to path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

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
