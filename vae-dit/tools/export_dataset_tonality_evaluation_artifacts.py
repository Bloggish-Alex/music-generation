#!/usr/bin/env python3
"""Thin entry point for dataset-tonality public export."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cli.export_dataset_tonality_evaluation_artifacts import main


if __name__ == "__main__":
    main()
