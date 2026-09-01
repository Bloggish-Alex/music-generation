#!/usr/bin/env python3
"""Thin entry point for artifact-only evaluation."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cli.evaluate import main


if __name__ == "__main__":
    main()
