#!/usr/bin/env python3
"""CLI wrapper for the semantic BarCodec harmony oracle."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cli.analyze_bar_codec_harmony_oracle import main  # noqa: E402


if __name__ == "__main__":
    main()
