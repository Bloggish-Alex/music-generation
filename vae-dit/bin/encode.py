#!/usr/bin/env python3
"""Executable wrapper for the encoding CLI."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    """Add src to sys.path and delegate to the package CLI."""
    src_dir = Path(__file__).resolve().parent.parent / "src"
    sys.path.insert(0, str(src_dir))
    from cli.encode import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()
