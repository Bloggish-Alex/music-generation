#!/usr/bin/env python3
"""Diagnostics containers for training and generation analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class DiagnosticsBase:
    """Append-only JSON diagnostics writer used by pipeline modules."""

    def __init__(self, kind: str) -> None:
        """Initialize an empty diagnostics payload for one pipeline kind."""
        self.payload: Dict[str, Any] = {
            "kind": kind,
            "stages": {},
            "events": [],
        }

    def set_request(self, request: Dict[str, Any]) -> None:
        """Record normalized request parameters for reproducibility."""
        self.payload["request"] = request

    def record_stage(self, stage: str, data: Dict[str, Any]) -> None:
        """Replace the summary for a named pipeline stage."""
        self.payload["stages"][stage] = data

    def append_event(self, stage: str, event: Dict[str, Any]) -> None:
        """Append an ordered event annotated with its stage name."""
        self.payload["events"].append({"stage": stage, **event})

    def to_dict(self) -> Dict[str, Any]:
        """Return the current JSON-safe diagnostics payload."""
        return self.payload

    def write(self, output_path: str | Path) -> None:
        """Write the payload as indented UTF-8 JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.payload, indent=2), encoding="utf-8")
