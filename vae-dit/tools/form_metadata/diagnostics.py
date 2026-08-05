#!/usr/bin/env python3
"""JSON diagnostics used while parsing form-metadata inputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


class DiagnosticsBase:
    """Collect stage summaries and append-only events in a JSON-safe payload."""

    def __init__(self, kind: str) -> None:
        """Initialize an empty diagnostics payload for one pipeline kind."""
        self.payload: Dict[str, Any] = {
            "kind": kind,
            "stages": {},
            "events": [],
        }

    def set_request(self, request: Dict[str, Any]) -> None:
        """Record the normalized request parameters that produced this run."""
        self.payload["request"] = request

    def record_stage(self, stage: str, data: Dict[str, Any]) -> None:
        """Replace the summary for a named processing stage."""
        self.payload["stages"][stage] = data

    def append_event(self, stage: str, event: Dict[str, Any]) -> None:
        """Append one ordered event and annotate it with its source stage."""
        self.payload["events"].append({"stage": stage, **event})

    def to_dict(self) -> Dict[str, Any]:
        """Return the current JSON-safe diagnostics payload."""
        return self.payload

    def write(self, output_path: str | Path) -> None:
        """Serialize diagnostics as indented UTF-8 JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.payload, indent=2), encoding="utf-8")


class TrainingDiagnostics(DiagnosticsBase):
    """Record the input-quality summary produced by the shared parser."""

    def __init__(self) -> None:
        """Initialize diagnostics with the historical ``training`` kind."""
        super().__init__("training")

    def record_input_summary(
        self,
        parsed_files: int,
        failed_files: List[Dict[str, Any]],
        bar_count: int,
    ) -> None:
        """Record parsed-file, failed-file, and total-bar counts."""
        self.record_stage("input", {
            "parsed_files": parsed_files,
            "failed_files": failed_files,
            "bar_count": bar_count,
        })
