#!/usr/bin/env python3
"""Diagnostics containers for training and generation analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class DiagnosticsBase:
    """Append-only JSON diagnostics writer used by pipeline modules."""

    def __init__(self, kind: str) -> None:
        self.payload: Dict[str, Any] = {
            "kind": kind,
            "stages": {},
            "events": [],
        }

    def set_request(self, request: Dict[str, Any]) -> None:
        self.payload["request"] = request

    def record_stage(self, stage: str, data: Dict[str, Any]) -> None:
        self.payload["stages"][stage] = data

    def append_event(self, stage: str, event: Dict[str, Any]) -> None:
        self.payload["events"].append({"stage": stage, **event})

    def to_dict(self) -> Dict[str, Any]:
        return self.payload

    def write(self, output_path: str | Path) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.payload, indent=2), encoding="utf-8")


class TrainingDiagnostics(DiagnosticsBase):
    """Training diagnostics focused on data quality and learned matrices."""

    def __init__(self) -> None:
        super().__init__("training")

    def record_input_summary(
        self,
        parsed_files: int,
        failed_files: List[Dict[str, Any]],
        bar_count: int,
    ) -> None:
        self.record_stage("input", {
            "parsed_files": parsed_files,
            "failed_files": failed_files,
            "bar_count": bar_count,
        })

    def record_clustering(self, diagnostics: Dict[str, Any]) -> None:
        self.record_stage("bar_clustering", diagnostics)

    def record_observation_vocab(self, diagnostics: Dict[str, Any]) -> None:
        self.record_stage("observation_vocab", diagnostics)

    def record_observation_bar_pools(self, diagnostics: Dict[str, Any]) -> None:
        self.record_stage("observation_bar_pools", diagnostics)

    def record_hmm(self, form_name: str, diagnostics: Dict[str, Any]) -> None:
        hmm_stage = self.payload["stages"].setdefault("form_hmm", {})
        hmm_stage[form_name] = diagnostics


class GenerationDiagnostics(DiagnosticsBase):
    """Generation diagnostics tying every sampled bar to its source data."""

    def __init__(self) -> None:
        super().__init__("generation")

    def record_section_plan(self, plan: List[Dict[str, Any]]) -> None:
        self.record_stage("section_plan", {"sections": plan})

    def record_sampled_bar(self, payload: Dict[str, Any]) -> None:
        self.append_event("sampled_bar", payload)

    def record_rare_bar_selection(self, payload: Dict[str, Any]) -> None:
        self.append_event("rare_bar_selection", payload)

    def record_render(self, diagnostics: Dict[str, Any]) -> None:
        self.record_stage("render", diagnostics)
