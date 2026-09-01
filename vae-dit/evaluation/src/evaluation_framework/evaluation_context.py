"""Execution contexts shared by exporters and evaluators."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .evaluation_artifact_store import EvaluationArtifactStore


@dataclass(frozen=True)
class ExportContext:
    run_id: str
    input_root: Path
    store: EvaluationArtifactStore
    options: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationContext:
    run_id: str
    input_root: Path
    store: EvaluationArtifactStore
    options: Mapping[str, Any] = field(default_factory=dict)
