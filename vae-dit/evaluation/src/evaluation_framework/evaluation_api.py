"""Stable public interfaces for artifact-only evaluation modules."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence


@dataclass(frozen=True)
class ArtifactBundle:
    """Artifacts exported for exactly one evaluation test point."""

    test_point: str
    artifacts: Mapping[str, str]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_point": self.test_point,
            "artifacts": dict(self.artifacts),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ArtifactBundle":
        return cls(
            test_point=str(payload["test_point"]),
            artifacts=dict(payload.get("artifacts") or {}),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass(frozen=True)
class EvaluationResult:
    """Evaluator output written by the shared runner, never by a module path."""

    report: Mapping[str, Any]
    markdown: str = ""
    figures: Mapping[str, bytes] = field(default_factory=dict)
    supplementary_json: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)


class ArtifactExporter(ABC):
    """Engineering adapter: diagnostics in, contract artifacts out."""

    test_point: str
    input_contract: str
    output_contract: str

    @abstractmethod
    def export(self, context: "ExportContext") -> ArtifactBundle:
        """Read engineering observations and write contract artifacts via context.store."""


class ArtifactEvaluator(ABC):
    """Artifact-only evaluator: no model, pipeline or checkpoint imports."""

    test_point: str
    required_artifacts: Sequence[str] = ()

    @abstractmethod
    def evaluate(self, context: "EvaluationContext", bundle: ArtifactBundle) -> EvaluationResult:
        """Calculate independent metrics and return a report payload."""


@dataclass(frozen=True)
class EvaluationModule:
    """One test point and its optional export/evaluation implementations."""

    test_point: str
    exporter: Optional[ArtifactExporter] = None
    evaluator: Optional[ArtifactEvaluator] = None
    requires: Sequence[str] = ()
    summary: str = ""

    def __post_init__(self) -> None:
        if not self.test_point:
            raise ValueError("Evaluation module requires a test_point name.")
        if self.exporter is not None and self.exporter.test_point != self.test_point:
            raise ValueError("Exporter test_point does not match its module.")
        if self.evaluator is not None and self.evaluator.test_point != self.test_point:
            raise ValueError("Evaluator test_point does not match its module.")


class ExportContext:  # pragma: no cover - imported lazily to avoid a runtime cycle
    pass


class EvaluationContext:  # pragma: no cover - imported lazily to avoid a runtime cycle
    pass
