"""Shared result vocabulary for artifact-only evaluators."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List


class AssessmentStatus(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"
    UNAVAILABLE = "UNAVAILABLE"
    MONITOR = "MONITOR"


@dataclass(frozen=True)
class MetricResult:
    """One independent metric; values are never combined into a total score."""

    metric_id: str
    value: Any
    status: AssessmentStatus
    availability: str = "available"
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "value": self.value,
            "status": self.status.value,
            "availability": self.availability,
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class AssessmentReport:
    """Stable report envelope shared by future evaluators."""

    assessment_type: str
    status: AssessmentStatus
    metrics: List[MetricResult] = field(default_factory=list)
    findings: List[Dict[str, Any]] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)
    missing_inputs: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": "assessment_report.v1",
            "assessment_type": self.assessment_type,
            "status": self.status.value,
            "metrics": {metric.metric_id: metric.to_dict() for metric in self.metrics},
            "findings": list(self.findings),
            "provenance": dict(self.provenance),
            "missing_inputs": list(self.missing_inputs),
        }
