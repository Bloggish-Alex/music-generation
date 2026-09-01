"""Persist evaluator results through the one sanctioned run-directory writer."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .evaluation_api import EvaluationResult
from .evaluation_artifact_store import EvaluationArtifactStore


@dataclass(frozen=True)
class EvaluationReportPublisher:
    """Adds execution provenance and writes every returned report artifact."""

    store: EvaluationArtifactStore
    code_revision: str

    def publish(self, test_point: str, result: EvaluationResult) -> Path:
        report = self._with_execution_provenance(result.report)
        report_path = self.store.write_json(test_point, "report", report)
        if result.markdown:
            self.store.write_text(test_point, "report", result.markdown, extension="md")
        for figure_role, figure_bytes in result.figures.items():
            self.store.write_bytes(test_point, figure_role, figure_bytes, extension="png")
        for role, payload in result.supplementary_json.items():
            self.store.write_json(test_point, role, payload)
        self.store.record_evaluation(test_point, result, report_path)
        return report_path

    def _with_execution_provenance(self, source: Mapping[str, Any]) -> dict[str, Any]:
        report = dict(source)
        provenance = dict(report.get("provenance") or {})
        provenance["evaluation_execution"] = {
            "executed_at": datetime.now(timezone.utc).isoformat(),
            "code_revision": self.code_revision,
        }
        report["provenance"] = provenance
        return report
