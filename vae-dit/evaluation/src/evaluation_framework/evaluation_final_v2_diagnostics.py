"""Framework-native reports for final Codec V2 parser/control diagnostics."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .core.artifacts import VerifiedArtifactResolver
from .evaluation_api import ArtifactBundle, ArtifactEvaluator, ArtifactExporter, EvaluationModule, EvaluationResult
from .evaluation_context import EvaluationContext, ExportContext


class FinalV2DiagnosticExporter(ArtifactExporter):
    def __init__(self, test_point: str) -> None:
        self.test_point = test_point; self.input_contract = f"{test_point}_raw_observation.v2"; self.output_contract = f"{test_point}_inputs.v2"

    def export(self, context: ExportContext) -> ArtifactBundle:
        path = context.input_root / f"{self.test_point}__raw_observation.v2.json"
        if not path.is_file():
            payload = {"schema_version": self.output_contract, "availability": "not_provided"}
        else:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if raw.get("schema_version") != self.input_contract:
                raise ValueError(f"Unsupported {self.test_point} raw schema.")
            payload = {"schema_version": self.output_contract, "availability": str(raw.get("status")), "raw": {"path": path.name, "sha256": VerifiedArtifactResolver.sha256(path)}}
        written = context.store.write_json(self.test_point, "inputs", payload)
        return ArtifactBundle(self.test_point, {"inputs": written.name})


class FinalV2DiagnosticEvaluator(ArtifactEvaluator):
    def __init__(self, test_point: str) -> None:
        self.test_point = test_point; self.required_artifacts: Sequence[str] = ("inputs",)

    def evaluate(self, context: EvaluationContext, bundle: ArtifactBundle) -> EvaluationResult:
        inputs = json.loads((context.store.run_dir / bundle.artifacts["inputs"]).read_text(encoding="utf-8"))
        if inputs.get("availability") != "AVAILABLE":
            report = {"schema_version": "assessment_report.v1", "status": "UNAVAILABLE", "metrics": {}, "findings": [], "provenance": {"inputs": inputs}, "missing_inputs": [{"field": self.test_point, "reason": str(inputs.get("availability"))}]}
            return EvaluationResult(report, f"# {self.test_point}\n\nRaw observation unavailable.\n")
        raw = VerifiedArtifactResolver(context.input_root).json(inputs["raw"])
        report = {"schema_version": "assessment_report.v1", "status": "MONITOR", "metrics": {"observation": _metrics(self.test_point, raw)}, "findings": [{"classification": "diagnostic_monitor", "text": "This report records Codec V2 source and parser facts; it is not a model quality score."}], "provenance": {"raw": inputs["raw"], "run": raw.get("run")}, "missing_inputs": []}
        return EvaluationResult(report, f"# {self.test_point}\n\nStatus: MONITOR\n")


def _metrics(test_point: str, raw: Mapping[str, Any]) -> Mapping[str, Any]:
    keys = {"parser_integrity": ("measure_map", "track_retention", "parser_failures"), "quantization_audit": ("grid_policy", "by_file_meter"), "performance_controls": ("tempo", "key", "velocity", "cc64"), "form_action_alignment": ("coverage", "confusion_table")}[test_point]
    return {key: raw.get(key) for key in keys}


def _module(name: str) -> EvaluationModule:
    return EvaluationModule(name, FinalV2DiagnosticExporter(name), FinalV2DiagnosticEvaluator(name), summary=f"Final Codec V2 {name.replace('_', ' ')} diagnostics.")


PARSER_INTEGRITY_MODULE = _module("parser_integrity")
QUANTIZATION_AUDIT_MODULE = _module("quantization_audit")
PERFORMANCE_CONTROLS_MODULE = _module("performance_controls")
FORM_ACTION_ALIGNMENT_MODULE = _module("form_action_alignment")
