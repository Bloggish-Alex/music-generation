"""Artifact-only information-flow evidence ledger across evaluation boundaries."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .core.artifacts import VerifiedArtifactResolver
from .core.oracle_ledger_registry import OracleBoundary, load_oracle_boundaries
from .evaluation_api import ArtifactBundle, ArtifactEvaluator, ArtifactExporter, EvaluationModule, EvaluationResult
from .evaluation_context import EvaluationContext, ExportContext


TEST_POINT = "oracle_ladder"
INPUT_SCHEMA = "oracle_ladder_inputs.v1"
BOUNDARIES = load_oracle_boundaries()
BOUNDARY_MODULES = {boundary.module for boundary in BOUNDARIES}


class OracleLadderExporter(ArtifactExporter):
    test_point = TEST_POINT
    input_contract = "assessment_report.v1"
    output_contract = INPUT_SCHEMA

    def export(self, context: ExportContext) -> ArtifactBundle:
        reports, index_reason = _completed_boundary_reports(context.input_root)
        payload = {"schema_version": INPUT_SCHEMA, "reports": reports, "index_reason": index_reason}
        artifact = context.store.write_json(TEST_POINT, "inputs", payload)
        return ArtifactBundle(TEST_POINT, {"inputs": artifact.name})


class OracleLadderEvaluator(ArtifactEvaluator):
    test_point = TEST_POINT
    required_artifacts: Sequence[str] = ("inputs",)

    def evaluate(self, context: EvaluationContext, bundle: ArtifactBundle) -> EvaluationResult:
        inputs = _read_json(context.store.run_dir / bundle.artifacts["inputs"])
        if inputs.get("schema_version") != INPUT_SCHEMA:
            raise ValueError("Unsupported oracle-ladder inputs schema.")
        resolver = VerifiedArtifactResolver(context.input_root)
        references = {item["test_point"]: item for item in inputs.get("reports", [])}
        reports = {test_point: resolver.json(reference) for test_point, reference in references.items()}
        boundaries = _boundaries(reports, references, inputs.get("index_reason"))
        report = {
            "schema_version": "oracle_ladder_report.v1",
            "status": _overall_status(boundaries),
            "boundaries": boundaries,
            "gaps": _unavailable_gaps(),
            "provenance": {"input_reports": references},
            "missing_inputs": [
                {"boundary": boundary["boundary_id"], "reason": boundary["reason"]}
                for boundary in boundaries
                if boundary["reason"]
            ],
        }
        return EvaluationResult(report, _markdown(report))


ORACLE_LADDER_MODULE = EvaluationModule(
    TEST_POINT,
    OracleLadderExporter(),
    OracleLadderEvaluator(),
    summary="Information-flow oracle ladder without cross-space score aggregation.",
)


def _completed_boundary_reports(root) -> tuple[list[dict[str, str]], str | None]:
    index_path = root / "evaluation_index.json"
    if not index_path.is_file():
        return [], "The run has no evaluation_index.json; oracle evidence is discovered only from the runner index."
    index = _read_json(index_path)
    if index.get("schema_version") != "evaluation_run_index.v1":
        return [], "The evaluation index has an unsupported schema."
    reports: list[dict[str, str]] = []
    for test_point, entry in index.get("modules", {}).items():
        evaluation = entry.get("evaluate", {})
        report_name = evaluation.get("report")
        if test_point in BOUNDARY_MODULES and evaluation.get("status") == "COMPLETE" and isinstance(report_name, str):
            report_path = root / report_name
            if report_path.is_file():
                reports.append({"test_point": test_point, **_reference(report_path)})
    return reports, None


def _boundaries(
    reports: Mapping[str, Mapping[str, Any]],
    references: Mapping[str, Mapping[str, str]],
    index_reason: str | None,
) -> list[dict[str, Any]]:
    boundaries: list[dict[str, Any]] = []
    for boundary in BOUNDARIES:
        report = reports.get(boundary.module)
        reason = None
        if report is None:
            reason = index_reason or "The runner index has no completed report for this boundary."
        boundaries.append({
            "boundary_id": boundary.boundary_id,
            "label": boundary.label,
            "source_module": boundary.module,
            "status": report.get("status", "UNAVAILABLE") if report else "UNAVAILABLE",
            "evidence_report": references.get(boundary.module),
            "reason": reason,
        })
    return boundaries


def _unavailable_gaps() -> list[dict[str, str]]:
    return [
        {
            "from_boundary": BOUNDARIES[index].boundary_id,
            "to_boundary": BOUNDARIES[index + 1].boundary_id,
            "status": "UNAVAILABLE",
            "reason": "The available reports use different representations or metric definitions; no same-unit paired metric gap is asserted.",
        }
        for index in range(len(BOUNDARIES) - 1)
    ]


def _overall_status(boundaries: Sequence[Mapping[str, Any]]) -> str:
    statuses = [boundary["status"] for boundary in boundaries]
    if "FAIL" in statuses:
        return "FAIL"
    if "WARN" in statuses:
        return "WARN"
    if any(status != "UNAVAILABLE" for status in statuses):
        return "MONITOR"
    return "UNAVAILABLE"


def _reference(path) -> dict[str, str]:
    return {"path": path.name, "sha256": VerifiedArtifactResolver.sha256(path)}


def _read_json(path) -> dict[str, Any]:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Oracle Ladder 信息流证据",
        "",
        "该表逐段列出已由 runner index 记录的证据。不同表示空间的数值不会被强制合成为总分或损失百分比。",
        "",
        "| 信息边界 | 证据模块 | 状态 |",
        "| --- | --- | --- |",
    ]
    lines.extend(
        f"| {boundary['label']} | {boundary['source_module']} | {boundary['status']} |"
        for boundary in report["boundaries"]
    )
    lines.extend([
        "",
        "## Metric Gap",
        "",
        "当前没有任何相邻边界同时提供同单位、同样本的配对指标，因此所有 gap 均为 `UNAVAILABLE`。这表示不可比较，而不是音乐质量结论。",
        "",
    ])
    return "\n".join(lines)
