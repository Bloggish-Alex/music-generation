"""Artifact-only validation of deterministic absolute-anchor transport."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .evaluation_api import ArtifactBundle, ArtifactEvaluator, ArtifactExporter, EvaluationModule, EvaluationResult
from .evaluation_context import EvaluationContext, ExportContext


TEST_POINT = "anchor_transport"
RAW_SCHEMA = "anchor_transport_raw_observation.v1"
INPUT_SCHEMA = "anchor_transport_inputs.v1"
_SPLITS = ("train", "validation", "excluded_unpaired")
_BOUNDARIES = (
    ("source_to_encoding", "source", "encoding"),
    ("encoding_to_export", "encoding", "exported"),
    ("export_to_trajectory_input", "exported", "trajectory_input"),
    ("trajectory_input_to_renderer_input", "trajectory_input", "renderer_input"),
)


class AnchorTransportExporter(ArtifactExporter):
    """Index raw anchor observations without deriving anchor values."""

    test_point = TEST_POINT
    input_contract = RAW_SCHEMA
    output_contract = INPUT_SCHEMA

    def export(self, context: ExportContext) -> ArtifactBundle:
        splits: dict[str, dict[str, str]] = {}
        availability: dict[str, str] = {}
        for split in _SPLITS:
            path = context.input_root / f"anchor_transport__raw_observation__{split}.v1.json"
            if not path.is_file():
                availability[split] = "not_provided"
                continue
            payload = _read_json(path)
            _validate_raw(payload, split)
            splits[split] = {"path": path.name, "sha256": _sha256(path), "schema_version": RAW_SCHEMA}
            availability[split] = "available"
        inputs = {"schema_version": INPUT_SCHEMA, "splits": splits, "availability": availability}
        path = context.store.write_json(TEST_POINT, "inputs", inputs)
        return ArtifactBundle(TEST_POINT, {"inputs": path.name})


class AnchorTransportEvaluator(ArtifactEvaluator):
    """Measure each deterministic boundary independently without a composite score."""

    test_point = TEST_POINT
    required_artifacts: Sequence[str] = ("inputs",)

    def evaluate(self, context: EvaluationContext, bundle: ArtifactBundle) -> EvaluationResult:
        inputs = _read_json(context.store.run_dir / bundle.artifacts["inputs"])
        if inputs.get("schema_version") != INPUT_SCHEMA:
            raise ValueError("Unsupported anchor-transport input schema.")
        split_reports = {
            split: _measure_split(_load_raw(context.input_root, reference), split)
            for split, reference in dict(inputs.get("splits") or {}).items()
        }
        if not split_reports:
            report = _unavailable_report(inputs, "No anchor-transport raw observation artifact was provided.")
            return EvaluationResult(report=report, markdown=_markdown(report))
        report = _report(split_reports, inputs)
        return EvaluationResult(report=report, markdown=_markdown(report), figures={"boundary_summary": _summary_png(split_reports)})


ANCHOR_TRANSPORT_MODULE = EvaluationModule(
    TEST_POINT,
    AnchorTransportExporter(),
    AnchorTransportEvaluator(),
    summary="Deterministic canonical-anchor transport from semantic codec input to runtime boundaries.",
)


def _validate_raw(payload: Mapping[str, Any], expected_split: str) -> None:
    if payload.get("schema_version") != RAW_SCHEMA:
        raise ValueError("Unsupported anchor-transport raw schema version.")
    dataset = payload.get("dataset")
    if not isinstance(dataset, Mapping) or dataset.get("split") != expected_split:
        raise ValueError(f"Anchor-transport raw split does not match filename: {expected_split}.")
    spec = payload.get("anchor_spec")
    if not isinstance(spec, Mapping) or spec.get("policy_id") != "canonical_bar_anchor_v1":
        raise ValueError("Anchor-transport raw observation does not declare canonical_bar_anchor_v1.")
    observations = payload.get("observations")
    if not isinstance(observations, list):
        raise ValueError("Anchor-transport raw observation must contain an observations array.")
    availability = payload.get("availability")
    if not isinstance(availability, Mapping):
        raise ValueError("Anchor-transport raw observation must declare availability.")
    expected_boundaries = [item[0] for item in _BOUNDARIES]
    actual_boundaries = [item.get("name") for item in payload.get("boundaries", []) if isinstance(item, Mapping)]
    if actual_boundaries != expected_boundaries:
        raise ValueError("Anchor-transport raw observation declares unsupported deterministic boundaries.")


def _load_raw(root: Path, reference: Mapping[str, Any]) -> dict[str, Any]:
    path = _resolve_reference(root, reference)
    if _sha256(path) != reference.get("sha256"):
        raise ValueError(f"Anchor-transport raw artifact hash mismatch: {path.name}")
    payload = _read_json(path)
    _validate_raw(payload, str(payload["dataset"]["split"]))
    return payload


def _measure_split(raw: Mapping[str, Any], split: str) -> dict[str, Any]:
    rows = raw.get("observations") or []
    availability = raw["availability"]
    boundaries = {
        name: _measure_boundary(rows, availability, name, source_name, destination_name)
        for name, source_name, destination_name in _BOUNDARIES
    }
    return {"dataset": raw["dataset"], "split": split, "observation_count": len(rows), "boundaries": boundaries}


def _measure_boundary(
    rows: Sequence[Mapping[str, Any]],
    availability: Mapping[str, Any],
    name: str,
    source_name: str,
    destination_name: str,
) -> dict[str, Any]:
    errors: list[float] = []
    unavailable_rows = 0
    for row in rows:
        anchors = row.get("anchors") if isinstance(row.get("anchors"), Mapping) else {}
        source = _anchor_number(anchors.get(source_name))
        destination = _anchor_number(anchors.get(destination_name))
        if source is None and destination is None:
            continue  # Canonical null for an empty bar is not a comparison failure.
        if source is None or destination is None:
            unavailable_rows += 1
            continue
        errors.append(destination - source)

    source_available = bool(availability.get(source_name))
    destination_available = bool(availability.get(destination_name))
    if not source_available or not destination_available or unavailable_rows:
        status = "UNAVAILABLE"
    elif not errors:
        status = "UNAVAILABLE"
    elif any(error != 0.0 for error in errors):
        status = "FAIL"
    else:
        status = "PASS"

    return {
        "status": status,
        "expected_relation": "exact_identity",
        "comparable_bar_count": len(errors),
        "unavailable_bar_count": unavailable_rows,
        "exact_match_ratio": float(np.mean(np.asarray(errors) == 0.0)) if errors else None,
        "signed_error_semitones": _distribution(errors),
        "mae_semitones": float(np.mean(np.abs(errors))) if errors else None,
        "rmse_semitones": float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
        "reasons": _boundary_reasons(availability, source_name, destination_name, unavailable_rows, errors),
    }


def _anchor_number(value: Any) -> float | None:
    if not isinstance(value, Mapping):
        return None
    number = value.get("value")
    return float(number) if isinstance(number, int) and not isinstance(number, bool) else None


def _boundary_reasons(
    availability: Mapping[str, Any], source_name: str, destination_name: str, unavailable_rows: int, errors: Sequence[float]
) -> list[str]:
    reasons: list[str] = []
    if not availability.get(source_name):
        reasons.append(f"{source_name} anchor is unavailable in this raw capture.")
    if not availability.get(destination_name):
        reasons.append(f"{destination_name} anchor is unavailable in this raw capture.")
    if unavailable_rows:
        reasons.append(f"{unavailable_rows} nonempty bar(s) do not provide both boundary values.")
    if errors and any(error != 0.0 for error in errors):
        reasons.append("A deterministic exact-identity boundary has nonzero anchor error.")
    return reasons


def _distribution(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"minimum": None, "p10": None, "median": None, "p90": None, "maximum": None}
    array = np.asarray(values, dtype=float)
    return {
        "minimum": float(array.min()), "p10": float(np.percentile(array, 10)), "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)), "maximum": float(array.max()),
    }


def _report(split_reports: Mapping[str, Mapping[str, Any]], inputs: Mapping[str, Any]) -> dict[str, Any]:
    statuses = [boundary["status"] for split in split_reports.values() for boundary in split["boundaries"].values()]
    status = "FAIL" if "FAIL" in statuses else "UNAVAILABLE" if "UNAVAILABLE" in statuses else "PASS"
    findings = []
    for split, split_report in split_reports.items():
        for name, boundary in split_report["boundaries"].items():
            if boundary["status"] == "FAIL":
                findings.append({"classification": "deterministic_transport_violation", "split": split, "boundary": name, "text": "确定性绝对锚点边界出现非零差异；这属于数据合同或对齐问题，不是模型预测误差。"})
            elif boundary["status"] == "UNAVAILABLE":
                findings.append({"classification": "missing_deterministic_evidence", "split": split, "boundary": name, "text": "该确定性边界缺少可比较的原始事实；未使用默认锚点推断结果。"})
    return {
        "schema_version": "assessment_report.v1", "status": status,
        "metrics": {"splits": split_reports}, "findings": findings,
        "provenance": {"input_artifacts": inputs.get("splits", {}), "anchor_policy": "canonical_bar_anchor_v1"}, "missing_inputs": [],
    }


def _unavailable_report(inputs: Mapping[str, Any], reason: str) -> dict[str, Any]:
    return {
        "schema_version": "assessment_report.v1", "status": "UNAVAILABLE", "metrics": {}, "findings": [],
        "provenance": {"input_availability": inputs.get("availability", {})},
        "missing_inputs": [{"artifact": "anchor_transport_raw_observation", "reason": reason}],
    }


def _markdown(report: Mapping[str, Any]) -> str:
    if report["status"] == "UNAVAILABLE" and not report["metrics"]:
        return "# 绝对锚点传递\n\n生成这份检查所需的锚点传递原始观察数据不可用。\n"
    lines = [
        "# 绝对锚点传递", "", "本报告检查已确定的音高坐标参考是否被原样传递。它不衡量模型对未来音区的预测。",
        "", "| 数据部分 | 确定性边界 | 状态 | 可比较小节 | 精确匹配率 | MAE（半音） |", "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for split, split_report in report["metrics"]["splits"].items():
        for name, boundary in split_report["boundaries"].items():
            ratio = "--" if boundary["exact_match_ratio"] is None else f"{boundary['exact_match_ratio']:.3f}"
            mae = "--" if boundary["mae_semitones"] is None else f"{boundary['mae_semitones']:.2f}"
            lines.append(f"| {split} | {name} | {boundary['status']} | {boundary['comparable_bar_count']} | {ratio} | {mae} |")
    lines += ["", "同一 policy、模型移调后坐标系且裁剪前的确定性边界，预期精确匹配率为 100%，MAE 为 0。非零差异应先按数据合同、移调参考系、单位或索引问题处理。`UNAVAILABLE` 表示未观察到所需事实，不代表音乐质量好或坏。", ""]
    return "\n".join(lines)


def _summary_png(split_reports: Mapping[str, Mapping[str, Any]]) -> bytes:
    try:
        import matplotlib.pyplot as plt

        split_count = len(split_reports)
        figure, axes = plt.subplots(
            nrows=split_count,
            ncols=1,
            figsize=(9, max(3.2, split_count * 2.8)),
            squeeze=False,
            sharex=True,
        )

        for axis, (split, split_report) in zip(axes[:, 0], split_reports.items()):
            boundaries = list(split_report["boundaries"].items())
            labels = [name.replace("_", " ") for name, _boundary in boundaries]
            ratios = [boundary["exact_match_ratio"] for _name, boundary in boundaries]
            positions = np.arange(len(boundaries))
            observed = [index for index, value in enumerate(ratios) if value is not None]
            unavailable = [index for index, value in enumerate(ratios) if value is None]

            if observed:
                observed_values = [float(ratios[index]) for index in observed]
                colors = ["#2E8B57" if value == 1.0 else "#C94C4C" for value in observed_values]
                axis.barh(np.asarray(observed), observed_values, color=colors, height=0.58)
            for index in unavailable:
                axis.text(0.02, index, "N/A", va="center", ha="left", color="#666666", fontweight="bold")
                axis.hlines(index, 0.0, 1.0, color="#B0B0B0", linewidth=1.0, linestyles="dashed")

            axis.set_yticks(positions, labels)
            axis.set_xlim(0.0, 1.05)
            axis.set_title(f"{split}: deterministic anchor boundaries", loc="left", fontsize=11)
            axis.grid(axis="x", color="#D9D9D9", linewidth=0.7)
            axis.set_axisbelow(True)

        axes[-1, 0].set_xlabel("Exact-match ratio; N/A = observation unavailable")
        figure.tight_layout()
        output = io.BytesIO()
        figure.savefig(output, format="png", dpi=160)
        plt.close(figure)
        return output.getvalue()
    except Exception:
        return bytes.fromhex("89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4890000000d49444154789c6360f8cff0ff3f0005fe02fe8e4cacf50000000049454e44ae426082")


def _resolve_reference(root: Path, reference: Mapping[str, Any]) -> Path:
    path = (root / str(reference["path"])).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("Anchor-transport artifact path must stay below input root.") from error
    if not path.is_file():
        raise FileNotFoundError(f"Missing anchor-transport artifact: {path.name}")
    return path


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"
