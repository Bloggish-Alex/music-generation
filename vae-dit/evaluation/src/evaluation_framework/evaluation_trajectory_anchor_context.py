"""Artifact-only verification of encoded anchor context entering trajectory training."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .evaluation_api import ArtifactBundle, ArtifactEvaluator, ArtifactExporter, EvaluationModule, EvaluationResult
from .evaluation_context import EvaluationContext, ExportContext


TEST_POINT = "trajectory_anchor_context"
PARENT_SCHEMA = "encoded_input_manifest.v1"
RAW_SCHEMAS = (
    "trajectory_training_input_lineage_raw.v2",
    "trajectory_training_input_lineage_raw.v1",
)
INPUT_SCHEMA = "trajectory_anchor_context_inputs.v1"
_STAGES = (
    "model_preparation_to_training_input",
    "empty_bar_fallback_to_training_input",
    "training_anchor_derivation",
    "training_offset_reconstruction",
    "generation_runtime_to_renderer",
)


class TrajectoryAnchorContextExporter(ArtifactExporter):
    """Index copied raw lineage facts without deriving musical values."""

    test_point = TEST_POINT
    input_contract = RAW_SCHEMAS[0]
    output_contract = INPUT_SCHEMA

    def export(self, context: ExportContext) -> ArtifactBundle:
        parent = _reference_if_present(
            context.input_root / "trajectory_anchor_context__encoded_input_manifest.v1.json",
            PARENT_SCHEMA,
        )
        lineage = _lineage_reference(context.input_root)
        payload = {
            "schema_version": INPUT_SCHEMA,
            "artifacts": {
                "encoded_input_manifest": parent,
                "training_lineage_raw": lineage,
            },
            "availability": {
                "model_preparation": "available" if parent else "not_provided",
                "trajectory_training": "available" if lineage else "not_provided",
                "generation_runtime": "not_provided",
            },
        }
        path = context.store.write_json(TEST_POINT, "inputs", payload)
        return ArtifactBundle(TEST_POINT, {"inputs": path.name})


class TrajectoryAnchorContextEvaluator(ArtifactEvaluator):
    """Validate training anchor transport and declare absent runtime evidence."""

    test_point = TEST_POINT
    required_artifacts: Sequence[str] = ("inputs",)

    def evaluate(self, context: EvaluationContext, bundle: ArtifactBundle) -> EvaluationResult:
        inputs = _read_json(context.store.run_dir / bundle.artifacts["inputs"])
        if inputs.get("schema_version") != INPUT_SCHEMA:
            raise ValueError("Unsupported trajectory anchor-context input schema.")

        parent_ref = _reference(inputs, "encoded_input_manifest")
        raw_ref = _reference(inputs, "training_lineage_raw")
        measurements = _unavailable_measurements(inputs)
        missing = _missing_inputs(parent_ref, raw_ref)
        if not missing:
            parent = _load_referenced(context.input_root, parent_ref, PARENT_SCHEMA)
            raw = _load_referenced(context.input_root, raw_ref, str(raw_ref["schema_version"]))
            measurements.update(_measure(parent, raw, str(parent_ref["sha256"])))

        report = _report(inputs, measurements, missing)
        return EvaluationResult(
            report=report,
            markdown=_markdown(report),
            figures={"context_summary": _summary_png(measurements)},
        )


TRAJECTORY_ANCHOR_CONTEXT_MODULE = EvaluationModule(
    TEST_POINT,
    TrajectoryAnchorContextExporter(),
    TrajectoryAnchorContextEvaluator(),
    summary="Verifies encoded base-pitch, declared song anchor and register-offset facts entering trajectory training.",
)


def _reference_if_present(path: Path, expected_schema: str) -> dict[str, str] | None:
    if not path.is_file():
        return None
    payload = _read_json(path)
    if payload.get("schema_version") != expected_schema:
        raise ValueError(f"Unsupported trajectory anchor-context artifact schema: {path.name}")
    return {"path": path.name, "sha256": _sha256(path), "schema_version": expected_schema}


def _lineage_reference(root: Path) -> dict[str, str] | None:
    for version in ("v2", "v1"):
        reference = _reference_if_present(
            root / f"trajectory_anchor_context__training_lineage_raw.{version}.json",
            f"trajectory_training_input_lineage_raw.{version}",
        )
        if reference is not None:
            return reference
    return None


def _reference(inputs: Mapping[str, Any], name: str) -> Mapping[str, Any] | None:
    artifacts = inputs.get("artifacts")
    value = artifacts.get(name) if isinstance(artifacts, Mapping) else None
    return value if isinstance(value, Mapping) else None


def _missing_inputs(parent: Mapping[str, Any] | None, raw: Mapping[str, Any] | None) -> list[dict[str, str]]:
    missing: list[dict[str, str]] = []
    if parent is None:
        missing.append({"artifact": "encoded_input_manifest", "reason": "Model-preparation parent manifest was not provided."})
    if raw is None:
        missing.append({"artifact": "trajectory_training_input_lineage_raw", "reason": "Trajectory-training raw lineage was not provided."})
    return missing


def _unavailable_measurements(inputs: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    availability = inputs.get("availability") if isinstance(inputs.get("availability"), Mapping) else {}
    return {
        stage: {
            "status": "UNAVAILABLE",
            "comparable_count": 0,
            "comparable_unit": "bar",
            "exact_match_ratio": None,
            "reasons": ["Required raw lineage evidence was not provided."],
        }
        for stage in _STAGES[:-1]
    } | {
        _STAGES[-1]: {
            "status": "UNAVAILABLE",
            "comparable_count": 0,
            "comparable_unit": "bar",
            "exact_match_ratio": None,
            "reasons": [
                "Generation runtime and renderer-input observations are not provided by this contract.",
                f"generation_runtime availability is {availability.get('generation_runtime', 'not_provided')}.",
            ],
        }
    }


def _measure(
    parent: Mapping[str, Any],
    raw: Mapping[str, Any],
    parent_sha256: str,
) -> dict[str, dict[str, Any]]:
    _validate_parent(parent)
    _validate_raw(raw)
    rows = raw["observations"]
    parent_errors = _parent_errors(parent, raw, parent_sha256)
    serialized_rows = [row for row in rows if row.get("base_pitch_origin", "serialized_anchor") == "serialized_anchor"]
    fallback_rows = [row for row in rows if row.get("base_pitch_origin") == "runtime_fallback"]
    base_errors = [row["trajectory_input_base_pitch"] - row["encoded_base_pitch"] for row in serialized_rows]
    fallback_errors = _fallback_errors(raw, fallback_rows)
    anchor_errors = _song_anchor_errors(rows)
    offset_errors = [row["register_offset"] - (row["trajectory_input_base_pitch"] - row["song_anchor"]) for row in rows]
    return {
        "model_preparation_to_training_input": _exact_measurement(
            len(serialized_rows),
            "bar",
            base_errors,
            parent_errors,
        ),
        "empty_bar_fallback_to_training_input": _fallback_measurement(fallback_rows, fallback_errors),
        "training_anchor_derivation": _exact_measurement(
            len(anchor_errors),
            "song",
            anchor_errors,
            [],
        ),
        "training_offset_reconstruction": _exact_measurement(
            len(rows),
            "bar",
            offset_errors,
            [],
        ),
        "generation_runtime_to_renderer": _unavailable_measurements({})["generation_runtime_to_renderer"],
    }


def _validate_parent(parent: Mapping[str, Any]) -> None:
    if parent.get("schema_version") != PARENT_SCHEMA:
        raise ValueError("Unsupported encoded-input manifest schema.")
    availability = parent.get("availability")
    required = ("bar_tensor_index", "bar_tensors", "songs", "stable_row_identity")
    if not isinstance(availability, Mapping) or any(availability.get(item) is not True for item in required):
        raise ValueError("Encoded-input manifest declares incomplete availability.")
    if parent.get("unavailable_reasons"):
        raise ValueError("Encoded-input manifest declares unavailable reasons.")


def _validate_raw(raw: Mapping[str, Any]) -> None:
    if raw.get("schema_version") not in RAW_SCHEMAS:
        raise ValueError("Unsupported trajectory-training raw lineage schema.")
    availability = raw.get("availability")
    required = (
        "parent_manifest",
        "encoded_index_hash_match",
        "encoded_tensor_hash_match",
        "row_identity_alignment",
        "base_pitch_alignment",
    )
    if not isinstance(availability, Mapping) or any(availability.get(name) is not True for name in required):
        raise ValueError("Trajectory-training raw lineage declares incomplete availability.")
    if raw.get("unavailable_reasons"):
        raise ValueError("Trajectory-training raw lineage declares unavailable reasons.")
    rows = raw.get("observations")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Trajectory-training raw lineage has no comparable observations.")


def _fallback_errors(raw: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> list[int]:
    parameterization = raw.get("register_parameterization")
    if not isinstance(parameterization, Mapping):
        return [1 for _ in rows]
    fallback = parameterization.get("fallback_base_pitch")
    if not isinstance(fallback, int) or isinstance(fallback, bool):
        return [1 for _ in rows]
    return [int(row["trajectory_input_base_pitch"]) - fallback for row in rows]


def _fallback_measurement(
    rows: Sequence[Mapping[str, Any]], errors: Sequence[int],
) -> dict[str, Any]:
    if not rows:
        return {
            "status": "UNAVAILABLE",
            "comparable_count": 0,
            "comparable_unit": "bar",
            "exact_match_ratio": None,
            "reasons": ["No empty bar used the configured runtime fallback."],
        }
    return _exact_measurement(len(rows), "bar", errors, [])


def _parent_errors(
    parent: Mapping[str, Any],
    raw: Mapping[str, Any],
    parent_sha256: str,
) -> list[str]:
    raw_parent = raw.get("parent_encoded_input_manifest")
    artifacts = parent.get("encoded_artifacts")
    if not isinstance(raw_parent, Mapping) or not isinstance(artifacts, Mapping):
        return ["Raw lineage does not provide parent-manifest provenance."]
    expected = {
        "sha256": parent_sha256,
        "run_identity": _value(parent, "run", "identity"),
        "bar_tensor_index_sha256": _value(artifacts, "bar_tensor_index", "sha256"),
        "bar_tensors_sha256": _value(artifacts, "bar_tensors", "sha256"),
        "tensor_schema_version": _value(parent, "tensor_identity", "tensor_schema_version"),
    }
    return [f"Parent provenance mismatch: {name}." for name, value in expected.items() if raw_parent.get(name) != value]


def _value(mapping: Mapping[str, Any], name: str, field: str) -> Any:
    nested = mapping.get(name)
    return nested.get(field) if isinstance(nested, Mapping) else None


def _song_anchor_errors(rows: Sequence[Mapping[str, Any]]) -> list[int]:
    grouped: dict[str, list[int]] = {}
    anchors: dict[str, int] = {}
    for row in rows:
        song_id = str(row["song_id"])
        grouped.setdefault(song_id, []).append(int(row["trajectory_input_base_pitch"]))
        anchors[song_id] = int(row["song_anchor"])
    return [anchors[song_id] - int(np.rint(np.median(pitches))) for song_id, pitches in grouped.items()]


def _exact_measurement(
    comparable_count: int,
    comparable_unit: str,
    errors: Sequence[int],
    reasons: Sequence[str],
) -> dict[str, Any]:
    if reasons:
        return {
            "status": "FAIL",
            "comparable_count": comparable_count,
            "comparable_unit": comparable_unit,
            "exact_match_ratio": 0.0,
            "reasons": list(reasons),
        }
    errors_array = np.asarray(errors, dtype=int)
    matches = errors_array == 0
    return {
        "status": "PASS" if bool(np.all(matches)) else "FAIL",
        "comparable_count": comparable_count,
        "comparable_unit": comparable_unit,
        "exact_match_ratio": float(np.mean(matches)),
        "mae_semitones": float(np.mean(np.abs(errors_array))),
        "reasons": [] if bool(np.all(matches)) else ["A declared deterministic anchor relation has nonzero error."],
    }


def _report(inputs: Mapping[str, Any], measurements: Mapping[str, Mapping[str, Any]], missing: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    statuses = [measurement["status"] for measurement in measurements.values()]
    status = "FAIL" if "FAIL" in statuses else "UNAVAILABLE" if "UNAVAILABLE" in statuses else "PASS"
    findings = [
        {
            "classification": "missing_runtime_evidence",
            "stage": "generation_runtime_to_renderer",
            "text": "生成与渲染阶段尚未输出可连接的运行时锚点观察，因此不能据此判断该段是否保持同一绝对音区参考。",
        }
    ]
    return {
        "schema_version": "assessment_report.v1",
        "status": status,
        "metrics": {"stages": measurements},
        "findings": findings,
        "provenance": {"input_artifacts": inputs.get("artifacts", {})},
        "missing_inputs": list(missing),
    }


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Trajectory 音区参考链路",
        "",
        "本报告检查编码后的绝对音高参考是否完整进入 trajectory 训练。它不评价模型未来音区预测，也不把训练整曲的参考音高与生成 primer 的参考音高当作同一数值。",
        "",
        "| 环节 | 状态 | 可比较对象 | 精确匹配率 |",
        "| --- | --- | ---: | ---: |",
    ]
    for stage, measurement in report["metrics"]["stages"].items():
        ratio = measurement.get("exact_match_ratio")
        ratio_text = "--" if ratio is None else f"{ratio:.3f}"
        count = measurement["comparable_count"]
        unit = "小节" if measurement["comparable_unit"] == "bar" else "首曲"
        lines.append(f"| {stage.replace('_', ' ')} | {measurement['status']} | {count} {unit} | {ratio_text} |")
    lines.extend([
        "",
        "前三项是确定性数据关系：编码 `base_pitch` 与训练输入必须逐小节相同；每首曲的 `song_anchor` 必须等于该曲输入 `base_pitch` 的中位数四舍五入；`register_offset` 必须等于 `base_pitch - song_anchor`。",
        "",
        "最后一项为 `UNAVAILABLE` 时表示尚未收集 generation runtime 与 renderer input 的共同观察。它是证据缺口，不代表音乐质量或模型质量差。",
        "",
    ])
    return "\n".join(lines)


def _summary_png(measurements: Mapping[str, Mapping[str, Any]]) -> bytes:
    try:
        import matplotlib.pyplot as plt

        labels = [stage.replace("_", " ") for stage in _STAGES]
        values = [measurements[stage].get("exact_match_ratio") for stage in _STAGES]
        figure, axis = plt.subplots(figsize=(9, 3.8))
        positions = np.arange(len(labels))
        observed = [index for index, value in enumerate(values) if value is not None]
        if observed:
            colors = ["#2E8B57" if values[index] == 1.0 else "#C94C4C" for index in observed]
            axis.barh(observed, [values[index] for index in observed], color=colors, height=0.55)
        for index, value in enumerate(values):
            if value is None:
                axis.text(0.02, index, "N/A", va="center", color="#666666", fontweight="bold")
                axis.hlines(index, 0.0, 1.0, color="#B0B0B0", linewidth=1.0, linestyles="dashed")
        axis.set_yticks(positions, labels)
        axis.set_xlim(0.0, 1.05)
        axis.set_xlabel("Exact-match ratio; N/A = runtime observation unavailable")
        axis.set_title("Trajectory anchor-context evidence", loc="left", fontsize=11)
        axis.grid(axis="x", color="#D9D9D9", linewidth=0.7)
        axis.set_axisbelow(True)
        figure.tight_layout()
        output = io.BytesIO()
        figure.savefig(output, format="png", dpi=160)
        plt.close(figure)
        return output.getvalue()
    except Exception:
        return bytes.fromhex("89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4890000000d49444154789c6360f8cff0ff3f0005fe02fe8e4cacf50000000049454e44ae426082")


def _load_referenced(root: Path, reference: Mapping[str, Any], expected_schema: str) -> dict[str, Any]:
    path = (root / str(reference["path"])).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("Trajectory anchor-context artifact path must stay below input root.") from error
    if not path.is_file():
        raise FileNotFoundError(f"Missing trajectory anchor-context artifact: {path.name}")
    if _sha256(path) != reference.get("sha256"):
        raise ValueError(f"Trajectory anchor-context artifact hash mismatch: {path.name}")
    payload = _read_json(path)
    if payload.get("schema_version") != expected_schema:
        raise ValueError(f"Unsupported trajectory anchor-context artifact schema: {path.name}")
    return payload


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"
