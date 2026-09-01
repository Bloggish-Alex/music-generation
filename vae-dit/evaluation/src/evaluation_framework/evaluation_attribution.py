"""Artifact-only paired theme-memory attribution comparison."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .core.tensor_schema import SemanticTensorDecoder
from .evaluation_api import (
    ArtifactBundle,
    ArtifactEvaluator,
    ArtifactExporter,
    EvaluationModule,
    EvaluationResult,
)
from .evaluation_context import EvaluationContext, ExportContext


TEST_POINT = "attribution"
INPUT_SCHEMA = "attribution_inputs.v1"
RAW_STATUS_SCHEMA = "attribution_raw_status.v3"


class AttributionExporter(ArtifactExporter):
    test_point = TEST_POINT
    input_contract = RAW_STATUS_SCHEMA
    output_contract = INPUT_SCHEMA

    def export(self, context: ExportContext) -> ArtifactBundle:
        status_path = context.input_root / "attribution__raw_status.v3.json"
        payload: dict[str, Any] = {
            "schema_version": INPUT_SCHEMA,
            "status": None,
            "status_reference": None,
        }
        if status_path.is_file():
            status = _read_json(status_path)
            if status.get("schema_version") != RAW_STATUS_SCHEMA:
                raise ValueError("Unsupported attribution status schema.")
            payload.update(
                status=status.get("status"),
                status_reference={"path": status_path.name, "sha256": _sha256(status_path)},
            )
        artifact = context.store.write_json(TEST_POINT, "inputs", payload)
        return ArtifactBundle(TEST_POINT, {"inputs": artifact.name})


class AttributionEvaluator(ArtifactEvaluator):
    test_point = TEST_POINT
    required_artifacts: Sequence[str] = ("inputs",)

    def evaluate(self, context: EvaluationContext, bundle: ArtifactBundle) -> EvaluationResult:
        inputs = _read_json(context.store.run_dir / bundle.artifacts["inputs"])
        status_reference = inputs.get("status_reference")
        if not status_reference:
            return _unavailable("No attribution raw status artifact was provided.")

        status = _read_reference(context.input_root, status_reference)
        if status.get("status") != "AVAILABLE":
            return _unavailable(_unavailable_reason(status, "Attribution pair is unavailable."))

        observation = _read_reference(context.input_root, status["artifacts"]["observation"])
        arms = {arm["arm_id"]: arm for arm in observation["arms"]}
        metrics = _compare_arms(
            _load_arm(context.input_root, arms["baseline"]),
            _load_arm(context.input_root, arms["ablated"]),
        )
        report = {
            "schema_version": "assessment_report.v1",
            "status": "MONITOR",
            "metrics": metrics,
            "findings": [
                {
                    "classification": "controlled_input_stream_attribution",
                    "text": "该比较在相同随机身份下，仅改变 theme-memory mask。它报告差异，不转换为单一质量分数或因果百分比。",
                },
                {
                    "classification": "ablation_interpretation",
                    "text": "masked 表示模型未读取 theme-memory；结果只适用于此明确 intervention。",
                },
            ],
            "provenance": {
                "experiment": observation["experiment"],
                "raw_observation_hash": status["artifacts"]["observation"]["sha256"],
            },
            "missing_inputs": _semantic_missing_inputs(metrics),
        }
        return EvaluationResult(report, _markdown(report), {"paired_difference": _png(metrics)})


ATTRIBUTION_MODULE = EvaluationModule(
    TEST_POINT,
    AttributionExporter(),
    AttributionEvaluator(),
    summary="Controlled paired input-stream attribution.",
)


def _load_arm(root: Path, arm: Mapping[str, Any]) -> dict[str, Any]:
    manifest_path = root / f"attribution_{arm['arm_id']}__evaluation_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing public generation manifest: {manifest_path.name}")
    manifest = _read_json(manifest_path)
    references = manifest["artifacts"]
    tensor_path = _reference_path(root, references["bar_tensors"])
    return {
        "arrays": np.load(tensor_path, allow_pickle=False),
        "manifest": manifest,
    }


def _compare_arms(baseline: Mapping[str, Any], ablated: Mapping[str, Any]) -> dict[str, Any]:
    baseline_arrays = baseline["arrays"]
    ablated_arrays = ablated["arrays"]
    try:
        baseline_bars, baseline_bases = _bar_arrays(baseline_arrays)
        ablated_bars, ablated_bases = _bar_arrays(ablated_arrays)
        count = min(len(baseline_bars), len(ablated_bars), len(baseline_bases), len(ablated_bases))
        baseline_bars, ablated_bars = baseline_bars[:count], ablated_bars[:count]
        baseline_bases, ablated_bases = baseline_bases[:count], ablated_bases[:count]
        return {
            "ablation": {"stream": "theme_memory", "type": "masked"},
            "bar_count": int(count),
            "tensor_mse": float(np.mean((baseline_bars - ablated_bars) ** 2)) if count else None,
            "register_path_mae_semitones": float(np.mean(np.abs(baseline_bases - ablated_bases))) if count else None,
            "register_path_max_difference_semitones": float(np.max(np.abs(baseline_bases - ablated_bases))) if count else None,
            "semantic_pitch_density": _semantic_pitch_density(
                baseline_bars,
                baseline_bases,
                ablated_bars,
                ablated_bases,
                baseline["manifest"].get("tensor_schema"),
                count,
            ),
        }
    finally:
        baseline_arrays.close()
        ablated_arrays.close()


def _bar_arrays(arrays: Any) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(arrays["bars"], dtype=float),
        np.asarray(arrays["render_base_pitches"], dtype=float).reshape(-1),
    )


def _semantic_pitch_density(
    baseline_bars: np.ndarray,
    baseline_bases: np.ndarray,
    ablated_bars: np.ndarray,
    ablated_bases: np.ndarray,
    schema: Any,
    count: int,
) -> dict[str, Any]:
    if not count:
        return _semantic_unavailable("The paired arms have no aligned bars.")
    if not isinstance(schema, Mapping):
        return _semantic_unavailable("The baseline public manifest has no bar_tensor_schema.v1 fact.")
    try:
        decoder = SemanticTensorDecoder.from_schema(schema)
    except ValueError as error:
        return _semantic_unavailable(str(error))

    baseline_active = decoder.active_mask(baseline_bars)
    ablated_active = decoder.active_mask(ablated_bars)
    baseline_pitch = decoder.absolute_pitch(baseline_bars, baseline_bases)
    ablated_pitch = decoder.absolute_pitch(ablated_bars, ablated_bases)
    chroma_cosines = [
        _cosine(_chroma(baseline_pitch[index][baseline_active[index]]), _chroma(ablated_pitch[index][ablated_active[index]]))
        for index in range(count)
    ]
    return {
        "status": "AVAILABLE",
        "chroma_cosine_mean": float(np.mean(chroma_cosines)),
        "density_slot_difference_mean": float(
            np.mean(np.abs(_active_slot_count(baseline_active) - _active_slot_count(ablated_active)))
        ),
        "unavailable_reasons": [],
    }


def _semantic_unavailable(reason: str) -> dict[str, Any]:
    return {
        "status": "UNAVAILABLE",
        "chroma_cosine_mean": None,
        "density_slot_difference_mean": None,
        "unavailable_reasons": [reason],
    }


def _active_slot_count(active: np.ndarray) -> np.ndarray:
    return np.sum(active, axis=(1, 2))


def _chroma(pitches: np.ndarray) -> np.ndarray:
    chroma = np.zeros(12)
    for pitch in pitches:
        chroma[int(round(float(pitch))) % 12] += 1
    return chroma


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(left, right) / max(np.linalg.norm(left) * np.linalg.norm(right), 1e-8))


def _semantic_missing_inputs(metrics: Mapping[str, Any]) -> list[dict[str, str]]:
    semantic = metrics["semantic_pitch_density"]
    return [
        {"field": "tensor_schema", "reason": reason}
        for reason in semantic["unavailable_reasons"]
    ]


def _unavailable_reason(status: Mapping[str, Any], fallback: str) -> str:
    reasons = [item.get("reason", "") for item in status.get("unavailable_reasons", [])]
    return "; ".join(reason for reason in reasons if reason) or fallback


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _reference_path(root: Path, reference: Mapping[str, Any]) -> Path:
    path = (root / str(reference["path"])).resolve()
    path.relative_to(root.resolve())
    if not path.is_file() or _sha256(path) != reference["sha256"]:
        raise ValueError(f"Attribution artifact hash mismatch: {path.name}")
    return path


def _read_reference(root: Path, reference: Mapping[str, Any]) -> dict[str, Any]:
    return _read_json(_reference_path(root, reference))


def _unavailable(reason: str) -> EvaluationResult:
    report = {
        "schema_version": "assessment_report.v1",
        "status": "UNAVAILABLE",
        "metrics": {},
        "findings": [],
        "provenance": {},
        "missing_inputs": [{"field": "attribution", "reason": reason}],
    }
    return EvaluationResult(report, _markdown(report))


def _markdown(report: Mapping[str, Any]) -> str:
    if report["status"] == "UNAVAILABLE":
        return "# 输入流归因\n\n生成这份受控对照报告所需的原始配对数据不可用。\n- " + report["missing_inputs"][0]["reason"] + "\n"

    metrics = report["metrics"]
    semantic = metrics["semantic_pitch_density"]
    value = lambda item: "--" if item is None else f"{item:.3f}"
    rows = [
        ("配对小节", str(metrics["bar_count"])),
        ("Tensor MSE", value(metrics["tensor_mse"])),
        ("音区路径 MAE（半音）", value(metrics["register_path_mae_semitones"])),
    ]
    if semantic["status"] == "AVAILABLE":
        rows.extend(
            [
                ("Chroma cosine", value(semantic["chroma_cosine_mean"])),
                ("活动 slot 密度差", value(semantic["density_slot_difference_mean"])),
            ]
        )
    else:
        rows.append(("语义音高与密度", "不可用：" + semantic["unavailable_reasons"][0]))
    table = "\n".join(f"| {label} | {result} |" for label, result in rows)
    return (
        "# Theme-memory 输入流归因\n\n"
        "同一生成身份下比较正常 theme memory 与 masked theme memory。数值是两臂差异，不是音乐质量总分。\n\n"
        "| 比较项 | 值 |\n| --- | ---: |\n"
        + table
        + "\n"
    )


def _png(metrics: Mapping[str, Any]) -> bytes:
    try:
        import matplotlib.pyplot as plt

        semantic = metrics["semantic_pitch_density"]
        chroma_gap = 0.0
        if semantic["chroma_cosine_mean"] is not None:
            chroma_gap = 1.0 - semantic["chroma_cosine_mean"]
        figure, axis = plt.subplots(figsize=(5, 3))
        axis.bar(
            ["Tensor MSE", "Register MAE", "1-Chroma cosine"],
            [metrics["tensor_mse"] or 0, metrics["register_path_mae_semitones"] or 0, chroma_gap],
        )
        figure.tight_layout()
        output = io.BytesIO()
        figure.savefig(output, format="png", dpi=150)
        plt.close(figure)
        return output.getvalue()
    except Exception:
        return b""
