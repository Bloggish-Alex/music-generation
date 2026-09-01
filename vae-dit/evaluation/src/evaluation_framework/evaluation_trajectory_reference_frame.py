"""Register reference-frame diagnostic from paired semantic observations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .evaluation_api import ArtifactBundle, ArtifactEvaluator, ArtifactExporter, EvaluationModule, EvaluationResult
from .evaluation_context import EvaluationContext, ExportContext


TEST_POINT = "trajectory_reference_frame"
PAIR_MANIFEST_NAME = "paired_experiment_manifest.json"


class TrajectoryReferenceFrameExporter(ArtifactExporter):
    """Publish the two semantic arm references needed for this diagnostic."""

    test_point = TEST_POINT
    input_contract = "paired_experiment_manifest.v2 + trajectory_teacher_forced_semantic_future_position.v1"
    output_contract = "trajectory_reference_frame_inputs.v1"

    def export(self, context: ExportContext) -> ArtifactBundle:
        pair_path = context.input_root / PAIR_MANIFEST_NAME
        pair = _read_json(pair_path)
        arm_references = _semantic_arm_references(pair_path, pair)
        payload = {
            "schema_version": self.output_contract,
            "pairing_group_id": pair["pairing_group_id"],
            "shared_identity": pair["shared_identity"],
            "arms": arm_references,
            "provenance": {
                "pair_manifest": _relative(pair_path, context.input_root),
                "pair_manifest_sha256": _sha256(pair_path),
            },
        }
        path = context.store.write_json(TEST_POINT, "inputs", payload)
        return ArtifactBundle(TEST_POINT, {"inputs": path.name})


class TrajectoryReferenceFrameEvaluator(ArtifactEvaluator):
    """Separate an absolute register error from an arm-specific anchor change."""

    test_point = TEST_POINT
    required_artifacts: Sequence[str] = ("inputs",)

    def evaluate(self, context: EvaluationContext, bundle: ArtifactBundle) -> EvaluationResult:
        inputs = _read_json(context.store.run_dir / bundle.artifacts["inputs"])
        arms = {name: _load_semantic(reference, context.input_root) for name, reference in inputs["arms"].items()}
        _validate_shared_source(arms)
        values = {name: _measure_first_future_register(data) for name, data in arms.items()}
        report = {
            "schema_version": "assessment_report.v1",
            "status": "MONITOR",
            "metrics": {"first_future_register": values},
            "findings": [{
                "classification": "reference_frame_decomposition",
                "text": "Common-anchor delta error equals absolute register error by construction. Native-anchor delta differs only by the recorded history-anchor gap; the residual verifies that identity.",
            }],
            "provenance": {**inputs["provenance"], "pairing_group_id": inputs["pairing_group_id"], "shared_identity": inputs["shared_identity"]},
            "causal_language_policy": "This decomposition identifies coordinate effects in the reported metric. It does not by itself establish why either arm predicted a register value.",
        }
        return EvaluationResult(report=report, markdown=_markdown(report))


TRAJECTORY_REFERENCE_FRAME_MODULE = EvaluationModule(
    test_point=TEST_POINT,
    exporter=TrajectoryReferenceFrameExporter(),
    evaluator=TrajectoryReferenceFrameEvaluator(),
    summary="First-future register error decomposed into absolute prediction and history-anchor coordinates.",
)


def _semantic_arm_references(pair_path: Path, pair: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    if pair.get("schema_version") not in {"paired_experiment_manifest.v1", "paired_experiment_manifest.v2"}:
        raise ValueError("Unsupported paired experiment manifest schema.")
    result: dict[str, dict[str, str]] = {}
    for arm in pair.get("arms", []):
        name = str(arm.get("arm", ""))
        manifest_path = pair_path.parent / str(arm.get("artifact_manifest", ""))
        manifest = _read_json(manifest_path)
        metric_reference = manifest.get("artifacts", {}).get("generation_metric_inputs", {})
        metric_path = manifest_path.parent / str(metric_reference.get("path", ""))
        _verify(metric_path, metric_reference)
        metric_inputs = _read_json(metric_path)
        semantic_path = metric_path.parent / str(metric_inputs.get("future_position_semantic_path", ""))
        digest = metric_inputs.get("future_position_semantic_sha256")
        if name not in {"free_running", "teacher_forced"} or not digest:
            continue
        if _sha256(semantic_path) != digest:
            raise ValueError(f"Artifact hash mismatch for {semantic_path.name}")
        result[name] = {"path": _relative(semantic_path, pair_path.parent), "sha256": str(digest)}
    if set(result) != {"free_running", "teacher_forced"}:
        raise ValueError("Both paired arms must export future_position_semantic observations.")
    return result


def _load_semantic(reference: Mapping[str, Any], root: Path) -> dict[str, np.ndarray]:
    path = root / str(reference["path"])
    if _sha256(path) != reference["sha256"]:
        raise ValueError(f"Artifact hash mismatch for {path.name}")
    required = (
        "schema_version", "predicted_render_base_pitches", "context_render_base_pitches",
        "source_render_base_pitches", "committed_source_stream_indices", "target_source_stream_indices",
    )
    with np.load(path, allow_pickle=False) as archive:
        missing = [name for name in required if name not in archive]
        if missing:
            raise ValueError(f"Semantic observation is missing: {', '.join(missing)}")
        return {name: np.asarray(archive[name]) for name in required}


def _validate_shared_source(arms: Mapping[str, Mapping[str, np.ndarray]]) -> None:
    free, teacher = arms["free_running"], arms["teacher_forced"]
    for name in ("source_render_base_pitches", "committed_source_stream_indices", "target_source_stream_indices"):
        if not np.array_equal(free[name], teacher[name]):
            raise ValueError(f"Paired reference-frame observations must share {name}.")


def _measure_first_future_register(data: Mapping[str, np.ndarray]) -> dict[str, float | int]:
    prediction = np.asarray(data["predicted_render_base_pitches"], dtype=np.float64)[:, 0]
    target_indexes = np.asarray(data["target_source_stream_indices"], dtype=np.int64)[:, 0]
    committed = np.asarray(data["committed_source_stream_indices"], dtype=np.int64)
    source = np.asarray(data["source_render_base_pitches"], dtype=np.float64)
    native_anchor = np.asarray(data["context_render_base_pitches"], dtype=np.float64)
    valid = (target_indexes >= 0) & (committed > 0) & (target_indexes < len(source)) & (committed <= len(source))
    if not np.any(valid):
        return {"valid_samples": 0}
    predicted = prediction[valid]
    target = source[target_indexes[valid]]
    true_anchor = source[committed[valid] - 1]
    native = native_anchor[valid]
    absolute = predicted - target
    common_delta = (predicted - true_anchor) - (target - true_anchor)
    native_delta = (predicted - native) - (target - true_anchor)
    anchor_gap = native - true_anchor
    residual = native_delta - (absolute - anchor_gap)
    return {
        "valid_samples": int(np.sum(valid)),
        "absolute_register_rmse_semitones": _rmse(absolute),
        "common_anchor_delta_rmse_semitones": _rmse(common_delta),
        "native_anchor_delta_rmse_semitones": _rmse(native_delta),
        "history_anchor_gap_rmse_semitones": _rmse(anchor_gap),
        "history_anchor_gap_mean_semitones": float(np.mean(anchor_gap)),
        "decomposition_residual_rmse_semitones": _rmse(residual),
    }


def _markdown(report: Mapping[str, Any]) -> str:
    arms = report["metrics"]["first_future_register"]
    rows = [
        "# +1 音区参考系检查",
        "",
        "本检查只看每个预测边界后的第一小节。它把模型给出的绝对音区，与同一条真实乐曲的目标音区直接比较；随后再显示不同历史参考点如何改变 delta 的表观误差。",
        "",
        "| 指标（半音） | Free-running | Teacher-forced |",
        "| --- | ---: | ---: |",
    ]
    labels = (
        ("absolute_register_rmse_semitones", "绝对音区误差 RMSE"),
        ("common_anchor_delta_rmse_semitones", "统一真实参考点下的 delta 误差"),
        ("native_anchor_delta_rmse_semitones", "各自历史参考点下的 delta 误差"),
        ("history_anchor_gap_rmse_semitones", "历史参考点差异 RMSE"),
        ("history_anchor_gap_mean_semitones", "历史参考点差异均值"),
        ("decomposition_residual_rmse_semitones", "分解恒等式残差 RMSE"),
    )
    for key, label in labels:
        rows.append(f"| {label} | {_value(arms['free_running'].get(key))} | {_value(arms['teacher_forced'].get(key))} |")
    rows.extend([
        "",
        "统一真实参考点下的 delta 误差与绝对音区误差必须相同：两者都等于预测音区减去真实音区。若“各自历史参考点下”的误差更大，应先查看历史参考点差异，而不是把它直接解释为模型的绝对音区预测恶化。",
        "",
        "分解关系为：各自历史参考点下的 delta 误差 = 绝对音区误差 - 历史参考点差异。残差接近零表示表格中的差异可由这个参考系关系完整解释。",
        "",
    ])
    return "\n".join(rows)


def _rmse(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def _value(value: Any) -> str:
    return f"{float(value):.3f}" if value is not None else "--"


def _verify(path: Path, reference: Mapping[str, Any]) -> None:
    if not path.is_file() or _sha256(path) != reference.get("sha256"):
        raise ValueError(f"Artifact hash mismatch for {path.name}")


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
