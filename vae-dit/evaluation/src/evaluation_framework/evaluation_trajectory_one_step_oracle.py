"""One-step teacher-forced oracle assessment from semantic future artifacts."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .evaluation_api import ArtifactBundle, ArtifactEvaluator, ArtifactExporter, EvaluationModule, EvaluationResult
from .evaluation_context import EvaluationContext, ExportContext


TEST_POINT = "trajectory_one_step_oracle"
_MAJOR = np.asarray([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88], dtype=np.float32)
_MINOR = np.asarray([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17], dtype=np.float32)


class TrajectoryOneStepOracleExporter(ArtifactExporter):
    test_point = TEST_POINT
    input_contract = "evaluation_manifest.v2 + trajectory_teacher_forced_semantic_future_position.v1"
    output_contract = "trajectory_one_step_oracle_inputs.v1"

    def export(self, context: ExportContext) -> ArtifactBundle:
        manifest_path = context.input_root / "teacher_forced__evaluation_manifest.json"
        manifest = _json(manifest_path)
        metric_ref = manifest.get("artifacts", {}).get("generation_metric_inputs", {})
        metric_path = manifest_path.parent / str(metric_ref.get("path", ""))
        _verify(metric_path, metric_ref)
        metric = _json(metric_path)
        if not metric.get("diagnostic_modes", {}).get("one_step_teacher_forced_oracle"):
            raise ValueError("Generation artifact was not produced by one-step teacher-forced oracle mode.")
        path = metric_path.parent / str(metric.get("future_position_semantic_path", ""))
        digest = metric.get("future_position_semantic_sha256")
        if not digest or _sha256(path) != digest:
            raise ValueError("One-step oracle requires a valid semantic future-position artifact.")
        payload = {"schema_version": self.output_contract, "semantic_future": {"path": _relative(path, context.input_root), "sha256": digest}, "provenance": {"generation_manifest": _relative(manifest_path, context.input_root), "generation_manifest_sha256": _sha256(manifest_path)}}
        inputs = context.store.write_json(TEST_POINT, "inputs", payload)
        return ArtifactBundle(TEST_POINT, {"inputs": inputs.name})


class TrajectoryOneStepOracleEvaluator(ArtifactEvaluator):
    test_point = TEST_POINT
    required_artifacts: Sequence[str] = ("inputs",)

    def evaluate(self, context: EvaluationContext, bundle: ArtifactBundle) -> EvaluationResult:
        inputs = _json(context.store.run_dir / bundle.artifacts["inputs"])
        data = _load(inputs["semantic_future"], context.input_root)
        metrics = _measure(data)
        report = {
            "schema_version": "assessment_report.v1",
            "status": "MONITOR",
            "metrics": metrics,
            "findings": [
                {"classification": "one_step_teacher_forced_oracle", "text": "Every evaluated +1 boundary was conditioned on real source history. Severe error here is evidence of a remaining one-step conditional prediction problem."},
                {"classification": "matched_free_running_control", "text": "UNAVAILABLE: this run has no matched free-running control with the same boundary set, target definition and sampling configuration. Rollout compounding contribution cannot be quantified from this oracle alone."},
            ],
            "provenance": inputs["provenance"],
        }
        return EvaluationResult(report=report, markdown=_markdown(report))


TRAJECTORY_ONE_STEP_ORACLE_MODULE = EvaluationModule(
    test_point=TEST_POINT,
    exporter=TrajectoryOneStepOracleExporter(),
    evaluator=TrajectoryOneStepOracleEvaluator(),
    summary="Per-boundary real-history +1 trajectory oracle without shortening the sampled plan horizon.",
)


def _load(reference: Mapping[str, Any], root: Path) -> dict[str, Any]:
    path = root / str(reference["path"])
    if _sha256(path) != reference["sha256"]:
        raise ValueError(f"Artifact hash mismatch for {path.name}")
    required = ("predicted_bar_tensors", "predicted_render_base_pitches", "source_bar_tensors", "source_render_base_pitches", "target_source_stream_indices", "codec_tensor_schema_json")
    with np.load(path, allow_pickle=False) as source:
        missing = [name for name in required if name not in source]
        if missing:
            raise ValueError(f"Semantic future artifact is missing: {', '.join(missing)}")
        data = {name: np.asarray(source[name]) for name in required}
    data["codec"] = json.loads(str(data.pop("codec_tensor_schema_json").reshape(-1)[0]))
    return data


def _measure(data: Mapping[str, Any]) -> dict[str, Any]:
    indexes = data["target_source_stream_indices"][:, 0]
    valid = indexes >= 0
    if not np.any(valid):
        return {"status": "UNAVAILABLE", "reason": "no real source target exists after an evaluated boundary"}
    target = data["source_bar_tensors"][indexes[valid]]
    prediction = data["predicted_bar_tensors"][valid, 0]
    target_bases = data["source_render_base_pitches"][indexes[valid]]
    prediction_bases = data["predicted_render_base_pitches"][valid, 0]
    scale = float(data["codec"]["pitch"]["pitch_scale"])
    target_chroma, predicted_chroma = _chroma(target, target_bases, scale), _chroma(prediction, prediction_bases, scale)
    target_keys = [_key(value) for value in target_chroma]
    predicted_keys = [_key(value) for value in predicted_chroma]
    return {
        "status": "MONITOR",
        "valid_boundaries": int(np.sum(valid)),
        "absolute_chroma_cosine_mean": float(np.mean([_cosine(left, right) for left, right in zip(target_chroma, predicted_chroma)])),
        "target_key_match_ratio": float(np.mean([left == right for left, right in zip(target_keys, predicted_keys)])),
        "diatonic_fit_to_target_key_mean": float(np.mean([_diatonic(key, chroma) for key, chroma in zip(target_keys, predicted_chroma)])),
        "absolute_register_rmse_semitones": float(np.sqrt(np.mean(np.square(prediction_bases - target_bases)))),
        "matched_free_running_control": {
            "status": "UNAVAILABLE",
            "reason": "No free-running artifact with matching boundaries, targets and sampling configuration was supplied.",
        },
    }


def _chroma(tensors: np.ndarray, bases: np.ndarray, scale: float) -> np.ndarray:
    result = np.zeros((len(tensors), 12), dtype=np.float32)
    active = (tensors[..., 2] > .5) | (tensors[..., 3] > .5)
    for row, tensor, base, mask in zip(result, tensors, bases, active):
        for pitch, enabled in zip(tensor[..., 0].reshape(-1), mask.reshape(-1)):
            if enabled:
                row[int(round(float(base) + float(pitch) * scale)) % 12] += 1.0
        if row.sum(): row /= row.sum()
    return result


def _key(chroma: np.ndarray) -> tuple[int, str]:
    choices = [(root, mode, _cosine(chroma, np.roll(template, root))) for root in range(12) for mode, template in (("major", _MAJOR), ("minor", _MINOR))]
    root, mode, _ = max(choices, key=lambda item: item[2])
    return int(root), str(mode)


def _diatonic(key: tuple[int, str], chroma: np.ndarray) -> float:
    root, mode = key
    intervals = (0, 2, 4, 5, 7, 9, 11) if mode == "major" else (0, 2, 3, 5, 7, 8, 10)
    mask = np.zeros(12, dtype=np.float32); mask[(root + np.asarray(intervals)) % 12] = 1.0
    return float(np.dot(_normalize(chroma), mask))


def _normalize(values: np.ndarray) -> np.ndarray:
    total = float(np.maximum(values, 0).sum())
    return values / total if total else values


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 1e-8 else 0.0


def _markdown(report: Mapping[str, Any]) -> str:
    markdown = _legacy_markdown(report)
    metric = report["metrics"]
    if metric.get("status") == "UNAVAILABLE":
        return markdown
    return "\n".join((
        markdown.rstrip(),
        "",
        "## Interpretation Limit",
        "",
        "Matched free-running control: UNAVAILABLE.",
        "This oracle shows the remaining one-step error under real history, but cannot quantify rollout compounding or establish that it is not the main cause.",
        "",
    ))


def _legacy_markdown(report: Mapping[str, Any]) -> str:
    metric = report["metrics"]
    if metric.get("status") == "UNAVAILABLE": return "# One-step Teacher-forced Oracle\n\n没有可比较的真实下一小节。\n"
    return "\n".join(["# One-step Teacher-forced Oracle", "", "每一个被测边界之前都只使用真实 source history。模型仍采样正常长度的 trajectory plan，本报告只读取其中的第一个未来小节。", "", "| 指标 | 结果 |", "| --- | ---: |", f"| 可比较边界数 | {metric['valid_boundaries']} |", f"| 与真实目标绝对和声轮廓相似度 | {metric['absolute_chroma_cosine_mean']:.3f} |", f"| 与真实目标调性匹配率 | {metric['target_key_match_ratio']:.3f} |", f"| 对真实目标调内音比例 | {metric['diatonic_fit_to_target_key_mean']:.3f} |", f"| 绝对音区误差 RMSE（半音） | {metric['absolute_register_rmse_semitones']:.3f} |", "", "真实 history 条件下仍存在一步预测误差。由于没有同边界、同目标和同采样设置的 free-running 对照，rollout contribution 尚不可量化。", ""])


def _verify(path: Path, reference: Mapping[str, Any]) -> None:
    if not path.is_file() or _sha256(path) != reference.get("sha256"): raise ValueError(f"Artifact hash mismatch for {path.name}")


def _relative(path: Path, root: Path) -> str: return path.resolve().relative_to(root.resolve()).as_posix()
def _json(path: Path) -> dict[str, Any]: return json.loads(path.read_text(encoding="utf-8"))
def _sha256(path: Path) -> str: return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
