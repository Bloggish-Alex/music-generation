"""Controlled history-swap sensitivity assessment from public artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .evaluation_api import ArtifactBundle, ArtifactEvaluator, ArtifactExporter, EvaluationModule, EvaluationResult
from .evaluation_context import EvaluationContext, ExportContext


TEST_POINT = "trajectory_history_swap"


class TrajectoryHistorySwapExporter(ArtifactExporter):
    test_point = TEST_POINT
    input_contract = "evaluation_manifest.v2 + trajectory_history_swap.v1"
    output_contract = "trajectory_history_swap_inputs.v1"

    def export(self, context: ExportContext) -> ArtifactBundle:
        manifest_path = context.input_root / "teacher_forced__evaluation_manifest.json"
        manifest = _json(manifest_path)
        metric_ref = manifest.get("artifacts", {}).get("generation_metric_inputs", {})
        metric_path = manifest_path.parent / str(metric_ref.get("path", ""))
        _verify(metric_path, metric_ref)
        metric = _json(metric_path)
        history_path = metric_path.parent / str(metric.get("history_swap_path", ""))
        digest = metric.get("history_swap_sha256")
        if not digest or _sha256(history_path) != digest:
            raise ValueError("Controlled history-swap public artifact is unavailable or has an invalid hash.")
        payload = {
            "schema_version": self.output_contract,
            "history_swap": {"path": _relative(history_path, context.input_root), "sha256": digest},
            "provenance": {"generation_manifest": _relative(manifest_path, context.input_root), "generation_manifest_sha256": _sha256(manifest_path)},
        }
        path = context.store.write_json(TEST_POINT, "inputs", payload)
        return ArtifactBundle(TEST_POINT, {"inputs": path.name})


class TrajectoryHistorySwapEvaluator(ArtifactEvaluator):
    test_point = TEST_POINT
    required_artifacts: Sequence[str] = ("inputs",)

    def evaluate(self, context: EvaluationContext, bundle: ArtifactBundle) -> EvaluationResult:
        inputs = _json(context.store.run_dir / bundle.artifacts["inputs"])
        archive = _load(inputs["history_swap"], context.input_root)
        report = {
            "schema_version": "assessment_report.v1",
            "status": "MONITOR",
            "metrics": _measure(archive),
            "findings": [{
                "classification": "controlled_history_sensitivity",
                "text": "This is one fixed boundary (n=1) with no uncertainty estimate. Differences describe local conditioning sensitivity only and must not be generalized to the model without multiple boundaries and seeds.",
            }],
            "provenance": inputs["provenance"],
        }
        return EvaluationResult(report=report, markdown=_markdown(report))


TRAJECTORY_HISTORY_SWAP_MODULE = EvaluationModule(
    test_point=TEST_POINT,
    exporter=TrajectoryHistorySwapExporter(),
    evaluator=TrajectoryHistorySwapEvaluator(),
    summary="Fixed-boundary sensitivity to real, transposed and alternate-song history.",
)


def _load(reference: Mapping[str, Any], root: Path) -> dict[str, np.ndarray | dict[str, Any]]:
    path = root / str(reference["path"])
    if _sha256(path) != reference["sha256"]:
        raise ValueError(f"Artifact hash mismatch for {path.name}")
    required = ("schema_version", "variant_names", "prediction_states", "predicted_bar_tensors", "predicted_render_base_pitches", "target_bar_tensor", "target_render_base_pitch", "target_source_stream_index", "history_song_ids", "history_bar_tensors", "codec_tensor_schema_json")
    with np.load(path, allow_pickle=False) as source:
        missing = [name for name in required if name not in source]
        if missing:
            raise ValueError(f"History-swap artifact is missing: {', '.join(missing)}")
        result: dict[str, np.ndarray | dict[str, Any]] = {name: np.asarray(source[name]) for name in required}
        for name in ("boundary_source_song_id", "primer_bars", "sampling_seed", "sampling_seed_algorithm", "sampling_seed_offset", "theme_memory_condition"):
            if name in source:
                result[name] = np.asarray(source[name])
    if str(np.asarray(result["schema_version"]).reshape(-1)[0]) != "trajectory_history_swap.v1":
        raise ValueError("Unsupported history-swap public schema.")
    if tuple(np.asarray(result["variant_names"]).tolist()) != ("H1_real", "H2_transposed_plus_6", "H3_alternate_song"):
        raise ValueError("Unexpected history-swap variant order.")
    result["codec"] = json.loads(str(np.asarray(result.pop("codec_tensor_schema_json")).reshape(-1)[0]))
    return result


def _measure(data: Mapping[str, Any]) -> dict[str, Any]:
    states = np.asarray(data["prediction_states"], dtype=np.float32)[:, 0]
    tensors = np.asarray(data["predicted_bar_tensors"], dtype=np.float32)[:, 0]
    bases = np.asarray(data["predicted_render_base_pitches"], dtype=np.float32)[:, 0]
    target_tensor = np.asarray(data["target_bar_tensor"], dtype=np.float32)[None]
    target_base = np.asarray([float(np.asarray(data["target_render_base_pitch"]).reshape(-1)[0])], dtype=np.float32)
    pitch_scale = float(data["codec"]["pitch"]["pitch_scale"])
    names = [str(item) for item in np.asarray(data["variant_names"]).tolist()]
    target_chroma = _chroma(target_tensor, target_base, pitch_scale)[0]
    values: dict[str, Any] = {}
    reference = states[0]
    for index, name in enumerate(names):
        chroma = _chroma(tensors[index:index + 1], bases[index:index + 1], pitch_scale)[0]
        values[name] = {
            "latent_l2_from_H1": float(np.linalg.norm(states[index, :-1] - reference[:-1])),
            "register_change_from_H1_semitones": float(bases[index] - bases[0]),
            "absolute_chroma_cosine_to_H1": _cosine(chroma, _chroma(tensors[:1], bases[:1], pitch_scale)[0]),
            "absolute_chroma_cosine_to_target": _cosine(chroma, target_chroma),
            "absolute_register_error_to_target_semitones": float(bases[index] - target_base[0]),
        }
    return {
        "fixed_future_position": 1,
        "experiment_context": {
            "sample_count": 1,
            "boundary_source_song_id": _optional_scalar(data, "boundary_source_song_id", _history_song_id(data)),
            "primer_bars": int(_optional_scalar(data, "primer_bars", _primer_bars(data))),
            "target_source_stream_index": _target_stream_index(data),
            "sampling_seed": _optional_scalar(data, "sampling_seed", None),
            "sampling_seed_algorithm": _optional_scalar(data, "sampling_seed_algorithm", "not_recorded"),
            "sampling_seed_offset": _optional_scalar(data, "sampling_seed_offset", "not_recorded"),
            "theme_memory_condition": _optional_scalar(data, "theme_memory_condition", "not_recorded"),
            "uncertainty": "UNAVAILABLE: one fixed boundary",
        },
        "variants": values,
    }


def _chroma(tensors: np.ndarray, bases: np.ndarray, pitch_scale: float) -> np.ndarray:
    result = np.zeros((len(tensors), 12), dtype=np.float32)
    active = (tensors[..., 2] > .5) | (tensors[..., 3] > .5)
    for row, tensor, base, mask in zip(result, tensors, bases, active):
        for pitch, enabled in zip(tensor[..., 0].reshape(-1), mask.reshape(-1)):
            if enabled:
                row[int(round(float(base) + float(pitch) * pitch_scale)) % 12] += 1.0
        total = row.sum()
        if total:
            row /= total
    return result


def _markdown(report: Mapping[str, Any]) -> str:
    variants = report["metrics"]["variants"]
    context = report["metrics"]["experiment_context"]
    rows = [
        "# Controlled History Swap",
        "",
        f"Fixed boundary experiment: n={context['sample_count']}; source song `{context['boundary_source_song_id']}`; primer bars {context['primer_bars']}; target source bar {context['target_source_stream_index']}; effective seed {context['sampling_seed']} ({context['sampling_seed_algorithm']}, offset {context['sampling_seed_offset']}); theme condition `{context['theme_memory_condition']}`.",
        "This report has no uncertainty estimate and does not support a model-wide conditioning conclusion.",
        "",
        "三种 history 在同一个预测边界、同一采样随机种子下比较。H1 是真实 history；H2 是同一 history 整体上移六个半音；H3 是另一首乐曲的等长真实 history。",
        "",
        "| History | Latent 与 H1 的距离 | 音区相对 H1（半音） | 绝对和声轮廓与 H1 相似度 | 与真实目标和声轮廓相似度 | 音区相对真实目标（半音） |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    labels = {"H1_real": "H1: 真实 history", "H2_transposed_plus_6": "H2: 同曲 +6", "H3_alternate_song": "H3: 异曲 history"}
    for name, value in variants.items():
        rows.append(f"| {labels[name]} | {value['latent_l2_from_H1']:.3f} | {value['register_change_from_H1_semitones']:+.3f} | {value['absolute_chroma_cosine_to_H1']:.3f} | {value['absolute_chroma_cosine_to_target']:.3f} | {value['absolute_register_error_to_target_semitones']:+.3f} |")
    rows.extend(["", "H2 与 H1 的差异显示模型是否会响应整体绝对调性/音区变化；H3 与 H1 的差异显示模型是否会响应不同的真实 musical context。需要同时阅读 latent、绝对和声轮廓和音区三列，不能把其中一列当作完整的条件敏感性结论。", ""])
    return "\n".join(rows)


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 1e-8 else 0.0


def _optional_scalar(data: Mapping[str, Any], name: str, default: Any) -> Any:
    if name not in data:
        return default
    values = np.asarray(data[name]).reshape(-1)
    return values[0].item() if values.size == 1 else default


def _history_song_id(data: Mapping[str, Any]) -> str:
    values = np.asarray(data.get("history_song_ids", ())).reshape(-1)
    return str(values[0]) if values.size else "not_recorded"


def _primer_bars(data: Mapping[str, Any]) -> int:
    history = np.asarray(data.get("history_bar_tensors", ()))
    return int(history.shape[1]) if history.ndim >= 2 else 0


def _target_stream_index(data: Mapping[str, Any]) -> int | str:
    values = np.asarray(data.get("target_source_stream_index", ())).reshape(-1)
    return int(values[0]) if values.size else "not_recorded"


def _verify(path: Path, reference: Mapping[str, Any]) -> None:
    if not path.is_file() or _sha256(path) != reference.get("sha256"):
        raise ValueError(f"Artifact hash mismatch for {path.name}")


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
