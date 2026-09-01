"""Paired teacher-forced trajectory assessment from public generation artifacts."""

from __future__ import annotations

import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .evaluation_api import ArtifactBundle, ArtifactEvaluator, ArtifactExporter, EvaluationModule, EvaluationResult
from .evaluation_context import EvaluationContext, ExportContext


TEST_POINT = "trajectory_teacher_forced"
PAIR_MANIFEST_NAME = "paired_experiment_manifest.json"
MAJOR_PROFILE = np.asarray([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88], dtype=np.float32)
MINOR_PROFILE = np.asarray([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17], dtype=np.float32)
KEY_NAMES = ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")


class TrajectoryTeacherForcedExporter(ArtifactExporter):
    """Normalize an existing paired generation manifest for this test point.

    The exporter intentionally consumes only already-written public artifacts. It
    never imports a pipeline or checkpoint and never requests a model rerun.
    """

    test_point = TEST_POINT
    input_contract = "paired_experiment_manifest.v1|paired_experiment_manifest.v2"
    output_contract = "trajectory_teacher_forced_inputs.v1"

    def export(self, context: ExportContext) -> ArtifactBundle:
        pair_path = context.input_root / PAIR_MANIFEST_NAME
        if not pair_path.is_file():
            raise FileNotFoundError(
                f"Missing {PAIR_MANIFEST_NAME} in {context.input_root}. "
                "Export the paired public generation manifests first."
            )
        pair = _read_json(pair_path)
        _require(pair, "schema_version", "pairing_group_id", "shared_identity", "arms")
        if pair["schema_version"] not in {"paired_experiment_manifest.v1", "paired_experiment_manifest.v2"}:
            raise ValueError(f"Unsupported paired manifest schema: {pair['schema_version']}")

        arms = _load_arms(pair_path, pair)
        availability = {
            "paired_generation": "available",
            "future_position_trajectory": "available" if all(arm["future_position"] is not None for arm in arms.values()) else "not_provided",
        }
        inputs = {
            "schema_version": self.output_contract,
            "pairing_group_id": pair["pairing_group_id"],
            "shared_identity": pair["shared_identity"],
            "history_source": {name: arm["history_source"] for name, arm in arms.items()},
            "arms": {name: arm["public"] for name, arm in arms.items()},
            "future_positions": _future_positions(arms),
            "availability": availability,
            "provenance": {
                "pair_manifest": _reference(pair_path, context.input_root),
                "pair_manifest_sha256": _sha256(pair_path),
            },
        }
        path = context.store.write_json(TEST_POINT, "inputs", inputs)
        return ArtifactBundle(TEST_POINT, {"inputs": path.name}, {"pairing_group_id": pair["pairing_group_id"]})


class TrajectoryTeacherForcedEvaluator(ArtifactEvaluator):
    """Measure paired rollout differences without turning them into a repair rule."""

    test_point = TEST_POINT
    required_artifacts: Sequence[str] = ("inputs",)

    def evaluate(self, context: EvaluationContext, bundle: ArtifactBundle) -> EvaluationResult:
        inputs_path = context.store.run_dir / bundle.artifacts["inputs"]
        inputs = _read_json(inputs_path)
        _require(inputs, "schema_version", "pairing_group_id", "shared_identity", "arms", "availability", "provenance")
        if inputs["schema_version"] != "trajectory_teacher_forced_inputs.v1":
            raise ValueError(f"Unsupported module input schema: {inputs['schema_version']}")
        arms = inputs["arms"]
        free = _measure_arm(arms["free_running"], context.input_root)
        teacher = _measure_arm(arms["teacher_forced"], context.input_root)
        gaps = _metric_gaps(free, teacher)
        classification = _classification(gaps)
        future_metric, missing_inputs = _future_position_metric(arms, context.input_root)
        semantic_metric = _semantic_future_metric(arms, context.input_root)
        if future_metric is None:
            missing_inputs.append({
                "artifact": "trajectory_teacher_forced_inputs",
                "field": "arms.*.future_position",
                "reason": "engineering export must add future_position_path and provide aligned target and prediction arrays for positions 1, 2, 3 and 4",
            })
        report = {
            "schema_version": "assessment_report.v1",
            "status": "WARN" if classification == "conditioning_boundary_unresolved" else "MONITOR",
            "metrics": {
                "free_running": free,
                "teacher_forced": teacher,
                "teacher_minus_free": gaps,
                "future_position_trajectory": future_metric or {"status": "UNAVAILABLE"},
                "future_position_music_features": semantic_metric,
            },
            "findings": [{
                "classification": classification,
                "text": _classification_text(classification),
            }],
            "provenance": {
                "pairing_group_id": inputs["pairing_group_id"],
                "shared_identity": inputs["shared_identity"],
                **inputs["provenance"],
            },
            "missing_inputs": missing_inputs,
            "causal_language_policy": "Teacher-forced and free-running gaps are reported as paired evidence, not as a percentage attribution or a generation constraint.",
        }
        return EvaluationResult(report=report, markdown=_markdown(report), figures={"comparison": _plot_png(free, teacher)})


TRAJECTORY_TEACHER_FORCED_MODULE = EvaluationModule(
    test_point=TEST_POINT,
    exporter=TrajectoryTeacherForcedExporter(),
    evaluator=TrajectoryTeacherForcedEvaluator(),
    summary="Paired free-running and teacher-forced trajectory boundary assessment.",
)


def _load_arms(pair_path: Path, pair: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for arm_ref in pair["arms"]:
        arm_name = str(arm_ref.get("arm", ""))
        if arm_name not in {"free_running", "teacher_forced"}:
            continue
        manifest_path = (pair_path.parent / str(arm_ref.get("artifact_manifest", ""))).resolve()
        manifest = _read_json(manifest_path)
        _require(manifest, "schema_version", "dataset", "run", "checkpoint_refs", "artifacts")
        if manifest["schema_version"] not in {"evaluation_manifest.v1", "evaluation_manifest.v2"}:
            raise ValueError(f"Unsupported generation manifest schema: {manifest['schema_version']}")
        _validate_identity(manifest, pair["shared_identity"])
        refs = manifest["artifacts"]
        metric_ref = refs.get("generation_metric_inputs")
        tensor_ref = refs.get("bar_tensors")
        if not metric_ref or not tensor_ref:
            raise ValueError("Generation manifest must reference generation_metric_inputs and bar_tensors.")
        metric_path = (manifest_path.parent / str(metric_ref["path"])).resolve()
        tensor_path = (manifest_path.parent / str(tensor_ref["path"])).resolve()
        _verify_reference(metric_path, metric_ref)
        _verify_reference(tensor_path, tensor_ref)
        metric_inputs = _read_json(metric_path)
        expected_history = "generated" if arm_name == "free_running" else "real_dataset"
        if metric_inputs.get("history_source") != expected_history:
            raise ValueError(f"{arm_name} artifact must use history_source={expected_history}.")
        future_path = metric_inputs.get("future_position_path")
        future_reference = None
        if future_path:
            resolved_future_path = (metric_path.parent / str(future_path)).resolve()
            expected_future_hash = metric_inputs.get("future_position_sha256")
            if not expected_future_hash or _sha256(resolved_future_path) != expected_future_hash:
                raise ValueError(f"Artifact hash mismatch for {resolved_future_path.name}")
            future_reference = {"path": _reference(resolved_future_path, pair_path.parent), "sha256": expected_future_hash}
        semantic_reference = _optional_reference(metric_inputs, "future_position_semantic", metric_path, pair_path.parent)
        result[arm_name] = {
            "history_source": expected_history,
            "future_position": future_reference,
            "future_position_semantic": semantic_reference,
            "public": {
                "generation_manifest": _reference(manifest_path, pair_path.parent),
                "generation_metric_inputs": _reference(metric_path, pair_path.parent),
                "bar_tensors": _reference(tensor_path, pair_path.parent),
                "primer_bars": int(metric_inputs.get("primer_bars", 0) or 0),
                "future_position": future_reference,
                "future_position_semantic": semantic_reference,
            },
        }
    if set(result) != {"free_running", "teacher_forced"}:
        raise ValueError("Pair manifest must contain exactly free_running and teacher_forced arms.")
    return result


def _measure_arm(arm: Mapping[str, Any], input_root: Path) -> dict[str, Any]:
    tensor_path = (input_root / arm["bar_tensors"]).resolve()
    arrays = np.load(tensor_path, allow_pickle=False)
    if "bars" not in arrays or "render_base_pitches" not in arrays:
        raise ValueError("Bar tensor artifact must contain bars and render_base_pitches.")
    bars = np.asarray(arrays["bars"], dtype=np.float32)
    bases = np.asarray(arrays["render_base_pitches"], dtype=np.float32).reshape(-1)
    if bars.ndim != 4 or len(bars) != len(bases):
        raise ValueError("bars must be [bar, track, step, feature] and align with render_base_pitches.")
    primer = int(arm["primer_bars"])
    chroma = _absolute_chroma_by_bar(bars, bases)
    primer_profile = _normalize(np.sum(chroma[:primer], axis=0)) if primer else np.zeros(12, dtype=np.float32)
    key = _estimate_key(primer_profile)
    summary = _sequence_summary(chroma[primer:], primer_profile, key)
    register_deltas = np.diff(bases)
    return {
        "primer_key": key["name"],
        "generated_bars": int(max(0, len(chroma) - primer)),
        **summary,
        "register_delta_abs_mean": float(np.mean(np.abs(register_deltas))) if len(register_deltas) else 0.0,
        "register_drift": float(np.max(bases) - np.min(bases)) if len(bases) else 0.0,
        "plan_overlap_disagreement": "UNAVAILABLE",
    }


def _future_position_metric(arms: Mapping[str, Any], input_root: Path) -> tuple[dict[str, Any] | None, list[dict[str, str]]]:
    paths = [arm.get("future_position") for arm in arms.values()]
    if not all(paths):
        return None, []
    errors: dict[str, Any] = {}
    shared_target = None
    for name, arm in arms.items():
        target, prediction, positions, valid_mask = _load_future_position(arm["future_position"], input_root)
        if shared_target is None:
            shared_target = (target, valid_mask, positions)
        elif not (np.array_equal(target, shared_target[0]) and np.array_equal(valid_mask, shared_target[1]) and positions == shared_target[2]):
            raise ValueError("Paired future-position target observations must be identical.")
        mse_by_position = []
        variance_by_position = []
        nmse_by_position = []
        groups = {"latent_mu": [], "register_delta": []}
        valid_samples_by_position = []
        for index in range(target.shape[1]):
            valid = valid_mask[:, index]
            valid_samples_by_position.append(int(np.sum(valid)))
            if not np.any(valid):
                mse_by_position.append(None); variance_by_position.append(None); nmse_by_position.append(None)
                for values in groups.values(): values.append(None)
                continue
            reference, estimate = target[valid, index], prediction[valid, index]
            mse = float(np.mean((reference - estimate) ** 2)); variance = float(np.mean((reference - np.mean(reference)) ** 2))
            mse_by_position.append(mse); variance_by_position.append(variance); nmse_by_position.append(mse / variance if variance > 1e-12 else None)
            groups["latent_mu"].append(_mse(reference[:, :-1], estimate[:, :-1]))
            groups["register_delta"].append(_mse(reference[:, -1:], estimate[:, -1:]))
        errors[name] = {
            "positions": positions,
            "mse_by_position": mse_by_position,
            "target_variance_by_position": variance_by_position,
            "nmse_by_position": nmse_by_position,
            "feature_groups_mse": groups,
            "valid_samples_by_position": valid_samples_by_position,
        }
    return {"status": "MONITOR", "arms": errors}, []


def _semantic_future_metric(arms: Mapping[str, Any], input_root: Path) -> dict[str, Any]:
    if not all(arm.get("future_position_semantic") for arm in arms.values()):
        return {"status": "UNAVAILABLE", "reason": "semantic future-position observation was not exported for both arms"}
    loaded = {name: _load_semantic_future(arm["future_position_semantic"], input_root) for name, arm in arms.items()}
    free, teacher = loaded["free_running"], loaded["teacher_forced"]
    for key in ("source_bar_tensors", "source_render_base_pitches", "source_bar_indices", "target_source_stream_indices"):
        if not np.array_equal(free[key], teacher[key]):
            raise ValueError(f"Paired semantic future observations must share {key}.")
    groups = {name: [] for name in ("absolute_chroma", "relative_chroma", "relative_pitch", "rhythm", "velocity", "density")}
    result = {"status": "MONITOR", "arms": {}}
    for arm_name, data in loaded.items():
        values = {name: [] for name in groups}
        for position in range(data["predicted_bar_tensors"].shape[1]):
            target, prediction, target_bases, prediction_bases = _semantic_target_prediction(data, position)
            if not len(target):
                for name in values:
                    values[name].append(None)
                continue
            pitch_scale = float(data["codec"]["pitch"]["pitch_scale"])
            for name, target_value, prediction_value in _semantic_group_values(target, prediction, target_bases, prediction_bases, pitch_scale):
                values[name].append(_mse(target_value, prediction_value))
        result["arms"][arm_name] = values
    return result


def _optional_reference(inputs: Mapping[str, Any], prefix: str, metric_path: Path, root: Path) -> dict[str, str] | None:
    path, digest = inputs.get(f"{prefix}_path"), inputs.get(f"{prefix}_sha256")
    if not path:
        return None
    resolved = (metric_path.parent / str(path)).resolve()
    if not digest or _sha256(resolved) != digest:
        raise ValueError(f"Artifact hash mismatch for {resolved.name}")
    return {"path": _reference(resolved, root), "sha256": str(digest)}


def _load_semantic_future(reference: Mapping[str, Any], root: Path) -> dict[str, np.ndarray]:
    path = (root / str(reference["path"])).resolve()
    if _sha256(path) != reference["sha256"]: raise ValueError(f"Artifact hash mismatch for {path.name}")
    with np.load(path, allow_pickle=False) as source:
        names = ("predicted_bar_tensors", "predicted_render_base_pitches", "source_bar_tensors", "source_render_base_pitches", "source_bar_indices", "target_source_stream_indices", "codec_tensor_schema_json")
        data = {name: np.asarray(source[name]) for name in names}
    data["codec"] = json.loads(str(data.pop("codec_tensor_schema_json").reshape(-1)[0]))
    return data


def _load_future_position(reference: Mapping[str, Any], input_root: Path) -> tuple[np.ndarray, np.ndarray, list[int], np.ndarray]:
    path = (input_root / str(reference["path"])).resolve()
    if _sha256(path) != reference.get("sha256"):
        raise ValueError(f"Artifact hash mismatch for {path.name}")
    if path.suffix.lower() == ".json":
        payload = _read_json(path)
        target = np.asarray(payload.get("target"), dtype=np.float32)
        prediction = np.asarray(payload.get("prediction"), dtype=np.float32)
        positions = payload.get("future_positions")
        mask_source = payload.get("valid_mask")
    elif path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            required = ("target", "prediction", "future_positions")
            missing = [name for name in required if name not in archive]
            if missing:
                raise ValueError(f"Future-position NPZ is missing: {', '.join(missing)}")
            target = np.asarray(archive["target"], dtype=np.float32)
            prediction = np.asarray(archive["prediction"], dtype=np.float32)
            positions = np.asarray(archive["future_positions"]).tolist()
            mask_source = np.asarray(archive["valid_mask"]) if "valid_mask" in archive else None
    else:
        raise ValueError("Future-position artifact must be JSON or NPZ.")
    if target.ndim != 3 or prediction.shape != target.shape or not np.isfinite(target).all() or not np.isfinite(prediction).all():
        raise ValueError("Future-position target and prediction must be finite aligned [sample, position, feature] arrays.")
    if not isinstance(positions, list) or len(positions) != target.shape[1] or not all(isinstance(value, (int, np.integer)) for value in positions):
        raise ValueError("future_positions must be integer-valued and match the position axis.")
    if mask_source is None:
        valid_mask = np.ones(target.shape[:2], dtype=bool)
    else:
        raw_mask = np.asarray(mask_source)
        binary_integer = np.issubdtype(raw_mask.dtype, np.integer) and np.isin(raw_mask, (0, 1)).all()
        if raw_mask.shape != target.shape[:2] or not (raw_mask.dtype == np.bool_ or binary_integer):
            raise ValueError("valid_mask must be bool or binary integer values aligned with [sample, position].")
        valid_mask = raw_mask.astype(bool, copy=False)
    return target, prediction, [int(value) for value in positions], valid_mask



from .evaluation_trajectory_teacher_forced_metrics import (
    _absolute_chroma_by_bar,
    _classification,
    _classification_text,
    _entropy,
    _estimate_key,
    _future_positions,
    _metric_gaps,
    _mse,
    _normalize,
    _semantic_group_values,
    _semantic_target_prediction,
    _sequence_summary,
)

from .evaluation_trajectory_teacher_forced_presentation import _markdown, _plot_png

def _validate_identity(manifest: Mapping[str, Any], shared: Mapping[str, Any]) -> None:
    dataset, run = manifest["dataset"], manifest["run"]
    observed = {
        "dataset_identity": dataset.get("identity", dataset.get("dataset_hash")),
        "dataset_identity_kind": dataset.get("identity_kind", "legacy_hash_field"),
        "split": dataset.get("split"), "song_id": dataset.get("song_id"),
        "seed": run.get("seed"), "primer_bars": run.get("primer_bars"), "total_bars": run.get("total_bars"),
        "initialization": run.get("initialization"), "sampling": run.get("sampling"), "checkpoint_refs": manifest.get("checkpoint_refs", []),
    }
    expected = dict(shared)
    if "dataset_hash" in expected:
        expected["dataset_identity"] = expected.pop("dataset_hash")
        expected.setdefault("dataset_identity_kind", "legacy_hash_field")
    mismatches = [key for key, value in expected.items() if observed.get(key) != value]
    if mismatches:
        raise ValueError(f"Paired generation artifacts do not share identity: {', '.join(mismatches)}")


def _verify_reference(path: Path, reference: Mapping[str, Any]) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing referenced artifact: {path}")
    expected = reference.get("sha256")
    if expected and expected != _sha256(path):
        raise ValueError(f"Artifact hash mismatch for {path.name}")


def _reference(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError("All paired artifacts must be located below --input-root.") from error


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _require(payload: Mapping[str, Any], *fields: str) -> None:
    missing = [field for field in fields if field not in payload]
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return f"sha256:{digest}"
