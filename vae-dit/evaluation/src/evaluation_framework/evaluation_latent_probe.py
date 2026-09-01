"""Artifact-only readability probes for frozen DVAE latent means."""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .evaluation_api import ArtifactBundle, ArtifactEvaluator, ArtifactExporter, EvaluationModule, EvaluationResult
from .evaluation_context import EvaluationContext, ExportContext
from .core.artifacts import VerifiedArtifactResolver


TEST_POINT = "latent_probe"
STATUS_SCHEMA = "dvae_fidelity_raw_status.v1"
OBSERVATION_SCHEMA = "dvae_fidelity_raw_observation.v1"
INPUT_SCHEMA = "latent_probe_inputs.v1"
_SPLITS = ("train", "validation")


class LatentProbeExporter(ArtifactExporter):
    """Index frozen DVAE observations; labels are deliberately evaluation-owned."""

    test_point = TEST_POINT
    input_contract = STATUS_SCHEMA
    output_contract = INPUT_SCHEMA

    def export(self, context: ExportContext) -> ArtifactBundle:
        resolver = VerifiedArtifactResolver(context.input_root)
        sources: dict[str, Mapping[str, Any] | None] = {}
        availability = {"latent_mu": True, "source_tensor": True, "row_alignment": True, "tensor_schema": True}
        for split in _SPLITS:
            status_path = context.input_root / f"dvae_fidelity__raw_status__{split}.v1.json"
            if not status_path.is_file():
                sources[split] = None
                availability[split] = "not_provided"
                continue
            status = _read_json(status_path)
            if status.get("schema_version") != STATUS_SCHEMA or status.get("dataset", {}).get("split") != split:
                raise ValueError(f"Invalid DVAE fidelity status for latent probe: {split}.")
            if status.get("status") != "AVAILABLE":
                sources[split] = None
                availability[split] = "unavailable"
                continue
            reference = status.get("artifacts", {}).get("observation")
            observation_path = resolver.path(reference)
            observation = resolver.json(reference)
            if observation.get("schema_version") != OBSERVATION_SCHEMA:
                raise ValueError(f"Unsupported DVAE observation schema for latent probe: {split}.")
            sources[split] = {
                "path": observation_path.name,
                "sha256": VerifiedArtifactResolver.sha256(observation_path),
                "schema_version": OBSERVATION_SCHEMA,
                "representation": "latent_mu",
                "dataset": {key: observation["dataset"][key] for key in ("identity", "identity_kind", "split")},
            }
            availability[split] = "available"
        payload = {
            "schema_version": INPUT_SCHEMA,
            "representation_sources": sources,
            "split_policy": {"split_unit": "base_song_id", "train_validation_overlap": False, "fitting_split": "train", "reporting_split": "validation"},
            "target_definitions": {
                "relative_chroma": "duration_or_active_slot_weighted_pitch_class_from_source_tensor_relative_pitch",
                "absolute_chroma": "relative_pitch_plus_base_pitch_pitch_class_when_base_pitch_available",
                "relative_register_center": "median_relative_pitch_semitones_over_active_slots",
                "absolute_register_center": "relative_register_center_plus_base_pitch_when_base_pitch_available",
                "state_ratio": "per_bar_rest_onset_hold_slot_ratio",
                "density_per_track": "per_track_active_slot_ratio",
            },
            "availability": availability,
        }
        path = context.store.write_json(TEST_POINT, "inputs", payload)
        return ArtifactBundle(TEST_POINT, {"inputs": path.name})


class LatentProbeEvaluator(ArtifactEvaluator):
    """Fit simple probes on train only and report held-out music facts."""

    test_point = TEST_POINT
    required_artifacts: Sequence[str] = ("inputs",)

    def evaluate(self, context: EvaluationContext, bundle: ArtifactBundle) -> EvaluationResult:
        inputs = _read_json(context.store.run_dir / bundle.artifacts["inputs"])
        if inputs.get("schema_version") != INPUT_SCHEMA:
            raise ValueError("Unsupported latent probe input schema.")
        source_refs = inputs["representation_sources"]
        if not all(source_refs.get(split) for split in _SPLITS):
            return EvaluationResult(_unavailable(inputs, "Both train and validation DVAE bundles are required."), _markdown_unavailable())

        resolver = VerifiedArtifactResolver(context.input_root)
        train = _load_profile(resolver, source_refs["train"])
        validation = _load_profile(resolver, source_refs["validation"])
        _assert_split_isolation(train.alignment, validation.alignment)

        results = {}
        for target in _derive_targets(train):
            validation_target = _Target(
                target.name,
                validation.targets.targets[target.name],
                validation.targets.masks[target.name],
                target.family,
            )
            results[target.name] = _evaluate_target(target, validation_target, train.latent, validation.latent)
        report = {
            "schema_version": "assessment_report.v1",
            "status": "MONITOR",
            "metrics": {"validation": results, "fit_split": "train", "reporting_split": "validation"},
            "findings": [{"classification": "latent_readability", "text": "每个目标分别报告线性与小型非线性探针；结果用于判断潜变量中信息的可读性，不合成为总分或模型修改规则。"}],
            "provenance": {"representation_sources": source_refs, "split_policy": inputs["split_policy"]},
            "missing_inputs": _target_missing(results),
        }
        return EvaluationResult(report, _markdown(report), {"readability_summary": _summary_png(results)})


@dataclass(frozen=True)
class _Targets:
    targets: Mapping[str, np.ndarray]
    masks: Mapping[str, np.ndarray]


@dataclass(frozen=True)
class _Profile:
    latent: np.ndarray
    tensor: np.ndarray
    schema: Mapping[str, Any]
    alignment: Sequence[Mapping[str, Any]]
    targets: _Targets


@dataclass(frozen=True)
class _Target:
    name: str
    values: np.ndarray
    mask: np.ndarray
    family: str


def _load_profile(resolver: VerifiedArtifactResolver, reference: Mapping[str, Any]) -> _Profile:
    observation = resolver.json(reference)
    with resolver.npz(observation["arrays"]) as archive:
        latent = np.asarray(archive["latent_mu"], dtype=np.float64)
        tensor = np.asarray(archive["source_tensor"], dtype=np.float64)
    if latent.ndim != 2 or tensor.ndim != 4 or len(latent) != len(tensor) or len(observation["alignment"]) != len(latent):
        raise ValueError("Latent probe row alignment does not match DVAE arrays.")
    if not np.isfinite(latent).all() or not np.isfinite(tensor).all():
        raise ValueError("Latent probe accepts finite DVAE arrays only.")
    targets = _targets(tensor, observation["tensor_schema"], observation["alignment"])
    return _Profile(latent, tensor, observation["tensor_schema"], observation["alignment"], targets)


def _derive_targets(profile: _Profile) -> list[_Target]:
    families = {
        "relative_chroma": "chroma", "absolute_chroma": "chroma",
        "relative_register_center": "register", "absolute_register_center": "register",
        "state_ratio": "rhythm", "density_per_track": "density",
    }
    return [_Target(name, values, profile.targets.masks[name], families[name]) for name, values in profile.targets.targets.items()]


def _targets(tensor: np.ndarray, schema: Mapping[str, Any], alignment: Sequence[Mapping[str, Any]]) -> _Targets:
    feature_names = list(schema["feature_names"])
    feature = {name: index for index, name in enumerate(feature_names)}
    state = np.argmax(tensor[..., [feature["is_rest"], feature["is_note_on"], feature["is_hold"]]], axis=-1)
    active = state != 0
    pitch = tensor[..., feature["relative_pitch"]] * float(schema["pitch_scale_semitones"])
    count = tensor.shape[0]
    relative_chroma = _chroma(pitch, active, np.zeros(count))
    anchors = np.asarray([item.get("base_pitch_semitones") if item.get("base_pitch_semitones") is not None else np.nan for item in alignment], dtype=float)
    anchored = np.isfinite(anchors) & np.any(active, axis=(1, 2))
    absolute_chroma = _chroma(pitch, active, np.nan_to_num(anchors))
    relative_register = _median_by_row(pitch, active)
    absolute_register = relative_register + anchors
    state_ratio = np.stack([np.mean(state == label, axis=(1, 2)) for label in range(3)], axis=1)
    density = np.mean(active, axis=2)
    nonempty = np.any(active, axis=(1, 2))
    values = {"relative_chroma": relative_chroma, "absolute_chroma": absolute_chroma, "relative_register_center": relative_register[:, None], "absolute_register_center": absolute_register[:, None], "state_ratio": state_ratio, "density_per_track": density}
    return _Targets(values, {"relative_chroma": nonempty, "absolute_chroma": anchored, "relative_register_center": nonempty, "absolute_register_center": anchored, "state_ratio": np.ones(count, dtype=bool), "density_per_track": np.ones(count, dtype=bool)})


def _evaluate_target(train: _Target, validation: _Target, train_latent: np.ndarray, validation_latent: np.ndarray) -> Mapping[str, Any]:
    # The target definition and availability mask are independent of fitting.
    train_valid = train.mask
    validation_mask = validation.mask
    if train_valid.sum() < 2 or validation_mask.sum() < 1:
        return {"status": "UNAVAILABLE", "family": train.family, "reason": "Target has insufficient available train or validation bars."}
    linear = _ridge_predict(train_latent[train_valid], train.values[train_valid], validation_latent[validation_mask])
    mlp = _mlp_predict(train_latent[train_valid], train.values[train_valid], validation_latent[validation_mask])
    actual = validation.values[validation_mask]
    return {"status": "MONITOR", "family": train.family, "train_bar_count": int(train_valid.sum()), "validation_bar_count": int(validation_mask.sum()), "linear": _scores(actual, linear, train.family), "mlp": _scores(actual, mlp, train.family)}


def _ridge_predict(train_x: np.ndarray, train_y: np.ndarray, validation_x: np.ndarray) -> np.ndarray:
    x_mean, x_scale = _mean_scale(train_x)
    y_mean, y_scale = _mean_scale(train_y)
    x = (train_x - x_mean) / x_scale
    y = (train_y - y_mean) / y_scale
    design = np.column_stack([np.ones(len(x)), x])
    weights = np.linalg.solve(design.T @ design + 1.0e-3 * np.eye(design.shape[1]), design.T @ y)
    return (np.column_stack([np.ones(len(validation_x)), (validation_x - x_mean) / x_scale]) @ weights) * y_scale + y_mean


def _mlp_predict(train_x: np.ndarray, train_y: np.ndarray, validation_x: np.ndarray) -> np.ndarray:
    """A deterministic one-hidden-layer regression MLP, fit only on train."""
    x_mean, x_scale = _mean_scale(train_x)
    y_mean, y_scale = _mean_scale(train_y)
    x, y = (train_x - x_mean) / x_scale, (train_y - y_mean) / y_scale
    rng = np.random.default_rng(17)
    hidden = min(32, max(4, train_x.shape[1] * 2))
    w1 = rng.normal(0.0, 0.15, (x.shape[1], hidden)); b1 = np.zeros(hidden)
    w2 = rng.normal(0.0, 0.10, (hidden, y.shape[1])); b2 = np.zeros(y.shape[1])
    for _ in range(300):
        pre = x @ w1 + b1; activation = np.maximum(pre, 0.0); error = (activation @ w2 + b2 - y) / len(x)
        grad_w2, grad_b2 = activation.T @ error, error.sum(axis=0)
        hidden_grad = (error @ w2.T) * (pre > 0.0)
        w1 -= 0.03 * (x.T @ hidden_grad); b1 -= 0.03 * hidden_grad.sum(axis=0)
        w2 -= 0.03 * grad_w2; b2 -= 0.03 * grad_b2
    validation = (validation_x - x_mean) / x_scale
    activation = np.maximum(validation @ w1 + b1, 0.0)
    return (activation @ w2 + b2) * y_scale + y_mean


def _scores(actual: np.ndarray, predicted: np.ndarray, family: str) -> Mapping[str, float]:
    mse = float(np.mean(np.square(actual - predicted)))
    variance = float(np.mean(np.square(actual - actual.mean(axis=0, keepdims=True))))
    result: dict[str, float] = {"mse": mse, "target_variance": variance, "nmse": mse / max(variance, 1.0e-8), "r2": float(1.0 - mse / max(variance, 1.0e-8))}
    if family == "chroma":
        cosine = np.sum(actual * predicted, axis=1) / np.maximum(np.linalg.norm(actual, axis=1) * np.linalg.norm(predicted, axis=1), 1.0e-8)
        result["chroma_cosine_mean"] = float(np.mean(cosine))
    return result


def _chroma(pitch: np.ndarray, active: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    result = np.zeros((len(pitch), 12), dtype=float)
    for row in range(len(pitch)):
        for value in pitch[row][active[row]]:
            result[row, int(np.rint(value + anchors[row])) % 12] += 1.0
    return result / np.maximum(result.sum(axis=1, keepdims=True), 1.0)


def _median_by_row(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.asarray([np.median(row[row_mask]) if np.any(row_mask) else np.nan for row, row_mask in zip(values, mask)])


def _mean_scale(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return values.mean(axis=0), np.maximum(values.std(axis=0), 1.0e-6)


def _assert_split_isolation(train: Sequence[Mapping[str, Any]], validation: Sequence[Mapping[str, Any]]) -> None:
    overlap = {str(row["base_song_id"]) for row in train} & {str(row["base_song_id"]) for row in validation}
    if overlap:
        raise ValueError("Latent probe train/validation base-song overlap is forbidden.")


def _target_missing(results: Mapping[str, Mapping[str, Any]]) -> list[Mapping[str, str]]:
    return [{"field": name, "reason": str(item["reason"])} for name, item in results.items() if item["status"] == "UNAVAILABLE"]


def _unavailable(inputs: Mapping[str, Any], reason: str) -> Mapping[str, Any]:
    return {"schema_version": "assessment_report.v1", "status": "UNAVAILABLE", "metrics": {}, "findings": [], "provenance": {"representation_sources": inputs.get("representation_sources", {})}, "missing_inputs": [{"field": "dvae_fidelity", "reason": reason}]}


def _markdown_unavailable() -> str:
    return "# 潜变量可读性探针\n\n生成这份报告所需的训练集和验证集 DVAE 原始观察数据不可用。\n"


def _markdown(report: Mapping[str, Any]) -> str:
    lines = ["# 潜变量可读性探针", "", "探针只用训练集拟合，并只在验证集报告。它回答潜变量是否保留可被后续模型读取的音乐信息；各项独立呈现，不合成为总分。", "", "| 音乐信息 | 验证小节 | 线性 R2 | MLP R2 | 线性 NMSE | MLP NMSE | Chroma cosine（线性 / MLP） |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    labels = {"relative_chroma": "相对 chroma", "absolute_chroma": "绝对 chroma", "relative_register_center": "相对音区中心", "absolute_register_center": "绝对音区中心", "state_ratio": "节奏状态比例", "density_per_track": "各声部密度"}
    for name, item in report["metrics"]["validation"].items():
        if item["status"] == "UNAVAILABLE":
            lines.append(f"| {labels[name]} | -- | -- | -- | -- | -- | 不可用 |")
            continue
        linear, mlp = item["linear"], item["mlp"]
        chroma = "--" if item["family"] != "chroma" else f"{linear['chroma_cosine_mean']:.3f} / {mlp['chroma_cosine_mean']:.3f}"
        lines.append(f"| {labels[name]} | {item['validation_bar_count']} | {linear['r2']:.3f} | {mlp['r2']:.3f} | {linear['nmse']:.3f} | {mlp['nmse']:.3f} | {chroma} |")
    lines.extend(["", "R2 接近 1 表示该目标在验证集上可由潜变量预测；0 表示不优于使用该验证集平均值，负值表示更差。线性和 MLP 同时较弱只说明该表示的可读性证据不足，不能单独证明模型没有保留音乐信息。", ""])
    return "\n".join(lines)


def _summary_png(results: Mapping[str, Mapping[str, Any]]) -> bytes:
    try:
        import matplotlib.pyplot as plt
        items = [(name, item) for name, item in results.items() if item["status"] != "UNAVAILABLE"]
        labels = [name.replace("_", "\n") for name, _ in items]
        x = np.arange(len(items)); width = 0.36
        figure, axis = plt.subplots(figsize=(9, 3.6))
        axis.bar(x - width / 2, [item["linear"]["r2"] for _, item in items], width, label="linear", color="#3B7EA1")
        axis.bar(x + width / 2, [item["mlp"]["r2"] for _, item in items], width, label="MLP", color="#2E8B57")
        axis.axhline(0.0, color="#666666", linewidth=0.8); axis.set_xticks(x, labels); axis.set_ylabel("validation R2"); axis.set_title("Latent readability by musical target", loc="left"); axis.legend(); axis.grid(axis="y", color="#DDDDDD", linewidth=0.7); axis.set_axisbelow(True)
        figure.tight_layout(); output = io.BytesIO(); figure.savefig(output, format="png", dpi=160); plt.close(figure)
        return output.getvalue()
    except Exception:
        return bytes.fromhex("89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4890000000d49444154789c6360f8cff0ff3f0005fe02fe8e4cacf50000000049454e44ae426082")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


LATENT_PROBE_MODULE = EvaluationModule(TEST_POINT, LatentProbeExporter(), LatentProbeEvaluator(), summary="Train-fit, validation-only probes for frozen DVAE latent readability.")
