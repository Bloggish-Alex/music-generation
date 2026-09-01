"""Artifact-only assessment of a single free-running trajectory rollout."""

from __future__ import annotations

import io
import json
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .evaluation_api import ArtifactBundle, ArtifactEvaluator, ArtifactExporter, EvaluationModule, EvaluationResult
from .evaluation_context import EvaluationContext, ExportContext
from .evaluation_trajectory_teacher_forced_metrics import _absolute_chroma_by_bar, _estimate_key, _entropy, _normalize


TEST_POINT = "trajectory_rollout"
MANIFEST_NAMES = ("free_running__evaluation_manifest.json", "evaluation_manifest.json")


class TrajectoryRolloutExporter(ArtifactExporter):
    """Normalize one already-exported free-running generation artifact."""

    test_point = TEST_POINT
    input_contract = "evaluation_manifest.v1|evaluation_manifest.v2"
    output_contract = "trajectory_rollout_inputs.v1"

    def export(self, context: ExportContext) -> ArtifactBundle:
        manifest_path = next((context.input_root / name for name in MANIFEST_NAMES if (context.input_root / name).is_file()), None)
        if manifest_path is None:
            raise FileNotFoundError(f"Missing one of {', '.join(MANIFEST_NAMES)} in {context.input_root}.")
        manifest = _read_json(manifest_path)
        if manifest.get("schema_version") not in {"evaluation_manifest.v1", "evaluation_manifest.v2"}:
            raise ValueError("trajectory_rollout requires evaluation_manifest.v1 or evaluation_manifest.v2.")
        if manifest.get("run", {}).get("arm") not in {None, "free_running"}:
            raise ValueError("trajectory_rollout only accepts a free_running generation artifact.")
        refs = manifest.get("artifacts") or {}
        inputs_ref, tensor_ref = refs.get("generation_metric_inputs"), refs.get("bar_tensors")
        if not inputs_ref or not tensor_ref:
            raise ValueError("Generation manifest must reference generation_metric_inputs and bar_tensors.")
        inputs_path = (manifest_path.parent / str(inputs_ref["path"])).resolve()
        tensor_path = (manifest_path.parent / str(tensor_ref["path"])).resolve()
        _verify_ref(inputs_path, inputs_ref)
        _verify_ref(tensor_path, tensor_ref)
        generation_inputs = _read_json(inputs_path)
        if generation_inputs.get("history_source") not in {None, "generated"}:
            raise ValueError("trajectory_rollout requires generated history.")
        trace_ref = refs.get("generation_trace")
        trace_path = (manifest_path.parent / str(trace_ref["path"])).resolve() if trace_ref else None
        if trace_path and not trace_path.is_file():
            trace_path = None
        payload = {
            "schema_version": self.output_contract,
            "generation_manifest": _relative(manifest_path, context.input_root),
            "bar_tensor": {"path": _relative(tensor_path, context.input_root), "sha256": tensor_ref.get("sha256")},
            "render_register": {"primer_bars": int(generation_inputs.get("primer_bars", 0) or 0), "base_pitch_path": generation_inputs.get("base_pitch_path", [])},
            "trace": {"path": _relative(trace_path, context.input_root)} if trace_path else {"availability": "not_provided"},
            "provenance": {"dataset": manifest.get("dataset", {}), "run": manifest.get("run", {}), "checkpoint_refs": manifest.get("checkpoint_refs", [])},
            "availability": {"trace": "available" if trace_path else "not_provided", "midi": manifest.get("availability", {}).get("midi", "not_provided"), "multi_seed_reference": "not_provided"},
        }
        path = context.store.write_json(TEST_POINT, "inputs", payload)
        return ArtifactBundle(TEST_POINT, {"inputs": path.name})


class TrajectoryRolloutEvaluator(ArtifactEvaluator):
    test_point = TEST_POINT
    required_artifacts: Sequence[str] = ("inputs",)

    def evaluate(self, context: EvaluationContext, bundle: ArtifactBundle) -> EvaluationResult:
        inputs = _read_json(context.store.run_dir / bundle.artifacts["inputs"])
        if inputs.get("schema_version") != "trajectory_rollout_inputs.v1":
            raise ValueError("Unsupported trajectory rollout input schema.")
        arrays = np.load((context.input_root / inputs["bar_tensor"]["path"]).resolve(), allow_pickle=False)
        if "bars" not in arrays or "render_base_pitches" not in arrays:
            raise ValueError("bar_tensors must contain bars and render_base_pitches.")
        bars = np.asarray(arrays["bars"], dtype=np.float32)
        bases = np.asarray(arrays["render_base_pitches"], dtype=np.float32).reshape(-1)
        if bars.ndim != 4 or len(bars) != len(bases):
            raise ValueError("bars and render_base_pitches must align by bar.")
        primer = int(inputs["render_register"]["primer_bars"])
        chroma = _absolute_chroma_by_bar(bars, bases)
        primer_profile = _normalize(chroma[:primer].sum(axis=0)) if primer else np.zeros(12, dtype=np.float32)
        key = _estimate_key(primer_profile)
        generated = chroma[primer:]
        per_bar = [_bar_metrics(row, primer_profile, key) for row in generated]
        trace = _trace(inputs, context.input_root)
        metrics = {
            "primer_key": key["name"],
            "generated_bars": int(len(generated)),
            "diatonic_fit_mean": _mean(per_bar, "diatonic_fit"),
            "key_match_ratio": _mean(per_bar, "key_match"),
            "pitch_class_entropy_mean": _mean(per_bar, "pitch_class_entropy"),
            "register_delta_abs_mean": float(np.mean(np.abs(np.diff(bases)))) if len(bases) > 1 else 0.0,
            "register_range": float(np.max(bases) - np.min(bases)) if len(bases) else 0.0,
            "tensor_self_similarity": _mean_pairwise_cosine(bars[primer:].reshape(max(0, len(bars) - primer), -1)),
            "latent_self_similarity": _mean_pairwise_cosine(np.asarray(arrays["latent_mu"], dtype=np.float32)[primer:]) if "latent_mu" in arrays else "UNAVAILABLE",
            "plan_overlap_register_disagreement": _mean_nested(trace.get("overlapping_plan_disagreement", []), "register_disagreement") if trace else "UNAVAILABLE",
            "per_bar": per_bar,
        }
        missing = []
        if not trace:
            missing.append({"artifact": "generation_trace", "field": "overlapping_plan_disagreement", "reason": "public generation trace was not provided"})
        missing.extend([
            {"artifact": "trajectory_rollout_inputs", "field": "multi_seed_reference", "reason": "reference distribution across matched seeds and primers was not provided"},
            {"artifact": "trajectory_rollout_inputs", "field": "midi_bar_alignment", "reason": "renderer consistency is owned by the separate renderer_consistency module"},
        ])
        report = {"schema_version": "assessment_report.v1", "status": "MONITOR", "metrics": metrics,
                  "findings": [{"classification": "rollout_structure_monitor", "text": "单次 rollout 的结构指标仅用于观察；必须与数据集自然变化和多 seed 结果共同解释。"}],
                  "provenance": inputs["provenance"], "missing_inputs": missing}
        return EvaluationResult(report=report, markdown=_markdown(report), figures={"tonality_timeline": _timeline_png(per_bar)})


TRAJECTORY_ROLLOUT_MODULE = EvaluationModule(TEST_POINT, TrajectoryRolloutExporter(), TrajectoryRolloutEvaluator(), summary="Free-running rollout structure assessment.")


def _bar_metrics(chroma: np.ndarray, primer: np.ndarray, key: Mapping[str, Any]) -> dict[str, Any]:
    estimated = _estimate_key(chroma)
    intervals = (0, 2, 4, 5, 7, 9, 11) if key["mode"] == "major" else (0, 2, 3, 5, 7, 8, 10)
    mask = np.zeros(12, dtype=np.float32); mask[(int(key["root"]) + np.asarray(intervals)) % 12] = 1.0
    return {"estimated_key": estimated["name"], "diatonic_fit": float(np.dot(_normalize(chroma), mask)), "key_match": float(estimated["name"] == key["name"]), "chroma_similarity": _cosine(chroma, primer), "pitch_class_entropy": _entropy(chroma)}


def _trace(inputs: Mapping[str, Any], root: Path) -> Mapping[str, Any]:
    path = inputs.get("trace", {}).get("path")
    return _read_json((root / str(path)).resolve()) if path else {}


def _mean(values: list[Mapping[str, Any]], key: str) -> float:
    return float(np.mean([item[key] for item in values])) if values else 0.0


def _mean_nested(values: list[Mapping[str, Any]], key: str) -> float:
    numbers = [float(item[key]) for item in values if key in item]
    return float(np.mean(numbers)) if numbers else 0.0


def _mean_pairwise_cosine(values: np.ndarray) -> float:
    if len(values) < 2: return 0.0
    unit = values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-8)
    return float(np.mean((unit @ unit.T)[np.triu_indices(len(values), 1)]))


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 1e-8 else 0.0


def _timeline_png(per_bar: list[Mapping[str, Any]]) -> bytes:
    try:
        import matplotlib.pyplot as plt
        figure, axis = plt.subplots(figsize=(8, 3)); axis.plot([item["diatonic_fit"] for item in per_bar], label="Diatonic fit"); axis.plot([item["chroma_similarity"] for item in per_bar], label="Primer chroma similarity"); axis.set_ylim(0, 1); axis.legend(); axis.set_xlabel("Generated bar"); figure.tight_layout(); output = io.BytesIO(); figure.savefig(output, format="png", dpi=160); plt.close(figure); return output.getvalue()
    except Exception:
        return bytes.fromhex("89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4890000000d49444154789c6360f8cff0ff3f0005fe02fe8e4cacf50000000049454e44ae426082")


def _markdown(report: Mapping[str, Any]) -> str:
    metrics = report["metrics"]
    return "\n".join([
        "# Free-running Trajectory Rollout",
        "",
        f"Primer key: **{metrics['primer_key']}**",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Generated bars | {metrics['generated_bars']} |",
        f"| Diatonic fit | {metrics['diatonic_fit_mean']:.3f} |",
        f"| Key-match ratio | {metrics['key_match_ratio']:.3f} |",
        f"| Pitch-class entropy | {metrics['pitch_class_entropy_mean']:.3f} |",
        f"| Register range | {metrics['register_range']:.3f} |",
        f"| Tensor self-similarity | {metrics['tensor_self_similarity']:.3f} |",
        "",
        "These single-rollout observations are monitors. They are not a total music-quality score or a generation constraint.",
        "",
    ])


def _read_json(path: Path) -> dict[str, Any]: return json.loads(path.read_text(encoding="utf-8"))

def _verify_ref(path: Path, reference: Mapping[str, Any]) -> None:
    if not path.is_file(): raise FileNotFoundError(f"Missing referenced artifact: {path}")
    expected = reference.get("sha256")
    if expected and expected != f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}":
        raise ValueError(f"Artifact hash mismatch for {path.name}")

def _relative(path: Path, root: Path) -> str:
    try: return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error: raise ValueError("All rollout artifacts must be below --input-root.") from error
