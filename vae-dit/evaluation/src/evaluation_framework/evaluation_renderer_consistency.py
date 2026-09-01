"""Artifact-only renderer input versus emitted MIDI consistency assessment."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .core.artifacts import VerifiedArtifactResolver
from .core.tensor_schema import SemanticTensorDecoder
from .evaluation_api import ArtifactBundle, ArtifactEvaluator, ArtifactExporter, EvaluationModule, EvaluationResult
from .evaluation_context import EvaluationContext, ExportContext


TEST_POINT = "renderer_consistency"
INPUT_SCHEMA = "renderer_consistency_inputs.v1"
RAW_STATUS_SCHEMA = "renderer_consistency_raw_status.v3"


class RendererConsistencyExporter(ArtifactExporter):
    test_point = TEST_POINT
    input_contract = RAW_STATUS_SCHEMA
    output_contract = INPUT_SCHEMA

    def export(self, context: ExportContext) -> ArtifactBundle:
        status_path = context.input_root / "renderer_consistency__raw_status.v3.json"
        payload: dict[str, Any] = {"schema_version": INPUT_SCHEMA, "status": None, "status_reference": None}
        if status_path.is_file():
            status = _read_json(status_path)
            if status.get("schema_version") != RAW_STATUS_SCHEMA:
                raise ValueError("Unsupported renderer status schema.")
            payload.update(status=str(status.get("status")), status_reference=_reference(status_path))
        artifact = context.store.write_json(TEST_POINT, "inputs", payload)
        return ArtifactBundle(TEST_POINT, {"inputs": artifact.name})


class RendererConsistencyEvaluator(ArtifactEvaluator):
    test_point = TEST_POINT
    required_artifacts: Sequence[str] = ("inputs",)

    def evaluate(self, context: EvaluationContext, bundle: ArtifactBundle) -> EvaluationResult:
        inputs = _read_json(context.store.run_dir / bundle.artifacts["inputs"])
        if inputs.get("schema_version") != INPUT_SCHEMA:
            raise ValueError("Unsupported renderer inputs schema.")
        reference = inputs.get("status_reference")
        if not reference:
            return _unavailable(inputs, "No renderer raw status artifact was provided.")

        resolver = VerifiedArtifactResolver(context.input_root)
        status = resolver.json(reference)
        if status.get("status") != "AVAILABLE":
            return _unavailable(inputs, _status_reason(status, "Renderer facts are unavailable."))

        observation = resolver.json(status["artifacts"]["observation"])
        arrays = resolver.npz(observation["arrays"])
        try:
            metrics = _measure(observation, arrays, resolver)
        finally:
            arrays.close()
        report = _report(observation, status, metrics)
        return EvaluationResult(report, _markdown(report), {"comparison": _png(metrics)})


RENDERER_CONSISTENCY_MODULE = EvaluationModule(
    TEST_POINT,
    RendererConsistencyExporter(),
    RendererConsistencyEvaluator(),
    summary="Renderer input tensor and emitted MIDI consistency.",
)


def _measure(observation: Mapping[str, Any], arrays: Any, resolver: VerifiedArtifactResolver) -> dict[str, Any]:
    bars = np.asarray(arrays["bar_tensors"], dtype=float)
    base_pitches = np.asarray(arrays["render_base_pitches"], dtype=float).reshape(-1)
    decoder = SemanticTensorDecoder.from_schema(observation["tensor_schema"])
    active = decoder.active_mask(bars)
    tensor_pitches = decoder.absolute_pitch(bars, base_pitches)
    midi_notes = _midi_note_on_events(resolver.path(observation["midi"]))
    midi_by_bar = _notes_by_bar(midi_notes, observation["bar_alignment"])
    tensor_chroma = [_chroma(tensor_pitches[index][active[index]]) for index in range(len(bars))]
    midi_chroma = [_chroma(notes) for notes in midi_by_bar]
    chroma_cosines = [_cosine(left, right) for left, right in zip(tensor_chroma, midi_chroma) if left.sum() and right.sum()]
    register_errors = _register_errors(tensor_pitches, active, midi_by_bar)
    density_errors = [abs(int(active[index].sum()) - len(notes)) for index, notes in enumerate(midi_by_bar)]
    return {
        "bar_count": int(len(bars)),
        "comparable_chroma_bar_count": len(chroma_cosines),
        "tensor_to_midi": {
            "chroma_cosine_mean": _mean_or_none(chroma_cosines),
            "chroma_cosine_p10": float(np.quantile(chroma_cosines, 0.1)) if chroma_cosines else None,
            "register_median_absolute_error_semitones": float(np.median(register_errors)) if register_errors else None,
            "register_mean_absolute_error_semitones": _mean_or_none(register_errors),
            "note_count_difference_mean": _mean_or_none(density_errors),
        },
        "unavailable_reasons": [] if chroma_cosines else ["No bars contain comparable tensor and MIDI pitches."],
    }


def _notes_by_bar(events: Sequence[tuple[int, int]], alignment: Sequence[Mapping[str, Any]]) -> list[list[int]]:
    return [[pitch for tick, pitch in events if int(row["start_tick"]) <= tick < int(row["end_tick"])] for row in alignment]


def _register_errors(tensor_pitches: np.ndarray, active: np.ndarray, midi_by_bar: Sequence[Sequence[int]]) -> list[float]:
    errors: list[float] = []
    for index, midi_pitches in enumerate(midi_by_bar):
        audible = tensor_pitches[index][active[index]]
        if audible.size and midi_pitches:
            errors.append(abs(float(np.median(audible)) - float(np.median(midi_pitches))))
    return errors


def _midi_note_on_events(path: Path) -> list[tuple[int, int]]:
    import mido

    events: list[tuple[int, int]] = []
    for track in mido.MidiFile(path).tracks:
        tick = 0
        for event in track:
            tick += event.time
            if event.type == "note_on" and event.velocity > 0:
                events.append((tick, int(event.note)))
    return events


def _chroma(pitches: Sequence[float]) -> np.ndarray:
    result = np.zeros(12)
    for pitch in pitches:
        result[int(round(float(pitch))) % 12] += 1
    return result


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(left, right) / max(np.linalg.norm(left) * np.linalg.norm(right), 1e-8))


def _mean_or_none(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _report(observation: Mapping[str, Any], status: Mapping[str, Any], metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "assessment_report.v1",
        "status": "MONITOR",
        "metrics": metrics,
        "findings": [{"classification": "renderer_boundary_measurement", "text": "本报告只比较送入渲染器的小节张量与实际 MIDI。它定位渲染边界，不评价生成音乐本身。"}],
        "provenance": {
            "run": observation["run"],
            "input_hashes": {
                "observation": status["artifacts"]["observation"]["sha256"],
                "arrays": observation["arrays"]["sha256"],
                "midi": observation["midi"]["sha256"],
            },
        },
        "missing_inputs": [],
    }


def _reference(path: Path) -> dict[str, str]:
    return {"path": path.name, "sha256": VerifiedArtifactResolver.sha256(path)}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _status_reason(status: Mapping[str, Any], fallback: str) -> str:
    reasons = [item.get("reason", "") for item in status.get("unavailable_reasons", [])]
    return "; ".join(reason for reason in reasons if reason) or fallback


def _unavailable(inputs: Mapping[str, Any], reason: str) -> EvaluationResult:
    report = {
        "schema_version": "assessment_report.v1",
        "status": "UNAVAILABLE",
        "metrics": {},
        "findings": [],
        "provenance": {"input_status": inputs.get("status")},
        "missing_inputs": [{"field": TEST_POINT, "reason": reason}],
    }
    return EvaluationResult(report, _markdown(report))


def _markdown(report: Mapping[str, Any]) -> str:
    if report["status"] == "UNAVAILABLE":
        return "# Renderer 一致性\n\n生成这份报告所需的 renderer 原始观察数据不可用。\n- " + report["missing_inputs"][0]["reason"] + "\n"
    values = report["metrics"]["tensor_to_midi"]
    show = lambda value: "--" if value is None else f"{value:.3f}"
    return (
        "# Renderer 一致性\n\n"
        "比较实际送入 renderer 的张量与实际写出的 MIDI；不评价旋律或和声质量。\n\n"
        "| 小节 | Chroma cosine | 音区中位绝对误差（半音） | 平均 note-count 差 |\n"
        "| ---: | ---: | ---: | ---: |\n"
        f"| {report['metrics']['bar_count']} | {show(values['chroma_cosine_mean'])} | "
        f"{show(values['register_median_absolute_error_semitones'])} | {show(values['note_count_difference_mean'])} |\n"
    )


def _png(metrics: Mapping[str, Any]) -> bytes:
    try:
        import matplotlib.pyplot as plt

        values = metrics["tensor_to_midi"]
        figure, axis = plt.subplots(figsize=(5, 3))
        axis.bar(["Chroma cosine", "Register MAE"], [values["chroma_cosine_mean"] or 0, values["register_mean_absolute_error_semitones"] or 0])
        figure.tight_layout()
        output = io.BytesIO()
        figure.savefig(output, format="png", dpi=150)
        plt.close(figure)
        return output.getvalue()
    except Exception:
        return b""
