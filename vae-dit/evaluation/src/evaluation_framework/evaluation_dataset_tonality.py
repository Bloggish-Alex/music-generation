"""Artifact-only reference profile for source-dataset tonality."""

from __future__ import annotations

import io
import json
import hashlib
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .evaluation_api import ArtifactBundle, ArtifactEvaluator, ArtifactExporter, EvaluationModule, EvaluationResult
from .evaluation_context import EvaluationContext, ExportContext


TEST_POINT = "dataset_tonality"
RAW_SCHEMA = "dataset_tonality_raw_source.v1"
INPUT_SCHEMA = "dataset_tonality_inputs.v1"
_SPLITS = ("train", "validation", "excluded_unpaired")
_KEY_TEMPLATES = {
    "major": np.asarray([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]),
    "minor": np.asarray([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]),
}


class DatasetTonalityExporter(ArtifactExporter):
    """Discover versioned raw-source observations without reading model state."""

    test_point = TEST_POINT
    input_contract = RAW_SCHEMA
    output_contract = INPUT_SCHEMA

    def export(self, context: ExportContext) -> ArtifactBundle:
        splits: dict[str, Any] = {}
        availability: dict[str, str] = {}
        bundle = _read_bundle(context.input_root)
        for split in _SPLITS:
            path, expected_hash = _raw_path(context.input_root, bundle, split)
            if not path.is_file():
                availability[split] = "not_provided"
                continue
            if expected_hash and expected_hash != f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}":
                raise ValueError(f"Dataset tonality raw artifact hash mismatch: {path.name}")
            payload = _read_json(path)
            _validate_raw(payload, split)
            splits[split] = {"path": _relative(path, context.input_root), "sha256": f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"}
            availability[split] = "available"
        inputs = {"schema_version": INPUT_SCHEMA, "splits": splits, "availability": availability}
        path = context.store.write_json(TEST_POINT, "inputs", inputs)
        return ArtifactBundle(TEST_POINT, {"inputs": path.name})


class DatasetTonalityEvaluator(ArtifactEvaluator):
    """Measure source-music distributions, keeping train and validation separate."""

    test_point = TEST_POINT
    required_artifacts: Sequence[str] = ("inputs",)

    def evaluate(self, context: EvaluationContext, bundle: ArtifactBundle) -> EvaluationResult:
        inputs = _read_json(context.store.run_dir / bundle.artifacts["inputs"])
        if inputs.get("schema_version") != INPUT_SCHEMA:
            raise ValueError("Unsupported dataset tonality input schema.")
        availability = dict(inputs.get("availability") or {})
        if availability.get("train") != "available" or availability.get("validation") != "available":
            missing = [split for split in ("train", "validation") if availability.get(split) != "available"]
            report = _unavailable_report(inputs, missing)
            return EvaluationResult(report=report, markdown=_markdown(report))

        raws = {split: _load_raw_reference(context.input_root, inputs["splits"][split], split) for split in ("train", "validation")}
        profiles = {split: _profile_split(raws[split]) for split in raws}
        if "excluded_unpaired" in inputs["splits"]:
            profiles["excluded_unpaired"] = _profile_split(_load_raw_reference(context.input_root, inputs["splits"]["excluded_unpaired"], "excluded_unpaired"))
        comparison = _compare(profiles["train"], profiles["validation"])
        provenance = {
            "dataset": profiles["train"]["dataset"],
            "input_artifacts": dict(inputs["splits"]),
        }
        report = {
            "schema_version": "assessment_report.v1",
            "status": "MONITOR",
            "metrics": {"profiles": profiles, "train_validation_gap": comparison},
            "findings": [{
                "classification": "reference_profile_monitor",
                "text": "本报告描述参考音乐集本身。训练集与验证集的差异分别作为音乐分布呈现，不合成为质量总分，也不构成生成规则。",
            }],
            "provenance": provenance,
            "missing_inputs": [],
        }
        return EvaluationResult(report=report, markdown=_markdown(report), figures={"pitch_class_profile": _profile_png(profiles)})


DATASET_TONALITY_MODULE = EvaluationModule(
    TEST_POINT,
    DatasetTonalityExporter(),
    DatasetTonalityEvaluator(),
    summary="Source-dataset tonality, register and rhythmic-density reference profile.",
)


def _validate_raw(payload: Mapping[str, Any], expected_split: str) -> None:
    if payload.get("schema_version") != RAW_SCHEMA:
        raise ValueError(f"{expected_split} raw source has an unsupported schema version.")
    dataset = payload.get("dataset")
    if not isinstance(dataset, Mapping) or dataset.get("split") != expected_split:
        raise ValueError(f"Raw source split does not match filename: {expected_split}.")
    if not isinstance(payload.get("songs"), list) or not payload["songs"]:
        raise ValueError(f"Raw source {expected_split} has no songs.")
    if not isinstance(payload.get("availability"), Mapping) or not payload["availability"].get("bar_note_events"):
        raise ValueError(f"Raw source {expected_split} does not provide bar note events.")


def _read_bundle(root: Path) -> Mapping[str, Any] | None:
    path = root / "dataset_tonality__source_bundle.v1.json"
    if not path.is_file():
        return None
    payload = _read_json(path)
    if payload.get("schema_version") != "dataset_tonality_source_bundle.v1" or not isinstance(payload.get("artifacts"), Mapping):
        raise ValueError("Unsupported dataset-tonality source bundle.")
    return payload


def _raw_path(root: Path, bundle: Mapping[str, Any] | None, split: str) -> tuple[Path, str | None]:
    if bundle is None:
        return root / f"dataset_tonality__raw_source__{split}.v1.json", None
    reference = bundle["artifacts"].get(split)
    if not isinstance(reference, Mapping):
        return root / f"dataset_tonality__raw_source__{split}.v1.json", None
    candidate = (root / str(reference.get("path", ""))).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("Dataset-tonality bundle artifact must be below input root.") from error
    return candidate, str(reference["sha256"]) if reference.get("sha256") else None


def _load_raw_reference(root: Path, reference: Mapping[str, Any], split: str) -> dict[str, Any]:
    path = (root / str(reference["path"])).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("Dataset-tonality input artifact must be below input root.") from error
    if not path.is_file():
        raise FileNotFoundError(f"Missing dataset-tonality raw artifact: {path.name}")
    expected = str(reference["sha256"])
    observed = f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"
    if observed != expected:
        raise ValueError(f"Dataset-tonality raw artifact hash mismatch: {path.name}")
    payload = _read_json(path)
    _validate_raw(payload, split)
    return payload


def _profile_split(raw: Mapping[str, Any]) -> dict[str, Any]:
    normalized = [_profile_song(song, restore_source_transpose=True) for song in raw["songs"]]
    augmented = [_profile_song(song, restore_source_transpose=False) for song in raw["songs"]]
    canonical_by_base: dict[str, dict[str, Any]] = {}
    for song in normalized:
        current = canonical_by_base.get(song["base_song_id"])
        if current is None or _canonical_sort_key(song) < _canonical_sort_key(current):
            canonical_by_base[song["base_song_id"]] = song
    base_profiles = [canonical_by_base[key] for key in sorted(canonical_by_base)]
    return {
        "dataset": raw["dataset"],
        "source": raw.get("source", {}),
        "base_work_count": len(base_profiles),
        "augmentation_record_count": len(normalized),
        "songs": [_public_song_profile(song) for song in base_profiles],
        "aggregate": _aggregate_profiles(base_profiles),
        "augmented_input_aggregate": _aggregate_profiles(augmented),
    }


def _profile_song(song: Mapping[str, Any], *, restore_source_transpose: bool) -> dict[str, Any]:
    bars = song.get("bars") or []
    pitches: list[int] = []
    chroma = np.zeros(12, dtype=float)
    onset_count = 0
    empty_bars = 0
    active_tracks: list[int] = []
    for bar in bars:
        notes = bar.get("notes") or []
        if not notes:
            empty_bars += 1
        active_tracks.append(len({int(note["track_index"]) for note in notes}))
        onset_count += len({(int(note["track_index"]), float(note["onset_ql"])) for note in notes})
        for note in notes:
            pitch, duration = int(note["pitch"]), float(note["duration_ql"])
            if restore_source_transpose:
                pitch -= int(song["applied_transpose_semitones"])
            pitches.append(pitch)
            chroma[pitch % 12] += duration
    return {
        "song_id": str(song["song_id"]),
        "base_song_id": str(song["base_song_id"]),
        "applied_transpose_semitones": int(song["applied_transpose_semitones"]),
        "bar_count": len(bars),
        "note_count": len(pitches),
        "onset_count": onset_count,
        "empty_bar_ratio": float(empty_bars / len(bars)) if bars else 1.0,
        "active_track_count_mean": float(np.mean(active_tracks)) if active_tracks else 0.0,
        "notes_per_bar": float(len(pitches) / len(bars)) if bars else 0.0,
        "chroma": _normalise(chroma).round(8).tolist(),
        "estimated_key": _estimate_key(chroma),
        "register": _distribution(pitches),
        "_pitches": pitches,
    }


def _canonical_sort_key(song: Mapping[str, Any]) -> tuple[int, str]:
    return (abs(int(song["applied_transpose_semitones"])), str(song["song_id"]))


def _public_song_profile(song: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in song.items() if key != "_pitches"}


def _aggregate_profiles(song_profiles: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    chroma = np.asarray([song["chroma"] for song in song_profiles], dtype=float).sum(axis=0) if song_profiles else np.zeros(12, dtype=float)
    pitches = [pitch for song in song_profiles for pitch in song["_pitches"]]
    note_counts = np.asarray([song["note_count"] for song in song_profiles], dtype=float)
    onset_counts = np.asarray([song["onset_count"] for song in song_profiles], dtype=float)
    density = np.asarray([song["notes_per_bar"] for song in song_profiles], dtype=float)
    key_counts = Counter(song["estimated_key"] for song in song_profiles)
    return {
        "pitch_class_distribution": _normalise(chroma).round(8).tolist(),
        "estimated_key_distribution": dict(sorted(key_counts.items())),
        "note_count_total": int(note_counts.sum()) if len(note_counts) else 0,
        "register": _distribution(pitches),
        "rhythm_density": {
            "notes_per_bar_mean": float(density.mean()) if len(density) else 0.0,
            "onsets_per_song_mean": float(onset_counts.mean()) if len(onset_counts) else 0.0,
            "empty_bar_ratio": float(np.mean([song["empty_bar_ratio"] for song in song_profiles])) if song_profiles else 0.0,
            "active_track_count_mean": float(np.mean([song["active_track_count_mean"] for song in song_profiles])) if song_profiles else 0.0,
        },
    }


def _compare(train: Mapping[str, Any], validation: Mapping[str, Any]) -> dict[str, Any]:
    left, right = np.asarray(train["aggregate"]["pitch_class_distribution"]), np.asarray(validation["aggregate"]["pitch_class_distribution"])
    return {
        "pitch_class_total_variation": float(0.5 * np.abs(left - right).sum()),
        "pitch_class_cosine_similarity": _cosine(left, right),
        "register_median_gap_semitones": float(validation["aggregate"]["register"]["median"] - train["aggregate"]["register"]["median"]),
        "notes_per_bar_mean_gap": float(validation["aggregate"]["rhythm_density"]["notes_per_bar_mean"] - train["aggregate"]["rhythm_density"]["notes_per_bar_mean"]),
    }


def _distribution(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"minimum": None, "median": None, "maximum": None, "p10": None, "p90": None}
    array = np.asarray(values, dtype=float)
    return {"minimum": float(array.min()), "median": float(np.median(array)), "maximum": float(array.max()), "p10": float(np.percentile(array, 10)), "p90": float(np.percentile(array, 90))}


def _estimate_key(chroma: np.ndarray) -> str:
    values = _normalise(chroma)
    if not np.any(values):
        return "UNAVAILABLE"
    candidates = []
    for mode, template in _KEY_TEMPLATES.items():
        for root in range(12):
            candidates.append((float(np.dot(values, _normalise(np.roll(template, root)))), root, mode))
    _, root, mode = max(candidates)
    return f"{('C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B')[root]} {mode}"


def _normalise(values: np.ndarray) -> np.ndarray:
    total = float(values.sum())
    return values / total if total > 0 else np.zeros_like(values, dtype=float)


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 1e-12 else 0.0


def _unavailable_report(inputs: Mapping[str, Any], missing: Sequence[str]) -> dict[str, Any]:
    return {
        "schema_version": "assessment_report.v1", "status": "UNAVAILABLE", "metrics": {}, "findings": [],
        "provenance": {"input_availability": inputs.get("availability", {})},
        "missing_inputs": [{"artifact": "dataset_tonality_raw_source", "field": split, "reason": "A train/validation reference profile requires this raw split."} for split in missing],
    }


def _profile_png(profiles: Mapping[str, Mapping[str, Any]]) -> bytes:
    try:
        import matplotlib.pyplot as plt
        figure, axis = plt.subplots(figsize=(8, 3.5))
        labels = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
        for split in ("train", "validation"):
            if split in profiles:
                axis.plot(labels, profiles[split]["aggregate"]["pitch_class_distribution"], marker="o", label=split)
        axis.set_ylabel("Relative sounding duration")
        axis.set_xlabel("Pitch class")
        axis.legend()
        figure.tight_layout()
        output = io.BytesIO(); figure.savefig(output, format="png", dpi=160); plt.close(figure)
        return output.getvalue()
    except Exception:
        return bytes.fromhex("89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4890000000d49444154789c6360f8cff0ff3f0005fe02fe8e4cacf50000000049454e44ae426082")


def _markdown(report: Mapping[str, Any]) -> str:
    if report["status"] == "UNAVAILABLE":
        return "# 数据集调性画像\n\n生成这份画像所需的训练集或验证集原始音乐观察数据不可用。\n"
    profiles, gap = report["metrics"]["profiles"], report["metrics"]["train_validation_gap"]
    lines = ["# 数据集调性画像", "", "这份报告描述参考音乐集里的音乐材料，不是模型分数，也不规定生成音乐应如何修改。调性和音区已还原至每首作品移调扩增前的原始参考系。", "", "| 数据部分 | 原始作品数 | 扩增版本数 | 音符数 | 音区中位数 | 每小节音符数 |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for split in ("train", "validation"):
        profile = profiles[split]
        lines.append(f"| {split} | {profile['base_work_count']} | {profile['augmentation_record_count']} | {profile['aggregate']['note_count_total']} | {profile['aggregate']['register']['median']:.1f} | {profile['aggregate']['rhythm_density']['notes_per_bar_mean']:.2f} |")
    lines += ["", "## 训练集与验证集的差异", "", "| 音乐方面 | 差异 |", "| --- | ---: |", f"| 音高类别分布差异 | {gap['pitch_class_total_variation']:.3f} |", f"| 音高类别相似度 | {gap['pitch_class_cosine_similarity']:.3f} |", f"| 音区中位数差异（半音） | {gap['register_median_gap_semitones']:.2f} |", f"| 每小节音符数差异 | {gap['notes_per_bar_mean_gap']:.2f} |", "", "分布较宽或两个数据部分有所不同，首先是这批音乐材料的性质；它本身不表示模型存在问题。", ""]
    return "\n".join(lines)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError("Dataset-tonality raw artifact must be below input root.") from error
