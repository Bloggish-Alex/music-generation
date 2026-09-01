"""Pure measurements for paired teacher-forced trajectory evaluation."""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
KEY_NAMES = ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")
MAJOR_PROFILE = np.asarray([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88], dtype=np.float32)
MINOR_PROFILE = np.asarray([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17], dtype=np.float32)

def _semantic_target_prediction(data: Mapping[str, Any], position: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    indexes = data["target_source_stream_indices"][:, position]
    valid = indexes >= 0
    return data["source_bar_tensors"][indexes[valid]], data["predicted_bar_tensors"][valid, position], data["source_render_base_pitches"][indexes[valid]], data["predicted_render_base_pitches"][valid, position]


def _semantic_group_values(target: np.ndarray, prediction: np.ndarray, target_bases: np.ndarray, prediction_bases: np.ndarray, pitch_scale: float) -> list[tuple[str, np.ndarray, np.ndarray]]:
    def active(values): return ((values[..., 2] > .5) | (values[..., 3] > .5)).astype(np.float32)
    def relative_pitch(values): return values[..., 0] * active(values)
    def rhythm(values): return values[..., 1:4]
    def velocity(values): return values[..., 4] * active(values)
    def density(values): return np.mean(active(values), axis=-1)
    def chroma(values, bases):
        out = np.zeros((len(values), 12), dtype=np.float32); pitches = values[..., 0]
        for row, pitch, weight, base in zip(out, pitches, active(values), bases):
            for item, amount in zip(pitch.reshape(-1), weight.reshape(-1)):
                if amount: row[int(round(float(base) + float(item) * pitch_scale)) % 12] += float(amount)
            total = row.sum(); row[:] = row / total if total else row
        return out
    def relative_chroma(values): return chroma(values, np.zeros(len(values), dtype=np.float32))
    return [("absolute_chroma", chroma(target, target_bases), chroma(prediction, prediction_bases)), ("relative_chroma", relative_chroma(target), relative_chroma(prediction)), ("relative_pitch", relative_pitch(target), relative_pitch(prediction)), ("rhythm", rhythm(target), rhythm(prediction)), ("velocity", velocity(target), velocity(prediction)), ("density", density(target), density(prediction))]


def _mse(target: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.mean((np.asarray(target, dtype=np.float32) - np.asarray(prediction, dtype=np.float32)) ** 2))



def _metric_gaps(free: Mapping[str, Any], teacher: Mapping[str, Any]) -> dict[str, float]:
    return {
        "teacher_minus_free_diatonic_fit": teacher["diatonic_fit_mean"] - free["diatonic_fit_mean"],
        "teacher_minus_free_key_match_ratio": teacher["key_match_ratio"] - free["key_match_ratio"],
        "teacher_minus_free_chroma_similarity": teacher["chroma_similarity_mean"] - free["chroma_similarity_mean"],
        "teacher_minus_free_register_delta": teacher["register_delta_abs_mean"] - free["register_delta_abs_mean"],
    }


def _classification(gaps: Mapping[str, float]) -> str:
    if gaps["teacher_minus_free_diatonic_fit"] >= 0.15 or gaps["teacher_minus_free_key_match_ratio"] >= 0.20:
        return "recurrent_drift_supported"
    return "conditioning_boundary_unresolved"


def _classification_text(classification: str) -> str:
    if classification == "recurrent_drift_supported":
        return "真实历史显著恢复了与 primer 的调性对齐，递归生成历史是已获支持的漂移来源。"
    return "真实历史未显著恢复与 primer 的调性对齐；轨迹目标或条件边界仍需要进一步证据。"




def _absolute_chroma_by_bar(tensors: np.ndarray, bases: np.ndarray) -> np.ndarray:
    result = np.zeros((len(tensors), 12), dtype=np.float32)
    active = (tensors[..., 2] > 0.5) | (tensors[..., 3] > 0.5)
    velocity = np.maximum(tensors[..., 4], 0.0)
    for index, bar in enumerate(tensors):
        pitches = bar[..., 0] * 24.0 + bases[index]
        weights = active[index].astype(np.float32) * np.maximum(velocity[index], 1.0e-3)
        for pitch, weight in zip(pitches.reshape(-1), weights.reshape(-1)):
            if weight > 0.0:
                result[index, int(round(float(pitch))) % 12] += float(weight)
        result[index] = _normalize(result[index])
    return result


def _sequence_summary(chroma: np.ndarray, primer: np.ndarray, key: Mapping[str, Any]) -> dict[str, Any]:
    if len(chroma) == 0:
        return {"diatonic_fit_mean": 0.0, "key_match_ratio": 0.0, "chroma_similarity_mean": 0.0, "pitch_class_entropy_mean": 0.0}
    estimated = [_estimate_key(row) for row in chroma]
    return {
        "diatonic_fit_mean": float(np.mean([_diatonic_fit(row, key) for row in chroma])),
        "key_match_ratio": float(np.mean([value["name"] == key["name"] for value in estimated])),
        "chroma_similarity_mean": float(np.mean([_cosine(row, primer) for row in chroma])),
        "pitch_class_entropy_mean": float(np.mean([_entropy(row) for row in chroma])),
    }


def _estimate_key(chroma: np.ndarray) -> dict[str, Any]:
    candidates = []
    for root, name in enumerate(KEY_NAMES):
        for mode, template in (("major", MAJOR_PROFILE), ("minor", MINOR_PROFILE)):
            candidates.append({"name": f"{name} {mode}", "root": root, "mode": mode, "score": _cosine(_normalize(chroma), _normalize(np.roll(template, root)))})
    return max(candidates, key=lambda value: value["score"])


def _diatonic_fit(chroma: np.ndarray, key: Mapping[str, Any]) -> float:
    intervals = (0, 2, 4, 5, 7, 9, 11) if key["mode"] == "major" else (0, 2, 3, 5, 7, 8, 10)
    mask = np.zeros(12, dtype=np.float32)
    mask[(int(key["root"]) + np.asarray(intervals)) % 12] = 1.0
    return float(np.dot(_normalize(chroma), mask))


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.maximum(np.asarray(values, dtype=np.float32), 0.0)
    total = float(values.sum())
    return values / total if total else values


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 1.0e-8 else 0.0


def _entropy(values: np.ndarray) -> float:
    active = _normalize(values)
    active = active[active > 0.0]
    return float(-np.sum(active * np.log2(active)) / math.log2(12)) if len(active) else 0.0


def _future_positions(arms: Mapping[str, Mapping[str, Any]]) -> list[int]:
    return [1, 2, 3, 4] if all(arm["future_position"] is not None for arm in arms.values()) else []


