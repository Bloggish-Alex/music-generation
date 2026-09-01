"""Pure numerical metrics for artifact-only DVAE fidelity assessment."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

def _feature_indices(names: Sequence[Any]) -> dict[str, Any]:
    normalized = [str(name) for name in names]
    required = ("relative_pitch", "is_rest", "is_note_on", "is_hold", "normalized_velocity")
    missing = [name for name in required if name not in normalized]
    if missing:
        raise ValueError(f"DVAE fidelity tensor schema is missing required features: {', '.join(missing)}")
    return {
        "relative_pitch": normalized.index("relative_pitch"),
        "state": [normalized.index(name) for name in ("is_rest", "is_note_on", "is_hold")],
        "velocity": normalized.index("normalized_velocity"),
        "chroma": [index for index, name in enumerate(normalized) if "chroma_embed_" in name or "chord_embed_" in name],
        "density": normalized.index("density_gradient") if "density_gradient" in normalized else None,
    }
def _state_labels(tensor: np.ndarray, features: Mapping[str, Any]) -> np.ndarray:
    return np.argmax(tensor[..., features["state"]], axis=-1)


def _state_metrics(source: np.ndarray, decoded: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(np.mean(source == decoded)),
        "onset_f1": _binary_f1(source == 1, decoded == 1),
        "hold_f1": _binary_f1(source == 2, decoded == 2),
    }


def _pitch_metrics(
    source: np.ndarray,
    decoded: np.ndarray,
    mask: np.ndarray,
    schema: Mapping[str, Any],
) -> dict[str, Any]:
    """Separate pitch-class retention from octave placement and voice roles."""
    if not np.any(mask):
        return {
            "mae_semitones": None,
            "rmse_semitones": None,
            "pitch_class_mae_semitones": None,
            "octave_component_mae_semitones": None,
            "octave_equivalent_error_rate": None,
            "error_bands": _empty_error_bands(),
            "voice_roles": _voice_role_metrics(source, decoded, mask, schema),
            "distance_to_anchor": _distance_to_anchor_metrics(source, decoded, mask, schema),
            "shared_active_slot_count": 0,
        }
    errors = decoded[mask] - source[mask]
    octave_component = 12.0 * np.rint(errors / 12.0)
    pitch_class_residual = errors - octave_component
    return {
        "mae_semitones": float(np.mean(np.abs(errors))),
        "rmse_semitones": float(np.sqrt(np.mean(np.square(errors)))),
        "pitch_class_mae_semitones": float(np.mean(np.abs(pitch_class_residual))),
        "octave_component_mae_semitones": float(np.mean(np.abs(octave_component))),
        "octave_equivalent_error_rate": float(np.mean((np.abs(pitch_class_residual) <= 0.5) & (np.abs(octave_component) >= 12.0))),
        "error_bands": _error_bands(errors),
        "voice_roles": _voice_role_metrics(source, decoded, mask, schema),
        "distance_to_anchor": _distance_to_anchor_metrics(source, decoded, mask, schema),
        "shared_active_slot_count": int(np.sum(mask)),
    }


def _error_bands(errors: np.ndarray) -> dict[str, float | int]:
    """Use tolerance bands because decoded pitch values may be continuous."""
    tolerance = 0.5
    bands = {"zero": 0.0, "plus_12": 12.0, "minus_12": -12.0, "plus_24": 24.0, "minus_24": -24.0}
    matches = {name: np.abs(errors - center) <= tolerance for name, center in bands.items()}
    recognized = np.logical_or.reduce(list(matches.values()))
    return {
        "tolerance_semitones": tolerance,
        "sample_count": int(len(errors)),
        "zero_rate": float(np.mean(matches["zero"])),
        "plus_12_rate": float(np.mean(matches["plus_12"])),
        "minus_12_rate": float(np.mean(matches["minus_12"])),
        "plus_24_rate": float(np.mean(matches["plus_24"])),
        "minus_24_rate": float(np.mean(matches["minus_24"])),
        "other_error_rate": float(np.mean(~recognized)),
    }


def _empty_error_bands() -> dict[str, float | int | None]:
    return {
        "tolerance_semitones": 0.5,
        "sample_count": 0,
        "zero_rate": None,
        "plus_12_rate": None,
        "minus_12_rate": None,
        "plus_24_rate": None,
        "minus_24_rate": None,
        "other_error_rate": None,
    }


def _voice_role_metrics(
    source: np.ndarray,
    decoded: np.ndarray,
    mask: np.ndarray,
    schema: Mapping[str, Any],
) -> dict[str, Mapping[str, float | int | str | None]]:
    track_names = [str(name) for name in schema.get("track_names") or []]
    if len(track_names) != source.shape[1]:
        raise ValueError("DVAE fidelity tensor schema track names do not match tensor tracks.")
    metrics: dict[str, Mapping[str, float | int | str | None]] = {}
    for role in ("melody", "harmony", "bass"):
        if role not in track_names:
            metrics[role] = {"status": "UNAVAILABLE", "reason": f"Tensor schema has no semantic {role} track."}
            continue
        track = track_names.index(role)
        metrics[role] = _voice_pitch_error(source[:, track], decoded[:, track], mask[:, track])
    return metrics


def _voice_pitch_error(source: np.ndarray, decoded: np.ndarray, mask: np.ndarray) -> dict[str, float | int | str | None]:
    if not np.any(mask):
        return {"status": "UNAVAILABLE", "reason": "No shared active slots for this voice role.", "mae_semitones": None, "shared_active_slot_count": 0}
    target = source[mask]
    values = decoded[mask]
    errors = values - target
    regression = _regression_metrics(target, values)
    return {
        "status": "MONITOR",
        "mae_semitones": float(np.mean(np.abs(errors))),
        "rmse_semitones": float(np.sqrt(np.mean(np.square(errors)))),
        "signed_bias_semitones": float(np.mean(errors)),
        "target_relative_pitch_mean_semitones": float(np.mean(target)),
        "target_relative_pitch_std_semitones": float(np.std(target)),
        "decoded_relative_pitch_mean_semitones": float(np.mean(values)),
        "decoded_relative_pitch_std_semitones": float(np.std(values)),
        "shared_active_slot_count": int(len(target)),
        **regression,
    }


def _regression_metrics(target: np.ndarray, decoded: np.ndarray) -> dict[str, float | None]:
    if len(target) < 2 or np.isclose(np.var(target), 0.0):
        return {"slope": None, "intercept_semitones": None, "pearson_correlation": None}
    slope, intercept = np.polyfit(target, decoded, 1)
    correlation = np.corrcoef(target, decoded)[0, 1] if not np.isclose(np.std(decoded), 0.0) else 0.0
    return {"slope": float(slope), "intercept_semitones": float(intercept), "pearson_correlation": float(correlation)}


def _distance_to_anchor_metrics(
    source: np.ndarray,
    decoded: np.ndarray,
    mask: np.ndarray,
    schema: Mapping[str, Any],
) -> dict[str, Any]:
    track_names = [str(name) for name in schema.get("track_names") or []]
    if len(track_names) != source.shape[1]:
        raise ValueError("DVAE fidelity tensor schema track names do not match tensor tracks.")
    result: dict[str, Any] = {"distance_definition": "absolute_source_relative_pitch_semitones", "all_shared_active_slots": _distance_bands(source[mask], decoded[mask])}
    voices: dict[str, Any] = {}
    for role in ("melody", "harmony", "bass"):
        if role not in track_names:
            voices[role] = {"status": "UNAVAILABLE", "reason": f"Tensor schema has no semantic {role} track."}
            continue
        track = track_names.index(role)
        if not np.any(mask[:, track]):
            voices[role] = {"status": "UNAVAILABLE", "reason": "No shared active slots for this voice role."}
            continue
        voices[role] = {"status": "MONITOR", "bands": _distance_bands(source[:, track][mask[:, track]], decoded[:, track][mask[:, track]])}
    result["voice_roles"] = voices
    return result


def _distance_bands(target: np.ndarray, decoded: np.ndarray) -> list[dict[str, float | int | str | None]]:
    bands = ((0.0, 6.0, "0-5"), (6.0, 12.0, "6-11"), (12.0, 18.0, "12-17"), (18.0, 24.0, "18-23"), (24.0, 30.0, "24-29"), (30.0, None, "30+"))
    distance = np.abs(target)
    errors = decoded - target
    result: list[dict[str, float | int | str | None]] = []
    for lower, upper, label in bands:
        selected = distance >= lower if upper is None else (distance >= lower) & (distance < upper)
        result.append({
            "range_semitones": label,
            "sample_count": int(np.sum(selected)),
            "mae_semitones": float(np.mean(np.abs(errors[selected]))) if np.any(selected) else None,
            "signed_error_semitones": float(np.mean(errors[selected])) if np.any(selected) else None,
        })
    return result


def _chroma_metrics(source_pitch: np.ndarray, decoded_pitch: np.ndarray, source_active: np.ndarray, decoded_active: np.ndarray) -> dict[str, float | None]:
    source = _bar_chroma(source_pitch, source_active)
    decoded = _bar_chroma(decoded_pitch, decoded_active)
    active = (source.sum(axis=1) > 0.0) & (decoded.sum(axis=1) > 0.0)
    if not np.any(active):
        return {"cosine_mean": None, "mse": None, "comparable_bar_count": 0}
    left, right = source[active], decoded[active]
    cosine = np.sum(left * right, axis=1) / np.maximum(np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1), 1.0e-8)
    return {"cosine_mean": float(np.mean(cosine)), "mse": float(np.mean(np.square(left - right))), "comparable_bar_count": int(np.sum(active))}


def _bar_chroma(pitches: np.ndarray, active: np.ndarray) -> np.ndarray:
    bars = np.zeros((pitches.shape[0], 12), dtype=np.float64)
    for row in range(pitches.shape[0]):
        for pitch in pitches[row][active[row]]:
            bars[row, int(np.rint(pitch)) % 12] += 1.0
    return bars / np.maximum(bars.sum(axis=1, keepdims=True), 1.0)


def _register_metrics(
    source_pitch: np.ndarray,
    decoded_pitch: np.ndarray,
    source_active: np.ndarray,
    decoded_active: np.ndarray,
    alignment: Sequence[Mapping[str, Any]],
) -> dict[str, float | int | None]:
    source_center = _bar_medians(source_pitch, source_active)
    decoded_center = _bar_medians(decoded_pitch, decoded_active)
    comparable = np.isfinite(source_center) & np.isfinite(decoded_center)
    absolute_mae = float(np.mean(np.abs(decoded_center[comparable] - source_center[comparable]))) if np.any(comparable) else None
    source_delta, decoded_delta = _adjacent_deltas(source_center, decoded_center, alignment)
    delta_mae = float(np.mean(np.abs(decoded_delta - source_delta))) if len(source_delta) else None
    return {
        "bar_center_mae_semitones": absolute_mae,
        "comparable_bar_count": int(np.sum(comparable)),
        "delta_mae_semitones": delta_mae,
        "comparable_transition_count": int(len(source_delta)),
    }


def _bar_medians(pitches: np.ndarray, active: np.ndarray) -> np.ndarray:
    return np.asarray([np.median(row[mask]) if np.any(mask) else np.nan for row, mask in zip(pitches, active)], dtype=float)


def _adjacent_deltas(source: np.ndarray, decoded: np.ndarray, alignment: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    grouped: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for row, item in enumerate(alignment):
        grouped[str(item["song_id"])].append((int(item["source_bar_index"]), row))
    source_deltas: list[float] = []
    decoded_deltas: list[float] = []
    for entries in grouped.values():
        ordered = sorted(entries)
        for (left_bar, left_row), (right_bar, right_row) in zip(ordered, ordered[1:]):
            if right_bar != left_bar + 1:
                continue
            values = (source[left_row], source[right_row], decoded[left_row], decoded[right_row])
            if not all(np.isfinite(value) for value in values):
                continue
            source_deltas.append(float(source[right_row] - source[left_row]))
            decoded_deltas.append(float(decoded[right_row] - decoded[left_row]))
    return np.asarray(source_deltas), np.asarray(decoded_deltas)


def _velocity_metrics(source: np.ndarray, decoded: np.ndarray, features: Mapping[str, Any], mask: np.ndarray) -> dict[str, float | None]:
    if not np.any(mask):
        return {"mae": None, "shared_active_slot_count": 0}
    index = int(features["velocity"])
    return {"mae": float(np.mean(np.abs(decoded[..., index][mask] - source[..., index][mask]))), "shared_active_slot_count": int(np.sum(mask))}


def _density_metrics(source_active: np.ndarray, decoded_active: np.ndarray) -> dict[str, float]:
    source_count = source_active.sum(axis=(1, 2))
    decoded_count = decoded_active.sum(axis=(1, 2))
    return {"bar_active_slot_mae": float(np.mean(np.abs(decoded_count - source_count))), "mean_signed_slot_difference": float(np.mean(decoded_count - source_count))}


def _texture_metrics(latent: np.ndarray, decoded: np.ndarray) -> dict[str, float | int | None]:
    if latent.shape[0] < 2:
        return {"nearest_latent_cosine_mean": None, "corresponding_decoded_cosine_mean": None, "pair_count": 0}
    latent_unit = _unit_rows(latent)
    similarity = latent_unit @ latent_unit.T
    np.fill_diagonal(similarity, -np.inf)
    neighbors = np.argmax(similarity, axis=1)
    decoded_unit = _unit_rows(decoded.reshape(decoded.shape[0], -1))
    decoded_similarity = np.sum(decoded_unit * decoded_unit[neighbors], axis=1)
    return {"nearest_latent_cosine_mean": float(np.mean(similarity[np.arange(len(neighbors)), neighbors])), "corresponding_decoded_cosine_mean": float(np.mean(decoded_similarity)), "pair_count": int(len(neighbors))}


def _feature_group_mse(source: np.ndarray, decoded: np.ndarray, features: Mapping[str, Any]) -> dict[str, float]:
    groups = {
        "relative_pitch": [features["relative_pitch"]],
        "state": list(features["state"]),
        "velocity": [features["velocity"]],
        "chroma_or_chord_embedding": list(features["chroma"]),
    }
    if features["density"] is not None:
        groups["density"] = [features["density"]]
    return {name: float(np.mean(np.square(decoded[..., indices] - source[..., indices]))) for name, indices in groups.items() if indices}


def _binary_f1(actual: np.ndarray, predicted: np.ndarray) -> float:
    true_positive = np.sum(actual & predicted)
    denominator = 2 * true_positive + np.sum(~actual & predicted) + np.sum(actual & ~predicted)
    return float(2 * true_positive / denominator) if denominator else 1.0


def _unit_rows(values: np.ndarray) -> np.ndarray:
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1.0e-8)


