#!/usr/bin/env python3
"""
Note Transform — atomic variation operators for section returns.

Implements the derivative layer of the Grundgestalt + developing variation
framework (Almada 2023 / Schoenberg).  Each transform is a pure function
on ``List[NoteEvent]`` producing a recognizably related variant.

Usage::

    from note_transform import select_transforms, apply_variation

    profile = select_transforms(centroids, cluster_labels, strength=0.3)
    varied_notes = apply_variation(original_notes, profile)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from hierarchical_generator import NoteEvent

TRANSFORM_REGISTRY: Dict[str, Callable] = {}


def _NoteEvent(pitch, duration_ql, velocity, beat_offset):
    """Lazy constructor to avoid circular import."""
    from hierarchical_generator import NoteEvent
    return NoteEvent(
        pitch=pitch, duration_ql=duration_ql,
        velocity=velocity, beat_offset=beat_offset,
    )


def _register(name: str):
    def decorator(fn):
        TRANSFORM_REGISTRY[name] = fn
        return fn
    return decorator


# ---------------------------------------------------------------------------
# Atomic transforms
# ---------------------------------------------------------------------------


def _clamp_pitch(p: int) -> int:
    return max(21, min(108, p))


@_register("shift_register")
def shift_register(
    notes: List[NoteEvent],
    strength: float,
    rng: Optional[np.random.RandomState] = None,
) -> List[NoteEvent]:
    """Shift all pitches by strength × 12 semitones (up or down)."""
    if rng is None:
        rng = np.random.RandomState()
    semitones = int(round(strength * 12))
    if semitones == 0:
        semitones = 12 if rng.random() < 0.5 else -12
    result = []
    for n in notes:
        if n.pitch >= 0:
            result.append(_NoteEvent(
                pitch=_clamp_pitch(n.pitch + semitones),
                duration_ql=n.duration_ql,
                velocity=n.velocity,
                beat_offset=n.beat_offset,
            ))
        else:
            result.append(n)
    return result


@_register("scale_duration")
def scale_duration(
    notes: List[NoteEvent],
    strength: float,
    rng: Optional[np.random.RandomState] = None,
) -> List[NoteEvent]:
    """Scale durations: diminution (faster) when note_density is high,
    augmentation (slower) otherwise.  strength controls the scale factor."""
    if rng is None:
        rng = np.random.RandomState()
    # strength ∈ [0, 0.6]: 0 = no change, 0.6 = 40% faster or 60% slower
    # Diminution for dense textures (factor < 1), augmentation for sparse (factor > 1)
    if rng.random() < 0.6:
        factor = 1.0 - strength * 0.5   # diminution: 1.0 → 0.7
    else:
        factor = 1.0 + strength * 0.8    # augmentation: 1.0 → 1.48
    factor = max(0.4, min(2.5, factor))
    result = []
    for n in notes:
        new_dur = max(0.25, n.duration_ql * factor)
        result.append(_NoteEvent(
            pitch=n.pitch,
            duration_ql=new_dur,
            velocity=n.velocity,
            beat_offset=n.beat_offset,
        ))
    return result


@_register("ornament")
def ornament(
    notes: List[NoteEvent],
    strength: float,
    rng: Optional[np.random.RandomState] = None,
) -> List[NoteEvent]:
    """Insert passing tones at pitch leaps ≥ 3 semitones.  Probability = strength."""
    if rng is None:
        rng = np.random.RandomState()
    result: List[NoteEvent] = []
    prev_pitch: Optional[int] = None
    for n in notes:
        if n.pitch >= 0 and prev_pitch is not None and prev_pitch >= 0:
            leap = abs(n.pitch - prev_pitch)
            if leap >= 3 and rng.random() < strength:
                # Insert a passing tone halfway between
                mid_pitch = (prev_pitch + n.pitch) // 2
                mid_dur = min(n.duration_ql * 0.3, 0.25)
                result.append(_NoteEvent(
                    pitch=mid_pitch,
                    duration_ql=mid_dur,
                    velocity=max(20, n.velocity - 20),
                    beat_offset=n.beat_offset - mid_dur,
                ))
                # Shorten the target note slightly
                n = _NoteEvent(
                    pitch=n.pitch,
                    duration_ql=max(0.25, n.duration_ql - mid_dur),
                    velocity=n.velocity,
                    beat_offset=n.beat_offset,
                )
        if n.pitch >= 0:
            prev_pitch = n.pitch
        result.append(n)
    return result


@_register("invert_contour")
def invert_contour(
    notes: List[NoteEvent],
    strength: float,
    rng: Optional[np.random.RandomState] = None,
) -> List[NoteEvent]:
    """Flip pitch direction: intervals are reversed with probability = strength."""
    if rng is None:
        rng = np.random.RandomState()
    sounding = [n for n in notes if n.pitch >= 0]
    if len(sounding) < 2:
        return list(notes)

    # Build intervals
    intervals = [
        sounding[i + 1].pitch - sounding[i].pitch
        for i in range(len(sounding) - 1)
    ]
    # Invert: flip sign
    inverted = [-d for d in intervals]

    # Blend: with probability = strength, use inverted interval
    blended = []
    for i, d in enumerate(intervals):
        if rng.random() < strength:
            blended.append(inverted[i])
        else:
            blended.append(d)

    # Reconstruct pitches
    new_pitches = [sounding[0].pitch]
    for d in blended:
        new_pitches.append(_clamp_pitch(new_pitches[-1] + d))

    # Apply to original notes
    result = []
    si = 0
    for n in notes:
        if n.pitch >= 0 and si < len(new_pitches):
            result.append(_NoteEvent(
                pitch=new_pitches[si],
                duration_ql=n.duration_ql,
                velocity=n.velocity,
                beat_offset=n.beat_offset,
            ))
            si += 1
        else:
            result.append(n)
    return result


@_register("deviation")
def deviation(
    notes: List[NoteEvent],
    strength: float,
    rng: Optional[np.random.RandomState] = None,
) -> List[NoteEvent]:
    """Small random jitter on pitch and duration — controlled 'humanization'."""
    if rng is None:
        rng = np.random.RandomState()
    result = []
    for n in notes:
        if n.pitch >= 0:
            shift = int(round(rng.normal(0, strength * 2.5)))
            dur = n.duration_ql + rng.normal(0, strength * 0.15)
            dur = max(0.25, dur)
            result.append(_NoteEvent(
                pitch=_clamp_pitch(n.pitch + shift),
                duration_ql=dur,
                velocity=max(30, min(127, int(n.velocity + rng.normal(0, 3)))),
                beat_offset=n.beat_offset,
            ))
        else:
            result.append(n)
    return result


# ---------------------------------------------------------------------------
# Transform selection
# ---------------------------------------------------------------------------


def select_transforms(
    centroids: np.ndarray,        # (n_clusters, 8)
    cluster_labels: List[int],    # labels for this section's measures
    strength: float,
    rng: Optional[np.random.RandomState] = None,
) -> List[Tuple[str, float]]:
    """Select 1–2 transforms weighted by the section's cluster centroids.

    Args:
        centroids: KMeans centroids, shape (n_clusters, 8).
        cluster_labels: Cluster labels for the measures in this section.
        strength: Variation strength s ∈ [0, 1].
        rng: Random state.

    Returns:
        List of (transform_name, strength) pairs.
    """
    if rng is None:
        rng = np.random.RandomState()
    if strength <= 0:
        return []

    # Average centroid across this section's measures
    indices = [c for c in cluster_labels if 0 <= c < len(centroids)]
    if not indices:
        return []
    avg_centroid = centroids[indices].mean(axis=0)

    note_density = float(avg_centroid[0])
    mean_dur = float(avg_centroid[1])
    dur_var = float(avg_centroid[2])
    silence_ratio = float(avg_centroid[4])
    syncopation = float(avg_centroid[6])
    entropy = float(avg_centroid[7])

    # Transform selection weights derived from centroid features
    weights = {
        "shift_register":  max(0.1, 1.0 - silence_ratio),
        "scale_duration":  max(0.1, note_density / 8.0),
        "ornament":        max(0.1, entropy / 2.0),
        "invert_contour":  max(0.05, syncopation),
        "deviation":       max(0.1, dur_var / 0.5),
    }

    # Normalize to probabilities
    names = list(weights.keys())
    probs = np.array([weights[n] for n in names], dtype=np.float64)
    probs /= probs.sum()

    # Sample 1–2 transforms (more at higher strength)
    n_transforms = 1 if rng.random() < (1.0 - strength) else 2
    n_transforms = min(n_transforms, len(names))

    chosen_idx = rng.choice(len(names), size=n_transforms, replace=False, p=probs)
    return [(names[i], strength) for i in chosen_idx]


# ---------------------------------------------------------------------------
# Apply variation profile
# ---------------------------------------------------------------------------


def apply_variation(
    notes: List[NoteEvent],
    profile: List[Tuple[str, float]],
    rng: Optional[np.random.RandomState] = None,
) -> List[NoteEvent]:
    """Apply a sequence of transforms to notes, preserving measure duration.

    After transforms, note durations are normalized so the total equals
    the original total duration.  This preserves cadence gaps and breathing
    space that the note sampler created, while keeping the measure grid
    intact regardless of which transforms are applied.

    Args:
        notes: Original NoteEvent list for a measure.
        profile: List of (transform_name, strength) pairs.
        rng: Random state passed to transforms.

    Returns:
        Transformed notes, with durations normalized to the original total.
    """
    if rng is None:
        rng = np.random.RandomState()
    if not profile:
        return list(notes)

    # Remember original total to preserve cadence gaps
    original_total = sum(n.duration_ql for n in notes)

    result = list(notes)
    for name, strength in profile:
        if name in TRANSFORM_REGISTRY:
            result = TRANSFORM_REGISTRY[name](result, strength, rng)

    # Normalize to the original total duration
    total_dur = sum(n.duration_ql for n in result)
    if total_dur <= 0 or abs(total_dur - original_total) < 0.01:
        return result

    scale = original_total / total_dur
    normalized = []
    for n in result:
        normalized.append(_NoteEvent(
            pitch=n.pitch,
            duration_ql=max(0.1, n.duration_ql * scale),
            velocity=n.velocity,
            beat_offset=n.beat_offset * scale if n.beat_offset > 0 else 0.0,
        ))
    return normalized
