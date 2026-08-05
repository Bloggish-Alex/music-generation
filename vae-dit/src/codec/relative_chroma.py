#!/usr/bin/env python3
"""Shared relative chroma utilities for bar-level encoding and diagnostics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def note_pitch(note: Any) -> int:
    """Return a MIDI pitch from a note object or serialized note mapping."""
    if isinstance(note, Mapping):
        return int(note["pitch"])
    return int(getattr(note, "pitch"))


def note_duration_ql(note: Any) -> float:
    """Return note duration in quarter lengths from object or mapping notes."""
    if isinstance(note, Mapping):
        return float(note.get("duration_ql", 0.0))
    return float(getattr(note, "duration_ql", 0.0))


def note_velocity(note: Any, default: float = 0.0) -> float:
    """Return MIDI velocity from object or mapping notes."""
    if isinstance(note, Mapping):
        return float(note.get("velocity", default))
    return float(getattr(note, "velocity", default))


def normalized_velocity(velocity: float, velocity_scale: float = 127.0) -> float:
    """Normalize MIDI velocity into [0, 1]."""
    scale = max(float(velocity_scale), 1.0e-8)
    return float(max(0.0, min(float(velocity), scale))) / scale


def bass_anchor_pitch(notes: Sequence[Any]) -> int | None:
    """Choose the physical lowest pitch as the bar-local harmonic anchor."""
    if not notes:
        return None
    return min(note_pitch(note) for note in notes)


def relative_chromagram(
    notes: Sequence[Any],
    base_pitch: int | None = None,
    *,
    velocity_scale: float = 127.0,
) -> np.ndarray:
    """Compute duration-and-velocity-weighted 12-bin chroma relative to bass anchor."""
    values = np.zeros(12, dtype=np.float32)
    if not notes:
        return values
    if base_pitch is None:
        base_pitch = bass_anchor_pitch(notes)
    if base_pitch is None:
        return values

    base_pc = int(base_pitch) % 12
    for note in notes:
        interval = (note_pitch(note) - base_pc) % 12
        duration = max(0.0, note_duration_ql(note))
        velocity = normalized_velocity(note_velocity(note), velocity_scale)
        values[interval] += duration * velocity

    total = float(values.sum())
    if total <= 0.0:
        return values
    return values / total
