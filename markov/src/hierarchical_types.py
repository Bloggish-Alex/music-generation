#!/usr/bin/env python3
"""Shared data structures for hierarchical generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class NoteEvent:
    """One note or rest within a measure."""

    pitch: int
    duration_ql: float
    velocity: int
    beat_offset: float
    voice: str = "melody"


@dataclass(frozen=True)
class SectionAffect:
    """Tonal-emotional identity for one section occurrence."""

    label: str
    role: str
    occurrence_id: int
    tonal_pc: int
    register_center: int
    intensity: float
    tension: float


@dataclass(frozen=True)
class CompositionPlan:
    """Global coherence plan shared by section, phrase, and note layers."""

    global_tonic_pc: int
    section_affects: Dict[int, SectionAffect]
    measure_targets: Dict[int, int]
    measure_affects: Dict[int, Dict[str, Any]]
    development_roles: Dict[int, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ThemeIdentity:
    """Learned identity of a section family from its first appearance."""

    label: str
    bar_intervals: Tuple[Tuple[int, ...], ...]
    bar_durations: Tuple[Tuple[float, ...], ...]
    bar_sizes: Tuple[int, ...]
    bar_mean_offsets: Tuple[float, ...]
    opening_intervals: Tuple[int, ...]
    opening_durations: Tuple[float, ...]
    cadence_intervals: Tuple[int, ...]
    cadence_durations: Tuple[float, ...]
    cadence_pc: int
    opening_size: int
    cadence_size: int


@dataclass(frozen=True)
class StructureEdge:
    """Long-range relationship between two bars in the structure graph."""

    source_bar: int
    target_bar: int
    relation: str
    label: str


@dataclass(frozen=True)
class BarSkeleton:
    """Theme-level bar skeleton independent of surface ornamentation."""

    rhythm_cell: Tuple[float, ...]
    contour: Tuple[int, ...]
    anchor_degree: int
    cadence_degree: int
    register_zone: float
    tension: float
    note_count: int


@dataclass(frozen=True)
class ThemeSkeleton:
    """A section family's reusable skeleton."""

    label: str
    bars: Tuple[BarSkeleton, ...]


@dataclass(frozen=True)
class BarGenerationTarget:
    """Skeleton constraint for rendering one bar."""

    relation: str
    source_bar: Optional[int]
    development_role: str
    rhythm_cell: Tuple[float, ...]
    contour: Tuple[int, ...]
    target_pitch: int
    target_degree: int
    register_target: float
    cadence_strength: float
    tension: float
    exact_copy_penalty: float
    similarity_min: float = 0.0
    similarity_max: float = 1.0
    development_strength: float = 1.0
    target_attraction: float = 0.55
    rhythm_change_scale: float = 1.0
    harmony: Optional[Dict[str, Any]] = None
    dual_theme: Optional[Dict[str, Any]] = None
