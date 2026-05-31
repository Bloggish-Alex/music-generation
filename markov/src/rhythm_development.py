"""Rhythm motif memory, planning, variation, and scoring.

This layer gives the existing narrative/theme system a rhythmic surface.  It
does not force a fixed pattern.  It stores first-statement rhythm cells, plans
phrase-level rhythm targets, creates rhythm variants, and scores candidates for
recognizable-but-not-literal rhythmic development.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from hierarchical_types import NoteEvent


RhythmCell = Tuple[float, ...]


@dataclass(frozen=True)
class RhythmTarget:
    enabled: bool
    phrase_role: str
    section_role: str
    narrative_role: str
    identity_min: float
    identity_max: float
    density_scale: float
    syncopation_shift: float
    cadence_lengthen: float
    avoid_exact_repeat: bool
    source_cell: RhythmCell = ()


class RhythmMotifModel:
    """Extract and compare compact rhythm cells."""

    @staticmethod
    def cell(notes: Sequence[NoteEvent]) -> RhythmCell:
        melody = RhythmMotifModel._melody(notes)
        if not melody:
            return ()
        return tuple(round(max(0.125, n.duration_ql), 3) for n in melody)

    @staticmethod
    def onset_cell(notes: Sequence[NoteEvent]) -> RhythmCell:
        melody = RhythmMotifModel._melody(notes)
        return tuple(round(max(0.0, n.beat_offset), 3) for n in melody)

    @staticmethod
    def similarity(a: Sequence[float], b: Sequence[float]) -> float:
        if not a or not b:
            return 0.0
        count = min(len(a), len(b))
        dist = sum(abs(float(a[i]) - float(b[i])) for i in range(count))
        dist += abs(len(a) - len(b)) * 0.5
        return float(1.0 / (1.0 + dist))

    @staticmethod
    def normalized_density(notes: Sequence[NoteEvent], bar_length: float = 4.0) -> float:
        melody = RhythmMotifModel._melody(notes)
        return len(melody) / max(1.0, bar_length)

    @staticmethod
    def _melody(notes: Sequence[NoteEvent]) -> List[NoteEvent]:
        return sorted(
            [n for n in notes if n.voice == "melody"],
            key=lambda n: (n.beat_offset, n.pitch),
        )


class RhythmMemory:
    """Stores first-statement rhythm cells by theme label and local bar."""

    def __init__(self) -> None:
        self._cells: Dict[str, Dict[int, RhythmCell]] = {}

    def remember(self, label: str, local_bar: int, notes: Sequence[NoteEvent]) -> None:
        cell = RhythmMotifModel.cell(notes)
        if not cell:
            return
        self._cells.setdefault(label, {})[int(local_bar)] = cell

    def get(self, label: str, local_bar: int) -> RhythmCell:
        by_bar = self._cells.get(label, {})
        if not by_bar:
            return ()
        if local_bar in by_bar:
            return by_bar[local_bar]
        nearest = min(by_bar, key=lambda idx: abs(idx - local_bar))
        return by_bar[nearest]


class RhythmPhrasePlanner:
    """Map narrative/phrase roles to rhythm-development targets."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        cfg = config.get("rhythm_development", {})
        self.config = cfg if isinstance(cfg, Mapping) else {}
        self.enabled = bool(self.config.get("enabled", True))

    def plan(
        self,
        *,
        label: str,
        local_bar: int,
        section_role: str,
        phrase_role: str,
        narrative_role: str,
        tension: float,
        source_cell: RhythmCell,
    ) -> RhythmTarget:
        if not self.enabled:
            return RhythmTarget(False, phrase_role, section_role, narrative_role, 0.0, 1.0, 1.0, 0.0, 0.0, False, source_cell)

        repeat_identity = self._range("repeat_identity", (0.70, 0.90))
        development_identity = self._range("development_identity", (0.45, 0.78))
        cadence_identity = self._range("cadence_identity", (0.35, 0.75))
        if section_role == "REPEAT":
            identity_min, identity_max = repeat_identity
        elif phrase_role == "CADENCE" or narrative_role == "CODA":
            identity_min, identity_max = cadence_identity
        elif section_role in {"RETURN", "VARIANT"} or narrative_role in {"DEVELOPMENT", "CLIMAX"}:
            identity_min, identity_max = development_identity
        else:
            identity_min, identity_max = (0.50, 0.88)

        density_scale = 1.0
        if narrative_role in {"DEVELOPMENT", "CLIMAX"}:
            density_scale += float(self.config.get("development_density_lift", 0.18)) * float(tension)
        if phrase_role == "CADENCE":
            density_scale *= float(self.config.get("cadence_density_scale", 0.78))

        return RhythmTarget(
            enabled=True,
            phrase_role=phrase_role,
            section_role=section_role,
            narrative_role=narrative_role,
            identity_min=identity_min,
            identity_max=identity_max,
            density_scale=float(max(0.55, min(1.45, density_scale))),
            syncopation_shift=float(self.config.get("syncopation_shift", 0.12)) if narrative_role in {"DEVELOPMENT", "CLIMAX"} else 0.0,
            cadence_lengthen=float(self.config.get("cadence_lengthen", 0.45)) if phrase_role == "CADENCE" else 0.0,
            avoid_exact_repeat=bool(self.config.get("avoid_exact_repeat", True)),
            source_cell=source_cell,
        )

    def _range(self, key: str, default: Tuple[float, float]) -> Tuple[float, float]:
        value = self.config.get(key, default)
        if isinstance(value, Sequence) and len(value) >= 2:
            return float(value[0]), float(value[1])
        return default


class RhythmVariation:
    """Generate rhythm variants while preserving pitch order."""

    @staticmethod
    def apply(
        notes: Sequence[NoteEvent],
        target: RhythmTarget,
        rng: np.random.RandomState,
        bar_length: float = 4.0,
    ) -> List[NoteEvent]:
        if not target.enabled:
            return list(notes)
        melody = RhythmMotifModel._melody(notes)
        others = [n for n in notes if n.voice != "melody"]
        if not melody:
            return list(notes)

        durations = RhythmVariation._target_durations(melody, target, rng, bar_length)
        offsets = RhythmVariation._offsets_from_durations(durations, target, rng, bar_length)
        varied: List[NoteEvent] = []
        for note, dur, offset in zip(melody, durations, offsets):
            varied.append(NoteEvent(
                pitch=note.pitch,
                duration_ql=dur,
                velocity=note.velocity,
                beat_offset=offset,
                voice=note.voice,
            ))
        return sorted(varied + others, key=lambda n: (n.beat_offset, n.pitch))

    @staticmethod
    def _target_durations(
        melody: Sequence[NoteEvent],
        target: RhythmTarget,
        rng: np.random.RandomState,
        bar_length: float,
    ) -> List[float]:
        source = list(target.source_cell)
        current = [max(0.125, n.duration_ql) for n in melody]
        if source:
            desired_count = max(2, int(round(len(source) * target.density_scale)))
        else:
            desired_count = max(2, int(round(len(current) * target.density_scale)))
        desired_count = min(16, max(1, desired_count))

        if source and target.section_role == "REPEAT":
            pattern = RhythmVariation._resample_pattern(source, desired_count)
            keep = rng.uniform(target.identity_min, target.identity_max)
            base = RhythmVariation._resample_pattern(current, desired_count)
            durations = [
                pattern[i] if rng.rand() < keep else base[i]
                for i in range(desired_count)
            ]
        elif source and target.section_role in {"RETURN", "VARIANT"}:
            pattern = RhythmVariation._resample_pattern(source, desired_count)
            durations = RhythmVariation._develop_pattern(pattern, rng, target)
        else:
            durations = RhythmVariation._develop_pattern(
                RhythmVariation._resample_pattern(current, desired_count),
                rng,
                target,
            )

        if target.cadence_lengthen > 0 and durations:
            take = min(sum(durations[:-1]) * 0.18, target.cadence_lengthen)
            if len(durations) > 1:
                scale = max(0.65, (sum(durations[:-1]) - take) / max(0.125, sum(durations[:-1])))
                durations[:-1] = [max(0.125, d * scale) for d in durations[:-1]]
            durations[-1] = max(durations[-1], durations[-1] + take)

        return RhythmVariation._normalize(durations, bar_length)

    @staticmethod
    def _develop_pattern(pattern: List[float], rng: np.random.RandomState, target: RhythmTarget) -> List[float]:
        result = list(pattern)
        for i, dur in enumerate(result):
            if rng.rand() < 0.35:
                factor = float(rng.choice([0.75, 1.25, 1.5]))
                if target.narrative_role in {"DEVELOPMENT", "CLIMAX"} and rng.rand() < 0.5:
                    factor = float(rng.choice([0.5, 0.75]))
                result[i] = max(0.125, min(2.5, dur * factor))
        return result

    @staticmethod
    def _resample_pattern(pattern: Sequence[float], count: int) -> List[float]:
        if not pattern:
            return [1.0] * count
        if len(pattern) == count:
            return [float(x) for x in pattern]
        xs = np.linspace(0, len(pattern) - 1, count)
        return [float(pattern[int(round(x))]) for x in xs]

    @staticmethod
    def _normalize(durations: Sequence[float], bar_length: float) -> List[float]:
        total = sum(max(0.125, d) for d in durations)
        if total <= 0:
            return [bar_length]
        scale = bar_length / total
        result = [max(0.125, round(d * scale, 3)) for d in durations]
        drift = bar_length - sum(result)
        if result:
            result[-1] = max(0.125, round(result[-1] + drift, 3))
        return result

    @staticmethod
    def _offsets_from_durations(
        durations: Sequence[float],
        target: RhythmTarget,
        rng: np.random.RandomState,
        bar_length: float,
    ) -> List[float]:
        offsets: List[float] = []
        beat = 0.0
        for i, dur in enumerate(durations):
            shift = 0.0
            if i > 0 and i < len(durations) - 1 and target.syncopation_shift > 0 and rng.rand() < 0.25:
                shift = float(rng.choice([-1.0, 1.0])) * target.syncopation_shift
            offsets.append(round(max(0.0, min(bar_length - 0.125, beat + shift)), 3))
            beat += dur
        return offsets


class RhythmCandidateScorer:
    """Soft rhythm score for candidate selection diagnostics."""

    def __init__(self, config: Mapping[str, Any], learned_prior: Optional[Any] = None) -> None:
        cfg = config.get("rhythm_development", {})
        self.config = cfg if isinstance(cfg, Mapping) else {}
        self.global_config = config
        self.learned_prior = learned_prior

    def score(
        self,
        notes: Sequence[NoteEvent],
        target: RhythmTarget,
        *,
        previous_cells: Sequence[RhythmCell] = (),
    ) -> Dict[str, float]:
        if not target.enabled:
            return {"active": 0.0, "score": 0.0}
        cell = RhythmMotifModel.cell(notes)
        source_similarity = RhythmMotifModel.similarity(cell, target.source_cell)
        exact_repeat_cost = self._exact_repeat_cost(cell, previous_cells)
        contour_score = self._phrase_contour_score(cell, target)
        prior_score = self._learned_prior_score(notes, target, previous_cells)

        score = 0.0
        if target.source_cell:
            if source_similarity < target.identity_min:
                score -= (target.identity_min - source_similarity) * float(self.config.get("identity_weight", 1.0))
            elif source_similarity > target.identity_max:
                score -= (source_similarity - target.identity_max) * float(self.config.get("copy_weight", 1.1))
            else:
                score += 0.25
        score += contour_score * float(self.config.get("phrase_contour_weight", 0.45))
        score -= exact_repeat_cost * float(self.config.get("consecutive_repeat_weight", 0.75))
        score += float(prior_score.get("weighted", 0.0))
        return {
            "active": 1.0,
            "score": float(score),
            "source_similarity": float(source_similarity),
            "exact_repeat_cost": float(exact_repeat_cost),
            "phrase_contour_score": float(contour_score),
            "note_count": float(len(cell)),
            "learned_prior_probability": float(prior_score.get("probability", 0.5)),
            "learned_prior_logit": float(prior_score.get("logit", 0.0)),
            "learned_prior_weighted": float(prior_score.get("weighted", 0.0)),
            "learned_prior_available": float(prior_score.get("model_available", 0.0)),
        }

    def _learned_prior_score(
        self,
        notes: Sequence[NoteEvent],
        target: RhythmTarget,
        previous_cells: Sequence[RhythmCell],
    ) -> Dict[str, float]:
        if self.learned_prior is None:
            return {
                "probability": 0.5,
                "logit": 0.0,
                "weighted": 0.0,
                "model_available": 0.0,
            }
        score = self.learned_prior.score_candidate(
            notes,
            target,
            self.global_config,
            previous_cells=previous_cells,
        )
        return {
            "probability": float(getattr(score, "probability", 0.5)),
            "logit": float(getattr(score, "logit", 0.0)),
            "weighted": float(getattr(score, "weighted", 0.0)),
            "model_available": 1.0 if bool(getattr(score, "model_available", False)) else 0.0,
        }

    @staticmethod
    def _exact_repeat_cost(cell: RhythmCell, previous_cells: Sequence[RhythmCell]) -> float:
        if not cell or not previous_cells:
            return 0.0
        cost = 0.0
        for prev in previous_cells[-3:]:
            if cell == prev:
                cost += 1.0
            elif RhythmMotifModel.similarity(cell, prev) > 0.92:
                cost += 0.5
        return cost

    @staticmethod
    def _phrase_contour_score(cell: RhythmCell, target: RhythmTarget) -> float:
        if not cell:
            return 0.0
        first = float(cell[0])
        last = float(cell[-1])
        if target.phrase_role == "CADENCE":
            return 1.0 if last >= max(first, 0.75) else -0.5
        if target.narrative_role in {"DEVELOPMENT", "CLIMAX"}:
            short_ratio = sum(1 for d in cell if d <= 0.5) / max(1, len(cell))
            return min(1.0, short_ratio * 1.5)
        return 0.25
