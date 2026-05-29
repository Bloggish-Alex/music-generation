"""Learned prior for rhythm-candidate selection.

The prior is trained from real corpus bars versus rhythm-only perturbations.
It does not generate notes and does not repair individual bars.  During
generation it gives the rhythm scorer a data-informed naturalness signal, so
rhythm selection can move away from accumulating hand-written local rules.
"""

from __future__ import annotations

import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from hierarchical_types import NoteEvent
from rhythm_development import RhythmCell, RhythmMotifModel, RhythmTarget


FeatureDict = Dict[str, float | str | int]


@dataclass(frozen=True)
class RhythmPriorScore:
    probability: float
    logit: float
    weighted: float
    enabled: bool
    model_available: bool


class RhythmCandidatePrior:
    """Lightweight learned naturalness prior for rhythm cells."""

    version = 1

    def __init__(
        self,
        pipeline: Any = None,
        feature_names: Optional[List[str]] = None,
        training_summary: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.pipeline = pipeline
        self.feature_names = feature_names or []
        self.training_summary = training_summary or {}

    @property
    def available(self) -> bool:
        return self.pipeline is not None

    @classmethod
    def fit(
        cls,
        file_map: Mapping[str, Sequence[Any]],
        *,
        negative_per_positive: int = 4,
        seed: int = 42,
        max_samples: Optional[int] = None,
    ) -> "RhythmCandidatePrior":
        """Train from real bar rhythms and synthetic rhythm failures."""
        try:
            from sklearn.feature_extraction import DictVectorizer
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
        except Exception:
            return cls(training_summary={
                "trained": False,
                "version": cls.version,
                "reason": "scikit-learn unavailable",
            })

        rng = np.random.RandomState(seed)
        items: List[Tuple[str, int, int, List[NoteEvent], RhythmCell, List[RhythmCell]]] = []
        for filepath, vectors in file_map.items():
            source_cell: RhythmCell = ()
            recent_cells: List[RhythmCell] = []
            for index, vector in enumerate(vectors):
                notes = cls._notes_from_vector(vector)
                if len(notes) < 2:
                    continue
                current_cell = RhythmMotifModel.cell(notes)
                if not source_cell:
                    source_cell = current_cell
                items.append((filepath, index, len(vectors), notes, source_cell, list(recent_cells[-3:])))
                if current_cell:
                    recent_cells.append(current_cell)

        if max_samples and max_samples > 0 and len(items) > max_samples:
            chosen = rng.choice(len(items), size=max_samples, replace=False)
            items = [items[int(i)] for i in chosen]

        features: List[FeatureDict] = []
        labels: List[int] = []
        weights: List[float] = []
        negative_type_counts: Dict[str, int] = {}
        role_counts: Dict[str, int] = {}

        for _filepath, index, total, notes, source_cell, previous_cells in items:
            role = cls._role_from_position(index, total)
            target = cls._training_target(role, source_cell)
            features.append(cls.extract_features(notes, target=target, previous_cells=previous_cells))
            labels.append(1)
            weights.append(1.0)
            role_counts[role] = role_counts.get(role, 0) + 1

            for variant in range(max(1, negative_per_positive)):
                negative_notes, negative_type = cls._perturb_rhythm(notes, rng, variant)
                features.append(cls.extract_features(negative_notes, target=target, previous_cells=previous_cells))
                labels.append(0)
                weights.append(cls._negative_weight(negative_type, role))
                negative_type_counts[negative_type] = negative_type_counts.get(negative_type, 0) + 1

        if len(set(labels)) < 2 or len(labels) < 20:
            return cls(training_summary={
                "trained": False,
                "version": cls.version,
                "reason": "not enough rhythm samples",
                "samples": len(labels),
            })

        pipeline = Pipeline(steps=[
            ("vectorizer", DictVectorizer(sparse=True)),
            ("scaler", StandardScaler(with_mean=False)),
            ("classifier", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=seed,
            )),
        ])
        pipeline.fit(features, labels, classifier__sample_weight=weights)
        vectorizer = pipeline.named_steps["vectorizer"]
        return cls(
            pipeline=pipeline,
            feature_names=list(vectorizer.get_feature_names_out()),
            training_summary={
                "trained": True,
                "version": cls.version,
                "positive_samples": len(items),
                "negative_samples": len(labels) - len(items),
                "negative_per_positive": negative_per_positive,
                "feature_count": len(vectorizer.get_feature_names_out()),
                "role_counts": role_counts,
                "negative_type_counts": negative_type_counts,
                "max_samples": max_samples,
            },
        )

    def score_candidate(
        self,
        notes: Sequence[NoteEvent],
        target: RhythmTarget,
        config: Mapping[str, Any],
        *,
        previous_cells: Sequence[RhythmCell] = (),
    ) -> RhythmPriorScore:
        cfg = self._config(config)
        enabled = bool(cfg.get("enabled", True)) and bool(cfg.get("learned_prior_enabled", True))
        weight = float(cfg.get("learned_prior_weight", 0.0))
        if not enabled or not self.available or weight == 0.0:
            return RhythmPriorScore(0.5, 0.0, 0.0, enabled, self.available)

        features = self.extract_features(notes, target=target, previous_cells=previous_cells)
        probability = float(self.pipeline.predict_proba([features])[0][1])
        eps = 1e-5
        probability = max(eps, min(1.0 - eps, probability))
        logit = math.log(probability / (1.0 - probability))
        clip = float(cfg.get("learned_prior_logit_clip", 2.0))
        logit = max(-clip, min(clip, logit))
        return RhythmPriorScore(
            probability=probability,
            logit=logit,
            weighted=weight * logit,
            enabled=True,
            model_available=True,
        )

    @classmethod
    def extract_features(
        cls,
        notes: Sequence[NoteEvent],
        *,
        target: RhythmTarget,
        previous_cells: Sequence[RhythmCell] = (),
    ) -> FeatureDict:
        melody = cls._melody(notes)
        durations = [float(n.duration_ql) for n in melody]
        offsets = [float(n.beat_offset) for n in melody]
        cell = RhythmMotifModel.cell(melody)
        source_similarity = RhythmMotifModel.similarity(cell, target.source_cell)
        previous_similarity = max(
            [RhythmMotifModel.similarity(cell, prev) for prev in previous_cells[-3:]],
            default=0.0,
        )
        bar_length = cls._bar_length(melody)
        intervals = [b - a for a, b in zip(offsets, offsets[1:])]
        final_ratio = durations[-1] / max(0.25, bar_length) if durations else 0.0
        short_ratio = cls._ratio(d <= 0.5 for d in durations)
        long_ratio = cls._ratio(d >= 1.25 for d in durations)
        equal_ratio = cls._equal_duration_ratio(durations)

        return {
            "version": cls.version,
            "phrase_role": target.phrase_role,
            "section_role": target.section_role,
            "narrative_role": target.narrative_role,
            "note_count": len(durations),
            "density": len(durations) / max(1.0, bar_length),
            "mean_duration": cls._mean(durations),
            "duration_std": cls._std(durations),
            "duration_entropy": cls._entropy(durations),
            "onset_entropy": cls._entropy([round(o % 1.0, 3) for o in offsets]),
            "short_note_ratio": short_ratio,
            "long_note_ratio": long_ratio,
            "equal_duration_ratio": equal_ratio,
            "final_duration_ratio": final_ratio,
            "final_is_long": 1 if final_ratio >= 0.25 else 0,
            "syncopated_onset_ratio": cls._ratio(abs(o % 1.0) > 1e-6 for o in offsets),
            "mean_ioi": cls._mean(intervals),
            "ioi_std": cls._std(intervals),
            "source_similarity": source_similarity,
            "source_similarity_below_min": max(0.0, target.identity_min - source_similarity),
            "source_similarity_above_max": max(0.0, source_similarity - target.identity_max),
            "previous_similarity": previous_similarity,
            "target_identity_width": target.identity_max - target.identity_min,
            "target_density_scale": target.density_scale,
            "target_syncopation_shift": target.syncopation_shift,
            "target_cadence_lengthen": target.cadence_lengthen,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "feature_names": self.feature_names,
            "training_summary": self.training_summary,
        }

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "rhythm_candidate_prior.pkl", "wb") as f:
            pickle.dump(self.pipeline, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "RhythmCandidatePrior":
        with open(path / "rhythm_candidate_prior.pkl", "rb") as f:
            pipeline = pickle.load(f)
        metadata_path = path / "metadata.json"
        metadata: Dict[str, Any] = {}
        if metadata_path.exists():
            with open(metadata_path, encoding="utf-8") as f:
                metadata = json.load(f)
        return cls(
            pipeline=pipeline,
            feature_names=list(metadata.get("feature_names", [])),
            training_summary=dict(metadata.get("training_summary", {})),
        )

    @staticmethod
    def _config(config: Mapping[str, Any]) -> Mapping[str, Any]:
        cfg = config.get("rhythm_development", {})
        return cfg if isinstance(cfg, Mapping) else {}

    @staticmethod
    def _notes_from_vector(vector: Any) -> List[NoteEvent]:
        events = getattr(vector, "melody_events", None)
        notes: List[NoteEvent] = []
        if not isinstance(events, list):
            return notes
        for event in events:
            if not isinstance(event, Mapping):
                continue
            pitch = int(event.get("pitch", -1))
            if pitch < 0:
                continue
            notes.append(NoteEvent(
                pitch=pitch,
                duration_ql=float(event.get("duration", event.get("quarterLength", 0.25))),
                velocity=int(event.get("velocity", 80)),
                beat_offset=float(event.get("onset", event.get("onset_in_measure", 0.0))),
                voice="melody",
            ))
        return sorted(notes, key=lambda n: (n.beat_offset, n.pitch))

    @staticmethod
    def _perturb_rhythm(
        notes: Sequence[NoteEvent],
        rng: np.random.RandomState,
        variant: int,
    ) -> Tuple[List[NoteEvent], str]:
        melody = RhythmCandidatePrior._melody(notes)
        if not melody:
            return list(notes), "empty"
        mode = variant % 4
        if mode == 0:
            return RhythmCandidatePrior._equal_run_long_tail(melody), "equal_run_long_tail"
        if mode == 1:
            return RhythmCandidatePrior._collapse_density(melody), "density_collapse"
        if mode == 2:
            durations = [n.duration_ql for n in melody]
            rng.shuffle(durations)
            return RhythmCandidatePrior._with_durations(melody, durations), "duration_shuffle"
        durations = [
            max(0.125, n.duration_ql * float(rng.choice([0.5, 0.5, 1.5, 2.0])))
            for n in melody
        ]
        return RhythmCandidatePrior._with_durations(melody, durations), "duration_noise"

    @staticmethod
    def _equal_run_long_tail(notes: Sequence[NoteEvent]) -> List[NoteEvent]:
        count = max(2, len(notes))
        head_count = max(1, count - 1)
        head = min(0.5, 3.0 / head_count)
        durations = [head] * head_count + [max(0.5, 4.0 - head * head_count)]
        return RhythmCandidatePrior._with_durations(notes, durations[:count])

    @staticmethod
    def _collapse_density(notes: Sequence[NoteEvent]) -> List[NoteEvent]:
        kept = [notes[0]]
        if len(notes) > 2:
            kept.append(notes[len(notes) // 2])
        if len(notes) > 1:
            kept.append(notes[-1])
        durations = [4.0 / len(kept)] * len(kept)
        return RhythmCandidatePrior._with_durations(kept, durations)

    @staticmethod
    def _with_durations(notes: Sequence[NoteEvent], durations: Sequence[float]) -> List[NoteEvent]:
        if not notes:
            return []
        durations = list(durations) or [4.0]
        if len(durations) < len(notes):
            durations.extend([durations[-1]] * (len(notes) - len(durations)))
        total = sum(max(0.125, d) for d in durations[:len(notes)])
        scale = 4.0 / max(0.125, total)
        result: List[NoteEvent] = []
        offset = 0.0
        for note, dur in zip(notes, durations):
            d = round(max(0.125, float(dur) * scale), 3)
            result.append(NoteEvent(
                pitch=note.pitch,
                duration_ql=d,
                velocity=note.velocity,
                beat_offset=round(offset, 3),
                voice=note.voice,
            ))
            offset += d
        drift = 4.0 - sum(n.duration_ql for n in result)
        if result:
            last = result[-1]
            result[-1] = NoteEvent(
                pitch=last.pitch,
                duration_ql=round(max(0.125, last.duration_ql + drift), 3),
                velocity=last.velocity,
                beat_offset=last.beat_offset,
                voice=last.voice,
            )
        return result

    @staticmethod
    def _training_target(role: str, source_cell: RhythmCell) -> RhythmTarget:
        if role == "CADENCE":
            identity = (0.35, 0.75)
            lengthen = 0.35
        elif role == "DEVELOPMENT":
            identity = (0.45, 0.78)
            lengthen = 0.0
        else:
            identity = (0.60, 0.90)
            lengthen = 0.0
        return RhythmTarget(
            enabled=True,
            phrase_role=role,
            section_role="TRAINING",
            narrative_role=role,
            identity_min=identity[0],
            identity_max=identity[1],
            density_scale=1.0,
            syncopation_shift=0.0,
            cadence_lengthen=lengthen,
            avoid_exact_repeat=True,
            source_cell=source_cell,
        )

    @staticmethod
    def _role_from_position(index: int, total: int) -> str:
        if total <= 0:
            return "CONTINUATION"
        frac = (index + 1) / total
        if frac >= 0.88:
            return "CADENCE"
        if frac >= 0.45:
            return "DEVELOPMENT"
        if index <= 1:
            return "OPENING"
        return "CONTINUATION"

    @staticmethod
    def _negative_weight(negative_type: str, role: str) -> float:
        base = {
            "equal_run_long_tail": 1.35,
            "density_collapse": 1.20,
            "duration_shuffle": 1.00,
            "duration_noise": 1.00,
        }.get(negative_type, 1.0)
        if role == "CADENCE" and negative_type == "equal_run_long_tail":
            base *= 0.85
        return float(base)

    @staticmethod
    def _melody(notes: Sequence[NoteEvent]) -> List[NoteEvent]:
        return sorted(
            [n for n in notes if n.pitch >= 0 and n.voice == "melody"],
            key=lambda n: (n.beat_offset, n.pitch),
        )

    @staticmethod
    def _bar_length(notes: Sequence[NoteEvent]) -> float:
        if not notes:
            return 4.0
        return max(4.0, max(n.beat_offset + n.duration_ql for n in notes))

    @staticmethod
    def _mean(values: Sequence[float | int]) -> float:
        return float(np.mean(values)) if values else 0.0

    @staticmethod
    def _std(values: Sequence[float | int]) -> float:
        return float(np.std(values)) if values else 0.0

    @staticmethod
    def _ratio(values: Iterable[bool]) -> float:
        data = list(values)
        return sum(1 for x in data if x) / max(1, len(data))

    @staticmethod
    def _entropy(values: Sequence[float]) -> float:
        if not values:
            return 0.0
        counts: Dict[float, int] = {}
        for value in values:
            key = round(float(value), 3)
            counts[key] = counts.get(key, 0) + 1
        total = sum(counts.values())
        probs = [count / total for count in counts.values()]
        return float(-sum(p * math.log(p + 1e-12) for p in probs))

    @staticmethod
    def _equal_duration_ratio(durations: Sequence[float]) -> float:
        if not durations:
            return 0.0
        counts: Dict[float, int] = {}
        for duration in durations:
            key = round(float(duration), 2)
            counts[key] = counts.get(key, 0) + 1
        return max(counts.values()) / max(1, len(durations))
