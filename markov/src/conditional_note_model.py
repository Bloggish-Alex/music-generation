#!/usr/bin/env python3
"""Learned conditional note priors for harmony-aware candidate selection.

The model is intentionally probabilistic and non-destructive: it does not
rewrite notes.  It learns how training melodies distribute pitch classes
relative to the estimated bar harmony, then scores generated candidates under
the same conditions.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from harmonic_planner import HarmonicPlanner
from hierarchical_types import BarGenerationTarget, NoteEvent
from learned_harmony import LearnedHarmonicModel


_RELATIONS: Tuple[str, ...] = (
    "root",
    "third",
    "fifth",
    "seventh",
    "other_chord",
    "upper_neighbor",
    "lower_neighbor",
    "other_non_chord",
)


class ConditionalNoteModel:
    """Corpus-informed prior over melody notes conditioned on harmony."""

    def __init__(
        self,
        counts: Dict[str, Dict[str, float]],
        total_count: float,
        version: int = 1,
    ) -> None:
        self.counts = counts
        self.total_count = float(total_count)
        self.version = int(version)

    @classmethod
    def fit(
        cls,
        file_map: Dict[str, List[Any]],
        file_labels: List[List[int]],
        harmonic_model: LearnedHarmonicModel,
        phrase_length: int = 4,
    ) -> "ConditionalNoteModel":
        """Fit relation priors from bar pitch-class histograms.

        `MeasureVector` stores duration-weighted pitch-class mass rather than
        individual notes.  That makes v1 conservative: it learns harmonic
        relation distributions robustly, while leaving rhythm/onset-specific
        learning for a later note-event extraction pass.
        """
        counts: Dict[str, Dict[str, float]] = {}
        total = 0.0
        for file_index, vectors in enumerate(file_map.values()):
            if not vectors:
                continue
            labels = file_labels[file_index] if file_index < len(file_labels) else []
            tonic_pc = harmonic_model._infer_tonic(vectors)
            for i, vec in enumerate(vectors):
                label = int(labels[i]) if i < len(labels) else int(getattr(vec, "cluster_label", -1))
                chord = harmonic_model.estimate_chord(vec, tonic_pc)
                phrase_role = harmonic_model._phrase_role(i, len(vectors), phrase_length)
                hist = np.asarray(getattr(vec, "pitch_class_histogram", np.zeros(12)), dtype=np.float64)
                if hist.shape != (12,) or float(hist.sum()) <= 0:
                    continue
                hist = hist / float(hist.sum())
                for pc, mass in enumerate(hist):
                    if mass <= 0:
                        continue
                    relation = cls._relation_for_pc(pc, chord.root_pc, chord.quality)
                    for key in cls._keys(label, phrase_role, chord.roman):
                        cls._inc(counts.setdefault(key, {}), relation, float(mass))
                    total += float(mass)
        return cls(counts=counts or {"global": {"root": 1.0}}, total_count=total)

    def score_candidate(
        self,
        notes: List[NoteEvent],
        target: BarGenerationTarget,
        cluster_label: int,
        config: Dict[str, Any],
    ) -> float:
        """Return log-likelihood plus optional diagnostics objective."""
        if not target.harmony:
            return 0.0
        cfg = config.get("conditional_note_model", {})
        if not isinstance(cfg, dict) or not cfg.get("enabled", True):
            return 0.0

        melody = sorted(
            [n for n in notes if n.pitch >= 0 and n.voice == "melody"],
            key=lambda n: (n.beat_offset, n.pitch),
        )
        if not melody:
            return -3.0

        roman = str(target.harmony.get("roman", "I"))
        quality = str(target.harmony.get("quality", "maj"))
        root_pc = int(target.harmony.get("root_pc", 0)) % 12
        keys = self._keys(cluster_label, str(target.development_role), roman)
        priors = [self._prior_for_key(key) for key in keys]
        mixture = self._mix_priors(priors)

        logp = 0.0
        for note in melody:
            relation = self._relation_for_pc(note.pitch % 12, root_pc, quality)
            logp += math.log(max(1e-6, mixture.get(relation, 1e-6)))
        logp /= max(1, len(melody))

        weights = cfg.get("diagnostic_targets", {})
        if not isinstance(weights, dict):
            weights = {}
        diagnostics = HarmonicPlanner.diagnostics(notes, target.harmony, config)
        objective = 0.0
        chord_ratio = diagnostics.get("chord_tone_ratio")
        strong_ratio = diagnostics.get("strong_beat_chord_tone_ratio")
        resolution_cost = diagnostics.get("non_chord_resolution_cost")
        if chord_ratio is not None:
            target_value = float(weights.get("chord_tone_ratio_target", 0.55))
            objective -= abs(float(chord_ratio) - target_value) * float(weights.get("chord_tone_ratio", 0.35))
        if strong_ratio is not None:
            target_value = float(weights.get("strong_beat_chord_tone_target", 0.85))
            objective -= max(0.0, target_value - float(strong_ratio)) * float(
                weights.get("strong_beat_chord_tone", 0.65)
            )
        if resolution_cost is not None:
            objective -= float(resolution_cost) * float(weights.get("non_chord_resolution", 0.55))

        return float(logp + objective)

    def _prior_for_key(self, key: str) -> Dict[str, float]:
        counts = self.counts.get(key) or self.counts.get("global") or {"root": 1.0}
        total = sum(float(v) for v in counts.values()) + 0.25 * len(_RELATIONS)
        return {rel: (float(counts.get(rel, 0.0)) + 0.25) / total for rel in _RELATIONS}

    @staticmethod
    def _mix_priors(priors: List[Dict[str, float]]) -> Dict[str, float]:
        weights = [0.50, 0.25, 0.15, 0.10]
        mixed = {rel: 0.0 for rel in _RELATIONS}
        for i, prior in enumerate(priors):
            weight = weights[i] if i < len(weights) else 0.05
            for rel in _RELATIONS:
                mixed[rel] += weight * prior.get(rel, 0.0)
        total = sum(mixed.values())
        return {rel: value / total for rel, value in mixed.items()} if total > 0 else priors[-1]

    @staticmethod
    def _keys(cluster_label: int, phrase_role: str, roman: str) -> List[str]:
        return [
            f"cluster:{cluster_label}|role:{phrase_role}|roman:{roman}",
            f"cluster:{cluster_label}|roman:{roman}",
            f"role:{phrase_role}|roman:{roman}",
            f"roman:{roman}",
            "global",
        ]

    @staticmethod
    def _relation_for_pc(pc: int, root_pc: int, quality: str) -> str:
        degree = (int(pc) - int(root_pc)) % 12
        if degree == 0:
            return "root"
        if degree in (3, 4):
            return "third"
        if degree == 7:
            return "fifth"
        if degree == 10:
            return "seventh"
        chord_intervals = {
            "maj": {0, 4, 7},
            "min": {0, 3, 7},
            "dim": {0, 3, 6},
            "dom7": {0, 4, 7, 10},
            "min7": {0, 3, 7, 10},
        }.get(quality, {0, 4, 7})
        if degree in chord_intervals:
            return "other_chord"
        if (degree - 1) % 12 in chord_intervals:
            return "upper_neighbor"
        if (degree + 1) % 12 in chord_intervals:
            return "lower_neighbor"
        return "other_non_chord"

    @staticmethod
    def _inc(counts: Dict[str, float], key: str, amount: float) -> None:
        counts[key] = counts.get(key, 0.0) + amount

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "total_count": self.total_count,
            "counts": self.counts,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConditionalNoteModel":
        return cls(
            counts={
                str(k): {str(kk): float(vv) for kk, vv in v.items()}
                for k, v in data.get("counts", {}).items()
            },
            total_count=float(data.get("total_count", 0.0)),
            version=int(data.get("version", 1)),
        )

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "conditional_notes.json", "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "ConditionalNoteModel":
        with open(path / "conditional_notes.json") as f:
            return cls.from_dict(json.load(f))
