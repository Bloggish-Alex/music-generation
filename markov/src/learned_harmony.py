#!/usr/bin/env python3
"""Learned harmonic model v1.

This model estimates one chord per bar from existing MeasureVector data, then
learns Roman-numeral priors and transitions. It is intentionally approximate:
the goal is to replace fixed rule progressions with corpus-informed harmonic
choices while keeping the module independent and optional.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from harmonic_planner import HarmonicContext


_QUALITIES: Dict[str, Tuple[int, ...]] = {
    "maj": (0, 4, 7),
    "min": (0, 3, 7),
    "dim": (0, 3, 6),
    "dom7": (0, 4, 7, 10),
}

_ROMAN_BY_DEGREE_QUALITY: Dict[Tuple[int, str], str] = {
    (0, "maj"): "I",
    (0, "dom7"): "I",
    (2, "min"): "ii",
    (2, "dom7"): "V/V",
    (5, "maj"): "IV",
    (7, "maj"): "V",
    (7, "dom7"): "V",
    (9, "min"): "vi",
}

_ROMAN_TO_DEGREE_QUALITY_FUNCTION: Dict[str, Tuple[int, str, str]] = {
    "I": (0, "maj", "T"),
    "ii": (2, "min", "PD"),
    "IV": (5, "maj", "PD"),
    "V": (7, "dom7", "D"),
    "vi": (9, "min", "T"),
    "V/V": (2, "dom7", "D"),
}

_FALLBACK_ROMANS = ["I", "vi", "IV", "V"]


@dataclass(frozen=True)
class EstimatedChord:
    """One estimated training-bar chord."""

    tonic_pc: int
    root_pc: int
    quality: str
    roman: str
    function: str
    bass_pc: int


class LearnedHarmonicModel:
    """Corpus-informed harmonic timeline model."""

    def __init__(
        self,
        roman_counts: Dict[str, int],
        start_counts: Dict[str, int],
        transition_counts: Dict[str, Dict[str, int]],
        role_counts: Dict[str, Dict[str, int]],
        cadence_counts: Dict[str, int],
        bass_interval_counts: Dict[str, int],
        cadence_transition_counts: Optional[Dict[str, Dict[str, int]]] = None,
        cadence_bigram_counts: Optional[Dict[str, int]] = None,
        cadence_trigram_counts: Optional[Dict[str, int]] = None,
    ) -> None:
        self.roman_counts = roman_counts
        self.start_counts = start_counts
        self.transition_counts = transition_counts
        self.role_counts = role_counts
        self.cadence_counts = cadence_counts
        self.bass_interval_counts = bass_interval_counts
        self.cadence_transition_counts = cadence_transition_counts or {}
        self.cadence_bigram_counts = cadence_bigram_counts or {}
        self.cadence_trigram_counts = cadence_trigram_counts or {}

    @classmethod
    def fit(cls, file_map: Dict[str, List[Any]], phrase_length: int = 4) -> "LearnedHarmonicModel":
        """Estimate chords from MeasureVectors and fit sequence statistics."""
        roman_counts: Dict[str, int] = {}
        start_counts: Dict[str, int] = {}
        transition_counts: Dict[str, Dict[str, int]] = {}
        role_counts: Dict[str, Dict[str, int]] = {}
        cadence_counts: Dict[str, int] = {}
        bass_interval_counts: Dict[str, int] = {}
        cadence_transition_counts: Dict[str, Dict[str, int]] = {}
        cadence_bigram_counts: Dict[str, int] = {}
        cadence_trigram_counts: Dict[str, int] = {}

        for vectors in file_map.values():
            if not vectors:
                continue
            tonic_pc = cls._infer_tonic(vectors)
            chords = [cls.estimate_chord(vec, tonic_pc) for vec in vectors]
            for i, chord in enumerate(chords):
                role = cls._phrase_role(i, len(chords), phrase_length)
                if role == "CADENCE":
                    if i >= 1:
                        cls._inc(cadence_bigram_counts, f"{chords[i - 1].roman}>{chord.roman}")
                    if i >= 2:
                        cls._inc(
                            cadence_trigram_counts,
                            f"{chords[i - 2].roman}>{chords[i - 1].roman}>{chord.roman}",
                        )
            previous: Optional[EstimatedChord] = None
            for index, chord in enumerate(chords):
                role = cls._phrase_role(index, len(chords), phrase_length)
                cls._inc(roman_counts, chord.roman)
                cls._inc(role_counts.setdefault(role, {}), chord.roman)
                if index == 0:
                    cls._inc(start_counts, chord.roman)
                if role == "CADENCE":
                    cls._inc(cadence_counts, chord.roman)
                if previous is not None:
                    cls._inc(transition_counts.setdefault(previous.roman, {}), chord.roman)
                    interval = (chord.bass_pc - previous.bass_pc) % 12
                    cls._inc(bass_interval_counts, str(interval))
                    if role == "CADENCE":
                        cls._inc(cadence_transition_counts.setdefault(previous.roman, {}), chord.roman)
                previous = chord

        return cls(
            roman_counts=roman_counts or {"I": 1},
            start_counts=start_counts or {"I": 1},
            transition_counts=transition_counts or {"I": {"V": 1}, "V": {"I": 1}},
            role_counts=role_counts or {},
            cadence_counts=cadence_counts or {"I": 1, "V": 1},
            bass_interval_counts=bass_interval_counts or {"0": 1, "7": 1, "5": 1},
            cadence_transition_counts=cadence_transition_counts or {"V": {"I": 1}},
            cadence_bigram_counts=cadence_bigram_counts or {"V>I": 1},
            cadence_trigram_counts=cadence_trigram_counts or {"ii>V>I": 1, "IV>V>I": 1},
        )

    @staticmethod
    def estimate_chord(vector: Any, tonic_pc: int) -> EstimatedChord:
        """Estimate a chord from pitch-class mass and bass pitch."""
        hist = np.asarray(getattr(vector, "pitch_class_histogram", np.zeros(12)), dtype=np.float64)
        if hist.shape != (12,) or float(hist.sum()) <= 0:
            hist = np.ones(12, dtype=np.float64) / 12.0
        else:
            hist = hist / float(hist.sum())
        bass_pc = int(getattr(vector, "bass_pitch", tonic_pc)) % 12

        best: Tuple[float, int, str] = (-1e9, tonic_pc, "maj")
        for root in range(12):
            for quality, intervals in _QUALITIES.items():
                tones = [(root + iv) % 12 for iv in intervals]
                non_tones = [pc for pc in range(12) if pc not in tones]
                score = float(hist[tones].sum()) - 0.35 * float(hist[non_tones].sum())
                if bass_pc == root:
                    score += 0.34
                elif bass_pc == (root + 7) % 12:
                    score += 0.13
                degree = (root - tonic_pc) % 12
                if (degree, quality) in _ROMAN_BY_DEGREE_QUALITY:
                    score += 0.10
                elif degree in (1, 6, 8, 10, 11):
                    score -= 0.08
                if quality == "dim":
                    score -= 0.06
                if quality == "dom7" and hist[(root + 10) % 12] < 0.04:
                    score -= 0.16
                if score > best[0]:
                    best = (score, root, quality)

        _, root_pc, quality = best
        degree = (root_pc - tonic_pc) % 12
        roman = _ROMAN_BY_DEGREE_QUALITY.get((degree, quality))
        if roman is None:
            roman = LearnedHarmonicModel._nearest_known_roman(degree, quality)
        _, _, function = _ROMAN_TO_DEGREE_QUALITY_FUNCTION.get(roman, (degree, quality, "T"))
        return EstimatedChord(
            tonic_pc=tonic_pc,
            root_pc=root_pc,
            quality=quality,
            roman=roman,
            function=function,
            bass_pc=bass_pc,
        )

    def build_plan(
        self,
        measure_context: List[Tuple[str, int, str, int, int]],
        measure_affects: Dict[int, Dict[str, float]],
        global_tonic_pc: int,
        seed: int = 0,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[int, HarmonicContext]:
        """Sample a harmonic context per generated bar."""
        rng = np.random.RandomState(seed)
        plan: Dict[int, HarmonicContext] = {}
        previous_roman: Optional[str] = None
        same_run = 0
        cadence_overrides = self._build_cadence_overrides(measure_context, rng, config)

        for bar_index, (_, local_bar, role, _, section_len) in enumerate(measure_context):
            affect = measure_affects.get(bar_index, {})
            tonic_pc = int(affect.get("tonal_pc", global_tonic_pc)) % 12
            phrase_role = (
                "CONTINUATION"
                if role in ("FREE", "FLAT")
                else self._section_phrase_role(local_bar, section_len)
            )
            force_cadence = local_bar >= section_len - 1 and role not in ("FREE", "FLAT")
            if bar_index in cadence_overrides:
                roman = cadence_overrides[bar_index]
            else:
                roman = self._sample_roman(previous_roman, phrase_role, force_cadence, same_run, rng)
            if force_cadence and not self._is_allowed_cadence_endpoint(roman, config):
                roman = self._repair_cadence_endpoint(previous_roman, config)
            plan[bar_index] = self._context_from_roman(tonic_pc, roman, force_cadence, phrase_role)
            same_run = same_run + 1 if roman == previous_roman else 1
            previous_roman = roman
        return plan

    def _build_cadence_overrides(
        self,
        measure_context: List[Tuple[str, int, str, int, int]],
        rng: np.random.RandomState,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[int, str]:
        """Choose cadence bigram/trigram patterns for structural cadences."""
        overrides: Dict[int, str] = {}
        cadence_endpoints = {
            bar_index
            for bar_index, (_, local_bar, role, _, section_len) in enumerate(measure_context)
            if role not in ("FREE", "FLAT") and local_bar >= section_len - 1
        }
        for bar_index, (_, local_bar, role, _, section_len) in enumerate(measure_context):
            if role in ("FREE", "FLAT") or local_bar < section_len - 1:
                continue
            # For the final cadence of the whole piece, use the strongest
            # validity constraint: end on I, approach from V if possible.
            is_global_final = bar_index >= len(measure_context) - 1
            pattern = self._sample_cadence_pattern(rng, final=is_global_final, config=config)
            start = bar_index - len(pattern) + 1
            if start < 0:
                pattern = pattern[-(bar_index + 1):]
                start = 0
            for offset, roman in enumerate(pattern):
                idx = start + offset
                if 0 <= idx <= bar_index:
                    # A later cadence pattern may start on a previous cadence
                    # endpoint.  Do not let a preparation chord relabel an
                    # already closed section as ii/IV.
                    if idx in cadence_endpoints and idx != bar_index:
                        continue
                    overrides[idx] = roman
            if not self._is_allowed_cadence_endpoint(overrides.get(bar_index, "I"), config):
                overrides[bar_index] = self._repair_cadence_endpoint(overrides.get(bar_index - 1), config)
        return overrides

    def _sample_cadence_pattern(
        self,
        rng: np.random.RandomState,
        final: bool = False,
        config: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        if final:
            return ["V", "I"]
        valid_trigrams = {
            pattern: count
            for pattern, count in self.cadence_trigram_counts.items()
            if self._is_valid_cadence_pattern(pattern.split(">"), config)
        }
        valid_bigrams = {
            pattern: count
            for pattern, count in self.cadence_bigram_counts.items()
            if self._is_valid_cadence_pattern(pattern.split(">"), config)
        }
        # Add tonal fallback patterns as priors, not hardcoded sole behavior.
        valid_trigrams = self._mix_counts(valid_trigrams, {"ii>V>I": 4, "IV>V>I": 3, "V/V>V>I": 2})
        valid_bigrams = self._mix_counts(valid_bigrams, {"V>I": 6, "V>V": 1})
        use_trigram = bool(valid_trigrams) and rng.random() < 0.55
        pattern_key = self._sample_from_counts(valid_trigrams if use_trigram else valid_bigrams, rng)
        pattern = pattern_key.split(">")
        if not self._is_valid_cadence_pattern(pattern, config):
            pattern = ["V", "I"]
        return pattern

    @staticmethod
    def _is_valid_cadence_pattern(pattern: List[str], config: Optional[Dict[str, Any]] = None) -> bool:
        if not pattern:
            return False
        if not LearnedHarmonicModel._is_allowed_cadence_endpoint(pattern[-1], config):
            return False
        if pattern[-1] == "I":
            return len(pattern) == 1 or pattern[-2] in ("V", "V/V", "ii", "IV")
        return pattern[-1] == "V"

    @staticmethod
    def _is_allowed_cadence_endpoint(roman: str, config: Optional[Dict[str, Any]] = None) -> bool:
        allowed = ("I", "V")
        if isinstance(config, dict):
            harmony_cfg = config.get("harmony", {})
            cadence_cfg = harmony_cfg.get("cadence", {}) if isinstance(harmony_cfg, dict) else {}
            if isinstance(cadence_cfg, dict):
                raw = cadence_cfg.get("allowed_endpoints")
                if isinstance(raw, list) and raw:
                    allowed = tuple(str(x) for x in raw)
        return roman in allowed

    @staticmethod
    def _repair_cadence_endpoint(
        previous_roman: Optional[str],
        config: Optional[Dict[str, Any]] = None,
    ) -> str:
        default_endpoint = "I"
        if isinstance(config, dict):
            harmony_cfg = config.get("harmony", {})
            cadence_cfg = harmony_cfg.get("cadence", {}) if isinstance(harmony_cfg, dict) else {}
            if isinstance(cadence_cfg, dict):
                default_endpoint = str(cadence_cfg.get("final", default_endpoint))
        if previous_roman in ("V", "V/V"):
            return "I"
        return default_endpoint if default_endpoint in ("I", "V") else "I"

    def _sample_roman(
        self,
        previous_roman: Optional[str],
        phrase_role: str,
        force_cadence: bool,
        same_run: int,
        rng: np.random.RandomState,
    ) -> str:
        if force_cadence:
            if previous_roman and previous_roman in self.cadence_transition_counts:
                pool = self._mix_counts(
                    self.cadence_transition_counts[previous_roman],
                    {"I": 8, "V": 2},
                )
            else:
                pool = self._mix_counts(self.cadence_counts, {"I": 8, "V": 2})
            pool = {k: v for k, v in pool.items() if k in ("I", "V")}
        elif previous_roman and previous_roman in self.transition_counts:
            role_counts = self.role_counts.get(phrase_role, {})
            pool = self._mix_counts(self.transition_counts[previous_roman], role_counts)
        elif previous_roman is None:
            pool = self.start_counts
        else:
            pool = self.role_counts.get(phrase_role, self.roman_counts)
        if phrase_role == "CADENCE_PREP":
            pool = self._mix_counts(pool, {"V": 5, "V/V": 2, "ii": 1, "IV": 1})
        if same_run >= 2 and previous_roman in pool and len(pool) > 1:
            pool = dict(pool)
            pool[previous_roman] = max(0, int(round(pool[previous_roman] * 0.15)))
        return self._sample_from_counts(pool, rng)

    @staticmethod
    def _context_from_roman(
        tonic_pc: int,
        roman: str,
        force_cadence: bool,
        phrase_role: str,
    ) -> HarmonicContext:
        degree, quality, function = _ROMAN_TO_DEGREE_QUALITY_FUNCTION.get(
            roman, _ROMAN_TO_DEGREE_QUALITY_FUNCTION["I"]
        )
        root_pc = (tonic_pc + degree) % 12
        chord_tones = tuple((root_pc + iv) % 12 for iv in _QUALITIES[quality])
        cadence_role = "CADENCE" if force_cadence else phrase_role
        return HarmonicContext(
            tonic_pc=tonic_pc,
            roman=roman,
            function=function,
            root_pc=root_pc,
            quality=quality,
            chord_tones=chord_tones,
            cadence_role=cadence_role,
            bass_pc=root_pc,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": 2,
            "roman_counts": self.roman_counts,
            "start_counts": self.start_counts,
            "transition_counts": self.transition_counts,
            "role_counts": self.role_counts,
            "cadence_counts": self.cadence_counts,
            "bass_interval_counts": self.bass_interval_counts,
            "cadence_transition_counts": self.cadence_transition_counts,
            "cadence_bigram_counts": self.cadence_bigram_counts,
            "cadence_trigram_counts": self.cadence_trigram_counts,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LearnedHarmonicModel":
        return cls(
            roman_counts={str(k): int(v) for k, v in data.get("roman_counts", {}).items()},
            start_counts={str(k): int(v) for k, v in data.get("start_counts", {}).items()},
            transition_counts={
                str(k): {str(kk): int(vv) for kk, vv in v.items()}
                for k, v in data.get("transition_counts", {}).items()
            },
            role_counts={
                str(k): {str(kk): int(vv) for kk, vv in v.items()}
                for k, v in data.get("role_counts", {}).items()
            },
            cadence_counts={str(k): int(v) for k, v in data.get("cadence_counts", {}).items()},
            bass_interval_counts={str(k): int(v) for k, v in data.get("bass_interval_counts", {}).items()},
            cadence_transition_counts={
                str(k): {str(kk): int(vv) for kk, vv in v.items()}
                for k, v in data.get("cadence_transition_counts", {}).items()
            },
            cadence_bigram_counts={str(k): int(v) for k, v in data.get("cadence_bigram_counts", {}).items()},
            cadence_trigram_counts={str(k): int(v) for k, v in data.get("cadence_trigram_counts", {}).items()},
        )

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "harmony.json", "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "LearnedHarmonicModel":
        with open(path / "harmony.json") as f:
            return cls.from_dict(json.load(f))

    @staticmethod
    def _infer_tonic(vectors: List[Any]) -> int:
        mass = np.zeros(12, dtype=np.float64)
        for vec in vectors:
            hist = np.asarray(getattr(vec, "pitch_class_histogram", np.zeros(12)), dtype=np.float64)
            if hist.shape == (12,):
                mass += hist
        return int(np.argmax(mass)) if float(mass.sum()) > 0 else 0

    @staticmethod
    def _nearest_known_roman(degree: int, quality: str) -> str:
        if degree in (0, 11, 1):
            return "I"
        if degree in (2, 3):
            return "ii" if quality != "dom7" else "V/V"
        if degree in (4, 5, 6):
            return "IV"
        if degree in (7, 8):
            return "V"
        if degree in (9, 10):
            return "vi"
        return "I"

    @staticmethod
    def _phrase_role(index: int, total_len: int, phrase_length: int) -> str:
        local = index % max(2, phrase_length)
        if index >= total_len - 1:
            return "CADENCE"
        if local == 0:
            return "OPENING"
        if local >= phrase_length - 1:
            return "CADENCE_PREP"
        return "CONTINUATION"

    @staticmethod
    def _section_phrase_role(local_bar: int, section_len: int) -> str:
        if section_len <= 1:
            return "CADENCE"
        if local_bar == 0:
            return "OPENING"
        if local_bar >= section_len - 1:
            return "CADENCE"
        if local_bar == section_len - 2:
            return "CADENCE_PREP"
        return "CONTINUATION"

    @staticmethod
    def _inc(counts: Dict[str, int], key: str, amount: int = 1) -> None:
        counts[key] = counts.get(key, 0) + amount

    @staticmethod
    def _mix_counts(primary: Dict[str, int], secondary: Dict[str, int]) -> Dict[str, int]:
        mixed = dict(primary)
        for key, value in secondary.items():
            mixed[key] = mixed.get(key, 0) + max(1, int(round(value * 0.35)))
        return mixed or {"I": 1}

    @staticmethod
    def _sample_from_counts(counts: Dict[str, int], rng: np.random.RandomState) -> str:
        valid = {k: max(0, int(v)) for k, v in counts.items() if k in _ROMAN_TO_DEGREE_QUALITY_FUNCTION}
        if not valid:
            valid = {k: 1 for k in _FALLBACK_ROMANS}
        labels = sorted(valid)
        weights = np.array([valid[k] for k in labels], dtype=np.float64) + 0.25
        weights /= weights.sum()
        return str(rng.choice(labels, p=weights))
