"""Learned candidate reranker for generated measure candidates.

This module is intentionally separate from the existing hand-written scorers.
It learns a soft naturalness prior from real training bars versus perturbed
negative bars, then uses that prior as one more candidate-selection signal.

The reranker does not edit notes and does not contain one-bar repair rules.
It converts the diagnostics we have been inspecting by hand into a reusable
feature surface for data-informed candidate ranking.
"""

from __future__ import annotations

import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from harmonic_planner import HarmonicPlanner
from hierarchical_types import BarGenerationTarget, NoteEvent
from candidate_reranker_calibration import CandidateRerankerCalibration


FeatureDict = Dict[str, float | str | int]


_QUALITY_INTERVALS: Dict[str, Tuple[int, ...]] = {
    "maj": (0, 4, 7),
    "min": (0, 3, 7),
    "dim": (0, 3, 6),
    "dom7": (0, 4, 7, 10),
}


@dataclass(frozen=True)
class RerankerScore:
    """Interpretable score returned for one candidate."""

    probability: float
    logit: float
    weighted: float
    enabled: bool
    model_available: bool
    raw_probability: float = 0.5
    raw_logit: float = 0.0
    calibrated_probability: float = 0.5
    calibrated_logit: float = 0.0
    calibration_adjustment: float = 0.0
    good_cadence_confidence: float = 0.0


class CandidateReranker:
    """Lightweight learned prior for candidate naturalness/style fit."""

    version = 2

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
        file_labels: Mapping[str, Sequence[int]] | Sequence[Sequence[int]],
        harmonic_model: Any,
        negative_per_positive: int = 3,
        seed: int = 42,
        max_samples: Optional[int] = None,
    ) -> "CandidateReranker":
        """Train from real bars and synthetic perturbations.

        Positive examples are untouched corpus bars.  Negative examples are
        pitch/rhythm/register/contour perturbations of those bars.  This gives
        the model a product-safe first objective: prefer candidates that look
        like real corpus bars under their harmonic context.
        """
        try:
            from sklearn.feature_extraction import DictVectorizer
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
        except Exception:
            return cls(
                training_summary={
                    "trained": False,
                    "version": cls.version,
                    "reason": "scikit-learn unavailable",
                }
            )

        rng = np.random.RandomState(seed)
        features: List[FeatureDict] = []
        labels: List[int] = []
        sample_weights: List[float] = []
        positive_count = 0
        negative_count = 0
        negative_type_counts: Dict[str, int] = {}
        positive_role_counts: Dict[str, int] = {}

        items: List[Tuple[str, int, Any, int, Dict[str, Any]]] = []
        if isinstance(file_labels, Mapping):
            label_items = [
                (filepath, vectors, list(file_labels.get(filepath, [])))
                for filepath, vectors in file_map.items()
            ]
        else:
            label_items = [
                (filepath, vectors, list(label_seq))
                for (filepath, vectors), label_seq in zip(file_map.items(), file_labels)
            ]

        for filepath, vectors, label_seq in label_items:
            tonic_pc = int(harmonic_model._infer_tonic(list(vectors))) if harmonic_model else 0
            for index, vector in enumerate(vectors):
                if index >= len(label_seq):
                    continue
                notes = cls._notes_from_vector(vector)
                if len(notes) < 2:
                    continue
                harmony = cls._harmony_from_vector(vector, tonic_pc, harmonic_model)
                items.append((filepath, index, vector, int(label_seq[index]), harmony))

        if max_samples and max_samples > 0 and len(items) > max_samples:
            chosen = rng.choice(len(items), size=max_samples, replace=False)
            items = [items[int(i)] for i in chosen]

        for filepath, index, vector, cluster_id, harmony in items:
            real_notes = cls._notes_from_vector(vector)
            context = cls._training_context(index, len(file_map.get(filepath, [])), harmony)
            phrase_role = str(context.harmony.get("cadence_role", "CONTINUATION")) if context.harmony else "CONTINUATION"
            features.append(
                cls.extract_features(
                    real_notes,
                    target=context,
                    cluster_id=cluster_id,
                    config=cls._training_config(),
                )
            )
            labels.append(1)
            sample_weights.append(1.45 if phrase_role == "CADENCE" else 1.0)
            positive_count += 1
            positive_role_counts[phrase_role] = positive_role_counts.get(phrase_role, 0) + 1

            for neg_i in range(max(1, negative_per_positive)):
                perturbed, negative_type = cls._perturb_notes(
                    real_notes,
                    rng,
                    neg_i,
                    harmony=context.harmony,
                    phrase_role=phrase_role,
                )
                features.append(
                    cls.extract_features(
                        perturbed,
                        target=context,
                        cluster_id=cluster_id,
                        config=cls._training_config(),
                        source_notes=real_notes,
                    )
                )
                labels.append(0)
                sample_weights.append(cls._negative_sample_weight(negative_type, phrase_role))
                negative_count += 1
                negative_type_counts[negative_type] = negative_type_counts.get(negative_type, 0) + 1

        if len(set(labels)) < 2 or len(labels) < 20:
            return cls(
                training_summary={
                    "trained": False,
                    "version": cls.version,
                    "reason": "not enough positive/negative training samples",
                    "samples": len(labels),
                }
            )

        pipeline = Pipeline(
            steps=[
                ("vectorizer", DictVectorizer(sparse=True)),
                ("scaler", StandardScaler(with_mean=False)),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        )
        pipeline.fit(features, labels, classifier__sample_weight=sample_weights)
        vectorizer = pipeline.named_steps["vectorizer"]
        return cls(
            pipeline=pipeline,
            feature_names=list(vectorizer.get_feature_names_out()),
            training_summary={
                "trained": True,
                "version": cls.version,
                "positive_samples": positive_count,
                "negative_samples": negative_count,
                "feature_count": len(vectorizer.get_feature_names_out()),
                "negative_per_positive": negative_per_positive,
                "positive_role_counts": positive_role_counts,
                "negative_type_counts": negative_type_counts,
                "max_samples": max_samples,
            },
        )

    def score_candidate(
        self,
        notes: Sequence[NoteEvent],
        target: Optional[BarGenerationTarget],
        cluster_id: int,
        config: Mapping[str, Any],
        *,
        source_notes: Optional[Sequence[NoteEvent]] = None,
        partner_notes: Optional[Sequence[NoteEvent]] = None,
        score_components: Optional[Mapping[str, float]] = None,
        proposal_kind: Optional[str] = None,
    ) -> RerankerScore:
        cfg = self._config(config)
        enabled = bool(cfg.get("enabled", True))
        weight = float(cfg.get("weight", 0.0))
        if not enabled or not self.available or weight == 0.0:
            return RerankerScore(
                probability=0.5,
                logit=0.0,
                weighted=0.0,
                enabled=enabled,
                model_available=self.available,
                raw_probability=0.5,
                raw_logit=0.0,
                calibrated_probability=0.5,
                calibrated_logit=0.0,
            )

        features = self.extract_features(
            notes,
            target=target,
            cluster_id=cluster_id,
            config=config,
            source_notes=source_notes,
            partner_notes=partner_notes,
            score_components=score_components,
            proposal_kind=proposal_kind,
        )
        probability = float(self.pipeline.predict_proba([features])[0][1])
        eps = 1e-5
        probability = max(eps, min(1.0 - eps, probability))
        logit = math.log(probability / (1.0 - probability))
        clip = float(cfg.get("logit_clip", 3.0))
        logit = max(-clip, min(clip, logit))
        calibration = CandidateRerankerCalibration.calibrate(logit, features, cfg)
        return RerankerScore(
            probability=calibration.calibrated_probability,
            logit=calibration.calibrated_logit,
            weighted=weight * calibration.calibrated_logit,
            enabled=True,
            model_available=True,
            raw_probability=probability,
            raw_logit=logit,
            calibrated_probability=calibration.calibrated_probability,
            calibrated_logit=calibration.calibrated_logit,
            calibration_adjustment=calibration.adjustment,
            good_cadence_confidence=calibration.good_cadence_confidence,
        )

    @classmethod
    def extract_features(
        cls,
        notes: Sequence[NoteEvent],
        *,
        target: Optional[BarGenerationTarget],
        cluster_id: int,
        config: Mapping[str, Any],
        source_notes: Optional[Sequence[NoteEvent]] = None,
        partner_notes: Optional[Sequence[NoteEvent]] = None,
        score_components: Optional[Mapping[str, float]] = None,
        proposal_kind: Optional[str] = None,
    ) -> FeatureDict:
        melody = cls._melody(notes)
        pitches = [n.pitch for n in melody]
        durations = [n.duration_ql for n in melody]
        intervals = [b - a for a, b in zip(pitches, pitches[1:])]
        harmony = getattr(target, "harmony", None) if target is not None else None
        diagnostics = HarmonicPlanner.diagnostics(list(notes), harmony, dict(config)) if harmony else {}

        features: FeatureDict = {
            "version": cls.version,
            "cluster_id": int(cluster_id),
            "note_count": len(melody),
            "mean_pitch": cls._mean(pitches),
            "pitch_range": float(max(pitches) - min(pitches)) if pitches else 0.0,
            "first_last_interval": float(pitches[-1] - pitches[0]) if len(pitches) >= 2 else 0.0,
            "mean_duration": cls._mean(durations),
            "duration_std": cls._std(durations),
            "short_note_ratio": cls._ratio(d < 0.5 for d in durations),
            "step_ratio": cls._ratio(abs(iv) <= 2 for iv in intervals),
            "leap_ratio": cls._ratio(abs(iv) >= 7 for iv in intervals),
            "large_leap_count": sum(1 for iv in intervals if abs(iv) >= 9),
            "direction_changes": cls._direction_changes(intervals),
            "mean_abs_interval": cls._mean([abs(iv) for iv in intervals]),
            "max_abs_interval": float(max([abs(iv) for iv in intervals], default=0)),
            "density": len(melody) / max(1.0, cls._bar_length(melody)),
            "silence_ratio": cls._silence_ratio(melody),
            "syncopated_onset_ratio": cls._ratio(
                abs((n.beat_offset % 1.0)) > 1e-6 for n in melody
            ),
            "harmony_score": float(diagnostics.get("score", 0.0) or 0.0),
            "chord_tone_ratio": float(diagnostics.get("chord_tone_ratio") or 0.0),
            "strong_beat_chord_tone_ratio": float(
                diagnostics.get("strong_beat_chord_tone_ratio") or 0.0
            ),
            "unresolved_dissonance_cost": float(
                diagnostics.get("unresolved_dissonance_cost") or 0.0
            ),
            "non_chord_resolution_cost": float(
                diagnostics.get("non_chord_resolution_cost") or 0.0
            ),
            "cadence_strength": float(getattr(target, "cadence_strength", 0.0) or 0.0),
            "tension": float(getattr(target, "tension", 0.0) or 0.0),
            "register_target_distance": cls._register_target_distance(melody, target),
            "target_pitch_distance": cls._target_pitch_distance(melody, target),
            "final_duration_ratio": (
                float(melody[-1].duration_ql) / max(0.25, cls._bar_length(melody))
                if melody else 0.0
            ),
            "final_is_chord_tone": cls._final_is_chord_tone(melody, harmony),
            "final_is_root": cls._final_is_root(melody, harmony),
            "terminal_non_chord": cls._terminal_non_chord(melody, harmony),
            "source_contour_distance": cls._contour_distance(melody, source_notes),
            "source_rhythm_distance": cls._rhythm_distance(melody, source_notes),
            "partner_contour_distance": cls._contour_distance(melody, partner_notes),
            "partner_rhythm_distance": cls._rhythm_distance(melody, partner_notes),
            "proposal_kind": proposal_kind or "none",
        }

        if harmony:
            features["roman"] = str(harmony.get("roman", "none"))
            features["harmonic_function"] = str(harmony.get("function", "none"))
            features["cadence_role"] = str(harmony.get("cadence_role", "none"))
            features["is_cadence"] = 1 if str(harmony.get("cadence_role", "")) == "CADENCE" else 0
        if target is not None:
            features["development_role"] = str(getattr(target, "development_role", "none"))
            features["relation"] = str(getattr(target, "relation", "none"))

        for key, value in (score_components or {}).items():
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                features[f"component_{key}"] = float(value)
        return features

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "feature_names": self.feature_names,
            "training_summary": self.training_summary,
        }

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "candidate_reranker.pkl", "wb") as f:
            pickle.dump(self.pipeline, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "CandidateReranker":
        with open(path / "candidate_reranker.pkl", "rb") as f:
            pipeline = pickle.load(f)
        metadata_path = path / "metadata.json"
        metadata = {}
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
        cfg = config.get("candidate_reranker", {})
        return cfg if isinstance(cfg, Mapping) else {}

    @staticmethod
    def _training_config() -> Dict[str, Any]:
        return {
            "harmony": {
                "enabled": True,
                "strong_beats": [0.0, 2.0],
                "strong_beat_tolerance": 0.08,
                "scoring": {
                    "chord_tone_ratio": 1.0,
                    "strong_beat_chord_tone": 1.8,
                    "cadence_fit": 2.2,
                    "unresolved_dissonance": 0.7,
                    "non_chord_resolution": 0.85,
                },
            }
        }

    @staticmethod
    def _training_context(
        index: int,
        total: int,
        harmony: Dict[str, Any],
        phrase_length: int = 4,
    ) -> BarGenerationTarget:
        phrase_pos = index / max(1, total - 1)
        local = index % max(2, phrase_length)
        if index >= total - 1 or local == phrase_length - 1:
            phrase_role = "CADENCE"
        elif local == phrase_length - 2:
            phrase_role = "CADENCE_PREP"
        elif local == 0:
            phrase_role = "OPENING"
        else:
            phrase_role = "CONTINUATION"
        cadence_strength = 1.0 if phrase_role == "CADENCE" else 0.45 if phrase_role == "CADENCE_PREP" else 0.0
        harmony = dict(harmony)
        harmony["cadence_role"] = phrase_role
        target_pitch = int(harmony.get("root_pc", 0)) + 60
        return BarGenerationTarget(
            relation="TRAINING",
            source_bar=None,
            development_role="CADENTIAL" if phrase_role == "CADENCE" else phrase_role,
            rhythm_cell=(),
            contour=(),
            target_pitch=target_pitch,
            target_degree=int(harmony.get("root_pc", 0)) % 12,
            register_target=float(target_pitch),
            cadence_strength=cadence_strength,
            tension=float(min(1.0, max(0.0, phrase_pos))),
            exact_copy_penalty=0.0,
            harmony=harmony,
        )

    @staticmethod
    def _harmony_from_vector(vector: Any, tonic_pc: int, harmonic_model: Any) -> Dict[str, Any]:
        chord = harmonic_model.estimate_chord(vector, tonic_pc) if harmonic_model else None
        if chord is None:
            root_pc = tonic_pc % 12
            quality = "maj"
            roman = "I"
            function = "T"
            bass_pc = root_pc
        else:
            root_pc = int(chord.root_pc) % 12
            quality = str(chord.quality)
            roman = str(chord.roman)
            function = str(chord.function)
            bass_pc = int(chord.bass_pc) % 12
        intervals = _QUALITY_INTERVALS.get(quality, _QUALITY_INTERVALS["maj"])
        return {
            "tonic_pc": tonic_pc % 12,
            "roman": roman,
            "function": function,
            "root_pc": root_pc,
            "quality": quality,
            "chord_tones": [(root_pc + iv) % 12 for iv in intervals],
            "cadence_role": function,
            "bass_pc": bass_pc,
        }

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
            notes.append(
                NoteEvent(
                    pitch=pitch,
                    duration_ql=float(event.get("duration", event.get("quarterLength", 0.25))),
                    velocity=int(event.get("velocity", 80)),
                    beat_offset=float(event.get("onset", event.get("onset_in_measure", 0.0))),
                    voice="melody",
                )
            )
        return sorted(notes, key=lambda n: (n.beat_offset, n.pitch))

    @staticmethod
    def _perturb_notes(
        notes: Sequence[NoteEvent],
        rng: np.random.RandomState,
        variant: int,
        *,
        harmony: Optional[Mapping[str, Any]] = None,
        phrase_role: str = "CONTINUATION",
    ) -> Tuple[List[NoteEvent], str]:
        result = list(notes)
        if not result:
            return [], "empty"
        if phrase_role == "CADENCE" and variant % 3 == 0:
            return CandidateReranker._bad_cadence_notes(result, rng, harmony), "bad_cadence"
        if variant % 3 == 1:
            return CandidateReranker._harmony_mismatch_notes(result, rng, harmony), "harmony_mismatch"
        if variant % 3 == 2:
            return CandidateReranker._unresolved_non_chord_notes(result, rng, harmony), "unresolved_non_chord"

        mode = variant % 5
        if mode == 0:
            shift = int(rng.choice([-5, -4, -3, 3, 4, 5, 7]))
            return [CandidateReranker._replace_note(n, pitch=n.pitch + shift) for n in result], "register_shift"
        if mode == 1:
            changed = []
            for n in result:
                if rng.rand() < 0.45:
                    changed.append(CandidateReranker._replace_note(n, pitch=n.pitch + int(rng.choice([-7, -5, -2, 2, 5, 7]))))
                else:
                    changed.append(n)
            return changed, "pitch_noise"
        if mode == 2 and len(result) >= 3:
            pitches = [n.pitch for n in result]
            rng.shuffle(pitches)
            return [CandidateReranker._replace_note(n, pitch=int(p)) for n, p in zip(result, pitches)], "contour_scramble"
        if mode == 3:
            changed = []
            for n in result:
                scale = float(rng.choice([0.5, 0.75, 1.5, 2.0]))
                changed.append(CandidateReranker._replace_note(n, duration_ql=max(0.125, min(4.0, n.duration_ql * scale))))
            return changed, "rhythm_distortion"
        inverted = []
        center = int(round(CandidateReranker._mean([n.pitch for n in result])))
        for n in result:
            inverted.append(CandidateReranker._replace_note(n, pitch=center - (n.pitch - center)))
        return inverted, "contour_inversion"

    @staticmethod
    def _negative_sample_weight(negative_type: str, phrase_role: str) -> float:
        base = {
            "bad_cadence": 1.90,
            "harmony_mismatch": 1.35,
            "unresolved_non_chord": 1.55,
            "register_shift": 1.00,
            "pitch_noise": 1.00,
            "contour_scramble": 1.10,
            "rhythm_distortion": 1.05,
            "contour_inversion": 1.10,
        }.get(negative_type, 1.0)
        if phrase_role == "CADENCE" and negative_type in {"bad_cadence", "unresolved_non_chord"}:
            base *= 1.25
        return float(base)

    @staticmethod
    def _bad_cadence_notes(
        notes: Sequence[NoteEvent],
        rng: np.random.RandomState,
        harmony: Optional[Mapping[str, Any]],
    ) -> List[NoteEvent]:
        result = list(notes)
        melody_indices = [i for i, n in enumerate(result) if n.voice == "melody" and n.pitch >= 0]
        if not melody_indices:
            return result
        idx = melody_indices[-1]
        bad_pitch = CandidateReranker._nearest_non_chord_pitch(result[idx].pitch, harmony, rng)
        result[idx] = CandidateReranker._replace_note(
            result[idx],
            pitch=bad_pitch,
            duration_ql=min(result[idx].duration_ql, 0.5),
        )
        return result

    @staticmethod
    def _harmony_mismatch_notes(
        notes: Sequence[NoteEvent],
        rng: np.random.RandomState,
        harmony: Optional[Mapping[str, Any]],
    ) -> List[NoteEvent]:
        result = []
        for note in notes:
            if note.voice != "melody" or note.pitch < 0 or rng.rand() > 0.55:
                result.append(note)
                continue
            result.append(
                CandidateReranker._replace_note(
                    note,
                    pitch=CandidateReranker._nearest_non_chord_pitch(note.pitch, harmony, rng),
                )
            )
        return result

    @staticmethod
    def _unresolved_non_chord_notes(
        notes: Sequence[NoteEvent],
        rng: np.random.RandomState,
        harmony: Optional[Mapping[str, Any]],
    ) -> List[NoteEvent]:
        result = list(notes)
        melody_indices = [i for i, n in enumerate(result) if n.voice == "melody" and n.pitch >= 0]
        if len(melody_indices) < 2:
            return result
        idx = int(rng.choice(melody_indices[:-1]))
        next_idx = melody_indices[melody_indices.index(idx) + 1]
        bad_pitch = CandidateReranker._nearest_non_chord_pitch(result[idx].pitch, harmony, rng)
        result[idx] = CandidateReranker._replace_note(result[idx], pitch=bad_pitch)
        leap = int(rng.choice([-7, -6, 6, 7]))
        result[next_idx] = CandidateReranker._replace_note(result[next_idx], pitch=bad_pitch + leap)
        return result

    @staticmethod
    def _nearest_non_chord_pitch(
        pitch: int,
        harmony: Optional[Mapping[str, Any]],
        rng: np.random.RandomState,
    ) -> int:
        chord_tones = set()
        if harmony:
            chord_tones = {int(pc) % 12 for pc in harmony.get("chord_tones", [])}
        candidates = [
            pitch + offset
            for offset in range(-6, 7)
            if not chord_tones or (pitch + offset) % 12 not in chord_tones
        ]
        if not candidates:
            candidates = [pitch + int(rng.choice([-5, -3, 3, 5]))]
        return int(min(candidates, key=lambda p: (abs(p - pitch), rng.rand())))

    @staticmethod
    def _replace_note(
        note: NoteEvent,
        *,
        pitch: Optional[int] = None,
        duration_ql: Optional[float] = None,
    ) -> NoteEvent:
        return NoteEvent(
            pitch=int(max(0, min(127, note.pitch if pitch is None else pitch))),
            duration_ql=float(note.duration_ql if duration_ql is None else duration_ql),
            velocity=note.velocity,
            beat_offset=note.beat_offset,
            voice=note.voice,
        )

    @staticmethod
    def _melody(notes: Sequence[NoteEvent]) -> List[NoteEvent]:
        return sorted(
            [n for n in notes if n.pitch >= 0 and n.voice == "melody"],
            key=lambda n: (n.beat_offset, n.pitch),
        )

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
    def _direction_changes(intervals: Sequence[int]) -> int:
        signs = [1 if iv > 0 else -1 if iv < 0 else 0 for iv in intervals]
        signs = [s for s in signs if s != 0]
        return sum(1 for a, b in zip(signs, signs[1:]) if a != b)

    @staticmethod
    def _bar_length(notes: Sequence[NoteEvent]) -> float:
        if not notes:
            return 4.0
        return max(4.0, max(n.beat_offset + n.duration_ql for n in notes))

    @staticmethod
    def _silence_ratio(notes: Sequence[NoteEvent]) -> float:
        if not notes:
            return 1.0
        total = sum(max(0.0, n.duration_ql) for n in notes)
        return max(0.0, min(1.0, 1.0 - total / CandidateReranker._bar_length(notes)))

    @staticmethod
    def _register_target_distance(
        melody: Sequence[NoteEvent],
        target: Optional[BarGenerationTarget],
    ) -> float:
        if not melody or target is None:
            return 0.0
        return abs(CandidateReranker._mean([n.pitch for n in melody]) - float(target.register_target))

    @staticmethod
    def _target_pitch_distance(
        melody: Sequence[NoteEvent],
        target: Optional[BarGenerationTarget],
    ) -> float:
        if not melody or target is None:
            return 0.0
        return abs(float(melody[-1].pitch) - float(target.target_pitch))

    @staticmethod
    def _final_is_chord_tone(
        melody: Sequence[NoteEvent],
        harmony: Optional[Mapping[str, Any]],
    ) -> int:
        if not melody or not harmony:
            return 0
        chord_tones = {int(pc) % 12 for pc in harmony.get("chord_tones", [])}
        return 1 if melody[-1].pitch % 12 in chord_tones else 0

    @staticmethod
    def _final_is_root(
        melody: Sequence[NoteEvent],
        harmony: Optional[Mapping[str, Any]],
    ) -> int:
        if not melody or not harmony:
            return 0
        return 1 if melody[-1].pitch % 12 == int(harmony.get("root_pc", -1)) % 12 else 0

    @staticmethod
    def _terminal_non_chord(
        melody: Sequence[NoteEvent],
        harmony: Optional[Mapping[str, Any]],
    ) -> int:
        if not melody or not harmony:
            return 0
        chord_tones = {int(pc) % 12 for pc in harmony.get("chord_tones", [])}
        tail = melody[-2:] if len(melody) >= 2 else melody[-1:]
        return 1 if any(n.pitch % 12 not in chord_tones for n in tail) else 0

    @staticmethod
    def _contour_distance(
        melody: Sequence[NoteEvent],
        other_notes: Optional[Sequence[NoteEvent]],
    ) -> float:
        other = CandidateReranker._melody(other_notes or [])
        if len(melody) < 2 or len(other) < 2:
            return 0.0
        a = [b.pitch - a.pitch for a, b in zip(melody, melody[1:])]
        b = [y.pitch - x.pitch for x, y in zip(other, other[1:])]
        count = min(len(a), len(b))
        if count <= 0:
            return 0.0
        return float(np.mean([abs(a[i] - b[i]) for i in range(count)]))

    @staticmethod
    def _rhythm_distance(
        melody: Sequence[NoteEvent],
        other_notes: Optional[Sequence[NoteEvent]],
    ) -> float:
        other = CandidateReranker._melody(other_notes or [])
        if not melody or not other:
            return 0.0
        count = min(len(melody), len(other))
        return float(np.mean([abs(melody[i].duration_ql - other[i].duration_ql) for i in range(count)]))
