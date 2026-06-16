#!/usr/bin/env python3
"""Learned candidate selection between decoder symbols and physical rendering."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from data.bar_density import TokenDensityAnalyzer
from common.config_loader import ConfigView
from data.generation_data import CodebookCandidate, CodebookEntry, HarmonyBarPlan, SampledBar


@dataclass(frozen=True)
class CandidateSelectorConfig:
    """Configuration for the learned post-emission candidate selector."""

    enabled: bool = True
    backend: str = "learned_ranker"
    model_type: str = "mlp"
    selection_stage: str = "candidate"
    train_enabled: bool = True
    negatives_per_positive: int = 3
    hidden_dim: int = 16
    epochs: int = 160
    learning_rate: float = 0.01
    l2: float = 0.0001
    random_seed: int = 42
    min_candidates: int = 2
    top_k: int = 32
    temperature: float = 0.75
    learned_weight: float = 1.0
    harmony_weight: float = 0.5
    position_weight: float = 0.25
    boundary_weight: float = 0.8
    boundary_sigma: float = 8.0
    latent_weight: float = 0.25
    latent_sigma: float = 4.0
    observation_top_k: int = 16
    candidates_per_observation: int = 5
    observation_emission_weight: float = 1.8
    min_score: float = 1.0e-9
    diagnostics_top_k: int = 5

    @classmethod
    def from_style_config(cls, config: Dict[str, Any]) -> "CandidateSelectorConfig":
        section = ConfigView(config).section("candidate_selector")
        return cls(
            enabled=bool(section.get("enabled", False)),
            backend=str(section.get("backend", "none")),
            model_type=str(section.get("model_type", "mlp")),
            selection_stage=str(section.get("selection_stage", "candidate")),
            train_enabled=bool(section.get("train_enabled", True)),
            negatives_per_positive=int(section.get("negatives_per_positive", 3)),
            hidden_dim=int(section.get("hidden_dim", 16)),
            epochs=int(section.get("epochs", 160)),
            learning_rate=float(section.get("learning_rate", 0.01)),
            l2=float(section.get("l2", 0.0001)),
            random_seed=int(section.get("random_seed", 42)),
            min_candidates=int(section.get("min_candidates", 2)),
            top_k=int(section.get("top_k", 32)),
            temperature=float(section.get("temperature", 0.75)),
            learned_weight=float(section.get("learned_weight", 1.0)),
            harmony_weight=float(section.get("harmony_weight", 0.5)),
            position_weight=float(section.get("position_weight", 0.25)),
            boundary_weight=float(section.get("boundary_weight", 0.8)),
            boundary_sigma=float(section.get("boundary_sigma", 8.0)),
            latent_weight=float(section.get("latent_weight", 0.25)),
            latent_sigma=float(section.get("latent_sigma", 4.0)),
            observation_top_k=int(section.get("observation_top_k", 16)),
            candidates_per_observation=int(section.get("candidates_per_observation", 5)),
            observation_emission_weight=float(section.get("observation_emission_weight", 1.8)),
            min_score=float(section.get("min_score", 1.0e-9)),
            diagnostics_top_k=int(section.get("diagnostics_top_k", 5)),
        )


@dataclass(frozen=True)
class CandidateSelectionContext:
    """Runtime context for choosing a concrete candidate bar."""

    sampled: SampledBar
    harmony: HarmonyBarPlan
    section_length: int
    previous_candidate: Optional[CodebookCandidate] = None
    previous_harmony: Optional[HarmonyBarPlan] = None


@dataclass(frozen=True)
class CandidateSelectionResult:
    """Selected codebook entry plus diagnostics."""

    entry: CodebookEntry
    diagnostics: Dict[str, Any]


class CandidateFeatureExtractor:
    """Extract small, stable features for candidate naturalness scoring."""

    FEATURE_NAMES = [
        "latent_distance",
        "boundary_abs_interval",
        "boundary_signed_interval",
        "density_delta_abs",
        "note_on_ratio_delta_abs",
        "rest_ratio_delta_abs",
        "sustain_ratio_delta_abs",
        "token_variance_delta_abs",
        "sharing_score_delta_abs",
        "same_source_song",
        "source_index_distance",
        "source_index_is_next",
        "position_ratio_delta_abs",
        "target_position_delta_abs",
        "harmony_fit",
        "candidate_note_on_ratio",
        "candidate_rest_ratio",
        "candidate_sustain_ratio",
    ]

    MAJOR_SCALE = {0, 2, 4, 5, 7, 9, 11}
    MINOR_SCALE = {0, 2, 3, 5, 7, 8, 10}
    MAJOR_TRIADS = {
        "I": {0, 4, 7},
        "II": {0, 3, 7},
        "III": {0, 3, 7},
        "IV": {0, 4, 7},
        "V": {0, 4, 7},
        "VI": {0, 3, 7},
        "VII": {0, 3, 6},
    }
    MINOR_TRIADS = {
        "I": {0, 3, 7},
        "II": {0, 3, 6},
        "III": {0, 4, 7},
        "IV": {0, 3, 7},
        "V": {0, 4, 7},
        "VI": {0, 4, 7},
        "VII": {0, 4, 7},
    }

    def __init__(self, mode: str = "major") -> None:
        self.mode = str(mode).lower()
        self.density = TokenDensityAnalyzer()

    def vector(
        self,
        previous: Optional[CodebookCandidate],
        candidate: CodebookCandidate,
        harmony: Optional[HarmonyBarPlan] = None,
        section_length: int = 1,
        target_position: Optional[float] = None,
    ) -> np.ndarray:
        features = self.describe(previous, candidate, harmony, section_length, target_position)
        return np.asarray([float(features[name]) for name in self.FEATURE_NAMES], dtype=np.float64)

    def describe(
        self,
        previous: Optional[CodebookCandidate],
        candidate: CodebookCandidate,
        harmony: Optional[HarmonyBarPlan] = None,
        section_length: int = 1,
        target_position: Optional[float] = None,
    ) -> Dict[str, float]:
        candidate_density = candidate.density or self.density.analyze(candidate.relative_tokens)
        previous_density = (
            previous.density
            if previous is not None and previous.density is not None
            else self.density.analyze(previous.relative_tokens) if previous is not None else None
        )
        if target_position is None:
            target_position = float(candidate.position_ratio)
        if previous is None:
            return {
                "latent_distance": 0.0,
                "boundary_abs_interval": 0.0,
                "boundary_signed_interval": 0.0,
                "density_delta_abs": 0.0,
                "note_on_ratio_delta_abs": 0.0,
                "rest_ratio_delta_abs": 0.0,
                "sustain_ratio_delta_abs": 0.0,
                "token_variance_delta_abs": 0.0,
                "sharing_score_delta_abs": 0.0,
                "same_source_song": 0.0,
                "source_index_distance": 0.0,
                "source_index_is_next": 0.0,
                "position_ratio_delta_abs": 0.0,
                "target_position_delta_abs": abs(float(candidate.position_ratio) - float(target_position)),
                "harmony_fit": self.harmony_fit(candidate.relative_tokens, harmony.degree if harmony else None),
                "candidate_note_on_ratio": float(candidate_density.note_on_ratio),
                "candidate_rest_ratio": float(candidate_density.rest_ratio),
                "candidate_sustain_ratio": float(candidate_density.sustain_ratio),
            }
        signed_interval = self._boundary_interval(previous, candidate, harmony)
        source_distance = self._source_index_distance(previous, candidate)
        return {
            "latent_distance": self._latent_distance(previous, candidate),
            "boundary_abs_interval": abs(float(signed_interval)),
            "boundary_signed_interval": float(signed_interval),
            "density_delta_abs": abs(float(candidate_density.active_duration_ql) - float(previous_density.active_duration_ql)),
            "note_on_ratio_delta_abs": abs(float(candidate_density.note_on_ratio) - float(previous_density.note_on_ratio)),
            "rest_ratio_delta_abs": abs(float(candidate_density.rest_ratio) - float(previous_density.rest_ratio)),
            "sustain_ratio_delta_abs": abs(float(candidate_density.sustain_ratio) - float(previous_density.sustain_ratio)),
            "token_variance_delta_abs": abs(float(candidate.token_variance) - float(previous.token_variance)),
            "sharing_score_delta_abs": abs(float(candidate.sharing_score) - float(previous.sharing_score)),
            "same_source_song": 1.0 if str(previous.source_file) == str(candidate.source_file) else 0.0,
            "source_index_distance": float(source_distance),
            "source_index_is_next": 1.0 if source_distance == 1 else 0.0,
            "position_ratio_delta_abs": abs(float(candidate.position_ratio) - float(previous.position_ratio)),
            "target_position_delta_abs": abs(float(candidate.position_ratio) - float(target_position)),
            "harmony_fit": self.harmony_fit(candidate.relative_tokens, harmony.degree if harmony else None),
            "candidate_note_on_ratio": float(candidate_density.note_on_ratio),
            "candidate_rest_ratio": float(candidate_density.rest_ratio),
            "candidate_sustain_ratio": float(candidate_density.sustain_ratio),
        }

    def harmony_fit(self, tokens: Sequence[int], degree: Optional[str]) -> float:
        if degree is None:
            return 1.0
        pitch_classes = [int(token) % 12 for token in tokens if int(token) >= 0]
        if not pitch_classes:
            return 0.1
        triads = self.MINOR_TRIADS if self.mode == "minor" else self.MAJOR_TRIADS
        scale = self.MINOR_SCALE if self.mode == "minor" else self.MAJOR_SCALE
        chord = triads.get(str(degree), {0, 4, 7})
        chord_ratio = sum(1 for pc in pitch_classes if pc in chord) / len(pitch_classes)
        scale_ratio = sum(1 for pc in pitch_classes if pc in scale) / len(pitch_classes)
        return max(0.05, float(0.7 * chord_ratio + 0.3 * scale_ratio))

    def _latent_distance(self, previous: CodebookCandidate, candidate: CodebookCandidate) -> float:
        if previous.latent_vector is None or candidate.latent_vector is None:
            return 0.0
        previous_vector = np.asarray(previous.latent_vector, dtype=np.float64)
        candidate_vector = np.asarray(candidate.latent_vector, dtype=np.float64)
        if previous_vector.shape != candidate_vector.shape:
            return 0.0
        return float(np.linalg.norm(candidate_vector - previous_vector))

    def _boundary_interval(
        self,
        previous: CodebookCandidate,
        candidate: CodebookCandidate,
        harmony: Optional[HarmonyBarPlan],
    ) -> float:
        previous_last = self._last_note_token(previous.relative_tokens)
        candidate_first = self._first_note_token(candidate.relative_tokens)
        if previous_last is None or candidate_first is None:
            return 0.0
        return float(int(candidate_first) - int(previous_last))

    def _first_note_token(self, tokens: Sequence[int]) -> Optional[int]:
        for token in tokens:
            value = int(token)
            if value >= 0:
                return value
        return None

    def _last_note_token(self, tokens: Sequence[int]) -> Optional[int]:
        for token in reversed(tokens):
            value = int(token)
            if value >= 0:
                return value
        return None

    def _source_index_distance(self, previous: CodebookCandidate, candidate: CodebookCandidate) -> int:
        if (
            previous.source_file is None
            or candidate.source_file is None
            or str(previous.source_file) != str(candidate.source_file)
            or previous.source_bar_index is None
            or candidate.source_bar_index is None
        ):
            return 99
        return abs(int(candidate.source_bar_index) - int(previous.source_bar_index))


@dataclass
class CandidateSelectorModel:
    """Serializable one-hidden-layer MLP for candidate naturalness."""

    feature_names: List[str]
    mean: List[float]
    std: List[float]
    w1: List[List[float]]
    b1: List[float]
    w2: List[float]
    b2: float
    diagnostics: Dict[str, Any]

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        mean = np.asarray(self.mean, dtype=np.float64)
        std = np.asarray(self.std, dtype=np.float64)
        x = (x - mean) / np.maximum(std, 1.0e-6)
        hidden = np.tanh(x @ np.asarray(self.w1, dtype=np.float64) + np.asarray(self.b1, dtype=np.float64))
        logits = hidden @ np.asarray(self.w2, dtype=np.float64) + float(self.b2)
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Optional[Dict[str, Any]]) -> Optional["CandidateSelectorModel"]:
        if not payload:
            return None
        return cls(
            feature_names=[str(value) for value in payload.get("feature_names", [])],
            mean=[float(value) for value in payload.get("mean", [])],
            std=[float(value) for value in payload.get("std", [])],
            w1=[[float(x) for x in row] for row in payload.get("w1", [])],
            b1=[float(value) for value in payload.get("b1", [])],
            w2=[float(value) for value in payload.get("w2", [])],
            b2=float(payload.get("b2", 0.0)),
            diagnostics=dict(payload.get("diagnostics", {})),
        )


class CandidateSelectorTrainer:
    """Train the selector from real consecutive bars and replacement negatives."""

    def __init__(self, config: Dict[str, Any], mode: str = "major") -> None:
        self.config = CandidateSelectorConfig.from_style_config(config)
        self.extractor = CandidateFeatureExtractor(mode=mode)
        self.rng = np.random.default_rng(self.config.random_seed)
        self.diagnostics: Dict[str, Any] = {}

    def fit(self, codebook: Dict[int, CodebookEntry]) -> Optional[CandidateSelectorModel]:
        if (
            not self.config.enabled
            or self.config.backend != "learned_ranker"
            or self.config.model_type != "mlp"
            or not self.config.train_enabled
        ):
            self.diagnostics = {
                "enabled": bool(self.config.enabled),
                "backend": self.config.backend,
                "model_type": self.config.model_type,
                "trained": False,
                "reason": "disabled_or_not_learned_ranker_mlp",
            }
            return None
        x, y = self._training_examples(codebook)
        if len(y) < 20 or len(set(int(value) for value in y.tolist())) < 2:
            self.diagnostics = {
                "enabled": True,
                "backend": self.config.backend,
                "trained": False,
                "reason": "insufficient_training_examples",
                "example_count": int(len(y)),
            }
            return None
        model = self._fit_mlp(x, y)
        probabilities = model.predict_proba(x)
        predictions = (probabilities >= 0.5).astype(int)
        accuracy = float(np.mean(predictions == y))
        positive_rate = float(np.mean(y))
        model.diagnostics.update({
            "enabled": True,
            "backend": self.config.backend,
            "trained": True,
            "example_count": int(len(y)),
            "positive_count": int(np.sum(y == 1)),
            "negative_count": int(np.sum(y == 0)),
            "positive_rate": round(positive_rate, 6),
            "training_accuracy": round(accuracy, 6),
            "config": asdict(self.config),
        })
        self.diagnostics = dict(model.diagnostics)
        return model

    def _training_examples(self, codebook: Dict[int, CodebookEntry]) -> Tuple[np.ndarray, np.ndarray]:
        by_source: Dict[str, Dict[int, Tuple[int, CodebookCandidate]]] = defaultdict(dict)
        for codebook_id, entry in codebook.items():
            for candidate in entry.candidates:
                if candidate.source_file is None or candidate.source_bar_index is None:
                    continue
                by_source[str(candidate.source_file)][int(candidate.source_bar_index)] = (int(codebook_id), candidate)
        features: List[np.ndarray] = []
        labels: List[int] = []
        for _, indexed in sorted(by_source.items()):
            for index in sorted(indexed):
                if index + 1 not in indexed:
                    continue
                _, previous = indexed[index]
                target_codebook_id, positive = indexed[index + 1]
                features.append(self.extractor.vector(previous, positive))
                labels.append(1)
                pool = [
                    candidate
                    for candidate in codebook[target_codebook_id].candidates
                    if not (
                        str(candidate.source_file) == str(positive.source_file)
                        and candidate.source_bar_index == positive.source_bar_index
                    )
                ]
                if not pool:
                    continue
                negatives = self._negative_examples(previous, pool)
                for negative in negatives:
                    features.append(self.extractor.vector(previous, negative))
                    labels.append(0)
        if not features:
            return np.empty((0, len(self.extractor.FEATURE_NAMES)), dtype=np.float64), np.asarray([], dtype=int)
        return np.vstack(features).astype(np.float64), np.asarray(labels, dtype=int)

    def _negative_examples(
        self,
        previous: CodebookCandidate,
        pool: Sequence[CodebookCandidate],
    ) -> List[CodebookCandidate]:
        negative_count = min(len(pool), max(1, self.config.negatives_per_positive))
        if negative_count <= 0:
            return []
        scored = [
            (
                float(self.extractor.describe(previous, candidate).get("boundary_abs_interval", 0.0)),
                index,
                candidate,
            )
            for index, candidate in enumerate(pool)
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        hard_count = min(len(scored), max(1, negative_count // 2))
        selected_indices = {int(index) for _, index, _ in scored[:hard_count]}
        remaining_indices = [index for index in range(len(pool)) if index not in selected_indices]
        random_count = negative_count - len(selected_indices)
        if random_count > 0 and remaining_indices:
            sampled = self.rng.choice(
                len(remaining_indices),
                size=min(random_count, len(remaining_indices)),
                replace=False,
            )
            selected_indices.update(int(remaining_indices[int(item)]) for item in sampled)
        return [pool[index] for index in sorted(selected_indices)]

    def _fit_mlp(self, x: np.ndarray, y: np.ndarray) -> CandidateSelectorModel:
        mean = np.mean(x, axis=0)
        std = np.std(x, axis=0)
        x_norm = (x - mean) / np.maximum(std, 1.0e-6)
        n_features = x_norm.shape[1]
        hidden_dim = max(2, int(self.config.hidden_dim))
        w1 = self.rng.normal(0.0, 0.1, size=(n_features, hidden_dim))
        b1 = np.zeros(hidden_dim, dtype=np.float64)
        w2 = self.rng.normal(0.0, 0.1, size=hidden_dim)
        b2 = 0.0
        learning_rate = float(self.config.learning_rate)
        l2 = float(self.config.l2)
        y_float = y.astype(np.float64)
        for _ in range(max(1, int(self.config.epochs))):
            hidden = np.tanh(x_norm @ w1 + b1)
            logits = hidden @ w2 + b2
            probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
            error = probs - y_float
            grad_w2 = hidden.T @ error / len(y_float) + l2 * w2
            grad_b2 = float(np.mean(error))
            grad_hidden = error[:, None] * w2[None, :] * (1.0 - hidden ** 2)
            grad_w1 = x_norm.T @ grad_hidden / len(y_float) + l2 * w1
            grad_b1 = np.mean(grad_hidden, axis=0)
            w2 -= learning_rate * grad_w2
            b2 -= learning_rate * grad_b2
            w1 -= learning_rate * grad_w1
            b1 -= learning_rate * grad_b1
        return CandidateSelectorModel(
            feature_names=list(self.extractor.FEATURE_NAMES),
            mean=[float(value) for value in mean.tolist()],
            std=[float(value) for value in std.tolist()],
            w1=[[float(x) for x in row] for row in w1.tolist()],
            b1=[float(value) for value in b1.tolist()],
            w2=[float(value) for value in w2.tolist()],
            b2=float(b2),
            diagnostics={},
        )


class LearnedCandidateSelector:
    """Runtime learned selector for candidates inside an emitted codebook ID."""

    def __init__(
        self,
        config: Dict[str, Any],
        model: Optional[CandidateSelectorModel],
        mode: str = "major",
    ) -> None:
        self.config = CandidateSelectorConfig.from_style_config(config)
        self.model = model
        self.extractor = CandidateFeatureExtractor(mode=mode)

    def select(
        self,
        entry: CodebookEntry,
        context: CandidateSelectionContext,
        rng: np.random.Generator,
    ) -> CandidateSelectionResult:
        if (
            not self.config.enabled
            or self.config.backend != "learned_ranker"
            or self.config.model_type != "mlp"
            or self.model is None
            or len(entry.candidates) < self.config.min_candidates
        ):
            return CandidateSelectionResult(entry=entry, diagnostics={
                "used": False,
                "reason": "disabled_missing_model_or_insufficient_candidates",
                "candidate_count": len(entry.candidates),
                "backend": self.config.backend,
                "model_type": self.config.model_type,
            })
        target_position = self._target_position(context.sampled, context.section_length)
        descriptions = [
            self.extractor.describe(
                context.previous_candidate,
                candidate,
                context.harmony,
                context.section_length,
                target_position,
            )
            for candidate in entry.candidates
        ]
        vectors = np.vstack([
            np.asarray([float(description[name]) for name in self.model.feature_names], dtype=np.float64)
            for description in descriptions
        ])
        learned = np.maximum(self.model.predict_proba(vectors), self.config.min_score)
        harmony = np.asarray([max(self.config.min_score, item["harmony_fit"]) for item in descriptions], dtype=np.float64)
        position = np.asarray([
            max(self.config.min_score, np.exp(-float(item["target_position_delta_abs"]) / 0.35))
            for item in descriptions
        ], dtype=np.float64)
        boundary = np.asarray([
            max(
                self.config.min_score,
                np.exp(-float(item["boundary_abs_interval"]) / max(float(self.config.boundary_sigma), 1.0e-6)),
            )
            for item in descriptions
        ], dtype=np.float64)
        latent = np.asarray([
            max(
                self.config.min_score,
                np.exp(-float(item["latent_distance"]) / max(float(self.config.latent_sigma), 1.0e-6)),
            )
            for item in descriptions
        ], dtype=np.float64)
        scores = (
            learned ** self.config.learned_weight
            * harmony ** self.config.harmony_weight
            * position ** self.config.position_weight
            * boundary ** self.config.boundary_weight
            * latent ** self.config.latent_weight
        )
        candidate_indices = self._candidate_indices(scores)
        probabilities = self._sampling_probabilities(scores, candidate_indices)
        if len(candidate_indices) == 0:
            selected_index = int(rng.integers(0, len(entry.candidates)))
            full_probabilities = np.full(len(entry.candidates), 1.0 / len(entry.candidates), dtype=np.float64)
        else:
            selected_local = int(rng.choice(len(candidate_indices), p=probabilities))
            selected_index = int(candidate_indices[selected_local])
            full_probabilities = np.zeros(len(entry.candidates), dtype=np.float64)
            for index, probability in zip(candidate_indices, probabilities):
                full_probabilities[int(index)] = float(probability)
        selected = entry.candidates[selected_index]
        return CandidateSelectionResult(
            entry=self._entry_from_candidate(entry.codebook_id, selected, entry.candidates),
            diagnostics={
                "used": True,
                "backend": self.config.backend,
                "model_type": self.config.model_type,
                "candidate_count": len(entry.candidates),
                "sampling_candidate_count": int(len(candidate_indices)),
                "selected_index": selected_index,
                "selected_probability": round(float(full_probabilities[selected_index]), 6),
                "selected": self._diagnostic_item(
                    selected_index,
                    descriptions[selected_index],
                    learned,
                    scores,
                    full_probabilities,
                    boundary,
                    latent,
                ),
                "top_candidates": self._top_candidates(descriptions, learned, scores, full_probabilities, boundary, latent),
            },
        )

    def select_from_observations(
        self,
        observation_entries: Sequence[Tuple[int, float, str, CodebookEntry]],
        context: CandidateSelectionContext,
        rng: np.random.Generator,
    ) -> Tuple[Optional[int], Optional[float], Optional[str], CandidateSelectionResult]:
        """Jointly choose an emitted observation and concrete candidate bar."""
        if (
            not self.config.enabled
            or self.config.backend != "learned_ranker"
            or self.config.model_type != "mlp"
            or self.model is None
        ):
            return None, None, None, CandidateSelectionResult(
                entry=observation_entries[0][3] if observation_entries else CodebookEntry(
                    codebook_id=-1,
                    source_song=None,
                    source_file=None,
                    source_bar_index=None,
                    relative_tokens=[],
                    absolute_tokens=[],
                ),
                diagnostics={
                    "used": False,
                    "reason": "disabled_or_missing_model",
                    "backend": self.config.backend,
                    "model_type": self.config.model_type,
                },
            )
        rows: List[Dict[str, Any]] = []
        for observation_id, emission_probability, composite_key, entry in observation_entries:
            if len(entry.candidates) < self.config.min_candidates:
                continue
            scored = self._score_entry(entry, context)
            candidate_indices = self._candidate_indices(scored["scores"])
            limit = int(self.config.candidates_per_observation)
            if limit > 0:
                candidate_indices = candidate_indices[:limit]
            for candidate_index in candidate_indices:
                index = int(candidate_index)
                emission_score = max(self.config.min_score, float(emission_probability))
                joint_score = float(scored["scores"][index]) * (
                    emission_score ** float(self.config.observation_emission_weight)
                )
                rows.append({
                    "observation_id": int(observation_id),
                    "emission_probability": float(emission_probability),
                    "composite_key": str(composite_key),
                    "entry": entry,
                    "candidate_index": index,
                    "candidate": entry.candidates[index],
                    "description": scored["descriptions"][index],
                    "learned": float(scored["learned"][index]),
                    "boundary": float(scored["boundary"][index]),
                    "latent": float(scored["latent"][index]),
                    "candidate_score": float(scored["scores"][index]),
                    "joint_score": joint_score,
                })
        if not rows:
            return None, None, None, CandidateSelectionResult(
                entry=observation_entries[0][3] if observation_entries else CodebookEntry(
                    codebook_id=-1,
                    source_song=None,
                    source_file=None,
                    source_bar_index=None,
                    relative_tokens=[],
                    absolute_tokens=[],
                ),
                diagnostics={
                    "used": False,
                    "reason": "no_scored_joint_candidates",
                    "backend": self.config.backend,
                    "model_type": self.config.model_type,
                    "observation_candidate_count": len(observation_entries),
                },
            )
        scores = np.asarray([max(self.config.min_score, row["joint_score"]) for row in rows], dtype=np.float64)
        indices = np.arange(len(rows), dtype=np.int64)
        probabilities = self._sampling_probabilities(scores, indices)
        selected_row_index = int(rng.choice(len(rows), p=probabilities))
        row = rows[selected_row_index]
        full_probabilities = np.asarray(probabilities, dtype=np.float64)
        selected_entry = self._entry_from_candidate(
            row["entry"].codebook_id,
            row["candidate"],
            row["entry"].candidates,
        )
        diagnostics = {
            "used": True,
            "backend": self.config.backend,
            "model_type": self.config.model_type,
            "selection_mode": "joint_observation_candidate",
            "observation_candidate_count": len(observation_entries),
            "scored_candidate_count": len(rows),
            "selected_observation_id": int(row["observation_id"]),
            "selected_emission_probability": round(float(row["emission_probability"]), 8),
            "selected_composite_key": str(row["composite_key"]),
            "selected_index": int(row["candidate_index"]),
            "selected_probability": round(float(full_probabilities[selected_row_index]), 6),
            "selected": self._joint_diagnostic_item(row, full_probabilities[selected_row_index]),
            "top_candidates": [
                self._joint_diagnostic_item(rows[index], full_probabilities[index])
                for index in sorted(
                    range(len(rows)),
                    key=lambda item: float(rows[item]["joint_score"]),
                    reverse=True,
                )[: max(0, self.config.diagnostics_top_k)]
            ],
        }
        return (
            int(row["observation_id"]),
            float(row["emission_probability"]),
            str(row["composite_key"]),
            CandidateSelectionResult(entry=selected_entry, diagnostics=diagnostics),
        )

    def _target_position(self, sampled: SampledBar, section_length: int) -> float:
        if section_length <= 1:
            return 0.0
        return float(int(sampled.section_local_index) / max(1, section_length - 1))

    def _score_entry(self, entry: CodebookEntry, context: CandidateSelectionContext) -> Dict[str, Any]:
        target_position = self._target_position(context.sampled, context.section_length)
        descriptions = [
            self.extractor.describe(
                context.previous_candidate,
                candidate,
                context.harmony,
                context.section_length,
                target_position,
            )
            for candidate in entry.candidates
        ]
        vectors = np.vstack([
            np.asarray([float(description[name]) for name in self.model.feature_names], dtype=np.float64)
            for description in descriptions
        ])
        learned = np.maximum(self.model.predict_proba(vectors), self.config.min_score)
        harmony = np.asarray([max(self.config.min_score, item["harmony_fit"]) for item in descriptions], dtype=np.float64)
        position = np.asarray([
            max(self.config.min_score, np.exp(-float(item["target_position_delta_abs"]) / 0.35))
            for item in descriptions
        ], dtype=np.float64)
        boundary = np.asarray([
            max(
                self.config.min_score,
                np.exp(-float(item["boundary_abs_interval"]) / max(float(self.config.boundary_sigma), 1.0e-6)),
            )
            for item in descriptions
        ], dtype=np.float64)
        latent = np.asarray([
            max(
                self.config.min_score,
                np.exp(-float(item["latent_distance"]) / max(float(self.config.latent_sigma), 1.0e-6)),
            )
            for item in descriptions
        ], dtype=np.float64)
        scores = (
            learned ** self.config.learned_weight
            * harmony ** self.config.harmony_weight
            * position ** self.config.position_weight
            * boundary ** self.config.boundary_weight
            * latent ** self.config.latent_weight
        )
        return {
            "descriptions": descriptions,
            "learned": learned,
            "boundary": boundary,
            "latent": latent,
            "scores": scores,
        }

    def _candidate_indices(self, scores: np.ndarray) -> np.ndarray:
        if len(scores) == 0:
            return np.array([], dtype=np.int64)
        top_k = int(self.config.top_k)
        if top_k <= 0 or top_k >= len(scores):
            return np.arange(len(scores), dtype=np.int64)
        return np.array(
            sorted(
                np.argpartition(scores, -top_k)[-top_k:],
                key=lambda index: float(scores[int(index)]),
                reverse=True,
            ),
            dtype=np.int64,
        )

    def _sampling_probabilities(self, scores: np.ndarray, indices: np.ndarray) -> np.ndarray:
        if len(indices) == 0:
            return np.array([], dtype=np.float64)
        selected = scores[indices].astype(np.float64)
        temperature = max(float(self.config.temperature), 1.0e-6)
        adjusted = np.power(np.maximum(selected, self.config.min_score), 1.0 / temperature)
        total = float(adjusted.sum())
        if total <= 0.0:
            return np.full(len(indices), 1.0 / len(indices), dtype=np.float64)
        return adjusted / total

    def _entry_from_candidate(
        self,
        codebook_id: int,
        candidate: CodebookCandidate,
        candidates: Sequence[CodebookCandidate],
    ) -> CodebookEntry:
        return CodebookEntry(
            codebook_id=int(codebook_id),
            source_song=candidate.source_song,
            source_file=candidate.source_file,
            source_bar_index=candidate.source_bar_index,
            relative_tokens=list(candidate.relative_tokens),
            absolute_tokens=list(candidate.absolute_tokens),
            density=candidate.density,
            token_variance=float(candidate.token_variance),
            sharing_score=float(candidate.sharing_score),
            candidates=list(candidates),
            latent_vector=(
                [float(value) for value in candidate.latent_vector]
                if candidate.latent_vector is not None
                else None
            ),
            position_ratio=float(candidate.position_ratio),
        )

    def _top_candidates(
        self,
        descriptions: Sequence[Dict[str, float]],
        learned: np.ndarray,
        scores: np.ndarray,
        probabilities: np.ndarray,
        boundary: np.ndarray,
        latent: np.ndarray,
    ) -> List[Dict[str, Any]]:
        ranked = sorted(
            range(len(descriptions)),
            key=lambda index: float(scores[index]),
            reverse=True,
        )[: max(0, self.config.diagnostics_top_k)]
        return [
            self._diagnostic_item(index, descriptions[index], learned, scores, probabilities, boundary, latent)
            for index in ranked
        ]

    def _diagnostic_item(
        self,
        index: int,
        description: Dict[str, float],
        learned: np.ndarray,
        scores: np.ndarray,
        probabilities: np.ndarray,
        boundary: np.ndarray,
        latent: np.ndarray,
    ) -> Dict[str, Any]:
        return {
            "candidate_index": int(index),
            "learned_probability": round(float(learned[index]), 6),
            "boundary_score": round(float(boundary[index]), 6),
            "latent_score": round(float(latent[index]), 6),
            "final_score": round(float(scores[index]), 12),
            "probability": round(float(probabilities[index]), 6),
            **{key: round(float(value), 6) for key, value in description.items()},
        }

    def _joint_diagnostic_item(self, row: Dict[str, Any], probability: float) -> Dict[str, Any]:
        description = row["description"]
        return {
            "observation_id": int(row["observation_id"]),
            "codebook_id": int(row["entry"].codebook_id),
            "candidate_index": int(row["candidate_index"]),
            "emission_probability": round(float(row["emission_probability"]), 8),
            "learned_probability": round(float(row["learned"]), 6),
            "boundary_score": round(float(row["boundary"]), 6),
            "latent_score": round(float(row["latent"]), 6),
            "candidate_score": round(float(row["candidate_score"]), 12),
            "final_score": round(float(row["joint_score"]), 12),
            "probability": round(float(probability), 6),
            **{key: round(float(value), 6) for key, value in description.items()},
        }
