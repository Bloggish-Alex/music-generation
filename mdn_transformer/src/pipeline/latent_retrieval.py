#!/usr/bin/env python3
"""Retrieval-based latent candidate selection for generation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class RetrievalConfig:
    """Configuration for retrieval_mdn generation."""

    enabled: bool = False
    top_k: int = 24
    temperature: float = 0.35
    distance_weight: float = 1.0
    energy_weight: float = 1.25
    position_weight: float = 0.25
    recent_penalty: float = 2.0
    recent_window: int = 8
    use_retrieved_tensors: bool = True
    energy_curve: str = "INTRODUCE:0.35,VARY:0.55,DEVELOP:0.9,RETURN:0.7,CADENCE:0.45"
    energy_arc_strength: float = 0.15


@dataclass
class CandidateSelection:
    """One selected retrieval candidate."""

    row_index: int
    mu: np.ndarray
    tensor: Optional[np.ndarray]
    diagnostics: Dict[str, Any]


class BarEnergyAnalyzer:
    """Compute objective energy/tension features from bar tensors."""

    def summarize(self, tensor: np.ndarray) -> Dict[str, float]:
        """Return energy features and an aggregate energy score in [0, 1]."""
        values = np.asarray(tensor, dtype=np.float32)
        note_on = values[..., 2] > 0.5
        active = (values[..., 2] > 0.5) | (values[..., 3] > 0.5)
        total_slots = max(1, int(values.shape[0] * values.shape[1]))
        note_count = int(note_on.sum())
        active_count = int(active.sum())
        velocities = values[..., 4][note_on]
        pitches = values[..., 0][note_on] * 24.0
        onset_by_slot = note_on.sum(axis=0).astype(np.float32) if values.ndim == 3 else np.zeros(16, dtype=np.float32)
        if float(onset_by_slot.sum()) > 0.0:
            probabilities = onset_by_slot / float(onset_by_slot.sum())
            rhythm_entropy = float(-(probabilities * np.log(np.clip(probabilities, 1.0e-8, 1.0))).sum() / np.log(len(probabilities)))
        else:
            rhythm_entropy = 0.0
        if len(pitches) > 1:
            pitch_range = float(np.max(pitches) - np.min(pitches))
            interval_intensity = float(np.mean(np.abs(np.diff(pitches))))
            register_mean = float(np.mean(pitches))
            register_peak = float(np.max(pitches))
        elif len(pitches) == 1:
            pitch_range = 0.0
            interval_intensity = 0.0
            register_mean = float(pitches[0])
            register_peak = float(pitches[0])
        else:
            pitch_range = 0.0
            interval_intensity = 0.0
            register_mean = 0.0
            register_peak = 0.0
        note_density = float(note_count / total_slots)
        active_density = float(active_count / total_slots)
        avg_velocity = float(np.mean(velocities)) if len(velocities) else 0.0
        energy = (
            0.35 * np.clip(note_density * 4.0, 0.0, 1.0)
            + 0.20 * np.clip(avg_velocity, 0.0, 1.0)
            + 0.20 * np.clip(pitch_range / 24.0, 0.0, 1.0)
            + 0.15 * np.clip(interval_intensity / 12.0, 0.0, 1.0)
            + 0.10 * np.clip(rhythm_entropy, 0.0, 1.0)
        )
        return {
            "energy": float(np.clip(energy, 0.0, 1.0)),
            "note_density": note_density,
            "active_density": active_density,
            "avg_velocity": avg_velocity,
            "pitch_range": pitch_range,
            "register_mean": register_mean,
            "register_peak": register_peak,
            "rhythm_entropy": rhythm_entropy,
            "interval_intensity": interval_intensity,
            "note_count": float(note_count),
        }


class EnergyCurve:
    """Action-conditioned target energy curve."""

    def __init__(self, config: RetrievalConfig) -> None:
        self.config = config
        self.by_action = self._parse_curve(config.energy_curve)

    def target(self, action: str, bar_index: int, total_bars: int) -> float:
        """Return target energy for one bar."""
        base = float(self.by_action.get(str(action), self.by_action.get("DEFAULT", 0.55)))
        progress = 0.0 if total_bars <= 1 else float(bar_index) / float(total_bars - 1)
        arc = float(self.config.energy_arc_strength) * float(np.sin(np.pi * progress))
        return float(np.clip(base + arc, 0.0, 1.0))

    def _parse_curve(self, text: str) -> Dict[str, float]:
        """Parse ACTION:value pairs."""
        result: Dict[str, float] = {}
        for raw in str(text).split(","):
            item = raw.strip()
            if not item or ":" not in item:
                continue
            key, value = item.split(":", 1)
            result[key.strip()] = float(value.strip())
        return result or {"DEFAULT": 0.55}


class LatentCandidateIndex:
    """Searchable index over real latent bars and optional encoded tensors."""

    def __init__(
        self,
        mu: np.ndarray,
        rows: Sequence[Dict[str, Any]],
        action_to_id: Dict[str, int],
        position_vocab_size: int,
        tensors_by_key: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        self.mu = np.asarray(mu, dtype=np.float32)
        self.rows = list(rows)
        self.action_to_id = action_to_id
        self.position_vocab_size = max(1, int(position_vocab_size))
        self.tensors_by_key = tensors_by_key or {}
        self.energy_analyzer = BarEnergyAnalyzer()
        self.energy = np.asarray([self._row_energy(row) for row in self.rows], dtype=np.float32)
        self.action_ids = np.asarray([self._action_id(row) for row in self.rows], dtype=np.int64)
        self.position_ids = np.asarray([int(row.get("bar_index", 0)) % self.position_vocab_size for row in self.rows], dtype=np.int64)
        self.source_base_ids = np.asarray([self._base_song_id(str(row.get("song_id", "UNKNOWN"))) for row in self.rows], dtype=object)
        self.row_indices = np.arange(len(self.rows), dtype=np.int64)

    @classmethod
    def from_model_dir(
        cls,
        model_dir: str | Path,
        mu: np.ndarray,
        rows: Sequence[Dict[str, Any]],
        action_to_id: Dict[str, int],
        position_vocab_size: int,
    ) -> "LatentCandidateIndex":
        """Build index and load encoded tensors when available."""
        tensors_by_key = cls._load_tensors_by_key(Path(model_dir) / "encoded" / "bar_tensors.npz")
        return cls(mu, rows, action_to_id, position_vocab_size, tensors_by_key=tensors_by_key)

    def select(
        self,
        predicted_mu: np.ndarray,
        action_id: int,
        action_name: str,
        position_id: int,
        bar_index: int,
        total_bars: int,
        recent_row_indices: Sequence[int],
        config: RetrievalConfig,
        rng: np.random.Generator,
    ) -> CandidateSelection:
        """Select one real candidate near predicted_mu and target energy."""
        candidate_indices, fallback_level = self._candidate_indices(action_id, position_id)
        predicted = np.asarray(predicted_mu, dtype=np.float32)
        candidate_mu = self.mu[candidate_indices]
        latent_distance = np.linalg.norm(candidate_mu - predicted[None, :], axis=1) / max(1.0, float(np.sqrt(predicted.shape[0])))
        target_energy = EnergyCurve(config).target(action_name, bar_index=bar_index, total_bars=total_bars)
        energy_distance = np.abs(self.energy[candidate_indices] - target_energy)
        position_distance = (self.position_ids[candidate_indices] != int(position_id)).astype(np.float32)
        recent = set(int(x) for x in recent_row_indices[-max(0, int(config.recent_window)):])
        recent_penalty = np.asarray([float(config.recent_penalty) if int(idx) in recent else 0.0 for idx in candidate_indices], dtype=np.float32)
        score = (
            float(config.distance_weight) * latent_distance
            + float(config.energy_weight) * energy_distance
            + float(config.position_weight) * position_distance
            + recent_penalty
        )
        order = np.argsort(score)
        top = order[: max(1, min(int(config.top_k), len(order)))]
        selected_local = self._sample_from_scores(score[top], rng, temperature=float(config.temperature))
        local_index = int(top[selected_local])
        row_index = int(candidate_indices[local_index])
        row = self.rows[row_index]
        tensor = self.tensor_for_row(row_index)
        diagnostics = {
            "selection_backend": "retrieval_mdn",
            "selected_row_index": row_index,
            "selected_song_id": str(row.get("song_id", "UNKNOWN")),
            "selected_source_base_id": str(self.source_base_ids[row_index]),
            "selected_bar_index": int(row.get("bar_index", 0)),
            "selected_tensor_key": str(row.get("tensor_key", "")),
            "candidate_count": int(len(candidate_indices)),
            "top_k": int(len(top)),
            "fallback_level": fallback_level,
            "target_energy": float(target_energy),
            "selected_energy": float(self.energy[row_index]),
            "selected_score": float(score[local_index]),
            "selected_latent_distance": float(latent_distance[local_index]),
            "selected_energy_distance": float(energy_distance[local_index]),
            "selected_position_distance": float(position_distance[local_index]),
            "selected_recent_penalty": float(recent_penalty[local_index]),
            "has_retrieved_tensor": tensor is not None,
            "top_candidates": [
                {
                    "row_index": int(candidate_indices[int(item)]),
                    "score": float(score[int(item)]),
                    "energy": float(self.energy[int(candidate_indices[int(item)])]),
                    "latent_distance": float(latent_distance[int(item)]),
                    "song_id": str(self.rows[int(candidate_indices[int(item)])].get("song_id", "UNKNOWN")),
                    "bar_index": int(self.rows[int(candidate_indices[int(item)])].get("bar_index", 0)),
                }
                for item in top[:5]
            ],
        }
        return CandidateSelection(
            row_index=row_index,
            mu=self.mu[row_index].astype(np.float32),
            tensor=tensor,
            diagnostics=diagnostics,
        )

    def select_memory(
        self,
        query_mu: np.ndarray,
        action_id: int,
        position_id: int,
        expected_row_index: Optional[int],
        config: RetrievalConfig,
        rng: np.random.Generator,
        allowed_source_base_ids: Optional[Sequence[str]] = None,
    ) -> CandidateSelection:
        """Select one candidate with the same cosine objective used by memory training."""
        candidate_indices, fallback_level = self._candidate_indices(action_id, position_id)
        scoped_indices, scope_applied = self._apply_source_scope(candidate_indices, allowed_source_base_ids)
        candidate_indices = scoped_indices
        query = np.asarray(query_mu, dtype=np.float32)
        candidate_mu = self.mu[candidate_indices]
        similarity = self._cosine_similarity(candidate_mu, query)
        euclidean_distance = np.linalg.norm(candidate_mu - query[None, :], axis=1) / max(1.0, float(np.sqrt(query.shape[0])))
        order = np.argsort(-similarity)
        top = order[: max(1, min(int(config.top_k), len(order)))]
        selected_local = self._sample_from_similarities(similarity[top], rng, temperature=float(config.temperature))
        local_index = int(top[selected_local])
        row_index = int(candidate_indices[local_index])
        row = self.rows[row_index]
        tensor = self.tensor_for_row(row_index)
        expected_rank = self._rank_for_expected_row(candidate_indices, order, expected_row_index)
        diagnostics = {
            "selection_backend": "memory_latent",
            "selected_row_index": row_index,
            "selected_song_id": str(row.get("song_id", "UNKNOWN")),
            "selected_source_base_id": str(self.source_base_ids[row_index]),
            "selected_bar_index": int(row.get("bar_index", 0)),
            "selected_tensor_key": str(row.get("tensor_key", "")),
            "candidate_count": int(len(candidate_indices)),
            "top_k": int(len(top)),
            "fallback_level": fallback_level,
            "memory_scope_enabled": allowed_source_base_ids is not None,
            "memory_scope_applied": bool(scope_applied),
            "memory_scope_source_base_ids": list(allowed_source_base_ids or []),
            "selected_score": float(similarity[local_index]),
            "selected_memory_similarity": float(similarity[local_index]),
            "selected_cosine_distance": float(1.0 - similarity[local_index]),
            "selected_latent_distance": float(euclidean_distance[local_index]),
            "selected_energy": float(self.energy[row_index]),
            "expected_row_index": int(expected_row_index) if expected_row_index is not None else None,
            "expected_row_rank": expected_rank,
            "expected_row_similarity": self._similarity_for_expected_row(candidate_indices, similarity, expected_row_index),
            "has_retrieved_tensor": tensor is not None,
            "top_candidates": [
                {
                    "row_index": int(candidate_indices[int(item)]),
                    "score": float(similarity[int(item)]),
                    "memory_similarity": float(similarity[int(item)]),
                    "cosine_distance": float(1.0 - similarity[int(item)]),
                    "latent_distance": float(euclidean_distance[int(item)]),
                    "energy": float(self.energy[int(candidate_indices[int(item)])]),
                    "song_id": str(self.rows[int(candidate_indices[int(item)])].get("song_id", "UNKNOWN")),
                    "bar_index": int(self.rows[int(candidate_indices[int(item)])].get("bar_index", 0)),
                }
                for item in top[:5]
            ],
        }
        return CandidateSelection(
            row_index=row_index,
            mu=self.mu[row_index].astype(np.float32),
            tensor=tensor,
            diagnostics=diagnostics,
        )

    def source_neighborhood(
        self,
        query_mu: np.ndarray,
        action_id: int,
        position_id: int,
        top_n: int,
    ) -> Dict[str, Any]:
        """Return the strongest source base ids for a query under action/position filters."""
        candidate_indices, fallback_level = self._candidate_indices(action_id, position_id)
        query = np.asarray(query_mu, dtype=np.float32)
        similarity = self._cosine_similarity(self.mu[candidate_indices], query)
        best_by_source: Dict[str, float] = {}
        count_by_source: Dict[str, int] = {}
        for local_index, row_index in enumerate(candidate_indices):
            source_base_id = str(self.source_base_ids[int(row_index)])
            best_by_source[source_base_id] = max(best_by_source.get(source_base_id, -1.0e9), float(similarity[int(local_index)]))
            count_by_source[source_base_id] = count_by_source.get(source_base_id, 0) + 1
        ranked = sorted(best_by_source, key=lambda item: best_by_source[item], reverse=True)
        selected = ranked[: max(1, int(top_n))]
        return {
            "source_base_ids": selected,
            "candidate_count": int(len(candidate_indices)),
            "source_count": int(len(ranked)),
            "fallback_level": fallback_level,
            "top_sources": [
                {
                    "source_base_id": source_base_id,
                    "best_similarity": float(best_by_source[source_base_id]),
                    "candidate_count": int(count_by_source[source_base_id]),
                }
                for source_base_id in selected[:10]
            ],
        }

    def tensor_for_row(self, row_index: int) -> Optional[np.ndarray]:
        """Return encoded tensor for a latent row if available."""
        key = str(self.rows[int(row_index)].get("tensor_key", ""))
        value = self.tensors_by_key.get(key)
        return None if value is None else np.asarray(value, dtype=np.float32)

    def diagnostics(self) -> Dict[str, Any]:
        """Return index diagnostics."""
        return {
            "candidate_count": int(len(self.rows)),
            "has_tensors": bool(self.tensors_by_key),
            "tensor_count": int(len(self.tensors_by_key)),
            "energy_mean": float(np.mean(self.energy)) if len(self.energy) else 0.0,
            "energy_std": float(np.std(self.energy)) if len(self.energy) else 0.0,
            "energy_min": float(np.min(self.energy)) if len(self.energy) else 0.0,
            "energy_max": float(np.max(self.energy)) if len(self.energy) else 0.0,
        }

    def _candidate_indices(self, action_id: int, position_id: int) -> tuple[np.ndarray, str]:
        """Return candidate row indices with progressively relaxed filters."""
        mask = (self.action_ids == int(action_id)) & (self.position_ids == int(position_id))
        if bool(mask.any()):
            return self.row_indices[mask], "action_and_position"
        mask = self.action_ids == int(action_id)
        if bool(mask.any()):
            return self.row_indices[mask], "action_only"
        mask = self.position_ids == int(position_id)
        if bool(mask.any()):
            return self.row_indices[mask], "position_only"
        return self.row_indices, "all"

    def _apply_source_scope(
        self,
        candidate_indices: np.ndarray,
        allowed_source_base_ids: Optional[Sequence[str]],
    ) -> tuple[np.ndarray, bool]:
        """Restrict candidate indices to a source neighborhood when possible."""
        if not allowed_source_base_ids:
            return candidate_indices, False
        allowed = {str(item) for item in allowed_source_base_ids}
        mask = np.asarray([str(self.source_base_ids[int(index)]) in allowed for index in candidate_indices], dtype=bool)
        if not bool(mask.any()):
            return candidate_indices, False
        return candidate_indices[mask], True

    def _action_id(self, row: Dict[str, Any]) -> int:
        """Map row action to action id."""
        action = str(row.get("action") or "UNKNOWN")
        return int(self.action_to_id.get(action, self.action_to_id.get("UNKNOWN", 1)))

    def _base_song_id(self, song_id: str) -> str:
        """Collapse transposed copies into the same source id."""
        return re.sub(r"_T[+-]?\d+$", "", str(song_id))

    def _row_energy(self, row: Dict[str, Any]) -> float:
        """Return tensor energy or a neutral fallback."""
        tensor = self.tensors_by_key.get(str(row.get("tensor_key", "")))
        if tensor is None:
            return 0.55
        return float(self.energy_analyzer.summarize(tensor)["energy"])

    def _sample_from_scores(self, scores: np.ndarray, rng: np.random.Generator, temperature: float) -> int:
        """Sample a local index from lower-is-better scores."""
        values = np.asarray(scores, dtype=np.float64)
        if len(values) <= 1 or temperature <= 0.0:
            return 0
        logits = -values / max(1.0e-6, float(temperature))
        logits = logits - float(np.max(logits))
        probs = np.exp(logits)
        total = float(probs.sum())
        if not np.isfinite(total) or total <= 0.0:
            return 0
        probs = probs / total
        return int(rng.choice(len(values), p=probs))

    def _sample_from_similarities(self, similarities: np.ndarray, rng: np.random.Generator, temperature: float) -> int:
        """Sample a local index from higher-is-better similarities."""
        values = np.asarray(similarities, dtype=np.float64)
        if len(values) <= 1 or temperature <= 0.0:
            return 0
        logits = values / max(1.0e-6, float(temperature))
        logits = logits - float(np.max(logits))
        probs = np.exp(logits)
        total = float(probs.sum())
        if not np.isfinite(total) or total <= 0.0:
            return 0
        probs = probs / total
        return int(rng.choice(len(values), p=probs))

    def _cosine_similarity(self, candidates: np.ndarray, query: np.ndarray) -> np.ndarray:
        """Return cosine similarity between each candidate and query."""
        candidate_values = np.asarray(candidates, dtype=np.float32)
        query_value = np.asarray(query, dtype=np.float32)
        candidate_norm = np.linalg.norm(candidate_values, axis=1, keepdims=True)
        query_norm = float(np.linalg.norm(query_value))
        candidate_unit = candidate_values / np.clip(candidate_norm, 1.0e-8, None)
        query_unit = query_value / max(1.0e-8, query_norm)
        return np.matmul(candidate_unit, query_unit.astype(np.float32)).astype(np.float32)

    def _rank_for_expected_row(
        self,
        candidate_indices: np.ndarray,
        ordered_local_indices: np.ndarray,
        expected_row_index: Optional[int],
    ) -> Optional[int]:
        """Return 1-based rank for an expected training row inside candidates."""
        if expected_row_index is None:
            return None
        matches = np.where(candidate_indices[ordered_local_indices] == int(expected_row_index))[0]
        if len(matches) == 0:
            return None
        return int(matches[0]) + 1

    def _similarity_for_expected_row(
        self,
        candidate_indices: np.ndarray,
        similarities: np.ndarray,
        expected_row_index: Optional[int],
    ) -> Optional[float]:
        """Return cosine similarity for an expected training row when present."""
        if expected_row_index is None:
            return None
        matches = np.where(candidate_indices == int(expected_row_index))[0]
        if len(matches) == 0:
            return None
        return float(similarities[int(matches[0])])

    @staticmethod
    def _load_tensors_by_key(path: Path) -> Dict[str, np.ndarray]:
        """Load encoded bar tensors keyed by tensor_key."""
        if not path.exists():
            return {}
        data = np.load(path)
        return {str(key): np.asarray(data[key], dtype=np.float32) for key in data.files}
