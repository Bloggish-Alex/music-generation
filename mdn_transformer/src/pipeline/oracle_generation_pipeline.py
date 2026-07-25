#!/usr/bin/env python3
"""Nearest-neighbor oracle generation for data upper-bound tests."""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from diagnostics.dvae_midi_render import DVAEMidiRenderConfig
from pipeline.latent_generation_pipeline import GenerationActionPlanner, LatentGenerationConfig, SequenceTensorMidiRenderer
from pipeline.latent_retrieval import RetrievalConfig
from pipeline.latent_transformer_training_pipeline import LatentDatasetReader


@dataclass(frozen=True)
class OracleGenerationConfig:
    """Configuration for nearest-neighbor oracle generation."""

    bars: int = 32
    primer_bars: int = 8
    selection_mode: str = "nearest_neighbor"
    source_scope: str = "same_base"
    top_k: int = 16
    temperature: float = 0.25
    query_context_bars: int = 1
    position_vocab_size: int = 8
    default_action: str = "VARY"
    action_plan: str = "sections"
    action_sections: str = "INTRODUCE:8,VARY:8,DEVELOP:8,RETURN:6,CADENCE:2"
    seed: int = 42
    base_pitch: int = 60
    tempo_bpm: int = 120

    def retrieval_config(self) -> RetrievalConfig:
        """Return the retrieval config used by the oracle selector."""
        return RetrievalConfig(
            enabled=True,
            top_k=max(1, int(self.top_k)),
            temperature=float(self.temperature),
            use_retrieved_tensors=True,
        )

    def action_config(self) -> LatentGenerationConfig:
        """Return the minimal config expected by GenerationActionPlanner."""
        return LatentGenerationConfig(
            bars=int(self.bars),
            primer_bars=int(self.primer_bars),
            default_action=str(self.default_action),
            action_plan=str(self.action_plan),
            action_sections=str(self.action_sections),
            seed=int(self.seed),
            base_pitch=int(self.base_pitch),
            tempo_bpm=int(self.tempo_bpm),
        )


@dataclass
class OracleGenerationResult:
    """Paths and diagnostics produced by oracle generation."""

    json_path: Path
    midi_path: Path
    tensor_path: Path
    diagnostics: Dict[str, Any]


@dataclass
class OracleCandidateSelection:
    """One selected nearest-neighbor candidate."""

    row_index: int
    mu: np.ndarray
    tensor: Optional[np.ndarray]
    diagnostics: Dict[str, Any]


class OracleCandidateIndex:
    """Lightweight candidate index for oracle lookup without energy scoring."""

    def __init__(
        self,
        model_dir: str | Path,
        mu: np.ndarray,
        rows: Sequence[Dict[str, Any]],
        action_to_id: Dict[str, int],
        position_vocab_size: int,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.mu = np.asarray(mu, dtype=np.float32)
        self.rows = list(rows)
        self.action_to_id = dict(action_to_id)
        self.position_vocab_size = max(1, int(position_vocab_size))
        self.tensor_path = self.model_dir / "encoded" / "bar_tensors.npz"
        if not self.tensor_path.exists():
            raise FileNotFoundError(f"Nearest Neighbor Oracle requires encoded tensors at {self.tensor_path}.")
        self.tensors_archive = np.load(self.tensor_path)
        self.tensor_keys = set(str(key) for key in self.tensors_archive.files)
        self.action_ids = np.asarray([self._action_id(row) for row in self.rows], dtype=np.int64)
        self.position_ids = np.asarray([int(row.get("bar_index", 0)) % self.position_vocab_size for row in self.rows], dtype=np.int64)
        self.song_ids = np.asarray([str(row.get("song_id", "UNKNOWN")) for row in self.rows], dtype=object)
        self.source_base_ids = np.asarray([self._base_song_id(str(row.get("song_id", "UNKNOWN"))) for row in self.rows], dtype=object)
        self.row_indices = np.arange(len(self.rows), dtype=np.int64)

    def select_memory(
        self,
        query_mu: np.ndarray,
        action_id: int,
        position_id: int,
        expected_row_index: Optional[int],
        config: RetrievalConfig,
        rng: np.random.Generator,
        allowed_source_base_ids: Optional[Sequence[str]] = None,
        allowed_song_ids: Optional[Sequence[str]] = None,
    ) -> OracleCandidateSelection:
        """Select one candidate using cosine nearest-neighbor retrieval."""
        candidate_indices, fallback_level, scope_applied = self._candidate_indices(
            action_id=action_id,
            position_id=position_id,
            allowed_source_base_ids=allowed_source_base_ids,
            allowed_song_ids=allowed_song_ids,
        )
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
            "selection_backend": "nearest_neighbor_oracle",
            "selected_row_index": row_index,
            "selected_song_id": str(row.get("song_id", "UNKNOWN")),
            "selected_source_base_id": str(self.source_base_ids[row_index]),
            "selected_bar_index": int(row.get("bar_index", 0)),
            "selected_tensor_key": str(row.get("tensor_key", "")),
            "candidate_count": int(len(candidate_indices)),
            "top_k": int(len(top)),
            "fallback_level": fallback_level,
            "source_scope_enabled": allowed_source_base_ids is not None,
            "song_scope_enabled": allowed_song_ids is not None,
            "source_scope_applied": bool(scope_applied),
            "source_scope_base_ids": list(allowed_source_base_ids or []),
            "source_scope_song_ids": list(allowed_song_ids or []),
            "selected_score": float(similarity[local_index]),
            "selected_memory_similarity": float(similarity[local_index]),
            "selected_cosine_distance": float(1.0 - similarity[local_index]),
            "selected_latent_distance": float(euclidean_distance[local_index]),
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
                    "song_id": str(self.rows[int(candidate_indices[int(item)])].get("song_id", "UNKNOWN")),
                    "bar_index": int(self.rows[int(candidate_indices[int(item)])].get("bar_index", 0)),
                }
                for item in top[:5]
            ],
        }
        return OracleCandidateSelection(
            row_index=row_index,
            mu=self.mu[row_index].astype(np.float32),
            tensor=tensor,
            diagnostics=diagnostics,
        )

    def tensor_for_row(self, row_index: int) -> Optional[np.ndarray]:
        """Return encoded tensor for a latent row if available."""
        key = str(self.rows[int(row_index)].get("tensor_key", ""))
        if key not in self.tensor_keys:
            return None
        return np.asarray(self.tensors_archive[key], dtype=np.float32)

    def diagnostics(self) -> Dict[str, Any]:
        """Return index diagnostics."""
        return {
            "candidate_count": int(len(self.rows)),
            "tensor_path": str(self.tensor_path),
            "tensor_count": int(len(self.tensor_keys)),
            "position_vocab_size": int(self.position_vocab_size),
            "song_count": int(len(set(str(item) for item in self.song_ids))),
            "source_base_count": int(len(set(str(item) for item in self.source_base_ids))),
        }

    def _candidate_indices(
        self,
        action_id: int,
        position_id: int,
        allowed_source_base_ids: Optional[Sequence[str]] = None,
        allowed_song_ids: Optional[Sequence[str]] = None,
    ) -> tuple[np.ndarray, str, bool]:
        """Return candidate row indices with progressively relaxed filters."""
        if allowed_song_ids:
            scoped = self._song_indices(allowed_song_ids)
            if len(scoped) > 0:
                return self._candidate_indices_inside_scope(scoped, action_id, position_id)
        if allowed_source_base_ids:
            scoped = self._source_base_indices(allowed_source_base_ids)
            if len(scoped) > 0:
                return self._candidate_indices_inside_scope(scoped, action_id, position_id)
        mask = (self.action_ids == int(action_id)) & (self.position_ids == int(position_id))
        if bool(mask.any()):
            return self.row_indices[mask], "action_and_position", False
        mask = self.action_ids == int(action_id)
        if bool(mask.any()):
            return self.row_indices[mask], "action_only", False
        mask = self.position_ids == int(position_id)
        if bool(mask.any()):
            return self.row_indices[mask], "position_only", False
        return self.row_indices, "all", False

    def _candidate_indices_inside_scope(
        self,
        scoped_indices: np.ndarray,
        action_id: int,
        position_id: int,
    ) -> tuple[np.ndarray, str, bool]:
        """Return candidates after locking source scope first."""
        action_match = self.action_ids[scoped_indices] == int(action_id)
        position_match = self.position_ids[scoped_indices] == int(position_id)
        mask = action_match & position_match
        if not bool(mask.any()):
            mask = action_match
            if bool(mask.any()):
                return scoped_indices[mask], "source_action_only", True
            mask = position_match
            if bool(mask.any()):
                return scoped_indices[mask], "source_position_only", True
            return scoped_indices, "source_only", True
        return scoped_indices[mask], "source_action_and_position", True

    def _source_base_indices(self, allowed_source_base_ids: Sequence[str]) -> np.ndarray:
        """Return all rows belonging to the allowed source base ids."""
        allowed = {str(item) for item in allowed_source_base_ids}
        mask = np.asarray([str(item) in allowed for item in self.source_base_ids], dtype=bool)
        return self.row_indices[mask]

    def _song_indices(self, allowed_song_ids: Sequence[str]) -> np.ndarray:
        """Return all rows belonging to the allowed exact song ids."""
        allowed = {str(item) for item in allowed_song_ids}
        mask = np.asarray([str(item) in allowed for item in self.song_ids], dtype=bool)
        return self.row_indices[mask]

    def _action_id(self, row: Dict[str, Any]) -> int:
        """Map row action to action id."""
        action = str(row.get("action") or "UNKNOWN")
        return int(self.action_to_id.get(action, self.action_to_id.get("UNKNOWN", 1)))

    def _base_song_id(self, song_id: str) -> str:
        """Collapse transposed copies into the same source id."""
        return re.sub(r"_T[+-]?\d+$", "", str(song_id))

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


class OracleGenerationPipeline:
    """Generate a sequence by nearest-neighbor lookup from real latent bars."""

    PAD_ACTION = "PAD"
    UNKNOWN_ACTION = "UNKNOWN"

    def __init__(self, config: OracleGenerationConfig) -> None:
        self.config = config
        self.reader = LatentDatasetReader()

    def run(
        self,
        model_dir: str | Path,
        latent_dir: str | Path,
        output_json: str | Path,
        output_midi: str | Path,
        seed_song_id: Optional[str] = None,
    ) -> OracleGenerationResult:
        """Run oracle generation and write diagnostics, tensors, and MIDI."""
        self._set_seed()
        model_directory = Path(model_dir)
        mu, rows, latent_summary = self.reader.load(latent_dir)
        action_to_id = self._action_vocab(rows)
        grouped = self._group_rows(rows)
        selected_song_id = self._select_song_id(grouped, seed_song_id)
        ordered_indices = grouped[selected_song_id]
        retrieval_index = OracleCandidateIndex(
            model_dir=model_directory,
            mu=mu,
            rows=rows,
            action_to_id=action_to_id,
            position_vocab_size=int(self.config.position_vocab_size),
        )

        generated_mu, tensors, selected_row_indices, steps = self._generate(
            mu=mu,
            rows=rows,
            ordered_indices=ordered_indices,
            action_to_id=action_to_id,
            retrieval_index=retrieval_index,
        )
        sequence_diagnostics = self._sequence_diagnostics(tensors, steps)
        tensor_path = Path(output_json).with_suffix(".bar_tensors.npz")
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            tensor_path,
            bars=tensors.astype(np.float32),
            latent_mu=generated_mu.astype(np.float32),
            selected_row_indices=np.asarray(selected_row_indices, dtype=np.int64),
        )
        midi_diag = SequenceTensorMidiRenderer(DVAEMidiRenderConfig(
            tempo_bpm=int(self.config.tempo_bpm),
            default_base_pitch=int(self.config.base_pitch),
        )).render(tensors, output_midi, base_pitch=int(self.config.base_pitch))
        diagnostics = {
            "generation_backend": "nearest_neighbor_oracle",
            "purpose": "data_upper_bound_test",
            "model_dir": str(model_directory),
            "latent_dir": str(latent_dir),
            "selected_song_id": selected_song_id,
            "seed_song_id": seed_song_id,
            "config": self.config.__dict__,
            "latent_summary": latent_summary,
            "primer_bars": int(min(self.config.primer_bars, len(ordered_indices), self.config.bars)),
            "generated_bar_count": int(generated_mu.shape[0]),
            "action_to_id": action_to_id,
            "action_plan": GenerationActionPlanner(self.config.action_config(), action_to_id).diagnostics(target_bars=int(self.config.bars)),
            "retrieval": retrieval_index.diagnostics(),
            "sequence_diagnostics": sequence_diagnostics,
            "steps": steps,
            "midi": midi_diag,
            "tensor_path": str(tensor_path),
            "json_path": str(output_json),
            "midi_path": str(output_midi),
        }
        Path(output_json).write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
        return OracleGenerationResult(
            json_path=Path(output_json),
            midi_path=Path(output_midi),
            tensor_path=tensor_path,
            diagnostics=diagnostics,
        )

    def _generate(
        self,
        mu: np.ndarray,
        rows: Sequence[Dict[str, Any]],
        ordered_indices: Sequence[int],
        action_to_id: Dict[str, int],
        retrieval_index: OracleCandidateIndex,
    ) -> tuple[np.ndarray, np.ndarray, List[int], List[Dict[str, Any]]]:
        """Generate bars by priming with source bars and continuing via nearest neighbor."""
        if str(self.config.selection_mode).strip().lower() == "sequential":
            return self._generate_sequential(
                mu=mu,
                rows=rows,
                ordered_indices=ordered_indices,
                retrieval_index=retrieval_index,
            )
        if str(self.config.selection_mode).strip().lower() != "nearest_neighbor":
            raise ValueError(f"Unsupported oracle selection_mode: {self.config.selection_mode}")
        rng = np.random.default_rng(int(self.config.seed))
        planner = GenerationActionPlanner(self.config.action_config(), action_to_id)
        generated: List[np.ndarray] = []
        tensors: List[np.ndarray] = []
        selected_row_indices: List[int] = []
        steps: List[Dict[str, Any]] = []
        primer_count = int(min(self.config.primer_bars, len(ordered_indices), self.config.bars))
        seed_song_id = str(rows[int(ordered_indices[0])].get("song_id", "UNKNOWN")) if ordered_indices else None
        seed_base_id = str(retrieval_index.source_base_ids[int(ordered_indices[0])]) if ordered_indices else None

        for bar_index in range(int(self.config.bars)):
            expected_row_index = int(ordered_indices[bar_index]) if bar_index < len(ordered_indices) else None
            if bar_index < primer_count:
                row_index = int(ordered_indices[bar_index])
                tensor = self._required_tensor(retrieval_index, row_index)
                generated.append(np.asarray(mu[row_index], dtype=np.float32))
                tensors.append(tensor)
                selected_row_indices.append(row_index)
                row = rows[row_index]
                steps.append({
                    "bar_index": int(bar_index),
                    "source": "primer",
                    "row_index": row_index,
                    "song_id": str(row.get("song_id", "UNKNOWN")),
                    "source_base_id": str(retrieval_index.source_base_ids[row_index]),
                    "source_bar_index": int(row.get("bar_index", 0)),
                    "tensor_key": str(row.get("tensor_key", "")),
                    "action": self._action_name(row),
                    "position_id": int(row.get("bar_index", 0)) % max(1, int(self.config.position_vocab_size)),
                    "expected_row_index": expected_row_index,
                })
                continue

            source_row = rows[expected_row_index] if expected_row_index is not None else {}
            action_name = planner.action_name(bar_index, source_row)
            action_id = int(action_to_id.get(action_name, action_to_id.get(self.UNKNOWN_ACTION, 1)))
            position_id = int(bar_index) % max(1, int(self.config.position_vocab_size))
            query_mu = self._query_mu(generated)
            query_context_count = int(min(len(generated), int(self.config.query_context_bars)))
            allowed_song_ids, allowed_source_base_ids = self._allowed_scope_ids(seed_song_id, seed_base_id)
            selection = retrieval_index.select_memory(
                query_mu=query_mu,
                action_id=action_id,
                position_id=position_id,
                expected_row_index=expected_row_index,
                config=self.config.retrieval_config(),
                rng=rng,
                allowed_source_base_ids=allowed_source_base_ids,
                allowed_song_ids=allowed_song_ids,
            )
            if selection.tensor is None:
                selection.tensor = self._required_tensor(retrieval_index, int(selection.row_index))
            generated.append(np.asarray(selection.mu, dtype=np.float32))
            tensors.append(np.asarray(selection.tensor, dtype=np.float32))
            selected_row_indices.append(int(selection.row_index))
            steps.append({
                "bar_index": int(bar_index),
                "source": "oracle_selected",
                "generation_model": "nearest_neighbor_oracle",
                "action_id": int(action_id),
                "action": str(action_name),
                "position_id": int(position_id),
                "query_context_bars": query_context_count,
                "query_norm": float(np.linalg.norm(query_mu)),
                "source_scope": str(self.config.source_scope),
                **selection.diagnostics,
            })
        return (
            np.stack(generated, axis=0).astype(np.float32),
            np.stack(tensors, axis=0).astype(np.float32),
            selected_row_indices,
            steps,
        )

    def _generate_sequential(
        self,
        mu: np.ndarray,
        rows: Sequence[Dict[str, Any]],
        ordered_indices: Sequence[int],
        retrieval_index: OracleCandidateIndex,
    ) -> tuple[np.ndarray, np.ndarray, List[int], List[Dict[str, Any]]]:
        """Generate by copying the seed song's real bar order."""
        if not ordered_indices:
            raise ValueError("Sequential oracle requires at least one source song row.")
        generated: List[np.ndarray] = []
        tensors: List[np.ndarray] = []
        selected_row_indices: List[int] = []
        steps: List[Dict[str, Any]] = []
        total = int(self.config.bars)
        available = min(total, len(ordered_indices))
        for bar_index in range(available):
            row_index = int(ordered_indices[bar_index])
            row = rows[row_index]
            tensor = self._required_tensor(retrieval_index, row_index)
            generated.append(np.asarray(mu[row_index], dtype=np.float32))
            tensors.append(tensor)
            selected_row_indices.append(row_index)
            steps.append({
                "bar_index": int(bar_index),
                "source": "sequential_oracle",
                "generation_model": "sequential_oracle",
                "row_index": row_index,
                "song_id": str(row.get("song_id", "UNKNOWN")),
                "source_base_id": str(retrieval_index.source_base_ids[row_index]),
                "source_bar_index": int(row.get("bar_index", 0)),
                "tensor_key": str(row.get("tensor_key", "")),
                "action": self._action_name(row),
                "position_id": int(row.get("bar_index", 0)) % max(1, int(self.config.position_vocab_size)),
                "expected_row_index": row_index,
            })
        if available < total:
            raise ValueError(
                f"Sequential oracle requested {total} bars, but selected song only has {available} bars. "
                "Use fewer --bars or choose another --seed-song-id."
            )
        return (
            np.stack(generated, axis=0).astype(np.float32),
            np.stack(tensors, axis=0).astype(np.float32),
            selected_row_indices,
            steps,
        )

    def _query_mu(self, generated: Sequence[np.ndarray]) -> np.ndarray:
        """Return the latent query vector for the next nearest-neighbor lookup."""
        if not generated:
            raise ValueError("Oracle generation requires at least one primer bar.")
        count = max(1, int(self.config.query_context_bars))
        values = [np.asarray(item, dtype=np.float32) for item in generated[-count:]]
        return np.mean(np.stack(values, axis=0), axis=0).astype(np.float32)

    def _allowed_scope_ids(
        self,
        seed_song_id: Optional[str],
        seed_base_id: Optional[str],
    ) -> tuple[Optional[List[str]], Optional[List[str]]]:
        """Return source scope ids according to oracle mode."""
        mode = str(self.config.source_scope).strip().lower()
        if mode in {"free", "none", "all"}:
            return None, None
        if mode in {"same_song", "same_song_strict", "song"}:
            return ([str(seed_song_id)] if seed_song_id else None), None
        if mode == "same_base":
            return None, ([str(seed_base_id)] if seed_base_id else None)
        raise ValueError(f"Unsupported oracle source_scope: {self.config.source_scope}")

    def _required_tensor(self, retrieval_index: OracleCandidateIndex, row_index: int) -> np.ndarray:
        """Return a tensor for a row or fail with a clear message."""
        tensor = retrieval_index.tensor_for_row(row_index)
        if tensor is None:
            key = str(retrieval_index.rows[int(row_index)].get("tensor_key", ""))
            raise KeyError(f"Missing encoded bar tensor for row_index={row_index}, tensor_key={key!r}.")
        return np.asarray(tensor, dtype=np.float32)

    def _sequence_diagnostics(self, tensors: np.ndarray, steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize whole-sequence continuity and source consistency."""
        bars = np.asarray(tensors, dtype=np.float32)
        note_counts: List[int] = []
        active_counts: List[int] = []
        pitch_ranges: List[float] = []
        first_pitches: List[Optional[float]] = []
        last_pitches: List[Optional[float]] = []
        for bar in bars:
            note_mask = bar[..., 2] > 0.5
            active_mask = (bar[..., 2] > 0.5) | (bar[..., 3] > 0.5)
            pitches = bar[..., 0][note_mask] * 24.0 + float(self.config.base_pitch)
            note_counts.append(int(note_mask.sum()))
            active_counts.append(int(active_mask.sum()))
            pitch_ranges.append(float(np.max(pitches) - np.min(pitches)) if len(pitches) else 0.0)
            first_pitch, last_pitch = self._bar_boundary_pitches(bar)
            first_pitches.append(first_pitch)
            last_pitches.append(last_pitch)

        boundary_jumps: List[float] = []
        adjacent_bar_l2: List[float] = []
        for index in range(1, len(bars)):
            adjacent_bar_l2.append(float(np.linalg.norm(bars[index] - bars[index - 1])))
            if last_pitches[index - 1] is not None and first_pitches[index] is not None:
                jump = float(abs(float(first_pitches[index]) - float(last_pitches[index - 1])))
                boundary_jumps.append(jump)
                if index < len(steps):
                    steps[index]["boundary_jump_from_previous"] = jump
            elif index < len(steps):
                steps[index]["boundary_jump_from_previous"] = None

        selected_song_ids = [
            str(step.get("selected_song_id") or step.get("song_id"))
            for step in steps
            if step.get("source") in {"oracle_selected", "primer", "sequential_oracle"} and (step.get("selected_song_id") or step.get("song_id"))
        ]
        selected_source_base_ids = [
            str(step.get("selected_source_base_id") or step.get("source_base_id"))
            for step in steps
            if step.get("source") in {"oracle_selected", "primer", "sequential_oracle"} and (step.get("selected_source_base_id") or step.get("source_base_id"))
        ]
        expected_ranks = [
            int(step["expected_row_rank"])
            for step in steps
            if isinstance(step.get("expected_row_rank"), int)
        ]
        similarities = [
            float(step["selected_memory_similarity"])
            for step in steps
            if isinstance(step.get("selected_memory_similarity"), (int, float))
        ]
        return {
            "note_count": self._numeric_summary(note_counts),
            "active_slot_count": self._numeric_summary(active_counts),
            "pitch_range": self._numeric_summary(pitch_ranges),
            "adjacent_bar_l2": self._numeric_summary(adjacent_bar_l2),
            "boundary_jump_abs": self._numeric_summary(boundary_jumps),
            "boundary_jump_gt12_count": int(sum(1 for value in boundary_jumps if value > 12.0)),
            "boundary_jump_gt24_count": int(sum(1 for value in boundary_jumps if value > 24.0)),
            "selected_source_song_count": int(len(selected_song_ids)),
            "selected_unique_source_song_count": int(len(set(selected_song_ids))),
            "source_switch_count": self._switch_count(selected_song_ids),
            "selected_unique_source_base_count": int(len(set(selected_source_base_ids))),
            "source_base_switch_count": self._switch_count(selected_source_base_ids),
            "memory_similarity": self._numeric_summary(similarities),
            "expected_row_rank": self._numeric_summary(expected_ranks),
        }

    def _bar_boundary_pitches(self, bar: np.ndarray) -> tuple[Optional[float], Optional[float]]:
        """Return first and last note-on pitch for one bar tensor."""
        events: List[tuple[int, int, float]] = []
        for slot_index in range(bar.shape[1]):
            for track_index in range(bar.shape[0]):
                if float(bar[track_index, slot_index, 2]) > 0.5:
                    pitch = float(bar[track_index, slot_index, 0] * 24.0 + float(self.config.base_pitch))
                    events.append((int(slot_index), int(track_index), pitch))
        if not events:
            return None, None
        return float(events[0][2]), float(events[-1][2])

    def _numeric_summary(self, values: Sequence[float | int]) -> Dict[str, Any]:
        """Return compact numeric summary for diagnostics."""
        if not values:
            return {"n": 0}
        array = np.asarray(values, dtype=np.float64)
        return {
            "n": int(array.size),
            "mean": float(np.mean(array)),
            "median": float(np.median(array)),
            "min": float(np.min(array)),
            "max": float(np.max(array)),
        }

    def _switch_count(self, values: Sequence[str]) -> int:
        """Return adjacent switch count in a categorical sequence."""
        return int(sum(1 for index in range(1, len(values)) if values[index] != values[index - 1]))

    def _action_vocab(self, rows: Sequence[Dict[str, Any]]) -> Dict[str, int]:
        """Create a stable action vocabulary from latent metadata."""
        actions = {self._action_name(row) for row in rows}
        actions.add(str(self.config.default_action))
        ordered = [self.PAD_ACTION, self.UNKNOWN_ACTION]
        ordered.extend(action for action in sorted(actions) if action not in set(ordered))
        return {action: index for index, action in enumerate(ordered)}

    def _action_name(self, row: Dict[str, Any]) -> str:
        """Return normalized action name."""
        return str(row.get("action") or self.UNKNOWN_ACTION)

    def _group_rows(self, rows: Sequence[Dict[str, Any]]) -> Dict[str, List[int]]:
        """Group row indices by song_id and sort by bar_index."""
        grouped: Dict[str, List[int]] = {}
        for index, row in enumerate(rows):
            grouped.setdefault(str(row.get("song_id", "UNKNOWN")), []).append(index)
        return {
            song_id: sorted(indices, key=lambda idx: (int(rows[idx].get("bar_index", 0)), int(rows[idx].get("row_index", idx))))
            for song_id, indices in grouped.items()
        }

    def _select_song_id(self, grouped: Dict[str, List[int]], seed_song_id: Optional[str]) -> str:
        """Choose the source song for primer and same-base oracle scope."""
        if not grouped:
            raise ValueError("No latent rows available for oracle generation.")
        if seed_song_id:
            if seed_song_id in grouped:
                return seed_song_id
            pattern = re.compile(str(seed_song_id))
            matches = [song_id for song_id in grouped if pattern.search(song_id)]
            if matches:
                return sorted(matches)[0]
            raise ValueError(f"seed_song_id not found: {seed_song_id}")
        rng = random.Random(int(self.config.seed))
        candidates = sorted(grouped.keys())
        if str(self.config.selection_mode).strip().lower() == "sequential":
            long_enough = [song_id for song_id in candidates if len(grouped[song_id]) >= int(self.config.bars)]
            if not long_enough:
                raise ValueError(f"No song has enough bars for sequential oracle: requested {self.config.bars}.")
            candidates = long_enough
        return rng.choice(candidates)

    def _set_seed(self) -> None:
        """Seed Python and numpy RNGs."""
        seed = int(self.config.seed)
        random.seed(seed)
        np.random.seed(seed)
