#!/usr/bin/env python3
"""Training samples for next-token decoder models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Sequence

import numpy as np

from common.model_store import ModelBundle


@dataclass(frozen=True)
class TokenSequence:
    """One source song represented as observation IDs and latent vectors."""

    song_id: str
    file_path: str
    token_ids: List[int]
    latent_vectors: List[List[float]]
    state_ids: List[int]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SequenceWindow:
    """One supervised sample: previous tokens predict the next token."""

    song_id: str
    start_index: int
    context_token_ids: List[int]
    target_token_id: int
    context_latents: List[List[float]]
    target_state_id: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SequenceDatasetSummary:
    sequence_count: int
    window_count: int
    vocab_size: int
    latent_dim: int
    input_dim: int
    hidden_state_count: int
    condition_on_hidden_state: bool
    state_condition_mode: str
    context_size: int
    skipped_bar_count: int
    skipped_sequence_count: int
    min_sequence_length: int
    max_sequence_length: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DecoderSequenceDatasetBuilder:
    """Build LSTM decoder training windows from a saved model bundle."""

    def __init__(self, context_size: int = 8, condition_on_hidden_state: bool = True) -> None:
        if int(context_size) < 1:
            raise ValueError("context_size must be >= 1.")
        self.context_size = int(context_size)
        self.condition_on_hidden_state = bool(condition_on_hidden_state)

    def build(self, bundle: ModelBundle) -> tuple[List[TokenSequence], List[SequenceWindow], SequenceDatasetSummary]:
        sequences, skipped_bars = self._sequences(bundle)
        windows = self._windows(sequences)
        lengths = [len(sequence.token_ids) for sequence in sequences]
        latent_dim = len(sequences[0].latent_vectors[0]) if sequences and sequences[0].latent_vectors else 0
        hidden_state_count = self._hidden_state_count(bundle)
        summary = SequenceDatasetSummary(
            sequence_count=len(sequences),
            window_count=len(windows),
            vocab_size=len(bundle.symbol_vocabulary.symbol_to_descriptor),
            latent_dim=latent_dim,
            input_dim=latent_dim,
            hidden_state_count=hidden_state_count,
            condition_on_hidden_state=self.condition_on_hidden_state,
            state_condition_mode="output_embedding",
            context_size=self.context_size,
            skipped_bar_count=skipped_bars,
            skipped_sequence_count=sum(1 for sequence in sequences if len(sequence.token_ids) <= self.context_size),
            min_sequence_length=min(lengths) if lengths else 0,
            max_sequence_length=max(lengths) if lengths else 0,
        )
        return sequences, windows, summary

    def _sequences(self, bundle: ModelBundle) -> tuple[List[TokenSequence], int]:
        bars_by_song: Dict[tuple[str, str], list[Any]] = {}
        for pool in bundle.observation_to_bars.values():
            for bar in pool:
                bars_by_song.setdefault((bar.song_id, bar.file_path), []).append(bar)
        sequences: List[TokenSequence] = []
        skipped_bars = 0
        for (song_id, file_path), bars in sorted(bars_by_song.items(), key=lambda item: item[0]):
            token_ids: List[int] = []
            latent_vectors: List[List[float]] = []
            kept_state_ids: List[int] = []
            sorted_bars = sorted(bars, key=lambda item: int(item.bar_index))
            state_ids_by_index = self._state_ids_for_bars(bundle, sorted_bars)
            for bar, state_id in zip(sorted_bars, state_ids_by_index):
                if bar.observation_id is None:
                    skipped_bars += 1
                    continue
                latent = self._latent_for_token(bundle, int(bar.observation_id))
                if latent is None:
                    skipped_bars += 1
                    continue
                token_ids.append(int(bar.observation_id))
                latent_vectors.append(latent)
                kept_state_ids.append(int(state_id))
            if token_ids:
                sequences.append(TokenSequence(
                    song_id=song_id,
                    file_path=file_path,
                    token_ids=token_ids,
                    latent_vectors=latent_vectors,
                    state_ids=kept_state_ids,
                ))
        return sequences, skipped_bars

    def _latent_for_token(self, bundle: ModelBundle, token_id: int) -> List[float] | None:
        if bundle.encoder_model is None:
            return None
        entry = bundle.encoder_model.codebook_entry_for_symbol(int(token_id))
        if entry.latent_vector is not None:
            return [float(value) for value in entry.latent_vector]
        descriptor = bundle.encoder_model.descriptor_for_symbol(int(token_id))
        fallback = bundle.global_codebook.get(int(descriptor.codebook_id))
        if fallback is not None and fallback.latent_vector is not None:
            return [float(value) for value in fallback.latent_vector]
        return None

    def _windows(self, sequences: Sequence[TokenSequence]) -> List[SequenceWindow]:
        windows: List[SequenceWindow] = []
        for sequence in sequences:
            for target_index in range(self.context_size, len(sequence.token_ids)):
                start = target_index - self.context_size
                windows.append(SequenceWindow(
                    song_id=sequence.song_id,
                    start_index=start,
                    context_token_ids=sequence.token_ids[start:target_index],
                    target_token_id=int(sequence.token_ids[target_index]),
                    context_latents=sequence.latent_vectors[start:target_index],
                    target_state_id=int(sequence.state_ids[target_index]),
                ))
        return windows

    def _hidden_state_count(self, bundle: ModelBundle) -> int:
        counts = [
            int(getattr(model, "n_states", 0))
            for model in getattr(bundle, "form_models", {}).values()
        ]
        return max(counts) if counts else 0

    def _state_ids_for_bars(self, bundle: ModelBundle, bars: Sequence[Any]) -> List[int]:
        if not bars:
            return []
        form_name = str(getattr(bars[0], "form", None) or "ternary")
        model = bundle.form_models.get(form_name)
        if model is None and bundle.form_models:
            model = next(iter(bundle.form_models.values()))
        if model is None:
            return [0 for _ in bars]
        return self._state_ids_for_length(model, len(bars))

    def _state_ids_for_length(self, model: Any, length: int) -> List[int]:
        state_count = max(1, int(getattr(model, "n_states", 1)))
        section_lengths = [
            max(1, int(value))
            for value in list(getattr(model, "section_lengths", []) or [])
        ]
        if len(section_lengths) < state_count:
            section_lengths.extend([1] * (state_count - len(section_lengths)))
        section_lengths = section_lengths[:state_count]
        template_total = max(1, int(sum(section_lengths)))
        state_ids: List[int] = []
        assigned = 0
        for state_id, section_length in enumerate(section_lengths):
            if state_id == state_count - 1:
                count = max(0, int(length) - assigned)
            else:
                count = int(round(float(length) * float(section_length) / float(template_total)))
                count = max(0, min(count, int(length) - assigned))
            state_ids.extend([int(state_id)] * count)
            assigned += count
        while len(state_ids) < int(length):
            state_ids.append(state_count - 1)
        return state_ids[:int(length)]


class WindowTensorBuilder:
    """Convert sequence windows into numpy arrays."""

    def build(
        self,
        windows: Sequence[SequenceWindow],
        hidden_state_count: int = 0,
        condition_on_hidden_state: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not windows:
            return (
                np.zeros((0, 0, 0), dtype=np.float32),
                np.zeros((0,), dtype=np.int64),
            )
        rows = [window.context_latents for window in windows]
        inputs = np.asarray(rows, dtype=np.float32)
        targets = np.asarray([int(window.target_token_id) for window in windows], dtype=np.int64)
        return inputs, targets

    def target_states(self, windows: Sequence[SequenceWindow]) -> np.ndarray:
        return np.asarray([int(window.target_state_id) for window in windows], dtype=np.int64)
