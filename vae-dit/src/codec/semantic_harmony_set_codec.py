"""Lossless slot-local melody, harmony-set, and bass bar codec."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from codec.relative_chroma import bass_anchor_pitch, relative_chromagram
from common.config_loader import ConfigView
from data.core import BarRecord, BarTensorRecord, NoteEvent, SongRecord
from codec.semantic_harmony_assignment import SemanticCodecSequenceState, assign
from codec.slot_grid import CAPACITY, SlotGrid


VOICE_NAMES = ["melody", *[f"harmony_{index:02d}" for index in range(16)], "bass"]
FEATURE_NAMES = ["relative_pitch", "is_rest", "is_note_on", "is_hold", "normalized_velocity", "velocity_ratio"]


@dataclass(frozen=True)
class SemanticHarmonySetConfig:
    steps_per_bar: int = 48
    pitch_scale: float = 24.0
    velocity_scale: float = 127.0
    max_harmony_notes: int = 16
    relative_pitch_max_semitones: float = 96.0
    melody_continuity_tolerance: int = 7
    slot_time_epsilon_ql: float = 1.0e-6

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "SemanticHarmonySetConfig":
        section = ConfigView(dict(config)).section("bar_tensor")
        if section.get("schema_version") != "bar_tensor_schema.v2" or section.get("overflow_policy") != "error":
            raise ValueError("semantic_harmony_set_v2 requires bar_tensor_schema.v2 and overflow_policy=error")
        result = cls(**{name: section.get(name, getattr(cls(), name)) for name in cls.__dataclass_fields__})
        if result.max_harmony_notes != 16 or result.steps_per_bar != CAPACITY or result.pitch_scale <= 0:
            raise ValueError("semantic_harmony_set_v2 configuration is invalid")
        return result


class SemanticHarmonySetCodec:
    """Encode every active source note into deterministic V2 semantic lanes."""

    def __init__(self, config: SemanticHarmonySetConfig) -> None:
        self.config = config

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "SemanticHarmonySetCodec":
        return cls(SemanticHarmonySetConfig.from_config(config))

    def encode_song(self, song: SongRecord) -> list[BarTensorRecord]:
        state = SemanticCodecSequenceState()
        return [self.encode(bar, state) for bar in song.bars]

    def encode(self, bar: BarRecord, state: SemanticCodecSequenceState | None = None) -> BarTensorRecord:
        notes = [note for track in bar.tracks for note in track.notes]
        base_pitch = bass_anchor_pitch(notes)
        grid = SlotGrid.for_bar(float(bar.bar_length_ql))
        tensor = np.zeros((18, self.config.steps_per_bar, 6), dtype=np.float32)
        tensor[:, np.asarray(grid.slot_valid_mask), 1] = 1.0
        if base_pitch is not None and max(int(note.pitch) for note in notes) - base_pitch > self.config.relative_pitch_max_semitones:
            raise ValueError("relative_pitch_range_overflow")
        previous: NoteEvent | None = None
        for slot in range(grid.valid_slot_count):
            start, end = grid.interval(slot)
            epsilon = self.config.slot_time_epsilon_ql
            active = [note for note in notes if note.onset_ql < end - epsilon and note.onset_ql + note.duration_ql > start + epsilon]
            if not active:
                continue
            prior = state.previous_note(active, bar.source_measure_index) if state is not None and slot == 0 else previous
            melody,bass,harmony = assign(active, prior, self.config.melody_continuity_tolerance)
            if len(harmony) > self.config.max_harmony_notes:
                raise ValueError(f"harmony_lane_overflow: song={bar.song_id} bar={bar.bar_index} slot={slot} count={len(harmony)}")
            assigned = [(0, melody), *[(index + 1, note) for index, note in enumerate(harmony)]]
            if bass is not None:
                assigned.append((17, bass))
            denominator = sum(max(0, int(note.velocity)) for _, note in assigned)
            for lane, note in assigned:
                self._write(tensor[lane, slot], note, base_pitch, start, denominator)
            previous = melody
        if state is not None:
            state.update(previous, bar.source_measure_index)
        context = relative_chromagram(notes, base_pitch, velocity_scale=self.config.velocity_scale)
        diagnostics = {"codec_backend": "semantic_harmony_set_v2", "schema_version": "bar_tensor_schema.v2", "base_pitch": base_pitch, "base_pitch_valid": base_pitch is not None, "bar_context": context.tolist() if base_pitch is not None else [0.0] * 12, "voice_names": VOICE_NAMES, "feature_names": FEATURE_NAMES, "slot_valid_mask": list(grid.slot_valid_mask), "slot_durations_ql": list(grid.slot_durations_ql)}
        return BarTensorRecord(bar.song_id, int(bar.bar_index), list(tensor.shape), tensor, diagnostics)

    def _write(self, features: np.ndarray, note: NoteEvent, base_pitch: int | None, slot_start: float, velocity_sum: int) -> None:
        features[:] = 0.0
        features[0] = 0.0 if base_pitch is None else (int(note.pitch) - base_pitch) / self.config.pitch_scale
        is_bar_continuation = note.continues_from_previous_bar and abs(slot_start) <= self.config.slot_time_epsilon_ql
        features[2] = 1.0 if not is_bar_continuation and abs(float(note.onset_ql) - slot_start) <= self.config.slot_time_epsilon_ql else 0.0
        features[3] = 1.0 - features[2]
        velocity = max(0.0, min(float(note.velocity), self.config.velocity_scale))
        features[4] = velocity / self.config.velocity_scale
        features[5] = velocity / velocity_sum if velocity_sum else 0.0
