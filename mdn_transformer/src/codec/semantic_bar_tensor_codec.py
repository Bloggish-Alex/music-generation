#!/usr/bin/env python3
"""Semantic Melody/Harmony/Bass bar tensor codec."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import numpy as np

from codec.bar_tensor_codec import BarTensorConfig, FixedChromagramProjector
from common.config_loader import ConfigView
from data.core import BarRecord, BarTensorRecord, NoteEvent


SEMANTIC_FEATURE_NAMES = [
    "relative_pitch",
    "is_rest",
    "is_note_on",
    "is_hold",
    "normalized_velocity",
    "velocity_ratio",
    "density_gradient",
    "relative_chroma_embed_00",
    "relative_chroma_embed_01",
    "relative_chroma_embed_02",
    "relative_chroma_embed_03",
    "relative_chroma_embed_04",
    "relative_chroma_embed_05",
    "relative_chroma_embed_06",
    "relative_chroma_embed_07",
    "relative_chroma_embed_08",
    "relative_chroma_embed_09",
    "relative_chroma_embed_10",
]

SEMANTIC_TRACK_NAMES = ["melody", "harmony", "bass"]


@dataclass(frozen=True)
class SemanticBarTensorConfig(BarTensorConfig):
    """Configuration for semantic 3-voice bar tensor encoding."""

    melody_continuity_tolerance: int = 7
    harmony_velocity_mode: str = "max"

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SemanticBarTensorConfig":
        """Build semantic codec configuration from style config."""
        section = ConfigView(config).section("bar_tensor")
        return cls(
            max_tracks=int(section.get("max_tracks", 3)),
            steps_per_bar=int(section.get("steps_per_bar", 16)),
            feature_dim=int(section.get("feature_dim", len(SEMANTIC_FEATURE_NAMES))),
            pitch_scale=float(section.get("pitch_scale", 24.0)),
            velocity_scale=float(section.get("velocity_scale", 127.0)),
            rest_feature_value=float(section.get("rest_feature_value", 1.0)),
            melody_continuity_tolerance=int(section.get("melody_continuity_tolerance", 7)),
            harmony_velocity_mode=str(section.get("harmony_velocity_mode", "max")),
        )


@dataclass
class SlotVoice:
    """One semantic voice value for a slot."""

    notes: List[NoteEvent]
    onset_notes: List[NoteEvent]

    @property
    def active(self) -> bool:
        """Return whether the voice has any active note."""
        return bool(self.notes)


class SemanticBarTensorCodec:
    """Encode bars as stable Melody/Harmony/Bass tensors."""

    def __init__(self, config: SemanticBarTensorConfig, projector: FixedChromagramProjector | None = None) -> None:
        self.config = config
        self.projector = projector or FixedChromagramProjector()

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SemanticBarTensorCodec":
        """Create a semantic codec from full style config."""
        return cls(SemanticBarTensorConfig.from_config(config))

    def encode(self, bar: BarRecord) -> BarTensorRecord:
        """Encode one bar into a [3, 16, 18] semantic tensor record."""
        if self.config.max_tracks != 3:
            raise ValueError("semantic_3voice requires bar_tensor.max_tracks=3.")
        if self.config.feature_dim != len(SEMANTIC_FEATURE_NAMES):
            raise ValueError(f"semantic_3voice feature_dim must be {len(SEMANTIC_FEATURE_NAMES)}.")

        tensor = np.zeros(
            (self.config.max_tracks, self.config.steps_per_bar, self.config.feature_dim),
            dtype=np.float32,
        )
        notes = bar.all_notes()
        base_pitch = self._bar_base_pitch(notes)
        relative_chroma = self._relative_chromagram(notes, base_pitch)
        chroma_embedding = self.projector.project(relative_chroma)
        slots = self._semantic_slots(bar, notes)
        densities = self._voice_densities(slots)
        gradients = self._density_gradients(densities)
        for slot_index, voices in enumerate(slots):
            velocity_sum = sum(self._voice_velocity(voice) for voice in voices)
            for track_index, voice in enumerate(voices):
                self._write_voice(
                    tensor[track_index, slot_index],
                    voice,
                    base_pitch,
                    chroma_embedding,
                    velocity_sum,
                    gradients[track_index],
                    track_index,
                )

        diagnostics = self._diagnostics(bar, tensor, base_pitch, relative_chroma, chroma_embedding, densities, slots)
        return BarTensorRecord(
            song_id=bar.song_id,
            bar_index=int(bar.bar_index),
            tensor_shape=[int(value) for value in tensor.shape],
            tensor=tensor,
            diagnostics=diagnostics,
        )

    def _semantic_slots(self, bar: BarRecord, notes: Sequence[NoteEvent]) -> List[List[SlotVoice]]:
        """Return slot-wise Melody/Harmony/Bass voice assignments."""
        slot_len = float(bar.bar_length_ql) / float(self.config.steps_per_bar)
        slots: List[List[SlotVoice]] = []
        previous_melody: NoteEvent | None = None
        for slot_index in range(self.config.steps_per_bar):
            slot_start = float(slot_index) * slot_len
            slot_end = slot_start + slot_len
            active = self._active_notes(notes, slot_start, slot_end)
            onset = self._onset_notes(notes, slot_start, slot_end)
            melody = self._melody_note(active, previous_melody)
            bass = self._bass_note(active, melody)
            excluded = {id(note) for note in [melody, bass] if note is not None}
            harmony = [note for note in active if id(note) not in excluded]
            voices = [
                SlotVoice(
                    notes=[melody] if melody is not None else [],
                    onset_notes=[note for note in onset if melody is not None and id(note) == id(melody)],
                ),
                SlotVoice(
                    notes=harmony,
                    onset_notes=[note for note in onset if id(note) in {id(item) for item in harmony}],
                ),
                SlotVoice(
                    notes=[bass] if bass is not None else [],
                    onset_notes=[note for note in onset if bass is not None and id(note) == id(bass)],
                ),
            ]
            slots.append(voices)
            if melody is not None:
                previous_melody = melody
        return slots

    def _melody_note(self, active: Sequence[NoteEvent], previous: NoteEvent | None) -> NoteEvent | None:
        """Choose the melody note with a small continuity preference."""
        if not active:
            return None
        highest = max(active, key=lambda note: (int(note.pitch), int(note.velocity)))
        if previous is None:
            return highest
        for note in active:
            if id(note) == id(previous) and int(note.pitch) >= int(highest.pitch) - int(self.config.melody_continuity_tolerance):
                return note
        return highest

    def _bass_note(self, active: Sequence[NoteEvent], melody: NoteEvent | None) -> NoteEvent | None:
        """Choose the bass note after reserving the melody note."""
        candidates = [note for note in active if melody is None or id(note) != id(melody)]
        if not candidates:
            return None
        return min(candidates, key=lambda note: (int(note.pitch), -int(note.velocity)))

    def _write_voice(
        self,
        features: np.ndarray,
        voice: SlotVoice,
        base_pitch: int | None,
        chroma_embedding: np.ndarray,
        slot_velocity_sum: float,
        density_gradient: float,
        track_index: int,
    ) -> None:
        """Write one semantic voice into one slot feature vector."""
        features[5] = 0.0
        features[6] = float(density_gradient)
        features[7:18] = chroma_embedding
        if not voice.active:
            features[0] = 0.0
            features[1] = self.config.rest_feature_value
            features[2] = 0.0
            features[3] = 0.0
            features[4] = 0.0
            return
        velocity = self._voice_velocity(voice)
        features[0] = self._voice_relative_pitch(voice, base_pitch, track_index)
        features[1] = 0.0
        features[2] = 1.0 if voice.onset_notes else 0.0
        features[3] = 0.0 if voice.onset_notes else 1.0
        features[4] = self._normalized_velocity(velocity)
        features[5] = float(velocity / slot_velocity_sum) if slot_velocity_sum > 0.0 else 0.0

    def _voice_relative_pitch(self, voice: SlotVoice, base_pitch: int | None, track_index: int) -> float:
        """Return normalized relative pitch for a semantic voice."""
        if base_pitch is None or not voice.notes:
            return 0.0
        if track_index == 1:
            pitch = float(np.mean([int(note.pitch) for note in voice.notes]))
        else:
            pitch = float(voice.notes[0].pitch)
        return float(pitch - int(base_pitch)) / float(self.config.pitch_scale)

    def _voice_velocity(self, voice: SlotVoice) -> float:
        """Return representative velocity for a semantic voice."""
        if not voice.notes:
            return 0.0
        velocities = [int(note.velocity) for note in voice.notes]
        if str(self.config.harmony_velocity_mode).lower() == "mean":
            return float(np.mean(velocities))
        return float(max(velocities))

    def _voice_densities(self, slots: Sequence[Sequence[SlotVoice]]) -> np.ndarray:
        """Return active-slot density for Melody/Harmony/Bass."""
        values = np.zeros(3, dtype=np.float32)
        if not slots:
            return values
        for voices in slots:
            for track_index, voice in enumerate(voices):
                if voice.active:
                    values[track_index] += 1.0
        return values / float(len(slots))

    def _density_gradients(self, densities: np.ndarray) -> np.ndarray:
        """Return track density minus the other tracks' mean density."""
        gradients = np.zeros(3, dtype=np.float32)
        for index in range(3):
            others = [float(densities[item]) for item in range(3) if item != index]
            gradients[index] = float(densities[index]) - float(np.mean(others))
        return gradients

    def _active_notes(self, notes: Sequence[NoteEvent], slot_start: float, slot_end: float) -> List[NoteEvent]:
        """Return notes active inside a slot."""
        return [
            note for note in notes
            if float(note.onset_ql) < slot_end and float(note.onset_ql) + float(note.duration_ql) > slot_start
        ]

    def _onset_notes(self, notes: Sequence[NoteEvent], slot_start: float, slot_end: float) -> List[NoteEvent]:
        """Return notes that start inside a slot."""
        return [note for note in notes if slot_start <= float(note.onset_ql) < slot_end]

    def _bar_base_pitch(self, notes: Sequence[NoteEvent]) -> int | None:
        """Choose a stable base pitch for relative-pitch and chroma rotation."""
        if not notes:
            return None
        pitches = np.asarray([int(note.pitch) for note in notes], dtype=np.int32)
        return int(np.median(pitches))

    def _relative_chromagram(self, notes: Sequence[NoteEvent], base_pitch: int | None) -> np.ndarray:
        """Compute base-pitch-relative 12-bin pitch-class profile."""
        values = np.zeros(12, dtype=np.float32)
        if base_pitch is None:
            return values
        base_pc = int(base_pitch) % 12
        for note in notes:
            values[(int(note.pitch) - base_pc) % 12] += max(0.0, float(note.duration_ql))
        total = float(values.sum())
        if total <= 0.0:
            return values
        return values / total

    def _normalized_velocity(self, velocity: float) -> float:
        """Normalize MIDI velocity into [0, 1]."""
        return float(max(0.0, min(float(velocity), float(self.config.velocity_scale)))) / float(self.config.velocity_scale)

    def _diagnostics(
        self,
        bar: BarRecord,
        tensor: np.ndarray,
        base_pitch: int | None,
        relative_chroma: np.ndarray,
        chroma_embedding: np.ndarray,
        densities: np.ndarray,
        slots: Sequence[Sequence[SlotVoice]],
    ) -> Dict[str, Any]:
        """Return JSON-safe semantic codec diagnostics."""
        note_on = tensor[..., 2] > 0.5
        active = note_on | (tensor[..., 3] > 0.5)
        harmony_polyphonic = sum(1 for voices in slots if len(voices[1].notes) > 1)
        track_rows = []
        for index, name in enumerate(SEMANTIC_TRACK_NAMES):
            pitches = tensor[index, active[index], 0] if bool(active[index].any()) else np.asarray([], dtype=np.float32)
            track_rows.append({
                "track_index": int(index),
                "name": name,
                "active_slot_count": int(active[index].sum()),
                "note_on_slot_count": int(note_on[index].sum()),
                "active_density": float(densities[index]),
                "relative_pitch_min": float(np.min(pitches)) if pitches.size else 0.0,
                "relative_pitch_max": float(np.max(pitches)) if pitches.size else 0.0,
            })
        return {
            "codec_backend": "semantic_3voice",
            "song_id": bar.song_id,
            "bar_index": int(bar.bar_index),
            "shape": [int(value) for value in tensor.shape],
            "feature_names": list(SEMANTIC_FEATURE_NAMES),
            "track_names": list(SEMANTIC_TRACK_NAMES),
            "base_pitch": int(base_pitch) if base_pitch is not None else None,
            "relative_chromagram": [float(value) for value in relative_chroma.tolist()],
            "relative_chroma_embedding": [float(value) for value in chroma_embedding.tolist()],
            "physical_track_count": int(len(bar.tracks)),
            "semantic_tracks": track_rows,
            "harmony_polyphonic_slot_count": int(harmony_polyphonic),
            "empty_semantic_track_count": int(sum(1 for item in track_rows if int(item["active_slot_count"]) == 0)),
        }
