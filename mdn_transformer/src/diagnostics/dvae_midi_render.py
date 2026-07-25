#!/usr/bin/env python3
"""Render DVAE target/reconstruction sample tensors to MIDI for listening tests."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


STATE_REST = 0
STATE_NOTE_ON = 1
STATE_HOLD = 2


@dataclass(frozen=True)
class DVAEMidiRenderConfig:
    """Configuration for rendering one-bar DVAE tensor samples."""

    ticks_per_beat: int = 480
    tempo_bpm: int = 120
    bar_length_ql: float = 4.0
    steps_per_bar: int = 16
    pitch_scale: float = 24.0
    default_base_pitch: int = 60
    default_velocity: int = 72
    min_pitch: int = 21
    max_pitch: int = 108
    gap_beats: float = 1.0
    audio_quality_enabled: bool = True


class TensorKeyParser:
    """Parse target/reconstruction sample keys produced by DVAE analysis."""

    SAMPLE_SUFFIX_RE = re.compile(r"__(target|reconstructed)$")

    def base_key(self, sample_key: str) -> str:
        """Return the original tensor key without target/reconstructed suffix."""
        return self.SAMPLE_SUFFIX_RE.sub("", str(sample_key))

    def sample_kind(self, sample_key: str) -> str:
        """Return target or reconstructed for a sample key."""
        match = self.SAMPLE_SUFFIX_RE.search(str(sample_key))
        return match.group(1) if match else "unknown"

    def safe_name(self, tensor_key: str) -> str:
        """Create a filesystem-safe short name from a tensor key."""
        value = re.sub(r"[^A-Za-z0-9_.+-]+", "_", str(tensor_key))
        return value[:180]


class TensorMidiRenderer:
    """Render a [3, 16, 16] tensor into scheduled MIDI note events."""

    def __init__(self, config: DVAEMidiRenderConfig) -> None:
        self.config = config

    def render_pair(self, target: np.ndarray, reconstructed: np.ndarray, base_pitch: Optional[int], output_path: str | Path) -> Dict[str, Any]:
        """Render target then reconstructed tensors into one A/B MIDI file."""
        import mido

        mid = mido.MidiFile(ticks_per_beat=int(self.config.ticks_per_beat))
        meta = mido.MidiTrack()
        meta.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(int(self.config.tempo_bpm)), time=0))
        meta.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
        mid.tracks.append(meta)
        base = int(base_pitch) if base_pitch is not None else int(self.config.default_base_pitch)
        target_events = self._tensor_events(target, base, start_tick=0)
        bar_ticks = self._bar_ticks()
        gap_ticks = int(round(float(self.config.gap_beats) * int(self.config.ticks_per_beat)))
        reconstructed_events = self._tensor_events(reconstructed, base, start_tick=bar_ticks + gap_ticks)
        diagnostics = {
            "base_pitch": int(base),
            "target": self._event_summary(target_events),
            "reconstructed": self._event_summary(reconstructed_events),
            "output_path": str(output_path),
        }
        for track_index in range(3):
            track = mido.MidiTrack()
            track.append(mido.MetaMessage("track_name", name=f"track_{track_index}", time=0))
            events = [
                event for event in [*target_events, *reconstructed_events]
                if int(event["track"]) == int(track_index)
            ]
            self._append_events(track, events)
            mid.tracks.append(track)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        mid.save(str(output_path))
        return diagnostics

    def _tensor_events(self, tensor: np.ndarray, base_pitch: int, start_tick: int) -> List[Dict[str, int]]:
        """Convert one tensor into absolute-tick note events."""
        events: List[Dict[str, int]] = []
        slot_ticks = self._slot_ticks()
        for track_index in range(min(3, int(tensor.shape[0]))):
            active_pitch: Optional[int] = None
            active_start: Optional[int] = None
            active_velocity = int(self.config.default_velocity)
            for slot_index in range(min(self.config.steps_per_bar, int(tensor.shape[1]))):
                features = tensor[track_index, slot_index]
                state = int(np.argmax(features[1:4]))
                current_tick = int(start_tick + slot_index * slot_ticks)
                if state == STATE_NOTE_ON:
                    if active_pitch is not None and active_start is not None:
                        events.extend(self._note_events(track_index, active_pitch, active_start, current_tick, active_velocity))
                    active_pitch = self._absolute_pitch(features, base_pitch)
                    active_start = current_tick
                    active_velocity = self._velocity(features)
                elif state == STATE_REST:
                    if active_pitch is not None and active_start is not None:
                        events.extend(self._note_events(track_index, active_pitch, active_start, current_tick, active_velocity))
                    active_pitch = None
                    active_start = None
                elif state == STATE_HOLD:
                    if active_pitch is None:
                        active_pitch = self._absolute_pitch(features, base_pitch)
                        active_start = current_tick
                        active_velocity = self._velocity(features)
            end_tick = int(start_tick + self._bar_ticks())
            if active_pitch is not None and active_start is not None:
                events.extend(self._note_events(track_index, active_pitch, active_start, end_tick, active_velocity))
        return sorted(events, key=lambda item: (int(item["tick"]), 0 if item["type"] == "off" else 1, int(item["pitch"])))

    def _absolute_pitch(self, features: np.ndarray, base_pitch: int) -> int:
        """Convert normalized relative pitch to MIDI pitch."""
        pitch = int(round(float(base_pitch) + float(features[0]) * float(self.config.pitch_scale)))
        return max(int(self.config.min_pitch), min(int(self.config.max_pitch), pitch))

    def _velocity(self, features: np.ndarray) -> int:
        """Convert normalized velocity to MIDI velocity."""
        velocity = int(round(float(features[4]) * 127.0))
        if velocity <= 0:
            velocity = int(self.config.default_velocity)
        return max(1, min(127, velocity))

    def _note_events(self, track: int, pitch: int, start: int, end: int, velocity: int) -> List[Dict[str, int]]:
        """Create note_on and note_off events if duration is positive."""
        if int(end) <= int(start):
            return []
        return [
            {"tick": int(start), "type": "on", "track": int(track), "pitch": int(pitch), "velocity": int(velocity)},
            {"tick": int(end), "type": "off", "track": int(track), "pitch": int(pitch), "velocity": 0},
        ]

    def _append_events(self, track: Any, events: Sequence[Dict[str, int]]) -> None:
        """Append sorted absolute-tick events to one MIDI track."""
        import mido

        previous_tick = 0
        for event in sorted(events, key=lambda item: (int(item["tick"]), 0 if item["type"] == "off" else 1)):
            tick = int(event["tick"])
            delta = max(0, tick - previous_tick)
            previous_tick = tick
            track.append(mido.Message(
                "note_on" if event["type"] == "on" else "note_off",
                note=int(event["pitch"]),
                velocity=int(event["velocity"]),
                time=delta,
            ))

    def _event_summary(self, events: Sequence[Dict[str, int]]) -> Dict[str, Any]:
        """Summarize rendered MIDI note events."""
        note_on = [event for event in events if event["type"] == "on"]
        pitches = [int(event["pitch"]) for event in note_on]
        return {
            "note_count": int(len(note_on)),
            "min_pitch": int(min(pitches)) if pitches else None,
            "max_pitch": int(max(pitches)) if pitches else None,
        }

    def _slot_ticks(self) -> int:
        """Return MIDI ticks per 16th-note slot."""
        return int(round(float(self.config.bar_length_ql) * int(self.config.ticks_per_beat) / int(self.config.steps_per_bar)))

    def _bar_ticks(self) -> int:
        """Return MIDI ticks per bar."""
        return int(round(float(self.config.bar_length_ql) * int(self.config.ticks_per_beat)))


class DVAESampleMidiBatchRenderer:
    """Render all sampled DVAE target/reconstructed pairs to MIDI files."""

    def __init__(self, config: DVAEMidiRenderConfig) -> None:
        self.config = config
        self.key_parser = TensorKeyParser()
        self.renderer = TensorMidiRenderer(config)

    def render(self, samples_path: str | Path, index_path: str | Path, output_dir: str | Path) -> Dict[str, Any]:
        """Render target/reconstructed sample pairs from a samples npz file."""
        samples = np.load(Path(samples_path))
        index = self._load_index(index_path)
        pairs = self._sample_pairs(samples)
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        rows = []
        for tensor_key, pair in pairs.items():
            if "target" not in pair or "reconstructed" not in pair:
                continue
            safe = self.key_parser.safe_name(tensor_key)
            midi_path = output / f"{safe}.target_vs_reconstructed.mid"
            base_pitch = self._base_pitch(index.get(tensor_key, {}))
            diagnostics = self.renderer.render_pair(pair["target"], pair["reconstructed"], base_pitch, midi_path)
            rows.append({
                "tensor_key": tensor_key,
                "midi_path": str(midi_path),
                **diagnostics,
            })
        report = {
            "samples_path": str(samples_path),
            "index_path": str(index_path),
            "output_dir": str(output),
            "rendered_count": int(len(rows)),
            "files": rows,
        }
        (output / "dvae_sample_midi_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    def _sample_pairs(self, samples: Any) -> Dict[str, Dict[str, np.ndarray]]:
        """Group target/reconstructed sample tensors by base tensor key."""
        pairs: Dict[str, Dict[str, np.ndarray]] = {}
        for sample_key in samples.files:
            tensor_key = self.key_parser.base_key(sample_key)
            kind = self.key_parser.sample_kind(sample_key)
            pairs.setdefault(tensor_key, {})[kind] = np.asarray(samples[sample_key], dtype=np.float32)
        return pairs

    def _load_index(self, index_path: str | Path) -> Dict[str, Dict[str, Any]]:
        """Load tensor index rows keyed by tensor_key."""
        rows = json.loads(Path(index_path).read_text(encoding="utf-8"))
        return {str(row["tensor_key"]): dict(row) for row in rows}

    def _base_pitch(self, index_row: Dict[str, Any]) -> Optional[int]:
        """Read base pitch from tensor diagnostics."""
        diagnostics = index_row.get("diagnostics", {}) if isinstance(index_row, dict) else {}
        value = diagnostics.get("base_pitch")
        return int(value) if value is not None else None
