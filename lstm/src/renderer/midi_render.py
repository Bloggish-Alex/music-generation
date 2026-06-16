#!/usr/bin/env python3
"""Render selected BarRecord objects to MIDI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

from common.config_loader import ConfigView
from data.core_data import BarRecord


@dataclass(frozen=True)
class MidiRenderConfig:
    tempo: int = 120
    time_signature: str = "4/4"
    velocity: int = 72
    channel: int = 0


class MidiRenderer:
    """Write BarRecord note content to a simple monophonic/polyphonic MIDI track."""

    def __init__(self, config: MidiRenderConfig) -> None:
        self.config = config

    @classmethod
    def from_style_config(cls, config: Dict) -> "MidiRenderer":
        section = ConfigView(config).section("midi_render")
        return cls(MidiRenderConfig(
            tempo=int(section.get("tempo", 120)),
            time_signature=str(section.get("time_signature", "4/4")),
            velocity=int(section.get("velocity", 72)),
            channel=int(section.get("channel", 0)),
        ))

    def write(self, bars: Sequence[BarRecord], output_path: str | Path) -> Dict:
        import mido

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ts_num, ts_den = self._parse_time_signature(self.config.time_signature)
        ticks_per_beat = 480
        bar_length_ql = ts_num * (4.0 / ts_den)
        us_per_beat = int(60_000_000 / self.config.tempo)

        mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
        track = mido.MidiTrack()
        mid.tracks.append(track)
        track.append(mido.MetaMessage("set_tempo", tempo=us_per_beat, time=0))
        track.append(mido.MetaMessage(
            "time_signature",
            numerator=ts_num,
            denominator=ts_den,
            clocks_per_click=24,
            notated_32nd_notes_per_beat=8,
            time=0,
        ))

        events: List[tuple[int, str, int, int]] = []
        for output_bar_index, bar in enumerate(bars):
            for note in bar.notes:
                start_ql = output_bar_index * bar_length_ql + note.onset_ql
                end_ql = start_ql + note.duration_ql
                start_tick = int(round(start_ql * ticks_per_beat))
                end_tick = int(round(end_ql * ticks_per_beat))
                if end_tick > start_tick:
                    events.append((start_tick, "on", int(note.pitch), int(note.velocity or self.config.velocity)))
                    events.append((end_tick, "off", int(note.pitch), 0))
        events.sort(key=lambda item: (item[0], 0 if item[1] == "off" else 1))
        previous_tick = 0
        for tick, kind, pitch, velocity in events:
            delta = max(0, tick - previous_tick)
            message = "note_on" if kind == "on" else "note_off"
            track.append(mido.Message(
                message,
                note=pitch,
                velocity=velocity,
                channel=self.config.channel,
                time=delta,
            ))
            previous_tick = tick
        mid.save(str(output_path))
        return {
            "output_path": str(output_path),
            "bar_count": len(bars),
            "event_count": len(events),
            "tempo": self.config.tempo,
            "time_signature": self.config.time_signature,
        }

    def _parse_time_signature(self, value: str) -> tuple[int, int]:
        parts = value.split("/")
        return int(parts[0]), int(parts[1])
