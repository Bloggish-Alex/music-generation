"""Typed source-only performance-control observations for Codec V2."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

@dataclass(frozen=True)
class ControlAvailability:
    """Raw control availability without creating a generated control target."""
    cc64_available: bool
    cc64_intervals: tuple[tuple[float, float, int, int], ...] = ()
    unavailable_reason: str | None = None

@dataclass(frozen=True)
class PerformanceControls:
    """Format-neutral parser facts used only by diagnostics and evaluation."""
    tempo_bpm: tuple[tuple[float, float], ...] = ()
    key_signature: str | None = None
    key_confidence: float | None = None
    cc64: ControlAvailability = field(default_factory=lambda: ControlAvailability(False, (), "not_collected"))


def collect_controls(score: Any, source_path: Path) -> PerformanceControls:
    """Collect source-only tempo, key, and MIDI CC64 availability facts."""
    from music21 import key, tempo
    tempos = tuple((float(item.offset), float(item.number)) for item in score.recurse().getElementsByClass(tempo.MetronomeMark) if item.number is not None)
    keys = list(score.recurse().getElementsByClass(key.KeySignature))
    key_name = str(keys[0]) if keys else None
    if source_path.suffix.lower() not in {".mid", ".midi"}:
        return PerformanceControls(tempos, key_name, None, ControlAvailability(False, (), "format_cc64_unavailable"))
    try:
        return PerformanceControls(tempos, key_name, None, ControlAvailability(True, tuple(_cc64_intervals(source_path)), None))
    except ValueError as error:
        return PerformanceControls(tempos, key_name, None, ControlAvailability(False, (), str(error)))


def _cc64_intervals(path: Path) -> list[tuple[float, float, int, int]]:
    """Read sustain-pedal intervals from Standard MIDI control-change events."""
    data = path.read_bytes()
    if data[:4] != b"MThd":
        raise ValueError("midi_header_unavailable")
    header_size = int.from_bytes(data[4:8], "big")
    division = int.from_bytes(data[12:14], "big")
    if division <= 0 or division & 0x8000:
        raise ValueError("midi_division_unavailable")
    position, intervals, track_index = 8 + header_size, [], 0
    while position + 8 <= len(data) and data[position:position + 4] == b"MTrk":
        size = int.from_bytes(data[position + 4:position + 8], "big"); track = data[position + 8:position + 8 + size]; position += 8 + size
        cursor = tick = 0; running = None; pedal_start: dict[int, float] = {}
        while cursor < len(track):
            delta, cursor = _varlen(track, cursor); tick += delta
            if cursor >= len(track): break
            first = track[cursor]
            if first < 0x80:
                if running is None: break
                status = running
            else:
                status = first; cursor += 1
                if status < 0xF0: running = status
            if status == 0xFF:
                cursor += 1; length, cursor = _varlen(track, cursor); cursor += length; continue
            if status in {0xF0, 0xF7}:
                length, cursor = _varlen(track, cursor); cursor += length; continue
            width = 1 if status & 0xF0 in {0xC0, 0xD0} else 2
            if cursor + width > len(track): break
            values = track[cursor:cursor + width]; cursor += width
            if status & 0xF0 == 0xB0 and values[0] == 64:
                time = tick / division
                channel = status & 0x0F
                if values[1] >= 64 and channel not in pedal_start: pedal_start[channel] = time
                if values[1] < 64 and channel in pedal_start:
                    intervals.append((pedal_start.pop(channel), time, track_index, channel))
        for channel, start in pedal_start.items(): intervals.append((start, tick / division, track_index, channel))
        track_index += 1
    return intervals


def _varlen(data: bytes, position: int) -> tuple[int, int]:
    value = 0
    while position < len(data):
        byte = data[position]; position += 1; value = (value << 7) | (byte & 0x7F)
        if not byte & 0x80: return value, position
    raise ValueError("midi_truncated_variable_length")
