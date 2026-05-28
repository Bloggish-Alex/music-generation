#!/usr/bin/env python3
"""Rendering constraints and MIDI writing for hierarchical generation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

from hierarchical_types import NoteEvent


def clamp_overlaps(all_notes: List[List[NoteEvent]], config: Dict) -> None:
    """Clamp melody durations so consecutive notes do not over-overlap."""
    max_ov = config.get("monophonic", {}).get("max_overlap", 0.15)
    min_dur = 0.1
    for mi in range(len(all_notes)):
        sounding = sorted(
            [n for n in all_notes[mi] if n.pitch >= 0 and n.voice == "melody"],
            key=lambda n: n.beat_offset,
        )
        for i in range(len(sounding) - 1):
            if sounding[i + 1].beat_offset - sounding[i].beat_offset < 0.02:
                continue
            cur_end = sounding[i].beat_offset + sounding[i].duration_ql
            allowed = sounding[i + 1].beat_offset + max_ov
            if cur_end > allowed:
                sounding[i] = NoteEvent(
                    pitch=sounding[i].pitch,
                    duration_ql=max(min_dur, allowed - sounding[i].beat_offset),
                    velocity=sounding[i].velocity,
                    beat_offset=sounding[i].beat_offset,
                    voice=sounding[i].voice,
                )
        others = [n for n in all_notes[mi] if n.pitch < 0 or n.voice != "melody"]
        all_notes[mi] = sorted(sounding + others, key=lambda n: n.beat_offset)


def clamp_measure_bounds(
    all_notes: List[List[NoteEvent]],
    time_signature: Tuple[int, int],
) -> None:
    """Clamp every event to its containing bar."""
    bar_length_ql = time_signature[0] * (4.0 / time_signature[1])
    min_dur = 0.05
    for mi, notes in enumerate(all_notes):
        bounded: List[NoteEvent] = []
        for note in notes:
            if note.beat_offset >= bar_length_ql - min_dur:
                continue
            max_duration = bar_length_ql - note.beat_offset
            duration = min(note.duration_ql, max_duration)
            if duration < min_dur:
                continue
            bounded.append(NoteEvent(
                pitch=note.pitch,
                duration_ql=duration,
                velocity=note.velocity,
                beat_offset=max(0.0, note.beat_offset),
                voice=note.voice,
            ))
        all_notes[mi] = sorted(bounded, key=lambda n: (n.beat_offset, n.pitch))


def ensure_final_bar_end(
    all_notes: List[List[NoteEvent]],
    time_signature: Tuple[int, int],
) -> None:
    """Guarantee the rendered MIDI reaches the requested final bar."""
    if not all_notes:
        return
    final_measure = all_notes[-1]
    sounding = [(idx, note) for idx, note in enumerate(final_measure) if note.pitch >= 0]
    if not sounding:
        return

    bar_length_ql = time_signature[0] * (4.0 / time_signature[1])
    idx, note = max(sounding, key=lambda pair: pair[1].beat_offset + pair[1].duration_ql)
    end = note.beat_offset + note.duration_ql
    if end >= bar_length_ql - 1e-6:
        return

    final_measure[idx] = NoteEvent(
        pitch=note.pitch,
        duration_ql=max(0.1, bar_length_ql - note.beat_offset),
        velocity=note.velocity,
        beat_offset=note.beat_offset,
        voice=note.voice,
    )


def write_midi(
    measures: List[List[NoteEvent]],
    output_path: Path,
    tempo: int,
    time_signature: Tuple[int, int],
) -> None:
    """Write MIDI via mido with direct tick-level control."""
    import mido

    ts_num, ts_den = time_signature
    bar_length_ql = ts_num * (4.0 / ts_den)
    tpb = 480
    us_per_beat = int(60_000_000 / tempo)

    mid = mido.MidiFile(ticks_per_beat=tpb)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    track.append(mido.MetaMessage('set_tempo', tempo=us_per_beat, time=0))
    track.append(mido.MetaMessage(
        'time_signature', numerator=ts_num, denominator=ts_den,
        clocks_per_click=24, notated_32nd_notes_per_beat=8, time=0,
    ))

    events: List[Tuple[int, str, int, int, int]] = []
    for measure_idx, nev_list in enumerate(measures):
        bar_base_ticks = measure_idx * bar_length_ql * tpb
        for nev in nev_list:
            if nev.pitch < 0:
                continue
            start_tick = int(round(bar_base_ticks + nev.beat_offset * tpb))
            end_tick = int(round(start_tick + nev.duration_ql * tpb))
            if end_tick <= start_tick:
                continue
            channel = 1 if nev.voice == "bass" else 0
            events.append((start_tick, 'on', nev.pitch, nev.velocity, channel))
            events.append((end_tick, 'off', nev.pitch, 0, channel))

    events.sort(key=lambda e: (e[0], 0 if e[1] == 'off' else 1))

    prev_tick = 0
    for tick, etype, pitch, velocity, channel in events:
        delta = tick - prev_tick
        if etype == 'on':
            track.append(mido.Message('note_on', note=pitch,
                                      velocity=velocity, channel=channel,
                                      time=delta))
        else:
            track.append(mido.Message('note_off', note=pitch,
                                      velocity=0, channel=channel,
                                      time=delta))
        prev_tick = tick

    mid.save(str(output_path))
