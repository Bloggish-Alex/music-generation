"""Raw-SMF performance controls used only by diagnostics and evaluation."""
from __future__ import annotations

from typing import Any


def collect_raw_smf_controls(midi: Any) -> dict[str, Any]:
    """Collect authored tempo, key-signature and CC64 facts from a mido MIDI."""
    ppqn = int(midi.ticks_per_beat)
    if ppqn <= 0:
        raise ValueError("midi_ppqn_unavailable")
    tempo_events: list[dict[str, Any]] = []
    key_events: list[dict[str, Any]] = []
    intervals: list[dict[str, Any]] = []
    cc64_event_count = orphan_release_count = repeated_down_count = 0
    for track_index, track in enumerate(midi.tracks):
        absolute_tick = 0
        active: dict[int, tuple[int, int, int]] = {}
        for event_ordinal, message in enumerate(track):
            absolute_tick += int(message.time)
            common = {"absolute_tick": absolute_tick, "ql_offset": absolute_tick / ppqn, "physical_track_index": track_index, "event_ordinal": event_ordinal}
            if message.type == "set_tempo":
                import mido
                tempo_events.append({**common, "microseconds_per_beat": int(message.tempo), "bpm": float(mido.tempo2bpm(message.tempo))})
            elif message.type == "key_signature":
                key_events.append({**common, "key": str(message.key)})
            elif message.type == "control_change" and int(message.control) == 64:
                cc64_event_count += 1
                channel, value = int(message.channel), int(message.value)
                if value >= 64:
                    if channel in active:
                        repeated_down_count += 1
                    else:
                        active[channel] = (absolute_tick, event_ordinal, value)
                elif channel in active:
                    start_tick, start_ordinal, start_value = active.pop(channel)
                    intervals.append(_interval(track_index, channel, start_tick, absolute_tick, ppqn, start_ordinal, event_ordinal, start_value, value, "cc64_release"))
                else:
                    orphan_release_count += 1
        for channel, (start_tick, start_ordinal, start_value) in sorted(active.items()):
            intervals.append(_interval(track_index, channel, start_tick, absolute_tick, ppqn, start_ordinal, None, start_value, None, "physical_track_end"))
    return {
        "collector_version": "raw_smf_performance_controls.v1",
        "tempo_readable": True, "tempo_present": bool(tempo_events), "tempo_events": tempo_events,
        "key_readable": True, "key_present": bool(key_events), "key_signature_events": key_events,
        "cc64_readable": True, "cc64_present": cc64_event_count > 0, "cc64_event_count": cc64_event_count,
        "cc64_intervals": intervals,
        "cc64_diagnostics": {
            "unterminated_interval_count": sum(item["end_rule"] == "physical_track_end" for item in intervals),
            "orphan_release_count": orphan_release_count, "repeated_down_count": repeated_down_count,
            "unterminated_end_rule": "physical_track_end",
        },
    }


def _interval(track: int, channel: int, start: int, end: int, ppqn: int, start_ordinal: int, end_ordinal: int | None, start_value: int, end_value: int | None, end_rule: str) -> dict[str, Any]:
    return {"physical_track_index": track, "channel": channel, "start_tick": start, "end_tick": end, "start_ql": start / ppqn, "end_ql": end / ppqn, "start_event_ordinal": start_ordinal, "end_event_ordinal": end_ordinal, "start_value": start_value, "end_value": end_value, "end_rule": end_rule}
