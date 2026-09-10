from __future__ import annotations

import mido

from data.performance_controls import collect_raw_smf_controls


def test_raw_smf_controls_keep_track_channel_state_and_close_terminal_pedals() -> None:
    midi = mido.MidiFile(ticks_per_beat=480)
    meta = mido.MidiTrack(); meta.extend([
        mido.MetaMessage("set_tempo", tempo=500000, time=0),
        mido.MetaMessage("key_signature", key="C", time=0),
        mido.MetaMessage("set_tempo", tempo=400000, time=240),
        mido.MetaMessage("key_signature", key="G", time=0),
    ])
    first = mido.MidiTrack(); first.extend([
        mido.Message("control_change", channel=0, control=64, value=127, time=120),
        mido.Message("control_change", channel=1, control=64, value=127, time=0),
        mido.Message("control_change", channel=0, control=64, value=0, time=120),
        mido.MetaMessage("end_of_track", time=120),
    ])
    second = mido.MidiTrack(); second.extend([
        mido.Message("control_change", channel=0, control=64, value=64, time=60),
        mido.Message("control_change", channel=0, control=64, value=0, time=60),
    ])
    midi.tracks.extend([meta, first, second])
    facts = collect_raw_smf_controls(midi)
    assert [event["absolute_tick"] for event in facts["tempo_events"]] == [0, 240]
    assert [event["key"] for event in facts["key_signature_events"]] == ["C", "G"]
    assert facts["cc64_event_count"] == 5
    intervals = facts["cc64_intervals"]
    assert {(item["physical_track_index"], item["channel"], item["start_tick"], item["end_tick"], item["end_rule"]) for item in intervals} == {(1, 0, 120, 240, "cc64_release"), (1, 1, 120, 360, "physical_track_end"), (2, 0, 60, 120, "cc64_release")}
    assert facts["cc64_diagnostics"]["unterminated_interval_count"] == 1


def test_raw_smf_controls_treat_no_cc64_as_readable_not_unavailable() -> None:
    midi = mido.MidiFile(ticks_per_beat=480); midi.tracks.append(mido.MidiTrack())
    facts = collect_raw_smf_controls(midi)
    assert facts["cc64_readable"] is True
    assert facts["cc64_present"] is False
    assert facts["cc64_event_count"] == 0
    assert facts["cc64_intervals"] == []
