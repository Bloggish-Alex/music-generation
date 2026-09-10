from __future__ import annotations

import numpy as np
import pytest

from codec.semantic_harmony_set_codec import SemanticHarmonySetCodec
from codec.slot_grid import CodecCapacityError
from data.core import BarRecord, NoteEvent, SongRecord, TrackRecord


def _config(capacity: int = 48) -> dict:
    return {"bar_tensor": {"backend": "semantic_harmony_set_v2", "schema_version": "bar_tensor_schema.v2", "overflow_policy": "error", "slot_grid": {"quantum_ql": .25, "capacity": capacity, "epsilon_ql": 1e-6}, "pitch_scale": 24.0, "velocity_scale": 127.0, "max_harmony_notes": 16, "relative_pitch_max_semitones": 96.0}}


@pytest.mark.parametrize(("meter", "length", "valid"), [("20/4", 20.0, 80), ("41/8", 20.5, 82), ("4/4", 4.0, 16)])
def test_configured_92_slot_capacity_is_run_fixed(meter, length, valid) -> None:
    bar = BarRecord("song", "fixture.mid", 7, length, time_signature=meter, canonical_bar_index=7, canonical_start_tick=10, canonical_end_tick=20)
    record = SemanticHarmonySetCodec.from_config(_config(92)).encode(bar)
    assert record.tensor.shape == (18, 92, 6)
    assert record.diagnostics["slot_valid_mask"][:valid] == [True] * valid
    assert not record.tensor[:, valid:, :].any()


@pytest.mark.parametrize(("capacity", "length", "required"), [(92, 23.25, 93), (48, 20.0, 80)])
def test_capacity_overflow_is_structured(capacity, length, required) -> None:
    bar = BarRecord("song", "fixture.mid", 7, length, time_signature="20/4", canonical_bar_index=7, canonical_start_tick=10, canonical_end_tick=20)
    song = SongRecord("song", "fixture.mid", metadata={"source_file_identity": "identity", "tune_index": 2}, bars=[bar])
    with pytest.raises(CodecCapacityError) as caught:
        SemanticHarmonySetCodec.from_config(_config(capacity)).encode_song(song)
    assert caught.value.details == {"reason": "slot_capacity_exceeded", "song_id": "song", "source_file_identity": "identity", "file_path": "fixture.mid", "tune_index": 2, "canonical_bar_index": 7, "canonical_start_tick": 10, "canonical_end_tick": 20, "meter": "20/4", "bar_length_ql": length, "quantum_ql": .25, "required_slot_count": required, "configured_slot_capacity": capacity}


def _note(pitch: int, ordinal: int, track: int = 0) -> NoteEvent:
    return NoteEvent(pitch=pitch, onset_ql=0.0, duration_ql=1.0, velocity=80, source_file_identity="fixture", physical_track_index=track, source_note_ordinal=ordinal, source_onset_ql=0.0)


def test_v2_preserves_duplicate_pitch_notes_in_distinct_harmony_lanes() -> None:
    bar = BarRecord("song", "fixture.mid", 0, 4.0, tracks=[TrackRecord(0, "track", [_note(72, 0), _note(60, 1), _note(64, 2, 1), _note(64, 3, 2)])])
    record = SemanticHarmonySetCodec.from_config(_config()).encode(bar)
    assert record.tensor.shape == (18, 48, 6)
    assert np.count_nonzero(record.tensor[1:17, 0, 1] < 0.5) == 2
    assert record.diagnostics["base_pitch"] == 60
    assert record.diagnostics["bar_context"] and len(record.diagnostics["bar_context"]) == 12
    assert record.diagnostics["slot_valid_mask"][:16] == [True] * 16
    assert record.diagnostics["slot_valid_mask"][16:] == [False] * 32
    assert not record.tensor[:, 16:, 1].any()  # padding is neither rest nor active


def test_v2_rejects_a_pitch_span_above_96_semitones() -> None:
    bar = BarRecord("song", "fixture.mid", 0, 4.0, tracks=[TrackRecord(0, "track", [_note(0, 0), _note(97, 1)])])
    with pytest.raises(ValueError, match="relative_pitch_range_overflow"):
        SemanticHarmonySetCodec.from_config(_config()).encode(bar)


def test_v2_uses_bar_local_onset_for_a_later_bar() -> None:
    note = _note(72, 0); note.source_onset_ql = 4.0
    bar = BarRecord("song", "fixture.mid", 1, 4.0, tracks=[TrackRecord(0, "track", [note])])
    record = SemanticHarmonySetCodec.from_config(_config()).encode(bar)
    assert record.tensor[0, 0, 2] == 1.0
    assert record.tensor[0, 0, 3] == 0.0


def test_v2_preserves_cross_bar_melody_identity_and_marks_continuation_hold() -> None:
    continued = _note(72, 0)
    continued.duration_ql = 4.0
    continued.continues_into_next_bar = True
    bar0 = BarRecord("song", "fixture.mid", 0, 4.0, source_measure_index=91, canonical_bar_index=0, tracks=[TrackRecord(0, "track", [continued])])
    next_bar_continued = _note(72, 0)
    next_bar_continued.duration_ql = 1.0
    next_bar_continued.continues_from_previous_bar = True
    # A new higher note would win without the frozen seven-semitone continuity rule.
    new_note = _note(76, 1)
    bar1 = BarRecord("song", "fixture.mid", 1, 4.0, source_measure_index=12, canonical_bar_index=1, tracks=[TrackRecord(0, "track", [next_bar_continued, new_note])])
    records = SemanticHarmonySetCodec.from_config(_config()).encode_song(SongRecord("song", "fixture.mid", bars=[bar0, bar1]))
    assert records[1].tensor[0, 0, 0] == records[0].tensor[0, 0, 0]
    assert records[1].tensor[0, 0, 2] == 0.0
    assert records[1].tensor[0, 0, 3] == 1.0


def test_v2_resets_melody_continuity_across_a_canonical_bar_gap() -> None:
    continued = _note(72, 0)
    continued.continues_into_next_bar = True
    first = BarRecord("song", "fixture.mid", 0, 4.0, canonical_bar_index=0, tracks=[TrackRecord(0, "track", [continued])])
    held = _note(72, 0)
    held.continues_from_previous_bar = True
    higher = _note(76, 1)
    gapped = BarRecord("song", "fixture.mid", 2, 4.0, canonical_bar_index=2, tracks=[TrackRecord(0, "track", [held, higher])])

    records = SemanticHarmonySetCodec.from_config(_config()).encode_song(SongRecord("song", "fixture.mid", bars=[first, gapped]))

    assert records[1].tensor[0, 0, 0] != records[0].tensor[0, 0, 0]


def test_v2_velocity_ratio_uses_clamped_assigned_velocities() -> None:
    high = _note(72, 0); high.velocity = 200
    bass = _note(60, 1); bass.velocity = 127
    bar = BarRecord("song", "fixture.mid", 0, 4.0, tracks=[TrackRecord(0, "track", [high, bass])])
    tensor = SemanticHarmonySetCodec.from_config(_config()).encode(bar).tensor
    ratios = tensor[[0, 17], 0, 5]
    assert np.allclose(ratios, [.5, .5]) and np.isclose(ratios.sum(), 1.0)
    high.velocity = bass.velocity = 0
    zeros = SemanticHarmonySetCodec.from_config(_config()).encode(bar).tensor[:, 0, 5]
    assert np.isfinite(zeros).all() and np.all(zeros == 0.0)
