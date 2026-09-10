"""Parser-path regression tests for the raw-SMF Codec V2 authority."""

from __future__ import annotations

from pathlib import Path

from data.music_parser import MusicDirectoryParser, MusicParserConfig
from codec.semantic_harmony_set_codec import SemanticHarmonySetCodec


def _write_five_thirty_second_midi(path: Path) -> None:
    import mido

    midi = mido.MidiFile(type=1, ticks_per_beat=480)
    meta = mido.MidiTrack()
    meta.append(mido.MetaMessage("time_signature", numerator=5, denominator=32, time=0))
    notes = mido.MidiTrack()
    notes.append(mido.Message("note_on", channel=0, note=60, velocity=96, time=360))
    notes.append(mido.Message("note_off", channel=0, note=60, velocity=0, time=120))
    midi.tracks.extend([meta, notes])
    midi.save(str(path))


def test_parser_uses_bar_local_grid_and_preserves_raw_note_identity(tmp_path: Path) -> None:
    """A second 5/32 bar must not inherit the global-quarter grid phase."""
    midi_path = tmp_path / "phase.mid"
    _write_five_thirty_second_midi(midi_path)
    parser = MusicDirectoryParser(MusicParserConfig())

    songs = parser.parse_file(midi_path, {}, dataset_root=tmp_path)

    assert len(songs) == 1
    song = songs[0]
    assert len(song.bars) == 2
    bar = song.bars[1]
    note = bar.tracks[0].notes[0]
    assert bar.canonical_bar_index == 1
    assert bar.source_measure_index == 1
    assert note.onset_ql == 0.25
    assert note.duration_ql == 0.25
    assert note.source_onset_ql == 0.75
    assert note.source_note_ordinal == 0
    assert note.source_note_id.endswith(":1:0")
    audit = song.metadata["quantization_audit"]
    assert audit["audit_unit"] == "source_note_fragment"
    assert audit["fragment_count"] == 1
    assert audit["by_meter"]["5/32"]["fragment_count"] == 1
    controls = song.metadata["performance_controls"]
    assert controls["collector_version"] == "raw_smf_performance_controls.v1"
    assert controls["cc64_readable"] is True
    assert controls["cc64_present"] is False
    samples = song.runtime_diagnostics["quantization_fragment_samples"]
    assert samples == [{
        "source_note_id": note.source_note_id,
        "canonical_bar_index": 1,
        "meter": "5/32",
        "raw_local_start_tick": 60,
        "raw_local_end_tick": 180,
        "ppqn": 480,
        "raw_local_start_ql": 0.125,
        "raw_local_end_ql": 0.375,
        "ordinary_quantized_local_start_ql": 0.25,
        "ordinary_quantized_local_end_ql": 0.5,
        "final_quantized_local_start_ql": 0.25,
        "final_quantized_local_end_ql": 0.5,
        "quantization_repair_kind": None,
        "repair_slot_index": None,
        "projection_overlap_ql": None,
        "projection_endpoint_error_ql": None,
        "onset_residual_ql": 0.125,
        "end_residual_ql": 0.125,
    }]


def test_parser_discovers_only_canonical_smf_inputs(tmp_path: Path) -> None:
    """First-release canonical parsing must not silently route text scores."""
    _write_five_thirty_second_midi(tmp_path / "canonical.mid")
    (tmp_path / "legacy.abc").write_text("X:1\nK:C\nC", encoding="utf-8")
    parser = MusicDirectoryParser(MusicParserConfig())

    assert [path.name for path in parser.discover_files(tmp_path)] == ["canonical.mid"]


def test_form_metadata_fails_closed_for_legacy_indexes_and_maps_bound_canonical_sections(tmp_path: Path) -> None:
    path = tmp_path / "canonical.mid"; _write_five_thirty_second_midi(path)
    parser = MusicDirectoryParser(MusicParserConfig())
    legacy = parser.parse_file(path, {"sections": [{"start_bar": 0, "end_bar": 2, "name": "A"}]}, dataset_root=tmp_path)[0]
    assert legacy.metadata["form_mapping_status"] == "unavailable_legacy_measure_index"
    assert all(bar.form is None for bar in legacy.bars)

    bound = parser.parse_file(path, {}, dataset_root=tmp_path)[0]
    metadata = {
        "coordinate_system": "canonical_bar_index.v1",
        "source_file_identity": bound.metadata["source_file_identity"],
        "canonical_parser_version": "raw_smf_v1",
        "canonical_timeline_sha256": bound.metadata["canonical_timeline_sha256"],
        "form": "binary",
        "sections": [{"name": "A", "canonical_start_bar_index": 0, "canonical_end_bar_index": 1}, {"name": "B", "canonical_start_bar_index": 1, "canonical_end_bar_index": 2}],
    }
    mapped = parser.parse_file(path, metadata, dataset_root=tmp_path)[0]
    assert mapped.metadata["form_mapping_status"] == "mapped"
    assert mapped.form == "binary"
    assert [(bar.form, bar.section_label, bar.section_index) for bar in mapped.bars] == [("A", "A", 0), ("B", "B", 1)]

    metadata["canonical_timeline_sha256"] = "sha256:" + "0" * 64
    mismatch = parser.parse_file(path, metadata, dataset_root=tmp_path)[0]
    assert mismatch.metadata["form_mapping_status"] == "unavailable_timeline_mismatch"
    assert all(bar.form is None for bar in mismatch.bars)

    metadata["canonical_timeline_sha256"] = bound.metadata["canonical_timeline_sha256"]
    metadata["sections"] = []
    empty = parser.parse_file(path, metadata, dataset_root=tmp_path)[0]
    assert empty.metadata["form_mapping_status"] == "unavailable_empty_canonical_sections"

    metadata["sections"] = [{"name": "A", "canonical_start_bar_index": 0, "canonical_end_bar_index": 1}, {"name": "B", "canonical_start_bar_index": 0, "canonical_end_bar_index": 2}]
    overlap = parser.parse_file(path, metadata, dataset_root=tmp_path)[0]
    assert overlap.metadata["form_mapping_status"] == "unavailable_timeline_mismatch"
    assert overlap.form is None
    assert all((bar.form, bar.section_label, bar.section_index) == (None, None, None) for bar in overlap.bars)


def test_cross_five_thirty_second_bar_keeps_raw_continuation_and_hold(tmp_path: Path) -> None:
    """A continuation is physical even when its later local onset is zero."""
    import mido

    path = tmp_path / "cross.mid"
    midi = mido.MidiFile(type=1, ticks_per_beat=480)
    meta = mido.MidiTrack(); meta.append(mido.MetaMessage("time_signature", numerator=5, denominator=32, time=0))
    notes = mido.MidiTrack(); notes.append(mido.Message("note_on", note=60, velocity=90, time=240)); notes.append(mido.Message("note_off", note=60, velocity=0, time=240))
    midi.tracks.extend([meta, notes]); midi.save(str(path))
    song = MusicDirectoryParser(MusicParserConfig()).parse_file(path, {}, dataset_root=tmp_path)[0]

    assert [len(bar.tracks[0].notes) for bar in song.bars] == [1, 1]
    first, second = song.bars
    assert first.tracks[0].notes[0].continues_into_next_bar is True
    assert second.tracks[0].notes[0].continues_from_previous_bar is True
    config = {"bar_tensor": {"backend": "semantic_harmony_set_v2", "schema_version": "bar_tensor_schema.v2", "overflow_policy": "error", "slot_grid": {"quantum_ql": .25, "capacity": 48, "epsilon_ql": 1e-6}, "pitch_scale": 24.0, "velocity_scale": 127.0, "max_harmony_notes": 16, "relative_pitch_max_semitones": 96.0}}
    record = SemanticHarmonySetCodec.from_config(config).encode_song(song)[1]
    assert record.tensor[0, 0, 2] == 0.0
    assert record.tensor[0, 0, 3] == 1.0


def test_missing_tick_zero_time_signature_is_default_error_or_explicitly_provenanced(tmp_path: Path) -> None:
    import mido
    path = tmp_path / "late_ts.mid"
    midi = mido.MidiFile(ticks_per_beat=480)
    meta = mido.MidiTrack(); meta.append(mido.MetaMessage("time_signature", numerator=3, denominator=4, time=480))
    notes = mido.MidiTrack(); notes.extend([mido.Message("note_on", note=60, velocity=90, time=0), mido.Message("note_off", note=60, velocity=0, time=1920)])
    midi.tracks.extend([meta, notes]); midi.save(path)
    with __import__("pytest").raises(ValueError, match="time_signature_initial_missing"):
        MusicDirectoryParser(MusicParserConfig()).parse_file(path, {}, dataset_root=tmp_path)
    song = MusicDirectoryParser(MusicParserConfig(initial_time_signature_policy="smf_default_4_4")).parse_file(path, {}, dataset_root=tmp_path)[0]
    assert song.metadata["initial_time_signature"] == {"policy": "smf_default_4_4", "origin": "smf_default", "injected_at_tick": 0, "meter": "4/4", "first_real_ts_tick": 480}


def test_no_time_signature_events_can_only_use_explicit_smf_default(tmp_path: Path) -> None:
    import mido
    path = tmp_path / "no_ts.mid"
    midi = mido.MidiFile(ticks_per_beat=480)
    notes = mido.MidiTrack(); notes.extend([mido.Message("note_on", note=60, velocity=90, time=0), mido.Message("note_off", note=60, velocity=0, time=480)])
    midi.tracks.append(notes); midi.save(path)
    with __import__("pytest").raises(ValueError, match="time_signature_initial_missing"):
        MusicDirectoryParser(MusicParserConfig()).parse_file(path, {}, dataset_root=tmp_path)
    song = MusicDirectoryParser(MusicParserConfig(initial_time_signature_policy="smf_default_4_4")).parse_file(path, {}, dataset_root=tmp_path)[0]
    assert song.metadata["initial_time_signature"]["first_real_ts_tick"] is None
