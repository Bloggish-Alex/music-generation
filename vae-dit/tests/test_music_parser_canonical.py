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
    samples = song.runtime_diagnostics["quantization_fragment_samples"]
    assert samples == [{
        "source_note_id": note.source_note_id,
        "canonical_bar_index": 1,
        "meter": "5/32",
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
    config = {"bar_tensor": {"backend": "semantic_harmony_set_v2", "schema_version": "bar_tensor_schema.v2", "overflow_policy": "error", "steps_per_bar": 48, "pitch_scale": 24.0, "velocity_scale": 127.0, "max_harmony_notes": 16, "relative_pitch_max_semitones": 96.0}}
    record = SemanticHarmonySetCodec.from_config(config).encode_song(song)[1]
    assert record.tensor[0, 0, 2] == 0.0
    assert record.tensor[0, 0, 3] == 1.0
