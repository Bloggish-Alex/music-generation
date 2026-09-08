"""Parser-path regression tests for the raw-SMF Codec V2 authority."""

from __future__ import annotations

from pathlib import Path

from data.music_parser import MusicDirectoryParser, MusicParserConfig


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


def test_parser_discovers_only_canonical_smf_inputs(tmp_path: Path) -> None:
    """First-release canonical parsing must not silently route text scores."""
    _write_five_thirty_second_midi(tmp_path / "canonical.mid")
    (tmp_path / "legacy.abc").write_text("X:1\nK:C\nC", encoding="utf-8")
    parser = MusicDirectoryParser(MusicParserConfig())

    assert [path.name for path in parser.discover_files(tmp_path)] == ["canonical.mid"]
