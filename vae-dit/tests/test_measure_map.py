from __future__ import annotations

import pytest
from music21 import chord, meter, note, stream

from data.measure_map import extract_measure_spans
from data.music_parser import MusicDirectoryParser, MusicParserConfig
from data.core import MeasureSpan, SongRecord
from data.music_parser import MusicDirectoryParser, MusicParserConfig


def _score(signature: str, duration: float) -> stream.Score:
    score = stream.Score(); part = stream.Part(); measure = stream.Measure(number=1)
    measure.insert(0, meter.TimeSignature(signature)); measure.insert(0, note.Note("C4", quarterLength=duration))
    part.append(measure); score.append(part)
    return score


def test_extract_measure_spans_uses_real_six_eight_duration() -> None:
    spans = extract_measure_spans(_score("6/8", 3.0))
    assert spans[0].time_signature == "6/8"
    assert spans[0].end_ql - spans[0].start_ql == 3.0


def test_extract_measure_spans_marks_pickup() -> None:
    spans = extract_measure_spans(_score("4/4", 1.0))
    assert spans[0].is_pickup is True


def test_extract_measure_spans_rejects_missing_meter() -> None:
    score = stream.Score(); part = stream.Part(); part.append(note.Note("C4")); score.append(part)
    with pytest.raises(ValueError, match="measure_map_unavailable"):
        extract_measure_spans(score)


def test_measure_map_rejects_part_mismatch() -> None:
    score = _score("4/4", 4.0)
    other = stream.Part(); measure = stream.Measure(number=1); measure.insert(0, meter.TimeSignature("3/4")); measure.insert(0, note.Note("D4", quarterLength=3.0)); other.append(measure); score.append(other)
    with pytest.raises(ValueError, match="measure_map_part_mismatch"):
        extract_measure_spans(score)


def test_single_part_retains_physical_track_zero_and_duplicate_audit_ids() -> None:
    parser = MusicDirectoryParser(MusicParserConfig())
    score = _score("4/4", 4.0)
    tracks = parser._collect_tracks(score)
    assert [physical for physical, _ in tracks] == [0]
    source = {(0, 0): (0.13, 0.88), (0, 1): (0.13, 0.88)}
    quantized = {(0, 0): (0.25, 1.0), (0, 1): (0.0, 0.75)}
    audit = parser._quantization_audit(source, quantized)
    assert audit["event_count"] == 2 and audit["nonzero_residual_count"] == 4


def test_quantized_parser_path_preserves_source_ids_through_reordering_and_chord() -> None:
    parser = MusicDirectoryParser(MusicParserConfig())
    score = stream.Score(); part = stream.Part(); measure = stream.Measure(number=1); measure.insert(0, meter.TimeSignature("4/4"))
    measure.insert(0.13, note.Note("E4", quarterLength=0.4)); measure.insert(0.12, note.Note("C4", quarterLength=0.4)); measure.insert(0.13, chord.Chord(["G4", "B4"], quarterLength=0.4)); part.append(measure); score.append(part)
    parser._tag_source_events(score)
    before = parser._all_event_boundaries(score)
    quantized = parser._quantize_score(score)
    after = parser._all_event_boundaries(quantized)
    assert set(before) == set(after) and len(after) == 4
    tracks = parser._collect_tracks(quantized)
    song = SongRecord("fixture", "fixture.mid", metadata={"source_file_identity": "file"})
    bar = parser._build_bar(song, tracks, 0, MeasureSpan(0, 0.0, 4.0, "4/4", 4, 4), 1)
    identifiers = [f"{item.physical_track_index}:{item.source_note_ordinal}" for track in bar.tracks for item in track.notes]
    assert len(identifiers) == len(set(identifiers)) == 4


def test_track_retention_defaults_to_error_and_records_explicit_truncation() -> None:
    tracks = [(index, [(0.0, 1.0, 60 + index, 80)]) for index in range(3)]
    failing = MusicDirectoryParser(MusicParserConfig(hard_safety_limit=2, track_retention_policy="error"))
    with pytest.raises(ValueError, match="track_limit_exceeded"):
        failing._select_tracks(tracks)
    truncating = MusicDirectoryParser(MusicParserConfig(hard_safety_limit=2, track_retention_policy="truncate"))
    assert len(truncating._select_tracks(tracks)) == 2
    event = truncating.track_retention_events[-1]
    assert event["dropped_part_count"] == 1 and event["dropped_note_count"] == 1
