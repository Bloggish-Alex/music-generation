from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mido
import pytest
from jsonschema import Draft202012Validator

from data.canonical_timeline import CanonicalTimelineError, collect_raw_smf_facts
from diagnostics.raw_pairing_census import census_file, run_census


def _write(path: Path, *tracks: list[mido.Message]) -> Path:
    midi = mido.MidiFile(type=1, ticks_per_beat=480)
    for events in tracks:
        track = mido.MidiTrack()
        track.extend(events)
        midi.tracks.append(track)
    midi.save(path)
    return path


def _on(note: int, velocity: int = 80, time: int = 0, channel: int = 0) -> mido.Message:
    return mido.Message("note_on", note=note, velocity=velocity, time=time, channel=channel)


def _off(note: int, time: int = 0, channel: int = 0) -> mido.Message:
    return mido.Message("note_off", note=note, velocity=0, time=time, channel=channel)


def _codes(path: Path, root: Path) -> list[str]:
    return [item["failure_code"] for item in census_file(path, root)]


def test_census_reports_orphan_with_same_key_same_tick_context(tmp_path: Path) -> None:
    path = _write(tmp_path / "orphan.mid", [_off(60)])
    finding = census_file(path, tmp_path)[0]
    assert finding["failure_code"] == "orphan_note_off"
    assert finding["tick_relation"] == "unmatched"
    assert finding["same_tick_events"] == [{"event_kind": "note_off", "absolute_tick": 0, "event_ordinal": 0, "velocity": 0}]


def test_census_reports_eof_unterminated_note_on(tmp_path: Path) -> None:
    path = _write(tmp_path / "dangling.mid", [_on(60)])
    finding = census_file(path, tmp_path)[0]
    assert finding["failure_code"] == "unterminated_note_on"
    assert finding["pending_on_queue"] == [{"absolute_tick": 0, "event_ordinal": 0, "velocity": 80}]


def test_census_reports_same_tick_nonpositive_duration(tmp_path: Path) -> None:
    path = _write(tmp_path / "same_tick.mid", [_on(60), _off(60)])
    finding = census_file(path, tmp_path)[0]
    assert finding["failure_code"] == "note_nonpositive_duration"
    assert finding["tick_relation"] == "same_tick"
    assert len(finding["same_tick_events"]) == 2


def test_census_uses_fifo_and_velocity_zero_note_on_is_note_off(tmp_path: Path) -> None:
    path = _write(tmp_path / "fifo.mid", [_on(60), _on(60, time=10), _off(60, time=10), _on(60, velocity=0, time=10)])
    assert census_file(path, tmp_path) == []
    _, notes = collect_raw_smf_facts(mido.MidiFile(path))
    assert [(note.start_tick, note.end_tick) for note in notes] == [(0, 20), (10, 30)]


def test_census_does_not_cross_pair_channels_or_physical_tracks(tmp_path: Path) -> None:
    path = _write(tmp_path / "isolated.mid", [_on(60, channel=0), _off(60, time=10, channel=1)], [_off(60, channel=0)])
    findings = census_file(path, tmp_path)
    assert sorted(item["failure_code"] for item in findings) == ["orphan_note_off", "orphan_note_off", "unterminated_note_on"]
    assert {item["physical_track_index"] for item in findings} == {0, 1}


def test_census_is_diagnostic_only_and_parser_remains_strict(tmp_path: Path) -> None:
    path = _write(tmp_path / "strict.mid", [_off(60)])
    assert _codes(path, tmp_path) == ["orphan_note_off"]
    with pytest.raises(CanonicalTimelineError, match="orphan_note_off"):
        collect_raw_smf_facts(mido.MidiFile(path))


def test_census_artifact_schema_manifest_hash_and_stable_input_sort(tmp_path: Path) -> None:
    _write(tmp_path / "z.mid", [_off(60)])
    _write(tmp_path / "a.mid", [_on(61)])
    output = tmp_path / "out"
    manifest_path = run_census(tmp_path, output, "deadbeef")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact_path = output / manifest["artifact"]["path"]
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    schema_path = Path(__file__).resolve().parents[1] / "contracts" / "diagnostics" / "raw_pairing_census.v1.schema.json"
    Draft202012Validator(json.loads(schema_path.read_text(encoding="utf-8"))).validate(artifact)
    assert [item["dataset_relative_posix_path"] for item in manifest["input_files"]] == ["a.mid", "z.mid"]
    assert manifest["input_sort"] == "dataset_relative_posix_path_nfc_ascending"
    assert manifest["code_revision"] == "deadbeef"
    assert manifest["artifact"]["sha256"] == "sha256:" + hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    assert manifest["dataset"]["content_sha256"].startswith("sha256:")
