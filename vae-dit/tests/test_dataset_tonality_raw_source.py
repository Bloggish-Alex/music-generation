from __future__ import annotations

import json
from pathlib import Path
import sys

from jsonschema import Draft202012Validator
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.core import BarRecord, NoteEvent, SongRecord, TrackRecord
from diagnostics.dataset_tonality_raw_source import (
    RAW_SOURCE_SCHEMA_VERSION,
    DatasetTonalityRawSourceRequest,
    JsonDatasetTonalityRawSourceWriter,
)


def _song(
    song_id: str,
    file_path: str,
    *,
    transpose: int = 0,
    empty_first_bar: bool = False,
) -> SongRecord:
    bars = [
        BarRecord(
            song_id=song_id,
            file_path=file_path,
            bar_index=0,
            bar_length_ql=4.0,
            time_signature="4/4",
            tracks=[] if empty_first_bar else [
                TrackRecord(
                    track_index=1,
                    name="harmony",
                    notes=[
                        NoteEvent(pitch=67, onset_ql=1.0, duration_ql=0.5, velocity=70),
                        NoteEvent(pitch=64, onset_ql=0.0, duration_ql=1.0, velocity=80),
                    ],
                ),
                TrackRecord(
                    track_index=0,
                    name="melody",
                    notes=[NoteEvent(pitch=72, onset_ql=0.0, duration_ql=0.5, velocity=90)],
                ),
            ],
        ),
        BarRecord(
            song_id=song_id,
            file_path=file_path,
            bar_index=1,
            bar_length_ql=3.0,
            time_signature="3/4",
            tracks=[],
        ),
    ]
    return SongRecord(
        song_id=song_id,
        file_path=file_path,
        metadata={"transpose_semitones": transpose},
        bars=bars,
    )


def test_writer_emits_schema_valid_path_free_split_artifacts(tmp_path: Path) -> None:
    absolute_path = r"D:\private_music\source_a.mid"
    request = DatasetTonalityRawSourceRequest(
        model_dir=tmp_path,
        dataset_identity="fixture_stage",
        dataset_identity_kind="stage_label_unverified",
        songs_by_split={
            "train": [_song("source_a", absolute_path, empty_first_bar=True)],
            "validation": [_song("source_b_T-2", r"C:\private_music\source_b.mid", transpose=-2)],
        },
    )

    paths = JsonDatasetTonalityRawSourceWriter().write(request)

    assert paths == {
        "train": tmp_path / "dataset_tonality__raw_source__train.v1.json",
        "validation": tmp_path / "dataset_tonality__raw_source__validation.v1.json",
    }
    schema = json.loads((ROOT / "contracts" / "evaluation" / "v1" / "dataset_tonality__raw_source.v1.schema.json").read_text(encoding="utf-8"))
    train = json.loads(paths["train"].read_text(encoding="utf-8"))
    validation = json.loads(paths["validation"].read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(train)
    Draft202012Validator(schema).validate(validation)

    assert train["schema_version"] == RAW_SOURCE_SCHEMA_VERSION
    assert train["availability"] == {
        "bar_note_events": True,
        "split_membership": True,
        "source_content_hashes": False,
    }
    assert train["source"]["encoding_artifact_sha256"] is None
    assert train["songs"][0]["bars"][0]["tempo_bpm"] is None
    assert train["songs"][0]["bars"][0]["notes"] == []
    assert validation["songs"][0]["base_song_id"] == "source_b"
    assert validation["songs"][0]["applied_transpose_semitones"] == -2
    assert [note["track_index"] for note in validation["songs"][0]["bars"][0]["notes"]] == [0, 1, 1]
    artifact_text = paths["train"].read_text(encoding="utf-8") + paths["validation"].read_text(encoding="utf-8")
    assert absolute_path not in artifact_text
    assert r"C:\private_music\source_b.mid" not in artifact_text


def test_writer_rejects_a_base_song_in_multiple_splits(tmp_path: Path) -> None:
    writer = JsonDatasetTonalityRawSourceWriter()
    request = DatasetTonalityRawSourceRequest(
        model_dir=tmp_path,
        dataset_identity="fixture_stage",
        dataset_identity_kind="stage_label_unverified",
        songs_by_split={
            "train": [_song("source_a", r"D:\a.mid")],
            "validation": [_song("source_a_T+3", r"D:\a.mid", transpose=3)],
        },
    )

    with pytest.raises(ValueError, match="appears in both"):
        writer.write(request)
