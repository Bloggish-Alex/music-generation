from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

from jsonschema import Draft202012Validator
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.core import BarTensorRecord, SongRecord
from diagnostics.codec_fidelity_raw_capture import (
    CodecFidelityRawCaptureRequest,
    JsonNpzCodecFidelityRawCapture,
)


def _song(song_id: str, *, transpose: int = 0) -> SongRecord:
    return SongRecord(
        song_id=song_id,
        file_path=rf"D:\private_music\{song_id}.mid",
        metadata={"transpose_semitones": transpose},
    )


def _record(song_id: str, bar_index: int, base_pitch: int | None = 60) -> BarTensorRecord:
    return BarTensorRecord(
        song_id=song_id,
        bar_index=bar_index,
        tensor_shape=[3, 2, 18],
        tensor=np.full((3, 2, 18), float(bar_index + 1), dtype=np.float32),
        diagnostics={"base_pitch": base_pitch},
    )


def _source_raw_paths(tmp_path: Path, splits: tuple[str, ...]) -> dict[str, Path]:
    paths = {}
    for split in splits:
        path = tmp_path / f"dataset_tonality__raw_source__{split}.v1.json"
        songs = {
            "train": [{"song_id": "source_a", "base_song_id": "source_a", "applied_transpose_semitones": 0, "bars": [{"bar_index": 0}, {"bar_index": 1}]}],
            "validation": [{"song_id": "source_b_T-2", "base_song_id": "source_b", "applied_transpose_semitones": -2, "bars": [{"bar_index": 0}]}],
            "excluded_unpaired": [{"song_id": "source_c", "base_song_id": "source_c", "applied_transpose_semitones": 0, "bars": [{"bar_index": 0}]}],
        }[split]
        path.write_text(json.dumps({
            "schema_version": "dataset_tonality_raw_source.v1",
            "dataset": {"split": split},
            "songs": songs,
            "availability": {"bar_note_events": True, "split_membership": True},
        }), encoding="utf-8")
        paths[split] = path
    return paths


def _request(tmp_path: Path) -> CodecFidelityRawCaptureRequest:
    return CodecFidelityRawCaptureRequest(
        model_dir=tmp_path,
        dataset_identity="fixture_stage",
        dataset_identity_kind="stage_label_unverified",
        dataset_content_sha256=None,
        source_raw_paths=_source_raw_paths(tmp_path, ("train", "validation", "excluded_unpaired")),
        songs=[_song("source_a"), _song("source_b_T-2", transpose=-2), _song("source_c")],
        tensors=[_record("source_a", 1), _record("source_a", 0), _record("source_b_T-2", 0), _record("source_c", 0)],
        train_base_song_ids=frozenset({"source_a"}),
        validation_base_song_ids=frozenset({"source_b"}),
        codec_config={"bar_tensor": {"backend": "semantic_3voice", "steps_per_bar": 2, "pitch_scale": 24.0}},
    )


def test_capture_writes_schema_valid_split_npz_and_complete_alignment(tmp_path: Path) -> None:
    result = JsonNpzCodecFidelityRawCapture().capture(_request(tmp_path))

    assert set(result.artifacts) == {"train", "validation", "excluded_unpaired"}
    assert set(result.status_artifacts) == {"train", "validation", "excluded_unpaired"}
    assert result.unavailable == {}
    schema = json.loads((ROOT / "contracts" / "evaluation" / "v1" / "codec_fidelity__raw_observation.v1.schema.json").read_text(encoding="utf-8"))
    train = json.loads(result.artifacts["train"].read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(train)
    tensor_path = tmp_path / train["tensor"]["path"]
    with np.load(tensor_path, allow_pickle=False) as archive:
        assert archive.files == ["bar_tensors"]
        assert archive["bar_tensors"].dtype == np.float32
        assert archive["bar_tensors"].shape == tuple(train["tensor"]["shape"])
    assert [row["tensor_row"] for row in train["alignment"]] == [0, 1]
    assert [row["bar_index"] for row in train["alignment"]] == [0, 1]
    assert train["source_raw"]["path"] == "dataset_tonality__raw_source__train.v1.json"
    assert train["source_raw"]["sha256"].startswith("sha256:")
    assert train["tensor_schema"]["axis_order"] == ["bar", "track", "step", "feature"]
    assert train["tensor_schema"]["track_names"] == ["melody", "harmony", "bass"]
    assert train["tensor_schema"]["feature_names"][0] == "relative_pitch"
    artifact_text = result.artifacts["train"].read_text(encoding="utf-8")
    assert "private_music" not in artifact_text
    assert "tensor_key" not in artifact_text
    assert "SemanticBarTensorCodec" not in artifact_text
    status_schema = json.loads((ROOT / "contracts" / "evaluation" / "v1" / "codec_fidelity__raw_status.v1.schema.json").read_text(encoding="utf-8"))
    status = json.loads(result.status_artifacts["train"].read_text(encoding="utf-8"))
    Draft202012Validator(status_schema).validate(status)
    assert status["status"] == "AVAILABLE"
    assert status["unavailable_reasons"] == []
    assert status["artifacts"]["observation"]["path"] == result.artifacts["train"].name
    assert status["artifacts"]["observation"]["sha256"] == f"sha256:{hashlib.sha256(result.artifacts['train'].read_bytes()).hexdigest()}"
    assert status["artifacts"]["tensors"]["sha256"] == f"sha256:{hashlib.sha256(tensor_path.read_bytes()).hexdigest()}"
    status_text = result.status_artifacts["train"].read_text(encoding="utf-8")
    assert "private_music" not in status_text
    assert "tensor_key" not in status_text
    assert "SemanticBarTensorCodec" not in status_text


def test_capture_rejects_cross_split_base_song_membership(tmp_path: Path) -> None:
    request = _request(tmp_path)
    invalid = CodecFidelityRawCaptureRequest(
        **{**request.__dict__, "validation_base_song_ids": frozenset({"source_a"})}
    )

    with pytest.raises(ValueError, match="both train and validation"):
        JsonNpzCodecFidelityRawCapture().capture(invalid)


def test_capture_marks_split_unavailable_when_existing_base_pitch_is_missing(tmp_path: Path) -> None:
    request = _request(tmp_path)
    records = list(request.tensors)
    records[0] = _record("source_a", 1, base_pitch=None)
    missing_anchor = CodecFidelityRawCaptureRequest(**{**request.__dict__, "tensors": records})

    result = JsonNpzCodecFidelityRawCapture().capture(missing_anchor)

    assert "train" in result.unavailable
    assert "base_pitch" in result.unavailable["train"]
    assert "train" not in result.artifacts
    assert set(result.artifacts) == {"validation", "excluded_unpaired"}
    status = json.loads(result.status_artifacts["train"].read_text(encoding="utf-8"))
    assert status["status"] == "UNAVAILABLE"
    assert status["availability"]["base_pitch"] is False
    assert status["artifacts"] == {}
    assert status["unavailable_reasons"][0]["field"] == "base_pitch"


def test_capture_rejects_invalid_or_misaligned_source_raw_observations(tmp_path: Path) -> None:
    request = _request(tmp_path)
    request.source_raw_paths["train"].write_text(json.dumps({"split": "train"}), encoding="utf-8")

    invalid_schema = JsonNpzCodecFidelityRawCapture().capture(request)

    assert "train" in invalid_schema.unavailable
    assert "schema_version" in invalid_schema.unavailable["train"]
    assert "train" not in invalid_schema.artifacts
    invalid_status = json.loads(invalid_schema.status_artifacts["train"].read_text(encoding="utf-8"))
    assert invalid_status["status"] == "UNAVAILABLE"
    assert invalid_status["unavailable_reasons"][0]["field"] == "source_raw_reference"

    aligned_request = _request(tmp_path)
    train_path = aligned_request.source_raw_paths["train"]
    payload = json.loads(train_path.read_text(encoding="utf-8"))
    payload["songs"][0]["base_song_id"] = "wrong_base"
    train_path.write_text(json.dumps(payload), encoding="utf-8")

    misaligned = JsonNpzCodecFidelityRawCapture().capture(aligned_request)

    assert "train" in misaligned.unavailable
    assert "base_song_id" in misaligned.unavailable["train"]
    assert "train" not in misaligned.artifacts


def test_unavailable_split_removes_stale_observation_and_tensor_artifacts(tmp_path: Path) -> None:
    capture = JsonNpzCodecFidelityRawCapture()
    available_request = _request(tmp_path)
    available = capture.capture(available_request)
    old_observation = available.artifacts["train"]
    old_tensor = tmp_path / json.loads(old_observation.read_text(encoding="utf-8"))["tensor"]["path"]
    records = list(available_request.tensors)
    records[0] = _record("source_a", 1, base_pitch=None)

    unavailable = capture.capture(CodecFidelityRawCaptureRequest(
        **{**available_request.__dict__, "tensors": records}
    ))

    assert not old_observation.exists()
    assert not old_tensor.exists()
    status = json.loads(unavailable.status_artifacts["train"].read_text(encoding="utf-8"))
    assert status["status"] == "UNAVAILABLE"
    assert status["artifacts"] == {}
    assert set(unavailable.artifacts) == {"validation", "excluded_unpaired"}
