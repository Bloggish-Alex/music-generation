from __future__ import annotations

import json
import numpy as np

from diagnostics.codec_fidelity_raw_capture import CodecFidelityV2RawCaptureRequest, JsonNpzCodecFidelityV2RawCapture
from export.codec_fidelity_artifact_export import CodecFidelityArtifactExportConfig, export_codec_fidelity_artifacts


def _source(path, split, song):
    path.write_text(json.dumps({"schema_version": "dataset_tonality_raw_source.v1", "dataset": {"split": split}, "availability": {"bar_note_events": True, "split_membership": True}, "songs": [{"song_id": song, "base_song_id": song, "applied_transpose_semitones": 0, "bars": [{"bar_index": 0, "notes": []}]}]}), encoding="utf-8")


def test_v2_capture_and_export_use_declared_row_aligned_arrays(tmp_path) -> None:
    voices = np.zeros((2, 18, 16, 6), dtype=np.float32); voices[:, :, :, 1] = 1.0
    np.savez_compressed(tmp_path / "codec_v2_arrays.npz", voice_tensors=voices, bar_contexts=np.zeros((2, 12), dtype=np.float32), base_pitches=np.asarray([60, 61], dtype=np.int16), base_pitch_valid=np.asarray([True, True]))
    rows = [{"row": 0, "song_id": "train_song", "base_song_id": "train_song", "source_bar_index": 0}, {"row": 1, "song_id": "validation_song", "base_song_id": "validation_song", "source_bar_index": 0}]
    index = tmp_path / "bar_tensor_index.json"
    index.write_text(json.dumps(rows), encoding="utf-8")
    import hashlib
    digest = lambda path: "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path / "encoding_manifest.json").write_text(json.dumps({"schema_version": "bar_tensor_schema.v2", "voice_names": [], "feature_names": [], "arrays": {"sha256": digest(tmp_path / "codec_v2_arrays.npz")}, "index": {"sha256": digest(index)}}), encoding="utf-8")
    train, validation = tmp_path / "train.json", tmp_path / "validation.json"; _source(train, "train", "train_song"); _source(validation, "validation", "validation_song")
    result = JsonNpzCodecFidelityV2RawCapture().capture(CodecFidelityV2RawCaptureRequest(tmp_path, "fixture", None, {"train": train, "validation": validation}, frozenset({"train_song"}), frozenset({"validation_song"})))
    assert set(result.artifacts) == {"train", "validation"}
    observation = json.loads(result.artifacts["train"].read_text())
    assert observation["schema_version"] == "codec_fidelity_raw_observation.v2"
    assert set(observation["arrays"]["names"]) == {"voice_tensors", "bar_contexts", "base_pitches", "base_pitch_valid"}
    output = tmp_path / "public"; export_codec_fidelity_artifacts(CodecFidelityArtifactExportConfig(tmp_path, output))
    assert (output / "codec_fidelity__raw_arrays__train.v2.npz").is_file()
    assert (output / "codec_fidelity__raw_status__train.v2.json").is_file()


def test_v2_capture_rejects_manifest_hash_or_note_identity_mismatch(tmp_path) -> None:
    voices = np.zeros((1, 18, 16, 6), dtype=np.float32)
    np.savez_compressed(tmp_path / "codec_v2_arrays.npz", voice_tensors=voices, bar_contexts=np.zeros((1, 12), dtype=np.float32), base_pitches=np.asarray([60], dtype=np.int16), base_pitch_valid=np.asarray([True]))
    rows = [{"row": 0, "song_id": "song", "base_song_id": "song", "source_bar_index": 0}]
    (tmp_path / "bar_tensor_index.json").write_text(json.dumps(rows), encoding="utf-8")
    (tmp_path / "encoding_manifest.json").write_text(json.dumps({"schema_version": "bar_tensor_schema.v2", "arrays": {"sha256": "sha256:wrong"}, "index": {"sha256": "sha256:wrong"}}), encoding="utf-8")
    source = tmp_path / "source.json"; _source(source, "train", "song")
    request = CodecFidelityV2RawCaptureRequest(tmp_path, "fixture", None, {"train": source}, frozenset({"song"}), frozenset())
    result = JsonNpzCodecFidelityV2RawCapture().capture(request)
    assert result.unavailable["train"].startswith("V2 manifest arrays.sha256")
    import hashlib
    digest = lambda path: "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path / "encoding_manifest.json").write_text(json.dumps({"schema_version": "bar_tensor_schema.v2", "arrays": {"sha256": digest(tmp_path / "codec_v2_arrays.npz")}, "index": {"sha256": digest(tmp_path / "bar_tensor_index.json")}}), encoding="utf-8")
    source.write_text(json.dumps({"schema_version": "dataset_tonality_raw_source.v1", "dataset": {"split": "train"}, "availability": {"bar_note_events": True, "split_membership": True}, "songs": [{"song_id": "song", "base_song_id": "song", "applied_transpose_semitones": 0, "bars": [{"bar_index": 0, "notes": [{"pitch": 60}]}]}]}), encoding="utf-8")
    result = JsonNpzCodecFidelityV2RawCapture().capture(request)
    assert "note identity" in result.unavailable["train"]
