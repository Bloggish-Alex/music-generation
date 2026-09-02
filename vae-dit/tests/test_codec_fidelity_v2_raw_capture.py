from __future__ import annotations

import hashlib
import json

import numpy as np

from diagnostics.codec_fidelity_raw_capture import CodecFidelityV2RawCaptureRequest, JsonNpzCodecFidelityV2RawCapture
from export.codec_fidelity_artifact_export import CodecFidelityArtifactExportConfig, export_codec_fidelity_artifacts


def _digest(path):
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _source(path, split, song, bars=None):
    bars = bars if bars is not None else [{"bar_index": 0, "bar_length_ql": 4.0, "notes": []}]
    path.write_text(json.dumps({"schema_version": "dataset_tonality_raw_source.v1", "dataset": {"split": split}, "availability": {"bar_note_events": True, "split_membership": True}, "songs": [{"song_id": song, "base_song_id": song, "applied_transpose_semitones": 0, "bars": bars}]}), encoding="utf-8")


def _canonical(tmp_path, rows):
    count = len(rows)
    voices = np.zeros((count, 18, 48, 6), dtype=np.float32)
    mask = np.zeros((count, 48), dtype=bool); mask[:, :16] = True
    voices[:, :, :16, 1] = 1.0
    durations = np.zeros((count, 48), dtype=np.float32); durations[:, :16] = .25
    arrays = tmp_path / "voice_tensors.npz"
    np.savez_compressed(arrays, voice_tensors=voices, slot_valid_mask=mask, slot_durations_ql=durations, bar_contexts=np.zeros((count, 12), dtype=np.float32), base_pitches=np.full(count, 60, dtype=np.int16), base_pitch_valid=np.ones(count, dtype=bool))
    index = tmp_path / "bar_tensor_index.json"; index.write_text(json.dumps(rows), encoding="utf-8")
    manifest = {"schema_version": "bar_tensor_schema.v2", "row_count": count, "voice_names": [], "feature_names": [], "arrays": {"path": arrays.name, "sha256": _digest(arrays)}, "index": {"sha256": _digest(index)}, "slot_grid_policy": {"quantum_ql": .25, "capacity": 48, "epsilon_ql": 1e-6}}
    (tmp_path / "encoding_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return arrays


def test_v2_capture_and_export_use_canonical_masked_arrays(tmp_path) -> None:
    rows = [{"row": 0, "song_id": "train_song", "base_song_id": "train_song", "source_bar_index": 0}, {"row": 1, "song_id": "validation_song", "base_song_id": "validation_song", "source_bar_index": 0}]
    _canonical(tmp_path, rows)
    train, validation = tmp_path / "train.json", tmp_path / "validation.json"; _source(train, "train", "train_song"); _source(validation, "validation", "validation_song")
    result = JsonNpzCodecFidelityV2RawCapture().capture(CodecFidelityV2RawCaptureRequest(tmp_path, "fixture", None, {"train": train, "validation": validation}, frozenset({"train_song"}), frozenset({"validation_song"})))
    assert set(result.artifacts) == {"train", "validation"}
    observation = json.loads(result.artifacts["train"].read_text())
    assert set(observation["arrays"]["names"]) == {"voice_tensors", "slot_valid_mask", "slot_durations_ql", "bar_contexts", "base_pitches", "base_pitch_valid"}
    output = tmp_path / "public"; export_codec_fidelity_artifacts(CodecFidelityArtifactExportConfig(tmp_path, output))
    assert (output / "codec_fidelity__raw_arrays__train.v2.npz").is_file()


def test_v2_capture_rejects_manifest_hash_and_mask_invariants(tmp_path) -> None:
    rows = [{"row": 0, "song_id": "song", "base_song_id": "song", "source_bar_index": 0}]
    arrays = _canonical(tmp_path, rows)
    source = tmp_path / "source.json"; _source(source, "train", "song")
    request = CodecFidelityV2RawCaptureRequest(tmp_path, "fixture", None, {"train": source}, frozenset({"song"}), frozenset())
    manifest_path = tmp_path / "encoding_manifest.json"; manifest = json.loads(manifest_path.read_text()); manifest["arrays"]["sha256"] = "sha256:wrong"; manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert "arrays.sha256" in JsonNpzCodecFidelityV2RawCapture().capture(request).unavailable["train"]
    manifest["arrays"]["sha256"] = _digest(arrays); manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with np.load(arrays) as archive:
        values = {key: archive[key] for key in archive.files}
    values["slot_durations_ql"][0, 20] = .25
    np.savez_compressed(arrays, **values); manifest["arrays"]["sha256"] = _digest(arrays); manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert "slot_valid_mask" in JsonNpzCodecFidelityV2RawCapture().capture(request).unavailable["train"]


def test_v2_capture_rejects_slot_duration_total_not_matching_source_bar(tmp_path) -> None:
    rows = [{"row": 0, "song_id": "song", "base_song_id": "song", "source_bar_index": 0}]
    arrays = _canonical(tmp_path, rows)
    source = tmp_path / "source.json"; _source(source, "train", "song")
    with np.load(arrays) as archive:
        values = {key: archive[key] for key in archive.files}
    values["slot_durations_ql"][0, 15] = .125
    np.savez_compressed(arrays, **values)
    manifest_path = tmp_path / "encoding_manifest.json"; manifest = json.loads(manifest_path.read_text()); manifest["arrays"]["sha256"] = _digest(arrays); manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    request = CodecFidelityV2RawCaptureRequest(tmp_path, "fixture", None, {"train": source}, frozenset({"song"}), frozenset())
    assert "does not equal" in JsonNpzCodecFidelityV2RawCapture().capture(request).unavailable["train"]


def test_v2_capture_allows_one_source_identity_in_two_clipped_bars(tmp_path) -> None:
    rows = [{"row": 0, "song_id": "song", "base_song_id": "song", "source_bar_index": 0}, {"row": 1, "song_id": "song", "base_song_id": "song", "source_bar_index": 1}]
    _canonical(tmp_path, rows)
    sustained = {"pitch": 60, "velocity": 80, "physical_track_index": 2, "source_note_ordinal": 7, "source_note_id": "file:2:7", "source_onset_ql": 3.5}
    source = tmp_path / "source.json"; _source(source, "train", "song", [{"bar_index": 0, "bar_length_ql": 4.0, "notes": [{**sustained, "onset_ql": 3.5, "duration_ql": .5}]}, {"bar_index": 1, "bar_length_ql": 4.0, "notes": [{**sustained, "onset_ql": 0., "duration_ql": 1.}]}])
    result = JsonNpzCodecFidelityV2RawCapture().capture(CodecFidelityV2RawCaptureRequest(tmp_path, "fixture", None, {"train": source}, frozenset({"song"}), frozenset()))
    assert "train" in result.artifacts
