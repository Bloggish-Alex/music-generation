from __future__ import annotations

import json
import numpy as np

from evaluation_framework.evaluation_api import ArtifactBundle
from evaluation_framework.evaluation_artifact_store import EvaluationArtifactStore
from evaluation_framework.evaluation_codec_fidelity import CodecFidelityEvaluator, CodecFidelityExporter
from evaluation_framework.evaluation_context import EvaluationContext, ExportContext
from evaluation_framework.evaluation_codec_fidelity import _multiset_f1


def test_v2_status_dispatches_to_monitor_report(tmp_path) -> None:
    public = tmp_path / "public"; public.mkdir(); run = EvaluationArtifactStore.create(tmp_path, "run")
    arrays = public / "codec_fidelity__raw_arrays__train.v2.npz"
    voices = np.zeros((1, 18, 16, 6), dtype=np.float32); voices[:, :, :, 1] = 1.0
    np.savez_compressed(arrays, voice_tensors=voices, bar_contexts=np.zeros((1, 12), dtype=np.float32), base_pitches=np.asarray([0], dtype=np.int16), base_pitch_valid=np.asarray([False]))
    import hashlib
    digest = lambda path: "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    source = public / "dataset_tonality__raw_source__train.v1.json"
    source.write_text(json.dumps({"songs": [{"song_id": "song", "bars": [{"bar_index": 0, "notes": []}]}]}), encoding="utf-8")
    manifest = public / "encoding_manifest.json"
    manifest.write_text(json.dumps({"configuration": {"slot_time_epsilon_ql": 1e-6, "melody_continuity_tolerance": 7}}), encoding="utf-8")
    observation = public / "codec_fidelity__raw_observation__train.v2.json"
    observation.write_text(json.dumps({"schema_version": "codec_fidelity_raw_observation.v2", "dataset": {"split": "train"}, "arrays": {"path": arrays.name, "sha256": digest(arrays)}, "source_raw": {"path": source.name, "sha256": digest(source)}, "encoding_manifest": {"path": manifest.name, "sha256": digest(manifest)}, "alignment": [], "availability": {}}), encoding="utf-8")
    status = public / "codec_fidelity__raw_status__train.v2.json"
    status.write_text(json.dumps({"schema_version": "codec_fidelity_raw_status.v2", "dataset": {"split": "train"}, "status": "AVAILABLE", "artifacts": {"observation": {"path": observation.name, "sha256": digest(observation)}, "arrays": {"path": arrays.name, "sha256": digest(arrays)}}, "unavailable_reasons": []}), encoding="utf-8")
    bundle = CodecFidelityExporter().export(ExportContext("run", public, run))
    result = CodecFidelityEvaluator().evaluate(EvaluationContext("run", public, run), ArtifactBundle("codec_fidelity", bundle.artifacts))
    assert result.report["status"] == "MONITOR"
    assert result.report["metrics"]["splits"]["train"]["schema_version"] == "bar_tensor_schema.v2"
    assert "Codec Fidelity V2" in result.markdown


def test_v2_harmony_pitch_state_multiset_detects_a_mismatch() -> None:
    precision, recall, f1 = _multiset_f1([(60, "onset")], [(61, "onset")])
    assert (precision, recall, f1) == (0.0, 0.0, 0.0)


def test_v2_measurement_uses_continuity_and_onset_hold_state(tmp_path) -> None:
    public = tmp_path / "public"; public.mkdir(); run = EvaluationArtifactStore.create(tmp_path, "run")
    voices = np.zeros((1, 18, 16, 6), dtype=np.float32); voices[:, :, :, 1] = 1.0
    for lane, pitch in ((0, 72), (17, 60)):
        voices[0, lane, :4, 0] = (pitch - 60) / 24.0
        voices[0, lane, :4, 1] = 0.0
        voices[0, lane, 0, 2] = 1.0
        voices[0, lane, 1:4, 3] = 1.0
    voices[0, 1, 1:4, 0] = (78 - 60) / 24.0
    voices[0, 1, 1:4, 1] = 0.0
    voices[0, 1, 1, 2] = 1.0
    voices[0, 1, 2:4, 3] = 1.0
    arrays = public / "codec_fidelity__raw_arrays__train.v2.npz"
    np.savez_compressed(arrays, voice_tensors=voices, bar_contexts=np.zeros((1, 12), dtype=np.float32), base_pitches=np.asarray([60], dtype=np.int16), base_pitch_valid=np.asarray([True]))
    import hashlib
    digest = lambda path: "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    notes = [
        {"pitch": 72, "onset_ql": 0.0, "duration_ql": 1.0, "velocity": 90, "physical_track_index": 0, "source_note_ordinal": 0, "source_note_id": "a:0:0", "source_onset_ql": 0.0},
        {"pitch": 60, "onset_ql": 0.0, "duration_ql": 1.0, "velocity": 80, "physical_track_index": 1, "source_note_ordinal": 0, "source_note_id": "a:1:0", "source_onset_ql": 0.0},
        {"pitch": 78, "onset_ql": 0.25, "duration_ql": 0.75, "velocity": 70, "physical_track_index": 2, "source_note_ordinal": 0, "source_note_id": "a:2:0", "source_onset_ql": 0.25},
    ]
    source = public / "dataset_tonality__raw_source__train.v1.json"
    source.write_text(json.dumps({"songs": [{"song_id": "song", "bars": [{"bar_index": 0, "bar_length_ql": 4.0, "notes": notes}]}]}), encoding="utf-8")
    manifest = public / "encoding_manifest.json"
    manifest.write_text(json.dumps({"configuration": {"slot_time_epsilon_ql": 1e-6, "melody_continuity_tolerance": 7}}), encoding="utf-8")
    observation = public / "codec_fidelity__raw_observation__train.v2.json"
    observation.write_text(json.dumps({"schema_version": "codec_fidelity_raw_observation.v2", "dataset": {"split": "train"}, "arrays": {"path": arrays.name, "sha256": digest(arrays)}, "source_raw": {"path": source.name, "sha256": digest(source)}, "encoding_manifest": {"path": manifest.name, "sha256": digest(manifest)}, "alignment": [{"tensor_row": 0, "song_id": "song", "source_bar_index": 0}], "availability": {}}), encoding="utf-8")
    status = public / "codec_fidelity__raw_status__train.v2.json"
    status.write_text(json.dumps({"schema_version": "codec_fidelity_raw_status.v2", "dataset": {"split": "train"}, "status": "AVAILABLE", "artifacts": {"observation": {"path": observation.name, "sha256": digest(observation)}, "arrays": {"path": arrays.name, "sha256": digest(arrays)}}, "unavailable_reasons": []}), encoding="utf-8")
    bundle = CodecFidelityExporter().export(ExportContext("run", public, run))
    result = CodecFidelityEvaluator().evaluate(EvaluationContext("run", public, run), ArtifactBundle("codec_fidelity", bundle.artifacts))
    metrics = result.report["metrics"]["splits"]["train"]
    assert metrics["melody"]["exact_pitch_state_rate"] == 1.0
    assert metrics["harmony"]["pitch_state_multiset_f1"] == 1.0
