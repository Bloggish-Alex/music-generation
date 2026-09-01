from __future__ import annotations

import json
import numpy as np

from evaluation_framework.evaluation_api import ArtifactBundle
from evaluation_framework.evaluation_artifact_store import EvaluationArtifactStore
from evaluation_framework.evaluation_codec_fidelity import CodecFidelityEvaluator, CodecFidelityExporter
from evaluation_framework.evaluation_context import EvaluationContext, ExportContext


def test_v2_status_dispatches_to_monitor_report(tmp_path) -> None:
    public = tmp_path / "public"; public.mkdir(); run = EvaluationArtifactStore.create(tmp_path, "run")
    arrays = public / "codec_fidelity__raw_arrays__train.v2.npz"
    voices = np.zeros((1, 18, 16, 6), dtype=np.float32); voices[:, :, :, 1] = 1.0
    np.savez_compressed(arrays, voice_tensors=voices, bar_contexts=np.zeros((1, 12), dtype=np.float32), base_pitches=np.asarray([0], dtype=np.int16), base_pitch_valid=np.asarray([False]))
    import hashlib
    digest = lambda path: "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    source = public / "dataset_tonality__raw_source__train.v1.json"
    source.write_text(json.dumps({"songs": [{"song_id": "song", "bars": [{"bar_index": 0, "notes": []}]}]}), encoding="utf-8")
    observation = public / "codec_fidelity__raw_observation__train.v2.json"
    observation.write_text(json.dumps({"schema_version": "codec_fidelity_raw_observation.v2", "dataset": {"split": "train"}, "arrays": {"path": arrays.name, "sha256": digest(arrays)}, "source_raw": {"path": source.name, "sha256": digest(source)}, "alignment": [], "availability": {}}), encoding="utf-8")
    status = public / "codec_fidelity__raw_status__train.v2.json"
    status.write_text(json.dumps({"schema_version": "codec_fidelity_raw_status.v2", "dataset": {"split": "train"}, "status": "AVAILABLE", "artifacts": {"observation": {"path": observation.name, "sha256": digest(observation)}, "arrays": {"path": arrays.name, "sha256": digest(arrays)}}, "unavailable_reasons": []}), encoding="utf-8")
    bundle = CodecFidelityExporter().export(ExportContext("run", public, run))
    result = CodecFidelityEvaluator().evaluate(EvaluationContext("run", public, run), ArtifactBundle("codec_fidelity", bundle.artifacts))
    assert result.report["status"] == "MONITOR"
    assert result.report["metrics"]["splits"]["train"]["schema_version"] == "bar_tensor_schema.v2"
    assert "Codec Fidelity V2" in result.markdown
