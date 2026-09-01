from __future__ import annotations

import json
import numpy as np

from pipeline.encoding_pipeline import EncodingPipeline
from data.core import BarTensorRecord, SongRecord


def _config() -> dict:
    return {"bar_tensor": {"schema_version": "bar_tensor_schema.v2", "backend": "semantic_harmony_set_v2"}}


def test_v2_artifacts_use_canonical_row_aligned_array_names(tmp_path) -> None:
    tensor = np.zeros((18, 16, 6), dtype=np.float32); tensor[:, :, 1] = 1.0
    record = BarTensorRecord("song", 0, [18, 16, 6], tensor, {"base_pitch": None, "base_pitch_valid": False, "bar_context": [0.0] * 12})
    EncodingPipeline(_config())._write_outputs(tmp_path, [SongRecord("song", "fixture.mid")], [record])
    with np.load(tmp_path / "codec_v2_arrays.npz", allow_pickle=False) as arrays:
        assert set(arrays.files) == {"voice_tensors", "bar_contexts", "base_pitches", "base_pitch_valid"}
        assert arrays["voice_tensors"].shape == (1, 18, 16, 6)
    index = json.loads((tmp_path / "bar_tensor_index.json").read_text())
    manifest = json.loads((tmp_path / "encoding_manifest.json").read_text())
    summary = json.loads((tmp_path / "bar_feature_summary.json").read_text())
    assert index[0]["row"] == 0 and index[0]["tensor_key"] == "song__bar_000000"
    assert manifest["arrays"]["names"]["bar_contexts"]["shape"] == [1, 12]
    assert summary["feature_count"] == 31
