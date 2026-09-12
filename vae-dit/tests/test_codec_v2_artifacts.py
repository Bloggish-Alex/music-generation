from __future__ import annotations

import json
import numpy as np

from pipeline.encoding_pipeline import EncodingPipeline
from data.core import BarRecord, BarTensorRecord, SongRecord


def _config() -> dict:
    return {
        "bar_tensor": {"schema_version": "bar_tensor_schema.v2", "backend": "semantic_harmony_set_v2", "slot_grid": {"quantum_ql": .25, "capacity": 92, "epsilon_ql": 1e-6}},
        "evaluation_splits": {
            "train_base_song_ids": ["song"],
            "validation_base_song_ids": ["validation"],
        },
    }


def test_v2_artifacts_use_canonical_row_aligned_array_names(tmp_path) -> None:
    tensor = np.zeros((18, 92, 6), dtype=np.float32); tensor[:, :12, 1] = 1.0
    diagnostics = {"base_pitch": None, "base_pitch_valid": False, "bar_context": [0.0] * 12, "slot_valid_mask": [True] * 12 + [False] * 80, "slot_durations_ql": [0.25] * 12 + [0.0] * 80}
    record = BarTensorRecord("song", 0, [18, 92, 6], tensor, diagnostics)
    song = SongRecord("song", "fixture.mid", bars=[BarRecord("song", "fixture.mid", 0, 3.0, source_measure_index=3)])
    EncodingPipeline(_config())._write_outputs(tmp_path, [song], [record])
    with np.load(tmp_path / "voice_tensors.npz", allow_pickle=False) as arrays:
        assert set(arrays.files) == {"voice_tensors", "slot_valid_mask", "slot_durations_ql", "bar_contexts", "base_pitches", "base_pitch_valid"}
        assert arrays["voice_tensors"].shape == (1, 18, 92, 6)
        assert arrays["slot_valid_mask"].sum() == 12
        assert arrays["slot_durations_ql"][0, 12] == 0.0
    index = json.loads((tmp_path / "bar_tensor_index.json").read_text())
    manifest = json.loads((tmp_path / "encoding_manifest.json").read_text())
    summary = json.loads((tmp_path / "bar_feature_summary.json").read_text())
    assert index[0]["row"] == 0 and index[0]["tensor_key"] == "song__bar_000000"
    assert index[0]["voice_tensor_shape"] == [18, 92, 6]
    assert manifest["arrays"]["names"]["slot_valid_mask"]["shape"] == [1, 92]
    assert manifest["slot_grid_policy"]["capacity"] == 92
    assert manifest["canonical_parser_version"] == "raw_smf_v1"
    assert manifest["supported_source_formats"] == ["smf_ppqn_type_0", "smf_ppqn_type_1"]
    assert summary["feature_count"] == 31
