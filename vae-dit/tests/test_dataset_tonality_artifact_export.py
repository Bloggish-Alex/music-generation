from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from export.dataset_tonality_artifact_export import DatasetTonalityArtifactExportConfig, export_dataset_tonality_artifacts


def _raw(split: str) -> dict:
    return {
        "schema_version": "dataset_tonality_raw_source.v1",
        "dataset": {"identity": "fixture", "identity_kind": "stage_label_unverified", "content_sha256": None, "split": split, "split_unit": "base_song_id"},
        "source": {"encoding_artifact_sha256": None, "note_representation": "bar_note_events", "bar_index_semantics": "zero_based_within_song"},
        "songs": [{"song_id": split, "base_song_id": split, "source_content_sha256": None, "applied_transpose_semitones": 0, "bars": []}],
        "availability": {"bar_note_events": True, "split_membership": True, "source_content_hashes": False},
    }


def test_export_copies_only_present_raw_splits_and_writes_hashed_bundle(tmp_path: Path) -> None:
    source, output = tmp_path / "model", tmp_path / "run"
    source.mkdir()
    for split in ("train", "validation"):
        (source / f"dataset_tonality__raw_source__{split}.v1.json").write_text(json.dumps(_raw(split)), encoding="utf-8")

    manifest = export_dataset_tonality_artifacts(DatasetTonalityArtifactExportConfig(source, output))

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "dataset_tonality_source_bundle.v1"
    assert set(payload["artifacts"]) == {"train", "validation"}
    assert (output / "dataset_tonality__raw_source__train.v1.json").is_file()
    assert "excluded_unpaired" not in payload["artifacts"]
