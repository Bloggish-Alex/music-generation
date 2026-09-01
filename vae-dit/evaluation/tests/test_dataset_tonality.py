from __future__ import annotations

import json
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluation_framework.evaluation_registry import DEFAULT_MODULE_REGISTRY
from evaluation_framework.evaluation_runner import EvaluationMode, EvaluationRunRequest, EvaluationRunner
from evaluation_framework.evaluation_dataset_tonality import _profile_split


def _raw(split: str, song_id: str, pitches: list[int]) -> dict:
    notes = [
        {"track_index": 0, "pitch": pitch, "onset_ql": float(index), "duration_ql": 1.0, "velocity": 80}
        for index, pitch in enumerate(pitches)
    ]
    return {
        "schema_version": "dataset_tonality_raw_source.v1",
        "dataset": {"identity": "fixture", "identity_kind": "stage_label_unverified", "content_sha256": None, "split": split, "split_unit": "base_song_id"},
        "source": {"encoding_artifact_sha256": None, "note_representation": "bar_note_events", "bar_index_semantics": "zero_based_within_song"},
        "songs": [{"song_id": song_id, "base_song_id": song_id, "source_content_sha256": None, "applied_transpose_semitones": 0, "bars": [
            {"bar_index": 0, "bar_length_ql": 4.0, "time_signature": "4/4", "tempo_bpm": None, "notes": notes},
            {"bar_index": 1, "bar_length_ql": 4.0, "time_signature": "4/4", "tempo_bpm": None, "notes": []},
        ]}],
        "availability": {"bar_note_events": True, "split_membership": True, "source_content_hashes": False},
    }


def _write(root: Path, split: str, song_id: str, pitches: list[int]) -> None:
    (root / f"dataset_tonality__raw_source__{split}.v1.json").write_text(json.dumps(_raw(split, song_id, pitches)), encoding="utf-8")


def test_dataset_tonality_writes_flat_artifact_only_profile(tmp_path: Path) -> None:
    input_root = tmp_path / "input"; input_root.mkdir()
    _write(input_root, "train", "train_piece", [60, 64, 67])
    _write(input_root, "validation", "validation_piece", [62, 65, 69])
    _write(input_root, "excluded_unpaired", "one_bar_piece", [72])
    artifacts = {}
    for split in ("train", "validation", "excluded_unpaired"):
        path = input_root / f"dataset_tonality__raw_source__{split}.v1.json"
        artifacts[split] = {"path": path.name, "sha256": f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"}
    (input_root / "dataset_tonality__source_bundle.v1.json").write_text(
        json.dumps({"schema_version": "dataset_tonality_source_bundle.v1", "artifacts": artifacts}), encoding="utf-8"
    )

    store = EvaluationRunner(DEFAULT_MODULE_REGISTRY).run(EvaluationRunRequest(
        input_root=input_root, output_root=tmp_path / "runs", run_id="dataset_1", modules=("dataset_tonality",), mode=EvaluationMode.ALL,
    ))

    report = json.loads((store.run_dir / "dataset_1__dataset_tonality__report.v1.json").read_text(encoding="utf-8"))
    assert report["status"] == "MONITOR"
    assert report["metrics"]["profiles"]["train"]["base_work_count"] == 1
    assert report["metrics"]["profiles"]["train"]["augmentation_record_count"] == 1
    assert report["metrics"]["profiles"]["train"]["aggregate"]["note_count_total"] == 3
    assert report["metrics"]["profiles"]["validation"]["aggregate"]["note_count_total"] == 3
    assert "excluded_unpaired" in report["metrics"]["profiles"]
    inputs = json.loads((store.run_dir / "dataset_1__dataset_tonality__inputs.v1.json").read_text(encoding="utf-8"))
    assert inputs["splits"]["train"] == {"path": "dataset_tonality__raw_source__train.v1.json", "sha256": artifacts["train"]["sha256"]}
    assert (store.run_dir / "dataset_1__dataset_tonality__pitch_class_profile.v1.png").is_file()


def test_dataset_tonality_is_unavailable_without_validation_source(tmp_path: Path) -> None:
    input_root = tmp_path / "input"; input_root.mkdir()
    _write(input_root, "train", "train_piece", [60, 64, 67])

    store = EvaluationRunner(DEFAULT_MODULE_REGISTRY).run(EvaluationRunRequest(
        input_root=input_root, output_root=tmp_path / "runs", run_id="dataset_2", modules=("dataset_tonality",), mode=EvaluationMode.ALL,
    ))

    report = json.loads((store.run_dir / "dataset_2__dataset_tonality__report.v1.json").read_text(encoding="utf-8"))
    assert report["status"] == "UNAVAILABLE"
    assert report["missing_inputs"][0]["field"] == "validation"


def test_profile_restores_transposed_variants_and_counts_base_works_once() -> None:
    original = _raw("train", "piece", [60, 64, 67])
    raw = json.loads(json.dumps(original))
    transposed = json.loads(json.dumps(raw["songs"][0]))
    transposed["song_id"] = "piece_T+1"
    transposed["applied_transpose_semitones"] = 1
    for note in transposed["bars"][0]["notes"]:
        note["pitch"] += 1
    raw["songs"].append(transposed)

    profile = _profile_split(raw)

    assert profile["base_work_count"] == 1
    assert profile["augmentation_record_count"] == 2
    assert profile["aggregate"]["pitch_class_distribution"] == _profile_split(original)["aggregate"]["pitch_class_distribution"]
    assert profile["aggregate"]["pitch_class_distribution"] != profile["augmented_input_aggregate"]["pitch_class_distribution"]


def test_default_registry_exposes_dataset_tonality() -> None:
    assert "dataset_tonality" in DEFAULT_MODULE_REGISTRY.names()
