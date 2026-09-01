from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluation_framework.evaluation_registry import DEFAULT_MODULE_REGISTRY
from evaluation_framework.evaluation_runner import EvaluationMode, EvaluationRunRequest, EvaluationRunner
from evaluation_framework.evaluation_trajectory_teacher_forced_presentation import _markdown


def _sha256(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _write_generation_arm(root: Path, name: str, history_source: str, generated_pitch: int, *, seed: int = 42, future_format: str | None = None) -> Path:
    directory = root / name
    directory.mkdir()
    bars = np.zeros((4, 1, 1, 5), dtype=np.float32)
    bars[:2, ..., 0] = 0.0  # C primer, relative to base pitch 60.
    bars[2:, ..., 0] = (generated_pitch - 60) / 24.0
    bars[..., 2] = 1.0
    bars[..., 4] = 1.0
    tensor_path = directory / "bars.npz"
    np.savez_compressed(tensor_path, bars=bars, render_base_pitches=np.asarray([60, 60, 60, 60], dtype=np.float32))
    inputs = {"schema_version": "generation_metric_inputs.v1", "history_source": history_source, "primer_bars": 2}
    if future_format == "json":
        future_path = directory / "future_position.json"
        future_path.write_text(json.dumps({"target": [[[0.0], [1.0]]], "prediction": [[[0.0], [99.0]]], "future_positions": [1, 2], "valid_mask": [[True, False]]}), encoding="utf-8")
        inputs.update({"future_position_path": future_path.name, "future_position_sha256": _sha256(future_path)})
    elif future_format == "npz":
        future_path = directory / "future_position.npz"
        np.savez_compressed(future_path, target=np.asarray([[[0.0], [1.0]]], dtype=np.float32), prediction=np.asarray([[[0.0], [99.0]]], dtype=np.float32), future_positions=np.asarray([1, 2], dtype=np.int64), valid_mask=np.asarray([[1, 0]], dtype=np.int64))
        inputs.update({"future_position_path": future_path.name, "future_position_sha256": _sha256(future_path)})
    inputs_path = directory / "generation_inputs.json"
    inputs_path.write_text(json.dumps(inputs), encoding="utf-8")
    manifest_path = directory / "generation_manifest.json"
    manifest_path.write_text(json.dumps({
        "schema_version": "evaluation_manifest.v1",
        "dataset": {"dataset_hash": "dataset-1", "split": "validation", "song_id": "song-1"},
        "run": {"seed": seed, "primer_bars": 2, "total_bars": 4, "initialization": "paired_seeded", "sampling": {"steps": 16, "noise_hash": "noise-1"}},
        "checkpoint_refs": [{"role": "trajectory", "sha256": "checkpoint-1"}],
        "artifacts": {
            "generation_metric_inputs": {"path": "generation_inputs.json", "sha256": _sha256(inputs_path)},
            "bar_tensors": {"path": "bars.npz", "sha256": _sha256(tensor_path)},
        },
    }), encoding="utf-8")
    return manifest_path


def _write_pair(root: Path, *, teacher_pitch: int = 60, teacher_seed: int = 42, future_format: str | None = None) -> None:
    free = _write_generation_arm(root, "free", "generated", 61, future_format=future_format)
    teacher = _write_generation_arm(root, "teacher", "real_dataset", teacher_pitch, seed=teacher_seed, future_format=future_format)
    shared_identity = {
        "dataset_hash": "dataset-1", "split": "validation", "song_id": "song-1", "seed": 42,
        "primer_bars": 2, "total_bars": 4, "initialization": "paired_seeded",
        "sampling": {"steps": 16, "noise_hash": "noise-1"}, "checkpoint_refs": [{"role": "trajectory", "sha256": "checkpoint-1"}],
    }
    (root / "paired_experiment_manifest.json").write_text(json.dumps({
        "schema_version": "paired_experiment_manifest.v1", "pairing_group_id": "pair-1", "shared_identity": shared_identity,
        "arms": [
            {"arm": "free_running", "artifact_manifest": free.relative_to(root).as_posix()},
            {"arm": "teacher_forced", "artifact_manifest": teacher.relative_to(root).as_posix()},
        ],
    }), encoding="utf-8")


def _run(tmp_path: Path):
    return EvaluationRunner(DEFAULT_MODULE_REGISTRY).run(EvaluationRunRequest(
        input_root=tmp_path / "input", output_root=tmp_path / "runs", run_id="teacher_pair",
        modules=("trajectory_teacher_forced",), mode=EvaluationMode.ALL,
    ))


def test_teacher_forced_markdown_uses_readable_utf8_chinese_text() -> None:
    report = {
        "metrics": {
            "free_running": {
                "diatonic_fit_mean": 0.1,
                "key_match_ratio": 0.2,
                "chroma_similarity_mean": 0.3,
                "register_delta_abs_mean": 0.4,
            },
            "teacher_forced": {
                "diatonic_fit_mean": 0.2,
                "key_match_ratio": 0.3,
                "chroma_similarity_mean": 0.4,
                "register_delta_abs_mean": 0.5,
            },
            "teacher_minus_free": {
                "teacher_minus_free_diatonic_fit": 0.1,
                "teacher_minus_free_key_match_ratio": 0.1,
                "teacher_minus_free_chroma_similarity": 0.1,
                "teacher_minus_free_register_delta": 0.1,
            },
            "future_position_trajectory": {"status": "UNAVAILABLE"},
            "future_position_music_features": {"status": "UNAVAILABLE"},
        },
        "findings": [{"classification": "insufficient_evidence", "text": "fixture"}],
    }

    markdown = _markdown(report)

    assert markdown.startswith("# Teacher-forced 与 Free-running 轨迹对照")
    assert "Future-position 轨迹误差" in markdown
    assert "未提供对齐的未来位置观察，因此无法计算该项误差。" in markdown


def test_registered_module_writes_flat_artifacts_and_reports_missing_future_data(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    input_root.mkdir()
    _write_pair(input_root)
    store = _run(tmp_path)
    report = json.loads((store.run_dir / "teacher_pair__trajectory_teacher_forced__report.v1.json").read_text(encoding="utf-8"))
    assert report["findings"][0]["classification"] == "recurrent_drift_supported"
    assert report["metrics"]["future_position_trajectory"]["status"] == "UNAVAILABLE"
    assert "future_position_path" in report["missing_inputs"][0]["reason"]
    assert (store.run_dir / "teacher_pair__trajectory_teacher_forced__report.v1.md").is_file()
    assert (store.run_dir / "teacher_pair__trajectory_teacher_forced__comparison.v1.png").is_file()


def test_pair_identity_mismatch_is_recorded_as_export_failure(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    input_root.mkdir()
    _write_pair(input_root, teacher_seed=43)
    store = _run(tmp_path)
    index = json.loads((store.run_dir / "index.json").read_text(encoding="utf-8"))
    export = index["modules"]["trajectory_teacher_forced"]["export"]
    assert export["status"] == "FAIL"
    assert "share identity" in export["message"]


@pytest.mark.parametrize("future_format", ["json", "npz"])
def test_full_runner_reads_masked_future_artifacts(tmp_path: Path, future_format: str) -> None:
    input_root = tmp_path / "input"
    input_root.mkdir()
    _write_pair(input_root, future_format=future_format)
    store = _run(tmp_path)
    report = json.loads((store.run_dir / "teacher_pair__trajectory_teacher_forced__report.v1.json").read_text(encoding="utf-8"))
    metric = report["metrics"]["future_position_trajectory"]
    assert metric["status"] == "MONITOR"
    for arm in ("free_running", "teacher_forced"):
        assert metric["arms"][arm]["valid_samples_by_position"] == [1, 0]
        assert metric["arms"][arm]["mse_by_position"] == [0.0, None]
    markdown = (store.run_dir / "teacher_pair__trajectory_teacher_forced__report.v1.md").read_text(encoding="utf-8")
    assert "| 1 | 0.000 | 0.000 |" in markdown
    assert "| 2 | -- | -- | -- | -- | -- | 0 |" in markdown


def test_default_registry_exposes_teacher_forced_module() -> None:
    assert "trajectory_teacher_forced" in DEFAULT_MODULE_REGISTRY.names()
