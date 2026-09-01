from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluation_framework.evaluation_registry import DEFAULT_MODULE_REGISTRY
from evaluation_framework.evaluation_runner import EvaluationMode, EvaluationRunRequest, EvaluationRunner


def _hash(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _write_input(root: Path, *, with_trace: bool = False) -> None:
    bars = np.zeros((4, 1, 1, 5), dtype=np.float32)
    bars[..., 2] = 1.0
    bars[..., 4] = 1.0
    bars[2:, ..., 0] = 1.0 / 24.0
    tensors = root / "bars.npz"
    np.savez_compressed(tensors, bars=bars, latent_mu=np.arange(16, dtype=np.float32).reshape(4, 4), render_base_pitches=np.asarray([60, 60, 61, 61]))
    generation_inputs = root / "inputs.json"
    generation_inputs.write_text(json.dumps({"schema_version": "generation_metric_inputs.v1", "history_source": "generated", "primer_bars": 2, "base_pitch_path": [60, 60, 61, 61]}), encoding="utf-8")
    artifacts = {"generation_metric_inputs": {"path": "inputs.json", "sha256": _hash(generation_inputs)}, "bar_tensors": {"path": "bars.npz", "sha256": _hash(tensors)}}
    if with_trace:
        trace = root / "trace.json"; trace.write_text(json.dumps({"overlapping_plan_disagreement": [{"register_disagreement": 2.0}]}), encoding="utf-8"); artifacts["generation_trace"] = {"path": "trace.json", "sha256": _hash(trace)}
    (root / "evaluation_manifest.json").write_text(json.dumps({"schema_version": "evaluation_manifest.v1", "dataset": {"dataset_hash": "d", "split": "validation"}, "run": {"arm": "free_running", "seed": 1}, "checkpoint_refs": [], "artifacts": artifacts, "availability": {}}), encoding="utf-8")


def test_rollout_writes_flat_outputs_and_marks_missing_trace(tmp_path: Path) -> None:
    input_root = tmp_path / "input"; input_root.mkdir(); _write_input(input_root)
    store = EvaluationRunner(DEFAULT_MODULE_REGISTRY).run(EvaluationRunRequest(input_root=input_root, output_root=tmp_path / "runs", run_id="rollout_1", modules=("trajectory_rollout",), mode=EvaluationMode.ALL))
    report = json.loads((store.run_dir / "rollout_1__trajectory_rollout__report.v1.json").read_text(encoding="utf-8"))
    assert report["metrics"]["latent_self_similarity"] != "UNAVAILABLE"
    assert report["metrics"]["plan_overlap_register_disagreement"] == "UNAVAILABLE"
    assert (store.run_dir / "rollout_1__trajectory_rollout__tonality_timeline.v1.png").is_file()


def test_rollout_discovers_trace_when_present(tmp_path: Path) -> None:
    input_root = tmp_path / "input"; input_root.mkdir(); _write_input(input_root, with_trace=True)
    store = EvaluationRunner(DEFAULT_MODULE_REGISTRY).run(EvaluationRunRequest(input_root=input_root, output_root=tmp_path / "runs", run_id="rollout_2", modules=("trajectory_rollout",), mode=EvaluationMode.ALL))
    report = json.loads((store.run_dir / "rollout_2__trajectory_rollout__report.v1.json").read_text(encoding="utf-8"))
    assert report["metrics"]["plan_overlap_register_disagreement"] == 2.0


def test_default_registry_exposes_rollout_module() -> None:
    assert "trajectory_rollout" in DEFAULT_MODULE_REGISTRY.names()
