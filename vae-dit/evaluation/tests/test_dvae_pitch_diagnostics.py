"""Artifact-only tests for DVAE relative-pitch wiring and gradient reports."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "evaluation" / "src"))

from evaluation_framework.evaluation_registry import DEFAULT_MODULE_REGISTRY
from evaluation_framework.evaluation_runner import EvaluationMode, EvaluationRunRequest, EvaluationRunner


def _hash(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write(root: Path, name: str, payload: dict) -> Path:
    path = root / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _status(schema: str, *, split: str | None = None, available: bool = True, observation: Path | None = None) -> dict:
    payload = {"schema_version": schema, "run": {"identity": "fixture", "dataset_identity": "stage3", "dataset_identity_kind": "stage_label_unverified", "seed": 3}, "status": "AVAILABLE" if available else "UNAVAILABLE", "availability": {"observation": available}, "artifacts": {}, "unavailable_reasons": [] if available else [{"field": "diagnostic", "reason": "disabled"}]}
    if split: payload["split"] = split
    if observation: payload["artifacts"] = {"observation": {"path": observation.name, "sha256": _hash(observation)}}
    return payload


def _fixture(root: Path, gradient_available: bool = True) -> None:
    audit = _write(root, "dvae_pitch_supervision_audit__raw.v1.json", {"schema_version": "dvae_pitch_supervision_audit_raw.v1", "run": {"identity": "fixture"}, "tensor_contract": {"relative_pitch_feature_index": 0, "relative_pitch_feature_name": "relative_pitch", "pitch_scale_semitones": 24.0}, "supervision": {"loss_enabled": True, "loss_weight": 1.0, "reduction": "mean", "target_decoder_shape_match": True, "active_mask_applied_to_pitch_loss": False, "active_mask_definition": "none", "normalization": {"decoder_output_activation": "identity"}, "gradient_path_declared": {"target_detached": True, "pitch_loss_requires_grad": True}}, "decoder_parameter_groups": [{"group_id": "decoder_pitch_output", "parameter_tensor_count": 2, "parameter_element_count": 10, "available": True}], "availability": {"resolved_loss_term": True}})
    _write(root, "dvae_pitch_supervision_audit__raw_status.v1.json", _status("dvae_pitch_supervision_audit_status.v1", observation=audit))
    for split in ("train", "validation"):
        raw = _write(root, f"dvae_pitch_gradient_probe__raw__{split}.v1.json", {"schema_version": "dvae_pitch_gradient_probe_raw.v1", "run": {"identity": "fixture"}, "split": split, "probe": {"requested_batch_count": 3, "selection_policy": "first_available_batches"}, "batches": [{"decoder_parameter_groups": [{"group_id": "decoder_pitch_output", "gradient_available": True, "gradient_l2_norm": 2.0, "gradient_max_abs": 1.0, "gradient_nonzero_element_count": 5, "parameter_element_count": 10}]}, {"decoder_parameter_groups": [{"group_id": "decoder_pitch_output", "gradient_available": True, "gradient_l2_norm": 0.0, "gradient_max_abs": 0.0, "gradient_nonzero_element_count": 0, "parameter_element_count": 10}]}], "unavailable_reasons": []})
        _write(root, f"dvae_pitch_gradient_probe__raw_status__{split}.v1.json", _status("dvae_pitch_gradient_probe_status.v1", split=split, available=gradient_available, observation=raw if gradient_available else None))


def _run(tmp_path: Path, gradient_available: bool = True) -> tuple[dict, dict]:
    source = tmp_path / "input"; source.mkdir(); _fixture(source, gradient_available)
    store = EvaluationRunner(DEFAULT_MODULE_REGISTRY).run(EvaluationRunRequest(input_root=source, output_root=tmp_path / "runs", run_id="pitch", modules=("dvae_pitch_supervision_audit", "dvae_pitch_gradient_probe"), mode=EvaluationMode.ALL))
    return tuple(json.loads((store.run_dir / f"pitch__{point}__report.v1.json").read_text(encoding="utf-8")) for point in ("dvae_pitch_supervision_audit", "dvae_pitch_gradient_probe"))


def test_reports_wiring_and_gradient_observations(tmp_path: Path) -> None:
    audit, gradient = _run(tmp_path)
    assert audit["status"] == "MONITOR"
    assert audit["metrics"]["supervision"]["loss_enabled"] is True
    group = gradient["metrics"]["splits"]["train"]["groups"]["decoder_pitch_output"]
    assert gradient["status"] == "MONITOR"
    assert group["gradient_available_batch_count"] == 2
    assert group["exact_zero_gradient_batch_count"] == 1
    assert group["l2_norm_median"] == 1.0


def test_gradient_is_unavailable_when_statuses_are_unavailable(tmp_path: Path) -> None:
    _, gradient = _run(tmp_path, gradient_available=False)
    assert gradient["status"] == "UNAVAILABLE"
    assert gradient["missing_inputs"]


def test_audit_is_unavailable_when_its_status_is_not_provided(tmp_path: Path) -> None:
    source = tmp_path / "input"; source.mkdir(); _fixture(source)
    (source / "dvae_pitch_supervision_audit__raw_status.v1.json").unlink()
    store = EvaluationRunner(DEFAULT_MODULE_REGISTRY).run(EvaluationRunRequest(input_root=source, output_root=tmp_path / "runs", run_id="audit_unavailable", modules=("dvae_pitch_supervision_audit",), mode=EvaluationMode.ALL))
    report = json.loads((store.run_dir / "audit_unavailable__dvae_pitch_supervision_audit__report.v1.json").read_text(encoding="utf-8"))
    assert report["status"] == "UNAVAILABLE"


def test_registry_exposes_both_dvae_pitch_modules() -> None:
    assert {"dvae_pitch_supervision_audit", "dvae_pitch_gradient_probe"}.issubset(DEFAULT_MODULE_REGISTRY.names())
