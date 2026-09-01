"""Artifact-only DVAE reconstruction and texture-diversity assessment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .evaluation_api import ArtifactBundle, ArtifactEvaluator, ArtifactExporter, EvaluationModule, EvaluationResult
from .evaluation_context import EvaluationContext, ExportContext


TEST_POINT = "dvae_fidelity"
STATUS_SCHEMA = "dvae_fidelity_raw_status.v1"
OBSERVATION_SCHEMA = "dvae_fidelity_raw_observation.v1"
INPUT_SCHEMA = "dvae_fidelity_inputs.v1"
_SPLITS = ("train", "validation", "excluded_unpaired")


class DVAEFidelityExporter(ArtifactExporter):
    """Index public DVAE raw status artifacts without interpreting arrays."""

    test_point = TEST_POINT
    input_contract = STATUS_SCHEMA
    output_contract = INPUT_SCHEMA

    def export(self, context: ExportContext) -> ArtifactBundle:
        splits: dict[str, dict[str, str]] = {}
        availability: dict[str, str] = {}
        for split in _SPLITS:
            path = context.input_root / f"dvae_fidelity__raw_status__{split}.v1.json"
            if not path.is_file():
                availability[split] = "not_provided"
                continue
            status = _read_json(path)
            _validate_status(status, split)
            splits[split] = {"path": path.name, "sha256": _sha256(path), "schema_version": STATUS_SCHEMA}
            availability[split] = "available" if status["status"] == "AVAILABLE" else "unavailable"
        path = context.store.write_json(
            TEST_POINT,
            "inputs",
            {"schema_version": INPUT_SCHEMA, "splits": splits, "availability": availability},
        )
        return ArtifactBundle(TEST_POINT, {"inputs": path.name})


class DVAEFidelityEvaluator(ArtifactEvaluator):
    """Measure reconstruction groups and decoded-texture diversity from arrays only."""

    test_point = TEST_POINT
    required_artifacts: Sequence[str] = ("inputs",)

    def evaluate(self, context: EvaluationContext, bundle: ArtifactBundle) -> EvaluationResult:
        inputs = _read_json(context.store.run_dir / bundle.artifacts["inputs"])
        if inputs.get("schema_version") != INPUT_SCHEMA:
            raise ValueError("Unsupported DVAE fidelity input schema.")

        profiles: dict[str, dict[str, Any]] = {}
        missing: list[dict[str, str]] = []
        for split, reference in dict(inputs.get("splits") or {}).items():
            status = _load_json_reference(context.input_root, reference)
            _validate_status(status, split)
            if status["status"] != "AVAILABLE":
                missing.extend({"split": split, **reason} for reason in status["unavailable_reasons"])
                continue
            profiles[split] = _measure_split(context.input_root, status)

        if not profiles:
            report = _unavailable_report(inputs, missing)
            return EvaluationResult(report=report, markdown=_markdown(report))

        report = _report(inputs, profiles, missing)
        return EvaluationResult(
            report=report,
            markdown=_markdown(report),
            figures={"reconstruction_summary": _summary_png(profiles)},
        )


DVAE_FIDELITY_MODULE = EvaluationModule(
    TEST_POINT,
    DVAEFidelityExporter(),
    DVAEFidelityEvaluator(),
    summary="Deterministic DVAE source-to-mu-to-decoded fidelity and texture diversity.",
)


def _measure_split(root: Path, status: Mapping[str, Any]) -> dict[str, Any]:
    observation = _load_json_reference(root, status["artifacts"]["observation"])
    if observation.get("schema_version") != OBSERVATION_SCHEMA:
        raise ValueError("Unsupported DVAE fidelity observation schema.")
    arrays = _load_arrays(root, observation["arrays"])
    source = arrays["source_tensor"]
    decoded = arrays["decoded_tensor"]
    latent = arrays["latent_mu"]
    schema = observation["tensor_schema"]
    alignment = observation["alignment"]
    _validate_observation_arrays(source, latent, decoded, schema, alignment)
    _validate_array_descriptors(observation["arrays"], source, latent, decoded)

    features = _feature_indices(schema["feature_names"])
    source_state = _state_labels(source, features)
    decoded_state = _state_labels(decoded, features)
    source_active = source_state != 0
    decoded_active = decoded_state != 0
    shared_active = source_active & decoded_active
    pitch_scale = float(schema["pitch_scale_semitones"])
    source_pitch = source[..., features["relative_pitch"]] * pitch_scale
    decoded_pitch = decoded[..., features["relative_pitch"]] * pitch_scale

    metrics: dict[str, Any] = {
        "state": _state_metrics(source_state, decoded_state),
        "relative_pitch": _pitch_metrics(source_pitch, decoded_pitch, shared_active, schema),
        "chroma": _chroma_metrics(source_pitch, decoded_pitch, source_active, decoded_active),
        "register": _register_metrics(source_pitch, decoded_pitch, source_active, decoded_active, alignment),
        "velocity": _velocity_metrics(source, decoded, features, shared_active),
        "density": _density_metrics(source_active, decoded_active),
        "texture_diversity": _texture_metrics(latent, decoded),
        "feature_group_mse": _feature_group_mse(source, decoded, features),
    }
    return {
        "dataset": observation["dataset"],
        "bar_count": int(source.shape[0]),
        "latent_dimension": int(latent.shape[1]),
        "metrics": metrics,
        "provenance": {
            "checkpoint": observation["checkpoint"],
            "parent_encoded_input_manifest": observation["parent_encoded_input_manifest"],
            "decoder_projection": observation["decoder_projection"],
        },
    }


def _validate_status(payload: Mapping[str, Any], split: str) -> None:
    dataset = payload.get("dataset")
    if payload.get("schema_version") != STATUS_SCHEMA:
        raise ValueError("Unsupported DVAE fidelity raw status schema.")
    if not isinstance(dataset, Mapping) or dataset.get("split") != split:
        raise ValueError("DVAE fidelity status split does not match its filename.")
    if payload.get("status") not in {"AVAILABLE", "UNAVAILABLE"}:
        raise ValueError("DVAE fidelity status has an invalid state.")


def _validate_observation_arrays(
    source: np.ndarray,
    latent: np.ndarray,
    decoded: np.ndarray,
    schema: Mapping[str, Any],
    alignment: Sequence[Mapping[str, Any]],
) -> None:
    if source.dtype != np.float32 or decoded.dtype != np.float32 or latent.dtype != np.float32:
        raise ValueError("DVAE fidelity arrays must be float32.")
    if source.ndim != 4 or decoded.shape != source.shape or latent.ndim != 2 or latent.shape[0] != source.shape[0]:
        raise ValueError("DVAE fidelity arrays have incompatible shapes.")
    if not np.isfinite(source).all() or not np.isfinite(decoded).all() or not np.isfinite(latent).all():
        raise ValueError("DVAE fidelity arrays contain non-finite values.")
    if len(alignment) != source.shape[0]:
        raise ValueError("DVAE fidelity alignment does not match array rows.")
    if list(schema.get("axis_order") or []) != ["bar", "track", "step", "feature"]:
        raise ValueError("DVAE fidelity tensor schema has unsupported axis order.")


def _validate_array_descriptors(
    descriptors: Mapping[str, Any],
    source: np.ndarray,
    latent: np.ndarray,
    decoded: np.ndarray,
) -> None:
    for name, values in (("source_tensor", source), ("latent_mu", latent), ("decoded_tensor", decoded)):
        descriptor = descriptors.get(name)
        if not isinstance(descriptor, Mapping):
            raise ValueError(f"DVAE fidelity observation is missing {name} descriptor.")
        if descriptor.get("array_name") != name or descriptor.get("dtype") != "float32":
            raise ValueError(f"DVAE fidelity {name} descriptor has unsupported name or dtype.")
        if descriptor.get("shape") != [int(value) for value in values.shape]:
            raise ValueError(f"DVAE fidelity {name} descriptor does not match array shape.")



from .evaluation_dvae_fidelity_metrics import (
    _adjacent_deltas,
    _bar_chroma,
    _binary_f1,
    _chroma_metrics,
    _density_metrics,
    _distance_to_anchor_metrics,
    _feature_group_mse,
    _feature_indices,
    _pitch_metrics,
    _register_metrics,
    _state_labels,
    _state_metrics,
    _texture_metrics,
    _velocity_metrics,
)


from .evaluation_dvae_fidelity_presentation import _markdown, _report, _summary_png, _unavailable_report

def _load_json_reference(root: Path, reference: Mapping[str, Any]) -> dict[str, Any]:
    path = _referenced_path(root, reference)
    if _sha256(path) != reference.get("sha256"):
        raise ValueError(f"DVAE fidelity artifact hash mismatch: {path.name}")
    return _read_json(path)


def _load_arrays(root: Path, reference: Mapping[str, Any]) -> Mapping[str, np.ndarray]:
    path = _referenced_path(root, reference)
    if _sha256(path) != reference.get("sha256"):
        raise ValueError(f"DVAE fidelity array hash mismatch: {path.name}")
    with np.load(path, allow_pickle=False) as archive:
        required = ("source_tensor", "latent_mu", "decoded_tensor")
        if set(archive.files) != set(required):
            raise ValueError("DVAE fidelity NPZ has unsupported array names.")
        return {name: np.asarray(archive[name]) for name in required}


def _referenced_path(root: Path, reference: Mapping[str, Any]) -> Path:
    if not isinstance(reference.get("path"), str):
        raise ValueError("DVAE fidelity artifact reference is missing a path.")
    path = (root / reference["path"]).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("DVAE fidelity artifact path must stay below input root.") from error
    if not path.is_file():
        raise FileNotFoundError(f"Missing DVAE fidelity artifact: {path.name}")
    return path


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"
