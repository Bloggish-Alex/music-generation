"""Project dataset-tonality diagnostics into a flat public bundle."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Mapping


RAW_SCHEMA = "dataset_tonality_raw_source.v1"
BUNDLE_SCHEMA = "dataset_tonality_source_bundle.v1"
_SPLITS = ("train", "validation", "excluded_unpaired")


@dataclass(frozen=True)
class DatasetTonalityArtifactExportConfig:
    """Locations for source diagnostics and their public bundle."""

    source_dir: Path
    output_dir: Path


def export_dataset_tonality_artifacts(config: DatasetTonalityArtifactExportConfig) -> Path:
    """Copy allowlisted raw observations and return their manifest path."""
    source_dir, output_dir = config.source_dir.resolve(), config.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, Mapping[str, Any]] = {}
    for split in _SPLITS:
        source = source_dir / f"dataset_tonality__raw_source__{split}.v1.json"
        if not source.is_file():
            continue
        payload = json.loads(source.read_text(encoding="utf-8"))
        _validate_raw(payload, split)
        target = output_dir / source.name
        if source != target:
            shutil.copyfile(source, target)
        artifacts[split] = {"path": target.name, "sha256": _sha256(target)}
    manifest = {"schema_version": BUNDLE_SCHEMA, "artifacts": artifacts}
    manifest_path = output_dir / "dataset_tonality__source_bundle.v1.json"
    _atomic_write(manifest_path, manifest)
    return manifest_path


def _validate_raw(payload: Mapping[str, Any], split: str) -> None:
    if payload.get("schema_version") != RAW_SCHEMA:
        raise ValueError(f"Unsupported raw-source schema for {split}.")
    dataset = payload.get("dataset")
    if not isinstance(dataset, Mapping) or dataset.get("split") != split:
        raise ValueError(f"Raw-source split does not match its filename: {split}.")
    if not isinstance(payload.get("songs"), list) or not payload["songs"]:
        raise ValueError(f"Raw-source artifact {split} contains no songs.")


def _sha256(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
