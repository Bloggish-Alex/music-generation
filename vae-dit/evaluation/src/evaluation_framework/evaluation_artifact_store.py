"""Single-directory storage and index management for one evaluation run."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping

from .evaluation_api import ArtifactBundle, EvaluationResult


_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_TEST_POINT = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass
class EvaluationArtifactStore:
    """Owns the only directory in which a framework run may write output."""

    run_dir: Path
    run_id: str
    index: Dict[str, Any]
    direct: bool = False

    @classmethod
    def create(cls, output_root: Path, run_id: str, *, resume: bool = False) -> "EvaluationArtifactStore":
        if not _RUN_ID.fullmatch(run_id):
            raise ValueError("run_id must contain only letters, digits, '_' and '-'.")
        run_dir = Path(output_root).resolve() / run_id
        if run_dir.exists() and not resume:
            raise FileExistsError(f"Evaluation run directory already exists: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)
        index_path = run_dir / "index.json"
        if resume and index_path.exists():
            index = json.loads(index_path.read_text(encoding="utf-8"))
        else:
            index = {
                "schema_version": "evaluation_run_index.v1",
                "run_id": run_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "modules": {},
            }
        store = cls(run_dir=run_dir, run_id=run_id, index=index)
        store._write_json("run_manifest.json", {
            "schema_version": "evaluation_run_manifest.v1",
            "run_id": run_id,
            "run_directory": run_dir.name,
        })
        store._write_index()
        return store

    @classmethod
    def create_direct(cls, run_dir: Path, *, resume: bool = True) -> "EvaluationArtifactStore":
        run_dir = Path(run_dir).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        index_path = run_dir / "evaluation_index.json"
        if resume and index_path.exists():
            index = json.loads(index_path.read_text(encoding="utf-8"))
        else:
            index = {
                "schema_version": "evaluation_run_index.v1",
                "run_id": run_dir.name,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "modules": {},
            }
        store = cls(run_dir=run_dir, run_id=run_dir.name, index=index, direct=True)
        store._write_json("evaluation_run_manifest.json", {
            "schema_version": "evaluation_run_manifest.v1",
            "run_id": run_dir.name,
            "run_directory": run_dir.name,
        })
        store._write_index()
        return store

    def write_json(self, test_point: str, role: str, payload: Mapping[str, Any], *, version: int = 1) -> Path:
        path = self.path_for(test_point, role, "json", version=version)
        self._write_json(path.name, payload)
        return path

    def write_text(self, test_point: str, role: str, value: str, *, extension: str, version: int = 1) -> Path:
        path = self.path_for(test_point, role, extension, version=version)
        self._atomic_write(path, value.encode("utf-8"))
        return path

    def write_bytes(self, test_point: str, role: str, value: bytes, *, extension: str, version: int = 1) -> Path:
        path = self.path_for(test_point, role, extension, version=version)
        self._atomic_write(path, value)
        return path

    def path_for(self, test_point: str, role: str, extension: str, *, version: int = 1) -> Path:
        if not _TEST_POINT.fullmatch(test_point):
            raise ValueError(f"Invalid test_point name: {test_point}")
        normalized_role = str(role).replace(" ", "_")
        if not _TEST_POINT.fullmatch(normalized_role):
            raise ValueError(f"Invalid artifact role: {role}")
        if int(version) <= 0:
            raise ValueError("Artifact version must be positive.")
        prefix = "" if self.direct else f"{self.run_id}__"
        return self.run_dir / f"{prefix}{test_point}__{normalized_role}.v{int(version)}.{extension}"

    def record_export(self, bundle: ArtifactBundle) -> None:
        self._module_entry(bundle.test_point)["export"] = {
            "status": "COMPLETE",
            "bundle": bundle.to_dict(),
        }
        self._write_index()

    def record_evaluation(self, test_point: str, result: EvaluationResult, report_path: Path) -> None:
        entry = self._module_entry(test_point)
        entry["evaluate"] = {
            "status": "COMPLETE",
            "report": report_path.name,
            "result_status": str(result.report.get("status", "MONITOR")),
        }
        self._write_index()

    def record_failure(self, test_point: str, phase: str, error: Exception) -> None:
        entry = self._module_entry(test_point)
        entry[phase] = {
            "status": "FAIL",
            "error_type": type(error).__name__,
            "message": str(error),
        }
        if phase == "export":
            entry.pop("evaluate", None)
        self._write_index()

    def bundle_for(self, test_point: str) -> ArtifactBundle:
        entry = self._module_entry(test_point).get("export")
        if not entry or entry.get("status") != "COMPLETE":
            raise FileNotFoundError(f"No exported artifact bundle recorded for '{test_point}'.")
        return ArtifactBundle.from_dict(entry["bundle"])

    def _module_entry(self, test_point: str) -> Dict[str, Any]:
        if not _TEST_POINT.fullmatch(test_point):
            raise ValueError(f"Invalid test_point name: {test_point}")
        return self.index.setdefault("modules", {}).setdefault(test_point, {})

    def _write_index(self) -> None:
        self._write_json("evaluation_index.json" if self.direct else "index.json", self.index)

    def _write_json(self, filename: str, payload: Mapping[str, Any]) -> None:
        self._atomic_write(self.run_dir / filename, json.dumps(payload, indent=2).encode("utf-8"))

    @staticmethod
    def _atomic_write(path: Path, value: bytes) -> None:
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_bytes(value)
        temporary.replace(path)
