from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluation_framework.evaluation_context import ExportContext
from evaluation_framework.evaluation_oracle_ladder import OracleLadderExporter
from evaluation_framework.evaluation_artifact_store import EvaluationArtifactStore
from evaluation_framework.core.oracle_ledger_registry import load_oracle_boundaries


def test_oracle_ladder_boundaries_are_declared_in_the_contract_registry() -> None:
    boundaries = load_oracle_boundaries()

    assert [boundary.module for boundary in boundaries] == [
        "codec_fidelity", "dvae_fidelity", "trajectory_teacher_forced",
        "trajectory_rollout", "renderer_consistency",
    ]


def test_oracle_ladder_discovers_boundary_reports_from_runner_index(tmp_path: Path) -> None:
    report_path = tmp_path / "codec_fidelity__report.v1.json"
    report_path.write_text(json.dumps({"schema_version": "assessment_report.v1", "status": "PASS"}), encoding="utf-8")
    (tmp_path / "evaluation_index.json").write_text(json.dumps({
        "schema_version": "evaluation_run_index.v1",
        "modules": {"codec_fidelity": {"evaluate": {"status": "COMPLETE", "report": report_path.name}}},
    }), encoding="utf-8")
    store = EvaluationArtifactStore.create_direct(tmp_path)

    bundle = OracleLadderExporter().export(ExportContext("ignored", tmp_path, store))

    payload = json.loads((store.run_dir / bundle.artifacts["inputs"]).read_text(encoding="utf-8"))
    assert payload["index_reason"] is None
    assert payload["reports"][0]["test_point"] == "codec_fidelity"


def test_oracle_ladder_does_not_scan_unnamed_report_files(tmp_path: Path) -> None:
    (tmp_path / "codec_fidelity__report.v1.json").write_text("{}", encoding="utf-8")
    store = EvaluationArtifactStore.create_direct(tmp_path)

    bundle = OracleLadderExporter().export(ExportContext("ignored", tmp_path, store))

    payload = json.loads((store.run_dir / bundle.artifacts["inputs"]).read_text(encoding="utf-8"))
    assert payload["reports"] == []
    assert payload["index_reason"] is None
