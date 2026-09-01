from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluation_framework.evaluation_api import (
    ArtifactBundle,
    ArtifactEvaluator,
    ArtifactExporter,
    EvaluationModule,
    EvaluationResult,
)
from evaluation_framework.evaluation_registry import EvaluationModuleRegistry
from evaluation_framework.evaluation_runner import EvaluationMode, EvaluationRunRequest, EvaluationRunner


class FixtureExporter(ArtifactExporter):
    test_point = "fixture_point"
    input_contract = "fixture_point__inputs.v1"
    output_contract = "fixture_point__inputs.v1"

    def export(self, context):
        artifact = context.store.write_json(self.test_point, "inputs", {"schema_version": self.output_contract, "value": 7})
        return ArtifactBundle(self.test_point, {"inputs": artifact.name})


class FixtureEvaluator(ArtifactEvaluator):
    test_point = "fixture_point"

    def evaluate(self, context, bundle):
        path = context.store.run_dir / bundle.artifacts["inputs"]
        value = json.loads(path.read_text(encoding="utf-8"))["value"]
        return EvaluationResult(
            report={"schema_version": "assessment_report.v1", "status": "PASS", "value": value},
            markdown=f"# Fixture\n\nValue: {value}\n",
            figures={"plot": b"fixture-png"},
        )


def _registry() -> EvaluationModuleRegistry:
    registry = EvaluationModuleRegistry()
    registry.register(EvaluationModule("fixture_point", FixtureExporter(), FixtureEvaluator()))
    return registry


def test_runner_writes_every_module_output_to_one_run_directory(tmp_path: Path) -> None:
    store = EvaluationRunner(_registry()).run(
        EvaluationRunRequest(
            input_root=tmp_path / "input",
            output_root=tmp_path / "runs",
            run_id="run_001",
            modules=("fixture_point",),
            mode=EvaluationMode.ALL,
        )
    )
    expected = {
        "run_manifest.json",
        "index.json",
        "run_001__fixture_point__inputs.v1.json",
        "run_001__fixture_point__report.v1.json",
        "run_001__fixture_point__report.v1.md",
        "run_001__fixture_point__plot.v1.png",
    }
    assert {path.name for path in store.run_dir.iterdir()} == expected
    index = json.loads((store.run_dir / "index.json").read_text(encoding="utf-8"))
    assert index["modules"]["fixture_point"]["export"]["status"] == "COMPLETE"
    assert index["modules"]["fixture_point"]["evaluate"]["result_status"] == "PASS"


def test_runner_rejects_invalid_run_id(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="run_id"):
        EvaluationRunner(_registry()).run(
            EvaluationRunRequest(
                input_root=tmp_path,
                output_root=tmp_path,
                run_id="contains space",
                modules=("fixture_point",),
            )
        )


def test_runner_writes_directly_into_existing_generation_directory(tmp_path: Path) -> None:
    run_dir = tmp_path / "model.with.dot__20260807_120000"
    store = EvaluationRunner(_registry()).run(
        EvaluationRunRequest(
            input_root=run_dir,
            output_root=tmp_path,
            run_id="ignored",
            run_dir=run_dir,
            modules=("fixture_point",),
            options={"code_revision": "commit-a"},
        )
    )
    assert store.run_dir == run_dir.resolve()
    assert (run_dir / "fixture_point__report.v1.json").is_file()
    report = json.loads((run_dir / "fixture_point__report.v1.json").read_text(encoding="utf-8"))
    assert report["provenance"]["evaluation_execution"]["code_revision"] == "commit-a"
    assert (run_dir / "evaluation_index.json").is_file()


def test_export_failure_clears_a_stale_evaluation_status(tmp_path: Path) -> None:
    store = EvaluationRunner(_registry()).run(
        EvaluationRunRequest(
            input_root=tmp_path,
            output_root=tmp_path,
            run_id="run_001",
            modules=("fixture_point",),
        )
    )

    store.record_failure("fixture_point", "export", RuntimeError("missing public input"))

    assert "evaluate" not in store.index["modules"]["fixture_point"]


def test_registry_rejects_duplicate_module_names() -> None:
    registry = _registry()
    with pytest.raises(ValueError, match="already registered"):
        registry.register(EvaluationModule("fixture_point"))
