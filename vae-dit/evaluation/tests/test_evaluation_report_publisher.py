from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluation_framework.evaluation_api import EvaluationResult
from evaluation_framework.evaluation_artifact_store import EvaluationArtifactStore
from evaluation_framework.evaluation_report_publisher import EvaluationReportPublisher


def test_publisher_writes_all_result_artifacts_and_execution_provenance(tmp_path: Path) -> None:
    store = EvaluationArtifactStore.create_direct(tmp_path)
    publisher = EvaluationReportPublisher(store, code_revision="abc123")
    result = EvaluationResult(
        report={"schema_version": "assessment_report.v1", "status": "PASS", "provenance": {"input": "raw"}},
        markdown="# Report\n",
        figures={"summary": b"png"},
        supplementary_json={"marker": {"status": "PASS"}},
    )

    path = publisher.publish("example", result)

    report = __import__("json").loads(path.read_text(encoding="utf-8"))
    assert report["provenance"]["input"] == "raw"
    assert report["provenance"]["evaluation_execution"]["code_revision"] == "abc123"
    assert (tmp_path / "example__report.v1.md").read_text(encoding="utf-8") == "# Report\n"
    assert (tmp_path / "example__summary.v1.png").read_bytes() == b"png"
    assert (tmp_path / "example__marker.v1.json").is_file()
