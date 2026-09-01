"""Shared export/evaluation orchestration for one flat result directory."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

from .evaluation_artifact_store import EvaluationArtifactStore
from .evaluation_context import EvaluationContext, ExportContext
from .evaluation_registry import EvaluationModuleRegistry
from .evaluation_report_publisher import EvaluationReportPublisher


class EvaluationMode(str, Enum):
    EXPORT = "export"
    EVALUATE = "evaluate"
    ALL = "all"


@dataclass(frozen=True)
class EvaluationRunRequest:
    input_root: Path
    output_root: Path
    run_id: str
    modules: Sequence[str] | str = "all"
    mode: EvaluationMode = EvaluationMode.ALL
    resume: bool = False
    run_dir: Path | None = None
    options: Mapping[str, Any] = field(default_factory=dict)


class EvaluationRunner:
    """Runs registered modules without allowing them to create result directories."""

    def __init__(self, registry: EvaluationModuleRegistry) -> None:
        self.registry = registry

    def run(self, request: EvaluationRunRequest) -> EvaluationArtifactStore:
        store = EvaluationArtifactStore.create_direct(request.run_dir, resume=True) if request.run_dir else EvaluationArtifactStore.create(request.output_root, request.run_id, resume=request.resume)
        modules = self.registry.select(request.modules)
        if not modules:
            raise ValueError("No evaluation modules were selected or registered.")
        export_context = ExportContext(request.run_id, Path(request.input_root).resolve(), store, request.options)
        evaluation_context = EvaluationContext(request.run_id, Path(request.input_root).resolve(), store, request.options)
        publisher = EvaluationReportPublisher(
            store=store,
            code_revision=str(request.options.get("code_revision", "unknown")),
        )
        for module in modules:
            if not self._dependencies_complete(module.requires, store):
                store.record_failure(module.test_point, "dependency", RuntimeError("Required module artifact is unavailable."))
                continue
            if request.mode in {EvaluationMode.EXPORT, EvaluationMode.ALL}:
                if module.exporter is None:
                    store.record_failure(module.test_point, "export", RuntimeError("Module has no exporter."))
                    continue
                try:
                    bundle = module.exporter.export(export_context)
                    store.record_export(bundle)
                except Exception as error:
                    store.record_failure(module.test_point, "export", error)
                    continue
            if request.mode in {EvaluationMode.EVALUATE, EvaluationMode.ALL}:
                if module.evaluator is None:
                    store.record_failure(module.test_point, "evaluate", RuntimeError("Module has no evaluator."))
                    continue
                try:
                    bundle = store.bundle_for(module.test_point)
                    result = module.evaluator.evaluate(evaluation_context, bundle)
                    publisher.publish(module.test_point, result)
                except Exception as error:
                    store.record_failure(module.test_point, "evaluate", error)
        return store

    @staticmethod
    def _dependencies_complete(requires: Sequence[str], store: EvaluationArtifactStore) -> bool:
        for test_point in requires:
            try:
                store.bundle_for(test_point)
            except FileNotFoundError:
                return False
        return True
