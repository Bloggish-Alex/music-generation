from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "evaluation_framework"


def test_artifact_only_evaluation_package_does_not_import_engineering_or_legacy_assessment() -> None:
    offenders = []
    for path in PACKAGE.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        if any(_is_forbidden_import(node) for node in ast.walk(tree)):
            offenders.append(path.relative_to(PACKAGE))

    assert offenders == []


def _is_forbidden_import(node: ast.AST) -> bool:
    if isinstance(node, ast.Import):
        return any(alias.name.split(".")[0] in {"assessment", "model", "pipeline"} for alias in node.names)
    if isinstance(node, ast.ImportFrom):
        return node.level == 0 and (node.module or "").split(".")[0] in {"assessment", "model", "pipeline"}
    return False
