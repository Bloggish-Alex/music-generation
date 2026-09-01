"""Verified, containment-safe access to flat evaluation artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class ArtifactReference:
    path: str
    sha256: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ArtifactReference":
        path, digest = value.get("path"), value.get("sha256")
        if not isinstance(path, str) or not path or Path(path).name != path:
            raise ValueError("Artifact reference path must be one flat filename.")
        if not isinstance(digest, str) or not digest.startswith("sha256:"):
            raise ValueError("Artifact reference requires a sha256 digest.")
        return cls(path, digest)


class VerifiedArtifactResolver:
    """Resolve only hash-verified files below one artifact root."""

    def __init__(self, root: Path) -> None:
        self._root = Path(root).resolve()

    def path(self, value: Mapping[str, Any]) -> Path:
        reference = ArtifactReference.from_mapping(value)
        candidate = (self._root / reference.path).resolve()
        try:
            candidate.relative_to(self._root)
        except ValueError as error:
            raise ValueError("Artifact reference escapes the configured root.") from error
        if not candidate.is_file() or self.sha256(candidate) != reference.sha256:
            raise ValueError(f"Artifact reference is missing or hash-mismatched: {reference.path}")
        return candidate

    def json(self, value: Mapping[str, Any]) -> dict[str, Any]:
        return json.loads(self.path(value).read_text(encoding="utf-8"))

    def npz(self, value: Mapping[str, Any]) -> np.lib.npyio.NpzFile:
        return np.load(self.path(value), allow_pickle=False)

    @staticmethod
    def sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return f"sha256:{digest.hexdigest()}"
