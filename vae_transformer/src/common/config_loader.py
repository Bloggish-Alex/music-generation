#!/usr/bin/env python3
"""Style configuration loading for the symbolic music engine."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT_DIR / "config" / "style_defaults.yaml"


class ConfigLoader:
    """Load default YAML config and optional overrides."""

    def __init__(self, default_path: Path = DEFAULT_CONFIG_PATH) -> None:
        self.default_path = Path(default_path)

    def load(self, override_path: Optional[str | Path] = None) -> Dict[str, Any]:
        config = self._read_yaml_with_includes(self.default_path)
        if override_path:
            self._deep_update(config, self._read_yaml_with_includes(Path(override_path)))
        return config

    def _read_yaml(self, path: Path) -> Dict[str, Any]:
        import yaml

        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}

    def _read_yaml_with_includes(self, path: Path) -> Dict[str, Any]:
        path = Path(path)
        config = self._read_yaml(path)
        include = config.pop("include", None)
        if not include:
            return config
        merged: Dict[str, Any] = {}
        for include_path in self._include_paths(include, path.parent):
            self._deep_update(merged, self._read_yaml_with_includes(include_path))
        self._deep_update(merged, config)
        return merged

    def _include_paths(self, include: Any, base_dir: Path) -> list[Path]:
        values: list[Any]
        if isinstance(include, str):
            values = [include]
        elif isinstance(include, list):
            values = include
        elif isinstance(include, dict):
            values = list(include.values())
        else:
            raise ValueError("YAML include must be a string, list, or mapping.")
        paths = []
        for value in values:
            include_path = Path(str(value))
            if not include_path.is_absolute():
                include_path = base_dir / include_path
            paths.append(include_path)
        return paths

    def _deep_update(self, base: Dict[str, Any], overrides: Dict[str, Any]) -> None:
        for key, value in overrides.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                self._deep_update(base[key], value)
            else:
                base[key] = deepcopy(value)


class ConfigView:
    """Small helper for safe nested config access."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

    def section(self, name: str) -> Dict[str, Any]:
        value = self.config.get(name, {})
        return value if isinstance(value, dict) else {}
