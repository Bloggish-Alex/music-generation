#!/usr/bin/env python3
"""Style configuration loading for the DFA/HMM engine."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = ROOT_DIR / "config" / "style_defaults.yaml"


class ConfigLoader:
    """Load default YAML config and optional overrides."""

    def __init__(self, default_path: Path = DEFAULT_CONFIG_PATH) -> None:
        """Set the tool-local YAML file used as the configuration base."""
        self.default_path = Path(default_path)

    def load(self, override_path: Optional[str | Path] = None) -> Dict[str, Any]:
        """Load tool defaults and recursively merge an optional override file."""
        config = self._read_yaml(self.default_path)
        if override_path:
            self._deep_update(config, self._read_yaml(Path(override_path)))
        return config

    def _read_yaml(self, path: Path) -> Dict[str, Any]:
        """Read a YAML mapping, returning an empty mapping when absent."""
        import yaml

        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}

    def _deep_update(self, base: Dict[str, Any], overrides: Dict[str, Any]) -> None:
        """Recursively merge mappings and replace scalar or sequence leaves."""
        for key, value in overrides.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                self._deep_update(base[key], value)
            else:
                base[key] = deepcopy(value)


class ConfigView:
    """Small helper for safe nested config access."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Wrap a configuration mapping for safe section lookup."""
        self.config = config

    def section(self, name: str) -> Dict[str, Any]:
        """Return a mapping section or an empty mapping for invalid values."""
        value = self.config.get(name, {})
        return value if isinstance(value, dict) else {}
