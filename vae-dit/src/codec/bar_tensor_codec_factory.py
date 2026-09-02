#!/usr/bin/env python3
"""Factory for selectable bar tensor codec backends."""

from __future__ import annotations

from typing import Any, Dict

from common.config_loader import ConfigView
from codec.semantic_harmony_set_codec import SemanticHarmonySetCodec


class BarTensorCodecFactory:
    """Create the configured bar tensor codec."""

    @staticmethod
    def create(config: Dict[str, Any]) -> Any:
        """Return a codec with an encode(bar) method."""
        section = ConfigView(config).section("bar_tensor")
        backend = str(section.get("backend", "")).strip().lower()
        if backend != "semantic_harmony_set_v2":
            raise ValueError("Only final Codec V2 backend semantic_harmony_set_v2 is supported; re-encode legacy artifacts.")
        return SemanticHarmonySetCodec.from_config(config)
