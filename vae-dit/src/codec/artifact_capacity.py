"""Shared run-capacity validation for downstream dataset consumers."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from codec.slot_grid import SlotGridPolicy


def require_uniform_slot_capacity(manifests: Iterable[Mapping[str, Any]]) -> SlotGridPolicy:
    """Return the one run policy or reject mixed/invalid encoded artifacts."""
    policies = [SlotGridPolicy.from_bar_tensor_config({"slot_grid": item.get("slot_grid_policy")}) for item in manifests]
    if not policies:
        raise ValueError("at least one encoding manifest is required")
    if any(policy != policies[0] for policy in policies[1:]):
        raise ValueError("mixed slot capacities or grid policies are not trainable")
    return policies[0]
