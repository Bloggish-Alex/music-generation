from __future__ import annotations

import pytest

from codec.artifact_capacity import require_uniform_slot_capacity


def _manifest(capacity: int):
    return {"slot_grid_policy": {"quantum_ql": .25, "capacity": capacity, "epsilon_ql": 1e-6}}


def test_uniform_capacity_helper_freezes_training_shape() -> None:
    assert require_uniform_slot_capacity([_manifest(92), _manifest(92)]).capacity == 92


def test_uniform_capacity_helper_rejects_mixed_artifacts() -> None:
    with pytest.raises(ValueError, match="mixed slot capacities"):
        require_uniform_slot_capacity([_manifest(48), _manifest(92)])
