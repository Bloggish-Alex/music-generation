"""Run-frozen fixed-quantum V2 measure grids."""
from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Any, Mapping

class CodecCapacityError(ValueError):
    def __init__(self, details: Mapping[str, Any]) -> None:
        self.details = {"reason": "slot_capacity_exceeded", **dict(details)}
        super().__init__("slot_capacity_exceeded: " + ", ".join(f"{k}={v}" for k, v in self.details.items() if k != "reason"))

@dataclass(frozen=True)
class SlotGridPolicy:
    quantum_ql: float
    capacity: int
    epsilon_ql: float

    @classmethod
    def from_bar_tensor_config(cls, section: Mapping[str, Any]) -> "SlotGridPolicy":
        raw = section.get("slot_grid")
        if not isinstance(raw, Mapping) or set(raw) != {"quantum_ql", "capacity", "epsilon_ql"}:
            raise ValueError("bar_tensor.slot_grid must explicitly define quantum_ql, capacity, and epsilon_ql")
        if type(raw["capacity"]) is not int or isinstance(raw["quantum_ql"], bool) or not isinstance(raw["quantum_ql"], (int, float)) or isinstance(raw["epsilon_ql"], bool) or not isinstance(raw["epsilon_ql"], (int, float)):
            raise ValueError("bar_tensor.slot_grid configuration is invalid")
        policy = cls(float(raw["quantum_ql"]), raw["capacity"], float(raw["epsilon_ql"]))
        if policy.quantum_ql != 0.25 or policy.capacity <= 0 or not 0 < policy.epsilon_ql < policy.quantum_ql:
            raise ValueError("bar_tensor.slot_grid configuration is invalid")
        return policy

    def to_dict(self) -> dict[str, int | float]:
        return {"quantum_ql": self.quantum_ql, "capacity": self.capacity, "epsilon_ql": self.epsilon_ql}

@dataclass(frozen=True)
class SlotGrid:
    """Valid measure slots and invalid padding for one actual bar length."""
    bar_length_ql: float
    policy: SlotGridPolicy
    slot_valid_mask: tuple[bool, ...]
    slot_durations_ql: tuple[float, ...]

    @property
    def valid_slot_count(self) -> int:
        return sum(self.slot_valid_mask)

    @property
    def capacity(self) -> int:
        return self.policy.capacity

    def interval(self, index: int) -> tuple[float, float]:
        if not self.slot_valid_mask[index]:
            raise ValueError("invalid padding slot has no musical interval")
        start = index * self.policy.quantum_ql
        return start, start + self.slot_durations_ql[index]

    @classmethod
    def for_bar(cls, bar_length_ql: float, policy: SlotGridPolicy, context: Mapping[str, Any] | None = None) -> "SlotGrid":
        if not math.isfinite(bar_length_ql) or bar_length_ql <= 0:
            raise ValueError("bar length must be finite and positive")
        nearest = round(bar_length_ql / policy.quantum_ql) * policy.quantum_ql
        length = nearest if nearest > 0 and abs(bar_length_ql - nearest) <= policy.epsilon_ql else bar_length_ql
        count = max(1, math.ceil((length - policy.epsilon_ql) / policy.quantum_ql))
        if count > policy.capacity:
            raise CodecCapacityError({**dict(context or {}), "bar_length_ql": float(length), "quantum_ql": policy.quantum_ql, "required_slot_count": count, "configured_slot_capacity": policy.capacity})
        durations = [0.0] * policy.capacity
        for index in range(count):
            durations[index] = min(policy.quantum_ql, length - index * policy.quantum_ql)
            if durations[index] <= 0:
                raise ValueError("slot duration is not positive")
        return cls(float(length), policy, tuple(index < count for index in range(policy.capacity)), tuple(durations))
