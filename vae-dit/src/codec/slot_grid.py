"""The canonical fixed-quantum V2 measure grid."""
from __future__ import annotations
from dataclasses import dataclass
import math

QUANTUM_QL = 0.25
CAPACITY = 48
EPSILON_QL = 1.0e-6

@dataclass(frozen=True)
class SlotGrid:
    """Valid measure slots and invalid padding for one actual bar length."""
    bar_length_ql: float
    slot_valid_mask: tuple[bool, ...]
    slot_durations_ql: tuple[float, ...]

    @property
    def valid_slot_count(self) -> int:
        return sum(self.slot_valid_mask)

    def interval(self, index: int) -> tuple[float, float]:
        if not self.slot_valid_mask[index]:
            raise ValueError("invalid padding slot has no musical interval")
        start = index * QUANTUM_QL
        return start, start + self.slot_durations_ql[index]

    @classmethod
    def for_bar(cls, bar_length_ql: float) -> "SlotGrid":
        if not math.isfinite(bar_length_ql) or bar_length_ql <= 0:
            raise ValueError("bar length must be finite and positive")
        nearest = round(bar_length_ql / QUANTUM_QL) * QUANTUM_QL
        length = nearest if nearest > 0 and abs(bar_length_ql - nearest) <= EPSILON_QL else bar_length_ql
        count = max(1, math.ceil((length - EPSILON_QL) / QUANTUM_QL))
        if count > CAPACITY:
            raise ValueError("slot_capacity_exceeded")
        durations = [0.0] * CAPACITY
        for index in range(count):
            durations[index] = min(QUANTUM_QL, length - index * QUANTUM_QL)
            if durations[index] <= 0:
                raise ValueError("slot duration is not positive")
        return cls(float(length), tuple(index < count for index in range(CAPACITY)), tuple(durations))
