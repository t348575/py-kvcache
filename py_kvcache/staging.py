from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StagingSlot:
    index: int


class StagingPool:
    def __init__(self, *, slot_count: int):
        if slot_count <= 0:
            raise ValueError("slot_count must be positive")

        self.slot_count = slot_count
        self._closed = False
        self._slots = [StagingSlot(index=i) for i in range(slot_count)]
        self._free: list[int] = list(range(slot_count))

    @staticmethod
    def compute_slot_count(
        *,
        staging_mem_gib: float,
        io_size: int,
        min_slots: int = 1,
    ) -> int:
        staging_bytes = int(staging_mem_gib * (1 << 30))
        return max(max(1, min_slots), max(1, staging_bytes // io_size))

    @property
    def free_count(self) -> int:
        return len(self._free)

    def try_reserve(self) -> StagingSlot | None:
        if self._closed:
            raise RuntimeError("staging pool is closed")
        if not self._free:
            return None
        index = self._free.pop()
        return self._slots[index]

    def release(self, slot_index: int) -> None:
        if self._closed:
            return
        if slot_index < 0 or slot_index >= self.slot_count:
            raise ValueError(f"invalid staging slot index {slot_index}")
        self._free.append(slot_index)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._free.clear()


__all__ = ["StagingPool", "StagingSlot"]
