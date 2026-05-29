from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Any

import numpy as np

from .profiling import add_event, is_active, now_ns


_LIBC = ctypes.CDLL(None, use_errno=True)
_LIBC.posix_memalign.argtypes = [
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.c_size_t,
    ctypes.c_size_t,
]
_LIBC.posix_memalign.restype = ctypes.c_int
_LIBC.free.argtypes = [ctypes.c_void_p]
_LIBC.free.restype = None


@dataclass(frozen=True)
class StagingSlot:
    index: int
    ptr: ctypes.c_void_p
    array: np.ndarray[Any, np.dtype[np.uint8]]
    payload_size: int

    @property
    def payload(self) -> np.ndarray[Any, np.dtype[np.uint8]]:
        return self.array[: self.payload_size]

    def clear_padding(self) -> None:
        if self.array.size > self.payload_size:
            self.array[self.payload_size :].fill(0)


class StagingPool:
    """Single-owner aligned staging buffer pool.

    Ownership invariant: only the I/O reactor thread reserves and releases
    slots, so the free list needs no lock. ``reserve``/``release`` are plain
    list operations; back pressure is applied by the reactor (it simply does
    not schedule a new file when ``try_reserve`` returns ``None``) instead of
    blocking a worker on a condition variable.
    """

    def __init__(self, *, slot_count: int, payload_size: int, io_size: int, alignment: int):
        if slot_count <= 0:
            raise ValueError("slot_count must be positive")
        if payload_size <= 0:
            raise ValueError("payload_size must be positive")
        if io_size < payload_size:
            raise ValueError("io_size must be at least payload_size")
        if alignment <= 0:
            raise ValueError("alignment must be positive")

        self.slot_count = slot_count
        self.payload_size = payload_size
        self.io_size = io_size
        self.alignment = max(alignment, ctypes.sizeof(ctypes.c_void_p))
        self._closed = False
        self._slots = [self._allocate_slot(index) for index in range(slot_count)]
        # Free list used as a stack; only the reactor thread touches it.
        self._free: list[int] = list(range(slot_count))

    @classmethod
    def from_memory_budget(
        cls,
        *,
        staging_mem_gib: float | None,
        payload_size: int,
        io_size: int,
        alignment: int,
        min_slots: int = 1,
    ) -> "StagingPool":
        staging_bytes = int(
            (staging_mem_gib if staging_mem_gib is not None else 1.0) * (1 << 30)
        )
        # ``min_slots`` is a floor (not a budget): the reactor needs enough slots
        # to keep ``iodepth`` reads in flight while extra slots drain through the
        # GPU copy phase, otherwise ``free_count`` re-couples read concurrency to
        # the copy pipeline. Honouring the floor never shrinks the budget.
        slot_count = max(max(1, min_slots), max(1, staging_bytes // io_size))
        return cls(
            slot_count=slot_count,
            payload_size=payload_size,
            io_size=io_size,
            alignment=alignment,
        )

    @property
    def free_count(self) -> int:
        return len(self._free)

    def try_reserve(self) -> StagingSlot | None:
        if self._closed:
            raise RuntimeError("staging pool is closed")
        if not self._free:
            return None
        index = self._free.pop()
        if is_active():
            free_count = len(self._free)
            now = now_ns()
            add_event(
                "py_kvcache.staging.reserve",
                "kv_offload",
                now,
                0,
                args={
                    "slot_index": index,
                    "waited": False,
                    "free_slots": free_count,
                    "total_slots": self.slot_count,
                },
            )
        return self._slots[index]

    def reserve(self) -> StagingSlot:
        slot = self.try_reserve()
        if slot is None:
            raise RuntimeError("no free staging slot (reactor should gate on try_reserve)")
        return slot

    def release(self, slot_index: int) -> None:
        if self._closed:
            return
        if slot_index < 0 or slot_index >= self.slot_count:
            raise ValueError(f"invalid staging slot index {slot_index}")
        self._free.append(slot_index)
        if is_active():
            now = now_ns()
            add_event(
                "py_kvcache.staging.release",
                "kv_offload",
                now,
                0,
                args={
                    "slot_index": slot_index,
                    "free_slots": len(self._free),
                    "total_slots": self.slot_count,
                },
            )

    def slot(self, slot_index: int) -> StagingSlot:
        if slot_index < 0 or slot_index >= self.slot_count:
            raise ValueError(f"invalid staging slot index {slot_index}")
        return self._slots[slot_index]

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._free.clear()
        for slot in self._slots:
            _LIBC.free(slot.ptr)

    def _allocate_slot(self, index: int) -> StagingSlot:
        ptr = ctypes.c_void_p()
        alloc_err = _LIBC.posix_memalign(
            ctypes.byref(ptr),
            ctypes.c_size_t(self.alignment),
            ctypes.c_size_t(self.io_size),
        )
        if alloc_err != 0 or not ptr.value:
            raise RuntimeError(f"posix_memalign() failed: {alloc_err}")
        array = np.ctypeslib.as_array(
            ctypes.cast(ptr, ctypes.POINTER(ctypes.c_uint8)),
            shape=(self.io_size,),
        )
        array.fill(0)
        return StagingSlot(
            index=index,
            ptr=ptr,
            array=array,
            payload_size=self.payload_size,
        )


__all__ = ["StagingPool", "StagingSlot"]
