"""Optional staging-data cache: retain completed CPU-staging buffers keyed by
block hash so a later load for the same hash is a staging-resident hit (one
staging->GPU copy, no disk read) instead of a re-open + re-read.

Safe because storage blocks are immutable and hash-addressed: a cached buffer is
always valid for its hash. This is read-side only -- store buffers are never
cached, so writes still persist to disk immediately.

Pure-Python (no torch/CUDA). The cache holds staging slot indices checked out of
the reactor's pool but never calls the pool itself -- it returns slot indices for
the reactor to release, preserving the pool's single-owner invariant.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass
from typing import Iterator, Protocol


class EvictionPolicy(Protocol):
    def admit(self, key: bytes) -> None: ...
    def touch(self, key: bytes) -> None: ...
    def evict(self, key: bytes) -> None: ...
    def victims(self) -> Iterator[bytes]: ...
    def clear(self) -> None: ...


class LruPolicy:
    """Least-recently-used ordering over resident keys."""

    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._od: "collections.OrderedDict[bytes, None]" = collections.OrderedDict()

    def admit(self, key: bytes) -> None:
        self._od[key] = None
        self._od.move_to_end(key)

    def touch(self, key: bytes) -> None:
        if key in self._od:
            self._od.move_to_end(key)

    def evict(self, key: bytes) -> None:
        self._od.pop(key, None)

    def victims(self) -> Iterator[bytes]:
        return iter(list(self._od))  # oldest first

    def clear(self) -> None:
        self._od.clear()


class ArcPolicy:
    """ARC-style adaptive policy (Megiddo & Modha): T1/T2 resident lists +
    B1/B2 ghost lists (keys only) with an adaptive target ``p`` for |T1|.

    Adapted for pull-based eviction: ``admit``/``touch`` drive adaptation on
    every access; the resident eviction (REPLACE) is deferred to ``victims()``
    (the reactor pulls a victim when it needs a slot), with evicted residents
    demoted to the matching ghost list. Scan resistance -- the reason to choose
    ARC over LRU -- comes from one-shot keys living in T1/B1 while reused keys
    are promoted to T2.
    """

    def __init__(self, capacity: int) -> None:
        self._c = max(1, capacity)
        self._p = 0
        self._t1: "collections.OrderedDict[bytes, None]" = collections.OrderedDict()
        self._t2: "collections.OrderedDict[bytes, None]" = collections.OrderedDict()
        self._b1: "collections.OrderedDict[bytes, None]" = collections.OrderedDict()
        self._b2: "collections.OrderedDict[bytes, None]" = collections.OrderedDict()

    def admit(self, key: bytes) -> None:
        if key in self._b1:
            delta = 1 if not self._b1 else max(1, len(self._b2) // max(1, len(self._b1)))
            self._p = min(self._c, self._p + delta)
            self._b1.pop(key, None)
            self._t2[key] = None
        elif key in self._b2:
            delta = 1 if not self._b2 else max(1, len(self._b1) // max(1, len(self._b2)))
            self._p = max(0, self._p - delta)
            self._b2.pop(key, None)
            self._t2[key] = None
        else:
            self._t1[key] = None
        self._trim_ghosts()

    def touch(self, key: bytes) -> None:
        if key in self._t1:
            self._t1.pop(key, None)
            self._t2[key] = None
        elif key in self._t2:
            self._t2.move_to_end(key)

    def evict(self, key: bytes) -> None:
        if key in self._t1:
            self._t1.pop(key, None)
            self._b1[key] = None
        elif key in self._t2:
            self._t2.pop(key, None)
            self._b2[key] = None
        self._trim_ghosts()

    def victims(self) -> Iterator[bytes]:
        # REPLACE preference: evict from T1 when it exceeds the target p.
        if len(self._t1) > self._p:
            first, second = self._t1, self._t2
        else:
            first, second = self._t2, self._t1
        yield from list(first)  # LRU-first
        yield from list(second)

    def clear(self) -> None:
        self._t1.clear()
        self._t2.clear()
        self._b1.clear()
        self._b2.clear()
        self._p = 0

    @property
    def p(self) -> int:
        return self._p

    def _trim_ghosts(self) -> None:
        while len(self._b1) > self._c:
            self._b1.popitem(last=False)
        while len(self._b2) > self._c:
            self._b2.popitem(last=False)


@dataclass
class _CacheSlot:
    block_hash: bytes
    slot_index: int
    copies_inflight: int = 0  # loads mid staging->GPU DMA out of this slot


def make_policy(kind: str, capacity: int) -> EvictionPolicy:
    if kind == "lru":
        return LruPolicy(capacity)
    if kind == "arc":
        return ArcPolicy(capacity)
    raise ValueError(f"unknown staging_cache policy {kind!r}")


class StagingDataCache:
    """Hash-addressed retention of completed staging slots, bounded at
    ``capacity`` slots, evictable on demand. Never touches the staging pool:
    returns slot indices for the reactor to release."""

    def __init__(self, *, policy: str, capacity: int) -> None:
        self._capacity = max(0, capacity)
        self._policy = make_policy(policy, self._capacity)
        self._slots: dict[bytes, _CacheSlot] = {}

    def __contains__(self, block_hash: bytes) -> bool:
        return block_hash in self._slots

    def __len__(self) -> int:
        return len(self._slots)

    @property
    def policy(self) -> EvictionPolicy:
        return self._policy

    def get(self, block_hash: bytes) -> _CacheSlot | None:
        slot = self._slots.get(block_hash)
        if slot is not None:
            self._policy.touch(block_hash)
        return slot

    def put(self, block_hash: bytes, slot_index: int) -> int | None:
        """Retain ``slot_index`` under ``block_hash``. Returns a slot index the
        caller must release: the passed index if the hash is already cached
        (dup -- keep the existing copy) or a self-evicted victim when at
        capacity; otherwise ``None``."""
        if self._capacity == 0:
            return slot_index
        if block_hash in self._slots:
            return slot_index
        freed: int | None = None
        if len(self._slots) >= self._capacity:
            freed = self.evict_one()
        self._slots[block_hash] = _CacheSlot(block_hash=block_hash, slot_index=slot_index)
        self._policy.admit(block_hash)
        return freed

    def pin(self, slot: _CacheSlot) -> None:
        slot.copies_inflight += 1

    def unpin(self, slot: _CacheSlot) -> None:
        slot.copies_inflight -= 1

    def evict_one(self) -> int | None:
        """Drop the policy victim with no in-flight copy; return its slot index
        for the reactor to release, or ``None`` if empty / all pinned."""
        for block_hash in self._policy.victims():
            slot = self._slots.get(block_hash)
            if slot is None or slot.copies_inflight > 0:
                continue
            del self._slots[block_hash]
            self._policy.evict(block_hash)
            return slot.slot_index
        return None

    def drain(self) -> list[int]:
        indices = [slot.slot_index for slot in self._slots.values()]
        self._slots.clear()
        self._policy.clear()
        return indices


__all__ = [
    "ArcPolicy",
    "EvictionPolicy",
    "LruPolicy",
    "StagingDataCache",
    "make_policy",
]
