"""Unit tests for the shared (ref-counted) preload staging mode.

When ``preload_share_staging`` is on, identical-prefix preload candidates share
one disk read into one staging slot; each demander gets its own staging->GPU
copy, and the slot is released only once it is no longer discoverable
(``cached`` False) and no copy still reads from it (``copies_inflight == 0``).

These exercise the pure refcount/lifecycle helpers on an ``IoReactor`` built via
``__new__`` with hand-set state (no CUDA/vLLM/ring). A stub records the
``_queue_load_copy`` call in place of the CUDA copy. They verify intake dedup +
refcount, claim accounting, the release invariant in both directions (retain
while a copy is in flight, release on the last claim), the eviction safety net,
read-complete serving every waiter, and store-deferred wake.
"""

from __future__ import annotations

import collections
import unittest
from types import SimpleNamespace

try:
    import numpy  # noqa: F401

    from py_kvcache.reactor import (
        IoReactor,
        _PreloadInfo,
        _PreloadRequest,
        _SharedPreloadSlot,
    )

    HAVE_DEPS = True
except Exception:  # pragma: no cover - minimal workspace
    HAVE_DEPS = False


class FakePool:
    def __init__(self, n: int) -> None:
        self.slot_count = n
        self._free = list(range(n))
        self.released: list[int] = []

    def try_reserve(self):
        if not self._free:
            return None
        return SimpleNamespace(index=self._free.pop())

    def release(self, idx: int) -> None:
        self._free.append(idx)
        self.released.append(idx)

    @property
    def free_count(self) -> int:
        return len(self._free)


IO_SIZE = 4096


def _shared_reactor(*, pool_slots: int = 8, iodepth: int = 4) -> "IoReactor":
    r = IoReactor.__new__(IoReactor)
    r.iodepth = iodepth
    r.staging_pool = FakePool(pool_slots)
    r.file_store = SimpleNamespace(io_size=IO_SIZE)
    r._share_preload = True
    r._preload_refcount = {}
    r._shared_cached = {}
    r._max_preload_slots = max(0, pool_slots - iodepth)
    r._ready_fds_load = collections.deque()
    r._ready_fds_preload = collections.deque()
    r._active = []
    r._inflight = {}
    r._pending_copies = []
    r._copy_ready = []
    r._preload_pending = collections.deque()
    r._preload_pending_count = {}
    r._preload_pending_cancel = {}
    r._preload_inflight_hashes = {}
    r._preload_slots = collections.OrderedDict()
    r._preload_cached_total = 0
    r._preload_inflight_total = 0
    r._preload_owned = collections.OrderedDict()
    r._max_preload_owned = 16384
    r._preload_waiters = {}
    r._store_inflight = {}
    r._preload_blocked_on_write = {}
    r._known_missing = collections.OrderedDict()
    r._max_known_missing = 4096
    r._data_inflight = 0
    r._stop = False
    r._staging_cache = None
    r._max_cache_slots = max(0, pool_slots - iodepth)
    # Record staging->GPU copies instead of launching CUDA.
    r._copy_calls = []  # type: ignore[attr-defined]

    def _fake_queue_load_copy(job, file_index, slot_index, *, shared=None, source="file"):
        r._copy_calls.append((job, file_index, slot_index, shared))

    r._queue_load_copy = _fake_queue_load_copy  # type: ignore[assignment]
    r._async_close = lambda fd: None  # type: ignore[assignment]
    return r


def _job(req_id: str, *, job_id: int = 0, hashes=(b"h",)):
    return SimpleNamespace(
        job_id=job_id,
        failed=None,
        future_set=False,
        is_store=False,
        next_file_index=0,
        inflight_files=0,
        total_files=len(hashes),
        block_hashes=list(hashes),
        profile=SimpleNamespace(req_id=req_id, profile_tid="kv_load"),
    )


def _info(h: bytes = b"h", *, req_id: str = "r0", file_index: int = 0):
    return _PreloadInfo(
        block_hash=h,
        preload_id=f"{req_id}:0:1",
        req_id=req_id,
        profile_tid="kv_preload",
        file_index=file_index,
        total_files=1,
    )


def _seed_cached(r, h: bytes, slot_index: int, *, copies_inflight: int = 0):
    slot = _SharedPreloadSlot(
        block_hash=h, slot_index=slot_index, preload_info=_info(h), cached=True
    )
    slot.copies_inflight = copies_inflight
    r._shared_cached[h] = slot
    r._preload_cached_total += 1
    return slot


@unittest.skipUnless(HAVE_DEPS, "numpy / py_kvcache not importable")
class SharedIntakeTest(unittest.TestCase):
    def test_n_requests_one_pending_refcount_n(self):
        r = _shared_reactor()
        h = b"shared"
        for i in range(3):
            r._intake(_PreloadRequest([h], preload_id=f"r{i}:0:1", req_id=f"r{i}"))
        # One disk read queued; demand counted three times.
        self.assertEqual([i.block_hash for i in r._preload_pending], [h])
        self.assertEqual(r._preload_refcount[h], 3)

    def test_same_req_dedups(self):
        r = _shared_reactor()
        h = b"shared"
        r._intake(_PreloadRequest([h], preload_id="r0:0:1", req_id="r0"))
        r._intake(_PreloadRequest([h], preload_id="r0:0:1", req_id="r0"))
        self.assertEqual(len(r._preload_pending), 1)
        self.assertEqual(r._preload_refcount[h], 1)

    def test_second_demander_of_staged_hash_only_bumps_refcount(self):
        r = _shared_reactor()
        h = b"shared"
        _seed_cached(r, h, 5)
        r._intake(_PreloadRequest([h], preload_id="r9:0:1", req_id="r9"))
        # Already staged -> no new pending, just demand.
        self.assertEqual(len(r._preload_pending), 0)
        self.assertEqual(r._preload_refcount[h], 1)


@unittest.skipUnless(HAVE_DEPS, "numpy / py_kvcache not importable")
class SharedRefcountReleaseTest(unittest.TestCase):
    def test_retain_until_last_demander_claims(self):
        r = _shared_reactor()
        h = b"h"
        slot = _seed_cached(r, h, 7)
        r._preload_refcount[h] = 2
        for rid in ("r0", "r1"):
            r._preload_owned[(rid, h)] = None

        # First claim: slot stays cached (one demander left).
        r._shared_claim_from_cache(_job("r0"), 0)
        self.assertIn(h, r._shared_cached)
        self.assertEqual(r._preload_refcount[h], 1)
        self.assertEqual(slot.copies_inflight, 1)
        self.assertEqual(r.staging_pool.released, [])

        # Second (last) claim: uncached now, but a copy is still in flight.
        r._shared_claim_from_cache(_job("r1"), 0)
        self.assertNotIn(h, r._shared_cached)
        self.assertNotIn(h, r._preload_refcount)
        self.assertEqual(slot.copies_inflight, 2)
        self.assertEqual(r.staging_pool.released, [])  # invariant: copies in flight

        # Copies drain (as _drain_cuda_copies would): release on the last.
        slot.copies_inflight -= 1
        r._maybe_release_shared(slot)
        self.assertEqual(r.staging_pool.released, [])
        slot.copies_inflight -= 1
        r._maybe_release_shared(slot)
        self.assertEqual(r.staging_pool.released, [7])

    def test_final_release_demotes_to_cache_when_enabled(self):
        from py_kvcache.staging_cache import StagingDataCache

        r = _shared_reactor(pool_slots=8, iodepth=4)
        r._staging_cache = StagingDataCache(policy="lru", capacity=4)
        h = b"h"
        slot = _seed_cached(r, h, 7)
        r._preload_refcount[h] = 1
        r._preload_owned[("r0", h)] = None
        r._shared_claim_from_cache(_job("r0"), 0)  # last demander -> uncached
        slot.copies_inflight -= 1
        r._maybe_release_shared(slot)  # no copies left -> demote, not release
        self.assertEqual(r.staging_pool.released, [])  # NOT returned to pool
        self.assertIn(h, r._staging_cache)  # demoted into the cache
        self.assertEqual(r._staging_cache.get(h).slot_index, 7)

    def test_decref_releases_immediately_when_no_copy_inflight(self):
        r = _shared_reactor()
        h = b"h"
        _seed_cached(r, h, 3)
        r._preload_refcount[h] = 1
        r._preload_owned[("r0", h)] = None
        # A demander dropped without ever claiming (e.g. served by GPU prefix cache
        # path that still decremented): refcount hits 0, slot free, release now.
        r._shared_decref("r0", h)
        self.assertEqual(r.staging_pool.released, [3])
        self.assertNotIn(h, r._shared_cached)

    def test_non_owner_load_does_not_touch_refcount(self):
        r = _shared_reactor()
        h = b"h"
        _seed_cached(r, h, 1)
        r._preload_refcount[h] = 2
        r._preload_owned[("r0", h)] = None
        r._preload_owned[("r1", h)] = None
        # A request that never preloaded h claims it: refcount untouched.
        r._shared_decref("stranger", h)
        self.assertEqual(r._preload_refcount[h], 2)
        self.assertEqual(r.staging_pool.released, [])


@unittest.skipUnless(HAVE_DEPS, "numpy / py_kvcache not importable")
class SharedEvictionTest(unittest.TestCase):
    def test_evict_orphan_reclaims_and_clears_refcount(self):
        r = _shared_reactor()
        h = b"h"
        _seed_cached(r, h, 2)
        r._preload_refcount[h] = 1  # leaked demander that never ran
        r._preload_owned[("r0", h)] = None
        self.assertTrue(r._evict_one_shared_slot())
        self.assertEqual(r.staging_pool.released, [2])
        self.assertNotIn(h, r._shared_cached)
        self.assertNotIn(h, r._preload_refcount)
        self.assertNotIn(("r0", h), r._preload_owned)
        self.assertEqual(r._preload_cached_total, 0)

    def test_evict_skips_slot_with_inflight_copy(self):
        r = _shared_reactor()
        busy = b"busy"
        free = b"free"
        _seed_cached(r, busy, 0, copies_inflight=1)
        _seed_cached(r, free, 1)
        self.assertTrue(r._evict_one_shared_slot())
        # The busy slot is not actually free; eviction takes the idle one.
        self.assertEqual(r.staging_pool.released, [1])
        self.assertIn(busy, r._shared_cached)

    def test_evict_empty_returns_false(self):
        r = _shared_reactor()
        self.assertFalse(r._evict_one_shared_slot())

    def test_reserve_foreground_slot_evicts_shared_when_pool_full(self):
        r = _shared_reactor(pool_slots=1, iodepth=1)
        # Drain the pool, then park a cached shared slot to reclaim.
        r.staging_pool.try_reserve()
        _seed_cached(r, b"h", 0)
        slot = r._reserve_foreground_slot()
        self.assertIsNotNone(slot)
        self.assertNotIn(b"h", r._shared_cached)

    def test_store_evicts_shared_preload_when_pool_full(self):
        r = _shared_reactor(pool_slots=1, iodepth=1)
        r.staging_pool.try_reserve()
        _seed_cached(r, b"h", 0)
        scheduled: list[int] = []
        r._schedule_store_copy = lambda job, fi, slot_index: scheduled.append(slot_index)
        store_job = SimpleNamespace(
            failed=None, future_set=False, is_store=True,
            next_file_index=0, total_files=1, block_hashes=[b"s"],
        )
        self.assertTrue(r._schedule_one(store_job))
        self.assertEqual(len(scheduled), 1)  # store reclaimed the slot
        self.assertNotIn(b"h", r._shared_cached)  # idle shared preload evicted


@unittest.skipUnless(HAVE_DEPS, "numpy / py_kvcache not importable")
class SharedReadCompleteTest(unittest.TestCase):
    def test_one_read_serves_all_waiters_then_caches(self):
        r = _shared_reactor()
        h = b"h"
        r._preload_refcount[h] = 3  # three demanders
        for rid in ("r0", "r1", "r2"):
            r._preload_owned[(rid, h)] = None
        r._preload_inflight_add(h)
        # Two of the three have arrived as real loads and parked as waiters.
        ja, jb = _job("r0", job_id=1), _job("r1", job_id=2)
        for j in (ja, jb):
            j.inflight_files = 1
        r._preload_waiter_add(h, ja, 0)
        r._preload_waiter_add(h, jb, 0)

        op = SimpleNamespace(
            preload_hash=h, preload_info=_info(h), slot_index=4, fd=9, start_ns=0
        )
        r._on_shared_preload_read_complete(op, IO_SIZE)

        # Both waiters served from the single read, each from the same slot.
        served = [(c[0].profile.req_id, c[2], c[3]) for c in r._copy_calls]
        self.assertEqual([s[0] for s in served], ["r0", "r1"])
        self.assertTrue(all(s[1] == 4 for s in served))
        slot = r._shared_cached[h]
        self.assertEqual(slot.copies_inflight, 2)
        # One demander (r2) still outstanding -> slot stays cached.
        self.assertEqual(r._preload_refcount[h], 1)
        self.assertTrue(slot.cached)
        self.assertEqual(r._preload_cached_total, 1)
        self.assertEqual(r.staging_pool.released, [])

    def test_read_with_no_remaining_demand_releases_after_copies(self):
        r = _shared_reactor()
        h = b"h"
        r._preload_refcount[h] = 1
        r._preload_owned[("r0", h)] = None
        r._preload_inflight_add(h)
        j = _job("r0")
        j.inflight_files = 1
        r._preload_waiter_add(h, j, 0)

        op = SimpleNamespace(
            preload_hash=h, preload_info=_info(h), slot_index=6, fd=3, start_ns=0
        )
        r._on_shared_preload_read_complete(op, IO_SIZE)

        # Sole demander served -> not cached, but its copy is still in flight.
        self.assertNotIn(h, r._shared_cached)
        self.assertEqual(r.staging_pool.released, [])
        self.assertEqual(len(r._copy_calls), 1)
        shared = r._copy_calls[0][3]
        self.assertIsNotNone(shared)
        shared.copies_inflight -= 1
        r._maybe_release_shared(shared)
        self.assertEqual(r.staging_pool.released, [6])


@unittest.skipUnless(HAVE_DEPS, "numpy / py_kvcache not importable")
class SharedStoreDeferredTest(unittest.TestCase):
    def test_write_requeues_one_readback_for_all_refholders(self):
        r = _shared_reactor()
        h = b"h"
        r._preload_refcount[h] = 3
        r._store_inflight[h] = 1
        r._preload_blocked_on_write[h] = collections.deque([_info(h)])
        r._release_store_inflight(h, ok=True)
        # Exactly one read-back queued; refcount carries the multiplicity.
        self.assertEqual([i.block_hash for i in r._preload_pending], [h])
        self.assertEqual(r._preload_refcount[h], 3)
        self.assertNotIn(h, r._store_inflight)


if __name__ == "__main__":
    unittest.main()
