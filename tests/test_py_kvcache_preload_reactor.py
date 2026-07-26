"""Unit tests for the reactor's preload scheduling decisions.

These exercise the pure decision helpers (gate, staging-slot cap, eviction,
load-arrival handling, multi-copy bookkeeping) on an ``IoReactor`` built via
``__new__`` with hand-set state, so they need neither CUDA/vLLM nor a live
io_uring ring. Trace-based validation on the GPU box covers the full read/copy
pipeline.

Each preload copy carries its own ``_PreloadInfo`` (block_hash + the demanding
request's id/preload_id/tid) through the pending / known-missing / blocked /
cached structures, so each speculative read attributes to the request that asked
for it rather than to whichever request first touched the hash.
"""

from __future__ import annotations

import collections
import unittest
from types import SimpleNamespace

try:
    import numpy  # noqa: F401

    import py_kvcache.reactor as rmod
    from py_kvcache.break_even import BreakEvenThresholds
    from py_kvcache.reactor import IoReactor, _PreloadInfo, _PreloadRequest

    HAVE_DEPS = True
except Exception:  # pragma: no cover - minimal workspace
    HAVE_DEPS = False


class FakePool:
    """Minimal staging pool: hands out / takes back integer slot indices."""

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


def _bare_reactor(*, pool_slots: int = 8, iodepth: int = 4) -> "IoReactor":
    r = IoReactor.__new__(IoReactor)
    r.iodepth = iodepth
    r.staging_pool = FakePool(pool_slots)
    r._share_preload = False
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
    r._break_even = BreakEvenThresholds()
    r._storage_block_tokens = 0
    return r


def _load_job(*, is_store: bool = False, needs_disk: bool = True):
    """A minimal job. ``needs_disk`` controls whether its next file still needs
    a storage read (drives the loosened load-pressure gate)."""
    return SimpleNamespace(
        failed=None,
        future_set=False,
        is_store=is_store,
        next_file_index=0 if needs_disk else 1,
        total_files=1,
        block_hashes=[b"diskblock"],
    )


def _info(
    profile_tid: str = "p",
    *,
    block_hash: bytes = b"h",
    preload_id: str = "p0",
    req_id: str = "r0",
    file_index: int = 0,
    total_files: int = 1,
):
    return _PreloadInfo(
        block_hash=block_hash,
        preload_id=preload_id,
        req_id=req_id,
        profile_tid=profile_tid,
        file_index=file_index,
        total_files=total_files,
    )


def _store_job(block_hashes, *, req_id: str = "producer"):
    return SimpleNamespace(
        is_store=True,
        block_hashes=block_hashes,
        profile=SimpleNamespace(req_id=req_id),
    )


def _pending_hashes(r) -> list:
    return [info.block_hash for info in r._preload_pending]


@unittest.skipUnless(HAVE_DEPS, "numpy / py_kvcache not importable")
class PreloadGateTest(unittest.TestCase):
    def test_idle_allows_preload(self):
        r = _bare_reactor()
        self.assertTrue(r._can_issue_speculative_preload())

    def test_store_allows_preload(self):
        r = _bare_reactor()
        r._active = [_load_job(is_store=True)]
        self.assertTrue(r._can_issue_speculative_preload())

    def test_load_job_needing_disk_blocks_preload(self):
        r = _bare_reactor()
        r._active = [_load_job(is_store=False, needs_disk=True)]
        self.assertFalse(r._can_issue_speculative_preload())

    def test_ready_load_fd_blocks_preload(self):
        r = _bare_reactor()
        r._ready_fds_load.append(SimpleNamespace())
        self.assertFalse(r._can_issue_speculative_preload())

    def test_inflight_load_read_blocks_preload(self):
        r = _bare_reactor()
        r._inflight = {1: SimpleNamespace(op_kind="read", job=_load_job(is_store=False))}
        self.assertFalse(r._can_issue_speculative_preload())

    def test_cuda_copy_only_load_allows_preload(self):
        """A load that has issued all its disk reads and is only draining CUDA
        copies (cpu<->gpu DMA) must NOT block preload; the disk is free."""
        r = _bare_reactor()
        r._active = [_load_job(is_store=False, needs_disk=False)]
        r._pending_copies = [SimpleNamespace(is_store=False)]  # staging->GPU
        r._copy_ready = [SimpleNamespace()]  # queued load copy
        self.assertTrue(r._can_issue_speculative_preload())

    def test_load_next_file_served_by_preload_does_not_block(self):
        """If the load's next file is already cached/in-flight from preload, no
        disk read is needed for it, so it is not disk pressure."""
        r = _bare_reactor()
        job = _load_job(is_store=False, needs_disk=True)
        job.block_hashes = [b"cachedblk"]
        r._preload_cache_push(b"cachedblk", 3, _info(block_hash=b"cachedblk"))
        r._active = [job]
        self.assertTrue(r._can_issue_speculative_preload())

    def test_inflight_load_close_does_not_block(self):
        """A trailing close op for a finished load is not disk pressure."""
        r = _bare_reactor()
        r._inflight = {1: SimpleNamespace(op_kind="close", job=_load_job(is_store=False))}
        self.assertTrue(r._can_issue_speculative_preload())


@unittest.skipUnless(HAVE_DEPS, "numpy / py_kvcache not importable")
class PreloadSlotBudgetTest(unittest.TestCase):
    def test_cap_blocks_when_full(self):
        r = _bare_reactor(pool_slots=8, iodepth=4)  # max preload slots = 4
        for i in range(4):
            h = bytes([i])
            r._preload_cache_push(h, i, _info(block_hash=h))
        self.assertEqual(r._preload_cached_total, 4)
        self.assertFalse(r._preload_slots_available())

    def test_cap_counts_inflight(self):
        r = _bare_reactor(pool_slots=8, iodepth=4)
        r._preload_cache_push(b"a", 0, _info(block_hash=b"a"))
        r._preload_cache_push(b"b", 1, _info(block_hash=b"b"))
        r._preload_inflight_add(b"c")
        r._preload_inflight_add(b"d")
        self.assertFalse(r._preload_slots_available())  # 2 + 2 == cap(4)
        r._preload_inflight_remove(b"d")
        self.assertTrue(r._preload_slots_available())  # 2 + 1 < 4

    def test_cap_counts_multiple_copies_of_one_hash(self):
        """Two cached copies of the same hash each count against the budget."""
        r = _bare_reactor(pool_slots=8, iodepth=4)
        r._preload_cache_push(b"a", 0, _info(block_hash=b"a"))
        r._preload_cache_push(b"a", 1, _info(block_hash=b"a"))  # second copy
        self.assertEqual(r._preload_cached_total, 2)
        self.assertEqual(len(r._preload_slots[b"a"]), 2)
        r._preload_inflight_add(b"a")
        r._preload_inflight_add(b"a")
        self.assertFalse(r._preload_slots_available())  # 2 + 2 == cap(4)

    def test_evict_one_is_fifo_and_releases(self):
        r = _bare_reactor()
        r._preload_cache_push(b"old", 5, _info(block_hash=b"old"))
        r._preload_cache_push(b"new", 6, _info(block_hash=b"new"))
        self.assertTrue(r._evict_one_preload_slot())
        self.assertEqual(r.staging_pool.released, [5])  # oldest first
        self.assertNotIn(b"old", r._preload_slots)
        self.assertIn(b"new", r._preload_slots)
        self.assertEqual(r._preload_cached_total, 1)

    def test_evict_drops_one_copy_keeps_others(self):
        r = _bare_reactor()
        r._preload_cache_push(b"h", 5, _info(block_hash=b"h"))
        r._preload_cache_push(b"h", 6, _info(block_hash=b"h"))  # two copies
        self.assertTrue(r._evict_one_preload_slot())
        self.assertEqual(r.staging_pool.released, [5])  # oldest copy first
        self.assertIn(b"h", r._preload_slots)  # one copy still cached
        self.assertEqual(r._preload_cached_total, 1)

    def test_evict_empty_returns_false(self):
        r = _bare_reactor()
        self.assertFalse(r._evict_one_preload_slot())

    def test_reserve_foreground_slot_evicts_when_pool_full(self):
        r = _bare_reactor(pool_slots=2, iodepth=2)
        # Drain the pool, parking both slots in unclaimed preloads.
        s0 = r.staging_pool.try_reserve()
        s1 = r.staging_pool.try_reserve()
        r._preload_cache_push(b"x", s0.index, _info(block_hash=b"x"))
        r._preload_cache_push(b"y", s1.index, _info(block_hash=b"y"))
        self.assertIsNone(r.staging_pool.try_reserve())  # pool empty
        slot = r._reserve_foreground_slot()  # must evict + reserve
        self.assertIsNotNone(slot)
        self.assertEqual(len(r._preload_slots), 1)  # one evicted


@unittest.skipUnless(HAVE_DEPS, "numpy / py_kvcache not importable")
class MultiCopyIntakeTest(unittest.TestCase):
    def test_same_req_dedups_to_one_copy(self):
        """A request re-emitting the same hash across steps stages one copy."""
        r = _bare_reactor()
        h = b"shared"
        for _ in range(3):
            r._intake(_PreloadRequest([h], preload_id="r0:0:1", req_id="r0"))
        self.assertEqual(_pending_hashes(r), [h])  # only one copy queued

    def test_distinct_reqs_each_get_a_copy(self):
        """Identical-prefix requests each enqueue their own copy of a hash."""
        r = _bare_reactor()
        h = b"shared"
        r._intake(_PreloadRequest([h], preload_id="r0:0:1", req_id="r0"))
        r._intake(_PreloadRequest([h], preload_id="r1:0:1", req_id="r1"))
        r._intake(_PreloadRequest([h], preload_id="r2:0:1", req_id="r2"))
        self.assertEqual(_pending_hashes(r), [h, h, h])  # three copies
        # Each pending copy carries its own requesting attribution.
        self.assertEqual([info.req_id for info in r._preload_pending], ["r0", "r1", "r2"])

    def test_owned_dedup_is_per_req_hash_pair(self):
        r = _bare_reactor()
        self.assertTrue(r._preload_own("r0", b"a"))
        self.assertFalse(r._preload_own("r0", b"a"))  # same pair -> dedup
        self.assertTrue(r._preload_own("r1", b"a"))  # other req -> own copy
        self.assertTrue(r._preload_own("r0", b"b"))  # other hash -> own copy

    def test_pending_push_tracks_count(self):
        r = _bare_reactor()
        r._pending_push(_info(block_hash=b"h"))
        r._pending_push(_info(block_hash=b"h"))
        self.assertEqual(r._preload_pending_count[b"h"], 2)
        r._pending_dec(b"h")
        self.assertEqual(r._preload_pending_count[b"h"], 1)
        r._pending_dec(b"h")
        self.assertNotIn(b"h", r._preload_pending_count)


@unittest.skipUnless(HAVE_DEPS, "numpy / py_kvcache not importable")
class PendingCancelTest(unittest.TestCase):
    def test_cancel_one_only_when_live_pending(self):
        r = _bare_reactor()
        self.assertFalse(r._cancel_one_pending(b"h"))  # nothing pending
        r._pending_push(_info(block_hash=b"h"))
        self.assertTrue(r._cancel_one_pending(b"h"))  # one live -> cancelled
        self.assertFalse(r._cancel_one_pending(b"h"))  # already cancelled

    def test_drain_cancelled_pending_skips_without_open(self):
        r = _bare_reactor()
        r._pending_push(_info(block_hash=b"h"))
        r._cancel_one_pending(b"h")
        r._drain_cancelled_pending()
        self.assertEqual(_pending_hashes(r), [])
        self.assertNotIn(b"h", r._preload_pending_count)
        self.assertNotIn(b"h", r._preload_pending_cancel)

    def test_cancelled_copy_not_opened_in_schedule(self):
        """A cancelled pending copy must not turn into a speculative read."""
        r = _bare_reactor()
        opened = []
        r._submit_open_preload = lambda info: opened.append(info.block_hash)
        r._can_open = lambda: True
        r._preload_slots_available = lambda: True
        r._can_issue_speculative_preload = lambda: True
        r._drain_ready_load_fds = lambda: False
        r._drain_ready_preload_fds = lambda: False
        r._pending_push(_info(block_hash=b"keep"))
        r._pending_push(_info(block_hash=b"cancel"))
        r._cancel_one_pending(b"cancel")
        r._schedule_work()
        self.assertEqual(opened, [b"keep"])  # only the live copy opened
        self.assertEqual(_pending_hashes(r), [])  # both drained from queue


@unittest.skipUnless(HAVE_DEPS, "numpy / py_kvcache not importable")
class PreloadWaiterTest(unittest.TestCase):
    def test_waiter_room_bounded_by_inflight(self):
        r = _bare_reactor()
        h = b"h"
        self.assertFalse(r._preload_waiter_room(h))  # no inflight -> no room
        r._preload_inflight_add(h)
        self.assertTrue(r._preload_waiter_room(h))  # 0 waiters < 1 inflight
        r._preload_waiter_add(h, _load_job(), 0)
        self.assertFalse(r._preload_waiter_room(h))  # 1 waiter == 1 inflight

    def test_waiters_are_fifo(self):
        r = _bare_reactor()
        h = b"h"
        j0, j1 = _load_job(), _load_job()
        r._preload_waiter_add(h, j0, 0)
        r._preload_waiter_add(h, j1, 1)
        self.assertIs(r._preload_waiter_pop(h)[0], j0)
        self.assertIs(r._preload_waiter_pop(h)[0], j1)
        self.assertIsNone(r._preload_waiter_pop(h))
        self.assertNotIn(h, r._preload_waiters)


@unittest.skipUnless(HAVE_DEPS, "numpy / py_kvcache not importable")
class LoadArrivalTest(unittest.TestCase):
    def test_inflight_and_cached_preloads_kept_for_reuse(self):
        """A load for an in-flight / cached preload hash must NOT discard it
        (step 1 claims it / registers a waiter instead of double-reading)."""
        r = _bare_reactor()
        h_inflight, h_cached = b"inflight", b"cached"
        r._preload_inflight_add(h_inflight)
        r._preload_cache_push(h_cached, 7, _info(block_hash=h_cached))

        job = SimpleNamespace(is_store=False, block_hashes=[h_inflight, h_cached])
        r._intake(job)

        self.assertTrue(r._has_inflight_preload(h_inflight))
        self.assertTrue(r._has_cached_preload(h_cached))
        self.assertIn(job, r._active)

    def test_pending_copies_for_other_reqs_survive_load_arrival(self):
        """Leftover pending copies (other identical requests' demand) are kept
        when one request's load arrives."""
        r = _bare_reactor()
        h = b"shared"
        r._pending_push(_info(block_hash=h, req_id="r1"))
        r._pending_push(_info(block_hash=h, req_id="r2"))
        r._intake(SimpleNamespace(is_store=False, block_hashes=[h]))
        self.assertEqual(_pending_hashes(r), [h, h])

    def test_load_arrival_clears_known_missing(self):
        r = _bare_reactor()
        h = b"x"
        r._known_missing[h] = collections.deque([_info(block_hash=h)])
        r._intake(SimpleNamespace(is_store=False, block_hashes=[h]))
        self.assertNotIn(h, r._known_missing)


@unittest.skipUnless(HAVE_DEPS, "numpy / py_kvcache not importable")
class StoreInflightTest(unittest.TestCase):
    def test_store_intake_registers_inflight(self):
        r = _bare_reactor()
        job = _store_job([b"a", b"b", b"a"])
        r._intake(job)
        self.assertEqual(r._store_inflight, {b"a": 2, b"b": 1})
        self.assertIn(job, r._active)

    def test_release_decrements_until_last(self):
        r = _bare_reactor()
        h = b"a"
        r._store_inflight = {h: 2}
        r._release_store_inflight(h, ok=True)
        self.assertEqual(r._store_inflight, {h: 1})  # one writer left
        r._release_store_inflight(h, ok=True)
        self.assertNotIn(h, r._store_inflight)

    def test_write_wakes_blocked_preload(self):
        r = _bare_reactor()
        h = b"x"
        r._store_inflight = {h: 1}
        r._preload_blocked_on_write = {h: collections.deque([_info(block_hash=h)])}
        r._release_store_inflight(h, ok=True)
        self.assertNotIn(h, r._store_inflight)
        self.assertNotIn(h, r._preload_blocked_on_write)
        self.assertEqual(_pending_hashes(r), [h])  # one read-back requeued

    def test_write_wakes_multiple_blocked_copies(self):
        """Multiple identical requests deferred on one store all get read-backs,
        each keeping its own attribution."""
        r = _bare_reactor()
        h = b"x"
        r._store_inflight = {h: 1}
        r._preload_blocked_on_write = {
            h: collections.deque([_info(block_hash=h, req_id=rid) for rid in ("r0", "r1", "r2")])
        }
        r._release_store_inflight(h, ok=True)
        self.assertEqual(_pending_hashes(r), [h, h, h])
        self.assertEqual([info.req_id for info in r._preload_pending], ["r0", "r1", "r2"])

    def test_failed_write_drops_blocked_preload(self):
        r = _bare_reactor()
        h = b"x"
        r._store_inflight = {h: 1}
        r._preload_blocked_on_write = {h: collections.deque([_info(block_hash=h)])}
        r._release_store_inflight(h, ok=False)
        self.assertNotIn(h, r._preload_blocked_on_write)
        self.assertEqual(_pending_hashes(r), [])  # not requeued

    def test_stop_does_not_requeue(self):
        r = _bare_reactor()
        r._stop = True
        h = b"x"
        r._store_inflight = {h: 1}
        r._preload_blocked_on_write = {h: collections.deque([_info(block_hash=h)])}
        r._release_store_inflight(h, ok=True)
        self.assertEqual(_pending_hashes(r), [])


@unittest.skipUnless(HAVE_DEPS, "numpy / py_kvcache not importable")
class KnownMissingTest(unittest.TestCase):
    def test_remember_caps_fifo(self):
        r = _bare_reactor()
        r._max_known_missing = 2
        for i in range(3):
            h = bytes([i])
            r._remember_known_missing(_info(block_hash=h))
        # Oldest (b"\x00") evicted by the FIFO cap.
        self.assertNotIn(b"\x00", r._known_missing)
        self.assertEqual(list(r._known_missing), [b"\x01", b"\x02"])

    def test_submit_open_preload_submits_without_exists_gate(self):
        # The async openat IS the existence check; no os.path.exists pre-stat.
        r = _bare_reactor()
        r._next_user_data = 0
        r._open_inflight = 0
        r.ring = SimpleNamespace()
        h = b"missing"
        info = _info(block_hash=h)
        r.file_mapper = SimpleNamespace(get_file_name=lambda block_hash: block_hash)
        r.file_store = SimpleNamespace(queue_open_read=lambda *args, **kwargs: b"buf")
        # exists() must NOT be consulted on the submit path anymore.
        orig_exists = rmod.os.path.exists

        def _boom(path):
            raise AssertionError("os.path.exists must not be called on submit")

        try:
            rmod.os.path.exists = _boom
            r._submit_open_preload(info)
        finally:
            rmod.os.path.exists = orig_exists

        # An openat op is now in flight for the (absent) hash, carrying the
        # demanding request's own info.
        self.assertEqual(len(r._inflight), 1)
        self.assertTrue(r._has_inflight_preload(h))
        self.assertEqual(r._open_inflight, 1)
        op = next(iter(r._inflight.values()))
        self.assertEqual(op.op_kind, "open")
        self.assertEqual(op.preload_hash, h)
        self.assertIs(op.preload_info, info)  # per-copy attribution rides the op

    def test_store_intake_rearms_known_missing(self):
        """A store for a hash a *reusing* preload already missed arms a write readback."""
        r = _bare_reactor()
        h = b"x"
        r._known_missing[h] = collections.deque([_info(block_hash=h, req_id="reuse")])
        r._intake(_store_job([h]))
        self.assertNotIn(h, r._known_missing)
        self.assertEqual(len(r._preload_blocked_on_write[h]), 1)
        self.assertEqual(r._store_inflight, {h: 1})
        # Write completing then requeues the preload for a read-back.
        r._release_store_inflight(h, ok=True)
        self.assertEqual(_pending_hashes(r), [h])

    def test_store_intake_drops_producers_own_missed_copy(self):
        """A FRESH request whose only candidate was itself must NOT re-arm a
        read-back of its just-written bytes (the producer never loads h)."""
        r = _bare_reactor()
        h = b"x"
        r._known_missing[h] = collections.deque([_info(block_hash=h, req_id="producer")])
        r._intake(_store_job([h], req_id="producer"))
        self.assertNotIn(h, r._known_missing)
        self.assertNotIn(h, r._preload_blocked_on_write)
        r._release_store_inflight(h, ok=True)
        self.assertEqual(_pending_hashes(r), [])

    def test_shared_store_intake_drops_producer_only_demand(self):
        """Shared mode: a FRESH unique-prefix store whose only demand is the
        producer itself withdraws that demand and re-arms nothing."""
        r = _bare_reactor()
        r._share_preload = True
        h = b"x"
        r._preload_own("producer", h)
        r._preload_refcount[h] = 1
        r._known_missing[h] = collections.deque([_info(block_hash=h, req_id="producer")])
        r._intake(_store_job([h], req_id="producer"))
        self.assertNotIn(h, r._preload_blocked_on_write)
        self.assertEqual(r._preload_refcount.get(h, 0), 0)
        r._release_store_inflight(h, ok=True)
        self.assertEqual(_pending_hashes(r), [])

    def test_shared_store_intake_keeps_reuse_demand(self):
        """Shared mode: when a reusing request also demands the hash, the store
        withdraws only the producer's demand and still re-arms one read-back."""
        r = _bare_reactor()
        r._share_preload = True
        h = b"x"
        r._preload_own("producer", h)
        r._preload_own("reuse", h)
        r._preload_refcount[h] = 2
        r._known_missing[h] = collections.deque([_info(block_hash=h, req_id="producer")])
        r._intake(_store_job([h], req_id="producer"))
        self.assertEqual(r._preload_refcount.get(h, 0), 1)
        self.assertEqual(len(r._preload_blocked_on_write[h]), 1)
        r._release_store_inflight(h, ok=True)
        self.assertEqual(_pending_hashes(r), [h])

    def test_store_intake_rearms_one_copy_per_missed_request(self):
        """N reusing requests that missed the same unwritten hash each get a copy
        armed; the producer's own missed copy is dropped."""
        r = _bare_reactor()
        h = b"x"
        r._known_missing[h] = collections.deque(
            [_info(block_hash=h, req_id=rid) for rid in ("producer", "r1", "r2")]
        )
        r._intake(_store_job([h], req_id="producer"))
        self.assertEqual(len(r._preload_blocked_on_write[h]), 2)
        # Write completing requeues one read-back per surviving missed copy.
        r._release_store_inflight(h, ok=True)
        self.assertEqual(_pending_hashes(r), [h, h])

    def test_preload_request_for_known_missing_bumps_count(self):
        """A new request for a known-missing hash registers extra demand (so the
        store arms its copy too) without issuing a doomed openat."""
        r = _bare_reactor()
        h = b"x"
        r._known_missing[h] = collections.deque([_info(block_hash=h, req_id="r0")])
        r._intake(_PreloadRequest([h], preload_id="p0", req_id="r1", profile_tid="p"))
        # No re-open (pending stays empty); demand recorded as a second copy.
        self.assertEqual(_pending_hashes(r), [])
        self.assertEqual(len(r._known_missing[h]), 2)
        self.assertEqual([info.req_id for info in r._known_missing[h]], ["r0", "r1"])


@unittest.skipUnless(HAVE_DEPS, "numpy / py_kvcache not importable")
class StorePriorityTest(unittest.TestCase):
    """Stores share the foreground priority tier with loads: when the pool is
    full of unclaimed preload slots, a store evicts one rather than stalling."""

    def _store_job(self, n_files: int = 1):
        return SimpleNamespace(
            failed=None,
            future_set=False,
            is_store=True,
            next_file_index=0,
            total_files=n_files,
            block_hashes=[bytes([i]) for i in range(n_files)],
        )

    def test_can_issue_store_ignores_free_count(self):
        r = _bare_reactor(pool_slots=2, iodepth=4)
        r.staging_pool.try_reserve()
        r.staging_pool.try_reserve()
        self.assertEqual(r.staging_pool.free_count, 0)
        self.assertTrue(r._can_issue_store())  # slot pressure is _schedule_one's job

    def test_can_issue_store_respects_iodepth(self):
        r = _bare_reactor(pool_slots=8, iodepth=4)
        r._data_inflight = 4
        self.assertFalse(r._can_issue_store())

    def test_schedule_one_evicts_preload_when_pool_full(self):
        r = _bare_reactor(pool_slots=2, iodepth=2)
        s0 = r.staging_pool.try_reserve()
        s1 = r.staging_pool.try_reserve()
        r._preload_cache_push(b"x", s0.index, _info(block_hash=b"x"))
        r._preload_cache_push(b"y", s1.index, _info(block_hash=b"y"))
        scheduled: list[int] = []
        r._schedule_store_copy = lambda job, fi, slot_index: scheduled.append(slot_index)

        self.assertTrue(r._schedule_one(self._store_job()))
        self.assertEqual(len(scheduled), 1)  # store got a slot
        self.assertEqual(len(r._preload_slots), 1)  # one preload evicted
        self.assertEqual(r._preload_cached_total, 1)

    def test_store_loop_evicts_preload_in_schedule_work(self):
        """End-to-end through _schedule_work: a store drains a full pool of
        unclaimed preloads instead of bailing on the free_count gate."""
        r = _bare_reactor(pool_slots=2, iodepth=2)
        s0 = r.staging_pool.try_reserve()
        s1 = r.staging_pool.try_reserve()
        r._preload_cache_push(b"x", s0.index, _info(block_hash=b"x"))
        r._preload_cache_push(b"y", s1.index, _info(block_hash=b"y"))
        r._drain_ready_load_fds = lambda: False
        r._drain_ready_preload_fds = lambda: False

        scheduled: list[int] = []

        def _store_copy(job, fi, slot_index):
            scheduled.append(slot_index)
            job.next_file_index += 1  # advance so the store loop terminates

        r._schedule_store_copy = _store_copy
        r._active = [self._store_job(n_files=1)]

        r._schedule_work()
        self.assertEqual(len(scheduled), 1)
        self.assertEqual(len(r._preload_slots), 1)  # evicted to make room


@unittest.skipUnless(HAVE_DEPS, "numpy / py_kvcache not importable")
class StagingCacheTest(unittest.TestCase):
    """Read-side staging-data cache: lowest slot tier, evicted before preload by
    foreground and reclaimable by preload; cached hashes serve loads as hits."""

    def _cache_reactor(self, *, policy="lru", pool_slots=8, iodepth=4):
        from py_kvcache.staging_cache import StagingDataCache

        r = _bare_reactor(pool_slots=pool_slots, iodepth=iodepth)
        r._staging_cache = StagingDataCache(policy=policy, capacity=pool_slots - iodepth)
        return r

    def _load_job_full(self, *, req_id="r0", job_id=0, hashes=(b"h",)):
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

    def test_release_or_cache_inserts_when_enabled(self):
        r = self._cache_reactor()
        slot = r.staging_pool.try_reserve()
        r._release_or_cache(b"h", slot.index)
        self.assertIn(b"h", r._staging_cache)
        self.assertNotIn(slot.index, r.staging_pool._free)  # retained, not released

    def test_release_or_cache_releases_when_disabled(self):
        r = _bare_reactor()  # _staging_cache is None
        slot = r.staging_pool.try_reserve()
        r._release_or_cache(b"h", slot.index)
        self.assertIn(slot.index, r.staging_pool._free)  # released to pool

    def test_write_complete_caches_slot(self):
        """write: gpu->cache->disk; a finished store retains its staging slot in
        the cache (byte-identical to a read of the same hash) instead of releasing."""
        r = self._cache_reactor()
        slot = r.staging_pool.try_reserve()
        job = SimpleNamespace(
            block_hashes=[b"h"],
            file_samples=[],
            profile=SimpleNamespace(req_id="producer"),
        )
        op = SimpleNamespace(
            job=job, slot_index=slot.index, file_index=0, fd=7,
            temp_path="/tmp/t", final_path="/tmp/f", start_ns=0,
        )
        r.file_store = SimpleNamespace(io_size=1024, finish_write=lambda **kw: None)
        r.layout = SimpleNamespace(storage_block_bytes=1024)
        r._file_terminal = lambda *a, **k: None
        r._release_store_inflight = lambda *a, **k: None
        r._on_write_complete(op, 1024)
        self.assertIn(b"h", r._staging_cache)               # write -> cache
        self.assertNotIn(slot.index, r.staging_pool._free)  # retained, not released

    def test_cached_store_suppresses_preload_rearm(self):
        """Once the store has cached the hash, the open-before-write re-arm is
        redundant (a reuse load hits the cache), so it is dropped and the shared
        demand is cleared without leaking refcount."""
        r = self._cache_reactor()
        r._share_preload = True
        h = b"x"
        r._preload_own("reuse", h)
        r._preload_refcount[h] = 1
        r._known_missing[h] = collections.deque([_info(block_hash=h, req_id="reuse")])
        r._intake(_store_job([h], req_id="producer"))
        self.assertEqual(len(r._preload_blocked_on_write[h]), 1)
        slot = r.staging_pool.try_reserve()
        r._release_or_cache(h, slot.index)  # write -> cache
        r._release_store_inflight(h, ok=True)
        self.assertEqual(_pending_hashes(r), [])             # no disk re-read armed
        self.assertEqual(r._preload_refcount.get(h, 0), 0)   # demand cleared
        self.assertEqual([k for k in r._preload_owned if k[1] == h], [])

    def test_load_hits_cache_no_open(self):
        r = self._cache_reactor()
        slot = r.staging_pool.try_reserve()
        r._release_or_cache(b"h", slot.index)
        copies = []
        r._queue_load_copy = lambda job, fi, idx, *, cache=None, source="file": copies.append((idx, cache))
        opened = []
        r._submit_open_read = lambda job, fi: opened.append(fi)
        r._drain_ready_load_fds = lambda: False
        r._drain_ready_preload_fds = lambda: False
        r._active = [self._load_job_full(hashes=(b"h",))]

        r._schedule_work()
        self.assertEqual(opened, [])  # served from cache, no disk open
        self.assertEqual(len(copies), 1)
        self.assertEqual(copies[0][0], slot.index)  # copy reads the cached slot
        self.assertEqual(r._staging_cache.get(b"h").copies_inflight, 1)  # pinned

    def test_foreground_evicts_cache_before_preload(self):
        r = self._cache_reactor(pool_slots=4, iodepth=2)
        # Drain pool; park one cache slot and one preload slot.
        slots = [r.staging_pool.try_reserve().index for _ in range(4)]
        r._release_or_cache(b"cache", slots[0])
        r._preload_cache_push(b"pre", slots[1], _info(block_hash=b"pre"))
        # Remaining two slots stay "in use" (foreground). Pool now empty.
        self.assertEqual(r.staging_pool.free_count, 0)

        slot = r._reserve_foreground_slot()
        self.assertIsNotNone(slot)
        self.assertNotIn(b"cache", r._staging_cache)  # cache died first
        self.assertIn(b"pre", r._preload_slots)  # preload untouched

    def test_preload_reserve_evicts_cache(self):
        r = self._cache_reactor(pool_slots=4, iodepth=2)
        slots = [r.staging_pool.try_reserve().index for _ in range(4)]
        r._release_or_cache(b"cache", slots[0])
        self.assertEqual(r.staging_pool.free_count, 0)
        slot = r._reserve_preload_slot()
        self.assertIsNotNone(slot)
        self.assertNotIn(b"cache", r._staging_cache)

    def test_evict_one_cache_skips_pinned(self):
        r = self._cache_reactor()
        s0 = r.staging_pool.try_reserve()
        s1 = r.staging_pool.try_reserve()
        r._release_or_cache(b"a", s0.index)
        r._release_or_cache(b"b", s1.index)
        r._staging_cache.pin(r._staging_cache.get(b"a"))  # a in flight
        self.assertTrue(r._evict_one_cache_slot("foreground_pressure"))
        self.assertIn(b"a", r._staging_cache)  # pinned survived
        self.assertNotIn(b"b", r._staging_cache)

    def test_preload_intake_suppressed_for_cached_hash(self):
        r = self._cache_reactor()
        slot = r.staging_pool.try_reserve()
        r._release_or_cache(b"h", slot.index)
        r._intake(_PreloadRequest([b"h"], preload_id="r0:0:1", req_id="r0"))
        self.assertEqual(_pending_hashes(r), [])  # no speculative read queued
        self.assertNotIn(("r0", b"h"), r._preload_owned)  # no ownership taken


@unittest.skipUnless(HAVE_DEPS, "numpy / py_kvcache not importable")
class BreakEvenDeclineTest(unittest.TestCase):
    def _gated_reactor(self, *, ssd_tokens=4096, mem_tokens=0, block_tokens=2048):
        r = _bare_reactor()
        r._break_even = BreakEvenThresholds(ssd_tokens=ssd_tokens, mem_tokens=mem_tokens)
        r._storage_block_tokens = block_tokens
        return r

    def _load_job(self, *, total_files, hashes, dst_blocks):
        from concurrent.futures import Future

        return SimpleNamespace(
            job_id=1,
            is_store=False,
            failed=None,
            future_set=False,
            total_files=total_files,
            block_hashes=hashes,
            block_chunks=[numpy.asarray(b, dtype=numpy.int64) for b in dst_blocks],
            profile=SimpleNamespace(profile_tid="p", req_id="r0"),
            future=Future(),
        )

    def test_short_ssd_load_declined(self):
        r = self._gated_reactor()  # P*_ssd=4096, block=2048 => need >=2 files
        job = self._load_job(total_files=1, hashes=[b"h0"], dst_blocks=[[5, 6]])
        r._intake(job)
        self.assertNotIn(job, r._active)
        self.assertTrue(job.future_set)
        with self.assertRaises(rmod.LoadDeclined) as ctx:
            job.future.result()
        self.assertEqual(ctx.exception.block_ids, [5, 6])

    def test_long_ssd_load_admitted(self):
        r = self._gated_reactor()
        job = self._load_job(
            total_files=2, hashes=[b"h0", b"h1"], dst_blocks=[[5, 6], [7, 8]]
        )
        r._intake(job)
        self.assertIn(job, r._active)
        self.assertFalse(job.future_set)

    def test_short_ram_resident_load_admitted(self):
        # Sub-P*_ssd but fully RAM-resident -> P*_mem(0) branch, never declined.
        r = self._gated_reactor()
        h = b"h0"
        r._preload_slots[h] = collections.deque([(0, None)])
        r._preload_cached_total = 1
        self.assertTrue(r._has_cached_preload(h))
        job = self._load_job(total_files=1, hashes=[h], dst_blocks=[[5, 6]])
        r._intake(job)
        self.assertIn(job, r._active)
        self.assertFalse(job.future_set)

    def test_disabled_thresholds_never_decline(self):
        r = _bare_reactor()
        r._break_even = BreakEvenThresholds()
        r._storage_block_tokens = 2048
        job = self._load_job(total_files=1, hashes=[b"h0"], dst_blocks=[[5, 6]])
        r._intake(job)
        self.assertIn(job, r._active)


if __name__ == "__main__":
    unittest.main()
