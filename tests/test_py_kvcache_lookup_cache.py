"""Unit tests for the manager's lookup existence cache and its invalidation.

The scheduler re-walks a waiting request's whole offload prefix (one
``os.path.exists`` per block) on every step it re-evaluates the request, with no
cross-step memoization. ``SharedStorageOffloadingManager.lookup`` caches positive
existence results. It does not cache negative misses: another process can publish
the file on shared storage between scheduler steps.

The manager holds the cache only until the next *regular* load or store for a NEW
request id (``prepare_load`` / ``prepare_store``) arrives; at that boundary it
drops the previous lookup's results. These tests pin that behavior. A counting
fake replaces ``os.path.exists`` so the workspace needs neither a real filesystem
layout nor vLLM.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import py_kvcache.vllm as vmod
from py_kvcache.vllm import SharedStorageOffloadingManager


class _FakeMapper:
    def get_file_name(self, block_hash: bytes) -> bytes:
        # The "file name" is the hash itself; the fake exists() consults a set.
        return block_hash


def _ctx(req_id: str) -> SimpleNamespace:
    return SimpleNamespace(req_id=req_id)


class LookupCacheTest(unittest.TestCase):
    def setUp(self) -> None:
        self.existing: set[bytes] = set()
        self.calls: list[bytes] = []

        def fake_exists(name):
            self.calls.append(name)
            return name in self.existing

        self._orig_exists = vmod.os.path.exists
        vmod.os.path.exists = fake_exists

        self.m = SharedStorageOffloadingManager(file_mapper=_FakeMapper())
        # Bypass vLLM offload-key parsing: the test keys are raw block hashes.
        self.m._get_block_hash = lambda key: key  # type: ignore[assignment]

    def tearDown(self) -> None:
        vmod.os.path.exists = self._orig_exists

    def test_repeated_hit_is_cached_no_restat(self) -> None:
        self.existing.add(b"a")
        ctx = _ctx("r0")
        self.assertTrue(self.m.lookup(b"a", ctx))
        self.assertFalse(self.m.lookup(b"b", ctx))
        self.assertEqual(self.calls, [b"a", b"b"])
        # Re-walk the same hit many times: zero new stats for immutable files.
        for _ in range(10):
            self.assertTrue(self.m.lookup(b"a", ctx))
        self.assertEqual(self.calls, [b"a", b"b"])

    def test_repeated_miss_restats_for_remote_publish(self) -> None:
        ctx = _ctx("r0")
        self.assertFalse(self.m.lookup(b"b", ctx))
        self.assertFalse(self.m.lookup(b"b", ctx))
        self.assertEqual(self.calls, [b"b", b"b"])
        self.existing.add(b"b")
        self.assertTrue(self.m.lookup(b"b", ctx))
        self.assertEqual(self.calls, [b"b", b"b", b"b"])
        # Once observed, the immutable final file is safe to cache.
        self.assertTrue(self.m.lookup(b"b", ctx))
        self.assertEqual(self.calls, [b"b", b"b", b"b"])

    def test_prepare_load_new_reqid_flushes(self) -> None:
        self.existing.add(b"a")
        self.m.lookup(b"a", _ctx("r0"))
        self.assertEqual(self.calls, [b"a"])
        # Regular load op for a NEW request id -> cache dropped.
        self.m.prepare_load([], _ctx("r1"))
        self.m.lookup(b"a", _ctx("r0"))
        self.assertEqual(self.calls, [b"a", b"a"])  # restat'd after flush

    def test_prepare_load_same_reqid_keeps_cache(self) -> None:
        self.existing.add(b"a")
        # Establish the current io req id, then cache a lookup.
        self.m.prepare_load([], _ctx("r0"))
        self.m.lookup(b"a", _ctx("r0"))
        self.assertEqual(self.calls, [b"a"])
        # Same request id load op -> no flush, cache survives.
        self.m.prepare_load([], _ctx("r0"))
        self.m.lookup(b"a", _ctx("r0"))
        self.assertEqual(self.calls, [b"a"])  # still one stat

    def test_prepare_store_new_reqid_flushes(self) -> None:
        self.existing.add(b"a")
        self.m.lookup(b"a", _ctx("r0"))
        self.m.prepare_store([], _ctx("r2"))  # new req id store op
        self.m.lookup(b"a", _ctx("r0"))
        self.assertEqual(self.calls, [b"a", b"a"])

    def test_reset_cache_clears(self) -> None:
        self.existing.add(b"a")
        self.m.lookup(b"a", _ctx("r0"))
        self.m.reset_cache()
        self.m.lookup(b"a", _ctx("r0"))
        self.assertEqual(self.calls, [b"a", b"a"])
        self.assertIsNone(self.m._last_io_req_id)

    def test_event_only_on_real_work(self) -> None:
        # Cache-hit lookups are silent: the scheduler re-probes each waiting
        # request's full prefix every step, so emitting an event per cached hit
        # buries the trace. Only the cold lookup (a real os.path.exists) emits.
        self.existing.add(b"a")
        events: list[dict] = []
        orig = vmod.add_event

        def cap(name, cat, start_ns, dur, args=None):
            if name == "py_kvcache.manager.lookup":
                events.append(args or {})
            return orig(name, cat, start_ns, dur, args=args)

        vmod.add_event = cap
        try:
            ctx = _ctx("r0")
            self.m.lookup(b"a", ctx)  # cold -> stat -> event
            self.m.lookup(b"a", ctx)  # cached -> silent
        finally:
            vmod.add_event = orig

        self.assertEqual(events, [{"hit": True, "cached": False, "deferred": False}])


class StoreInflightDeferTest(unittest.TestCase):
    """lookup() defers (None) while a block is mid-write so the scheduler waits
    for the store instead of recomputing a reusable prefix."""

    def setUp(self) -> None:
        self.existing: set[bytes] = set()
        self.calls: list[bytes] = []

        def fake_exists(name):
            self.calls.append(name)
            return name in self.existing

        self._orig_exists = vmod.os.path.exists
        vmod.os.path.exists = fake_exists
        self.m = SharedStorageOffloadingManager(file_mapper=_FakeMapper())
        self.m._get_block_hash = lambda key: key  # type: ignore[assignment]

    def tearDown(self) -> None:
        vmod.os.path.exists = self._orig_exists

    def test_inflight_store_defers_lookup(self) -> None:
        self.m.prepare_store([b"a"], _ctx("s0"))  # absent -> in flight
        self.assertEqual(self.m._store_inflight, {b"a": 1})
        self.assertIsNone(self.m.lookup(b"a", _ctx("s0")))  # defer, not miss

    def test_complete_store_landed_yields_hit(self) -> None:
        self.m.prepare_store([b"a"], _ctx("s0"))
        self.assertIsNone(self.m.lookup(b"a", _ctx("s0")))
        self.existing.add(b"a")  # write landed on disk
        self.m.complete_store([b"a"], _ctx("s0"))
        self.assertNotIn(b"a", self.m._store_inflight)
        self.assertTrue(self.m.lookup(b"a", _ctx("s0")))  # re-stat -> hit

    def test_failed_store_falls_to_miss(self) -> None:
        self.m.prepare_store([b"a"], _ctx("s0"))
        self.m.complete_store([b"a"], _ctx("s0"))  # block never landed
        self.assertNotIn(b"a", self.m._store_inflight)
        self.assertFalse(self.m.lookup(b"a", _ctx("s0")))  # miss, not defer

    def test_count_balances_concurrent_stores(self) -> None:
        self.m.prepare_store([b"a"], _ctx("s0"))
        self.m.prepare_store([b"a"], _ctx("s1"))  # second writer, still absent
        self.assertEqual(self.m._store_inflight, {b"a": 2})
        self.m.complete_store([b"a"], _ctx("s0"))
        self.assertEqual(self.m._store_inflight, {b"a": 1})  # one writer left
        self.assertIsNone(self.m.lookup(b"a", _ctx("s0")))  # still deferring
        self.m.complete_store([b"a"], _ctx("s1"))
        self.assertNotIn(b"a", self.m._store_inflight)

    def test_existing_block_skips_inflight(self) -> None:
        self.existing.add(b"a")
        self.m.prepare_store([b"a"], _ctx("s0"))  # already on disk -> not stored
        self.assertNotIn(b"a", self.m._store_inflight)
        self.assertTrue(self.m.lookup(b"a", _ctx("s0")))

    def test_reset_cache_clears_inflight(self) -> None:
        self.m.prepare_store([b"a"], _ctx("s0"))
        self.m.reset_cache()
        self.assertEqual(self.m._store_inflight, {})
        self.assertFalse(self.m.lookup(b"a", _ctx("s0")))  # absent, not deferred


if __name__ == "__main__":
    unittest.main()
