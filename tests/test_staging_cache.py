"""Unit tests for the optional staging-data cache (policies + slot manager).

Pure-Python (no torch/CUDA/ring). Covers LRU ordering, ARC ghost adaptation +
scan resistance, and the StagingDataCache slot bookkeeping (dup-put release,
capacity self-eviction, copies_inflight pinning, drain).
"""

from __future__ import annotations

import unittest

from py_kvcache.staging_cache import ArcPolicy, LruPolicy, StagingDataCache


def H(i: int) -> bytes:
    return bytes([i])


class LruPolicyTest(unittest.TestCase):
    def test_victim_order_is_oldest_first(self):
        p = LruPolicy(8)
        for i in range(3):
            p.admit(H(i))
        self.assertEqual(list(p.victims()), [H(0), H(1), H(2)])

    def test_touch_moves_to_mru(self):
        p = LruPolicy(8)
        for i in range(3):
            p.admit(H(i))
        p.touch(H(0))  # now MRU
        self.assertEqual(list(p.victims()), [H(1), H(2), H(0)])

    def test_evict_removes(self):
        p = LruPolicy(8)
        p.admit(H(0))
        p.admit(H(1))
        p.evict(H(0))
        self.assertEqual(list(p.victims()), [H(1)])


class ArcPolicyTest(unittest.TestCase):
    def test_reused_key_promoted_to_t2_survives_scan(self):
        """A reused key (in T2) is evicted after one-shot scan keys (T1)."""
        p = ArcPolicy(capacity=4)
        p.admit(H(1))  # T1
        p.touch(H(1))  # -> T2 (frequent)
        for i in range(2, 5):  # scan: one-shot keys into T1
            p.admit(H(i))
        order = list(p.victims())
        # T1 (one-shot scan) drains before the reused T2 key.
        self.assertLess(order.index(H(2)), order.index(H(1)))
        self.assertEqual(order[-1], H(1))  # reused key last to die

    def test_b1_ghost_hit_raises_p(self):
        p = ArcPolicy(capacity=4)
        p.admit(H(1))
        p.evict(H(1))  # -> B1 ghost
        self.assertIn(H(1), p._b1)
        before = p.p
        p.admit(H(1))  # ghost hit in B1 -> recency pressure, p up
        self.assertGreater(p.p, before)
        self.assertIn(H(1), p._t2)  # promoted

    def test_b2_ghost_hit_lowers_p(self):
        p = ArcPolicy(capacity=4)
        p.admit(H(1))
        p.touch(H(1))  # T2
        p.evict(H(1))  # -> B2 ghost
        self.assertIn(H(1), p._b2)
        p._p = 3
        p.admit(H(1))  # ghost hit in B2 -> frequency pressure, p down
        self.assertLess(p.p, 3)

    def test_ghost_lists_bounded_by_capacity(self):
        p = ArcPolicy(capacity=2)
        for i in range(6):
            p.admit(H(i))
            p.evict(H(i))  # straight to B1
        self.assertLessEqual(len(p._b1), 2)


class StagingDataCacheTest(unittest.TestCase):
    def test_put_get_roundtrip(self):
        c = StagingDataCache(policy="lru", capacity=4)
        self.assertIsNone(c.put(H(1), 10))
        slot = c.get(H(1))
        self.assertIsNotNone(slot)
        self.assertEqual(slot.slot_index, 10)
        self.assertIsNone(c.get(H(2)))  # miss

    def test_dup_put_returns_new_slot_for_release(self):
        c = StagingDataCache(policy="lru", capacity=4)
        c.put(H(1), 10)
        # Second put of same hash keeps slot 10, hands back 11 to release.
        self.assertEqual(c.put(H(1), 11), 11)
        self.assertEqual(c.get(H(1)).slot_index, 10)

    def test_capacity_self_evicts(self):
        c = StagingDataCache(policy="lru", capacity=2)
        self.assertIsNone(c.put(H(1), 10))
        self.assertIsNone(c.put(H(2), 11))
        freed = c.put(H(3), 12)  # at capacity -> evict oldest (H1/slot10)
        self.assertEqual(freed, 10)
        self.assertNotIn(H(1), c)
        self.assertEqual(len(c), 2)

    def test_evict_one_skips_pinned(self):
        c = StagingDataCache(policy="lru", capacity=4)
        c.put(H(1), 10)
        c.put(H(2), 11)
        c.pin(c.get(H(1)))  # H1 pinned by an in-flight copy
        self.assertEqual(c.evict_one(), 11)  # skips pinned H1, evicts H2
        self.assertIn(H(1), c)

    def test_evict_one_all_pinned_returns_none(self):
        c = StagingDataCache(policy="lru", capacity=4)
        c.put(H(1), 10)
        c.pin(c.get(H(1)))
        self.assertIsNone(c.evict_one())

    def test_zero_capacity_never_caches(self):
        c = StagingDataCache(policy="lru", capacity=0)
        self.assertEqual(c.put(H(1), 10), 10)  # handed straight back
        self.assertNotIn(H(1), c)

    def test_drain_returns_all_slots(self):
        c = StagingDataCache(policy="arc", capacity=4)
        c.put(H(1), 10)
        c.put(H(2), 11)
        self.assertEqual(sorted(c.drain()), [10, 11])
        self.assertEqual(len(c), 0)


if __name__ == "__main__":
    unittest.main()
