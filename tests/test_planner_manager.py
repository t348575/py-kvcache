"""Manager-side wiring for the cost-model load planner.

Covers the glue Task 3 adds to `py_kvcache/vllm.py`: turning candidate prefixes
into `CandidateCostInput`s (probing existence/mid-write without `lookup`'s side
effects), `lookup` reading planner decisions with the mid-write fence always
winning, and config validation at spec construction time. vLLM is not installed in this workspace, so these
tests exercise `py_kvcache.vllm`'s fallback classes exactly like
`test_py_kvcache_lookup_cache.py` and `test_vllm_compat.py` do.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import py_kvcache.vllm as vmod
from py_kvcache.load_planner import LoadDecision, PlanResult
from py_kvcache.vllm import (
    PlanCandidate,
    PlanDecision,
    PlanOutcome,
    PyKvCacheOffloadingSpec,
    SharedStorageOffloadingManager,
)


class _FakeMapper:
    def get_file_name(self, block_hash: bytes) -> bytes:
        return block_hash


def _ctx(req_id: str) -> SimpleNamespace:
    return SimpleNamespace(req_id=req_id)


class _RecordingPlanner:
    """Records the `CandidateCostInput`s it is handed and returns a fixed
    outcome per req_id, so probing/pricing can be asserted without a real
    cost model."""

    def __init__(self, outcome=LoadDecision.ADMIT) -> None:
        self.received = None
        self.outstanding = None
        self._outcome = outcome

    def plan(self, candidates, *, outstanding_load_blocks=()):
        self.received = list(candidates)
        self.outstanding = list(outstanding_load_blocks)
        return {c.req_id: PlanResult(self._outcome, 0) for c in candidates}


class _FakeDecisionPlanner:
    """Only implements `decision_for`, for `lookup`-side tests."""

    def __init__(self, decisions: dict[str, LoadDecision]) -> None:
        self._decisions = decisions

    def decision_for(self, req_id):
        return self._decisions.get(req_id)


class ProbingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.existing: set[bytes] = set()
        patcher = mock.patch("os.path.exists", side_effect=lambda name: name in self.existing)
        self.addCleanup(patcher.stop)
        patcher.start()

        self.planner = _RecordingPlanner()
        self.m = SharedStorageOffloadingManager(
            file_mapper=_FakeMapper(), planner=self.planner
        )
        self.m._get_block_hash = lambda key: key  # type: ignore[assignment]

    def test_first_block_missing_yields_zero_load_blocks(self) -> None:
        candidate = PlanCandidate(
            position=0, req_id="r0", recompute_tokens=48, keys=(b"a", b"b", b"c")
        )
        self.m.plan_candidates([candidate])
        (c,) = self.planner.received
        self.assertEqual(c.load_blocks, 0)
        self.assertEqual(c.loadable_hashes, ())
        self.assertFalse(c.blocked_on_store)

    def test_third_block_missing_yields_two_load_blocks(self) -> None:
        self.existing.update({b"a", b"b"})
        candidate = PlanCandidate(
            position=0, req_id="r0", recompute_tokens=48, keys=(b"a", b"b", b"c")
        )
        self.m.plan_candidates([candidate])
        (c,) = self.planner.received
        self.assertEqual(c.load_blocks, 2)
        self.assertEqual(c.loadable_hashes, (b"a", b"b"))
        self.assertFalse(c.blocked_on_store)

    def test_mid_write_block_counts_toward_prefix_and_flags_blocked(self) -> None:
        self.existing.add(b"a")
        self.m._store_inflight[b"b"] = 1  # mid-write, not yet on disk
        candidate = PlanCandidate(
            position=0, req_id="r0", recompute_tokens=48, keys=(b"a", b"b", b"c")
        )
        self.m.plan_candidates([candidate])
        (c,) = self.planner.received
        self.assertEqual(c.load_blocks, 2)
        self.assertEqual(c.loadable_hashes, (b"a", b"b"))
        self.assertTrue(c.blocked_on_store)

    def test_outstanding_load_blocks_forwarded_and_dram_resident_false(self) -> None:
        self.existing.update({b"a", b"b"})
        candidate = PlanCandidate(
            position=0, req_id="r0", recompute_tokens=48, keys=(b"a", b"b", b"c")
        )
        self.m.plan_candidates([candidate], outstanding_load_blocks=[2, 5])
        self.assertEqual(self.planner.outstanding, [2, 5])
        (c,) = self.planner.received
        self.assertFalse(c.predicted_dram_resident)


class SharedPrefixForwardingTest(unittest.TestCase):
    """The manager reports each candidate's own prefix verbatim; deciding which
    of those hashes actually cost a read is the planner's job (it can only be
    answered against the outcomes it assigns as it walks the window)."""

    def setUp(self) -> None:
        self.existing: set[bytes] = {b"a", b"b", b"c", b"d"}
        patcher = mock.patch("os.path.exists", side_effect=lambda name: name in self.existing)
        self.addCleanup(patcher.stop)
        patcher.start()

        self.planner = _RecordingPlanner()
        self.m = SharedStorageOffloadingManager(
            file_mapper=_FakeMapper(), planner=self.planner
        )
        self.m._get_block_hash = lambda key: key  # type: ignore[assignment]

    def test_identical_prefixes_are_both_reported_in_full(self) -> None:
        c0 = PlanCandidate(position=0, req_id="r0", recompute_tokens=64, keys=(b"a", b"b"))
        c1 = PlanCandidate(position=1, req_id="r1", recompute_tokens=64, keys=(b"a", b"b"))
        self.m.plan_candidates([c0, c1])
        got0, got1 = self.planner.received
        self.assertEqual(got0.loadable_hashes, (b"a", b"b"))
        self.assertEqual(got1.loadable_hashes, (b"a", b"b"))
        self.assertEqual(got1.load_blocks, 2)
        self.assertEqual({got0.req_id, got1.req_id}, {"r0", "r1"})

    def test_partial_overlap_reports_each_full_prefix_in_order(self) -> None:
        c0 = PlanCandidate(position=0, req_id="r0", recompute_tokens=64, keys=(b"a", b"b"))
        c1 = PlanCandidate(
            position=1, req_id="r1", recompute_tokens=96, keys=(b"a", b"b", b"c", b"d")
        )
        self.m.plan_candidates([c0, c1])
        got0, got1 = self.planner.received
        self.assertEqual(got0.loadable_hashes, (b"a", b"b"))
        self.assertEqual(got1.loadable_hashes, (b"a", b"b", b"c", b"d"))
        self.assertEqual(got1.load_blocks, 4)


class LookupDecisionReadTest(unittest.TestCase):
    def setUp(self) -> None:
        self.existing: set[bytes] = set()
        patcher = mock.patch("os.path.exists", side_effect=lambda name: name in self.existing)
        self.addCleanup(patcher.stop)
        patcher.start()

    def _manager(self, decisions: dict[str, LoadDecision]) -> SharedStorageOffloadingManager:
        m = SharedStorageOffloadingManager(
            file_mapper=_FakeMapper(), planner=_FakeDecisionPlanner(decisions)
        )
        m._get_block_hash = lambda key: key  # type: ignore[assignment]
        return m

    def test_decline_overrides_to_false(self) -> None:
        self.existing.add(b"a")  # file exists...
        m = self._manager({"r0": LoadDecision.DECLINE})
        self.assertFalse(m.lookup(b"a", _ctx("r0")))  # ...but planner declines it

    def test_defer_overrides_to_none(self) -> None:
        m = self._manager({"r0": LoadDecision.DEFER})
        self.assertIsNone(m.lookup(b"a", _ctx("r0")))

    def test_admit_falls_through_to_filesystem_result(self) -> None:
        self.existing.add(b"a")
        m = self._manager({"r0": LoadDecision.ADMIT})
        self.assertTrue(m.lookup(b"a", _ctx("r0")))
        m2 = self._manager({"r0": LoadDecision.ADMIT})
        self.assertFalse(m2.lookup(b"b", _ctx("r0")))

    def test_unscored_req_id_falls_through_to_filesystem_result(self) -> None:
        self.existing.add(b"a")
        m = self._manager({"other": LoadDecision.DECLINE})
        self.assertTrue(m.lookup(b"a", _ctx("r0")))


class FenceWinsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.existing: set[bytes] = set()
        patcher = mock.patch("os.path.exists", side_effect=lambda name: name in self.existing)
        self.addCleanup(patcher.stop)
        patcher.start()

    def _manager(self, decisions: dict[str, LoadDecision]) -> SharedStorageOffloadingManager:
        m = SharedStorageOffloadingManager(
            file_mapper=_FakeMapper(), planner=_FakeDecisionPlanner(decisions)
        )
        m._get_block_hash = lambda key: key  # type: ignore[assignment]
        return m

    def test_mid_write_defers_despite_decline(self) -> None:
        m = self._manager({"r0": LoadDecision.DECLINE})
        m._store_inflight[b"a"] = 1
        self.assertIsNone(m.lookup(b"a", _ctx("r0")))

    def test_mid_write_defers_despite_admit(self) -> None:
        m = self._manager({"r0": LoadDecision.ADMIT})
        m._store_inflight[b"a"] = 1
        self.assertIsNone(m.lookup(b"a", _ctx("r0")))


class PlannerOffTest(unittest.TestCase):
    def setUp(self) -> None:
        self.existing: set[bytes] = set()
        patcher = mock.patch("os.path.exists", side_effect=lambda name: name in self.existing)
        self.addCleanup(patcher.stop)
        patcher.start()

        self.m = SharedStorageOffloadingManager(file_mapper=_FakeMapper())
        self.m._get_block_hash = lambda key: key  # type: ignore[assignment]

    def test_supports_planned_defer_preload_false(self) -> None:
        self.assertFalse(self.m.supports_planned_defer_preload)

    def test_plan_candidates_returns_empty_mapping(self) -> None:
        candidate = PlanCandidate(position=0, req_id="r0", recompute_tokens=32, keys=(b"a",))
        self.assertEqual(self.m.plan_candidates([candidate]), {})

    def test_lookup_unchanged(self) -> None:
        self.existing.add(b"a")
        self.assertTrue(self.m.lookup(b"a", _ctx("r0")))
        self.assertFalse(self.m.lookup(b"b", _ctx("r0")))
        self.m.prepare_store([b"c"], _ctx("s0"))
        self.assertIsNone(self.m.lookup(b"c", _ctx("s0")))


class _MappingPlanner:
    """Returns a caller-supplied `PlanResult` per candidate req_id, so the
    `LoadDecision` -> `PlanDecision` boundary conversion and `preload_blocks`
    propagation can be asserted independent of the real cost model."""

    def __init__(self, results: dict[str, PlanResult]) -> None:
        self._results = results

    def plan(self, candidates, *, outstanding_load_blocks=()):
        return {c.req_id: self._results[c.req_id] for c in candidates}


class PlanCandidatesConversionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.existing: set[bytes] = {b"a", b"b", b"c"}
        patcher = mock.patch("os.path.exists", side_effect=lambda name: name in self.existing)
        self.addCleanup(patcher.stop)
        patcher.start()

    def test_all_three_decisions_and_preload_blocks_propagate(self) -> None:
        planner = _MappingPlanner(
            {
                "admit": PlanResult(LoadDecision.ADMIT, 0),
                "defer": PlanResult(LoadDecision.DEFER, 3),
                "decline": PlanResult(LoadDecision.DECLINE, 0),
            }
        )
        m = SharedStorageOffloadingManager(file_mapper=_FakeMapper(), planner=planner)
        m._get_block_hash = lambda key: key  # type: ignore[assignment]
        candidates = [
            PlanCandidate(position=0, req_id="admit", recompute_tokens=16, keys=(b"a",)),
            PlanCandidate(position=1, req_id="defer", recompute_tokens=16, keys=(b"b",)),
            PlanCandidate(position=2, req_id="decline", recompute_tokens=16, keys=(b"c",)),
        ]
        outcomes = m.plan_candidates(candidates)
        self.assertEqual(
            outcomes,
            {
                "admit": PlanOutcome(PlanDecision.ADMIT, 0),
                "defer": PlanOutcome(PlanDecision.DEFER, 3),
                "decline": PlanOutcome(PlanDecision.DECLINE, 0),
            },
        )


class ProbeThenLookupEventTest(unittest.TestCase):
    """Demonstrates F2: probing a hash must not suppress the
    `py_kvcache.manager.lookup` event on the first real `lookup` of that hash,
    but repeat lookups still must not re-emit."""

    def setUp(self) -> None:
        self.existing: set[bytes] = {b"a"}
        exists_patcher = mock.patch(
            "os.path.exists", side_effect=lambda name: name in self.existing
        )
        self.addCleanup(exists_patcher.stop)
        exists_patcher.start()

        self.events: list[str] = []
        event_patcher = mock.patch(
            "py_kvcache.vllm.add_event",
            side_effect=lambda name, *a, **kw: self.events.append(name),
        )
        self.addCleanup(event_patcher.stop)
        event_patcher.start()

    def test_probe_then_lookup_emits_once_repeat_lookup_does_not(self) -> None:
        m = SharedStorageOffloadingManager(file_mapper=_FakeMapper())
        m._get_block_hash = lambda key: key  # type: ignore[assignment]

        m._probe(b"a")
        self.assertEqual(self.events, [])  # probing alone never emits

        m.lookup(b"a", _ctx("r0"))
        self.assertEqual(self.events.count("py_kvcache.manager.lookup"), 1)

        m.lookup(b"a", _ctx("r0"))
        self.assertEqual(self.events.count("py_kvcache.manager.lookup"), 1)  # not re-emitted


class _ResettablePlanner:
    def __init__(self, decisions: dict[str, LoadDecision]) -> None:
        self._decisions = dict(decisions)
        self.reset_called = False

    def decision_for(self, req_id):
        return self._decisions.get(req_id)

    def reset(self) -> None:
        self.reset_called = True
        self._decisions.clear()


class ResetCacheResetsPlannerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.existing: set[bytes] = set()
        patcher = mock.patch("os.path.exists", side_effect=lambda name: name in self.existing)
        self.addCleanup(patcher.stop)
        patcher.start()

    def test_reset_cache_resets_planner_state(self) -> None:
        planner = _ResettablePlanner({"r0": LoadDecision.DECLINE})
        m = SharedStorageOffloadingManager(file_mapper=_FakeMapper(), planner=planner)
        m._get_block_hash = lambda key: key  # type: ignore[assignment]

        self.existing.add(b"a")
        self.assertFalse(m.lookup(b"a", _ctx("r0")))  # DECLINE overrides the hit

        m.reset_cache()
        self.assertTrue(planner.reset_called)

        # A stale DECLINE surviving reset would wrongly suppress this fresh hit.
        self.assertTrue(m.lookup(b"a", _ctx("r0")))


# --- Config validation -------------------------------------------------------

# Last knot == the fixtures' max_model_len, so building these tables never trips
# the past-the-measured-range extrapolation warning.
_V2_CURVES = {
    "curves": {
        "f": {"floor": 0.0, "knots": {"0": 0.0, "32768": 0.09}},
        "g_ssd": {"floor": 0.031, "knots": {"0": 0.031, "32768": 0.05}},
        "g_mem": {"floor": 0.003, "knots": {"0": 0.003, "32768": 0.006}},
    },
    "kv_bytes_per_token": 131072,
}


@unittest.skipIf(vmod.VLLM_AVAILABLE, "exercises the no-vLLM fallback OffloadingSpec")
class SpecConfigValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.root_dir = self._tmpdir.name

    def _write_json(self, payload: object) -> str:
        fd, path = tempfile.mkstemp(suffix=".json", dir=self.root_dir)
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f)
        return path

    def _vllm_config(
        self, *, enable_prefix_caching: bool = False, max_model_len: int | None = 32768
    ) -> SimpleNamespace:
        return SimpleNamespace(
            model_config=SimpleNamespace(model="test-model", max_model_len=max_model_len),
            cache_config=SimpleNamespace(
                cache_dtype="auto", enable_prefix_caching=enable_prefix_caching
            ),
            parallel_config=SimpleNamespace(
                tensor_parallel_size=1,
                pipeline_parallel_size=1,
                prefill_context_parallel_size=1,
                rank=0,
            ),
        )

    def _extra_config(self, **overrides: object) -> dict:
        cfg = {
            "shared_storage_path": self.root_dir,
            "enable_preload": True,
            "preload_lookahead_requests": 8,
            "load_planner": "on",
            "prefix_cache_break_even_path": self._write_json(dict(_V2_CURVES)),
        }
        cfg.update(overrides)
        return cfg

    def _build(
        self,
        *,
        enable_prefix_caching: bool = False,
        max_model_len: int | None = 32768,
        **extra_overrides: object,
    ):
        vllm_config = self._vllm_config(
            enable_prefix_caching=enable_prefix_caching, max_model_len=max_model_len
        )
        vllm_config.kv_connector_extra_config = self._extra_config(**extra_overrides)
        return PyKvCacheOffloadingSpec(vllm_config, SimpleNamespace())

    def test_v1_file_with_planner_on_raises(self) -> None:
        v1_path = self._write_json({"break_even_ssd_tokens": 100, "break_even_mem_tokens": 10})
        with self.assertRaises(ValueError):
            self._build(prefix_cache_break_even_path=v1_path)

    def test_enable_preload_false_raises(self) -> None:
        with self.assertRaises(ValueError):
            self._build(enable_preload=False)

    def test_preload_lookahead_requests_absent_raises(self) -> None:
        with self.assertRaises(ValueError):
            self._build(preload_lookahead_requests=None)

    def test_preload_lookahead_requests_zero_raises(self) -> None:
        with self.assertRaises(ValueError):
            self._build(preload_lookahead_requests=0)

    def test_preload_share_staging_off_raises(self) -> None:
        # The planner charges one slot per distinct hash; multi-copy staging holds
        # one per (req_id, hash), so its budget would be N times optimistic.
        with self.assertRaises(ValueError):
            self._build(preload_share_staging=False)

    def test_unpatched_vllm_raises(self) -> None:
        # vLLM installed but without the planned-defer preload API: plan_candidates
        # would never be called, so load_planner=on would silently no-op.
        with mock.patch.object(vmod, "VLLM_AVAILABLE", True), mock.patch.object(
            vmod, "PLAN_API_AVAILABLE", False
        ):
            with self.assertRaises(ValueError):
                self._build()

    def test_absent_vllm_still_builds(self) -> None:
        # The pure-Python workspace has no vLLM at all; that must stay buildable.
        self.assertFalse(vmod.VLLM_AVAILABLE)
        self.assertIsNotNone(self._build()._planner)

    def test_missing_kv_bytes_per_token_raises(self) -> None:
        payload = dict(_V2_CURVES)
        payload = {k: v for k, v in payload.items() if k != "kv_bytes_per_token"}
        with self.assertRaises(ValueError):
            self._build(prefix_cache_break_even_path=self._write_json(payload))

    def test_missing_max_model_len_raises(self) -> None:
        with self.assertRaises(ValueError):
            self._build(max_model_len=None)

    def test_non_positive_max_model_len_raises(self) -> None:
        with self.assertRaises(ValueError):
            self._build(max_model_len=0)

    def test_valid_config_builds_planner(self) -> None:
        spec = self._build()
        self.assertIsNotNone(spec._planner)
        manager = spec.get_manager()
        self.assertTrue(manager.supports_planned_defer_preload)

    def test_realistic_block_size_builds_planner(self) -> None:
        # The no-vLLM stub defaults gpu_block_size and block_size_factor to 1,
        # so pass nontrivial values to exercise the real io_size arithmetic.
        spec = self._build(gpu_block_size=16, block_size_factor=4)
        self.assertEqual(spec.storage_block_tokens, 64)
        self.assertEqual(spec._planner._tables.block_tokens, 64)
        self.assertEqual(spec._planner._max_preload_slots, 112)

    def test_planner_off_by_default_skips_all_validation(self) -> None:
        vllm_config = self._vllm_config()
        vllm_config.kv_connector_extra_config = {
            "shared_storage_path": self.root_dir,
            # No load_planner key at all, and an otherwise-invalid combination
            # (no enable_preload) that would fail validation if load_planner were on.
        }
        spec = PyKvCacheOffloadingSpec(vllm_config, SimpleNamespace())
        self.assertIsNone(spec._planner)
        self.assertFalse(spec.get_manager().supports_planned_defer_preload)


if __name__ == "__main__":
    unittest.main()
