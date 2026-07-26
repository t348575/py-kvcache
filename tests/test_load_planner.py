from __future__ import annotations

import unittest

from py_kvcache.break_even import Curve, CurveData, GoldenPoint
from py_kvcache.cost_model import build_cost_tables
from py_kvcache.load_planner import CandidateCostInput, LoadDecision, LoadPlanner

# Synthetic affine cost curves with a break-even crossover around 5.5 blocks.
_MAX_BLOCKS = 100
_F_RATE = 1.0
_SSD_FIXED = 5.0
_SSD_RATE = 0.1
_MEM_FIXED = 0.5
_MEM_RATE = 0.05


def _f(n: int) -> float:
    return _F_RATE * n


def _g_ssd(n: int) -> float:
    return _SSD_FIXED + _SSD_RATE * n


def _g_mem(n: int) -> float:
    return _MEM_FIXED + _MEM_RATE * n


def _service(n: int) -> float:
    return max(0.0, _g_ssd(n) - _g_mem(n))


def _build_curve(fn) -> Curve:
    knots = tuple((n, fn(n)) for n in range(0, _MAX_BLOCKS + 1))
    return Curve(floor=0.0, knots=knots)


def _make_tables():
    return _make_tables_with_block_tokens(1)


def _make_tables_with_block_tokens(block_tokens: int):
    curves = CurveData(
        f=_build_curve(_f),
        g_ssd=_build_curve(_g_ssd),
        g_mem=_build_curve(_g_mem),
        golden=(GoldenPoint(tokens=50, f=_f(50), g_ssd=_g_ssd(50), g_mem=_g_mem(50)),),
        kv_bytes_per_token=None,
    )
    return build_cost_tables(curves, block_tokens=block_tokens, max_model_len=_MAX_BLOCKS)


class _FakeClock:
    def __init__(self, start: float) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now


def _own(req_id: str, count: int) -> tuple[bytes, ...]:
    """Hashes unique to `req_id` -- every block is a distinct read."""
    return tuple(f"{req_id}:{i}".encode() for i in range(count))


def _shared(count: int, *, start: int = 0) -> tuple[bytes, ...]:
    """Hashes drawn from one pool, so candidates sharing a prefix share reads."""
    return tuple(f"shared:{i}".encode() for i in range(start, start + count))


def _candidate(
    req_id: str,
    *,
    position: int = 0,
    recompute_tokens: int = 0,
    load_blocks: int = 0,
    loadable_hashes: tuple[bytes, ...] | None = None,
    blocked_on_store: bool = False,
    predicted_dram_resident: bool = False,
) -> CandidateCostInput:
    if loadable_hashes is None:
        loadable_hashes = _own(req_id, load_blocks)
    return CandidateCostInput(
        position=position,
        req_id=req_id,
        recompute_tokens=recompute_tokens,
        load_blocks=load_blocks,
        loadable_hashes=loadable_hashes,
        blocked_on_store=blocked_on_store,
        predicted_dram_resident=predicted_dram_resident,
    )


class LoadPlannerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tables = _make_tables()

    def _planner(
        self,
        *,
        max_preload_slots=100,
        defer_tolerance=2.0,
        defer_deadline_min_s=0.05,
        defer_deadline_max_s=2.0,
        clock=None,
    ):
        return LoadPlanner(
            self.tables,
            max_preload_slots=max_preload_slots,
            defer_tolerance=defer_tolerance,
            defer_deadline_min_s=defer_deadline_min_s,
            defer_deadline_max_s=defer_deadline_max_s,
            time_source=clock or (lambda: 1000.0),
        )

    def _fixed_deadline_planner(self, grace_s, *, max_preload_slots=100, clock=None):
        """Planner whose defer deadline is exactly `grace_s`, whatever the prefix.

        Pinning the clamp to a single value takes the cost model out of the
        deadline tests so they assert the deadline mechanism itself.
        """
        return self._planner(
            max_preload_slots=max_preload_slots,
            defer_deadline_min_s=grace_s,
            defer_deadline_max_s=grace_s,
            clock=clock,
        )

    def test_predicted_resident_admits(self) -> None:
        planner = self._planner()
        c = _candidate("r1", recompute_tokens=1, load_blocks=1, predicted_dram_resident=True)
        result = planner.plan([c])["r1"]
        self.assertEqual(result.decision, LoadDecision.ADMIT)
        self.assertEqual(result.preload_blocks, 0)

    def test_missing_prefix_declines(self) -> None:
        planner = self._planner()
        c = _candidate("r1", recompute_tokens=5, load_blocks=0)
        result = planner.plan([c])["r1"]
        self.assertEqual(result.decision, LoadDecision.DECLINE)
        self.assertEqual(result.preload_blocks, 0)

    def test_mid_write_defer_is_budgeted_and_on_the_timeline(self) -> None:
        planner = self._planner(max_preload_slots=6)
        blocked = _candidate("blocked", position=0, recompute_tokens=1, load_blocks=5,
                              blocked_on_store=True)
        follower = _candidate("follower", position=1, recompute_tokens=20, load_blocks=6)

        results = planner.plan([blocked, follower])

        self.assertEqual(results["blocked"].decision, LoadDecision.DEFER)
        self.assertEqual(results["blocked"].preload_blocks, 5)
        # The fence never arms a defer deadline.
        self.assertNotIn("blocked", planner._deadline_at)
        self.assertEqual(len(planner._entries), 1)
        self.assertEqual(planner._entries[0].req_id, "blocked")
        self.assertAlmostEqual(planner._entries[0].service_s, _service(5))
        # Only 1 of 6 slots remain, so the follower's 6 distinct reads don't fit.
        self.assertEqual(results["follower"].decision, LoadDecision.ADMIT)
        self.assertEqual(results["follower"].preload_blocks, 0)

    def test_mid_write_defer_over_budget_emits_nothing(self) -> None:
        # Still DEFER (the fence is unconditional) but with no preload spec.
        planner = self._planner(max_preload_slots=4)
        blocked = _candidate("blocked", recompute_tokens=1, load_blocks=5,
                              blocked_on_store=True)

        result = planner.plan([blocked])["blocked"]

        self.assertEqual(result.decision, LoadDecision.DEFER)
        self.assertEqual(result.preload_blocks, 0)
        self.assertEqual(len(planner._entries), 0)

    def test_mid_write_redefer_never_reaches_the_deadline(self) -> None:
        # A long store must not become a deadline decision or a second shadow entry.
        clock = _FakeClock(1000.0)
        planner = self._fixed_deadline_planner(0.01, max_preload_slots=5, clock=clock)
        blocked = _candidate("blocked", recompute_tokens=1, load_blocks=5,
                              blocked_on_store=True)

        for _ in range(4):
            result = planner.plan([blocked])["blocked"]
            self.assertEqual(result.decision, LoadDecision.DEFER)
            self.assertEqual(result.preload_blocks, 5)
            clock.now += 1.0  # far past any deadline the fence might have armed

        self.assertEqual(len(planner._entries), 1)
        self.assertNotIn("blocked", planner._deadline_at)

    def test_break_even_declines_when_recompute_cheaper(self) -> None:
        planner = self._planner()
        # n=5: f(5)=5.0 <= service(5)+mem(5) == t_load -> decline.
        c = _candidate("small", recompute_tokens=5, load_blocks=5)
        self.assertLessEqual(_f(5), _service(5) + _g_mem(5))
        result = planner.plan([c])["small"]
        self.assertEqual(result.decision, LoadDecision.DECLINE)

    def test_position_changes_decision(self) -> None:
        target_kwargs = dict(recompute_tokens=5, load_blocks=5)

        alone = self._planner()
        target_alone = _candidate("target", position=0, **target_kwargs)
        self.assertEqual(
            alone.plan([target_alone])["target"].decision, LoadDecision.DECLINE
        )

        preceded = self._planner()
        precedents = [
            _candidate(f"p{i}", position=i, recompute_tokens=1, load_blocks=1)
            for i in range(3)
        ]
        target_at_3 = _candidate("target", position=3, **target_kwargs)
        results = preceded.plan([*precedents, target_at_3])
        for i in range(3):
            self.assertEqual(results[f"p{i}"].decision, LoadDecision.DECLINE)
        self.assertEqual(results["target"].decision, LoadDecision.DEFER)
        self.assertEqual(results["target"].preload_blocks, 5)

    def test_gpu_wait_overlaps_storage_wait(self) -> None:
        # t_load must be max(storage_done, gpu_wait) + mem. Summing all three
        # instead would wrongly decline j here.
        planner = self._planner()
        p = _candidate("p", position=0, recompute_tokens=10, load_blocks=0)
        j = _candidate("j", position=1, recompute_tokens=1, load_blocks=1)

        self.assertLess(_service(1), _f(10))  # gpu_wait will exceed storage_done
        t_load_correct = _f(10) + _g_mem(1)
        t_load_wrong_sum = _service(1) + _f(10) + _g_mem(1)
        t_recompute_j = _f(10) + _f(1)
        self.assertGreater(t_recompute_j, t_load_correct)
        self.assertLessEqual(t_recompute_j, t_load_wrong_sum)

        results = planner.plan([p, j])
        self.assertEqual(results["p"].decision, LoadDecision.DECLINE)
        self.assertEqual(results["j"].decision, LoadDecision.DEFER)

    def test_shared_prefix_charges_only_the_new_distinct_reads(self) -> None:
        planner = self._planner(max_preload_slots=100)
        c1 = _candidate("c1", position=0, recompute_tokens=8, load_blocks=8,
                         loadable_hashes=_shared(8))
        # c2 shares 6 of c1's blocks; only 2 are new distinct reads.
        c2 = _candidate("c2", position=1, recompute_tokens=20, load_blocks=8,
                         loadable_hashes=_shared(6) + _own("c2", 2))

        results = planner.plan([c1, c2])

        self.assertEqual(results["c1"].decision, LoadDecision.DEFER)
        self.assertEqual(results["c1"].preload_blocks, 8)  # identity: distinct == load
        self.assertEqual(results["c2"].decision, LoadDecision.DEFER)
        # Full 8-block prefix emitted though only 2 blocks are charged.
        self.assertEqual(results["c2"].preload_blocks, 8)

        self.assertEqual(len(planner._entries), 2)
        self.assertAlmostEqual(planner._entries[0].service_s, _service(8))
        self.assertAlmostEqual(planner._entries[1].service_s, _service(2))

    def test_shadow_residual_uses_final_completion_not_sum(self) -> None:
        clock = _FakeClock(1000.0)
        planner = self._planner(max_preload_slots=100, clock=clock)
        c1 = _candidate("c1", position=0, recompute_tokens=6, load_blocks=6)
        c2 = _candidate("c2", position=1, recompute_tokens=15, load_blocks=6)
        planner.plan([c1, c2])

        self.assertEqual(len(planner._entries), 2)

        # Derive expected completion times independently, not by reading
        # planner._entries back -- that would mask a chaining bug in lockstep.
        service = _service(6)
        end0 = 1000.0 + service                    # first entry starts at now
        end1 = max(1000.0, end0) + service          # second queues behind the first
        unchained_end1 = 1000.0 + service           # what starts_at = now would give

        now = 1002.0
        correct = max(0.0, end1 - now)
        wrong_sum = max(0.0, end0 - now) + max(0.0, end1 - now)
        unchained = max(0.0, unchained_end1 - now)
        self.assertNotAlmostEqual(correct, wrong_sum)
        self.assertNotAlmostEqual(correct, unchained)
        self.assertAlmostEqual(planner._shadow_residual(now), correct)

    def test_jobs_floor_wins_over_idle_shadow_model(self) -> None:
        idle = self._planner()
        d_idle = _candidate("d", recompute_tokens=6, load_blocks=6)
        self.assertEqual(idle.plan([d_idle])["d"].decision, LoadDecision.DEFER)

        busy = self._planner()
        d_busy = _candidate("d", recompute_tokens=6, load_blocks=6)
        result = busy.plan([d_busy], outstanding_load_blocks=[50])["d"]
        self.assertEqual(result.decision, LoadDecision.DECLINE)

    def test_budget_exhaustion_forces_admit_without_shadow_entry(self) -> None:
        planner = self._planner(max_preload_slots=10)
        a = _candidate("a", position=0, recompute_tokens=10, load_blocks=10)
        b = _candidate("b", position=1, recompute_tokens=100, load_blocks=100)
        # z's break-even hinges on b's budget-exhausted ADMIT still advancing
        # storage_wait to storage_done_b.
        z = _candidate("z", position=2, recompute_tokens=15, load_blocks=1)

        results = planner.plan([a, b, z])

        self.assertEqual(results["a"].decision, LoadDecision.DEFER)
        self.assertEqual(results["a"].preload_blocks, 10)
        self.assertEqual(results["b"].decision, LoadDecision.ADMIT)
        self.assertEqual(results["b"].preload_blocks, 0)
        self.assertEqual(len(planner._entries), 1)
        self.assertEqual(planner._entries[0].req_id, "a")
        self.assertEqual(results["z"].decision, LoadDecision.DECLINE)

    def test_decision_map_rebuilt_wholesale(self) -> None:
        planner = self._planner()
        c = _candidate("gone", recompute_tokens=5, load_blocks=5)
        self.assertEqual(planner.plan([c])["gone"].decision, LoadDecision.DECLINE)
        self.assertEqual(planner.decision_for("gone"), LoadDecision.DECLINE)

        other = _candidate("other", recompute_tokens=5, load_blocks=5)
        planner.plan([other])
        self.assertIsNone(planner.decision_for("gone"))

    def test_unscored_request_returns_none(self) -> None:
        planner = self._planner()
        self.assertIsNone(planner.decision_for("never-seen"))

    def test_deferred_past_deadline_admits_instead_of_recomputing(self) -> None:
        # Past the deadline, admit foreground rather than discard a real cache hit.
        clock = _FakeClock(1000.0)
        planner = self._fixed_deadline_planner(0.5, clock=clock)
        h = _candidate("h", recompute_tokens=1000, load_blocks=6)

        first = planner.plan([h])["h"]
        clock.now += 0.4  # inside the grace period
        second = planner.plan([h])["h"]
        clock.now += 0.2  # past it
        third = planner.plan([h])["h"]

        self.assertEqual(first.decision, LoadDecision.DEFER)
        self.assertEqual(second.decision, LoadDecision.DEFER)
        self.assertEqual(third.decision, LoadDecision.ADMIT)
        self.assertEqual(third.preload_blocks, 0)

    def test_deadline_does_not_trip_on_plan_call_rate(self) -> None:
        # Deadline is wall-clock, not a plan()-call count.
        clock = _FakeClock(1000.0)
        planner = self._fixed_deadline_planner(0.5, clock=clock)
        h = _candidate("h", recompute_tokens=1000, load_blocks=6)

        for _ in range(200):
            self.assertEqual(planner.plan([h])["h"].decision, LoadDecision.DEFER)
            clock.now += 0.001  # 200 calls, 0.2s of wall clock

    def test_deadline_scales_with_the_predicted_wait(self) -> None:
        # A big prefix gets a proportionally longer grace than a small one.
        clock = _FakeClock(1000.0)
        planner = self._planner(defer_tolerance=2.0, defer_deadline_min_s=0.0,
                                 defer_deadline_max_s=1e9, clock=clock)
        small = _candidate("small", position=0, recompute_tokens=1000, load_blocks=2)
        big = _candidate("big", position=1, recompute_tokens=1000, load_blocks=60)

        planner.plan([small, big])

        small_grace = planner._deadline_at["small"] - 1000.0
        big_grace = planner._deadline_at["big"] - 1000.0
        self.assertGreater(big_grace, small_grace)
        self.assertAlmostEqual(small_grace, 2.0 * _service(2))

    def test_deadline_ceiling_caps_the_grace_period(self) -> None:
        # Ceiling must win over tolerance x predicted_wait.
        clock = _FakeClock(1000.0)
        planner = self._planner(defer_tolerance=1000.0, defer_deadline_max_s=0.25,
                                 defer_deadline_min_s=0.0, clock=clock)
        h = _candidate("h", recompute_tokens=1000, load_blocks=6)

        self.assertEqual(planner.plan([h])["h"].decision, LoadDecision.DEFER)
        self.assertAlmostEqual(planner._deadline_at["h"] - 1000.0, 0.25)

        clock.now += 0.3
        self.assertEqual(planner.plan([h])["h"].decision, LoadDecision.ADMIT)

    def test_deadline_admit_is_sticky(self) -> None:
        # A deadline ADMIT stays admitted on the next call, no defer relapse.
        clock = _FakeClock(1000.0)
        planner = self._fixed_deadline_planner(0.1, clock=clock)
        h = _candidate("h", recompute_tokens=1000, load_blocks=6)

        self.assertEqual(planner.plan([h])["h"].decision, LoadDecision.DEFER)
        clock.now += 0.2
        for _ in range(3):
            self.assertEqual(planner.plan([h])["h"].decision, LoadDecision.ADMIT)
            clock.now += 0.01

    def test_deadline_admit_still_declines_when_recompute_wins(self) -> None:
        # Break-even is evaluated before the deadline.
        clock = _FakeClock(1000.0)
        planner = self._fixed_deadline_planner(0.1, clock=clock)
        cheap = _candidate("cheap", recompute_tokens=1, load_blocks=6)

        self.assertEqual(planner.plan([cheap])["cheap"].decision, LoadDecision.DECLINE)
        clock.now += 0.2
        self.assertEqual(planner.plan([cheap])["cheap"].decision, LoadDecision.DECLINE)

    def test_shadow_retirement_makes_resident_next_call(self) -> None:
        clock = _FakeClock(1000.0)
        planner = self._planner(max_preload_slots=100, clock=clock)
        e = _candidate("e", recompute_tokens=6, load_blocks=6)

        first = planner.plan([e])["e"]
        self.assertEqual(first.decision, LoadDecision.DEFER)
        ends_at = planner._entries[0].ends_at

        clock.now = ends_at + 1.0
        second = planner.plan([e])["e"]
        self.assertEqual(second.decision, LoadDecision.ADMIT)
        self.assertEqual(second.preload_blocks, 0)
        # The resident ADMIT must consume the predicted-resident hint.
        self.assertNotIn("e", planner._staged)

    def test_staged_hint_garbage_collected_after_prolonged_absence(self) -> None:
        clock = _FakeClock(1000.0)
        planner = self._planner(max_preload_slots=100, clock=clock)
        e = _candidate("e", recompute_tokens=6, load_blocks=6)
        filler = _candidate("filler", recompute_tokens=1, load_blocks=1)

        first = planner.plan([e])["e"]
        self.assertEqual(first.decision, LoadDecision.DEFER)
        ends_at = planner._entries[0].ends_at

        clock.now = ends_at + 1.0
        planner.plan([filler])  # retires "e" into _staged; "e" absent this call
        self.assertIn("e", planner._staged)

        from py_kvcache.load_planner import _STAGED_ABSENCE_LIMIT

        for _ in range(_STAGED_ABSENCE_LIMIT):
            planner.plan([filler])

        self.assertNotIn("e", planner._staged)

    def test_staging_by_req_popped_at_retirement_allows_fresh_redefer(self) -> None:
        # Retirement must remove the stale _staging_by_req entry, or a later
        # re-defer reuses its frozen emitted_blocks instead of rescoring.
        clock = _FakeClock(1000.0)
        planner = self._planner(max_preload_slots=100, clock=clock)
        filler = _candidate("filler", recompute_tokens=1, load_blocks=1)

        e = _candidate("e", recompute_tokens=6, load_blocks=6)
        first = planner.plan([e])["e"]
        self.assertEqual(first.decision, LoadDecision.DEFER)
        self.assertEqual(first.preload_blocks, 6)

        ends_at = 1000.0 + _service(6)
        clock.now = ends_at + 1.0
        planner.plan([filler])  # retires "e" into _staged; "e" absent this call
        self.assertIn("e", planner._staged)

        from py_kvcache.load_planner import _STAGED_ABSENCE_LIMIT

        for _ in range(_STAGED_ABSENCE_LIMIT):
            planner.plan([filler])
        self.assertNotIn("e", planner._staged)

        # "e" reappears with a different prefix (4 blocks, not 6); the defer
        # must price off the new shape, not a stale outstanding entry.
        e2 = _candidate("e", recompute_tokens=6, load_blocks=4)
        result = planner.plan([e2])["e"]

        self.assertEqual(result.decision, LoadDecision.DEFER)
        self.assertEqual(result.preload_blocks, 4)
        self.assertAlmostEqual(planner._entries[-1].service_s, _service(4))

    def test_defer_deadline_survives_brief_candidate_absence(self) -> None:
        # A request briefly absent from the candidate set must not have its
        # defer deadline re-armed while its shadow entry is still outstanding.
        clock = _FakeClock(1000.0)
        planner = self._fixed_deadline_planner(0.1, clock=clock)
        h = _candidate("h", recompute_tokens=1000, load_blocks=6)
        filler = _candidate("filler", recompute_tokens=1, load_blocks=1)

        first = planner.plan([h])["h"]
        self.assertEqual(first.decision, LoadDecision.DEFER)
        armed_at = planner._deadline_at["h"]

        clock.now += 0.2
        planner.plan([filler])  # "h" absent this call; its shadow entry persists

        second = planner.plan([h])["h"]
        self.assertEqual(second.decision, LoadDecision.ADMIT)
        self.assertEqual(planner._deadline_at["h"], armed_at)

    def test_redeferred_request_does_not_inflate_storage_wait(self) -> None:
        # A request re-deferred across several calls must reuse its outstanding
        # shadow entry, not append a new one and re-charge storage. "h" reappears
        # with a longer prefix (6) than its outstanding entry covers (4), so the
        # two counts stay distinguishable.
        planner = self._fixed_deadline_planner(1e9, max_preload_slots=4)
        h = _candidate("h", recompute_tokens=1000, load_blocks=4)

        first = planner.plan([h])["h"]
        self.assertEqual(first.decision, LoadDecision.DEFER)
        self.assertEqual(first.preload_blocks, 4)
        self.assertEqual(len(planner._entries), 1)

        # A peer after h's redefer must see budget and storage_wait untouched
        # by the redefer branch (it must not re-subtract or re-charge them).
        h_grown = _candidate("h", recompute_tokens=1000, load_blocks=6,
                              loadable_hashes=_own("h", 6))
        peer = _candidate("peer", position=1, recompute_tokens=12, load_blocks=3)

        results = planner.plan([h_grown, peer])

        self.assertEqual(results["h"].decision, LoadDecision.DEFER)
        self.assertEqual(results["h"].preload_blocks, 4)  # reused, not c.load_blocks=6
        self.assertEqual(results["peer"].decision, LoadDecision.DEFER)
        self.assertEqual(results["peer"].preload_blocks, 3)

        self.assertEqual(len(planner._entries), 2)
        self.assertAlmostEqual(planner._entries[0].service_s, _service(4))
        self.assertAlmostEqual(planner._entries[1].service_s, _service(3))

    def test_fully_shared_candidate_defers_for_free(self) -> None:
        # A candidate fully covered by an earlier read must DEFER at zero cost
        # even with the budget exhausted (c1 takes all 8 slots).
        planner = self._planner(max_preload_slots=8)
        c1 = _candidate("c1", position=0, recompute_tokens=8, load_blocks=8,
                         loadable_hashes=_shared(8))
        c2 = _candidate("c2", position=1, recompute_tokens=20, load_blocks=8,
                         loadable_hashes=_shared(8))

        results = planner.plan([c1, c2])

        self.assertEqual(results["c1"].decision, LoadDecision.DEFER)
        self.assertEqual(results["c2"].decision, LoadDecision.DEFER)
        self.assertEqual(results["c2"].preload_blocks, 8)

    def test_fully_shared_candidate_advances_storage_wait_by_nothing(self) -> None:
        # A zero-distinct candidate must be charged exactly 0.0, not
        # service_s_of(0) (nonzero here since g_ssd/g_mem have different floors).
        clock = _FakeClock(1000.0)
        planner = self._planner(max_preload_slots=100, clock=clock)
        c1 = _candidate("c1", position=0, recompute_tokens=8, load_blocks=8,
                         loadable_hashes=_shared(8))
        c2 = _candidate("c2", position=1, recompute_tokens=20, load_blocks=8,
                         loadable_hashes=_shared(8))
        # A peer after c2 whose break-even hinges on storage_wait carrying only c1's charge.
        peer = _candidate("peer", position=2, recompute_tokens=12, load_blocks=1)
        self.assertGreater(_service(0), 0.0)

        results = planner.plan([c1, c2, peer])

        self.assertEqual(results["c2"].decision, LoadDecision.DEFER)
        self.assertEqual(results["c2"].preload_blocks, 8)
        self.assertAlmostEqual(planner._entries[1].service_s, 0.0)
        self.assertEqual(results["peer"].decision, LoadDecision.DEFER)

        # The shared candidate's shadow entry retires in lockstep with the peer
        # it rode on, becoming predicted-resident exactly when that read completes.
        c1_ends_at = 1000.0 + _service(8)
        clock.now = c1_ends_at + 1.0
        resident = planner.plan([c2])["c2"]
        self.assertEqual(resident.decision, LoadDecision.ADMIT)
        self.assertEqual(resident.preload_blocks, 0)

    def test_decision_pricing_uses_distinct_reads_not_load_blocks(self) -> None:
        # t_load must be priced off the distinct read count, not the full
        # (mostly-shared) load_blocks, or c2 wrongly declines.
        planner = self._planner(max_preload_slots=100)
        c1 = _candidate("c1", position=0, recompute_tokens=49, load_blocks=49,
                         loadable_hashes=_shared(49))
        c2 = _candidate("c2", position=1, recompute_tokens=15, load_blocks=50,
                         loadable_hashes=_shared(49) + _own("c2", 1))

        results = planner.plan([c1, c2])

        self.assertEqual(results["c1"].decision, LoadDecision.DEFER)
        self.assertEqual(results["c2"].decision, LoadDecision.DEFER)

    def test_budget_shortfall_admits_instead_of_deferring_a_partial_prefix(self) -> None:
        # All-or-ADMIT: reads that don't fit the remaining budget go foreground
        # rather than staging a partial prefix that would freeze forever.
        planner = self._planner(max_preload_slots=6)
        c = _candidate("c", position=0, recompute_tokens=100, load_blocks=10)

        result = planner.plan([c])["c"]

        self.assertEqual(result.decision, LoadDecision.ADMIT)
        self.assertEqual(result.preload_blocks, 0)
        self.assertEqual(len(planner._entries), 0)
        self.assertNotIn("c", planner._deadline_at)

    def test_short_request_rides_the_front_of_a_long_shared_prefix(self) -> None:
        # A short request sharing the leading blocks of a long read must be
        # priced off when its own blocks land, not the whole read's completion.
        planner = self._planner(max_preload_slots=1000)
        prefix = _shared(200)
        big = _candidate("big", position=0, recompute_tokens=100_000,
                          load_blocks=200, loadable_hashes=prefix)
        small = _candidate("small", position=1, recompute_tokens=8,
                            load_blocks=8, loadable_hashes=prefix[:8])

        results = planner.plan([big, small])

        self.assertEqual(results["big"].decision, LoadDecision.DEFER)
        self.assertEqual(results["small"].decision, LoadDecision.DEFER)
        self.assertEqual(results["small"].preload_blocks, 8)

    def test_covered_blocks_are_timed_across_the_read_not_at_its_end(self) -> None:
        # A hash at the front of a read must be marked ready earlier than one at the back.
        from py_kvcache.load_planner import _record_staged

        staged: dict[bytes, float] = {}
        hashes = tuple(f"h{i}".encode() for i in range(10))
        _record_staged(staged, hashes, 0.0, 1.0)

        self.assertLess(staged[hashes[0]], staged[hashes[-1]])
        self.assertAlmostEqual(staged[hashes[-1]], 1.0)
        self.assertAlmostEqual(staged[hashes[0]], 0.1)

        # Earliest covering read wins.
        _record_staged(staged, hashes, 0.0, 0.5)
        self.assertAlmostEqual(staged[hashes[-1]], 0.5)

    def test_declined_candidate_does_not_make_its_hashes_free(self) -> None:
        # A DECLINE reads nothing, so an identical-prefix follower still owes
        # all its blocks.
        planner = self._planner(max_preload_slots=0)
        prefix = _shared(5)
        head = _candidate("head", position=0, recompute_tokens=5, load_blocks=5,
                           loadable_hashes=prefix)
        follower = _candidate("follower", position=1, recompute_tokens=100,
                               load_blocks=5, loadable_hashes=prefix)

        results = planner.plan([head, follower])

        self.assertEqual(results["head"].decision, LoadDecision.DECLINE)
        self.assertEqual(results["follower"].decision, LoadDecision.ADMIT)
        self.assertEqual(results["follower"].preload_blocks, 0)
        self.assertEqual(len(planner._entries), 0)

    def test_admitted_candidate_does_not_make_its_hashes_free(self) -> None:
        # An ADMIT reads foreground; the preload sharing tier never serves from it.
        planner = self._planner(max_preload_slots=8)
        prefix = _shared(10)
        head = _candidate("head", position=0, recompute_tokens=100, load_blocks=10,
                           loadable_hashes=prefix)
        follower = _candidate("follower", position=1, recompute_tokens=100,
                               load_blocks=10, loadable_hashes=prefix)

        results = planner.plan([head, follower])

        self.assertEqual(results["head"].decision, LoadDecision.ADMIT)
        self.assertEqual(results["follower"].decision, LoadDecision.ADMIT)
        self.assertEqual(len(planner._entries), 0)

    def test_recompute_tokens_use_ceil_block_conversion(self) -> None:
        # recompute_tokens must convert to blocks via ceil, not floor or a
        # raw-token passthrough.

        # Case 1 pins against floor division: floor(5/4)=1 wrongly DECLINEs;
        # only the correct ceil(5/4)=2 DEFERs.
        block_tokens = 4
        tables = _make_tables_with_block_tokens(block_tokens)
        planner = LoadPlanner(
            tables, max_preload_slots=100, defer_tolerance=2.0,
            time_source=lambda: 1000.0,
        )
        c = CandidateCostInput(
            position=0, req_id="r1", recompute_tokens=5, load_blocks=2,
            loadable_hashes=_own("r1", 2),
        )
        result = planner.plan([c])["r1"]
        self.assertEqual(result.decision, LoadDecision.DEFER)

        # Case 2 pins against raw-token passthrough: block_tokens=2,
        # recompute_tokens=3 -> ceil(3/2)=2 (correct) vs raw 3 used directly
        # as a block count (bug), which flips DECLINE to DEFER.
        block_tokens2 = 2
        tables2 = _make_tables_with_block_tokens(block_tokens2)
        planner2 = LoadPlanner(
            tables2, max_preload_slots=100, defer_tolerance=2.0,
            time_source=lambda: 1000.0,
        )
        c2 = CandidateCostInput(
            position=0, req_id="r2", recompute_tokens=3, load_blocks=1,
            loadable_hashes=_own("r2", 1),
        )
        result2 = planner2.plan([c2])["r2"]
        self.assertEqual(result2.decision, LoadDecision.DECLINE)


if __name__ == "__main__":
    unittest.main()
