from __future__ import annotations

import enum
import time
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

from .cost_model import CostTables
from .profiling import add_event, now_ns

# Max consecutive plan() calls a staged req_id may be absent before its
# predicted-resident hint is garbage-collected.
_STAGED_ABSENCE_LIMIT = 4

# Clamp on the per-request defer deadline; not user-configurable.
_DEFER_DEADLINE_MIN_S = 0.05
_DEFER_DEADLINE_MAX_S = 2.0


def _distinct_count(hashes: Sequence[bytes], staged_at: Mapping[bytes, float]) -> int:
    return sum(1 for block_hash in hashes if block_hash not in staged_at)


def _record_staged(
    staged_at: dict[bytes, float],
    hashes: Sequence[bytes],
    start_in_s: float,
    end_in_s: float,
) -> None:
    """Note when each hash of a scheduled read is modelled to be delivered.

    Spread linearly across the read (leading blocks land near `start_in_s`)
    instead of stamping every block with the read's completion time, so a
    short request riding the front of a longer prefix waits only for its own
    blocks. Earliest wins across overlapping reads.
    """
    total = len(hashes)
    if total == 0:
        return
    span = max(0.0, end_in_s - start_in_s)
    for index, block_hash in enumerate(hashes):
        ready_in_s = start_in_s + span * (index + 1) / total
        previous = staged_at.get(block_hash)
        if previous is None or ready_in_s < previous:
            staged_at[block_hash] = ready_in_s


def _covered_ready_in_s(
    hashes: Sequence[bytes], staged_at: Mapping[bytes, float]
) -> float:
    """When the last of `hashes` already covered by a scheduled read lands."""
    return max((staged_at[h] for h in hashes if h in staged_at), default=0.0)


class LoadDecision(enum.Enum):
    ADMIT = "admit"
    DEFER = "defer"
    DECLINE = "decline"


@dataclass(frozen=True)
class CandidateCostInput:
    position: int
    req_id: str
    recompute_tokens: int
    load_blocks: int
    loadable_hashes: tuple[bytes, ...]
    blocked_on_store: bool = False
    predicted_dram_resident: bool = False


@dataclass(frozen=True)
class PlanResult:
    decision: LoadDecision
    preload_blocks: int = 0


@dataclass
class _Staging:
    req_id: str
    service_s: float
    starts_at: float
    ends_at: float
    emitted_blocks: int


class LoadPlanner:
    def __init__(
        self,
        tables: CostTables,
        *,
        max_preload_slots: int,
        defer_tolerance: float,
        defer_deadline_max_s: float = _DEFER_DEADLINE_MAX_S,
        # Overridable for tests; not exposed as config.
        defer_deadline_min_s: float = _DEFER_DEADLINE_MIN_S,
        time_source: Callable[[], float] = time.monotonic,
    ) -> None:
        self._tables = tables
        self._max_preload_slots = max_preload_slots
        self._defer_tolerance = defer_tolerance
        self._defer_deadline_min_s = defer_deadline_min_s
        self._defer_deadline_max_s = defer_deadline_max_s
        self._time_source = time_source
        self._entries: deque[_Staging] = deque()
        self._staging_by_req: dict[str, _Staging] = {}
        self._staged: dict[str, int] = {}
        # req_id -> absolute deadline after which a deferred request is
        # admitted for a foreground load instead of waiting on its
        # speculative read. Wall-clock, not a plan()-call count.
        self._deadline_at: dict[str, float] = {}
        self._deadline_absent: dict[str, int] = {}
        self._decisions: dict[str, PlanResult] = {}

    def plan(
        self,
        candidates: Sequence[CandidateCostInput],
        *,
        outstanding_load_blocks: Sequence[int] = (),
    ) -> dict[str, PlanResult]:
        call_start_ns = now_ns()
        now = self._time_source()
        candidate_ids = {c.req_id for c in candidates}

        self._retire_shadow_entries(now)
        self._gc_staged(candidate_ids)
        self._gc_deadlines(candidate_ids)

        storage_wait = max(
            self._shadow_residual(now), self._jobs_floor(outstanding_load_blocks)
        )
        gpu_wait = 0.0
        budget = self._max_preload_slots
        # hash -> seconds from now at which its scheduled read delivers it,
        # so a candidate sharing that hash prices off the read's completion
        # rather than the whole queue's.
        staged_at: dict[bytes, float] = {}
        decisions: dict[str, PlanResult] = {}
        initial_storage_wait = storage_wait

        for c in candidates:
            if c.blocked_on_store:
                # Mid-write fence: never takes a defer-deadline count, but
                # emission is still budgeted.
                outstanding = self._staging_by_req.get(c.req_id)
                if outstanding is not None:
                    emitted_blocks = outstanding.emitted_blocks
                    _record_staged(
                        staged_at,
                        c.loadable_hashes[:emitted_blocks],
                        max(0.0, outstanding.starts_at - now),
                        max(0.0, outstanding.ends_at - now),
                    )
                else:
                    distinct = _distinct_count(c.loadable_hashes, staged_at)
                    if distinct <= budget:
                        budget -= distinct
                        emitted_blocks = c.load_blocks
                        service_s = (
                            0.0 if distinct == 0 else self._tables.service_s_of(distinct)
                        )
                        read_starts_at = storage_wait
                        storage_wait += service_s
                        self._append_shadow_entry(c.req_id, service_s, now, emitted_blocks)
                        _record_staged(
                            staged_at, c.loadable_hashes, read_starts_at, storage_wait
                        )
                    else:
                        emitted_blocks = 0
                decisions[c.req_id] = PlanResult(LoadDecision.DEFER, emitted_blocks)
                add_event(
                    "py_kvcache.planner.defer_mid_write",
                    "kv_planner",
                    now_ns(),
                    0,
                    args={
                        "position": c.position,
                        "reason": "mid_write",
                        "preload_blocks": emitted_blocks,
                    },
                )
                continue

            if c.load_blocks == 0:
                recompute_s = self._recompute_s_for_tokens(c.recompute_tokens)
                decisions[c.req_id] = PlanResult(LoadDecision.DECLINE, 0)
                gpu_wait += recompute_s
                add_event(
                    "py_kvcache.planner.decline",
                    "kv_planner",
                    now_ns(),
                    0,
                    args={
                        "position": c.position,
                        "prefix_tokens": c.recompute_tokens,
                        "t_load_s": None,
                        "t_recompute_s": recompute_s,
                        "reason": "missing",
                    },
                )
                continue

            if c.predicted_dram_resident or c.req_id in self._staged:
                decisions[c.req_id] = PlanResult(LoadDecision.ADMIT, 0)
                self._staged.pop(c.req_id, None)
                add_event(
                    "py_kvcache.planner.admit",
                    "kv_planner",
                    now_ns(),
                    0,
                    args={"position": c.position, "reason": "resident"},
                )
                continue

            outstanding = self._staging_by_req.get(c.req_id)
            if outstanding is not None:
                # Already deferred: re-emit the same prefix rather than
                # re-scoring and double-charging storage for it.
                deadline_at = self._deadline_at.get(c.req_id)
                if deadline_at is not None and now >= deadline_at:
                    # Deadline hit but the blocks exist: admit foreground
                    # instead of discarding a real cache hit.
                    decisions[c.req_id] = PlanResult(LoadDecision.ADMIT, 0)
                    self._staged.pop(c.req_id, None)
                    add_event(
                        "py_kvcache.planner.admit",
                        "kv_planner",
                        now_ns(),
                        0,
                        args={"position": c.position, "reason": "deadline"},
                    )
                    continue
                decisions[c.req_id] = PlanResult(LoadDecision.DEFER, outstanding.emitted_blocks)
                deadline_at = self._arm_deadline(
                    c.req_id, now, max(0.0, outstanding.ends_at - now)
                )
                _record_staged(
                    staged_at,
                    c.loadable_hashes[: outstanding.emitted_blocks],
                    max(0.0, outstanding.starts_at - now),
                    max(0.0, outstanding.ends_at - now),
                )
                add_event(
                    "py_kvcache.planner.defer",
                    "kv_planner",
                    now_ns(),
                    0,
                    args={
                        "position": c.position,
                        "prefix_tokens": outstanding.emitted_blocks * self._tables.block_tokens,
                        "service_s": 0.0,
                        "storage_done_s": storage_wait,
                        "gpu_wait_s": gpu_wait,
                        "deadline_in_s": deadline_at - now,
                    },
                )
                continue

            distinct = _distinct_count(c.loadable_hashes, staged_at)
            # A block another candidate already scheduled this call arrives
            # when that read delivers it, not when the whole backlog drains.
            covered_ready = _covered_ready_in_s(c.loadable_hashes, staged_at)
            if distinct == 0:
                storage_done = covered_ready
            else:
                storage_done = max(
                    covered_ready, storage_wait + self._tables.service_s_of(distinct)
                )
            # g_mem after the max(): DRAM->GPU copies ride async streams and
            # don't serialise with prefill like an SSD read/recompute does.
            t_load = max(storage_done, gpu_wait) + self._tables.load_mem_s(c.load_blocks)
            t_recompute = gpu_wait + self._recompute_s_for_tokens(c.recompute_tokens)

            if t_recompute <= t_load:
                decisions[c.req_id] = PlanResult(LoadDecision.DECLINE, 0)
                gpu_wait += self._recompute_s_for_tokens(c.recompute_tokens)
                add_event(
                    "py_kvcache.planner.decline",
                    "kv_planner",
                    now_ns(),
                    0,
                    args={
                        "position": c.position,
                        "prefix_tokens": c.recompute_tokens,
                        "t_load_s": t_load,
                        "t_recompute_s": t_recompute,
                        "reason": "break_even",
                    },
                )
                continue

            # Only an already-armed deadline is checked; arming here would
            # start the clock for a candidate that ADMITs on budget below.
            deadline_at = self._deadline_at.get(c.req_id)
            if deadline_at is not None and now >= deadline_at:
                decisions[c.req_id] = PlanResult(LoadDecision.ADMIT, 0)
                storage_wait = storage_done
                add_event(
                    "py_kvcache.planner.admit",
                    "kv_planner",
                    now_ns(),
                    0,
                    args={
                        "position": c.position,
                        "reason": "deadline",
                        "t_load_s": t_load,
                        "t_recompute_s": t_recompute,
                    },
                )
                continue

            # All-or-ADMIT: a DEFER is priced and charged for the full
            # distinct count, else it ADMITs foreground instead of staging
            # a partial prefix. distinct == 0 still charges an explicit 0.0
            # (service_s_of(0) is non-zero with a non-zero curve floor).
            if distinct <= budget:
                budget -= distinct
                decisions[c.req_id] = PlanResult(LoadDecision.DEFER, c.load_blocks)
                deadline_at = self._arm_deadline(c.req_id, now, storage_done)
                service_s = 0.0 if distinct == 0 else self._tables.service_s_of(distinct)
                read_starts_at = storage_wait
                storage_wait = storage_wait + service_s
                self._append_shadow_entry(c.req_id, service_s, now, c.load_blocks)
                _record_staged(
                    staged_at,
                    c.loadable_hashes,
                    max(read_starts_at, covered_ready if distinct == 0 else 0.0),
                    max(storage_wait, covered_ready),
                )
                add_event(
                    "py_kvcache.planner.defer",
                    "kv_planner",
                    now_ns(),
                    0,
                    args={
                        "position": c.position,
                        "prefix_tokens": c.load_blocks * self._tables.block_tokens,
                        "service_s": service_s,
                        "storage_done_s": storage_done,
                        "gpu_wait_s": gpu_wait,
                        "deadline_in_s": deadline_at - now,
                    },
                )
            else:
                decisions[c.req_id] = PlanResult(LoadDecision.ADMIT, 0)
                storage_wait = storage_done
                add_event(
                    "py_kvcache.planner.admit",
                    "kv_planner",
                    now_ns(),
                    0,
                    args={"position": c.position, "reason": "budget"},
                )

        self._decisions = decisions
        add_event(
            "py_kvcache.planner.plan",
            "kv_planner",
            call_start_ns,
            now_ns() - call_start_ns,
            args={
                "candidates": len(candidates),
                "storage_wait_s": initial_storage_wait,
                "gpu_wait_s": gpu_wait,
            },
        )
        return decisions

    def decision_for(self, req_id: str) -> LoadDecision | None:
        result = self._decisions.get(req_id)
        return result.decision if result is not None else None

    def reset(self) -> None:
        self._entries.clear()
        self._staging_by_req.clear()
        self._staged.clear()
        self._deadline_at.clear()
        self._deadline_absent.clear()
        self._decisions.clear()

    def _recompute_s_for_tokens(self, tokens: int) -> float:
        blocks = -(-tokens // self._tables.block_tokens)
        return self._tables.recompute_s(blocks)

    def _retire_shadow_entries(self, now: float) -> None:
        while self._entries and self._entries[0].ends_at <= now:
            entry = self._entries.popleft()
            self._staging_by_req.pop(entry.req_id, None)
            self._staged[entry.req_id] = 0
            add_event(
                "py_kvcache.planner.shadow_retire",
                "kv_planner",
                now_ns(),
                0,
                args={"modelled_s": entry.service_s},
            )

    def _shadow_residual(self, now: float) -> float:
        if not self._entries:
            return 0.0
        # Last entry's completion time already includes every entry ahead of
        # it; summing per-entry residuals would double-count the queue.
        return max(0.0, self._entries[-1].ends_at - now)

    def _jobs_floor(self, outstanding_load_blocks: Sequence[int]) -> float:
        return sum(self._tables.service_s_of(n) for n in outstanding_load_blocks)

    def _append_shadow_entry(
        self, req_id: str, service_s: float, now: float, emitted_blocks: int
    ) -> None:
        starts_at = max(now, self._entries[-1].ends_at) if self._entries else now
        ends_at = starts_at + service_s
        entry = _Staging(
            req_id=req_id,
            service_s=service_s,
            starts_at=starts_at,
            ends_at=ends_at,
            emitted_blocks=emitted_blocks,
        )
        self._entries.append(entry)
        self._staging_by_req[req_id] = entry

    def _gc_staged(self, candidate_ids: set[str]) -> None:
        for req_id in list(self._staged):
            if req_id in candidate_ids:
                self._staged[req_id] = 0
                continue
            absent = self._staged[req_id] + 1
            if absent > _STAGED_ABSENCE_LIMIT:
                del self._staged[req_id]
            else:
                self._staged[req_id] = absent

    def _arm_deadline(self, req_id: str, now: float, predicted_wait_s: float) -> float:
        """Absolute time this request stops waiting for its speculative read.

        Idempotent: only the first defer of a request arms it.
        """
        existing = self._deadline_at.get(req_id)
        if existing is not None:
            return existing
        grace = min(
            self._defer_deadline_max_s,
            max(self._defer_deadline_min_s, self._defer_tolerance * predicted_wait_s),
        )
        deadline_at = now + grace
        self._deadline_at[req_id] = deadline_at
        return deadline_at

    def _gc_deadlines(self, candidate_ids: set[str]) -> None:
        # Tolerate a brief absence (e.g. scheduler flicker) before dropping,
        # same as _staged.
        for req_id in list(self._deadline_at):
            if req_id in candidate_ids:
                self._deadline_absent.pop(req_id, None)
                continue
            absent = self._deadline_absent.get(req_id, 0) + 1
            if absent > _STAGED_ABSENCE_LIMIT:
                del self._deadline_at[req_id]
                self._deadline_absent.pop(req_id, None)
            else:
                self._deadline_absent[req_id] = absent

__all__ = ["CandidateCostInput", "LoadDecision", "LoadPlanner", "PlanResult"]
