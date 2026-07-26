from __future__ import annotations

import collections
import enum
import logging
import os
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from concurrent.futures import Future
from concurrent.futures import wait as futures_wait
from dataclasses import dataclass
from typing import Any

from .break_even import load_break_even, load_curves
from .cost_model import build_cost_tables
from .file_mapper import FileMapper
from .fs_config import SharedFileConfig
from .liburing_file import align_up
from .load_planner import CandidateCostInput, LoadDecision, LoadPlanner
from .profiling import add_event, now_ns
from .reactor import LoadDeclined, TransferCoordinator
from .staging import StagingPool
from .transfer import ParsedKvLayout

logger = logging.getLogger(__name__)

try:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.kv_offload.base import (
        CanonicalKVCaches,
        GPULoadStoreSpec,
        LoadStoreSpec,
        OffloadingManager,
        OffloadingSpec,
        OffloadKey,
        PrepareStoreOutput,
        ReqContext,
        get_offload_block_hash,
        get_offload_group_idx,
    )
    from vllm.v1.kv_offload.worker.worker import (
        OffloadingHandler,
        TransferResult,
        TransferSpec,
    )

    VLLM_AVAILABLE = True
except Exception:
    VLLM_AVAILABLE = False
    VllmConfig = Any
    KVCacheConfig = Any
    OffloadKey = Any
    ReqContext = Any
    CanonicalKVCaches = Any
    TransferSpec = tuple[Any, Any]

    class LoadStoreSpec:
        @staticmethod
        def medium() -> str:
            return "NOOP"

    class OffloadingManager:
        pass

    @dataclass(frozen=True)
    class PrepareStoreOutput:
        keys_to_store: list[Any]
        store_spec: LoadStoreSpec
        evicted_keys: list[Any]

    def get_offload_block_hash(key: Any) -> bytes:
        raise RuntimeError("vLLM is required for scheduler offload-key parsing")

    def get_offload_group_idx(key: Any) -> int:
        raise RuntimeError("vLLM is required for scheduler offload-key parsing")

    class GPULoadStoreSpec(LoadStoreSpec):
        @staticmethod
        def medium() -> str:
            return "GPU"

    class OffloadingSpec:
        def __init__(self, vllm_config: Any, kv_cache_config: Any) -> None:
            self.vllm_config = vllm_config
            self.kv_cache_config = kv_cache_config
            self.extra_config = _read_extra_config(vllm_config, kv_cache_config)
            self.gpu_block_size = [int(self.extra_config.get("gpu_block_size", 1))]
            self.block_size_factor = int(self.extra_config.get("block_size_factor", 1))

    class OffloadingHandler:
        pass

    @dataclass(frozen=True)
    class TransferResult:
        job_id: int
        success: bool = True

# Guarded separately: an installed-but-not-yet-patched vLLM (missing these
# planned-defer types) must not flip VLLM_AVAILABLE off.
try:
    from vllm.v1.kv_offload.base import PlanCandidate, PlanDecision, PlanOutcome

    PLAN_API_AVAILABLE = True
except Exception:
    PLAN_API_AVAILABLE = False

    @dataclass(frozen=True)
    class PlanCandidate:
        position: int
        req_id: str
        recompute_tokens: int
        keys: tuple[OffloadKey, ...]

    class PlanDecision(str, enum.Enum):
        ADMIT = "admit"
        DEFER = "defer"
        DECLINE = "decline"

    @dataclass(frozen=True)
    class PlanOutcome:
        decision: PlanDecision
        preload_blocks: int = 0


_DECISION_TO_PLAN = {
    LoadDecision.ADMIT: PlanDecision.ADMIT,
    LoadDecision.DEFER: PlanDecision.DEFER,
    LoadDecision.DECLINE: PlanDecision.DECLINE,
}


def _read_extra_config(vllm_config: Any, kv_cache_config: Any) -> Mapping[str, object]:
    for owner in (kv_cache_config, vllm_config):
        for attr_name in ("kv_connector_extra_config", "extra_config"):
            value = getattr(owner, attr_name, None)
            if isinstance(value, Mapping):
                return value
    kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)
    value = getattr(kv_transfer_config, "kv_connector_extra_config", None)
    if isinstance(value, Mapping):
        return value
    return {}


def _get_nested_attr(root: Any, path: tuple[str, ...], default: Any) -> Any:
    value = root
    for attr_name in path:
        value = getattr(value, attr_name, None)
        if value is None:
            return default
    return value


def _first_int(value: Any, default: int) -> int:
    if isinstance(value, (list, tuple)):
        if not value:
            return default
        value = value[0]
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _parse_non_negative_int(value: Any, *, name: str) -> int:
    # Mirrors vllm's offloading connector scheduler's own parse of this key
    # (absent -> 0) so config validation and the runtime scheduler never disagree.
    if value is None:
        return 0
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a non-negative integer") from exc
    if parsed < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return parsed


class SharedStorageLoadStoreSpec(LoadStoreSpec):
    def __init__(self, block_hashes: Iterable[bytes] = ()):
        self.block_hashes = list(block_hashes)

    @staticmethod
    def medium() -> str:
        return "SHARED_STORAGE"


class SharedStorageOffloadingManager(OffloadingManager):
    def __init__(
        self,
        file_mapper: FileMapper,
        *,
        planner: LoadPlanner | None = None,
    ):
        self.file_mapper = file_mapper
        # Positive hits are immutable; misses not cached (another process can publish between steps).
        self._lookup_cache: dict[bytes, bool] = {}
        # Hashes whose cache entry came from `_probe`, not yet observed by a real
        # `lookup` call; drives the emit-once-per-hash profiling gate below.
        self._probed_hashes: set[bytes] = set()
        self._last_io_req_id: str | None = None
        # hash -> count of in-flight stores; lookup() returns None (defer) so the scheduler
        # waits for the write and loads rather than recomputing a reusable prefix.
        self._store_inflight: dict[bytes, int] = {}
        self._planner = planner

    @staticmethod
    def _get_block_hash(key: OffloadKey) -> bytes:
        group_idx = get_offload_group_idx(key)
        if group_idx != 0:
            raise ValueError(
                "py-kvcache shared-storage offload currently supports exactly one KV cache group"
            )
        return get_offload_block_hash(key)

    @staticmethod
    def _req_id(req_context: ReqContext) -> str | None:
        return getattr(req_context, "req_id", None)

    def _maybe_flush_lookup_cache(self, req_context: ReqContext) -> None:
        req_id = self._req_id(req_context)
        if req_id != self._last_io_req_id:
            self._lookup_cache.clear()
            self._probed_hashes.clear()
            self._last_io_req_id = req_id

    def _probe(self, block_hash: bytes) -> tuple[bool, bool]:
        """(exists, mid_write) via the same `_lookup_cache`/`_store_inflight` state
        `lookup` uses, but with no profiling event and no defer semantics — so the
        planner and `lookup` can never disagree about the filesystem."""
        if block_hash in self._lookup_cache:
            return True, False
        exists = os.path.exists(self.file_mapper.get_file_name(block_hash))
        if exists:
            self._lookup_cache[block_hash] = True
            self._probed_hashes.add(block_hash)
            return True, False
        return False, block_hash in self._store_inflight

    @property
    def supports_planned_defer_preload(self) -> bool:
        return self._planner is not None

    def plan_candidates(
        self,
        candidates: Sequence[PlanCandidate],
        outstanding_load_blocks: Sequence[int] = (),
    ) -> dict[str, PlanOutcome]:
        if self._planner is None:
            return {}

        inputs: list[CandidateCostInput] = []
        for candidate in candidates:
            loadable_hashes: list[bytes] = []
            blocked_on_store = False
            for key in candidate.keys:
                block_hash = self._get_block_hash(key)
                exists, mid_write = self._probe(block_hash)
                if exists:
                    loadable_hashes.append(block_hash)
                    continue
                if mid_write:
                    loadable_hashes.append(block_hash)
                    blocked_on_store = True
                    continue
                break

            inputs.append(
                CandidateCostInput(
                    position=candidate.position,
                    req_id=candidate.req_id,
                    recompute_tokens=candidate.recompute_tokens,
                    load_blocks=len(loadable_hashes),
                    loadable_hashes=tuple(loadable_hashes),
                    blocked_on_store=blocked_on_store,
                    predicted_dram_resident=False,
                )
            )

        results = self._planner.plan(inputs, outstanding_load_blocks=outstanding_load_blocks)
        return {
            req_id: PlanOutcome(
                decision=_DECISION_TO_PLAN[result.decision],
                preload_blocks=result.preload_blocks,
            )
            for req_id, result in results.items()
        }

    def lookup(self, key: OffloadKey, req_context: ReqContext) -> bool | None:
        start_ns = now_ns()
        block_hash = self._get_block_hash(key)
        from_cache = block_hash in self._lookup_cache
        if from_cache:
            cached = True
        else:
            cached = os.path.exists(self.file_mapper.get_file_name(block_hash))
            if cached:
                self._lookup_cache[block_hash] = True
        deferred = not cached and block_hash in self._store_inflight
        was_probed = block_hash in self._probed_hashes
        self._probed_hashes.discard(block_hash)
        # Emit once per hash; cache hits repeat 100k+ times per file otherwise.
        if not from_cache or was_probed:
            add_event(
                "py_kvcache.manager.lookup",
                "kv_offload",
                start_ns,
                now_ns() - start_ns,
                args={"hit": bool(cached), "cached": from_cache, "deferred": deferred},
            )
        if deferred:
            # Correctness fence: mid-write always defers, regardless of any
            # planner decision (stale or otherwise).
            return None
        if self._planner is not None:
            req_id = self._req_id(req_context)
            decision = self._planner.decision_for(req_id) if req_id is not None else None
            if decision == LoadDecision.DECLINE:
                return False
            if decision == LoadDecision.DEFER:
                return None
        return cached

    def prepare_load(self, keys: Iterable[OffloadKey], req_context: ReqContext) -> LoadStoreSpec:
        self._maybe_flush_lookup_cache(req_context)
        block_hashes = [self._get_block_hash(key) for key in keys]
        return SharedStorageLoadStoreSpec(block_hashes)

    def touch(self, keys: Iterable[OffloadKey], req_context: ReqContext) -> None:
        return None

    def complete_load(self, keys: Iterable[OffloadKey], req_context: ReqContext) -> None:
        return None

    def prepare_store(
        self, keys: Iterable[OffloadKey], req_context: ReqContext
    ) -> PrepareStoreOutput | None:
        self._maybe_flush_lookup_cache(req_context)
        start_ns = now_ns()
        keys_to_store: list[OffloadKey] = []
        block_hashes_to_store: list[bytes] = []
        checked = 0

        for key in keys:
            checked += 1
            block_hash = self._get_block_hash(key)
            file_name = self.file_mapper.get_file_name(block_hash)
            exists = os.path.exists(file_name)
            if exists:
                continue
            keys_to_store.append(key)
            block_hashes_to_store.append(block_hash)
            self._store_inflight[block_hash] = self._store_inflight.get(block_hash, 0) + 1

        add_event(
            "py_kvcache.manager.prepare_store",
            "kv_offload",
            start_ns,
            now_ns() - start_ns,
            args={
                "checked_keys": checked,
                "keys_to_store": len(keys_to_store),
            },
        )
        return PrepareStoreOutput(
            keys_to_store=keys_to_store,
            store_spec=SharedStorageLoadStoreSpec(block_hashes_to_store),
            evicted_keys=[],
        )

    def complete_store(
        self,
        keys: Iterable[OffloadKey],
        req_context: ReqContext,
        success: bool = True,
    ) -> None:
        for key in keys:
            block_hash = self._get_block_hash(key)
            count = self._store_inflight.get(block_hash, 0) - 1
            if count > 0:
                self._store_inflight[block_hash] = count
            else:
                self._store_inflight.pop(block_hash, None)
            self._lookup_cache.pop(block_hash, None)
            self._probed_hashes.discard(block_hash)
        return None

    def reset_cache(self) -> None:
        self._lookup_cache.clear()
        self._probed_hashes.clear()
        self._last_io_req_id = None
        self._store_inflight.clear()
        if self._planner is not None:
            self._planner.reset()
        return None


class NoopSharedStorageOffloadingHandler(OffloadingHandler):
    def __init__(
        self,
        *,
        coordinator: TransferCoordinator,
    ) -> None:
        self.coordinator = coordinator
        self._active: dict[int, tuple[Future[int], float, tuple[str, str]]] = {}
        self._finished: dict[int, TransferResult] = {}
        # Break-even declines reported to vLLM for recompute.
        self._declined_block_ids: set[int] = set()
        # Sticky req_ids suppress repeated declined loads.
        self._declined_req_ids: set[str] = set()
        # Bounded FIFO dedup; evicted ids can be re-submitted (reactor deduplicates by hash).
        self._submitted_preload_ids: "collections.OrderedDict[str, None]" = (
            collections.OrderedDict()
        )
        self._max_submitted_preload_ids = 1024
        self.is_shutdown = False

    def preload_async(
        self,
        preload_id: str,
        src_spec: LoadStoreSpec,
        profile_tid: str = "kv_preload",
        req_id: str = "",
    ) -> bool:
        if self.is_shutdown:
            raise RuntimeError("py-kvcache handler is shut down")
        if not isinstance(src_spec, SharedStorageLoadStoreSpec):
            return False
        if preload_id in self._submitted_preload_ids:
            self._submitted_preload_ids.move_to_end(preload_id)
            return False
        self._submitted_preload_ids[preload_id] = None
        if len(self._submitted_preload_ids) > self._max_submitted_preload_ids:
            self._submitted_preload_ids.popitem(last=False)
        self.coordinator.submit_preload(
            list(src_spec.block_hashes),
            preload_id=preload_id,
            req_id=req_id,
            profile_tid=profile_tid,
        )
        return True

    def load_from_preload_async(
        self,
        job_id: int,
        preload_id: str,
        src_spec: LoadStoreSpec,
        dst_spec: LoadStoreSpec,
        profile_tid: str = "kv_transfer",
        req_id: str = "",
    ) -> bool:
        if not isinstance(src_spec, SharedStorageLoadStoreSpec) or not isinstance(
            dst_spec, GPULoadStoreSpec
        ):
            return False
        return self.transfer_async(
            job_id,
            (src_spec, dst_spec),
            profile_tid=profile_tid,
            req_id=req_id,
        )

    def transfer_async(
        self,
        job_id: int,
        spec: TransferSpec,
        profile_tid: str = "kv_transfer",
        req_id: str = "",
    ) -> bool:
        if self.is_shutdown:
            raise RuntimeError("py-kvcache handler is shut down")
        if job_id in self._active or job_id in self._finished:
            raise ValueError(f"job {job_id} is already finished but not collected")

        src_spec, dst_spec = spec
        started = time.perf_counter()
        if isinstance(src_spec, GPULoadStoreSpec) and isinstance(
            dst_spec, SharedStorageLoadStoreSpec
        ):
            future = self.coordinator.submit_store(
                src_spec,
                dst_spec,
                job_id=job_id,
                profile_tid=profile_tid,
                req_id=req_id,
            )
            transfer_type = (
                GPULoadStoreSpec.medium(),
                SharedStorageLoadStoreSpec.medium(),
            )
        elif isinstance(src_spec, SharedStorageLoadStoreSpec) and isinstance(
            dst_spec, GPULoadStoreSpec
        ):
            future = self.coordinator.submit_load(
                src_spec,
                dst_spec,
                job_id=job_id,
                profile_tid=profile_tid,
                req_id=req_id,
            )
            transfer_type = (
                SharedStorageLoadStoreSpec.medium(),
                GPULoadStoreSpec.medium(),
            )
        else:
            raise TypeError(f"unsupported transfer spec: {type(src_spec)!r} -> {type(dst_spec)!r}")

        self._active[job_id] = (future, started, transfer_type)
        return True

    def get_finished(self) -> list[TransferResult]:
        for job_id, (future, started, transfer_type) in list(self._active.items()):
            if not future.done():
                continue
            self._finished[job_id] = self._future_to_transfer_result(
                job_id,
                future,
                started,
                transfer_type,
            )
            self._active.pop(job_id, None)
        results = list(self._finished.values())
        self._finished.clear()
        return results

    def get_declined_block_ids(self) -> set[int]:
        if not self._declined_block_ids:
            return set()
        declined = self._declined_block_ids
        self._declined_block_ids = set()
        return declined

    def get_declined_req_ids(self) -> set[str]:
        if not self._declined_req_ids:
            return set()
        declined = self._declined_req_ids
        self._declined_req_ids = set()
        return declined

    def wait(self, job_ids: set[int]) -> None:
        start_ns = now_ns()
        futures = [
            active[0] for job_id in job_ids if (active := self._active.get(job_id)) is not None
        ]
        if futures:
            futures_wait(futures)
        add_event(
            "py_kvcache.handler.wait",
            "kv_offload",
            start_ns,
            now_ns() - start_ns,
            args={"requested_jobs": len(job_ids), "waited_jobs": len(futures)},
        )

    def shutdown(self) -> None:
        self.is_shutdown = True
        for job_id, (future, started, transfer_type) in list(self._active.items()):
            if future.done():
                self._finished[job_id] = self._future_to_transfer_result(
                    job_id,
                    future,
                    started,
                    transfer_type,
                )
        self._finished.clear()
        self._active.clear()
        self._submitted_preload_ids.clear()
        self.coordinator.shutdown()

    def _future_to_transfer_result(
        self,
        job_id: int,
        future: Future[int],
        started: float,
        transfer_type: tuple[str, str],
    ) -> TransferResult:
        try:
            transfer_size = future.result()
            return _make_transfer_result(
                job_id=job_id,
                success=True,
                transfer_size=transfer_size,
                transfer_time=time.perf_counter() - started,
                transfer_type=transfer_type,
            )
        except LoadDeclined as declined:
            # Declined loads complete with recompute metadata.
            self._declined_block_ids.update(declined.block_ids)
            if declined.req_id:
                self._declined_req_ids.add(declined.req_id)
            return _make_transfer_result(
                job_id=job_id,
                success=True,
                transfer_size=0,
                transfer_time=time.perf_counter() - started,
                transfer_type=transfer_type,
            )
        except BaseException as exc:
            logger.exception("py-kvcache transfer job %s failed", job_id)
            fail_start_ns = now_ns()
            add_event(
                "py_kvcache.handler.transfer_failure",
                "kv_offload",
                fail_start_ns,
                0,
                args={"job_id": job_id, "exception": type(exc).__name__},
            )
            return _make_transfer_result(
                job_id=job_id,
                success=False,
                transfer_size=0,
                transfer_time=time.perf_counter() - started,
                transfer_type=transfer_type,
                message=str(exc),
            )


class PyKvCacheOffloadingSpec(OffloadingSpec):
    def __init__(self, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig):
        super().__init__(vllm_config, kv_cache_config)

        self.shared_file_config = SharedFileConfig.from_extra_config(self.extra_config)
        self._manager: SharedStorageOffloadingManager | None = None
        self._handler: NoopSharedStorageOffloadingHandler | None = None

        parallel_config = getattr(vllm_config, "parallel_config", None)
        model_name = str(_get_nested_attr(vllm_config, ("model_config", "model"), "unknown-model"))
        dtype = str(_get_nested_attr(vllm_config, ("cache_config", "cache_dtype"), "auto")).replace(
            "torch.", ""
        )
        gpu_block_size = _first_int(getattr(self, "gpu_block_size", None), 1)
        block_size_factor = _first_int(getattr(self, "block_size_factor", None), 1)

        self.file_mapper = FileMapper(
            root_dir=self.shared_file_config.root_dir,
            model_name=model_name,
            gpu_block_size=gpu_block_size,
            gpu_blocks_per_file=block_size_factor,
            tp_size=_first_int(getattr(parallel_config, "tensor_parallel_size", None), 1),
            pp_size=_first_int(getattr(parallel_config, "pipeline_parallel_size", None), 1),
            pcp_size=_first_int(getattr(parallel_config, "prefill_context_parallel_size", None), 1),
            rank=_first_int(getattr(parallel_config, "rank", None), 0),
            dtype=dtype,
        )
        os.makedirs(self.file_mapper.base_path, exist_ok=True)

        # Worker load gate; disabled when no file is configured. Not used
        # scheduler-side -- stores always happen.
        self.break_even = load_break_even(
            self.shared_file_config.prefix_cache_break_even_path,
            model_name=model_name,
            kv_dtype=dtype,
        )
        self.storage_block_tokens = gpu_block_size * block_size_factor

        self._planner: LoadPlanner | None = None
        if self.shared_file_config.load_planner == "on":
            self._planner = self._build_planner(vllm_config, model_name=model_name, dtype=dtype)

    def _build_planner(
        self, vllm_config: VllmConfig, *, model_name: str, dtype: str
    ) -> LoadPlanner:
        curves = load_curves(
            self.shared_file_config.prefix_cache_break_even_path,
            model_name=model_name,
            kv_dtype=dtype,
        )
        if curves is None:
            raise ValueError(
                "load_planner=on requires prefix_cache_break_even_path to point at "
                "a v2 break-even file with a 'curves' section; a v1 file (or none) "
                "has no curves to plan with"
            )
        if VLLM_AVAILABLE and not PLAN_API_AVAILABLE:
            raise ValueError(
                "load_planner=on requires a vLLM with the planned-defer preload API "
                "(vllm.v1.kv_offload.base.PlanDecision); the installed vLLM does not "
                "expose it, so plan_candidates would never be called"
            )
        if not self.shared_file_config.enable_preload:
            raise ValueError("load_planner=on requires enable_preload=true")
        if not self.shared_file_config.preload_share_staging:
            raise ValueError(
                "load_planner=on requires preload_share_staging=true; with per-"
                "(req_id, hash) staging the planner's shared-prefix pricing charges "
                "one slot where the reactor holds one per requesting request"
            )

        preload_lookahead_requests = _parse_non_negative_int(
            self.extra_config.get("preload_lookahead_requests"),
            name="preload_lookahead_requests",
        )
        if preload_lookahead_requests <= 0:
            raise ValueError("load_planner=on requires preload_lookahead_requests > 0")

        if bool(_get_nested_attr(vllm_config, ("cache_config", "enable_prefix_caching"), False)):
            raise ValueError(
                "load_planner=on is incompatible with local GPU prefix caching "
                "(vllm_config.cache_config.enable_prefix_caching); the planner prices "
                "recompute_tokens as the 0%-prefix token count"
            )

        if curves.kv_bytes_per_token is None:
            raise ValueError(
                "load_planner=on requires kv_bytes_per_token in the break-even file"
            )

        max_model_len = _first_int(
            _get_nested_attr(vllm_config, ("model_config", "max_model_len"), 0), 0
        )
        if max_model_len <= 0:
            raise ValueError(
                "load_planner=on requires a positive vllm_config.model_config.max_model_len"
            )
        tables = build_cost_tables(
            curves,
            block_tokens=self.storage_block_tokens,
            max_model_len=max_model_len,
        )

        # The scheduler has no staging pool; mirror the reactor's own
        # max_preload_slots derivation so the planner's budget tracks it. A
        # wrong estimate just degrades to a foreground read.
        io_size = align_up(curves.kv_bytes_per_token * self.storage_block_tokens)
        copy_headroom = max(4, self.shared_file_config.iodepth // 2)
        slot_count = StagingPool.compute_slot_count(
            staging_mem_gib=self.shared_file_config.staging_mem,
            io_size=io_size,
            min_slots=self.shared_file_config.iodepth + copy_headroom,
        )
        max_preload_slots = max(0, slot_count - self.shared_file_config.iodepth)

        return LoadPlanner(
            tables,
            max_preload_slots=max_preload_slots,
            defer_tolerance=self.shared_file_config.load_planner_defer_tolerance,
            defer_deadline_max_s=self.shared_file_config.load_planner_defer_deadline_max_s,
        )

    def get_manager(self) -> OffloadingManager:
        if self._manager is None:
            self._manager = SharedStorageOffloadingManager(
                self.file_mapper,
                planner=self._planner,
            )
        return self._manager

    def get_handlers(
        self, kv_caches: CanonicalKVCaches
    ) -> Iterator[tuple[type[LoadStoreSpec], type[LoadStoreSpec], OffloadingHandler]]:
        if self._handler is None:
            gpu_block_size = _first_int(getattr(self, "gpu_block_size", None), 1)
            block_size_factor = _first_int(
                getattr(self, "block_size_factor", None),
                1,
            )
            storage_block_size = gpu_block_size * block_size_factor
            layout = ParsedKvLayout.from_canonical_kv_caches(
                gpu_block_size=gpu_block_size,
                storage_block_size=storage_block_size,
                kv_caches=kv_caches,
            )
            coordinator = TransferCoordinator(
                config=self.shared_file_config,
                file_mapper=self.file_mapper,
                layout=layout,
                break_even=self.break_even,
                storage_block_tokens=storage_block_size,
            )
            self._handler = NoopSharedStorageOffloadingHandler(
                coordinator=coordinator,
            )

        yield GPULoadStoreSpec, SharedStorageLoadStoreSpec, self._handler
        yield SharedStorageLoadStoreSpec, GPULoadStoreSpec, self._handler


PyKvcacheOffloadingSpec = PyKvCacheOffloadingSpec
SharedStorageOffloadingSpec = PyKvCacheOffloadingSpec


def _make_transfer_result(
    *,
    job_id: int,
    success: bool,
    transfer_size: int = 0,
    transfer_time: float = 0.0,
    transfer_type: tuple[str, str] | None = None,
    message: str | None = None,
) -> TransferResult:
    if transfer_type is None:
        transfer_type = (
            SharedStorageLoadStoreSpec.medium(),
            GPULoadStoreSpec.medium(),
        )
    result_kwargs = {
        "job_id": job_id,
        "success": success,
        "transfer_size": transfer_size,
        "transfer_time": transfer_time,
        "transfer_type": transfer_type,
    }
    if message is not None:
        try:
            return TransferResult(**result_kwargs, error_msg=message)
        except TypeError:
            pass
    try:
        return TransferResult(**result_kwargs)
    except TypeError:
        try:
            return TransferResult(job_id=job_id, success=success)
        except TypeError:
            return TransferResult(job_id, success)


__all__ = [
    "NoopSharedStorageOffloadingHandler",
    "PyKvCacheOffloadingSpec",
    "PyKvcacheOffloadingSpec",
    "SharedStorageLoadStoreSpec",
    "SharedStorageOffloadingManager",
    "SharedStorageOffloadingSpec",
    "PLAN_API_AVAILABLE",
    "VLLM_AVAILABLE",
]
