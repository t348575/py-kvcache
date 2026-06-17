from __future__ import annotations

import collections
import logging
import os
import queue
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .break_even import BreakEvenThresholds, should_load
from .file_mapper import FileMapper
from .fs_config import SharedFileConfig
from .liburing_file import (
    DEFAULT_IODEPTH,
    DirectIoFileStore,
    LiburingRing,
    align_up,
)
from .profiling import add_event, now_ns
from .staging import StagingPool
from .staging_cache import StagingDataCache, _CacheSlot
from .transfer import (
    ParsedKvLayout,
    _TransferProfile,
    block_ids_of,
    build_block_mapping,
    emit_transfer_events,
    first_group_block_index,
    safe_num_blocks,
    split_block_ids_for_files,
)

try:
    import torch
    from vllm import _custom_ops as ops

    TORCH_COPY_AVAILABLE = True
except Exception:
    torch = Any
    ops = None
    TORCH_COPY_AVAILABLE = False

logger = logging.getLogger(__name__)


class LoadDeclined(Exception):
    """Load skipped by break-even; block ids must be recomputed."""

    def __init__(self, block_ids: list[int], req_id: str = "") -> None:
        super().__init__("load declined by prefix-cache break-even gate")
        self.block_ids = block_ids
        self.req_id = req_id


@dataclass
class _ReactorJob:
    job_id: int
    is_store: bool
    block_hashes: list[bytes]
    # Per-file GPU block ids: source blocks for store, destination blocks for load.
    block_chunks: list[np.ndarray]
    profile: _TransferProfile
    future: "Future[int]"
    total_files: int
    transfer_size: int
    num_blocks: int
    next_file_index: int = 0
    inflight_files: int = 0
    done_files: int = 0
    failed: BaseException | None = None
    future_set: bool = False
    file_samples: list = field(default_factory=list)
    cuda_samples: list = field(default_factory=list)
    n_from_file: int = 0
    n_from_preload: int = 0
    n_from_cache: int = 0
    # Store-only: compute-stream event the store copy waits on before reading GPU KV.
    compute_event: Any = None


@dataclass
class _PreloadRequest:
    block_hashes: list[bytes]
    preload_id: str = ""
    req_id: str = ""
    profile_tid: str = "kv_preload"


@dataclass(frozen=True)
class _PreloadInfo:
    block_hash: bytes
    preload_id: str
    req_id: str
    profile_tid: str
    file_index: int
    total_files: int


@dataclass
class _SharedPreloadSlot:
    # Released only when not cached and no copy still reads from it.
    block_hash: bytes
    slot_index: int
    preload_info: "_PreloadInfo | None"
    copies_inflight: int = 0
    cached: bool = True


@dataclass
class _RingOp:
    job: _ReactorJob | None  # None for preload reads
    file_index: int
    slot_index: int
    fd: int
    is_write: bool
    start_ns: int
    op_kind: str = "read"  # "open" | "read" | "write" | "close"
    final_path: str | None = None
    temp_path: str | None = None
    preload_hash: bytes | None = None
    preload_info: _PreloadInfo | None = None
    path_buf: Any = None  # keeps openat path alive until open CQE is reaped


@dataclass
class _ReadyFd:
    fd: int
    job: _ReactorJob | None
    file_index: int
    preload_hash: bytes | None
    open_start_ns: int
    preload_info: _PreloadInfo | None = None


@dataclass
class _CopyOp:
    job: _ReactorJob | None
    file_index: int
    slot_index: int
    is_store: bool
    start_ns: int
    start_event: Any
    end_event: Any
    nbytes: int
    members: list[tuple[int, "_ReactorJob", int, "_SharedPreloadSlot | None"]] | None = None


@dataclass
class _ReadyCopy:
    job: _ReactorJob
    file_index: int
    slot_index: int
    mapping_len: int
    nbytes: int
    # Per-copy snapshot so shared siblings reusing one slot_index don't clobber it.
    mapping: Any = None
    shared: "_SharedPreloadSlot | None" = None
    cache: "_CacheSlot | None" = None


class IoReactor:
    _STOP = object()

    def __init__(
        self,
        *,
        config: SharedFileConfig,
        file_mapper: FileMapper,
        layout: ParsedKvLayout,
        break_even: BreakEvenThresholds | None = None,
        storage_block_tokens: int = 0,
    ) -> None:
        if not TORCH_COPY_AVAILABLE:
            raise RuntimeError("vLLM and torch are required for the I/O reactor")
        self.config = config
        self.file_mapper = file_mapper
        self.layout = layout
        # Load-side break-even gate: decline sub-threshold loads to GPU recompute.
        # Disabled (no gating) when thresholds are off or block-token size unknown.
        self._break_even = break_even or BreakEvenThresholds()
        self._storage_block_tokens = storage_block_tokens
        payload_size = layout.storage_block_bytes
        io_size = align_up(payload_size)
        self.file_store = DirectIoFileStore(config, payload_size=payload_size)
        self.iodepth = max(1, config.iodepth or DEFAULT_IODEPTH)
        # Floor above iodepth so slots held by CUDA copies don't starve new reads.
        self._copy_headroom = max(4, self.iodepth // 2)
        min_slots = self.iodepth + self._copy_headroom
        slot_count = StagingPool.compute_slot_count(
            staging_mem_gib=config.staging_mem,
            io_size=io_size,
            min_slots=min_slots,
        )
        self.staging_buffer = layout.allocate_staging_buffer(slot_count)
        self._staging_base_ptr = self.staging_buffer.data_ptr()
        self.staging_pool = StagingPool(slot_count=slot_count)
        if slot_count <= min_slots:
            logger.warning(
                "staging slots floored to %d (iodepth=%d + copy_headroom=%d); "
                "raise staging_mem so reads do not stall on copy-held slots",
                slot_count,
                self.iodepth,
                self._copy_headroom,
            )
        self.open_lookahead = max(1, config.open_lookahead or self.iodepth)
        ring_depth = max(self.iodepth + self.open_lookahead + 16, 16)
        self.ring = LiburingRing(ring_depth, ring_id=0)

        self._streams = [torch.cuda.Stream() for _ in range(slot_count)]
        self._copy_stream = torch.cuda.Stream()
        max_mapping_entries = max(1, layout.storage_block_size_factor)
        self._mapping_buffers = [
            np.empty((max_mapping_entries, 2), dtype=np.int64) for _ in range(slot_count)
        ]
        self._mapping_tensors = [torch.from_numpy(buffer) for buffer in self._mapping_buffers]
        self._slot_block_ids = [
            np.asarray([slot_index], dtype=np.int64) for slot_index in range(slot_count)
        ]
        max_copy_entries = max(1, len(layout.gpu_tensors) * max_mapping_entries)
        self._batch_src_ptrs = [
            np.empty(max_copy_entries, dtype=np.int64) for _ in range(slot_count)
        ]
        self._batch_dst_ptrs = [
            np.empty(max_copy_entries, dtype=np.int64) for _ in range(slot_count)
        ]
        self._batch_sizes = [np.empty(max_copy_entries, dtype=np.int64) for _ in range(slot_count)]
        self._batch_src_tensors = [torch.from_numpy(buffer) for buffer in self._batch_src_ptrs]
        self._batch_dst_tensors = [torch.from_numpy(buffer) for buffer in self._batch_dst_ptrs]
        self._batch_size_tensors = [torch.from_numpy(buffer) for buffer in self._batch_sizes]

        self._incoming: "queue.Queue[_ReactorJob | object]" = queue.Queue()
        self._active: list[_ReactorJob] = []
        self._inflight: dict[int, _RingOp] = {}
        self._pending_copies: list[_CopyOp] = []
        self._copy_ready: list[_ReadyCopy] = []
        self._fused_capacity = 0
        self._fused_src_np: np.ndarray | None = None
        self._fused_dst_np: np.ndarray | None = None
        self._fused_sizes_np: np.ndarray | None = None
        self._fused_src_t: Any = None
        self._fused_dst_t: Any = None
        self._fused_sizes_t: Any = None
        self._next_user_data = 0
        self._stop = False
        self._closed = False
        self._submit_lock = threading.Lock()

        self._data_inflight = 0
        self._open_inflight = 0
        self._ready_fds_load: collections.deque[_ReadyFd] = collections.deque()
        self._ready_fds_preload: collections.deque[_ReadyFd] = collections.deque()

        # Shared preload: one read + one slot per hash, released by refcount.
        self._share_preload = bool(getattr(config, "preload_share_staging", False))
        # hash -> count of distinct outstanding requests demanding it.
        self._preload_refcount: dict[bytes, int] = {}
        self._shared_cached: dict[bytes, _SharedPreloadSlot] = {}

        # Multi-copy preload: each requesting req_id gets its own staged copy of a
        # block hash (dedup per (req_id, hash) via _preload_owned).
        self._preload_pending: "collections.deque[_PreloadInfo]" = collections.deque()
        # Live and cancel counts per hash for pending copies.
        self._preload_pending_count: dict[bytes, int] = {}
        self._preload_pending_cancel: dict[bytes, int] = {}
        # hash -> count of open/read ops in flight.
        self._preload_inflight_hashes: dict[bytes, int] = {}
        # hash -> deque of (slot_index, _PreloadInfo); FIFO so oldest is evicted first.
        self._preload_slots: "collections.OrderedDict[bytes, collections.deque[tuple[int, _PreloadInfo]]]" = collections.OrderedDict()
        self._preload_cached_total = 0
        self._preload_inflight_total = 0
        # FIFO-capped per-(req_id, hash) dedup set.
        self._preload_owned: "collections.OrderedDict[tuple[str, bytes], None]" = (
            collections.OrderedDict()
        )
        self._max_preload_owned = 16384
        # Reserve iodepth slots for foreground loads.
        self._max_preload_slots = max(0, self.staging_pool.slot_count - self.iodepth)
        # hash -> deque of (job, file_index); waiter count <= inflight count for hash.
        self._preload_waiters: dict[bytes, collections.deque[tuple[_ReactorJob, int]]] = {}
        # hash -> count of in-flight stores; preload defers until write completes.
        self._store_inflight: dict[bytes, int] = {}
        # hash -> deque of _PreloadInfo parked on an in-flight store.
        self._preload_blocked_on_write: dict[bytes, collections.deque[_PreloadInfo]] = {}
        # hash -> deque of _PreloadInfo for openat misses; re-armed on store completion.
        self._known_missing: "collections.OrderedDict[bytes, collections.deque[_PreloadInfo]]" = (
            collections.OrderedDict()
        )
        self._max_known_missing = 4096

        cache_kind = str(getattr(config, "staging_cache", "off") or "off").lower()
        self._max_cache_slots = max(0, self.staging_pool.slot_count - self.iodepth)
        self._staging_cache: StagingDataCache | None = (
            StagingDataCache(policy=cache_kind, capacity=self._max_cache_slots)
            if cache_kind != "off" and self._max_cache_slots > 0
            else None
        )

        self._worker = threading.Thread(target=self._run, name="py-kvcache-reactor", daemon=True)
        self._worker.start()
        logger.info(
            "IoReactor started: iodepth=%d staging_slots=%d payload_size=%d (direct-staging)",
            self.iodepth,
            slot_count,
            payload_size,
        )

    def submit_job(self, job: _ReactorJob) -> None:
        with self._submit_lock:
            if self._closed:
                if not job.future_set:
                    job.future_set = True
                    job.future.set_exception(RuntimeError("reactor is shut down"))
                return
            self._incoming.put(job)

    def enqueue_preload(
        self,
        block_hashes: list[bytes],
        *,
        preload_id: str = "",
        req_id: str = "",
        profile_tid: str = "kv_preload",
    ) -> None:
        with self._submit_lock:
            if self._closed:
                return
            self._incoming.put(
                _PreloadRequest(
                    block_hashes,
                    preload_id=preload_id,
                    req_id=req_id,
                    profile_tid=profile_tid,
                )
            )

    def shutdown(self, wait: bool = True) -> None:
        with self._submit_lock:
            if self._closed:
                return
            self._closed = True
            self._incoming.put(self._STOP)
        if wait:
            self._worker.join()
            self.staging_pool.close()
            self.ring.close()

    def _preload_cache_push(
        self, block_hash: bytes, slot_index: int, info: "_PreloadInfo | None"
    ) -> None:
        slots = self._preload_slots.get(block_hash)
        if slots is None:
            slots = collections.deque()
            self._preload_slots[block_hash] = slots
        slots.append((slot_index, info))
        self._preload_cached_total += 1

    def _preload_cache_pop(self, block_hash: bytes) -> "tuple[int, _PreloadInfo | None] | None":
        slots = self._preload_slots.get(block_hash)
        if not slots:
            return None
        entry = slots.popleft()
        self._preload_cached_total -= 1
        if not slots:
            del self._preload_slots[block_hash]
        return entry

    def _has_cached_preload(self, block_hash: bytes) -> bool:
        return bool(self._preload_slots.get(block_hash))

    def _preload_inflight_add(self, block_hash: bytes) -> None:
        self._preload_inflight_hashes[block_hash] = (
            self._preload_inflight_hashes.get(block_hash, 0) + 1
        )
        self._preload_inflight_total += 1

    def _preload_inflight_remove(self, block_hash: bytes) -> None:
        count = self._preload_inflight_hashes.get(block_hash, 0)
        if count <= 0:
            return
        self._preload_inflight_total -= 1
        if count == 1:
            del self._preload_inflight_hashes[block_hash]
        else:
            self._preload_inflight_hashes[block_hash] = count - 1

    def _has_inflight_preload(self, block_hash: bytes) -> bool:
        return self._preload_inflight_hashes.get(block_hash, 0) > 0

    def _preload_waiter_add(self, block_hash: bytes, job: "_ReactorJob", file_index: int) -> None:
        waiters = self._preload_waiters.get(block_hash)
        if waiters is None:
            waiters = collections.deque()
            self._preload_waiters[block_hash] = waiters
        waiters.append((job, file_index))

    def _preload_waiter_pop(self, block_hash: bytes):
        waiters = self._preload_waiters.get(block_hash)
        if not waiters:
            return None
        item = waiters.popleft()
        if not waiters:
            del self._preload_waiters[block_hash]
        return item

    def _has_preload_waiter(self, block_hash: bytes) -> bool:
        return bool(self._preload_waiters.get(block_hash))

    def _preload_waiter_room(self, block_hash: bytes) -> bool:
        # Waiter count must not exceed inflight count: every waiter is served by some CQE.
        return len(self._preload_waiters.get(block_hash, ())) < (
            self._preload_inflight_hashes.get(block_hash, 0)
        )

    def _preload_own(self, req_id: str, block_hash: bytes) -> bool:
        key = (req_id, block_hash)
        if key in self._preload_owned:
            return False
        self._preload_owned[key] = None
        while len(self._preload_owned) > self._max_preload_owned:
            self._preload_owned.popitem(last=False)
        return True

    def _pending_push(self, info: "_PreloadInfo") -> None:
        self._preload_pending.append(info)
        self._preload_pending_count[info.block_hash] = (
            self._preload_pending_count.get(info.block_hash, 0) + 1
        )

    def _pending_dec(self, block_hash: bytes) -> None:
        count = self._preload_pending_count.get(block_hash, 0)
        if count <= 1:
            self._preload_pending_count.pop(block_hash, None)
        else:
            self._preload_pending_count[block_hash] = count - 1

    def _cancel_one_pending(self, block_hash: bytes) -> bool:
        # The load now reads this hash itself; the pending copy is redundant.
        live = self._preload_pending_count.get(block_hash, 0) - self._preload_pending_cancel.get(
            block_hash, 0
        )
        if live <= 0:
            return False
        self._preload_pending_cancel[block_hash] = (
            self._preload_pending_cancel.get(block_hash, 0) + 1
        )
        return True

    # --- Shared (ref-counted) preload helpers -------------------------------

    def _has_any_shared_state(self, block_hash: bytes) -> bool:
        return (
            block_hash in self._shared_cached
            or self._preload_inflight_hashes.get(block_hash, 0) > 0
            or self._preload_pending_count.get(block_hash, 0) > 0
            or block_hash in self._preload_blocked_on_write
            or block_hash in self._known_missing
        )

    def _preload_has_copy(self, block_hash: bytes) -> bool:
        if self._staging_cache is not None and block_hash in self._staging_cache:
            return True
        if self._has_inflight_preload(block_hash):
            return True
        if self._share_preload:
            return block_hash in self._shared_cached
        return self._has_cached_preload(block_hash)

    def _make_preload_info_for_job(self, job: "_ReactorJob", file_index: int) -> "_PreloadInfo":
        return _PreloadInfo(
            block_hash=job.block_hashes[file_index],
            preload_id="",
            req_id=job.profile.req_id,
            profile_tid="kv_preload",
            file_index=file_index,
            total_files=job.total_files,
        )

    def _shared_uncache(self, slot: "_SharedPreloadSlot") -> None:
        if self._shared_cached.get(slot.block_hash) is slot:
            del self._shared_cached[slot.block_hash]
            self._preload_cached_total -= 1
        slot.cached = False

    def _maybe_release_shared(self, slot: "_SharedPreloadSlot") -> None:
        if slot.cached or slot.copies_inflight > 0:
            return
        self._release_or_cache(slot.block_hash, slot.slot_index)

    def _purge_owned_for_hash(self, block_hash: bytes) -> None:
        stale = [k for k in self._preload_owned if k[1] == block_hash]
        for k in stale:
            del self._preload_owned[k]

    def _shared_decref(self, req_id: str, block_hash: bytes) -> None:
        # Only an owner's claim consumes demand; over-counts fall to eviction.
        key = (req_id, block_hash)
        if key not in self._preload_owned:
            return
        del self._preload_owned[key]
        count = self._preload_refcount.get(block_hash, 0)
        if count > 1:
            self._preload_refcount[block_hash] = count - 1
            return
        self._preload_refcount.pop(block_hash, None)
        slot = self._shared_cached.get(block_hash)
        if slot is not None:
            self._shared_uncache(slot)
            self._maybe_release_shared(slot)

    def _fail_all_preload_waiters(self, block_hash: bytes, exc: BaseException) -> bool:
        failed = False
        while True:
            waiter = self._preload_waiter_pop(block_hash)
            if waiter is None:
                break
            self._file_terminal(waiter[0], ok=False, exc=exc)
            failed = True
        return failed

    def _evict_one_shared_slot(self) -> bool:
        for block_hash, slot in list(self._shared_cached.items()):
            if slot.copies_inflight > 0:
                continue  # in use, not actually free
            had_ref = self._preload_refcount.get(block_hash, 0) > 0
            self._shared_uncache(slot)
            self._preload_refcount.pop(block_hash, None)
            self._purge_owned_for_hash(block_hash)
            add_event(
                "py_kvcache.preload.evict_slot",
                "kv_preload",
                now_ns(),
                0,
                tid=slot.preload_info.profile_tid if slot.preload_info else None,
                args=self._preload_event_args(
                    block_hash,
                    slot.preload_info,
                    reason="refcount_leak_reclaim" if had_ref else "load_slot_pressure",
                ),
            )
            self.staging_pool.release(slot.slot_index)
            return True
        return False

    def _has_work(self) -> bool:
        return bool(
            self._active
            or self._inflight
            or self._pending_copies
            or self._copy_ready
            or self._preload_pending
            or self._preload_slots
            or self._shared_cached
            or self._ready_fds_load
            or self._ready_fds_preload
        )

    def _run(self) -> None:
        try:
            while not (self._stop and not self._has_work()):
                self._drain_incoming(block=not self._has_work())
                if self._stop and not self._has_work():
                    break
                self._pump_once()
        except BaseException as exc:
            # The reactor owns every outstanding transfer Future. If this thread
            # dies, vLLM blocks forever (loads never free GPU blocks, stores never
            # signal complete_store -> requests wedge as Running:0 / Deferred:N).
            # Surface the traceback and fail all outstanding work so the engine
            # gets errors (recompute/retry) instead of a silent deadlock.
            logger.exception("py-kvcache reactor thread crashed; failing all outstanding transfers")
            self._fail_everything(exc)

    def _fail_everything(self, exc: BaseException) -> None:
        self._stop = True
        wrapped = RuntimeError("py-kvcache reactor thread crashed")
        wrapped.__cause__ = exc
        for job in self._active:
            if not job.future_set:
                job.future_set = True
                try:
                    job.future.set_exception(wrapped)
                except BaseException:
                    pass
        self._active = []
        for waiters in list(self._preload_waiters.values()):
            for job, _fi in waiters:
                if not job.future_set:
                    job.future_set = True
                    try:
                        job.future.set_exception(wrapped)
                    except BaseException:
                        pass
        self._preload_waiters.clear()
        # Drain any queued work so submit_job after this fails fast too.
        with self._submit_lock:
            self._closed = True
            while True:
                try:
                    item = self._incoming.get_nowait()
                except queue.Empty:
                    break
                if isinstance(item, _ReactorJob) and not item.future_set:
                    item.future_set = True
                    try:
                        item.future.set_exception(wrapped)
                    except BaseException:
                        pass

    def _drain_incoming(self, *, block: bool) -> None:
        if block:
            try:
                item = self._incoming.get(timeout=0.5)
            except queue.Empty:
                return
            self._intake(item)
        while True:
            try:
                item = self._incoming.get_nowait()
            except queue.Empty:
                return
            self._intake(item)

    def _intake(self, item: Any) -> None:
        if item is self._STOP:
            self._stop = True
            unclaimed = self._preload_cached_total
            if unclaimed:
                add_event(
                    "py_kvcache.preload.unclaimed_at_shutdown",
                    "kv_preload",
                    now_ns(),
                    0,
                    args={"count": unclaimed},
                )
            for slots in self._preload_slots.values():
                for slot_index, _info in slots:
                    self.staging_pool.release(slot_index)
            self._preload_slots.clear()
            for slot in self._shared_cached.values():
                self.staging_pool.release(slot.slot_index)
            self._shared_cached.clear()
            if self._staging_cache is not None:
                retained = len(self._staging_cache)
                for slot_index in self._staging_cache.drain():
                    self.staging_pool.release(slot_index)
                if retained:
                    add_event(
                        "py_kvcache.staging_cache.retained_at_shutdown",
                        "kv_staging_cache",
                        now_ns(),
                        0,
                        args={"count": retained},
                    )
            self._preload_refcount.clear()
            self._preload_cached_total = 0
            self._preload_inflight_total = 0
            self._preload_pending.clear()
            self._preload_pending_count.clear()
            self._preload_pending_cancel.clear()
            self._preload_owned.clear()
            self._preload_blocked_on_write.clear()
            self._known_missing.clear()
            err = RuntimeError("reactor shutting down")
            for _h, waiters in list(self._preload_waiters.items()):
                for job, _fi in waiters:
                    self._file_terminal(job, ok=False, exc=err)
            self._preload_waiters.clear()
            return
        if isinstance(item, _PreloadRequest):
            total_files = len(item.block_hashes)
            for file_index, h in enumerate(item.block_hashes):
                if self._staging_cache is not None and h in self._staging_cache:
                    continue  # already resident; the real load will hit the cache
                if not self._preload_own(item.req_id, h):
                    continue
                info = _PreloadInfo(
                    block_hash=h,
                    preload_id=item.preload_id,
                    req_id=item.req_id,
                    profile_tid=item.profile_tid,
                    file_index=file_index,
                    total_files=total_files,
                )
                if self._share_preload:
                    # Demand counts; a read is queued only for the first demander.
                    self._preload_refcount[h] = self._preload_refcount.get(h, 0) + 1
                    if not self._has_any_shared_state(h):
                        self._pending_push(info)
                    continue
                known = self._known_missing.get(h)
                if known is not None:
                    # Hash already missed; join the deque so store completion re-arms this copy too.
                    known.append(info)
                    self._known_missing.move_to_end(h)
                else:
                    self._pending_push(info)
            return
        job = item
        if not job.is_store:
            if self._break_even.enabled and self._should_decline_load(job):
                self._decline_load(job)
                return
            # Clear stale known-missing marks; in-flight/cached copies stay for their owners.
            for h in job.block_hashes:
                self._known_missing.pop(h, None)
        else:
            producer = job.profile.req_id
            for h in job.block_hashes:
                # A request never loads a prefix it produced, so withdraw its own
                # speculative demand; else a FRESH unique-prefix store would re-arm
                # a pointless read-back of its just-written bytes.
                if self._share_preload:
                    self._shared_decref(producer, h)
                missed = self._known_missing.pop(h, None)
                if missed:
                    if self._share_preload:
                        rearm = missed if self._preload_refcount.get(h, 0) > 0 else None
                    else:
                        rearm = collections.deque(
                            info for info in missed if info.req_id != producer
                        )
                        rearm = rearm or None
                    if rearm:
                        blocked = self._preload_blocked_on_write.get(h)
                        if blocked is None:
                            self._preload_blocked_on_write[h] = rearm
                        else:
                            blocked.extend(rearm)
                self._store_inflight[h] = self._store_inflight.get(h, 0) + 1
        self._active.append(job)

    def _should_decline_load(self, job: "_ReactorJob") -> bool:
        # Medium is RAM only if every block is already cheaply available (cached
        # or an in-flight/cached preload copy) -> no fresh disk read on this load.
        # Otherwise a fresh SSD read is required -> SSD threshold applies.
        prefix_tokens = job.total_files * self._storage_block_tokens
        if prefix_tokens <= 0:
            return False
        ram_resident = all(self._preload_has_copy(h) for h in job.block_hashes)
        return not should_load(prefix_tokens, self._break_even, ram_resident=ram_resident)

    def _decline_load(self, job: "_ReactorJob") -> None:
        block_ids = [
            int(b)
            for chunk in job.block_chunks
            for b in np.asarray(chunk).reshape(-1)
        ]
        add_event(
            "py_kvcache.break_even.decline_load",
            "kv_offload",
            now_ns(),
            0,
            tid=job.profile.profile_tid,
            args={
                "req_id": job.profile.req_id,
                "job_id": job.job_id,
                "prefix_tokens": job.total_files * self._storage_block_tokens,
                "num_blocks": len(block_ids),
            },
        )
        job.future_set = True
        job.future.set_exception(LoadDeclined(block_ids, req_id=job.profile.req_id))

    def _pump_once(self) -> bool:
        made = False
        made = self._drain_cuda_copies() or made
        made = self._poll_ring_completions() or made
        made = self._schedule_work() or made
        # Fuse all load copies queued this pump into one swap_blocks_batch.
        made = self._flush_copy_batch() or made
        self.ring.submit_pending()
        made = self._finish_jobs() or made
        if not made and self._has_work():
            time.sleep(0)
        return made

    @staticmethod
    def _preload_event_args(
        block_hash: bytes,
        info: _PreloadInfo | None,
        **extra: Any,
    ) -> dict[str, Any]:
        args: dict[str, Any] = {
            "block_hash_prefix": block_hash.hex()[:16],
        }
        if info is not None:
            args.update(
                {
                    "req_id": info.req_id,
                    "preload_id": info.preload_id,
                    "preload_file_index": info.file_index,
                    "preload_num_files": info.total_files,
                }
            )
        args.update(extra)
        return args

    def _drain_cuda_copies(self) -> bool:
        if not self._pending_copies:
            return False
        made = False
        for copy in list(self._pending_copies):
            if not copy.end_event.query():
                continue
            self._pending_copies.remove(copy)
            duration_ns = int(copy.start_event.elapsed_time(copy.end_event) * 1_000_000)
            if copy.is_store:
                assert copy.job is not None
                copy.job.cuda_samples.append((copy.start_ns, duration_ns, copy.nbytes, 0))
                self._on_store_copy_done(copy)
            else:
                assert copy.members is not None
                per_job: dict[int, list[Any]] = {}
                order: list[int] = []
                for _slot_index, job, _file_index, _shared, _cache in copy.members:
                    rec = per_job.get(id(job))
                    if rec is None:
                        per_job[id(job)] = [job, 1]
                        order.append(id(job))
                    else:
                        rec[1] += 1
                for key in order:
                    job, count = per_job[key]
                    job.cuda_samples.append(
                        (
                            copy.start_ns,
                            duration_ns,
                            count * self.layout.storage_block_bytes,
                            0,
                        )
                    )
                for slot_index, job, file_index, shared, cache in copy.members:
                    self._settle_load_slot(job, file_index, slot_index, shared, cache)
                    self._file_terminal(job, ok=True)
            made = True
        return made

    def _on_store_copy_done(self, copy: _CopyOp) -> None:
        job = copy.job
        io_view = self._slot_view(copy.slot_index)
        final_path = self.file_mapper.get_file_name(job.block_hashes[copy.file_index])
        try:
            fd, temp_path = self.file_store.open_temp_write(final_path)
        except BaseException as exc:
            self._data_inflight -= 1
            self.staging_pool.release(copy.slot_index)
            self._file_terminal(job, ok=False, exc=exc)
            self._release_store_inflight(job.block_hashes[copy.file_index], ok=False)
            return
        user_data = self._new_user_data()
        start_ns = now_ns()
        try:
            self.file_store.queue_write(
                self.ring,
                user_data=user_data,
                fd=fd,
                io_array=io_view,
            )
        except BaseException as exc:
            self._data_inflight -= 1
            self.file_store.cleanup_temp(fd=fd, temp_path=temp_path)
            self.staging_pool.release(copy.slot_index)
            self._file_terminal(job, ok=False, exc=exc)
            self._release_store_inflight(job.block_hashes[copy.file_index], ok=False)
            return
        self._inflight[user_data] = _RingOp(
            job=job,
            file_index=copy.file_index,
            slot_index=copy.slot_index,
            fd=fd,
            is_write=True,
            start_ns=start_ns,
            op_kind="write",
            final_path=final_path,
            temp_path=temp_path,
        )

    def _poll_ring_completions(self) -> bool:
        completions = self.ring.poll_all()
        if not completions:
            return False
        for user_data, result in completions:
            op = self._inflight.pop(user_data, None)
            if op is None:
                continue
            kind = op.op_kind
            if kind == "read":
                self._data_inflight -= 1
                self._on_read_complete(op, result)
            elif kind == "open":
                self._on_open_complete(op, result)
            elif kind == "write":
                self._data_inflight -= 1
                self._on_write_complete(op, result)
            # else: close — result ignored.
        return True

    def _on_open_complete(self, op: _RingOp, result_fd: int) -> None:
        self._open_inflight -= 1
        op.path_buf = None
        preload_info = op.preload_info
        if result_fd < 0:
            exc = OSError(-result_fd, "io_uring openat failed")
            if op.preload_hash is not None:
                h = op.preload_hash
                self._preload_inflight_remove(h)
                if self._share_preload:
                    if self._fail_all_preload_waiters(h, exc):
                        return
                    if h in self._store_inflight:
                        blocked = self._preload_blocked_on_write.get(h)
                        if blocked is None:
                            blocked = collections.deque()
                            self._preload_blocked_on_write[h] = blocked
                        if preload_info is not None:
                            blocked.append(preload_info)
                    elif preload_info is not None:
                        self._remember_known_missing(preload_info)
                    return
                waiter = self._preload_waiter_pop(h)
                if waiter is not None:
                    job, _fi = waiter
                    self._file_terminal(job, ok=False, exc=exc)
                elif h in self._store_inflight:
                    # Store started while open was in flight; arm a readback when the write completes.
                    blocked = self._preload_blocked_on_write.get(h)
                    if blocked is None:
                        blocked = collections.deque()
                        self._preload_blocked_on_write[h] = blocked
                    if preload_info is not None:
                        blocked.append(preload_info)
                elif preload_info is not None:
                    self._remember_known_missing(preload_info)
            else:
                assert op.job is not None
                self._file_terminal(op.job, ok=False, exc=exc)
            return
        ready = _ReadyFd(
            fd=result_fd,
            job=op.job,
            file_index=op.file_index,
            preload_hash=op.preload_hash,
            open_start_ns=op.start_ns,
            preload_info=preload_info,
        )
        if op.preload_hash is not None:
            self._ready_fds_preload.append(ready)
        else:
            self._ready_fds_load.append(ready)

    def _on_read_complete(self, op: _RingOp, result_nbytes: int) -> None:
        if op.preload_hash is not None:
            self._on_preload_read_complete(op, result_nbytes)
            return
        job = op.job
        assert job is not None
        try:
            if result_nbytes < 0:
                raise OSError(-result_nbytes, "io_uring read failed")
            if result_nbytes != self.file_store.io_size:
                raise IOError(
                    f"io_uring read transferred {result_nbytes} bytes, "
                    f"expected {self.file_store.io_size}"
                )
            duration_ns = now_ns() - op.start_ns
            job.file_samples.append((op.start_ns, duration_ns, self.layout.storage_block_bytes, op.file_index))
            self._queue_load_copy(job, op.file_index, op.slot_index)
        except BaseException as exc:
            self.staging_pool.release(op.slot_index)
            self._file_terminal(job, ok=False, exc=exc)
        finally:
            self._async_close(op.fd)

    def _on_preload_read_complete(self, op: _RingOp, result_nbytes: int) -> None:
        if self._share_preload:
            self._on_shared_preload_read_complete(op, result_nbytes)
            return
        block_hash = op.preload_hash
        assert block_hash is not None
        preload_info = op.preload_info
        self._preload_inflight_remove(block_hash)
        try:
            if result_nbytes < 0:
                raise OSError(-result_nbytes, "io_uring preload read failed")
            if result_nbytes != self.file_store.io_size:
                raise IOError(
                    f"io_uring preload read transferred {result_nbytes} bytes, "
                    f"expected {self.file_store.io_size}"
                )
            duration_ns = now_ns() - op.start_ns
            waiter = self._preload_waiter_pop(block_hash)
            add_event(
                "py_kvcache.preload.file_read",
                "kv_preload",
                op.start_ns,
                duration_ns,
                tid=preload_info.profile_tid if preload_info else None,
                args=self._preload_event_args(
                    block_hash,
                    preload_info,
                    success=True,
                    nbytes=result_nbytes,
                    has_waiter=waiter is not None,
                    cached=waiter is None and not self._stop,
                ),
            )
            if waiter is not None:
                job, file_index = waiter
                add_event(
                    "py_kvcache.preload.place_from_read",
                    "kv_preload",
                    now_ns(),
                    0,
                    tid=job.profile.profile_tid,
                    args=self._preload_event_args(
                        block_hash,
                        preload_info,
                        load_job_id=job.job_id,
                        load_req_id=job.profile.req_id,
                        load_file_index=file_index,
                    ),
                )
                self._queue_load_copy(job, file_index, op.slot_index, source="preload")
            elif self._stop:
                self.staging_pool.release(op.slot_index)
            else:
                self._preload_cache_push(block_hash, op.slot_index, preload_info)
        except BaseException as exc:
            add_event(
                "py_kvcache.preload.file_read",
                "kv_preload",
                op.start_ns,
                now_ns() - op.start_ns,
                tid=preload_info.profile_tid if preload_info else None,
                args=self._preload_event_args(
                    block_hash,
                    preload_info,
                    success=False,
                    error=type(exc).__name__,
                ),
            )
            self.staging_pool.release(op.slot_index)
            waiter = self._preload_waiter_pop(block_hash)
            if waiter is not None:
                job, _fi = waiter
                self._file_terminal(job, ok=False, exc=exc)
            else:
                logger.warning("py-kvcache preload read failed (no waiter): %s", exc)
        finally:
            self._async_close(op.fd)

    def _on_shared_preload_read_complete(self, op: _RingOp, result_nbytes: int) -> None:
        block_hash = op.preload_hash
        assert block_hash is not None
        preload_info = op.preload_info
        slot_index = op.slot_index
        self._preload_inflight_remove(block_hash)
        try:
            if result_nbytes < 0:
                raise OSError(-result_nbytes, "io_uring preload read failed")
            if result_nbytes != self.file_store.io_size:
                raise IOError(
                    f"io_uring preload read transferred {result_nbytes} bytes, "
                    f"expected {self.file_store.io_size}"
                )
            duration_ns = now_ns() - op.start_ns
            has_waiter = self._has_preload_waiter(block_hash)
            add_event(
                "py_kvcache.preload.file_read",
                "kv_preload",
                op.start_ns,
                duration_ns,
                tid=preload_info.profile_tid if preload_info else None,
                args=self._preload_event_args(
                    block_hash,
                    preload_info,
                    success=True,
                    nbytes=result_nbytes,
                    has_waiter=has_waiter,
                    shared=True,
                ),
            )
            slot = _SharedPreloadSlot(
                block_hash=block_hash,
                slot_index=slot_index,
                preload_info=preload_info,
                cached=False,
            )
            # One read serves every parked waiter for this hash.
            while True:
                waiter = self._preload_waiter_pop(block_hash)
                if waiter is None:
                    break
                job, file_index = waiter
                add_event(
                    "py_kvcache.preload.place_from_read",
                    "kv_preload",
                    now_ns(),
                    0,
                    tid=job.profile.profile_tid,
                    args=self._preload_event_args(
                        block_hash,
                        preload_info,
                        load_job_id=job.job_id,
                        load_req_id=job.profile.req_id,
                        load_file_index=file_index,
                        shared=True,
                    ),
                )
                slot.copies_inflight += 1
                try:
                    self._queue_load_copy(job, file_index, slot_index, shared=slot, source="preload")
                except BaseException as exc:
                    slot.copies_inflight -= 1
                    self._file_terminal(job, ok=False, exc=exc)
                self._shared_decref(job.profile.req_id, block_hash)
            if self._stop:
                self._maybe_release_shared(slot)
            elif self._preload_refcount.get(block_hash, 0) > 0:
                slot.cached = True
                self._shared_cached[block_hash] = slot
                self._preload_cached_total += 1
            else:
                self._maybe_release_shared(slot)
        except BaseException as exc:
            add_event(
                "py_kvcache.preload.file_read",
                "kv_preload",
                op.start_ns,
                now_ns() - op.start_ns,
                tid=preload_info.profile_tid if preload_info else None,
                args=self._preload_event_args(
                    block_hash,
                    preload_info,
                    success=False,
                    error=type(exc).__name__,
                    shared=True,
                ),
            )
            if not self._fail_all_preload_waiters(block_hash, exc):
                logger.warning("py-kvcache shared preload read failed (no waiter): %s", exc)
            self.staging_pool.release(slot_index)
        finally:
            self._async_close(op.fd)

    def _on_write_complete(self, op: _RingOp, result_nbytes: int) -> None:
        job = op.job
        try:
            if result_nbytes < 0:
                raise OSError(-result_nbytes, "io_uring write failed")
            if result_nbytes != self.file_store.io_size:
                raise IOError(
                    f"io_uring write transferred {result_nbytes} bytes, "
                    f"expected {self.file_store.io_size}"
                )
            self.file_store.finish_write(
                fd=op.fd,
                temp_path=op.temp_path,
                final_path=op.final_path,
            )
            duration_ns = now_ns() - op.start_ns
            job.file_samples.append((op.start_ns, duration_ns, self.layout.storage_block_bytes, op.file_index))
            # The store slot view is byte-identical to a read of the same hash, so
            # retain it as a cache entry instead of releasing (write-back).
            self._release_or_cache(job.block_hashes[op.file_index], op.slot_index)
            self._file_terminal(job, ok=True)
            self._release_store_inflight(job.block_hashes[op.file_index], ok=True)
        except BaseException as exc:
            self.file_store.cleanup_temp(fd=op.fd, temp_path=op.temp_path)
            self.staging_pool.release(op.slot_index)
            self._file_terminal(job, ok=False, exc=exc)
            self._release_store_inflight(job.block_hashes[op.file_index], ok=False)

    def _can_issue_store(self) -> bool:
        # Device budget released at CQE reap, not at CUDA copy completion.
        # Slot acquisition is _schedule_one's job (evicts a preload slot if full).
        return self._data_inflight < self.iodepth

    def _can_open(self) -> bool:
        primed = self._open_inflight + len(self._ready_fds_load) + len(self._ready_fds_preload)
        return primed < self.open_lookahead

    def _has_real_load_pressure(self) -> bool:
        # CUDA copies (staging->GPU) don't count: they're async DMA, not disk I/O.
        if self._ready_fds_load:
            return True
        if any(
            op.op_kind in ("open", "read") and op.job is not None and not op.job.is_store
            for op in self._inflight.values()
        ):
            return True
        for job in self._active:
            if job.is_store or job.failed is not None or job.future_set:
                continue
            idx = job.next_file_index
            if idx < job.total_files:
                h = job.block_hashes[idx]
                if not self._preload_has_copy(h):
                    return True
        return False

    def _can_issue_speculative_preload(self) -> bool:
        return not self._has_real_load_pressure()

    def _preload_slots_available(self) -> bool:
        occupied = self._preload_cached_total + self._preload_inflight_total
        return occupied < self._max_preload_slots

    def _evict_one_preload_slot(self) -> bool:
        if self._share_preload:
            return self._evict_one_shared_slot()
        if not self._preload_slots:
            return False
        block_hash = next(iter(self._preload_slots))
        entry = self._preload_cache_pop(block_hash)
        assert entry is not None
        slot_index, preload_info = entry
        add_event(
            "py_kvcache.preload.evict_slot",
            "kv_preload",
            now_ns(),
            0,
            tid=preload_info.profile_tid if preload_info else None,
            args=self._preload_event_args(block_hash, preload_info, reason="load_slot_pressure"),
        )
        self.staging_pool.release(slot_index)
        return True

    def _evict_one_cache_slot(self, reason: str) -> bool:
        if self._staging_cache is None:
            return False
        slot_index = self._staging_cache.evict_one()
        if slot_index is None:
            return False
        self.staging_pool.release(slot_index)
        add_event(
            "py_kvcache.staging_cache.evict",
            "kv_staging_cache",
            now_ns(),
            0,
            args={"reason": reason, "size": len(self._staging_cache)},
        )
        return True

    def _reserve_foreground_slot(self):
        # Slot priority: foreground > preload > cache, so reclaim cache then preload.
        slot = self.staging_pool.try_reserve()
        if slot is None and self._evict_one_cache_slot("foreground_pressure"):
            slot = self.staging_pool.try_reserve()
        if slot is None and self._evict_one_preload_slot():
            slot = self.staging_pool.try_reserve()
        return slot

    def _reserve_preload_slot(self):
        # Preload outranks the cache: a speculative read may reclaim a cache slot.
        slot = self.staging_pool.try_reserve()
        if slot is None and self._evict_one_cache_slot("preload_pressure"):
            slot = self.staging_pool.try_reserve()
        return slot

    def _release_or_cache(self, block_hash: bytes, slot_index: int) -> None:
        if self._staging_cache is None:
            self.staging_pool.release(slot_index)
            return
        to_release = self._staging_cache.put(block_hash, slot_index)
        if to_release is not None:
            self.staging_pool.release(to_release)
        if to_release != slot_index:
            add_event(
                "py_kvcache.staging_cache.insert",
                "kv_staging_cache",
                now_ns(),
                0,
                args={"size": len(self._staging_cache)},
            )

    def _settle_load_slot(
        self,
        job: "_ReactorJob",
        file_index: int,
        slot_index: int,
        shared: "_SharedPreloadSlot | None",
        cache: "_CacheSlot | None",
    ) -> None:
        # Disk bytes stay valid regardless of the GPU copy outcome (immutable
        # hash), so success, empty, and error completions all settle here.
        if cache is not None:
            self._staging_cache.unpin(cache)
        elif shared is None:
            self._release_or_cache(job.block_hashes[file_index], slot_index)
        else:
            shared.copies_inflight -= 1
            self._maybe_release_shared(shared)

    def _schedule_cache_hit(
        self, job: "_ReactorJob", file_index: int, cache_slot: "_CacheSlot"
    ) -> None:
        # Non-destructive: the slot stays in the cache, pinned for the copy's DMA.
        add_event(
            "py_kvcache.staging_cache.hit",
            "kv_staging_cache",
            now_ns(),
            0,
            tid=job.profile.profile_tid,
            args={"load_job_id": job.job_id, "load_req_id": job.profile.req_id},
        )
        self._staging_cache.pin(cache_slot)
        job.next_file_index += 1
        job.inflight_files += 1
        try:
            self._queue_load_copy(job, file_index, cache_slot.slot_index, cache=cache_slot, source="cache")
        except BaseException as exc:
            self._staging_cache.unpin(cache_slot)
            self._file_terminal(job, ok=False, exc=exc)

    def _schedule_work(self) -> bool:
        made = False

        # 0. Drain opened load fds first (latency-critical).
        made = self._drain_ready_load_fds() or made

        # 1. Load jobs: claim cached/inflight preload copies or open files.
        for job in self._active:
            if job.failed is not None or job.is_store:
                continue
            while job.failed is None and job.next_file_index < job.total_files:
                file_index = job.next_file_index
                h = job.block_hashes[file_index]
                if self._staging_cache is not None:
                    cs = self._staging_cache.get(h)
                    if cs is not None:
                        self._schedule_cache_hit(job, file_index, cs)
                        made = True
                        continue
                if self._share_preload:
                    if h in self._shared_cached:
                        self._shared_claim_from_cache(job, file_index)
                        made = True
                    elif self._has_inflight_preload(h):
                        # One read serves all waiters, so no waiter_room cap here.
                        self._schedule_preload_inflight(job, file_index)
                        made = True
                    elif self._preload_pending_count.get(h, 0) > 0:
                        # Queued but unopened: promote to a shared read, don't wait on the gate.
                        if not self._can_open():
                            break
                        self._promote_pending_to_shared_read(job, file_index)
                        made = True
                    else:
                        if not self._can_open():
                            break
                        self._submit_open_read(job, file_index)
                        self._shared_decref(job.profile.req_id, h)
                        made = True
                    continue
                if self._has_cached_preload(h):
                    self._schedule_preload_done(job, file_index)
                    made = True
                elif self._has_inflight_preload(h) and self._preload_waiter_room(h):
                    self._schedule_preload_inflight(job, file_index)
                    made = True
                else:
                    if not self._can_open():
                        break
                    self._submit_open_read(job, file_index)
                    self._cancel_one_pending(h)
                    made = True

        # 2. Store jobs (background relative to loads).
        for job in self._active:
            if job.failed is not None or not job.is_store:
                continue
            while (
                job.failed is None
                and job.next_file_index < job.total_files
                and self._can_issue_store()
            ):
                if not self._schedule_one(job):
                    break
                made = True

        # 3. Speculative preloads: only during store windows or idle.
        made = self._drain_ready_preload_fds() or made
        self._drain_cancelled_pending()
        if self._can_issue_speculative_preload():
            while self._preload_pending and self._can_open() and self._preload_slots_available():
                info = self._preload_pending[0]
                h = info.block_hash
                # No cross-copy dedup here: each pending entry is a distinct
                # request's demand for its own staged copy (intake already
                # deduped per (req_id, hash)). An already-inflight/cached copy
                # belongs to another request and does not satisfy this one.
                if self._preload_pending_cancel.get(h, 0) > 0:
                    self._preload_pending.popleft()
                    self._consume_pending_cancel(h)
                    self._pending_dec(h)
                    continue
                if h in self._store_inflight:
                    self._preload_pending.popleft()
                    self._pending_dec(h)
                    blocked = self._preload_blocked_on_write.get(h)
                    if blocked is None:
                        blocked = collections.deque()
                        self._preload_blocked_on_write[h] = blocked
                    blocked.append(info)
                    add_event(
                        "py_kvcache.preload.defer_on_write",
                        "kv_preload",
                        now_ns(),
                        0,
                        tid=info.profile_tid,
                        args=self._preload_event_args(h, info),
                    )
                    continue
                self._preload_pending.popleft()
                self._pending_dec(h)
                self._submit_open_preload(info)
                made = True

        return made

    def _consume_pending_cancel(self, block_hash: bytes) -> None:
        count = self._preload_pending_cancel.get(block_hash, 0)
        if count <= 1:
            self._preload_pending_cancel.pop(block_hash, None)
        else:
            self._preload_pending_cancel[block_hash] = count - 1

    def _drain_cancelled_pending(self) -> None:
        while self._preload_pending and (
            self._preload_pending_cancel.get(self._preload_pending[0].block_hash, 0) > 0
        ):
            h = self._preload_pending.popleft().block_hash
            self._consume_pending_cancel(h)
            self._pending_dec(h)

    def _drain_ready_load_fds(self) -> bool:
        made = False
        while self._ready_fds_load and self._data_inflight < self.iodepth:
            slot = self._reserve_foreground_slot()
            if slot is None:
                return made
            ready = self._ready_fds_load.popleft()
            self._submit_read_from_ready(ready, slot.index)
            made = True

        return made

    def _requeue_preload_hash(self, info: "_PreloadInfo") -> None:
        self._pending_push(info)

    def _remember_known_missing(self, info: "_PreloadInfo") -> None:
        block_hash = info.block_hash
        missed = self._known_missing.get(block_hash)
        if missed is None:
            missed = collections.deque()
            self._known_missing[block_hash] = missed
        missed.append(info)
        self._known_missing.move_to_end(block_hash)
        while len(self._known_missing) > self._max_known_missing:
            self._known_missing.popitem(last=False)

    def _release_store_inflight(self, block_hash: bytes, *, ok: bool) -> None:
        count = self._store_inflight.get(block_hash, 0) - 1
        if count > 0:
            self._store_inflight[block_hash] = count
            return
        self._store_inflight.pop(block_hash, None)
        blocked = self._preload_blocked_on_write.pop(block_hash, None)
        if not blocked:
            return
        if not ok or self._stop:
            return
        if self._staging_cache is not None and block_hash in self._staging_cache:
            # Store already cached the bytes; a reuse load hits the cache, so the
            # re-arm read is redundant. Clear shared demand so refcount does not leak.
            if self._share_preload:
                self._purge_owned_for_hash(block_hash)
                self._preload_refcount.pop(block_hash, None)
            return
        if self._share_preload:
            # refcount carries multiplicity, so one read-back serves all refholders.
            self._requeue_preload_hash(blocked[0])
        else:
            for info in blocked:
                self._requeue_preload_hash(info)
        add_event(
            "py_kvcache.preload.wake_after_write",
            "kv_preload",
            now_ns(),
            0,
            tid=blocked[0].profile_tid,
            args=self._preload_event_args(block_hash, blocked[0], copies=len(blocked)),
        )

    def _drain_ready_preload_fds(self) -> bool:
        made = False
        while self._ready_fds_preload:
            ready = self._ready_fds_preload[0]
            preload_hash = ready.preload_hash
            waited = preload_hash is not None and self._has_preload_waiter(preload_hash)
            if waited:
                if self._data_inflight >= self.iodepth:
                    return made
                slot = self._reserve_foreground_slot()
                if slot is None:
                    return made
                self._ready_fds_preload.popleft()
                self._submit_read_from_ready(ready, slot.index)
                made = True
                continue
            if preload_hash is not None and self._has_real_load_pressure():
                preload_info = ready.preload_info
                self._ready_fds_preload.popleft()
                self._preload_inflight_remove(preload_hash)
                if preload_info is not None:
                    self._requeue_preload_hash(preload_info)
                add_event(
                    "py_kvcache.preload.yield_open_fd",
                    "kv_preload",
                    now_ns(),
                    0,
                    tid=preload_info.profile_tid if preload_info else None,
                    args=self._preload_event_args(
                        preload_hash,
                        preload_info,
                        reason="real_load_pressure",
                    ),
                )
                self._async_close(ready.fd)
                made = True
                continue
            if self._data_inflight >= self.iodepth or not self._can_issue_speculative_preload():
                return made
            slot = self._reserve_preload_slot()
            if slot is None:
                return made
            self._ready_fds_preload.popleft()
            self._submit_read_from_ready(ready, slot.index)
            made = True
        return made

    def _schedule_one(self, job: _ReactorJob) -> bool:
        slot = self._reserve_foreground_slot()
        if slot is None:
            return False
        self._schedule_store_copy(job, job.next_file_index, slot.index)
        return True

    def _schedule_preload_done(self, job: _ReactorJob, file_index: int) -> None:
        h = job.block_hashes[file_index]
        entry = self._preload_cache_pop(h)
        assert entry is not None
        slot_index, preload_info = entry
        add_event(
            "py_kvcache.preload.place_from_cache",
            "kv_preload",
            now_ns(),
            0,
            tid=job.profile.profile_tid,
            args=self._preload_event_args(
                h,
                preload_info,
                load_job_id=job.job_id,
                load_req_id=job.profile.req_id,
                load_file_index=file_index,
            ),
        )
        job.next_file_index += 1
        job.inflight_files += 1
        try:
            self._queue_load_copy(job, file_index, slot_index, source="preload")
        except BaseException as exc:
            self.staging_pool.release(slot_index)
            self._file_terminal(job, ok=False, exc=exc)

    def _schedule_preload_inflight(self, job: _ReactorJob, file_index: int) -> None:
        h = job.block_hashes[file_index]
        add_event(
            "py_kvcache.preload.wait_for_inflight",
            "kv_preload",
            now_ns(),
            0,
            tid=job.profile.profile_tid,
            args=self._preload_event_args(
                h,
                None,
                load_job_id=job.job_id,
                load_req_id=job.profile.req_id,
                load_file_index=file_index,
            ),
        )
        job.next_file_index += 1
        job.inflight_files += 1
        self._preload_waiter_add(h, job, file_index)

    def _shared_claim_from_cache(self, job: _ReactorJob, file_index: int) -> None:
        h = job.block_hashes[file_index]
        slot = self._shared_cached[h]
        add_event(
            "py_kvcache.preload.place_from_cache",
            "kv_preload",
            now_ns(),
            0,
            tid=job.profile.profile_tid,
            args=self._preload_event_args(
                h,
                slot.preload_info,
                load_job_id=job.job_id,
                load_req_id=job.profile.req_id,
                load_file_index=file_index,
                shared=True,
                refcount=self._preload_refcount.get(h, 0),
            ),
        )
        job.next_file_index += 1
        job.inflight_files += 1
        slot.copies_inflight += 1
        try:
            self._queue_load_copy(job, file_index, slot.slot_index, shared=slot, source="preload")
        except BaseException as exc:
            slot.copies_inflight -= 1
            self._file_terminal(job, ok=False, exc=exc)
            self._maybe_release_shared(slot)
        self._shared_decref(job.profile.req_id, h)

    def _promote_pending_to_shared_read(self, job: _ReactorJob, file_index: int) -> None:
        h = job.block_hashes[file_index]
        self._cancel_one_pending(h)
        info = self._make_preload_info_for_job(job, file_index)
        self._submit_open_preload(info)
        self._schedule_preload_inflight(job, file_index)

    def _submit_open_read(self, job: _ReactorJob, file_index: int) -> None:
        # Slot reserved at read issue, not here — slow opens never tie up buffers.
        job.next_file_index += 1
        job.inflight_files += 1
        final_path = self.file_mapper.get_file_name(job.block_hashes[file_index])
        user_data = self._new_user_data()
        start_ns = now_ns()
        try:
            path_buf = self.file_store.queue_open_read(
                self.ring, user_data=user_data, path=final_path
            )
        except BaseException as exc:
            self._file_terminal(job, ok=False, exc=exc)
            return
        self._open_inflight += 1
        self._inflight[user_data] = _RingOp(
            job=job,
            file_index=file_index,
            slot_index=-1,
            fd=-1,
            is_write=False,
            start_ns=start_ns,
            op_kind="open",
            path_buf=path_buf,
        )

    def _submit_open_preload(self, info: "_PreloadInfo") -> None:
        # The async openat IS the existence check: a miss flows to _known_missing
        # so a later store re-arms the readback. Do NOT pre-gate with os.path.exists
        # — it stalls the pump and breaks the store->preload re-arm chain.
        block_hash = info.block_hash
        final_path = self.file_mapper.get_file_name(block_hash)
        user_data = self._new_user_data()
        start_ns = now_ns()
        try:
            path_buf = self.file_store.queue_open_read(
                self.ring, user_data=user_data, path=final_path
            )
        except BaseException as exc:
            logger.warning("py-kvcache preload openat submit failed: %s", exc)
            return
        self._open_inflight += 1
        self._preload_inflight_add(block_hash)
        self._inflight[user_data] = _RingOp(
            job=None,
            file_index=-1,
            slot_index=-1,
            fd=-1,
            is_write=False,
            start_ns=start_ns,
            op_kind="open",
            preload_hash=block_hash,
            preload_info=info,
            path_buf=path_buf,
        )

    def _submit_read_from_ready(self, ready: _ReadyFd, slot_index: int) -> None:
        io_view = self._slot_view(slot_index)
        preload_info = ready.preload_info
        user_data = self._new_user_data()
        start_ns = now_ns()
        try:
            self.file_store.queue_read(
                self.ring,
                user_data=user_data,
                fd=ready.fd,
                io_array=io_view,
            )
        except BaseException as exc:
            self._async_close(ready.fd)
            self.staging_pool.release(slot_index)
            if ready.preload_hash is not None:
                add_event(
                    "py_kvcache.preload.file_read",
                    "kv_preload",
                    start_ns,
                    now_ns() - start_ns,
                    tid=preload_info.profile_tid if preload_info else None,
                    args=self._preload_event_args(
                        ready.preload_hash,
                        preload_info,
                        success=False,
                        error=type(exc).__name__,
                    ),
                )
                self._preload_inflight_remove(ready.preload_hash)
                if self._share_preload:
                    if not self._fail_all_preload_waiters(ready.preload_hash, exc):
                        logger.warning(
                            "py-kvcache shared preload queue_read failed (no waiter): %s", exc
                        )
                else:
                    waiter = self._preload_waiter_pop(ready.preload_hash)
                    if waiter is not None:
                        self._file_terminal(waiter[0], ok=False, exc=exc)
                    else:
                        logger.warning(
                            "py-kvcache preload queue_read failed (no waiter): %s", exc
                        )
            else:
                assert ready.job is not None
                self._file_terminal(ready.job, ok=False, exc=exc)
            return
        self._data_inflight += 1
        self._inflight[user_data] = _RingOp(
            job=ready.job,
            file_index=ready.file_index,
            slot_index=slot_index,
            fd=ready.fd,
            is_write=False,
            start_ns=start_ns,
            op_kind="read",
            preload_hash=ready.preload_hash,
            preload_info=preload_info,
        )

    def _async_close(self, fd: int) -> None:
        # Tracked in _inflight so shutdown drains it cleanly.
        if fd < 0:
            return
        user_data = self._new_user_data()
        try:
            self.file_store.queue_close(self.ring, user_data=user_data, fd=fd)
        except BaseException:
            try:
                os.close(fd)
            except OSError:
                pass
            return
        self._inflight[user_data] = _RingOp(
            job=None,
            file_index=-1,
            slot_index=-1,
            fd=fd,
            is_write=False,
            start_ns=now_ns(),
            op_kind="close",
        )

    def _schedule_store_copy(self, job: _ReactorJob, file_index: int, slot_index: int) -> bool:
        job.next_file_index += 1
        job.inflight_files += 1
        try:
            src_to_dst = self._build_mapping(
                slot_index,
                job.block_chunks[file_index],
                1,
                self._slot_block_ids[slot_index],
                self.layout.storage_block_size_factor,
            )
            copy = self._launch_swap_blocks(
                slot_index,
                src_to_dst,
                self.layout.storage_block_bytes,
                is_store=True,
                job=job,
                file_index=file_index,
            )
        except BaseException as exc:
            self.staging_pool.release(slot_index)
            self._file_terminal(job, ok=False, exc=exc)
            self._release_store_inflight(job.block_hashes[file_index], ok=False)
            return False
        # A store occupies one device-budget unit from copy-launch through
        # write-completion (one unit, two phases): without this, _can_issue_store
        # never sees in-flight store copies and launches up to slot_count of them,
        # then drains every write into the SQ in one pump -> SQ overflow.
        self._data_inflight += 1
        self._pending_copies.append(copy)
        return True

    def _build_mapping(
        self,
        slot_index: int,
        src_blocks: np.ndarray,
        src_block_size_factor: int,
        dst_blocks: np.ndarray,
        dst_block_size_factor: int,
    ) -> Any:
        output = self._mapping_buffers[slot_index]
        mapping_length = build_block_mapping(
            src_blocks,
            src_block_size_factor,
            dst_blocks,
            dst_block_size_factor,
            output,
        )
        return self._mapping_tensors[slot_index][:mapping_length]

    def _queue_load_copy(
        self,
        job: _ReactorJob,
        file_index: int,
        slot_index: int,
        *,
        shared: "_SharedPreloadSlot | None" = None,
        cache: "_CacheSlot | None" = None,
        source: str = "file",
    ) -> None:
        if source == "preload":
            job.n_from_preload += 1
        elif source == "cache":
            job.n_from_cache += 1
        else:
            job.n_from_file += 1
        output = self._mapping_buffers[slot_index]
        mapping_length = build_block_mapping(
            self._slot_block_ids[slot_index],
            self.layout.storage_block_size_factor,
            job.block_chunks[file_index],
            1,
            output,
        )
        # Snapshot: the per-slot buffer is reused, including by shared siblings in a pump.
        mapping = output[:mapping_length].copy()
        self._copy_ready.append(
            _ReadyCopy(
                job=job,
                file_index=file_index,
                slot_index=slot_index,
                mapping_len=mapping_length,
                nbytes=self.layout.storage_block_bytes,
                mapping=mapping,
                shared=shared,
                cache=cache,
            )
        )

    def _ensure_fused_capacity(self, entries: int) -> None:
        if entries <= self._fused_capacity:
            return
        cap = max(entries, self._fused_capacity * 2, 1024)
        self._fused_src_np = np.empty(cap, dtype=np.int64)
        self._fused_dst_np = np.empty(cap, dtype=np.int64)
        self._fused_sizes_np = np.empty(cap, dtype=np.int64)
        self._fused_src_t = torch.from_numpy(self._fused_src_np)
        self._fused_dst_t = torch.from_numpy(self._fused_dst_np)
        self._fused_sizes_t = torch.from_numpy(self._fused_sizes_np)
        self._fused_capacity = cap

    def _fill_swap_ptrs(
        self,
        mapping: np.ndarray,
        slot_index: int,
        staging_is_src: bool,
        src_arr: np.ndarray,
        dst_arr: np.ndarray,
        sizes_arr: np.ndarray,
        offset: int,
    ) -> int:
        """Fill swap_blocks pointer arrays for one staging slot."""
        layout = self.layout
        factor = layout.storage_block_size_factor
        slot_base = self._staging_base_ptr + slot_index * layout.storage_block_bytes
        if staging_is_src:
            staging_local = mapping[:, 0] - slot_index * factor
            gpu_blocks = mapping[:, 1]
        else:
            gpu_blocks = mapping[:, 0]
            staging_local = mapping[:, 1] - slot_index * factor
        rows = mapping.shape[0]
        end = offset
        for gpu_tensor, block_bytes, layer_offset in zip(
            layout.gpu_tensors, layout.bytes_per_kernel_block, layout.staging_layer_offsets
        ):
            start = end
            end = start + rows
            staging_ptr = slot_base + layer_offset + staging_local * block_bytes
            gpu_ptr = gpu_tensor.data_ptr() + gpu_blocks * block_bytes
            if staging_is_src:
                src_arr[start:end] = staging_ptr
                dst_arr[start:end] = gpu_ptr
            else:
                src_arr[start:end] = gpu_ptr
                dst_arr[start:end] = staging_ptr
            sizes_arr[start:end] = block_bytes
        return end

    def _flush_copy_batch(self) -> bool:
        ready = self._copy_ready
        if not ready:
            return False
        self._copy_ready = []
        total = sum(rc.mapping_len for rc in ready)
        if total <= 0:
            for rc in ready:
                self._settle_load_slot(rc.job, rc.file_index, rc.slot_index, rc.shared, rc.cache)
                self._file_terminal(rc.job, ok=True)
            return True
        self._ensure_fused_capacity(total * len(self.layout.gpu_tensors))
        src = self._fused_src_np
        dst = self._fused_dst_np
        sizes = self._fused_sizes_np
        start_ns = now_ns()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        try:
            with torch.cuda.stream(self._copy_stream):
                start_event.record(self._copy_stream)
                offset = 0
                for rc in ready:
                    offset = self._fill_swap_ptrs(
                        rc.mapping,
                        rc.slot_index,
                        True,
                        src,
                        dst,
                        sizes,
                        offset,
                    )
                ops.swap_blocks_batch(
                    self._fused_src_t[:offset],
                    self._fused_dst_t[:offset],
                    self._fused_sizes_t[:offset],
                )
                end_event.record(self._copy_stream)
        except BaseException as exc:
            for rc in ready:
                self._settle_load_slot(rc.job, rc.file_index, rc.slot_index, rc.shared, rc.cache)
                self._file_terminal(rc.job, ok=False, exc=exc)
            return True
        self._pending_copies.append(
            _CopyOp(
                job=None,
                file_index=-1,
                slot_index=-1,
                is_store=False,
                start_ns=start_ns,
                start_event=start_event,
                end_event=end_event,
                nbytes=len(ready) * self.layout.storage_block_bytes,
                members=[
                    (rc.slot_index, rc.job, rc.file_index, rc.shared, rc.cache) for rc in ready
                ],
            )
        )
        return True

    def _launch_swap_blocks(
        self,
        slot_index: int,
        src_to_dst: Any,
        sample_nbytes: int,
        *,
        is_store: bool,
        job: _ReactorJob,
        file_index: int,
    ) -> _CopyOp:
        stream = self._streams[slot_index]
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_ns = now_ns()
        with torch.cuda.stream(stream):
            if is_store and job.compute_event is not None:
                stream.wait_event(job.compute_event)
            start_event.record(stream)
            mapping_len = int(src_to_dst.shape[0])
            if mapping_len > 0:
                mapping = self._mapping_buffers[slot_index][:mapping_len]
                offset = self._fill_swap_ptrs(
                    mapping,
                    slot_index,
                    not is_store,
                    self._batch_src_ptrs[slot_index],
                    self._batch_dst_ptrs[slot_index],
                    self._batch_sizes[slot_index],
                    0,
                )
                ops.swap_blocks_batch(
                    self._batch_src_tensors[slot_index][:offset],
                    self._batch_dst_tensors[slot_index][:offset],
                    self._batch_size_tensors[slot_index][:offset],
                )
            end_event.record(stream)
        return _CopyOp(
            job=job,
            file_index=file_index,
            slot_index=slot_index,
            is_store=is_store,
            start_ns=start_ns,
            start_event=start_event,
            end_event=end_event,
            nbytes=sample_nbytes,
        )

    def _file_terminal(
        self, job: _ReactorJob, *, ok: bool, exc: BaseException | None = None
    ) -> None:
        job.inflight_files -= 1
        if ok:
            job.done_files += 1
        if job.future_set:
            return
        if exc is not None:
            job.failed = exc
            job.future_set = True
            logger.exception("py-kvcache transfer job %s failed", job.job_id, exc_info=exc)
            self._emit(job, success=False)
            job.future.set_exception(exc)
        elif job.done_files == job.total_files:
            job.future_set = True
            self._emit(job, success=True)
            job.future.set_result(job.transfer_size)

    def _finish_jobs(self) -> bool:
        if not self._active:
            return False
        remaining: list[_ReactorJob] = []
        removed = False
        for job in self._active:
            drained = job.inflight_files == 0 and (
                job.next_file_index >= job.total_files or job.failed is not None
            )
            if drained and job.future_set:
                removed = True
                continue
            remaining.append(job)
        if removed:
            self._active = remaining
        return removed

    def _emit(self, job: _ReactorJob, *, success: bool) -> None:
        emit_transfer_events(
            self.layout,
            job.profile,
            success=success,
            transfer_size=job.transfer_size if success else 0,
            file_samples=job.file_samples,
            cuda_samples=job.cuda_samples,
            num_files=job.total_files,
            num_blocks=job.num_blocks,
            n_from_file=job.n_from_file,
            n_from_preload=job.n_from_preload,
            n_from_cache=job.n_from_cache,
        )

    def _slot_view(self, slot_index: int) -> np.ndarray[Any, np.dtype[np.uint8]]:
        return self.layout.storage_slot_view(self.staging_buffer, slot_index)

    def _new_user_data(self) -> int:
        self._next_user_data += 1
        return self._next_user_data


class TransferCoordinator:
    def __init__(
        self,
        *,
        config: SharedFileConfig,
        file_mapper: FileMapper,
        layout: ParsedKvLayout,
        break_even: BreakEvenThresholds | None = None,
        storage_block_tokens: int = 0,
    ) -> None:
        self.layout = layout
        self.reactor = IoReactor(
            config=config,
            file_mapper=file_mapper,
            layout=layout,
            break_even=break_even,
            storage_block_tokens=storage_block_tokens,
        )

    def submit_store(
        self,
        src_spec: Any,
        dst_spec: Any,
        *,
        job_id: int = -1,
        profile_tid: str = "kv_store",
        req_id: str = "",
    ) -> "Future[int]":
        start_ns = now_ns()
        profile = _TransferProfile(
            job_id=job_id,
            direction="gpu_to_storage",
            profile_tid=profile_tid,
            req_id=req_id,
            start_ns=start_ns,
        )
        future: "Future[int]" = Future()
        block_hashes = list(dst_spec.block_hashes)
        total = len(block_hashes)
        if total == 0:
            future.set_result(0)
            return future
        src_chunks = split_block_ids_for_files(self.layout, block_ids_of(src_spec), total)
        # Capture the compute stream here (worker thread); the reactor thread cannot.
        compute_event = torch.cuda.Event()
        compute_event.record(torch.cuda.current_stream())
        job = _ReactorJob(
            job_id=job_id,
            is_store=True,
            block_hashes=block_hashes,
            block_chunks=src_chunks,
            profile=profile,
            future=future,
            total_files=total,
            transfer_size=total * self.layout.storage_block_bytes,
            num_blocks=safe_num_blocks(src_spec),
            compute_event=compute_event,
        )
        future.set_running_or_notify_cancel()
        self.reactor.submit_job(job)
        return future

    def submit_load(
        self,
        src_spec: Any,
        dst_spec: Any,
        *,
        job_id: int = -1,
        profile_tid: str = "kv_load",
        req_id: str = "",
    ) -> "Future[int]":
        start_ns = now_ns()
        profile = _TransferProfile(
            job_id=job_id,
            direction="storage_to_gpu",
            profile_tid=profile_tid,
            req_id=req_id,
            start_ns=start_ns,
        )
        future: "Future[int]" = Future()
        block_hashes = list(src_spec.block_hashes)
        total = len(block_hashes)
        if total == 0:
            future.set_result(0)
            return future
        dst_chunks = split_block_ids_for_files(
            self.layout,
            block_ids_of(dst_spec),
            total,
            first_group_block_index(dst_spec),
        )
        job = _ReactorJob(
            job_id=job_id,
            is_store=False,
            block_hashes=block_hashes,
            block_chunks=dst_chunks,
            profile=profile,
            future=future,
            total_files=total,
            transfer_size=total * self.layout.storage_block_bytes,
            num_blocks=safe_num_blocks(dst_spec),
        )
        future.set_running_or_notify_cancel()
        self.reactor.submit_job(job)
        return future

    def submit_preload(
        self,
        block_hashes: list[bytes],
        *,
        preload_id: str = "",
        req_id: str = "",
        profile_tid: str = "kv_preload",
    ) -> None:
        self.reactor.enqueue_preload(
            block_hashes,
            preload_id=preload_id,
            req_id=req_id,
            profile_tid=profile_tid,
        )

    def shutdown(self) -> None:
        self.reactor.shutdown()


__all__ = ["IoReactor", "TransferCoordinator"]
