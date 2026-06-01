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

from .file_mapper import FileMapper
from .fs_config import SharedFileConfig
from .liburing_file import (
    DEFAULT_IODEPTH,
    DIRECT_IO_ALIGNMENT,
    DirectIoFileStore,
    LiburingRing,
    align_up,
)
from .profiling import add_event, now_ns
from .staging import StagingPool
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
    pack_samples: list = field(default_factory=list)
    unpack_samples: list = field(default_factory=list)


@dataclass
class _PreloadRequest:
    block_hashes: list[bytes]


@dataclass
class _RingOp:
    job: _ReactorJob | None  # None for preload reads
    file_index: int
    slot_index: int
    fd: int
    is_write: bool
    start_ns: int
    direct: bool
    # "open" | "read" | "write" | "close". Opens carry no slot/fd yet; their CQE
    # res is the opened fd. Closes are fire-and-forget (result ignored).
    op_kind: str = "read"
    final_path: str | None = None
    temp_path: str | None = None
    preload_hash: bytes | None = None
    # Holds the openat path buffer alive until the open CQE is reaped.
    path_buf: Any = None


@dataclass
class _ReadyFd:
    """An fd opened by an async ``openat``, waiting for a staging slot + read
    budget before its read is issued. Slot-free so opens never tie up buffers."""
    fd: int
    job: _ReactorJob | None
    file_index: int
    preload_hash: bytes | None
    open_start_ns: int


@dataclass
class _CopyOp:
    job: _ReactorJob
    file_index: int
    slot_index: int
    is_store: bool  # True: GPU -> staging (store); False: staging -> GPU (load)
    start_ns: int
    start_event: Any
    end_event: Any
    nbytes: int


class IoReactor:
    """Single I/O reactor thread.

    Owns the io_uring ring, the staging pool, the direct-I/O file store, and the
    CUDA copy streams. One thread runs a pump loop that batches async io_uring
    submission (never a blocking ``submit_and_wait`` per op), polls completions
    non-blocking, drains CUDA copies via ``event.query()``, and refills the
    pipeline up to ``iodepth`` while staging slots are free. Planning (splitting
    block ids, building specs) happens on the caller thread; the reactor only
    receives ready jobs through a queue. This reproduces the proven gap-free
    hot loop of the old ``storage_kvcache`` multithreaded-liburing engine while
    using exactly one thread.
    """

    _STOP = object()

    def __init__(
        self,
        *,
        config: SharedFileConfig,
        file_mapper: FileMapper,
        layout: ParsedKvLayout,
    ) -> None:
        if not TORCH_COPY_AVAILABLE:
            raise RuntimeError("vLLM and torch are required for the I/O reactor")
        self.config = config
        self.file_mapper = file_mapper
        self.layout = layout
        payload_size = layout.storage_block_bytes
        self.file_store = DirectIoFileStore(config, payload_size=payload_size)
        self.iodepth = max(1, config.iodepth or DEFAULT_IODEPTH)
        # Reads are bounded by ``iodepth`` (device queue depth); their staging
        # slots stay reserved through the GPU copy phase that follows the read.
        # Floor the pool above ``iodepth`` so reads never starve waiting for a
        # slot that is merely draining through a copy.
        self._copy_headroom = max(4, self.iodepth // 2)
        min_slots = self.iodepth + self._copy_headroom
        self.staging_pool = StagingPool.from_memory_budget(
            staging_mem_gib=config.staging_mem,
            payload_size=payload_size,
            io_size=align_up(payload_size),
            alignment=DIRECT_IO_ALIGNMENT,
            min_slots=min_slots,
        )
        slot_count = self.staging_pool.slot_count
        if slot_count <= min_slots:
            logger.warning(
                "staging slots floored to %d (iodepth=%d + copy_headroom=%d); "
                "raise staging_mem so reads do not stall on copy-held slots",
                slot_count,
                self.iodepth,
                self._copy_headroom,
            )
        self.staging_tensors = layout.allocate_staging_tensors(slot_count)
        # Pre-open lookahead: how many async ``openat`` ops + opened-but-not-read
        # fds may be outstanding. Opens are slot-free, so this is independent of
        # ``iodepth`` (which bounds reads+writes / device queue depth).
        self.open_lookahead = max(1, config.open_lookahead or self.iodepth)
        # The ring must hold reads (iodepth) + pre-opens (open_lookahead) + the
        # trailing async closes simultaneously.
        ring_depth = max(self.iodepth + self.open_lookahead + 16, 16)
        self.ring = LiburingRing(ring_depth, ring_id=0)

        # Per-slot, pre-allocated CUDA copy scratch (no hot-path allocation).
        self._streams = [torch.cuda.Stream() for _ in range(slot_count)]
        max_mapping_entries = max(1, layout.storage_block_size_factor)
        self._mapping_buffers = [
            np.empty((max_mapping_entries, 2), dtype=np.int64)
            for _ in range(slot_count)
        ]
        self._mapping_tensors = [
            torch.from_numpy(buffer) for buffer in self._mapping_buffers
        ]
        self._slot_block_ids = [
            np.asarray([slot_index], dtype=np.int64)
            for slot_index in range(slot_count)
        ]
        max_copy_entries = max(1, len(layout.gpu_tensors) * max_mapping_entries)
        self._batch_src_ptrs = [
            np.empty(max_copy_entries, dtype=np.int64) for _ in range(slot_count)
        ]
        self._batch_dst_ptrs = [
            np.empty(max_copy_entries, dtype=np.int64) for _ in range(slot_count)
        ]
        self._batch_sizes = [
            np.empty(max_copy_entries, dtype=np.int64) for _ in range(slot_count)
        ]
        self._batch_src_tensors = [
            torch.from_numpy(buffer) for buffer in self._batch_src_ptrs
        ]
        self._batch_dst_tensors = [
            torch.from_numpy(buffer) for buffer in self._batch_dst_ptrs
        ]
        self._batch_size_tensors = [
            torch.from_numpy(buffer) for buffer in self._batch_sizes
        ]

        self._incoming: "queue.Queue[_ReactorJob | object]" = queue.Queue()
        self._active: list[_ReactorJob] = []
        self._inflight: dict[int, _RingOp] = {}
        self._pending_copies: list[_CopyOp] = []
        self._next_user_data = 0
        self._stop = False
        self._closed = False
        self._submit_lock = threading.Lock()

        # Async-open pipeline state (reactor thread only).
        # Reads + writes in flight on the ring (the ``iodepth`` device budget).
        self._data_inflight = 0
        # Async ``openat`` ops in flight on the ring.
        self._open_inflight = 0
        # Opened fds awaiting a staging slot + read budget. Loads drained before
        # preloads so latency-critical reads win the device.
        self._ready_fds_load: collections.deque[_ReadyFd] = collections.deque()
        self._ready_fds_preload: collections.deque[_ReadyFd] = collections.deque()

        # Preload state — all accessed from the reactor thread only.
        # Hashes queued for preload reads but not yet started.
        self._preload_pending: collections.deque[bytes] = collections.deque()
        self._preload_pending_set: set[bytes] = set()
        # In-flight preload reads: hash -> user_data key in _inflight.
        self._preload_inflight_hashes: dict[bytes, int] = {}
        # Completed preload reads waiting for a job: hash -> slot_index.
        self._preload_slots: dict[bytes, int] = {}
        # Jobs waiting for a preload CQE: hash -> (job, file_index).
        self._preload_waiters: dict[bytes, tuple[_ReactorJob, int]] = {}

        self._worker = threading.Thread(
            target=self._run, name="py-kvcache-reactor", daemon=True
        )
        self._worker.start()
        logger.info(
            "IoReactor started: iodepth=%d staging_slots=%d payload_size=%d",
            self.iodepth,
            slot_count,
            payload_size,
        )

    # -- public API (called from the planner / handler thread) --------------

    def submit_job(self, job: _ReactorJob) -> None:
        with self._submit_lock:
            if self._closed:
                if not job.future_set:
                    job.future_set = True
                    job.future.set_exception(RuntimeError("reactor is shut down"))
                return
            self._incoming.put(job)

    def enqueue_preload(self, block_hashes: list[bytes]) -> None:
        """Called from the coordinator/handler thread to enqueue preload reads."""
        with self._submit_lock:
            if self._closed:
                return
            self._incoming.put(_PreloadRequest(block_hashes))

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

    # -- reactor thread ------------------------------------------------------

    def _has_work(self) -> bool:
        return bool(
            self._active
            or self._inflight
            or self._pending_copies
            or self._preload_pending
            or self._preload_slots
            or self._ready_fds_load
            or self._ready_fds_preload
        )

    def _run(self) -> None:
        while not (self._stop and not self._has_work()):
            self._drain_incoming(block=not self._has_work())
            if self._stop and not self._has_work():
                break
            self._pump_once()

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
            # Release slots held by done preloads with no pending job.
            for slot_index in self._preload_slots.values():
                self.staging_pool.release(slot_index)
            self._preload_slots.clear()
            self._preload_pending.clear()
            self._preload_pending_set.clear()
            # Fail any jobs waiting on preload CQEs.
            err = RuntimeError("reactor shutting down")
            for _h, (job, _fi) in list(self._preload_waiters.items()):
                self._file_terminal(job, ok=False, exc=err)
            self._preload_waiters.clear()
            return
        if isinstance(item, _PreloadRequest):
            for h in item.block_hashes:
                if (
                    h not in self._preload_pending_set
                    and h not in self._preload_inflight_hashes
                    and h not in self._preload_slots
                ):
                    self._preload_pending.append(h)
                    self._preload_pending_set.add(h)
            return
        # It's a _ReactorJob.
        job = item
        if not job.is_store:
            # Cut-off: pending (not-yet-started) preload reads for this job's
            # hashes are superseded — fall back to the normal read path.
            # Since _PreloadRequests arrive in _incoming before the job (FIFO),
            # any such hashes are already in _preload_pending_set here.
            for h in job.block_hashes:
                self._preload_pending_set.discard(h)
        self._active.append(job)

    def _pump_once(self) -> bool:
        made = False
        made = self._drain_cuda_copies() or made
        made = self._poll_ring_completions() or made
        made = self._schedule_work() or made
        self.ring.submit_pending()
        made = self._finish_jobs() or made
        if not made and self._has_work():
            time.sleep(0)
        return made

    # -- CUDA copy draining --------------------------------------------------

    def _drain_cuda_copies(self) -> bool:
        if not self._pending_copies:
            return False
        made = False
        for copy in list(self._pending_copies):
            if not copy.end_event.query():
                continue
            self._pending_copies.remove(copy)
            duration_ns = int(
                copy.start_event.elapsed_time(copy.end_event) * 1_000_000
            )
            copy.job.cuda_samples.append(
                (copy.start_ns, duration_ns, copy.nbytes, 0)
            )
            if copy.is_store:
                self._on_store_copy_done(copy)
            else:
                # load staging -> GPU complete: file finished, slot free.
                self.staging_pool.release(copy.slot_index)
                self._file_terminal(copy.job, ok=True)
            made = True
        return made

    def _on_store_copy_done(self, copy: _CopyOp) -> None:
        job = copy.job
        slot = self.staging_pool.slot(copy.slot_index)
        direct_view = self._direct_view(copy.slot_index)
        if direct_view is None:
            pack_start = now_ns()
            self.layout.pack_storage_slot_into(
                self.staging_tensors, copy.slot_index, slot.payload
            )
            job.pack_samples.append(
                (pack_start, now_ns() - pack_start, self.layout.storage_block_bytes)
            )
        final_path = self.file_mapper.get_file_name(job.block_hashes[copy.file_index])
        try:
            fd, temp_path = self.file_store.open_temp_write(final_path)
        except BaseException as exc:
            self.staging_pool.release(copy.slot_index)
            self._file_terminal(job, ok=False, exc=exc)
            return
        user_data = self._new_user_data()
        start_ns = now_ns()
        try:
            self.file_store.queue_write(
                self.ring,
                user_data=user_data,
                fd=fd,
                slot=slot,
                io_array=direct_view,
            )
        except BaseException as exc:
            self.file_store.cleanup_temp(fd=fd, temp_path=temp_path)
            self.staging_pool.release(copy.slot_index)
            self._file_terminal(job, ok=False, exc=exc)
            return
        self._data_inflight += 1
        self._inflight[user_data] = _RingOp(
            job=job,
            file_index=copy.file_index,
            slot_index=copy.slot_index,
            fd=fd,
            is_write=True,
            start_ns=start_ns,
            direct=direct_view is not None,
            op_kind="write",
            final_path=final_path,
            temp_path=temp_path,
        )

    # -- ring completion handling -------------------------------------------

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
            # else: close — fire-and-forget, result ignored.
        return True

    def _on_open_complete(self, op: _RingOp, result_fd: int) -> None:
        self._open_inflight -= 1
        op.path_buf = None  # kernel no longer reads the path; allow GC
        if result_fd < 0:
            exc = OSError(-result_fd, "io_uring openat failed")
            if op.preload_hash is not None:
                self._preload_inflight_hashes.pop(op.preload_hash, None)
                waiter = self._preload_waiters.pop(op.preload_hash, None)
                if waiter is not None:
                    job, _fi = waiter
                    self._file_terminal(job, ok=False, exc=exc)
                else:
                    logger.warning(
                        "py-kvcache preload openat failed (no waiter): %s", exc
                    )
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
            job.file_samples.append(
                (op.start_ns, duration_ns, self.layout.storage_block_bytes)
            )
            slot = self.staging_pool.slot(op.slot_index)
            if not op.direct:
                unpack_start = now_ns()
                self.layout.unpack_storage_slot(
                    slot.payload, self.staging_tensors, op.slot_index
                )
                job.unpack_samples.append(
                    (
                        unpack_start,
                        now_ns() - unpack_start,
                        self.layout.storage_block_bytes,
                    )
                )
            src_to_dst = self._build_mapping(
                op.slot_index,
                self._slot_block_ids[op.slot_index],
                self.layout.storage_block_size_factor,
                job.block_chunks[op.file_index],
                self.layout.gpu_block_size_factor,
            )
            copy = self._launch_swap_blocks(
                op.slot_index,
                self.staging_tensors,
                self.layout.gpu_tensors,
                src_to_dst,
                self.layout.storage_block_bytes,
                is_store=False,
                job=job,
                file_index=op.file_index,
            )
            self._pending_copies.append(copy)
        except BaseException as exc:
            self.staging_pool.release(op.slot_index)
            self._file_terminal(job, ok=False, exc=exc)
        finally:
            self._async_close(op.fd)

    def _on_preload_read_complete(self, op: _RingOp, result_nbytes: int) -> None:
        block_hash = op.preload_hash
        assert block_hash is not None
        self._preload_inflight_hashes.pop(block_hash, None)
        try:
            if result_nbytes < 0:
                raise OSError(-result_nbytes, "io_uring preload read failed")
            if result_nbytes != self.file_store.io_size:
                raise IOError(
                    f"io_uring preload read transferred {result_nbytes} bytes, "
                    f"expected {self.file_store.io_size}"
                )
            slot = self.staging_pool.slot(op.slot_index)
            if not op.direct:
                self.layout.unpack_storage_slot(
                    slot.payload, self.staging_tensors, op.slot_index
                )
            duration_ns = now_ns() - op.start_ns
            add_event(
                "py_kvcache.preload.read_complete",
                "kv_preload",
                op.start_ns,
                duration_ns,
                args={"nbytes": result_nbytes},
            )
            waiter = self._preload_waiters.pop(block_hash, None)
            if waiter is not None:
                job, file_index = waiter
                src_to_dst = self._build_mapping(
                    op.slot_index,
                    self._slot_block_ids[op.slot_index],
                    self.layout.storage_block_size_factor,
                    job.block_chunks[file_index],
                    self.layout.gpu_block_size_factor,
                )
                copy = self._launch_swap_blocks(
                    op.slot_index,
                    self.staging_tensors,
                    self.layout.gpu_tensors,
                    src_to_dst,
                    self.layout.storage_block_bytes,
                    is_store=False,
                    job=job,
                    file_index=file_index,
                )
                self._pending_copies.append(copy)
            elif self._stop:
                # Shutting down: no job will ever claim this preloaded slot.
                self.staging_pool.release(op.slot_index)
            else:
                self._preload_slots[block_hash] = op.slot_index
        except BaseException as exc:
            self.staging_pool.release(op.slot_index)
            waiter = self._preload_waiters.pop(block_hash, None)
            if waiter is not None:
                job, _fi = waiter
                self._file_terminal(job, ok=False, exc=exc)
            else:
                logger.warning(
                    "py-kvcache preload read failed (no waiter): %s", exc
                )
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
            job.file_samples.append(
                (op.start_ns, duration_ns, self.layout.storage_block_bytes)
            )
            self.staging_pool.release(op.slot_index)
            self._file_terminal(job, ok=True)
        except BaseException as exc:
            self.file_store.cleanup_temp(fd=op.fd, temp_path=op.temp_path)
            self.staging_pool.release(op.slot_index)
            self._file_terminal(job, ok=False, exc=exc)

    # -- scheduling ----------------------------------------------------------

    def _can_issue_read(self) -> bool:
        # ``iodepth`` bounds reads + writes in flight (the device queue depth).
        # A read also needs a free staging slot for its destination buffer.
        # In-flight CUDA copies are GPU-side, not storage I/O, so they do not
        # consume the device budget; the budget unit is released at CQE reap.
        return self._data_inflight < self.iodepth and self.staging_pool.free_count > 0

    def _can_open(self) -> bool:
        # Pre-opens are slot-free; bound only the count of in-flight ``openat``
        # ops plus opened-but-not-yet-read fds, so the reactor reads ahead
        # without holding unbounded file descriptors.
        primed = (
            self._open_inflight
            + len(self._ready_fds_load)
            + len(self._ready_fds_preload)
        )
        return primed < self.open_lookahead

    def _schedule_work(self) -> bool:
        made = False

        # 0. Drain already-opened fds into reads first — keep the device full.
        made = self._drain_ready_fds() or made

        # 1. Load jobs (latency-critical). Pre-open upcoming files; preload-done
        #    and preload-inflight files skip the disk read entirely.
        for job in self._active:
            if job.failed is not None or job.is_store:
                continue
            while job.failed is None and job.next_file_index < job.total_files:
                file_index = job.next_file_index
                h = job.block_hashes[file_index]
                if h in self._preload_slots:
                    self._schedule_preload_done(job, file_index)
                    made = True
                elif h in self._preload_inflight_hashes:
                    self._schedule_preload_inflight(job, file_index)
                    made = True
                else:
                    if not self._can_open():
                        break
                    self._submit_open_read(job, file_index)
                    made = True

        # 2. Preload opens (lower priority than loads, higher than stores).
        while self._preload_pending and self._can_open():
            h = self._preload_pending[0]
            if h not in self._preload_pending_set:
                # Hash was cut-off: removed from set but not yet from deque.
                self._preload_pending.popleft()
                continue
            if h in self._preload_inflight_hashes or h in self._preload_slots:
                self._preload_pending.popleft()
                self._preload_pending_set.discard(h)
                continue
            self._preload_pending.popleft()
            self._preload_pending_set.discard(h)
            self._submit_open_preload(h)
            made = True

        # 3. Store jobs (background): synchronous open + queued write.
        for job in self._active:
            if job.failed is not None or not job.is_store:
                continue
            while (
                job.failed is None
                and job.next_file_index < job.total_files
                and self._can_issue_read()
            ):
                if not self._schedule_one(job):
                    break
                made = True

        return made

    def _drain_ready_fds(self) -> bool:
        """Issue reads for opened fds while read budget + slots allow.

        Loads drain before preloads so the device queue favours latency-critical
        reads. A read reserves its staging slot only here — opens are slot-free.
        """
        made = False
        for dq in (self._ready_fds_load, self._ready_fds_preload):
            while dq and self._data_inflight < self.iodepth:
                slot = self.staging_pool.try_reserve()
                if slot is None:
                    return made
                ready = dq.popleft()
                self._submit_read_from_ready(ready, slot.index)
                made = True
        return made

    def _schedule_one(self, job: _ReactorJob) -> bool:
        # Stores only — reads flow through the async-open pipeline.
        slot = self.staging_pool.try_reserve()
        if slot is None:
            return False
        self._schedule_store_copy(job, job.next_file_index, slot.index)
        return True

    def _schedule_preload_done(self, job: _ReactorJob, file_index: int) -> None:
        """File already in staging from a completed preload read — skip disk I/O."""
        h = job.block_hashes[file_index]
        slot_index = self._preload_slots.pop(h)
        job.next_file_index += 1
        job.inflight_files += 1
        try:
            src_to_dst = self._build_mapping(
                slot_index,
                self._slot_block_ids[slot_index],
                self.layout.storage_block_size_factor,
                job.block_chunks[file_index],
                self.layout.gpu_block_size_factor,
            )
            copy = self._launch_swap_blocks(
                slot_index,
                self.staging_tensors,
                self.layout.gpu_tensors,
                src_to_dst,
                self.layout.storage_block_bytes,
                is_store=False,
                job=job,
                file_index=file_index,
            )
            self._pending_copies.append(copy)
        except BaseException as exc:
            self.staging_pool.release(slot_index)
            self._file_terminal(job, ok=False, exc=exc)

    def _schedule_preload_inflight(self, job: _ReactorJob, file_index: int) -> None:
        """Preload read is in-flight — register a waiter; no slot needed yet."""
        h = job.block_hashes[file_index]
        job.next_file_index += 1
        job.inflight_files += 1
        self._preload_waiters[h] = (job, file_index)

    def _submit_open_read(self, job: _ReactorJob, file_index: int) -> None:
        """Submit an async ``openat`` for a load file. The staging slot is
        reserved later, when the opened fd's read is issued (opens are slot-free
        so a slow open never ties up a buffer)."""
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
            direct=False,
            op_kind="open",
            path_buf=path_buf,
        )

    def _submit_open_preload(self, block_hash: bytes) -> None:
        """Submit an async ``openat`` for a preload (no job attached yet)."""
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
        self._preload_inflight_hashes[block_hash] = user_data
        self._inflight[user_data] = _RingOp(
            job=None,
            file_index=-1,
            slot_index=-1,
            fd=-1,
            is_write=False,
            start_ns=start_ns,
            direct=False,
            op_kind="open",
            preload_hash=block_hash,
            path_buf=path_buf,
        )

    def _submit_read_from_ready(self, ready: _ReadyFd, slot_index: int) -> None:
        """Queue the read for an already-opened fd into the reserved slot."""
        direct_view = self._direct_view(slot_index)
        slot = self.staging_pool.slot(slot_index)
        user_data = self._new_user_data()
        start_ns = now_ns()
        try:
            self.file_store.queue_read(
                self.ring,
                user_data=user_data,
                fd=ready.fd,
                slot=slot,
                io_array=direct_view,
            )
        except BaseException as exc:
            self._async_close(ready.fd)
            self.staging_pool.release(slot_index)
            if ready.preload_hash is not None:
                self._preload_inflight_hashes.pop(ready.preload_hash, None)
                waiter = self._preload_waiters.pop(ready.preload_hash, None)
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
            direct=direct_view is not None,
            op_kind="read",
            preload_hash=ready.preload_hash,
        )

    def _async_close(self, fd: int) -> None:
        """Close an fd via the ring, off the critical path. Result ignored, but
        the op is tracked in ``_inflight`` so shutdown drains it cleanly."""
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
            direct=False,
            op_kind="close",
        )

    def _schedule_store_copy(
        self, job: _ReactorJob, file_index: int, slot_index: int
    ) -> bool:
        job.next_file_index += 1
        job.inflight_files += 1
        try:
            src_to_dst = self._build_mapping(
                slot_index,
                job.block_chunks[file_index],
                self.layout.gpu_block_size_factor,
                self._slot_block_ids[slot_index],
                self.layout.storage_block_size_factor,
            )
            copy = self._launch_swap_blocks(
                slot_index,
                self.layout.gpu_tensors,
                self.staging_tensors,
                src_to_dst,
                self.layout.storage_block_bytes,
                is_store=True,
                job=job,
                file_index=file_index,
            )
        except BaseException as exc:
            self.staging_pool.release(slot_index)
            self._file_terminal(job, ok=False, exc=exc)
            return False
        self._pending_copies.append(copy)
        return True

    # -- CUDA copy launch ----------------------------------------------------

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

    def _launch_swap_blocks(
        self,
        slot_index: int,
        src_tensors: list[Any],
        dst_tensors: list[Any],
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
            start_event.record(stream)
            mapping_len = int(src_to_dst.shape[0])
            if mapping_len > 0 and ops is not None and hasattr(ops, "swap_blocks_batch"):
                mapping = self._mapping_buffers[slot_index][:mapping_len]
                src_ptrs = self._batch_src_ptrs[slot_index]
                dst_ptrs = self._batch_dst_ptrs[slot_index]
                sizes = self._batch_sizes[slot_index]
                offset = 0
                for src_tensor, dst_tensor, block_bytes in zip(
                    src_tensors,
                    dst_tensors,
                    self.layout.bytes_per_kernel_block,
                ):
                    end = offset + mapping_len
                    src_ptrs[offset:end] = (
                        src_tensor.data_ptr() + mapping[:, 0] * block_bytes
                    )
                    dst_ptrs[offset:end] = (
                        dst_tensor.data_ptr() + mapping[:, 1] * block_bytes
                    )
                    sizes[offset:end] = block_bytes
                    offset = end
                if offset > 0:
                    ops.swap_blocks_batch(
                        self._batch_src_tensors[slot_index][:offset],
                        self._batch_dst_tensors[slot_index][:offset],
                        self._batch_size_tensors[slot_index][:offset],
                    )
            else:
                for src_tensor, dst_tensor, block_bytes in zip(
                    src_tensors,
                    dst_tensors,
                    self.layout.bytes_per_kernel_block,
                ):
                    ops.swap_blocks(src_tensor, dst_tensor, block_bytes, src_to_dst)
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

    # -- job completion ------------------------------------------------------

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
            pack_samples=job.pack_samples,
            unpack_samples=job.unpack_samples,
            num_files=job.total_files,
            num_blocks=job.num_blocks,
        )

    # -- helpers -------------------------------------------------------------

    def _direct_view(self, slot_index: int) -> np.ndarray[Any, np.dtype[np.uint8]] | None:
        if self.file_store.io_size != self.layout.storage_block_bytes:
            return None
        return self.layout.direct_storage_slot_view(
            self.staging_tensors,
            slot_index,
            alignment=self.staging_pool.alignment,
        )

    def _new_user_data(self) -> int:
        self._next_user_data += 1
        return self._next_user_data


class TransferCoordinator:
    """Caller-thread planner that builds jobs and hands them to the reactor.

    The lock-touching / CPU planning work (splitting GPU block ids into per-file
    chunks, computing the first-group block offset) runs here, on the vLLM
    worker thread. The reactor then owns all I/O and CUDA work on one thread.
    """

    def __init__(
        self,
        *,
        config: SharedFileConfig,
        file_mapper: FileMapper,
        layout: ParsedKvLayout,
    ) -> None:
        self.layout = layout
        self.reactor = IoReactor(config=config, file_mapper=file_mapper, layout=layout)

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
        src_chunks = split_block_ids_for_files(
            self.layout, block_ids_of(src_spec), total
        )
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

    def submit_preload(self, block_hashes: list[bytes]) -> None:
        self.reactor.enqueue_preload(block_hashes)

    def shutdown(self) -> None:
        self.reactor.shutdown()


__all__ = ["IoReactor", "TransferCoordinator"]
