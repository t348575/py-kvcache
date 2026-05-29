from __future__ import annotations

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
class _RingOp:
    job: _ReactorJob
    file_index: int
    slot_index: int
    fd: int
    is_write: bool
    start_ns: int
    direct: bool
    final_path: str | None = None
    temp_path: str | None = None


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
        self.ring = LiburingRing(max(self.iodepth, 8), ring_id=0)

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
        return bool(self._active or self._inflight or self._pending_copies)

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
            return
        self._active.append(item)

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
        self._inflight[user_data] = _RingOp(
            job=job,
            file_index=copy.file_index,
            slot_index=copy.slot_index,
            fd=fd,
            is_write=True,
            start_ns=start_ns,
            direct=direct_view is not None,
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
            if op.is_write:
                self._on_write_complete(op, result)
            else:
                self._on_read_complete(op, result)
        return True

    def _on_read_complete(self, op: _RingOp, result_nbytes: int) -> None:
        job = op.job
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
            try:
                os.close(op.fd)
            except OSError:
                pass

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

    def _can_schedule(self) -> bool:
        # ``iodepth`` bounds in-flight ring/device ops (reads + writes) only.
        # In-flight CUDA copies are GPU-side, not storage I/O, so they must not
        # consume the device queue budget. Releasing a unit at CQE reap (rather
        # than at copy completion) lets ``_schedule_work`` refill the ring to
        # ``iodepth`` in the same pump; copies are bounded by staging slots.
        return len(self._inflight) < self.iodepth and self.staging_pool.free_count > 0

    def _schedule_work(self) -> bool:
        made = False
        # Loads first (latency-critical), then stores.
        for want_store in (False, True):
            for job in self._active:
                if job.failed is not None or job.is_store != want_store:
                    continue
                while (
                    job.failed is None
                    and job.next_file_index < job.total_files
                    and self._can_schedule()
                ):
                    if not self._schedule_one(job):
                        break
                    made = True
        return made

    def _schedule_one(self, job: _ReactorJob) -> bool:
        slot = self.staging_pool.try_reserve()
        if slot is None:
            return False
        file_index = job.next_file_index
        if job.is_store:
            ok = self._schedule_store_copy(job, file_index, slot.index)
        else:
            ok = self._schedule_read(job, file_index, slot.index)
        if not ok:
            return True  # a failure was recorded; treat as progress and move on
        return True

    def _schedule_read(self, job: _ReactorJob, file_index: int, slot_index: int) -> bool:
        job.next_file_index += 1
        job.inflight_files += 1
        final_path = self.file_mapper.get_file_name(job.block_hashes[file_index])
        try:
            fd = self.file_store.open_read(final_path)
        except BaseException as exc:
            self.staging_pool.release(slot_index)
            self._file_terminal(job, ok=False, exc=exc)
            return False
        direct_view = self._direct_view(slot_index)
        slot = self.staging_pool.slot(slot_index)
        user_data = self._new_user_data()
        start_ns = now_ns()
        try:
            self.file_store.queue_read(
                self.ring,
                user_data=user_data,
                fd=fd,
                slot=slot,
                io_array=direct_view,
            )
        except BaseException as exc:
            try:
                os.close(fd)
            except OSError:
                pass
            self.staging_pool.release(slot_index)
            self._file_terminal(job, ok=False, exc=exc)
            return False
        self._inflight[user_data] = _RingOp(
            job=job,
            file_index=file_index,
            slot_index=slot_index,
            fd=fd,
            is_write=False,
            start_ns=start_ns,
            direct=direct_view is not None,
        )
        return True

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
        add_event(
            "py_kvcache.transfer.submit",
            "kv_offload",
            start_ns,
            now_ns() - start_ns,
            tid=profile_tid,
            args={"job_id": job_id, "req_id": req_id, "direction": profile.direction},
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
        add_event(
            "py_kvcache.transfer.submit",
            "kv_offload",
            start_ns,
            now_ns() - start_ns,
            tid=profile_tid,
            args={"job_id": job_id, "req_id": req_id, "direction": profile.direction},
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

    def shutdown(self) -> None:
        self.reactor.shutdown()


__all__ = ["IoReactor", "TransferCoordinator"]
