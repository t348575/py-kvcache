from __future__ import annotations

import ctypes
import os
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.v1.kv_offload.mediums import GPULoadStoreSpec
from vllm.v1.kv_offload.worker.worker import (
    OffloadingHandler,
    TransferResult,
    TransferSpec,
)
import xnvme.ctypes_bindings.api as xnvme

from simple_profiler import profile_scope, profiler

from ..file_mapper import FileMapper
from ..vllm_xnvme import (
    ParsedKvLayout,
    Sample,
    SharedStorageLoadStoreSpec,
    _build_block_mapping,
)
from ..xnvme_file import (
    XNvmeFileStore,
    _AlignedIoBuffer,
    _TimedIoResult,
    _xnvme_cmd_ctx_failed,
    _xnvme_cmd_ctx_result,
)

logger = init_logger("vllm.storage_kvcache.xnvme_engines.single_threaded")


@dataclass
class _InstrumentedTransferResult:
    result: TransferResult
    file_io_samples: list[Sample]
    cuda_copy_samples: list[Sample]


@dataclass
class _PendingCudaCopy:
    slot_index: int
    start_ns: int
    start_event: torch.cuda.Event
    end_event: torch.cuda.Event
    sample_num_bytes: int


@dataclass
class _QueuedFileIo:
    path: str
    handle: Any
    queue: Any
    ctx: Any
    buffer: _AlignedIoBuffer
    start_ns: int
    write: bool
    temp_path: str | None = None
    final_path: str | None = None
    parent_path: str | None = None
    closed: bool = False
    string_refs: list[Any] | None = None


@dataclass
class _ActiveTransferJob:
    job_id: int
    direction: str
    transfer_type: tuple[str, str]
    started: float
    meta: tuple[int, str, str, str]
    done: threading.Event
    block_hashes: list[bytes]
    blocks_per_file: list[np.ndarray]
    total_file_count: int
    next_file_index: int = 0
    file_io_samples: list[Sample] = field(default_factory=list)
    cuda_copy_samples: list[Sample] = field(default_factory=list)
    parent_paths: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class _SubmittedTransferJob:
    job_id: int
    spec: TransferSpec
    meta: tuple[int, str, str, str]
    done: threading.Event


@dataclass(frozen=True)
class _CompletedTransferJob:
    job_id: int
    item: _InstrumentedTransferResult | BaseException
    meta: tuple[int, str, str, str]


class SingleThreadedXNvmeOffloadingHandler(OffloadingHandler):
    def __init__(
        self,
        *,
        layout: ParsedKvLayout,
        file_mapper: FileMapper,
        file_store: XNvmeFileStore,
        staging_slot_capacity: int,
    ):
        self.layout = layout
        self.file_mapper = file_mapper
        self.file_store = file_store
        self.staging_slot_capacity = staging_slot_capacity
        self._io_parallelism = max(
            1, min(staging_slot_capacity, self.file_store.parallel_files)
        )
        self._staging_tensors = self.layout.allocate_staging_tensors(
            staging_slot_capacity
        )
        self._io_buffers = self.file_store.allocate_buffers(staging_slot_capacity)
        self._free_slots = list(range(staging_slot_capacity))
        self._swap_streams = [
            torch.cuda.Stream() for _ in range(staging_slot_capacity)
        ]
        self._mapping_buffers = [
            np.empty((self.layout.storage_block_size_factor, 2), dtype=np.int64)
            for _ in range(staging_slot_capacity)
        ]
        self._mapping_tensors = [
            torch.from_numpy(buffer) for buffer in self._mapping_buffers
        ]
        self._slot_block_ids = [
            np.asarray([slot_index], dtype=np.int64)
            for slot_index in range(staging_slot_capacity)
        ]

        self._pending_jobs: deque[_SubmittedTransferJob] = deque()
        self._active_jobs: dict[int, _ActiveTransferJob] = {}
        self._completion_cache: dict[int, _CompletedTransferJob] = {}
        self._profiled_jobs: set[int] = set()
        self._known_jobs: set[int] = set()
        self._job_done_events: dict[int, threading.Event] = {}
        self._submission_queue: queue.SimpleQueue[_SubmittedTransferJob | None] = (
            queue.SimpleQueue()
        )
        self._completion_queue: queue.SimpleQueue[_CompletedTransferJob] = (
            queue.SimpleQueue()
        )
        self._wake_worker = threading.Event()

        self._inflight_file_ios: dict[int, tuple[int, int, _QueuedFileIo]] = {}
        self._pending_cuda_copies: dict[int, tuple[int, int, _PendingCudaCopy]] = {}
        self._ready_writes: deque[tuple[int, int, int]] = deque()
        self._round_robin_job_ids: deque[int] = deque()
        self._shutdown = False
        self._worker = threading.Thread(
            target=self._run_worker,
            name="kv-xnvme-single",
            daemon=True,
        )
        self._worker.start()

    def transfer_async(
        self,
        job_id: int,
        spec: TransferSpec,
        profile_tid: str = "kv_transfer",
        req_id: str = "",
    ) -> bool:
        src_spec, dst_spec = spec
        if isinstance(src_spec, GPULoadStoreSpec) and isinstance(
            dst_spec, SharedStorageLoadStoreSpec
        ):
            direction = "gpu_to_storage"
            if profile_tid == "kv_transfer":
                profile_tid = "kv_store"
        elif isinstance(src_spec, SharedStorageLoadStoreSpec) and isinstance(
            dst_spec, GPULoadStoreSpec
        ):
            direction = "storage_to_gpu"
            if profile_tid == "kv_transfer":
                profile_tid = "kv_load"
        else:
            raise TypeError(
                f"unsupported transfer spec: {type(src_spec)!r} -> {type(dst_spec)!r}"
            )

        if job_id in self._known_jobs:
            raise ValueError(f"job {job_id} is already active")
        done = threading.Event()
        self._known_jobs.add(job_id)
        self._job_done_events[job_id] = done
        with profile_scope(
            "storage_kvcache.single.enqueue_transfer",
            "kv_offload",
            args={"job_id": job_id, "direction": direction},
        ):
            self._submission_queue.put(
                _SubmittedTransferJob(
                    job_id=job_id,
                    spec=spec,
                    meta=(time.perf_counter_ns(), profile_tid, req_id, direction),
                    done=done,
                )
            )
        self._wake_worker.set()
        return True

    def _run_worker(self) -> None:
        while True:
            self._drain_submission_queue()
            if self._shutdown:
                return
            has_work = bool(
                self._pending_jobs
                or self._active_jobs
                or self._inflight_file_ios
                or self._pending_cuda_copies
                or self._ready_writes
            )
            if not has_work:
                self._wake_worker.wait(timeout=0.00001)
                self._wake_worker.clear()
                continue
            made_progress = self._pump_once_locked()
            if not made_progress:
                time.sleep(0)

    def _drain_submission_queue(self) -> None:
        while True:
            try:
                item = self._submission_queue.get_nowait()
            except queue.Empty:
                return
            if item is None:
                self._shutdown = True
                return
            self._pending_jobs.append(item)

    def _has_job(self, job_id: int) -> bool:
        return (
            job_id in self._active_jobs
            or any(item.job_id == job_id for item in self._pending_jobs)
        )

    def _activate_pending_jobs(self) -> bool:
        made_progress = False
        while self._pending_jobs:
            item = self._pending_jobs.popleft()
            self._active_jobs[item.job_id] = self._create_active_job(item)
            self._round_robin_job_ids.append(item.job_id)
            made_progress = True
        return made_progress

    def _create_active_job(self, item: _SubmittedTransferJob) -> _ActiveTransferJob:
        src_spec, dst_spec = item.spec
        if isinstance(src_spec, GPULoadStoreSpec) and isinstance(
            dst_spec, SharedStorageLoadStoreSpec
        ):
            total_file_count = len(dst_spec.block_hashes)
            blocks_per_file = self._split_block_ids_for_files(
                src_spec.block_ids, total_file_count
            )
            return _ActiveTransferJob(
                job_id=item.job_id,
                direction="gpu_to_storage",
                transfer_type=(
                    GPULoadStoreSpec.medium(),
                    SharedStorageLoadStoreSpec.medium(),
                ),
                started=time.perf_counter(),
                meta=item.meta,
                done=item.done,
                block_hashes=dst_spec.block_hashes,
                blocks_per_file=blocks_per_file,
                total_file_count=total_file_count,
            )
        if isinstance(src_spec, SharedStorageLoadStoreSpec) and isinstance(
            dst_spec, GPULoadStoreSpec
        ):
            total_file_count = len(src_spec.block_hashes)
            blocks_per_file = self._split_block_ids_for_files(
                dst_spec.block_ids,
                total_file_count,
                self._get_first_group_block_index(dst_spec),
            )
            return _ActiveTransferJob(
                job_id=item.job_id,
                direction="storage_to_gpu",
                transfer_type=(
                    SharedStorageLoadStoreSpec.medium(),
                    GPULoadStoreSpec.medium(),
                ),
                started=time.perf_counter(),
                meta=item.meta,
                done=item.done,
                block_hashes=src_spec.block_hashes,
                blocks_per_file=blocks_per_file,
                total_file_count=total_file_count,
            )
        raise TypeError(
            f"unsupported transfer spec: {type(src_spec)!r} -> {type(dst_spec)!r}"
        )

    def _submit_file_io(
        self,
        *,
        path: str,
        buffer: _AlignedIoBuffer,
        write: bool,
        create: bool,
        truncate: bool,
        temp_path: str | None = None,
        final_path: str | None = None,
        parent_path: str | None = None,
    ) -> _QueuedFileIo:
        opts, string_refs, be, mem, sync, async_backend = self.file_store._make_opts(
            create=create, truncate=truncate, write=write
        )
        uri_ref = ctypes.create_string_buffer(path.encode("utf-8"))
        string_refs.append(uri_ref)
        open_start_ns = time.perf_counter_ns()
        handle = xnvme.xnvme_file_open(
            ctypes.cast(uri_ref, ctypes.POINTER(ctypes.c_char)), ctypes.byref(opts)
        )
        if profiler._active:
            profiler.add_event(
                "storage_kvcache.single.file_open",
                "fs",
                open_start_ns,
                time.perf_counter_ns() - open_start_ns,
                args={"write": write, "create": create, "truncate": truncate},
            )
        if not handle:
            raise RuntimeError(
                "xnvme_file_open() failed for "
                f"{path} with be={be!r}, mem={mem!r}, sync={sync!r}, async={async_backend!r}, "
                f"direct={bool(opts.direct)}, create={bool(opts.create)}, "
                f"truncate={bool(opts.truncate)}, rdonly={bool(opts.rdonly)}, "
                f"wronly={bool(opts.wronly)}"
            )

        queue = ctypes.POINTER(xnvme.xnvme_queue)()
        with profile_scope("storage_kvcache.single.queue_init", "fs"):
            queue_err = xnvme.xnvme_queue_init(handle, 1, 0, ctypes.byref(queue))
        if queue_err or not queue:
            xnvme.xnvme_file_close(handle)
            op = "write" if write else "read"
            raise IOError(f"xnvme_queue_init() failed for {op} of {path}: {queue_err}")

        with profile_scope("storage_kvcache.single.queue_get_ctx", "fs"):
            ctx = xnvme.xnvme_queue_get_cmd_ctx(queue)
        if not ctx:
            xnvme.xnvme_queue_term(queue)
            xnvme.xnvme_file_close(handle)
            raise IOError(f"xnvme_queue_get_cmd_ctx() failed for {path}")

        ptr = ctypes.cast(buffer.ptr, ctypes.POINTER(None))
        start_ns = time.perf_counter_ns()
        submit_start_ns = start_ns
        if write:
            io_err = xnvme.xnvme_file_pwrite(ctx, ptr, self.file_store.io_size, 0)
            op_name = "xnvme_file_pwrite"
        else:
            io_err = xnvme.xnvme_file_pread(ctx, ptr, self.file_store.io_size, 0)
            op_name = "xnvme_file_pread"
        if profiler._active:
            profiler.add_event(
                f"storage_kvcache.single.{op_name}_submit",
                "fs",
                submit_start_ns,
                time.perf_counter_ns() - submit_start_ns,
                args={"io_size": self.file_store.io_size},
            )
        if io_err:
            xnvme.xnvme_queue_put_cmd_ctx(queue, ctx)
            xnvme.xnvme_queue_term(queue)
            xnvme.xnvme_file_close(handle)
            raise IOError(f"{op_name}() submit failed for {path}: {io_err}")

        return _QueuedFileIo(
            path=path,
            handle=handle,
            queue=queue,
            ctx=ctx,
            buffer=buffer,
            start_ns=start_ns,
            write=write,
            temp_path=temp_path,
            final_path=final_path,
            parent_path=parent_path,
            string_refs=string_refs,
        )

    def _poll_file_io(self, op: _QueuedFileIo) -> _TimedIoResult | None:
        poll_start_ns = time.perf_counter_ns()
        if xnvme.xnvme_queue_get_outstanding(op.queue):
            poke_err = xnvme.xnvme_queue_poke(op.queue, 0)
            if poke_err < 0:
                self._close_file_io(op)
                raise IOError(f"xnvme_queue_poke() failed for {op.path}: {poke_err}")
            if xnvme.xnvme_queue_get_outstanding(op.queue):
                if profiler._active:
                    profiler.add_event(
                        "storage_kvcache.single.poll_file_io_pending",
                        "fs",
                        poll_start_ns,
                        time.perf_counter_ns() - poll_start_ns,
                    )
                return None

        result = op.ctx.contents
        if _xnvme_cmd_ctx_failed(result):
            self._close_file_io(op)
            op_name = "xnvme_file_pwrite" if op.write else "xnvme_file_pread"
            raise IOError(f"{op_name}() failed for {op.path}")

        result_nbytes = _xnvme_cmd_ctx_result(result)
        if result_nbytes != self.file_store.io_size:
            self._close_file_io(op)
            op_name = "xnvme_file_pwrite" if op.write else "xnvme_file_pread"
            raise IOError(
                f"{op_name}() transferred {result_nbytes} bytes for {op.path}, "
                f"expected {self.file_store.io_size}"
            )
        duration_ns = time.perf_counter_ns() - op.start_ns
        if op.write and self.file_store.config.sync_on_store:
            sync_start_ns = time.perf_counter_ns()
            sync_err = xnvme.xnvme_file_sync(op.handle)
            if profiler._active:
                profiler.add_event(
                    "storage_kvcache.single.file_sync",
                    "fs",
                    sync_start_ns,
                    time.perf_counter_ns() - sync_start_ns,
                )
            if sync_err:
                self._close_file_io(op)
                raise IOError(f"xnvme_file_sync() failed for {op.path}")
        self._close_file_io(op)
        if profiler._active:
            profiler.add_event(
                "storage_kvcache.single.poll_file_io_complete",
                "fs",
                poll_start_ns,
                time.perf_counter_ns() - poll_start_ns,
                args={"write": op.write},
            )
        return _TimedIoResult(start_ns=op.start_ns, duration_ns=duration_ns)

    def _close_file_io(self, op: _QueuedFileIo) -> None:
        if op.closed:
            return
        op.closed = True
        if op.ctx:
            with profile_scope("storage_kvcache.single.queue_put_ctx", "fs"):
                xnvme.xnvme_queue_put_cmd_ctx(op.queue, op.ctx)
        with profile_scope("storage_kvcache.single.queue_term", "fs"):
            xnvme.xnvme_queue_term(op.queue)
        with profile_scope("storage_kvcache.single.file_close", "fs"):
            xnvme.xnvme_file_close(op.handle)

    def _launch_swap_blocks(
        self,
        slot_index: int,
        src_tensors: list[torch.Tensor],
        dst_tensors: list[torch.Tensor],
        src_to_dst: torch.Tensor,
        sample_num_bytes: int,
    ) -> _PendingCudaCopy:
        sample_start_ns = time.perf_counter_ns()
        stream = self._swap_streams[slot_index]
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(stream):
            start_event.record(stream)
            for src_tensor, dst_tensor, block_bytes in zip(
                src_tensors,
                dst_tensors,
                self.layout.bytes_per_kernel_block,
            ):
                ops.swap_blocks(src_tensor, dst_tensor, block_bytes, src_to_dst)
            end_event.record(stream)
        return _PendingCudaCopy(
            slot_index=slot_index,
            start_ns=sample_start_ns,
            start_event=start_event,
            end_event=end_event,
            sample_num_bytes=sample_num_bytes,
        )

    @staticmethod
    def _complete_cuda_copy(
        pending: _PendingCudaCopy, cuda_copy_samples: list[Sample]
    ) -> None:
        pending.end_event.synchronize()
        duration_ns = int(
            pending.start_event.elapsed_time(pending.end_event) * 1_000_000
        )
        cuda_copy_samples.append(
            (pending.start_ns, duration_ns, pending.sample_num_bytes)
        )

    @staticmethod
    def _cuda_copy_done(pending: _PendingCudaCopy) -> bool:
        return bool(pending.end_event.query())

    def _build_block_mapping(
        self,
        slot_index: int,
        src_blocks: np.ndarray,
        src_block_size_factor: int,
        dst_blocks: np.ndarray,
        dst_block_size_factor: int,
    ) -> torch.Tensor:
        with profile_scope(
            "storage_kvcache.single.build_block_mapping",
            "kv_offload",
            args={"src_blocks": int(src_blocks.size), "dst_blocks": int(dst_blocks.size)},
        ):
            output = self._mapping_buffers[slot_index]
            mapping_length = _build_block_mapping(
                src_blocks, src_block_size_factor, dst_blocks, dst_block_size_factor, output
            )
            return self._mapping_tensors[slot_index][:mapping_length]

    def _split_block_ids_for_files(
        self, block_ids: np.ndarray, file_count: int, first_block_index: int = 0
    ) -> list[np.ndarray]:
        if file_count == 0:
            return []

        blocks_per_file = self.layout.gpu_blocks_per_storage_block
        first_block_skip = first_block_index % blocks_per_file
        first_size = min(len(block_ids), blocks_per_file - first_block_skip)
        chunks: list[np.ndarray] = []
        start = 0
        size = first_size
        for _ in range(file_count):
            end = min(start + size, len(block_ids))
            chunks.append(block_ids[start:end])
            start = end
            size = blocks_per_file

        if start != len(block_ids):
            raise ValueError("source and destination block counts do not line up")
        return chunks

    @staticmethod
    def _get_first_group_block_index(dst_spec: GPULoadStoreSpec) -> int:
        if len(dst_spec.group_sizes) != 1:
            raise ValueError(
                "xNVMe shared-storage offload currently supports exactly one KV cache group"
            )
        if dst_spec.block_indices is None:
            return 0
        if len(dst_spec.block_indices) != 1:
            raise ValueError(
                "xNVMe shared-storage offload currently supports exactly one KV cache group"
            )
        return int(dst_spec.block_indices[0])

    def _pump_once(self) -> bool:
        with profile_scope("storage_kvcache.single.pump_once", "kv_offload"):
            made_progress = self._activate_pending_jobs()
            made_progress = self._drain_cuda_copies() or made_progress
            made_progress = self._poll_file_ios() or made_progress
            made_progress = self._submit_ready_writes() or made_progress
            made_progress = self._schedule_new_work() or made_progress
            made_progress = self._finish_completed_jobs() or made_progress
            return made_progress

    def _drain_cuda_copies(self) -> bool:
        made_progress = False
        for slot_index, (job_id, file_index, pending) in list(
            self._pending_cuda_copies.items()
        ):
            if not self._cuda_copy_done(pending):
                continue
            job = self._active_jobs[job_id]
            self._complete_cuda_copy(pending, job.cuda_copy_samples)
            self._pending_cuda_copies.pop(slot_index)
            if job.direction == "storage_to_gpu":
                self._free_slots.append(slot_index)
            else:
                self._ready_writes.append((slot_index, job_id, file_index))
            made_progress = True
        return made_progress

    def _poll_file_ios(self) -> bool:
        for slot_index, (job_id, file_index, op) in list(self._inflight_file_ios.items()):
            result = self._poll_file_io(op)
            if result is None:
                continue
            self._inflight_file_ios.pop(slot_index)
            job = self._active_jobs[job_id]
            job.file_io_samples.append(
                (result.start_ns, result.duration_ns, self.layout.storage_block_bytes)
            )
            if job.direction == "storage_to_gpu":
                self._complete_read(slot_index, job, file_index, op)
            else:
                self._complete_write(slot_index, job, op)
            return True
        return False

    def _complete_read(
        self,
        slot_index: int,
        job: _ActiveTransferJob,
        file_index: int,
        op: _QueuedFileIo,
    ) -> None:
        with profile_scope(
            "storage_kvcache.single.unpack_storage_slot",
            "kv_offload",
            args={"payload_size": self.file_store.payload_size},
        ):
            self.layout.unpack_storage_slot(
                op.buffer.array[: self.file_store.payload_size],
                self._staging_tensors,
                slot_index,
            )
        src_to_dst = self._build_block_mapping(
            slot_index,
            self._slot_block_ids[slot_index],
            self.layout.storage_block_size_factor,
            job.blocks_per_file[file_index],
            self.layout.gpu_block_size_factor,
        )
        self._pending_cuda_copies[slot_index] = (
            job.job_id,
            file_index,
            self._launch_swap_blocks(
                slot_index,
                self._staging_tensors,
                self.layout.gpu_tensors,
                src_to_dst,
                self.layout.storage_block_bytes,
            ),
        )

    def _complete_write(
        self, slot_index: int, job: _ActiveTransferJob, op: _QueuedFileIo
    ) -> None:
        try:
            with profile_scope("storage_kvcache.single.link_temp", "fs"):
                os.link(op.temp_path, op.final_path)  # type: ignore[arg-type]
            if self.file_store.config.sync_on_store and op.parent_path:
                job.parent_paths.append(op.parent_path)
        except FileExistsError:
            pass
        finally:
            try:
                with profile_scope("storage_kvcache.single.unlink_temp", "fs"):
                    os.unlink(op.temp_path)  # type: ignore[arg-type]
            except FileNotFoundError:
                pass
        self._free_slots.append(slot_index)

    def _submit_ready_writes(self) -> bool:
        if (
            not self._ready_writes
            or len(self._inflight_file_ios) >= self._io_parallelism
        ):
            return False
        slot_index, job_id, file_index = self._ready_writes.popleft()
        job = self._active_jobs[job_id]
        self._submit_write(slot_index, job, file_index)
        return True

    def _schedule_new_work(self) -> bool:
        made_progress = False
        if not self._active_jobs:
            return made_progress

        idle_rounds = 0
        while idle_rounds < len(self._round_robin_job_ids):
            job_id = self._round_robin_job_ids.popleft()
            if job_id not in self._active_jobs:
                continue
            job = self._active_jobs[job_id]
            self._round_robin_job_ids.append(job_id)
            if job.next_file_index >= job.total_file_count:
                idle_rounds += 1
                continue
            if job.direction == "storage_to_gpu":
                if len(self._inflight_file_ios) >= self._io_parallelism:
                    idle_rounds += 1
                    continue
                if not self._free_slots:
                    idle_rounds += 1
                    continue
                slot_index = self._free_slots.pop()
                self._submit_read(slot_index, job)
                made_progress = True
                idle_rounds = 0
            else:
                if not self._free_slots:
                    idle_rounds += 1
                    continue
                slot_index = self._free_slots.pop()
                self._launch_store_copy(slot_index, job)
                made_progress = True
                idle_rounds = 0
        return made_progress

    def _submit_read(self, slot_index: int, job: _ActiveTransferJob) -> None:
        file_index = job.next_file_index
        job.next_file_index += 1
        self._inflight_file_ios[slot_index] = (
            job.job_id,
            file_index,
            self._submit_file_io(
                path=self.file_mapper.get_file_name(job.block_hashes[file_index]),
                buffer=self._io_buffers[slot_index],
                write=False,
                create=False,
                truncate=False,
            ),
        )

    def _launch_store_copy(self, slot_index: int, job: _ActiveTransferJob) -> None:
        file_index = job.next_file_index
        job.next_file_index += 1
        src_to_dst = self._build_block_mapping(
            slot_index,
            job.blocks_per_file[file_index],
            self.layout.gpu_block_size_factor,
            self._slot_block_ids[slot_index],
            self.layout.storage_block_size_factor,
        )
        self._pending_cuda_copies[slot_index] = (
            job.job_id,
            file_index,
            self._launch_swap_blocks(
                slot_index,
                self.layout.gpu_tensors,
                self._staging_tensors,
                src_to_dst,
                self.layout.storage_block_bytes,
            ),
        )

    def _submit_write(
        self, slot_index: int, job: _ActiveTransferJob, file_index: int
    ) -> None:
        final_path = self.file_mapper.get_file_name(job.block_hashes[file_index])
        start_ns = time.perf_counter_ns()
        exists_start_ns = time.perf_counter_ns()
        exists = os.path.exists(final_path)
        if profiler._active:
            profiler.add_event(
                "storage_kvcache.single.store_exists",
                "fs",
                exists_start_ns,
                time.perf_counter_ns() - exists_start_ns,
                args={"exists": exists},
            )
        if exists:
            job.file_io_samples.append((start_ns, time.perf_counter_ns() - start_ns, 0))
            self._free_slots.append(slot_index)
            return
        parent = os.path.dirname(final_path)
        with profile_scope("storage_kvcache.single.makedirs", "fs"):
            os.makedirs(parent, exist_ok=True)
        temp_path = self.file_store._temp_path(final_path)
        buffer = self._io_buffers[slot_index]
        with profile_scope(
            "storage_kvcache.single.zero_buffer",
            "fs",
            args={"io_size": self.file_store.io_size},
        ):
            buffer.array.fill(0)
        with profile_scope(
            "storage_kvcache.single.pack_storage_slot",
            "kv_offload",
            args={"payload_size": self.file_store.payload_size},
        ):
            self.layout.pack_storage_slot_into(
                self._staging_tensors,
                slot_index,
                buffer.array[: self.file_store.payload_size],
            )
        self._inflight_file_ios[slot_index] = (
            job.job_id,
            file_index,
            self._submit_file_io(
                path=temp_path,
                buffer=buffer,
                write=True,
                create=True,
                truncate=True,
                temp_path=temp_path,
                final_path=final_path,
                parent_path=parent,
            ),
        )

    def _finish_completed_jobs(self) -> bool:
        made_progress = False
        for job_id, job in list(self._active_jobs.items()):
            if job.next_file_index < job.total_file_count:
                continue
            if self._job_has_inflight_work(job_id):
                continue
            if job.parent_paths:
                with profile_scope(
                    "storage_kvcache.single.fsync_parent_dirs",
                    "fs",
                    args={"num_dirs": len(set(job.parent_paths))},
                ):
                    self.file_store.fsync_dirs(job.parent_paths)
            transfer_size = job.total_file_count * self.layout.storage_block_bytes
            completed = _InstrumentedTransferResult(
                result=TransferResult(
                    job_id=job_id,
                    success=True,
                    transfer_size=transfer_size,
                    transfer_time=time.perf_counter() - job.started,
                    transfer_type=job.transfer_type,
                ),
                file_io_samples=job.file_io_samples,
                cuda_copy_samples=job.cuda_copy_samples,
            )
            self._completion_queue.put(
                _CompletedTransferJob(job_id=job_id, item=completed, meta=job.meta)
            )
            job.done.set()
            self._active_jobs.pop(job_id)
            made_progress = True
        return made_progress

    def _job_has_inflight_work(self, job_id: int) -> bool:
        return (
            any(owner_job_id == job_id for owner_job_id, _, _ in self._inflight_file_ios.values())
            or any(owner_job_id == job_id for owner_job_id, _, _ in self._pending_cuda_copies.values())
            or any(owner_job_id == job_id for _, owner_job_id, _ in self._ready_writes)
        )

    def _emit_profile_events(
        self,
        job_id: int,
        instrumented_result: _InstrumentedTransferResult,
        end_ns: int,
        meta: tuple[int, str, str, str],
    ) -> None:
        if job_id in self._profiled_jobs:
            return

        if not getattr(profiler, "_active", False):
            logger.warning_once(
                "xNVMe KV profiling events are being generated but simple_profiler is inactive in this process"
            )
            return

        def merged_duration_ns(samples: list[Sample]) -> int:
            intervals = sorted(
                (start_ns, start_ns + duration_ns)
                for start_ns, duration_ns, _sample_num_bytes in samples
                if duration_ns > 0
            )
            if not intervals:
                return 0

            merged_start, merged_end = intervals[0]
            merged_duration = 0
            for start_ns, end_ns in intervals[1:]:
                if start_ns <= merged_end:
                    merged_end = max(merged_end, end_ns)
                else:
                    merged_duration += merged_end - merged_start
                    merged_start, merged_end = start_ns, end_ns
            return merged_duration + (merged_end - merged_start)

        result = instrumented_result.result
        wall_start_ns, profile_tid, req_id, direction = meta
        is_store = direction == "gpu_to_storage"
        file_io_samples = sorted(instrumented_result.file_io_samples)
        cuda_copy_samples = sorted(instrumented_result.cuda_copy_samples)
        transfer_size = result.transfer_size or 0
        file_io_wall_ns = merged_duration_ns(file_io_samples)
        cuda_copy_wall_ns = merged_duration_ns(cuda_copy_samples)
        job_args = {
            "req_id": req_id,
            "success": result.success,
            "num_bytes": transfer_size,
            "file_io_bw_GBps": round(
                (transfer_size / file_io_wall_ns) if file_io_wall_ns > 0 else 0.0,
                3,
            ),
            "cuda_copy_bw_GBps": round(
                (transfer_size / cuda_copy_wall_ns) if cuda_copy_wall_ns > 0 else 0.0,
                3,
            ),
        }
        profiler.add_event(
            name=f"xnvme_transfer({direction}, job={job_id})",
            category="xnvme_transfer",
            start_ns=wall_start_ns,
            duration_ns=end_ns - wall_start_ns,
            tid=profile_tid,
            args=job_args,
        )
        for sample_idx, (start_ns, duration_ns, sample_num_bytes) in enumerate(
            cuda_copy_samples
        ):
            profiler.add_event(
                name=(
                    f"cuda_staging({'gpu_to_cpu' if is_store else 'cpu_to_gpu'}, "
                    f"job={job_id}, sample={sample_idx})"
                ),
                category="cuda",
                start_ns=start_ns,
                duration_ns=duration_ns,
                tid=profile_tid,
                args={
                    "req_id": req_id,
                    "num_bytes": sample_num_bytes,
                    "bw_GBps": round(
                        (sample_num_bytes / duration_ns) if duration_ns > 0 else 0.0,
                        3,
                    ),
                },
            )
        for sample_idx, (start_ns, duration_ns, sample_num_bytes) in enumerate(
            file_io_samples
        ):
            profiler.add_event(
                name=(
                    f"file_{'write' if is_store else 'read'}"
                    f"(job={job_id}, sample={sample_idx})"
                ),
                category="fs",
                start_ns=start_ns,
                duration_ns=duration_ns,
                tid=profile_tid,
                args={
                    "req_id": req_id,
                    "num_bytes": sample_num_bytes,
                    "bw_GBps": round(
                        (sample_num_bytes / duration_ns) if duration_ns > 0 else 0.0,
                        3,
                    ),
                },
            )
        self._profiled_jobs.add(job_id)

    def get_finished(self) -> list[TransferResult]:
        poll_start_ns = time.perf_counter_ns()
        self._drain_completion_queue()
        results: list[TransferResult] = []
        completed_items = list(self._completion_cache.items())
        self._completion_cache.clear()
        if profiler._active:
            profiler.add_event(
                "storage_kvcache.single.get_finished_poll",
                "kv_offload",
                poll_start_ns,
                time.perf_counter_ns() - poll_start_ns,
                args={"num_finished": len(completed_items)},
            )
        for job_id, completed in completed_items:
            item = completed.item
            try:
                if isinstance(item, BaseException):
                    raise item
                self._emit_profile_events(
                    job_id, item, time.perf_counter_ns(), completed.meta
                )
                self._profiled_jobs.discard(job_id)
                results.append(item.result)
            except Exception:
                logger.exception("shared KV transfer job %s failed", job_id)
                self._profiled_jobs.discard(job_id)
                results.append(TransferResult(job_id=job_id, success=False))
            finally:
                self._known_jobs.discard(job_id)
                self._job_done_events.pop(job_id, None)
        return results

    def wait(self, job_ids: set[int]) -> None:
        events = [self._job_done_events[job_id] for job_id in job_ids if job_id in self._job_done_events]
        for event in events:
            event.wait()
        self._drain_completion_queue()
        for job_id in job_ids:
            completed = self._completion_cache.get(job_id)
            if completed is None:
                continue
            item = completed.item
            if isinstance(item, BaseException):
                raise item
            if item is not None:
                self._emit_profile_events(
                    job_id, item, time.perf_counter_ns(), completed.meta
                )

    def _drain_completion_queue(self) -> None:
        while True:
            try:
                completed = self._completion_queue.get_nowait()
            except queue.Empty:
                return
            self._completion_cache[completed.job_id] = completed

    def _pump_once_locked(self) -> bool:
        try:
            return self._pump_once()
        except BaseException as exc:
            failed_job_id = next(iter(self._active_jobs), None)
            if failed_job_id is None:
                raise
            job = self._active_jobs[failed_job_id]
            self._completion_queue.put(
                _CompletedTransferJob(
                    job_id=failed_job_id,
                    item=exc,
                    meta=job.meta,
                )
            )
            job.done.set()
            self._drop_job_state(failed_job_id)
            return True

    def _drop_job_state(self, job_id: int) -> None:
        self._active_jobs.pop(job_id, None)
        self._round_robin_job_ids = deque(
            active_job_id
            for active_job_id in self._round_robin_job_ids
            if active_job_id != job_id
        )
        for slot_index, (owner_job_id, _file_index, op) in list(
            self._inflight_file_ios.items()
        ):
            if owner_job_id == job_id:
                self._close_file_io(op)
                self._inflight_file_ios.pop(slot_index, None)
                self._free_slots.append(slot_index)
        for slot_index, (owner_job_id, _file_index, _pending) in list(
            self._pending_cuda_copies.items()
        ):
            if owner_job_id == job_id:
                self._pending_cuda_copies.pop(slot_index, None)
                self._free_slots.append(slot_index)
        kept_writes: deque[tuple[int, int, int]] = deque()
        for item in self._ready_writes:
            slot_index, owner_job_id, _file_index = item
            if owner_job_id == job_id:
                self._free_slots.append(slot_index)
            else:
                kept_writes.append(item)
        self._ready_writes = kept_writes

    def shutdown(self) -> None:
        self._submission_queue.put(None)
        self._wake_worker.set()
        self._worker.join()
        for _slot_index, (_job_id, _file_index, op) in list(
            self._inflight_file_ios.items()
        ):
            self._close_file_io(op)
        self._inflight_file_ios.clear()
        self.file_store.free_buffers(self._io_buffers)
        self.file_store.close()


__all__ = ["SingleThreadedXNvmeOffloadingHandler"]
