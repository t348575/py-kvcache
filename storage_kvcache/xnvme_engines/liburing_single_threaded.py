from __future__ import annotations

import ctypes
import os
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
from vllm.logger import init_logger

from ..file_mapper import FileMapper
from ..liburing_file import (
    LiburingFileStore,
    LiburingRing,
    _AlignedIoBuffer,
    _TimedIoResult,
)
from ..vllm_xnvme import ParsedKvLayout
from .single_threaded import (
    SingleThreadedXNvmeOffloadingHandler,
    _CompletedTransferJob,
    _SubmittedTransferJob,
)

logger = init_logger("vllm.storage_kvcache.xnvme_engines.liburing_single_threaded")


@dataclass
class _QueuedLiburingFileIo:
    path: str
    fd: int
    user_data: int
    buffer: _AlignedIoBuffer
    start_ns: int
    write: bool
    temp_path: str | None = None
    final_path: str | None = None
    parent_path: str | None = None
    closed: bool = False


class LiburingSingleThreadedOffloadingHandler(SingleThreadedXNvmeOffloadingHandler):
    def __init__(
        self,
        *,
        layout: ParsedKvLayout,
        file_mapper: FileMapper,
        file_store: LiburingFileStore,
        staging_slot_capacity: int,
    ):
        self.layout = layout
        self.file_mapper = file_mapper
        self.file_store = file_store
        self.staging_slot_capacity = staging_slot_capacity
        self._io_parallelism = max(
            1, min(staging_slot_capacity, self.file_store.uring_depth)
        )
        self._ring = LiburingRing(self.file_store.uring_depth)
        self._next_user_data = 1
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
        self._active_jobs = {}
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

        self._inflight_file_ios = {}
        self._pending_cuda_copies = {}
        self._ready_writes = deque()
        self._round_robin_job_ids: deque[int] = deque()
        self._shutdown = False
        self._worker = threading.Thread(
            target=self._run_worker,
            name="kv-liburing-single",
            daemon=True,
        )
        self._worker.start()

        logger.info(
            "Liburing single-threaded KV backend initialized: uring_depth=%d effective_io_parallelism=%d staging_slot_capacity=%d",
            self.file_store.uring_depth,
            self._io_parallelism,
            staging_slot_capacity,
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
    ) -> _QueuedLiburingFileIo:
        fd = self.file_store.open_fd(
            path,
            write=write,
            create=create,
            truncate=truncate,
        )
        user_data = self._next_user_data
        self._next_user_data += 1
        start_ns = time.perf_counter_ns()
        try:
            self._ring.queue_rw(
                user_data=user_data,
                fd=fd,
                ptr=buffer.ptr,
                nbytes=self.file_store.io_size,
                write=write,
            )
        except BaseException:
            os.close(fd)
            raise
        return _QueuedLiburingFileIo(
            path=path,
            fd=fd,
            user_data=user_data,
            buffer=buffer,
            start_ns=start_ns,
            write=write,
            temp_path=temp_path,
            final_path=final_path,
            parent_path=parent_path,
        )

    def _pump_once(self) -> bool:
        made_progress = self._activate_pending_jobs()
        made_progress = self._drain_cuda_copies() or made_progress
        made_progress = self._poll_file_ios() or made_progress
        made_progress = self._submit_ready_writes() or made_progress
        made_progress = self._schedule_new_work() or made_progress
        self._ring.submit_pending()
        made_progress = self._finish_completed_jobs() or made_progress
        return made_progress

    def _poll_file_ios(self) -> bool:
        made_progress = False
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
            made_progress = True
        return made_progress

    def _poll_file_io(
        self, op: _QueuedLiburingFileIo
    ) -> _TimedIoResult | None:
        result_nbytes = self._ring.poll(op.user_data)
        if result_nbytes is None:
            return None
        if result_nbytes < 0:
            self._close_file_io(op)
            raise OSError(-result_nbytes, f"io_uring {'write' if op.write else 'read'} failed for {op.path}")
        if result_nbytes != self.file_store.io_size:
            self._close_file_io(op)
            op_name = "write" if op.write else "read"
            raise IOError(
                f"io_uring {op_name} transferred {result_nbytes} bytes for {op.path}, "
                f"expected {self.file_store.io_size}"
            )
        duration_ns = time.perf_counter_ns() - op.start_ns
        if op.write and self.file_store.config.sync_on_store:
            os.fsync(op.fd)
        self._close_file_io(op)
        return _TimedIoResult(start_ns=op.start_ns, duration_ns=duration_ns)

    def _close_file_io(self, op: _QueuedLiburingFileIo) -> None:
        if op.closed:
            return
        op.closed = True
        os.close(op.fd)

    def shutdown(self) -> None:
        self._submission_queue.put(None)
        self._wake_worker.set()
        self._worker.join()
        for _slot_index, (_job_id, _file_index, op) in list(
            self._inflight_file_ios.items()
        ):
            self._close_file_io(op)
        self._inflight_file_ios.clear()
        self._ring.close()
        self.file_store.free_buffers(self._io_buffers)
        self.file_store.close()


__all__ = ["LiburingSingleThreadedOffloadingHandler"]
