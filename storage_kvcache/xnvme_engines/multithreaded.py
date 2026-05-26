from __future__ import annotations

import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, wait
from dataclasses import dataclass, field
from functools import partial
from queue import SimpleQueue
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

from simple_profiler import profile_scope, profiler

from ..file_mapper import FileMapper
from ..liburing_file import LiburingFileStore
from ..preload_policy import (
    preload_submission_capacity,
    resolve_preload_target_io_depth,
)
from ..vllm_xnvme import (
    ParsedKvLayout,
    Sample,
    SharedStorageLoadStoreSpec,
    SharedStoragePreloadMessage,
    _build_block_mapping,
)
from ..xnvme_file import XNvmeFileStore, _TimedIoResult, _TimedStoreResult

logger = init_logger("vllm.storage_kvcache.xnvme_engines.multithreaded")


class _WorkerPool:
    _STOP = object()

    def __init__(self, *, max_workers: int, thread_name_prefix: str):
        self._queue: SimpleQueue[object] = SimpleQueue()
        self._lock = threading.Lock()
        self._shutdown = False
        self._threads = [
            threading.Thread(
                target=self._worker,
                name=f"{thread_name_prefix}_{index}",
            )
            for index in range(max_workers)
        ]
        for thread in self._threads:
            thread.start()

    def submit(self, fn: Any, /, *args: Any, **kwargs: Any) -> Future[Any]:
        future: Future[Any] = Future()
        with self._lock:
            if self._shutdown:
                raise RuntimeError("cannot schedule new futures after shutdown")
            self._queue.put((future, fn, args, kwargs))
        return future

    def shutdown(self, wait: bool = True) -> None:
        with self._lock:
            if self._shutdown:
                return
            self._shutdown = True
            for _thread in self._threads:
                self._queue.put(self._STOP)
        if wait:
            for thread in self._threads:
                thread.join()

    def _worker(self) -> None:
        while True:
            item = self._queue.get()
            if item is self._STOP:
                return
            future, fn, args, kwargs = item
            if not future.set_running_or_notify_cancel():
                continue
            try:
                future.set_result(fn(*args, **kwargs))
            except BaseException as exc:
                future.set_exception(exc)


@dataclass
class _InstrumentedTransferResult:
    result: TransferResult
    file_io_samples: list[Sample]
    cuda_copy_samples: list[Sample]


@dataclass
class _PreloadedBlockState:
    block_hash: bytes
    slot_index: int
    file_io_samples: list[Sample]
    io_depth_pending: bool = True
    future: Future[None] | None = None
    copy_future: Future[_PendingCudaCopy] | None = None
    dst_blocks: np.ndarray | None = None
    pending_copy: _PendingCudaCopy | None = None
    done_reading: bool = False
    done_copying: bool = False
    failed: BaseException | None = None
    condition: threading.Condition = field(default_factory=threading.Condition)


@dataclass
class _PendingCudaCopy:
    slot_index: int
    start_ns: int
    start_event: torch.cuda.Event
    end_event: torch.cuda.Event
    sample_num_bytes: int


class MultithreadedXNvmeOffloadingHandler(OffloadingHandler):
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
        self._load_parallelism = max(
            1, min(staging_slot_capacity, self.file_store.parallel_files)
        )
        self._executor = _WorkerPool(
            max_workers=self._load_parallelism, thread_name_prefix="kv-xnvme"
        )
        self._preload_executor = _WorkerPool(
            max_workers=self._load_parallelism, thread_name_prefix="kv-xnvme-preload"
        )
        self._pipeline_executor = _WorkerPool(
            max_workers=self._load_parallelism,
            thread_name_prefix="kv-xnvme-pipeline",
        )
        self._futures: dict[int, Future[_InstrumentedTransferResult]] = {}
        self._lock = threading.Lock()
        self._staging_cv = threading.Condition()
        self._available_staging_slots = set(range(staging_slot_capacity))
        self._io_queue_depth = max(
            1,
            min(
                staging_slot_capacity,
                getattr(self.file_store, "uring_depth", staging_slot_capacity),
            ),
        )
        self._io_queue_cv = threading.Condition()
        self._available_io_queue_slots = self._io_queue_depth
        self._preload_target_io_depth = resolve_preload_target_io_depth(
            getattr(self.file_store.config, "preload_target_io_depth", None),
            self._io_queue_depth,
        )
        self._pending_io_depth_reservations = 0
        self._transfer_jobs: dict[int, tuple[int, str, str, str]] = {}
        self._profiled_jobs: set[int] = set()
        self._preloaded_blocks: dict[bytes, _PreloadedBlockState] = {}
        self._staging_tensors = self.layout.allocate_staging_tensors(
            staging_slot_capacity
        )
        self._swap_streams = [
            torch.cuda.Stream() for _ in range(staging_slot_capacity)
        ]
        self._preloaded_copy_batch_size = min(16, staging_slot_capacity)
        self._mapping_buffers = [
            np.empty(
                (
                    self._preloaded_copy_batch_size
                    * self.layout.storage_block_size_factor,
                    2,
                ),
                dtype=np.int64,
            )
            for _ in range(staging_slot_capacity)
        ]
        self._mapping_tensors = [
            torch.from_numpy(buffer) for buffer in self._mapping_buffers
        ]
        self._slot_block_ids = [
            np.asarray([slot_index], dtype=np.int64)
            for slot_index in range(staging_slot_capacity)
        ]
        max_copy_entries = (
            len(self.layout.gpu_tensors)
            * self._preloaded_copy_batch_size
            * self.layout.storage_block_size_factor
        )
        self._batch_src_ptrs = [
            np.empty(max_copy_entries, dtype=np.int64)
            for _ in range(staging_slot_capacity)
        ]
        self._batch_dst_ptrs = [
            np.empty(max_copy_entries, dtype=np.int64)
            for _ in range(staging_slot_capacity)
        ]
        self._batch_sizes = [
            np.empty(max_copy_entries, dtype=np.int64)
            for _ in range(staging_slot_capacity)
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

    def _reserve_staging_slots(self, count: int) -> list[int]:
        if count > self.staging_slot_capacity:
            raise ValueError(
                f"cannot reserve {count} staging slots from capacity "
                f"{self.staging_slot_capacity}"
            )
        with self._staging_cv:
            while len(self._available_staging_slots) < count:
                self._staging_cv.wait()
            slots = sorted(self._available_staging_slots)[:count]
            self._available_staging_slots.difference_update(slots)
            return slots

    def _release_staging_slots(self, slot_indexes: list[int]) -> None:
        if not slot_indexes:
            return
        with self._staging_cv:
            self._available_staging_slots.update(slot_indexes)
            self._staging_cv.notify_all()

    def _try_reserve_staging_slot(self) -> int | None:
        with self._staging_cv:
            if not self._available_staging_slots:
                return None
            slot_index = min(self._available_staging_slots)
            self._available_staging_slots.remove(slot_index)
            return slot_index

    def _try_reserve_staging_slots(self, count: int) -> list[int] | None:
        if count <= 0:
            return []
        with self._staging_cv:
            if len(self._available_staging_slots) < count:
                return None
            slots = sorted(self._available_staging_slots)[:count]
            self._available_staging_slots.difference_update(slots)
            return slots

    def _reserve_available_staging_slots(self, max_count: int) -> list[int]:
        if max_count <= 0:
            return []
        with self._staging_cv:
            count = min(max_count, len(self._available_staging_slots))
            slots = sorted(self._available_staging_slots)[:count]
            self._available_staging_slots.difference_update(slots)
            return slots

    def _reserve_io_queue_slots(self, count: int, *, wait: bool) -> int:
        if count <= 0:
            return 0
        with self._io_queue_cv:
            while wait and self._available_io_queue_slots == 0:
                self._io_queue_cv.wait()
            reserved = min(count, self._available_io_queue_slots)
            self._available_io_queue_slots -= reserved
            return reserved

    def _release_io_queue_slots(self, count: int) -> None:
        if count <= 0:
            return
        with self._io_queue_cv:
            self._available_io_queue_slots += count
            if self._available_io_queue_slots > self._io_queue_depth:
                self._available_io_queue_slots = self._io_queue_depth
            self._io_queue_cv.notify_all()

    def _reserved_io_depth(self) -> int:
        with self._io_queue_cv:
            return self._io_queue_depth - self._available_io_queue_slots

    def _mark_preload_io_queued(self, states: list[_PreloadedBlockState]) -> None:
        released_pending = 0
        with self._lock:
            for state in states:
                if state.io_depth_pending:
                    state.io_depth_pending = False
                    released_pending += 1
            self._pending_io_depth_reservations = max(
                0, self._pending_io_depth_reservations - released_pending
            )

    def _release_pending_preload_io(
        self, states: list[_PreloadedBlockState]
    ) -> None:
        self._mark_preload_io_queued(states)

    def handle_worker_message(self, message: Any) -> bool:
        if not isinstance(message, SharedStoragePreloadMessage):
            return False
        start_ns = time.perf_counter_ns()
        supported = isinstance(self.file_store, LiburingFileStore)
        submitted = 0
        if supported:
            if self._can_submit_preload_batch(message.block_hashes):
                submitted = self._submit_preload_batch(message.block_hashes)
        if profiler._active:
            profiler.add_event(
                "storage_kvcache.engine.handle_preload_message",
                "kv_offload",
                start_ns,
                time.perf_counter_ns() - start_ns,
                args={
                    "supported": supported,
                    "submitted": submitted,
                    "num_blocks": len(message.block_hashes),
                },
            )
        return supported

    def _can_submit_preload_batch(self, block_hashes: tuple[bytes, ...]) -> bool:
        reserved_io_depth = self._reserved_io_depth()
        with self._lock:
            unique_missing = {
                block_hash
                for block_hash in block_hashes
                if block_hash not in self._preloaded_blocks
            }
            requested_blocks = len(unique_missing)
            if requested_blocks == 0:
                return True
            capacity = preload_submission_capacity(
                target_io_depth=self._preload_target_io_depth,
                reserved_io_depth=reserved_io_depth,
                pending_io_depth_reservations=self._pending_io_depth_reservations,
                requested_blocks=requested_blocks,
            )
        if capacity < requested_blocks:
            return False

        with self._staging_cv:
            return len(self._available_staging_slots) >= requested_blocks

    def _submit_preload_batch(self, block_hashes: tuple[bytes, ...]) -> int:
        submit_start_ns = time.perf_counter_ns()
        states: list[_PreloadedBlockState] = []
        duplicate_count = 0
        no_slot_count = 0
        target_depth_rejected_count = 0
        reserved_io_depth = self._reserved_io_depth()
        system_io_depth = reserved_io_depth

        with self._lock:
            pending_io_depth_reservations = self._pending_io_depth_reservations
            system_io_depth = reserved_io_depth + pending_io_depth_reservations
            unique_missing: list[bytes] = []
            seen_missing: set[bytes] = set()
            for block_hash in block_hashes:
                if block_hash in self._preloaded_blocks or block_hash in seen_missing:
                    duplicate_count += 1
                    continue
                seen_missing.add(block_hash)
                unique_missing.append(block_hash)
            capacity = preload_submission_capacity(
                target_io_depth=self._preload_target_io_depth,
                reserved_io_depth=reserved_io_depth,
                pending_io_depth_reservations=pending_io_depth_reservations,
                requested_blocks=len(unique_missing),
            )
            if capacity < len(unique_missing):
                target_depth_rejected_count = len(unique_missing)
                unique_missing = []
            slot_indexes = self._try_reserve_staging_slots(len(unique_missing))
            if slot_indexes is None:
                no_slot_count = len(unique_missing)
                unique_missing = []
                slot_indexes = []
            for block_hash, slot_index in zip(unique_missing, slot_indexes):
                state = _PreloadedBlockState(
                    block_hash=block_hash,
                    slot_index=slot_index,
                    file_io_samples=[],
                )
                self._preloaded_blocks[block_hash] = state
                self._pending_io_depth_reservations += 1
                states.append(state)

        if duplicate_count and profiler._active:
            profiler.add_event(
                "storage_kvcache.engine.preload_duplicate",
                "kv_offload",
                submit_start_ns,
                time.perf_counter_ns() - submit_start_ns,
                args={"num_blocks": duplicate_count},
            )
        if no_slot_count and profiler._active:
            profiler.add_event(
                "storage_kvcache.engine.preload_no_slot",
                "kv_offload",
                submit_start_ns,
                time.perf_counter_ns() - submit_start_ns,
                args={"num_blocks": no_slot_count},
            )
        if target_depth_rejected_count and profiler._active:
            profiler.add_event(
                "storage_kvcache.engine.preload_target_io_depth_rejected",
                "kv_offload",
                submit_start_ns,
                time.perf_counter_ns() - submit_start_ns,
                args={
                    "num_blocks": target_depth_rejected_count,
                    "target_io_depth": self._preload_target_io_depth,
                    "system_io_depth": system_io_depth,
                },
            )

        if not states:
            return 0

        try:
            with profile_scope(
                "storage_kvcache.engine.submit_preload_batch",
                "kv_offload",
                args={"num_blocks": len(states)},
            ):
                future = self._preload_executor.submit(self._run_preload_batch, states)
        except BaseException as exc:
            self._release_pending_preload_io(states)
            release_slots: list[int] = []
            with self._lock:
                for state in states:
                    if self._preloaded_blocks.get(state.block_hash) is state:
                        self._preloaded_blocks.pop(state.block_hash, None)
                        release_slots.append(state.slot_index)
            for state in states:
                with state.condition:
                    state.failed = exc
                    state.condition.notify_all()
            self._release_staging_slots(release_slots)
            raise

        for state in states:
            state.future = future
        if profiler._active:
            profiler.add_event(
                "storage_kvcache.engine.preload_submit",
                "kv_offload",
                submit_start_ns,
                time.perf_counter_ns() - submit_start_ns,
                args={
                    "num_blocks": len(states),
                    "target_io_depth": self._preload_target_io_depth,
                    "system_io_depth": system_io_depth,
                },
            )
        return len(states)

    def _run_preload_batch(self, states: list[_PreloadedBlockState]) -> None:
        start_ns = time.perf_counter_ns()
        block_hashes = [state.block_hash for state in states]
        slot_indexes = [state.slot_index for state in states]
        batch_file_io_samples: list[Sample] = []

        try:
            self._run_batched_io_uring_pipeline(
                block_hashes,
                None,
                batch_file_io_samples,
                [],
                slot_indexes=slot_indexes,
                preload_states=states,
                release_slots=False,
            )
            if profiler._active:
                profiler.add_event(
                    "storage_kvcache.engine.preload_batch_complete",
                    "kv_offload",
                    start_ns,
                    time.perf_counter_ns() - start_ns,
                    args={"num_blocks": len(states)},
                )
        except BaseException as exc:
            self._release_pending_preload_io(states)
            failed_states: list[_PreloadedBlockState] = []
            for state in states:
                with state.condition:
                    if not state.done_reading and not state.done_copying:
                        state.failed = exc
                        failed_states.append(state)
                    state.condition.notify_all()
            release_slots: list[int] = []
            with self._lock:
                for state in failed_states:
                    if self._preloaded_blocks.get(state.block_hash) is state:
                        self._preloaded_blocks.pop(state.block_hash, None)
                        release_slots.append(state.slot_index)
            self._release_staging_slots(release_slots)
            if profiler._active:
                profiler.add_event(
                    "storage_kvcache.engine.preload_batch_failed",
                    "kv_offload",
                    start_ns,
                    time.perf_counter_ns() - start_ns,
                    args={"num_blocks": len(states)},
                )
            raise

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

        with self._lock:
            if job_id in self._futures:
                raise ValueError(f"job {job_id} is already active")
            self._transfer_jobs[job_id] = (
                time.perf_counter_ns(),
                profile_tid,
                req_id,
                direction,
            )
            with profile_scope(
                "storage_kvcache.engine.submit_transfer_future",
                "kv_offload",
                args={"job_id": job_id, "direction": direction},
            ):
                self._futures[job_id] = self._executor.submit(
                    self._run_transfer, job_id, spec
                )
        return True

    def _run_transfer(
        self, job_id: int, spec: TransferSpec
    ) -> _InstrumentedTransferResult:
        started = time.perf_counter()
        src_spec, dst_spec = spec
        file_io_samples: list[Sample] = []
        cuda_copy_samples: list[Sample] = []
        if isinstance(src_spec, GPULoadStoreSpec) and isinstance(
            dst_spec, SharedStorageLoadStoreSpec
        ):
            with profile_scope(
                "storage_kvcache.engine.store_total",
                "kv_offload",
                args={"job_id": job_id, "num_files": len(dst_spec.block_hashes)},
            ):
                transfer_size = self._store(
                    src_spec, dst_spec, file_io_samples, cuda_copy_samples
                )
            transfer_type = (
                GPULoadStoreSpec.medium(),
                SharedStorageLoadStoreSpec.medium(),
            )
        elif isinstance(src_spec, SharedStorageLoadStoreSpec) and isinstance(
            dst_spec, GPULoadStoreSpec
        ):
            with profile_scope(
                "storage_kvcache.engine.load_total",
                "kv_offload",
                args={"job_id": job_id, "num_files": len(src_spec.block_hashes)},
            ):
                transfer_size = self._load(
                    src_spec, dst_spec, file_io_samples, cuda_copy_samples
                )
            transfer_type = (
                SharedStorageLoadStoreSpec.medium(),
                GPULoadStoreSpec.medium(),
            )
        else:
            raise TypeError(
                f"unsupported transfer spec: {type(src_spec)!r} -> {type(dst_spec)!r}"
            )

        return _InstrumentedTransferResult(
            result=TransferResult(
                job_id=job_id,
                success=True,
                transfer_size=transfer_size,
                transfer_time=time.perf_counter() - started,
                transfer_type=transfer_type,
            ),
            file_io_samples=file_io_samples,
            cuda_copy_samples=cuda_copy_samples,
        )

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
            mapping_len = int(src_to_dst.shape[0])
            if mapping_len > 0 and hasattr(ops, "swap_blocks_batch"):
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
            (
                pending.start_ns,
                duration_ns,
                pending.sample_num_bytes,
            )
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
            "storage_kvcache.engine.build_block_mapping",
            "kv_offload",
            args={"src_blocks": int(src_blocks.size), "dst_blocks": int(dst_blocks.size)},
        ):
            output = self._mapping_buffers[slot_index]
            mapping_length = _build_block_mapping(
                src_blocks, src_block_size_factor, dst_blocks, dst_block_size_factor, output
            )
            return self._mapping_tensors[slot_index][:mapping_length]

    def _launch_storage_to_gpu_copy(
        self,
        *,
        copy_slot_index: int,
        storage_block_index: int,
        storage_tensors: list[torch.Tensor],
        dst_blocks: np.ndarray,
    ) -> _PendingCudaCopy:
        src_to_dst = self._build_block_mapping(
            copy_slot_index,
            np.asarray([storage_block_index], dtype=np.int64),
            self.layout.storage_block_size_factor,
            dst_blocks,
            self.layout.gpu_block_size_factor,
        )
        return self._launch_swap_blocks(
            copy_slot_index,
            storage_tensors,
            self.layout.gpu_tensors,
            src_to_dst,
            self.layout.storage_block_bytes,
        )

    def _launch_preloaded_copy_batch(
        self, items: list[tuple[int, np.ndarray]]
    ) -> _PendingCudaCopy:
        if not items:
            raise ValueError("preloaded copy batch is empty")
        if len(items) > self._preloaded_copy_batch_size:
            raise ValueError("preloaded copy batch exceeds configured capacity")

        launch_start_ns = time.perf_counter_ns()
        copy_slot_index = items[0][0]
        src_blocks = np.asarray(
            [slot_index for slot_index, _dst in items], dtype=np.int64
        )
        if len(items) == 1:
            dst_blocks = items[0][1]
        else:
            dst_blocks = np.concatenate([dst for _slot_index, dst in items])
        src_to_dst = self._build_block_mapping(
            copy_slot_index,
            src_blocks,
            self.layout.storage_block_size_factor,
            dst_blocks,
            self.layout.gpu_block_size_factor,
        )
        pending = self._launch_swap_blocks(
            copy_slot_index,
            self._staging_tensors,
            self.layout.gpu_tensors,
            src_to_dst,
            len(items) * self.layout.storage_block_bytes,
        )
        if profiler._active:
            profiler.add_event(
                "storage_kvcache.engine.preloaded_copy_launch",
                "kv_offload",
                launch_start_ns,
                time.perf_counter_ns() - launch_start_ns,
                args={"slot_index": copy_slot_index, "num_blocks": len(items)},
            )
        return pending

    def _unpack_storage_slot_from_payload(
        self, slot_index: int, payload: np.ndarray
    ) -> None:
        with profile_scope(
            "storage_kvcache.engine.unpack_storage_slot",
            "kv_offload",
            args={"payload_size": len(payload)},
        ):
            self.layout.unpack_storage_slot(payload, self._staging_tensors, slot_index)

    def _read_storage_blocks_batched(
        self,
        block_hashes: list[bytes],
        staging_tensors: list[torch.Tensor],
        *,
        slot_indexes: list[int],
        on_read_complete: Any,
        file_io_samples: list[Sample],
        max_queued_reads: int,
        drain_ready: Any | None = None,
        wait_when_idle: Any | None = None,
        on_reads_queued: Any | None = None,
        read_source: str = "regular_load",
        owned_slot_indexes: set[int] | None = None,
        reuse_slots: bool = True,
    ) -> None:
        total_file_count = len(block_hashes)
        if total_file_count == 0:
            return
        if reuse_slots:
            if not slot_indexes and drain_ready is None:
                raise ValueError("slot_indexes does not cover requested reads")
        elif len(slot_indexes) < total_file_count:
            raise ValueError("slot_indexes does not cover requested reads")
        if owned_slot_indexes is None:
            owned_slot_indexes = set(slot_indexes)
        else:
            owned_slot_indexes.update(slot_indexes)

        queued_reads: dict[int, Any] = {}
        slot_to_file_index: dict[int, int] = {}
        queued_io_slots: dict[int, int] = {}
        free_slots = list(slot_indexes[:max_queued_reads]) if reuse_slots else []
        next_file_index = 0

        def unpack_storage_slot(slot_index: int, payload: np.ndarray) -> None:
            self.layout.unpack_storage_slot(payload, staging_tensors, slot_index)

        def submit_reads() -> None:
            nonlocal next_file_index
            if drain_ready is not None:
                drain_ready(free_slots)
                owned_slot_indexes.update(free_slots)
            if next_file_index >= total_file_count:
                return
            available = max_queued_reads - len(queued_reads)
            if available <= 0:
                return

            if reuse_slots:
                available = min(available, len(free_slots))
            else:
                available = min(available, total_file_count - next_file_index)
            wait_for_io_slot = not queued_reads
            reserved_io_slots = self._reserve_io_queue_slots(
                available, wait=wait_for_io_slot
            )
            if reserved_io_slots <= 0:
                return
            available = reserved_io_slots

            paths: list[str] = []
            read_slot_indexes: list[int] = []
            direct_targets: list[np.ndarray[Any, np.dtype[np.uint8]] | None] = []
            queued_file_indexes: list[int] = []
            while available > 0 and next_file_index < total_file_count:
                file_index = next_file_index
                if reuse_slots:
                    if not free_slots:
                        break
                    slot_index = free_slots.pop()
                else:
                    slot_index = slot_indexes[file_index]
                owned_slot_indexes.add(slot_index)
                block_hash = block_hashes[file_index]
                slot_to_file_index[slot_index] = file_index
                queued_io_slots[slot_index] = 1
                paths.append(self.file_mapper.get_file_name(block_hash))
                read_slot_indexes.append(slot_index)
                direct_target = None
                if self.file_store.io_size == self.layout.storage_block_bytes:
                    direct_target = self.layout.direct_storage_slot_view(
                        staging_tensors,
                        slot_index,
                        alignment=self.file_store.config.io_alignment,
                    )
                direct_targets.append(direct_target)
                queued_file_indexes.append(file_index)
                next_file_index += 1
                available -= 1

            unused_io_slots = reserved_io_slots - len(paths)
            if unused_io_slots:
                self._release_io_queue_slots(unused_io_slots)

            if paths:
                queued_reads.update(
                    self.file_store.queue_read_paths(
                        paths,
                        read_slot_indexes,
                        unpack_storage_slot,
                        direct_targets=direct_targets,
                        read_source=read_source,
                    )
                )
                if on_reads_queued is not None:
                    on_reads_queued(queued_file_indexes)

        try:
            submit_reads()
            while queued_reads or next_file_index < total_file_count:
                if drain_ready is not None:
                    drain_ready(free_slots)
                submit_reads()

                completed_read = None
                if queued_reads:
                    completed_read = self.file_store.poll_queued_read(queued_reads)

                if completed_read is None:
                    if not queued_reads and wait_when_idle is not None:
                        wait_when_idle(free_slots)
                        owned_slot_indexes.update(free_slots)
                    else:
                        time.sleep(0)
                    continue

                slot_index, result = completed_read
                self._release_io_queue_slots(queued_io_slots.pop(slot_index, 0))
                file_io_samples.append(
                    (
                        result.start_ns,
                        result.duration_ns,
                        self.layout.storage_block_bytes,
                        getattr(result, "read_source", read_source),
                    )
                )
                file_index = slot_to_file_index.pop(slot_index)
                on_read_complete(slot_index, file_index)
                if reuse_slots and drain_ready is None:
                    free_slots.append(slot_index)
        finally:
            self._release_io_queue_slots(sum(queued_io_slots.values()))
            self.file_store.close_queued_reads(queued_reads)

    def _run_batched_io_uring_pipeline(
        self,
        block_hashes: list[bytes],
        dst_blocks_per_file: list[np.ndarray] | None,
        file_io_samples: list[Sample],
        cuda_copy_samples: list[Sample],
        *,
        slot_indexes: list[int],
        preload_states: list[_PreloadedBlockState] | None = None,
        extra_drain_ready: Any | None = None,
        allow_dynamic_slots: bool = False,
        release_slots: bool = True,
    ) -> int:
        total_file_count = len(block_hashes)
        if total_file_count == 0:
            return 0
        if preload_states is None and dst_blocks_per_file is None:
            raise ValueError("dst_blocks_per_file is required for regular loads")
        if preload_states is not None and len(preload_states) != total_file_count:
            raise ValueError("preload states do not match block hashes")

        slot_capacity = self._io_queue_depth if allow_dynamic_slots else len(slot_indexes)
        max_queued_reads = max(
            1,
            min(
                slot_capacity,
                total_file_count,
                self._io_queue_depth,
            ),
        )
        pending_copies: dict[int, _PendingCudaCopy] = {}
        owned_slot_indexes = set(slot_indexes)

        def drain_ready_copies(
            free_slots: list[int], *, force: bool = False
        ) -> None:
            if preload_states is None:
                for slot_index, pending in list(pending_copies.items()):
                    if force or self._cuda_copy_done(pending):
                        self._complete_cuda_copy(pending, cuda_copy_samples)
                        pending_copies.pop(slot_index)
                        free_slots.append(slot_index)

            if extra_drain_ready is not None:
                extra_drain_ready(free_slots, force=force)

        def on_read_complete(slot_index: int, file_index: int) -> None:
            if preload_states is not None:
                state = preload_states[file_index]
                sample = file_io_samples[-1]
                with state.condition:
                    state.file_io_samples.append(sample)
                    state.done_reading = True
                    state.condition.notify_all()
                return

            assert dst_blocks_per_file is not None
            pending_copies[slot_index] = self._launch_storage_to_gpu_copy(
                copy_slot_index=slot_index,
                storage_block_index=slot_index,
                storage_tensors=self._staging_tensors,
                dst_blocks=dst_blocks_per_file[file_index],
            )

        try:
            self._read_storage_blocks_batched(
                block_hashes,
                self._staging_tensors,
                slot_indexes=slot_indexes,
                on_read_complete=on_read_complete,
                file_io_samples=file_io_samples,
                max_queued_reads=max_queued_reads,
                drain_ready=drain_ready_copies,
                wait_when_idle=partial(drain_ready_copies, force=True),
                on_reads_queued=(
                    lambda file_indexes: self._mark_preload_io_queued(
                        [preload_states[file_index] for file_index in file_indexes]
                    )
                    if preload_states is not None
                    else None
                ),
                read_source="preload" if preload_states is not None else "regular_load",
                owned_slot_indexes=owned_slot_indexes,
                reuse_slots=preload_states is None,
            )
            while pending_copies:
                released_slots: list[int] = []
                drain_ready_copies(released_slots, force=True)
                owned_slot_indexes.update(released_slots)
        finally:
            if release_slots:
                self._release_staging_slots(sorted(owned_slot_indexes))

        return total_file_count * self.layout.storage_block_bytes

    def _load_slot(
        self,
        slot_index: int,
        block_hash: bytes,
        dst_blocks: np.ndarray,
    ) -> tuple[int, _TimedIoResult, _PendingCudaCopy]:
        with profile_scope("storage_kvcache.engine.file_name", "kv_offload"):
            file_name = self.file_mapper.get_file_name(block_hash)
        with profile_scope("storage_kvcache.engine.read_slot_file", "kv_offload"):
            result = self.file_store.read_path_into(
                file_name,
                slot_index,
                self._unpack_storage_slot_from_payload,
            )
        src_to_dst = self._build_block_mapping(
            slot_index,
            self._slot_block_ids[slot_index],
            self.layout.storage_block_size_factor,
            dst_blocks,
            self.layout.gpu_block_size_factor,
        )
        pending_copy = self._launch_swap_blocks(
            slot_index,
            self._staging_tensors,
            self.layout.gpu_tensors,
            src_to_dst,
            self.layout.storage_block_bytes,
        )
        return slot_index, result, pending_copy

    def _store_slot(
        self,
        slot_index: int,
        block_hash: bytes,
    ) -> tuple[int, _TimedStoreResult]:
        with profile_scope("storage_kvcache.engine.file_name", "kv_offload"):
            file_name = self.file_mapper.get_file_name(block_hash)
        with profile_scope("storage_kvcache.engine.store_slot_file", "kv_offload"):
            result = self.file_store.store_slot_if_missing(
                file_name,
                slot_index,
                partial(self.layout.pack_storage_slot_into, self._staging_tensors),
            )
        return slot_index, result

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

    def _store(
        self,
        src_spec: GPULoadStoreSpec,
        dst_spec: SharedStorageLoadStoreSpec,
        file_io_samples: list[Sample],
        cuda_copy_samples: list[Sample],
    ) -> int:
        total_file_count = len(dst_spec.block_hashes)
        if total_file_count == 0:
            return 0

        with profile_scope(
            "storage_kvcache.engine.split_store_block_ids",
            "kv_offload",
            args={"num_blocks": int(src_spec.block_ids.size), "num_files": total_file_count},
        ):
            src_blocks_per_file = self._split_block_ids_for_files(
                src_spec.block_ids, total_file_count
            )
        slots = self._reserve_staging_slots(
            min(self.staging_slot_capacity, total_file_count)
        )
        free_slots = list(slots)
        inflight: dict[Future[tuple[int, _TimedStoreResult]], int] = {}
        pending_copies: dict[int, tuple[_PendingCudaCopy, int]] = {}
        next_file_index = 0
        parent_paths: list[str] = []

        def launch_next_copy(slot_index: int) -> None:
            nonlocal next_file_index
            if next_file_index >= total_file_count:
                return
            file_index = next_file_index
            src_blocks = src_blocks_per_file[next_file_index]
            src_to_dst = self._build_block_mapping(
                slot_index,
                src_blocks,
                self.layout.gpu_block_size_factor,
                self._slot_block_ids[slot_index],
                self.layout.storage_block_size_factor,
            )
            pending_copies[slot_index] = (
                self._launch_swap_blocks(
                    slot_index,
                    self.layout.gpu_tensors,
                    self._staging_tensors,
                    src_to_dst,
                    self.layout.storage_block_bytes,
                ),
                file_index,
            )
            next_file_index += 1

        def submit_store(slot_index: int, file_index: int) -> None:
            with profile_scope(
                "storage_kvcache.engine.submit_store_slot",
                "kv_offload",
                args={"file_index": file_index},
            ):
                future = self._pipeline_executor.submit(
                    self._store_slot,
                    slot_index,
                    dst_spec.block_hashes[file_index],
                )
            inflight[future] = slot_index

        def drain_ready_copies(*, force: bool = False) -> None:
            for slot_index, (pending, file_index) in list(pending_copies.items()):
                if force or self._cuda_copy_done(pending):
                    self._complete_cuda_copy(pending, cuda_copy_samples)
                    pending_copies.pop(slot_index)
                    submit_store(slot_index, file_index)

        try:
            for slot_index in free_slots:
                launch_next_copy(slot_index)

            while inflight or pending_copies:
                drain_ready_copies()
                if not inflight:
                    drain_ready_copies(force=True)
                    continue
                done, _pending = wait(
                    inflight, timeout=0.001, return_when=FIRST_COMPLETED
                )
                if not done:
                    continue
                for future in done:
                    inflight.pop(future)
                    slot_index, item = future.result()
                    file_io_samples.append(
                        (
                            item.start_ns,
                            item.duration_ns,
                            self.layout.storage_block_bytes,
                        )
                    )
                    if item.parent_path:
                        parent_paths.append(item.parent_path)
                    launch_next_copy(slot_index)
        finally:
            self._release_staging_slots(slots)

        if parent_paths:
            with profile_scope(
                "storage_kvcache.engine.fsync_parent_dirs",
                "fs",
                args={"num_dirs": len(set(parent_paths))},
            ):
                self.file_store.fsync_dirs(parent_paths)

        return total_file_count * self.layout.storage_block_bytes

    def _load(
        self,
        src_spec: SharedStorageLoadStoreSpec,
        dst_spec: GPULoadStoreSpec,
        file_io_samples: list[Sample],
        cuda_copy_samples: list[Sample],
    ) -> int:
        if all(
            hasattr(self.file_store, name)
            for name in ("queue_read_paths", "poll_queued_read", "close_queued_reads")
        ):
            with self._lock:
                has_preloaded_blocks = any(
                    block_hash in self._preloaded_blocks
                    for block_hash in src_spec.block_hashes
                )
            if has_preloaded_blocks:
                return self._load_with_partial_preloads(
                    src_spec, dst_spec, file_io_samples, cuda_copy_samples
                )
            return self._load_batched_io_uring(
                src_spec, dst_spec, file_io_samples, cuda_copy_samples
            )

        total_file_count = len(src_spec.block_hashes)
        logger.debug("Loading %d blocks", total_file_count)
        if total_file_count == 0:
            return 0

        with profile_scope(
            "storage_kvcache.engine.split_load_block_ids",
            "kv_offload",
            args={"num_blocks": int(dst_spec.block_ids.size), "num_files": total_file_count},
        ):
            dst_blocks_per_file = self._split_block_ids_for_files(
                dst_spec.block_ids,
                total_file_count,
                self._get_first_group_block_index(dst_spec),
            )
        slots = self._reserve_staging_slots(
            min(self.staging_slot_capacity, total_file_count)
        )
        free_slots = list(slots)
        inflight: dict[Future[tuple[int, _TimedIoResult, _PendingCudaCopy]], int] = {}
        pending_copies: dict[int, _PendingCudaCopy] = {}
        next_file_index = 0

        def submit_next(slot_index: int) -> None:
            nonlocal next_file_index
            if next_file_index >= total_file_count:
                return
            with profile_scope(
                "storage_kvcache.engine.submit_load_slot",
                "kv_offload",
                args={"file_index": next_file_index},
            ):
                future = self._pipeline_executor.submit(
                    self._load_slot,
                    slot_index,
                    src_spec.block_hashes[next_file_index],
                    dst_blocks_per_file[next_file_index],
                )
            inflight[future] = slot_index
            next_file_index += 1

        def drain_ready_copies(*, force: bool = False) -> None:
            for slot_index, pending in list(pending_copies.items()):
                if force or self._cuda_copy_done(pending):
                    self._complete_cuda_copy(pending, cuda_copy_samples)
                    pending_copies.pop(slot_index)
                    submit_next(slot_index)

        try:
            for slot_index in free_slots:
                submit_next(slot_index)

            while inflight or pending_copies:
                drain_ready_copies()
                if inflight:
                    done, _pending = wait(
                        inflight, timeout=0.001, return_when=FIRST_COMPLETED
                    )
                else:
                    done = set()
                    if pending_copies:
                        drain_ready_copies(force=True)
                if not done:
                    continue
                for future in done:
                    inflight.pop(future)
                    slot_index, result, pending_copy = future.result()
                    file_io_samples.append(
                        (
                            result.start_ns,
                            result.duration_ns,
                            self.layout.storage_block_bytes,
                            "regular_load",
                        )
                    )
                    pending_copies[slot_index] = pending_copy
        finally:
            self._release_staging_slots(slots)

        return total_file_count * self.layout.storage_block_bytes

    def _take_preloaded_block_states(
        self, block_hashes: list[bytes]
    ) -> dict[int, _PreloadedBlockState]:
        with self._lock:
            return {
                file_index: state
                for file_index, block_hash in enumerate(block_hashes)
                if (state := self._preloaded_blocks.pop(block_hash, None)) is not None
            }

    def _copy_preloaded_blocks(
        self,
        states_by_file_index: dict[int, _PreloadedBlockState],
        dst_blocks_per_file: list[np.ndarray],
        file_io_samples: list[Sample],
        cuda_copy_samples: list[Sample],
    ) -> None:
        if not states_by_file_index:
            return

        wait_start_ns = time.perf_counter_ns()
        pending_states = list(states_by_file_index.values())
        release_slots = {state.slot_index for state in pending_states}

        def finish_state(state: _PreloadedBlockState) -> None:
            pending_states.remove(state)
            file_io_samples.extend(state.file_io_samples)
            if state.slot_index in release_slots:
                release_slots.remove(state.slot_index)
                self._release_staging_slots([state.slot_index])

        launch_futures: dict[Future[_PendingCudaCopy], list[_PreloadedBlockState]] = {}
        pending_copy_batches: list[
            tuple[_PendingCudaCopy, list[_PreloadedBlockState]]
        ] = []

        def submit_ready_copy_batches() -> bool:
            ready_states: list[_PreloadedBlockState] = []
            for state in pending_states:
                with state.condition:
                    if state.failed is not None:
                        raise state.failed
                    if (
                        state.done_reading
                        and state.dst_blocks is not None
                        and state.copy_future is None
                        and state.pending_copy is None
                        and not state.done_copying
                    ):
                        ready_states.append(state)

            submitted = False
            batch_size = self._preloaded_copy_batch_size
            for start in range(0, len(ready_states), batch_size):
                batch_states = ready_states[start : start + batch_size]
                items = [
                    (state.slot_index, state.dst_blocks)
                    for state in batch_states
                    if state.dst_blocks is not None
                ]
                if not items:
                    continue
                future = self._pipeline_executor.submit(
                    self._launch_preloaded_copy_batch, items
                )
                for state in batch_states:
                    with state.condition:
                        state.copy_future = future
                launch_futures[future] = batch_states
                submitted = True
            return submitted

        def drain_copy_launches() -> bool:
            made_progress = False
            for future, batch_states in list(launch_futures.items()):
                if not future.done():
                    continue
                try:
                    pending = future.result()
                except BaseException as exc:
                    for state in batch_states:
                        with state.condition:
                            state.failed = exc
                            state.copy_future = None
                            state.condition.notify_all()
                    raise
                launch_futures.pop(future, None)
                for state in batch_states:
                    with state.condition:
                        state.copy_future = None
                        state.pending_copy = pending
                        state.condition.notify_all()
                pending_copy_batches.append((pending, batch_states))
                made_progress = True
            return made_progress

        def drain_pending_copy_batches(*, force: bool = False) -> bool:
            made_progress = False
            for pending, batch_states in list(pending_copy_batches):
                if not force and not self._cuda_copy_done(pending):
                    continue
                self._complete_cuda_copy(pending, cuda_copy_samples)
                pending_copy_batches[:] = [
                    item for item in pending_copy_batches if item[0] is not pending
                ]
                for state in list(batch_states):
                    with state.condition:
                        state.pending_copy = None
                        state.done_copying = True
                        state.condition.notify_all()
                    if state in pending_states:
                        finish_state(state)
                made_progress = True
            return made_progress

        try:
            for file_index, state in states_by_file_index.items():
                with state.condition:
                    state.dst_blocks = dst_blocks_per_file[file_index]
                    state.condition.notify_all()

            while pending_states:
                waiting_for_reads = False
                for state in list(pending_states):
                    with state.condition:
                        if state.failed is not None:
                            raise state.failed
                        if not state.done_reading:
                            waiting_for_reads = True

                made_progress = submit_ready_copy_batches()
                made_progress = drain_copy_launches() or made_progress
                made_progress = drain_pending_copy_batches() or made_progress

                if not pending_states:
                    break
                if made_progress:
                    continue

                if not waiting_for_reads and not launch_futures:
                    drain_pending_copy_batches(force=True)
                    continue

                # Keep letting preload IO completions make more blocks launchable;
                # only force-sync once every readable copy has been issued.
                time.sleep(0.001)
        finally:
            self._release_staging_slots(list(release_slots))

        if profiler._active:
            profiler.add_event(
                "storage_kvcache.engine.wait_preloaded_copies",
                "kv_offload",
                wait_start_ns,
                time.perf_counter_ns() - wait_start_ns,
                args={"num_blocks": len(states_by_file_index)},
            )

    def _load_regular_blocks_while_copying_preloads(
        self,
        block_hashes: list[bytes],
        dst_blocks_per_file: list[np.ndarray],
        missing_file_indexes: list[int],
        states_by_file_index: dict[int, _PreloadedBlockState],
        file_io_samples: list[Sample],
        cuda_copy_samples: list[Sample],
    ) -> None:
        wait_start_ns = time.perf_counter_ns()
        pending_states = list(states_by_file_index.values())
        release_slots = {state.slot_index for state in pending_states}
        launch_futures: dict[Future[_PendingCudaCopy], list[_PreloadedBlockState]] = {}
        pending_copy_batches: list[
            tuple[_PendingCudaCopy, list[_PreloadedBlockState]]
        ] = []

        def finish_state(
            state: _PreloadedBlockState, free_slots: list[int] | None
        ) -> None:
            pending_states.remove(state)
            file_io_samples.extend(state.file_io_samples)
            if state.slot_index not in release_slots:
                return
            release_slots.remove(state.slot_index)
            if free_slots is None:
                self._release_staging_slots([state.slot_index])
            else:
                free_slots.append(state.slot_index)

        def submit_ready_copy_batches() -> bool:
            ready_states: list[_PreloadedBlockState] = []
            for state in pending_states:
                with state.condition:
                    if state.failed is not None:
                        raise state.failed
                    if (
                        state.done_reading
                        and state.dst_blocks is not None
                        and state.copy_future is None
                        and state.pending_copy is None
                        and not state.done_copying
                    ):
                        ready_states.append(state)

            submitted = False
            batch_size = self._preloaded_copy_batch_size
            for start in range(0, len(ready_states), batch_size):
                batch_states = ready_states[start : start + batch_size]
                items = [
                    (state.slot_index, state.dst_blocks)
                    for state in batch_states
                    if state.dst_blocks is not None
                ]
                if not items:
                    continue
                future = self._pipeline_executor.submit(
                    self._launch_preloaded_copy_batch, items
                )
                for state in batch_states:
                    with state.condition:
                        state.copy_future = future
                launch_futures[future] = batch_states
                submitted = True
            return submitted

        def drain_copy_launches() -> bool:
            made_progress = False
            for future, batch_states in list(launch_futures.items()):
                if not future.done():
                    continue
                try:
                    pending = future.result()
                except BaseException as exc:
                    for state in batch_states:
                        with state.condition:
                            state.failed = exc
                            state.copy_future = None
                            state.condition.notify_all()
                    raise
                launch_futures.pop(future, None)
                for state in batch_states:
                    with state.condition:
                        state.copy_future = None
                        state.pending_copy = pending
                        state.condition.notify_all()
                pending_copy_batches.append((pending, batch_states))
                made_progress = True
            return made_progress

        def drain_pending_copy_batches(
            free_slots: list[int] | None, *, force: bool = False
        ) -> bool:
            made_progress = False
            for pending, batch_states in list(pending_copy_batches):
                if not force and not self._cuda_copy_done(pending):
                    continue
                self._complete_cuda_copy(pending, cuda_copy_samples)
                pending_copy_batches[:] = [
                    item for item in pending_copy_batches if item[0] is not pending
                ]
                for state in list(batch_states):
                    with state.condition:
                        state.pending_copy = None
                        state.done_copying = True
                        state.condition.notify_all()
                    if state in pending_states:
                        finish_state(state, free_slots)
                made_progress = True
            return made_progress

        def drain_preloaded_copies(
            free_slots: list[int] | None, *, force: bool = False
        ) -> bool:
            made_progress = submit_ready_copy_batches()
            made_progress = drain_copy_launches() or made_progress
            made_progress = (
                drain_pending_copy_batches(free_slots, force=force) or made_progress
            )
            return made_progress

        def wait_for_preloaded_copies() -> None:
            while pending_states:
                waiting_for_reads = False
                for state in list(pending_states):
                    with state.condition:
                        if state.failed is not None:
                            raise state.failed
                        if not state.done_reading:
                            waiting_for_reads = True

                made_progress = drain_preloaded_copies(None)
                if not pending_states:
                    break
                if made_progress:
                    continue
                if not waiting_for_reads and not launch_futures:
                    drain_pending_copy_batches(None, force=True)
                    continue
                time.sleep(0.001)

        try:
            self._load_batched_io_uring_indexes(
                block_hashes,
                dst_blocks_per_file,
                missing_file_indexes,
                file_io_samples,
                cuda_copy_samples,
                extra_drain_ready=drain_preloaded_copies,
                reserve_available_only=True,
            )
            wait_for_preloaded_copies()
        finally:
            self._release_staging_slots(sorted(release_slots))

        if profiler._active:
            profiler.add_event(
                "storage_kvcache.engine.wait_preloaded_copies",
                "kv_offload",
                wait_start_ns,
                time.perf_counter_ns() - wait_start_ns,
                args={
                    "num_blocks": len(states_by_file_index),
                    "interleaved_regular_loads": True,
                },
            )

    def _load_with_partial_preloads(
        self,
        src_spec: SharedStorageLoadStoreSpec,
        dst_spec: GPULoadStoreSpec,
        file_io_samples: list[Sample],
        cuda_copy_samples: list[Sample],
    ) -> int:
        total_file_count = len(src_spec.block_hashes)
        if total_file_count == 0:
            return 0

        dst_blocks_per_file = self._split_block_ids_for_files(
            dst_spec.block_ids,
            total_file_count,
            self._get_first_group_block_index(dst_spec),
        )
        states_by_file_index = self._take_preloaded_block_states(src_spec.block_hashes)
        missing_file_indexes = [
            file_index
            for file_index in range(total_file_count)
            if file_index not in states_by_file_index
        ]
        start_ns = time.perf_counter_ns()
        if profiler._active:
            profiler.add_event(
                "storage_kvcache.engine.load_partial_preloads",
                "kv_offload",
                start_ns,
                0,
                args={
                    "num_files": total_file_count,
                    "num_preloaded": len(states_by_file_index),
                    "num_missing": len(missing_file_indexes),
                },
            )

        for file_index, state in states_by_file_index.items():
            with state.condition:
                state.dst_blocks = dst_blocks_per_file[file_index]
                state.condition.notify_all()

        if missing_file_indexes:
            self._load_regular_blocks_while_copying_preloads(
                src_spec.block_hashes,
                dst_blocks_per_file,
                missing_file_indexes,
                states_by_file_index,
                file_io_samples,
                cuda_copy_samples,
            )
        else:
            self._copy_preloaded_blocks(
                states_by_file_index,
                dst_blocks_per_file,
                file_io_samples,
                cuda_copy_samples,
            )

        if profiler._active:
            profiler.add_event(
                "storage_kvcache.engine.load_partial_preloads_complete",
                "kv_offload",
                start_ns,
                time.perf_counter_ns() - start_ns,
                args={
                    "num_files": total_file_count,
                    "num_preloaded": len(states_by_file_index),
                    "num_missing": len(missing_file_indexes),
                },
            )

        return total_file_count * self.layout.storage_block_bytes

    def _load_batched_io_uring(
        self,
        src_spec: SharedStorageLoadStoreSpec,
        dst_spec: GPULoadStoreSpec,
        file_io_samples: list[Sample],
        cuda_copy_samples: list[Sample],
    ) -> int:
        total_file_count = len(src_spec.block_hashes)
        logger.debug("Loading %d blocks with batched io_uring", total_file_count)
        if total_file_count == 0:
            return 0

        dst_blocks_per_file = self._split_block_ids_for_files(
            dst_spec.block_ids,
            total_file_count,
            self._get_first_group_block_index(dst_spec),
        )
        return self._load_batched_io_uring_indexes(
            src_spec.block_hashes,
            dst_blocks_per_file,
            list(range(total_file_count)),
            file_io_samples,
            cuda_copy_samples,
        )

    def _load_batched_io_uring_indexes(
        self,
        block_hashes: list[bytes],
        dst_blocks_per_file: list[np.ndarray],
        file_indexes: list[int],
        file_io_samples: list[Sample],
        cuda_copy_samples: list[Sample],
        *,
        extra_drain_ready: Any | None = None,
        reserve_available_only: bool = False,
    ) -> int:
        total_file_count = len(file_indexes)
        if total_file_count == 0:
            return 0
        selected_block_hashes = [
            block_hashes[file_index] for file_index in file_indexes
        ]
        selected_dst_blocks = [
            dst_blocks_per_file[file_index] for file_index in file_indexes
        ]
        max_slots = min(self._io_queue_depth, total_file_count)
        if reserve_available_only:
            slots = self._reserve_available_staging_slots(max_slots)
        else:
            slots = self._reserve_staging_slots(max_slots)
        return self._run_batched_io_uring_pipeline(
            selected_block_hashes,
            selected_dst_blocks,
            file_io_samples,
            cuda_copy_samples,
            slot_indexes=slots,
            extra_drain_ready=extra_drain_ready,
            allow_dynamic_slots=reserve_available_only,
        )

    def _emit_profile_events(
        self, job_id: int, instrumented_result: _InstrumentedTransferResult, end_ns: int
    ) -> None:
        if job_id in self._profiled_jobs:
            return

        if not getattr(profiler, "_active", False):
            logger.warning_once(
                "xNVMe KV profiling events are being generated but simple_profiler is inactive in this process"
            )
            return

        job_meta = self._transfer_jobs.get(job_id)
        if job_meta is None:
            return

        def merged_duration_ns(samples: list[Sample]) -> int:
            intervals = sorted(
                (sample[0], sample[0] + sample[1])
                for sample in samples
                if sample[1] > 0
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
        wall_start_ns, profile_tid, req_id, direction = job_meta
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
        for sample_idx, sample in enumerate(file_io_samples):
            start_ns, duration_ns, sample_num_bytes = sample[:3]
            file_args = {
                "req_id": req_id,
                "num_bytes": sample_num_bytes,
                "bw_GBps": round(
                    (sample_num_bytes / duration_ns) if duration_ns > 0 else 0.0,
                    3,
                ),
            }
            if not is_store:
                file_args["read_source"] = (
                    sample[3] if len(sample) > 3 else "regular_load"
                )
            profiler.add_event(
                name=(
                    f"file_{'write' if is_store else 'read'}"
                    f"(job={job_id}, sample={sample_idx})"
                ),
                category="fs",
                start_ns=start_ns,
                duration_ns=duration_ns,
                tid=profile_tid,
                args=file_args,
            )
        self._profiled_jobs.add(job_id)

    def get_finished(self) -> list[TransferResult]:
        poll_start_ns = time.perf_counter_ns()
        with self._lock:
            finished_items = [
                (job_id, future)
                for job_id, future in self._futures.items()
                if future.done()
            ]
        if profiler._active:
            profiler.add_event(
                "storage_kvcache.engine.get_finished_poll",
                "kv_offload",
                poll_start_ns,
                time.perf_counter_ns() - poll_start_ns,
                args={"num_finished": len(finished_items)},
            )

        results: list[TransferResult] = []
        for job_id, future in finished_items:
            try:
                instrumented_result = future.result()
                result = instrumented_result.result
                end_ns = time.perf_counter_ns()
                self._emit_profile_events(job_id, instrumented_result, end_ns)
                self._transfer_jobs.pop(job_id, None)
                self._profiled_jobs.discard(job_id)
                results.append(result)
            except Exception:
                logger.exception("shared KV transfer job %s failed", job_id)
                self._transfer_jobs.pop(job_id, None)
                self._profiled_jobs.discard(job_id)
                results.append(TransferResult(job_id=job_id, success=False))
            finally:
                with self._lock:
                    self._futures.pop(job_id, None)
        return results

    def wait(self, job_ids: set[int]) -> None:
        with self._lock:
            futures = {
                job_id: self._futures[job_id]
                for job_id in job_ids
                if job_id in self._futures
            }
        for job_id, future in futures.items():
            instrumented_result = future.result()
            self._emit_profile_events(
                job_id, instrumented_result, time.perf_counter_ns()
            )

    def shutdown(self) -> None:
        self._preload_executor.shutdown(wait=True)
        self._executor.shutdown(wait=True)
        self._pipeline_executor.shutdown(wait=True)
        self.file_store.close()


__all__ = ["MultithreadedXNvmeOffloadingHandler"]
