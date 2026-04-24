from __future__ import annotations

import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from functools import partial

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

from simple_profiler import profiler

from ..file_mapper import FileMapper
from ..vllm_xnvme import (
    ParsedKvLayout,
    Sample,
    SharedStorageLoadStoreSpec,
    _build_block_mapping,
)
from ..xnvme_file import XNvmeFileStore, _TimedIoResult, _TimedStoreResult

logger = init_logger("vllm.storage_kvcache.xnvme_engines.multithreaded")


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
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="kv-xnvme"
        )
        self._pipeline_executor = ThreadPoolExecutor(
            max_workers=self._load_parallelism,
            thread_name_prefix="kv-xnvme-pipeline",
        )
        self._futures: dict[int, Future[_InstrumentedTransferResult]] = {}
        self._lock = threading.Lock()
        self._transfer_jobs: dict[int, tuple[int, str, str, str]] = {}
        self._profiled_jobs: set[int] = set()
        self._staging_tensors = self.layout.allocate_staging_tensors(
            staging_slot_capacity
        )
        self._swap_streams = [
            torch.cuda.Stream() for _ in range(staging_slot_capacity)
        ]
        self._mapping_buffers = [
            np.empty((self.layout.storage_block_size_factor, 2), dtype=np.int64)
            for _ in range(staging_slot_capacity)
        ]
        self._mapping_tensors = [torch.from_numpy(buffer) for buffer in self._mapping_buffers]
        self._slot_block_ids = [
            np.asarray([slot_index], dtype=np.int64)
            for slot_index in range(staging_slot_capacity)
        ]

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
        output = self._mapping_buffers[slot_index]
        mapping_length = _build_block_mapping(
            src_blocks, src_block_size_factor, dst_blocks, dst_block_size_factor, output
        )
        return self._mapping_tensors[slot_index][:mapping_length]

    def _unpack_storage_slot_from_payload(
        self, slot_index: int, payload: np.ndarray
    ) -> None:
        self.layout.unpack_storage_slot(payload, self._staging_tensors, slot_index)

    def _load_slot(
        self,
        slot_index: int,
        block_hash: bytes,
        dst_blocks: np.ndarray,
    ) -> tuple[int, _TimedIoResult, _PendingCudaCopy]:
        result = self.file_store.read_path_into(
            self.file_mapper.get_file_name(block_hash),
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
        result = self.file_store.store_slot_if_missing(
            self.file_mapper.get_file_name(block_hash),
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

        src_blocks_per_file = self._split_block_ids_for_files(
            src_spec.block_ids, total_file_count
        )
        free_slots = list(range(min(self.staging_slot_capacity, total_file_count)))
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

        for slot_index in free_slots:
            launch_next_copy(slot_index)

        while inflight or pending_copies:
            drain_ready_copies()
            if not inflight:
                drain_ready_copies(force=True)
                continue
            done, _pending = wait(inflight, timeout=0.001, return_when=FIRST_COMPLETED)
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

        if parent_paths:
            self.file_store.fsync_dirs(parent_paths)

        return total_file_count * self.layout.storage_block_bytes

    def _load(
        self,
        src_spec: SharedStorageLoadStoreSpec,
        dst_spec: GPULoadStoreSpec,
        file_io_samples: list[Sample],
        cuda_copy_samples: list[Sample],
    ) -> int:
        total_file_count = len(src_spec.block_hashes)
        logger.debug("Loading %d blocks", total_file_count)
        if total_file_count == 0:
            return 0

        dst_blocks_per_file = self._split_block_ids_for_files(
            dst_spec.block_ids,
            total_file_count,
            self._get_first_group_block_index(dst_spec),
        )
        free_slots = list(range(min(self.staging_slot_capacity, total_file_count)))
        inflight: dict[Future[tuple[int, _TimedIoResult, _PendingCudaCopy]], int] = {}
        pending_copies: dict[int, _PendingCudaCopy] = {}
        next_file_index = 0

        def submit_next(slot_index: int) -> None:
            nonlocal next_file_index
            if next_file_index >= total_file_count:
                return
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
                    )
                )
                pending_copies[slot_index] = pending_copy

        return total_file_count * self.layout.storage_block_bytes

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
        with self._lock:
            finished_items = [
                (job_id, future)
                for job_id, future in self._futures.items()
                if future.done()
            ]

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
        self._executor.shutdown(wait=True)
        self._pipeline_executor.shutdown(wait=True)
        self.file_store.close()


__all__ = ["MultithreadedXNvmeOffloadingHandler"]
