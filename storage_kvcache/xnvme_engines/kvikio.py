from __future__ import annotations

import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from vllm.logger import init_logger
from vllm.v1.kv_offload.mediums import GPULoadStoreSpec
from vllm.v1.kv_offload.worker.worker import (
    OffloadingHandler,
    TransferResult,
    TransferSpec,
)

from simple_profiler import profiler

from ..file_mapper import FileMapper
from ..kvikio_file import KvikioFileStore, _TimedKvikioResult
from ..vllm_xnvme import (
    ParsedKvLayout,
    Sample,
    SharedStorageLoadStoreSpec,
    _build_block_mapping,
)

logger = init_logger("vllm.storage_kvcache.xnvme_engines.kvikio")


@dataclass
class _InstrumentedTransferResult:
    result: TransferResult
    file_io_samples: list[Sample]
    cuda_copy_samples: list[Sample]


class KvikioDirectOffloadingHandler(OffloadingHandler):
    def __init__(
        self,
        *,
        layout: ParsedKvLayout,
        file_mapper: FileMapper,
        file_store: KvikioFileStore,
        staging_slot_capacity: int,
    ):
        self.layout = layout
        self.file_mapper = file_mapper
        self.file_store = file_store
        self.staging_slot_capacity = staging_slot_capacity
        self._parallelism = max(
            1, min(staging_slot_capacity, self.file_store.parallel_files)
        )
        self._executor = ThreadPoolExecutor(
            max_workers=self._parallelism, thread_name_prefix="kv-kvikio"
        )
        self._file_executor = ThreadPoolExecutor(
            max_workers=self._parallelism, thread_name_prefix="kv-kvikio-file"
        )
        self._futures: dict[int, Future[_InstrumentedTransferResult]] = {}
        self._lock = threading.Lock()
        self._transfer_jobs: dict[int, tuple[int, str, str, str]] = {}
        self._profiled_jobs: set[int] = set()

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
            transfer_size = self._store(src_spec, dst_spec, file_io_samples)
            transfer_type = (
                GPULoadStoreSpec.medium(),
                SharedStorageLoadStoreSpec.medium(),
            )
        elif isinstance(src_spec, SharedStorageLoadStoreSpec) and isinstance(
            dst_spec, GPULoadStoreSpec
        ):
            transfer_size = self._load(src_spec, dst_spec, file_io_samples)
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
                "KvikIO shared-storage offload currently supports exactly one KV cache group"
            )
        if dst_spec.block_indices is None:
            return 0
        if len(dst_spec.block_indices) != 1:
            raise ValueError(
                "KvikIO shared-storage offload currently supports exactly one KV cache group"
            )
        return int(dst_spec.block_indices[0])

    def _mapping_pairs(
        self,
        src_blocks: np.ndarray,
        src_block_size_factor: int,
        dst_blocks: np.ndarray,
        dst_block_size_factor: int,
    ) -> np.ndarray:
        max_len = max(
            1,
            max(
                len(src_blocks) * src_block_size_factor,
                len(dst_blocks) * dst_block_size_factor,
            ),
        )
        mapping = np.empty((max_len, 2), dtype=np.int64)
        mapping_len = _build_block_mapping(
            src_blocks,
            src_block_size_factor,
            dst_blocks,
            dst_block_size_factor,
            mapping,
        )
        return mapping[:mapping_len]

    def _store(
        self,
        src_spec: GPULoadStoreSpec,
        dst_spec: SharedStorageLoadStoreSpec,
        file_io_samples: list[Sample],
    ) -> int:
        total_file_count = len(dst_spec.block_hashes)
        if total_file_count == 0:
            return 0

        src_blocks_per_file = self._split_block_ids_for_files(
            src_spec.block_ids, total_file_count
        )
        futures = [
            self._file_executor.submit(
                self._store_file,
                dst_spec.block_hashes[file_index],
                src_blocks_per_file[file_index],
            )
            for file_index in range(total_file_count)
        ]
        parent_paths: list[str] = []
        for future in futures:
            result = future.result()
            file_io_samples.append(
                (result.start_ns, result.duration_ns, self.layout.storage_block_bytes)
            )
            if result.parent_path:
                parent_paths.append(result.parent_path)
        if parent_paths:
            self.file_store.fsync_dirs(parent_paths)
        return total_file_count * self.layout.storage_block_bytes

    def _load(
        self,
        src_spec: SharedStorageLoadStoreSpec,
        dst_spec: GPULoadStoreSpec,
        file_io_samples: list[Sample],
    ) -> int:
        total_file_count = len(src_spec.block_hashes)
        if total_file_count == 0:
            return 0

        dst_blocks_per_file = self._split_block_ids_for_files(
            dst_spec.block_ids,
            total_file_count,
            self._get_first_group_block_index(dst_spec),
        )
        futures = [
            self._file_executor.submit(
                self._load_file,
                src_spec.block_hashes[file_index],
                dst_blocks_per_file[file_index],
            )
            for file_index in range(total_file_count)
        ]
        for future in futures:
            result = future.result()
            file_io_samples.append(
                (result.start_ns, result.duration_ns, self.layout.storage_block_bytes)
            )
        return total_file_count * self.layout.storage_block_bytes

    def _store_file(
        self, block_hash: bytes, src_blocks: np.ndarray
    ) -> _TimedKvikioResult:
        path = self.file_mapper.get_file_name(block_hash)
        start_ns = time.perf_counter_ns()
        if os.path.exists(path):
            return _TimedKvikioResult(
                start_ns=start_ns,
                duration_ns=time.perf_counter_ns() - start_ns,
                stored=False,
            )

        temp_path, handle = self.file_store.open_write_temp(path)
        try:
            mapping = self._mapping_pairs(
                src_blocks,
                self.layout.gpu_block_size_factor,
                np.asarray([0], dtype=np.int64),
                self.layout.storage_block_size_factor,
            )
            self._write_mapping(handle, mapping)
            handle.close()
            self.file_store.sync_file(temp_path)
            parent_path = self.file_store.publish_temp(temp_path, path)
            return _TimedKvikioResult(
                start_ns=start_ns,
                duration_ns=time.perf_counter_ns() - start_ns,
                parent_path=parent_path,
                stored=parent_path is not None,
            )
        except Exception:
            handle.close()
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass
            raise

    def _load_file(
        self, block_hash: bytes, dst_blocks: np.ndarray
    ) -> _TimedKvikioResult:
        path = self.file_mapper.get_file_name(block_hash)
        start_ns = time.perf_counter_ns()
        with self.file_store.open_read(path) as handle:
            mapping = self._mapping_pairs(
                np.asarray([0], dtype=np.int64),
                self.layout.storage_block_size_factor,
                dst_blocks,
                self.layout.gpu_block_size_factor,
            )
            self._read_mapping(handle, mapping)
        return _TimedKvikioResult(
            start_ns=start_ns,
            duration_ns=time.perf_counter_ns() - start_ns,
        )

    def _write_mapping(self, handle, mapping: np.ndarray) -> None:
        pending_ios = []
        tensor_file_offset = 0
        for tensor, block_bytes in zip(
            self.layout.gpu_tensors, self.layout.bytes_per_kernel_block
        ):
            for src_block, dst_block in mapping:
                pending_ios.append(
                    self.file_store.submit_write_tensor_slice(
                        handle,
                        tensor,
                        size=block_bytes,
                        file_offset=tensor_file_offset + int(dst_block) * block_bytes,
                        tensor_offset=int(src_block) * block_bytes,
                    )
                )
            tensor_file_offset += block_bytes * self.layout.storage_block_size_factor

        for pending in pending_ios:
            result = self.file_store.wait_kvikio_io(pending)
            if result != pending.expected_nbytes:
                raise IOError(
                    "KvikIO async write transferred "
                    f"{result} bytes, expected {pending.expected_nbytes}"
                )

    def _read_mapping(self, handle, mapping: np.ndarray) -> None:
        pending_ios = []
        tensor_file_offset = 0
        for tensor, block_bytes in zip(
            self.layout.gpu_tensors, self.layout.bytes_per_kernel_block
        ):
            for src_block, dst_block in mapping:
                pending_ios.append(
                    self.file_store.submit_read_tensor_slice(
                        handle,
                        tensor,
                        size=block_bytes,
                        file_offset=tensor_file_offset + int(src_block) * block_bytes,
                        tensor_offset=int(dst_block) * block_bytes,
                    )
                )
            tensor_file_offset += block_bytes * self.layout.storage_block_size_factor

        for pending in pending_ios:
            result = self.file_store.wait_kvikio_io(pending)
            if result != pending.expected_nbytes:
                raise IOError(
                    "KvikIO async read transferred "
                    f"{result} bytes, expected {pending.expected_nbytes}"
                )

    def _emit_profile_events(
        self, job_id: int, instrumented_result: _InstrumentedTransferResult, end_ns: int
    ) -> None:
        if job_id in self._profiled_jobs:
            return
        if not getattr(profiler, "_active", False):
            return

        job_meta = self._transfer_jobs.get(job_id)
        if job_meta is None:
            return

        result = instrumented_result.result
        wall_start_ns, profile_tid, req_id, direction = job_meta
        is_store = direction == "gpu_to_storage"
        transfer_size = result.transfer_size or 0
        file_io_wall_ns = sum(
            duration_ns
            for _start_ns, duration_ns, _sample_num_bytes in (
                instrumented_result.file_io_samples
            )
        )
        profiler.add_event(
            name=f"kvikio_transfer({direction}, job={job_id})",
            category="kvikio_transfer",
            start_ns=wall_start_ns,
            duration_ns=end_ns - wall_start_ns,
            tid=profile_tid,
            args={
                "req_id": req_id,
                "success": result.success,
                "num_bytes": transfer_size,
                "file_io_bw_GBps": round(
                    (transfer_size / file_io_wall_ns) if file_io_wall_ns > 0 else 0.0,
                    3,
                ),
                "cuda_copy_bw_GBps": 0.0,
            },
        )
        for sample_idx, (start_ns, duration_ns, sample_num_bytes) in enumerate(
            instrumented_result.file_io_samples
        ):
            profiler.add_event(
                name=(
                    f"kvikio_file_{'write' if is_store else 'read'}"
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
                self._emit_profile_events(
                    job_id, instrumented_result, time.perf_counter_ns()
                )
                self._transfer_jobs.pop(job_id, None)
                self._profiled_jobs.discard(job_id)
                results.append(result)
            except Exception:
                logger.exception("KvikIO shared KV transfer job %s failed", job_id)
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
        self._file_executor.shutdown(wait=True)
        self.file_store.close()


__all__ = ["KvikioDirectOffloadingHandler"]
