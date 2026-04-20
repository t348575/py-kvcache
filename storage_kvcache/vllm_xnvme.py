from __future__ import annotations

import os
import threading
import time
from collections.abc import Iterable, Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from functools import partial

import numpy as np
import torch
from vllm import _custom_ops as ops
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.kv_offload.abstract import (
    LoadStoreSpec,
    OffloadingManager,
    OffloadKey,
    PrepareStoreOutput,
    get_offload_block_hash,
    get_offload_group_idx,
)
from vllm.v1.kv_offload.mediums import GPULoadStoreSpec
from vllm.v1.kv_offload.spec import CanonicalKVCaches, OffloadingSpec
from vllm.v1.kv_offload.worker.worker import (
    OffloadingHandler,
    TransferResult,
    TransferSpec,
)

from .file_mapper import FileMapper
from .fs_config import SharedFileConfig
from simple_profiler import profiler
from .xnvme_file import XNvmeFileStore

logger = init_logger("vllm.storage_kvcache.vllm_xnvme")

Sample = tuple[int, int, int]


def _expand_block_ids(
    block_ids: np.ndarray,
    block_size_factor: int,
    output: np.ndarray,
    *,
    skip_count: int = 0,
) -> None:
    if skip_count >= block_size_factor:
        raise ValueError("skip_count must be smaller than block_size_factor")

    first_range = np.arange(skip_count, block_size_factor, dtype=np.int64)
    full_range = np.arange(0, block_size_factor, dtype=np.int64)

    output_index = 0
    for index, block_id in enumerate(block_ids):
        base_block_id = int(block_id) * block_size_factor
        indices = first_range if index == 0 else full_range
        output_end = output_index + len(indices)
        output[output_index:output_end] = base_block_id + indices
        output_index = output_end


def _build_block_mapping(
    src_blocks: np.ndarray,
    src_block_size_factor: int,
    dst_blocks: np.ndarray,
    dst_block_size_factor: int,
    output: np.ndarray,
) -> int:
    src_sub_block_count = src_blocks.size * src_block_size_factor
    dst_sub_block_count = dst_blocks.size * dst_block_size_factor
    src_sub_blocks_to_skip = -dst_blocks.size % src_block_size_factor

    if dst_sub_block_count != src_sub_block_count - src_sub_blocks_to_skip:
        raise ValueError("source and destination block counts do not line up")

    _expand_block_ids(
        src_blocks,
        src_block_size_factor,
        output[:dst_sub_block_count, 0],
        skip_count=src_sub_blocks_to_skip,
    )
    _expand_block_ids(
        dst_blocks, dst_block_size_factor, output[:dst_sub_block_count, 1]
    )
    return dst_sub_block_count


class SharedStorageLoadStoreSpec(LoadStoreSpec):
    def __init__(self, block_hashes: Iterable[bytes]):
        self.block_hashes = list(block_hashes)

    @staticmethod
    def medium() -> str:
        return "SHARED_STORAGE"


class SharedStorageOffloadingManager(OffloadingManager):
    def __init__(self, file_mapper: FileMapper):
        self.file_mapper = file_mapper

    @staticmethod
    def _get_block_hash(key: OffloadKey) -> bytes:
        group_idx = get_offload_group_idx(key)
        if group_idx != 0:
            raise ValueError(
                "xNVMe shared-storage offload currently supports exactly one KV cache group"
            )
        return get_offload_block_hash(key)

    def lookup(self, keys: Iterable[OffloadKey]) -> int:
        hit_count = 0
        for key in keys:
            block_hash = self._get_block_hash(key)
            if not os.path.exists(self.file_mapper.get_file_name(block_hash)):
                break
            hit_count += 1
        return hit_count

    def prepare_load(self, keys: Iterable[OffloadKey]) -> LoadStoreSpec:
        return SharedStorageLoadStoreSpec(self._get_block_hash(key) for key in keys)

    def touch(self, keys: Iterable[OffloadKey]):
        return

    def complete_load(self, keys: Iterable[OffloadKey]):
        return

    def prepare_store(self, keys: Iterable[OffloadKey]) -> PrepareStoreOutput | None:
        keys_to_store: list[OffloadKey] = []
        block_hashes_to_store: list[bytes] = []
        for key in keys:
            block_hash = self._get_block_hash(key)
            if os.path.exists(self.file_mapper.get_file_name(block_hash)):
                continue
            keys_to_store.append(key)
            block_hashes_to_store.append(block_hash)
        return PrepareStoreOutput(
            keys_to_store=keys_to_store,
            store_spec=SharedStorageLoadStoreSpec(block_hashes_to_store),
            evicted_keys=[],
        )

    def complete_store(self, keys: Iterable[OffloadKey], success: bool = True):
        return


@dataclass
class ParsedKvLayout:
    gpu_tensors: list[torch.Tensor]
    bytes_per_kernel_block: list[int]
    gpu_block_size_factor: int
    storage_block_size_factor: int
    pin_memory: bool

    @property
    def storage_block_bytes(self) -> int:
        return self.storage_block_size_factor * sum(self.bytes_per_kernel_block)

    @property
    def gpu_blocks_per_storage_block(self) -> int:
        if self.storage_block_size_factor % self.gpu_block_size_factor != 0:
            raise ValueError(
                "storage block size must be a multiple of the GPU block size"
            )
        return self.storage_block_size_factor // self.gpu_block_size_factor

    @classmethod
    def from_canonical_kv_caches(
        cls,
        *,
        gpu_block_size: int,
        storage_block_size: int,
        kv_caches: CanonicalKVCaches,
    ) -> "ParsedKvLayout":
        if len(kv_caches.group_data_refs) != 1:
            raise ValueError(
                "xNVMe shared-storage offload currently supports exactly one KV cache group"
            )

        if not kv_caches.tensors:
            raise ValueError("at least one KV cache tensor must be registered")

        parsed_gpu_tensors = [kv_cache.tensor for kv_cache in kv_caches.tensors]

        for tensor in parsed_gpu_tensors:
            if tensor.dtype != torch.int8:
                raise ValueError("canonical KV cache tensors must use int8 dtype")
            if tensor.ndim != 2:
                raise ValueError(
                    "canonical KV cache tensors must have shape (num_blocks, page_size_bytes)"
                )
            if not tensor.is_cuda:
                raise ValueError("canonical KV cache tensors must reside on CUDA")

        kernel_block_size = gpu_block_size
        if gpu_block_size % kernel_block_size != 0:
            raise ValueError("gpu block size must align to kernel block size")
        if storage_block_size % kernel_block_size != 0:
            raise ValueError("storage block size must align to kernel block size")

        return cls(
            gpu_tensors=parsed_gpu_tensors,
            bytes_per_kernel_block=[
                tensor.element_size() * tensor.stride(0)
                for tensor in parsed_gpu_tensors
            ],
            gpu_block_size_factor=gpu_block_size // kernel_block_size,
            storage_block_size_factor=storage_block_size // kernel_block_size,
            pin_memory=is_pin_memory_available(),
        )

    def allocate_staging_tensors(self, slot_count: int) -> list[torch.Tensor]:
        num_kernel_blocks = slot_count * self.storage_block_size_factor
        staging_tensors: list[torch.Tensor] = []
        for gpu_tensor in self.gpu_tensors:
            shape = list(gpu_tensor.shape)
            shape[0] = num_kernel_blocks
            staging_tensors.append(
                torch.empty(
                    shape,
                    dtype=gpu_tensor.dtype,
                    device="cpu",
                    pin_memory=self.pin_memory,
                )
            )
        return staging_tensors

    @staticmethod
    def _byte_array_view(tensor: torch.Tensor) -> np.ndarray:
        # Reinterpret as raw bytes so unsupported NumPy dtypes like bfloat16
        # still serialize and deserialize losslessly.
        array = tensor.view(torch.uint8).numpy()
        if not array.flags.c_contiguous:
            raise ValueError("staging tensor slice must be contiguous")
        return array.reshape(-1)

    def pack_storage_slot_into(
        self,
        staging_tensors: list[torch.Tensor],
        slot_index: int,
        payload: np.ndarray,
    ) -> None:
        if payload.size < self.storage_block_bytes:
            raise ValueError(
                f"payload size {payload.size} is smaller than storage block size {self.storage_block_bytes}"
            )
        start = slot_index * self.storage_block_size_factor
        end = start + self.storage_block_size_factor
        offset = 0
        for tensor, kernel_block_bytes in zip(
            staging_tensors, self.bytes_per_kernel_block
        ):
            nbytes = kernel_block_bytes * self.storage_block_size_factor
            payload[offset : offset + nbytes] = self._byte_array_view(tensor[start:end])
            offset += nbytes

    def unpack_storage_slot(
        self,
        payload: bytes | bytearray | memoryview | np.ndarray,
        staging_tensors: list[torch.Tensor],
        slot_index: int,
    ) -> None:
        payload_view = memoryview(payload).cast("B")
        if payload_view.nbytes != self.storage_block_bytes:
            raise ValueError(
                f"payload size {payload_view.nbytes} does not match storage block size {self.storage_block_bytes}"
            )

        start = slot_index * self.storage_block_size_factor
        end = start + self.storage_block_size_factor
        offset = 0
        for tensor, kernel_block_bytes in zip(
            staging_tensors, self.bytes_per_kernel_block
        ):
            expected_nbytes = kernel_block_bytes * self.storage_block_size_factor
            byte_view = self._byte_array_view(tensor[start:end])
            byte_view[:] = np.frombuffer(
                payload_view[offset : offset + expected_nbytes],
                dtype=np.uint8,
                count=expected_nbytes,
            )
            offset += expected_nbytes


@dataclass
class _InstrumentedTransferResult:
    result: TransferResult
    file_io_samples: list[Sample]
    cuda_copy_samples: list[Sample]


class XNvmeOffloadingHandler(OffloadingHandler):
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
            1, min(self.staging_slot_capacity, self.file_store.parallel_files)
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
        self._swap_stream = torch.cuda.Stream()
        self._swap_event = torch.cuda.Event(enable_timing=False)
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

    def _swap_blocks(
        self,
        src_tensors: list[torch.Tensor],
        dst_tensors: list[torch.Tensor],
        src_to_dst: torch.Tensor,
        sample_num_bytes: int,
        cuda_copy_samples: list[Sample],
    ) -> None:
        sample_start_ns = time.perf_counter_ns()
        with torch.cuda.stream(self._swap_stream):
            for src_tensor, dst_tensor, block_bytes in zip(
                src_tensors,
                dst_tensors,
                self.layout.bytes_per_kernel_block,
            ):
                ops.swap_blocks(src_tensor, dst_tensor, block_bytes, src_to_dst)
            self._swap_event.record(self._swap_stream)
        self._swap_event.synchronize()
        cuda_copy_samples.append(
            (
                sample_start_ns,
                time.perf_counter_ns() - sample_start_ns,
                sample_num_bytes,
            )
        )

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
    ) -> tuple[int, np.ndarray, _TimedIoResult]:
        result = self.file_store.read_path_into(
            self.file_mapper.get_file_name(block_hash),
            slot_index,
            self._unpack_storage_slot_from_payload,
        )
        return slot_index, dst_blocks, result

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
        next_file_index = 0
        parent_paths: list[str] = []

        def submit_next(slot_index: int) -> None:
            nonlocal next_file_index
            if next_file_index >= total_file_count:
                return
            src_blocks = src_blocks_per_file[next_file_index]
            src_to_dst = self._build_block_mapping(
                slot_index,
                src_blocks,
                self.layout.gpu_block_size_factor,
                self._slot_block_ids[slot_index],
                self.layout.storage_block_size_factor,
            )
            self._swap_blocks(
                self.layout.gpu_tensors,
                self._staging_tensors,
                src_to_dst,
                self.layout.storage_block_bytes,
                cuda_copy_samples,
            )
            future = self._pipeline_executor.submit(
                self._store_slot,
                slot_index,
                dst_spec.block_hashes[next_file_index],
            )
            inflight[future] = slot_index
            next_file_index += 1

        for slot_index in free_slots:
            submit_next(slot_index)

        while inflight:
            done, _pending = wait(inflight, return_when=FIRST_COMPLETED)
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
                submit_next(slot_index)

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
        inflight: dict[Future[tuple[int, np.ndarray, _TimedIoResult]], int] = {}
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

        for slot_index in free_slots:
            submit_next(slot_index)

        while inflight:
            done, _pending = wait(inflight, return_when=FIRST_COMPLETED)
            for future in done:
                slot_index = inflight.pop(future)
                slot_index, dst_blocks, result = future.result()
                file_io_samples.append(
                    (
                        result.start_ns,
                        result.duration_ns,
                        self.layout.storage_block_bytes,
                    )
                )
                src_to_dst = self._build_block_mapping(
                    slot_index,
                    self._slot_block_ids[slot_index],
                    self.layout.storage_block_size_factor,
                    dst_blocks,
                    self.layout.gpu_block_size_factor,
                )
                self._swap_blocks(
                    self._staging_tensors,
                    self.layout.gpu_tensors,
                    src_to_dst,
                    self.layout.storage_block_bytes,
                    cuda_copy_samples,
                )
                submit_next(slot_index)

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


class XNvmeOffloadingSpec(OffloadingSpec):
    def __init__(self, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig):
        super().__init__(vllm_config, kv_cache_config)

        self.shared_file_config = SharedFileConfig.from_extra_config(self.extra_config)
        self._manager: OffloadingManager | None = None
        self._handler: XNvmeOffloadingHandler | None = None

        assert len(self.gpu_block_size) == 1, (
            f"Expected exactly one KV cache group, got {len(self.gpu_block_size)}"
        )

        parallel_config = vllm_config.parallel_config
        dtype = str(vllm_config.cache_config.cache_dtype).replace("torch.", "")
        self.file_mapper = FileMapper(
            root_dir=self.shared_file_config.root_dir,
            model_name=vllm_config.model_config.model,
            gpu_block_size=self.gpu_block_size[0],
            gpu_blocks_per_file=self.block_size_factor,
            tp_size=parallel_config.tensor_parallel_size,
            pp_size=parallel_config.pipeline_parallel_size,
            pcp_size=parallel_config.prefill_context_parallel_size,
            rank=parallel_config.rank,
            dtype=dtype,
        )
        os.makedirs(self.file_mapper.base_path, exist_ok=True)

    def get_manager(self) -> OffloadingManager:
        if self._manager is None:
            self._manager = SharedStorageOffloadingManager(self.file_mapper)
        return self._manager

    def get_handlers(
        self, kv_caches: CanonicalKVCaches
    ) -> Iterator[tuple[type[LoadStoreSpec], type[LoadStoreSpec], OffloadingHandler]]:
        if self._handler is None:
            if not current_platform.is_cuda_alike():
                raise RuntimeError(
                    "xNVMe KV offloading is currently only supported on CUDA-like GPUs"
                )

            gpu_block_size = self.gpu_block_size[0]
            layout = ParsedKvLayout.from_canonical_kv_caches(
                gpu_block_size=gpu_block_size,
                storage_block_size=gpu_block_size * self.block_size_factor,
                kv_caches=kv_caches,
            )
            staging_slot_capacity = self._resolve_staging_slot_capacity(layout)
            logger.info(
                "Initializing xNVMe KV offloading: root_dir=%s model=%s dtype=%s gpu_block_size=%d storage_block_size=%d storage_block_bytes=%d block_size_factor=%d staging_slot_capacity=%d direct_io=%s sync_on_store=%s profiler_active=%s file_mapper_base=%s",
                self.shared_file_config.root_dir,
                self.vllm_config.model_config.model,
                str(self.vllm_config.cache_config.cache_dtype).replace("torch.", ""),
                gpu_block_size,
                gpu_block_size * self.block_size_factor,
                layout.storage_block_bytes,
                self.block_size_factor,
                staging_slot_capacity,
                self.shared_file_config.direct,
                self.shared_file_config.sync_on_store,
                getattr(profiler, "_active", False),
                self.file_mapper.base_path,
            )
            self._handler = XNvmeOffloadingHandler(
                layout=layout,
                file_mapper=self.file_mapper,
                file_store=XNvmeFileStore(
                    self.shared_file_config,
                    payload_size=layout.storage_block_bytes,
                ),
                staging_slot_capacity=staging_slot_capacity,
            )

        yield GPULoadStoreSpec, SharedStorageLoadStoreSpec, self._handler
        yield SharedStorageLoadStoreSpec, GPULoadStoreSpec, self._handler

    def _resolve_staging_slot_capacity(self, layout: ParsedKvLayout) -> int:
        configured_num_blocks = self.extra_config.get("staging_num_blocks")
        if configured_num_blocks is not None:
            slot_capacity = int(configured_num_blocks)
        else:
            configured_bytes = self.extra_config.get("staging_bytes_to_use")
            staging_bytes_to_use = (
                int(configured_bytes) if configured_bytes is not None else 256 << 20
            )
            slot_capacity = staging_bytes_to_use // layout.storage_block_bytes

        if slot_capacity <= 0:
            slot_capacity = 1
        return slot_capacity


SharedStorageOffloadingSpec = XNvmeOffloadingSpec

__all__ = [
    "SharedStorageLoadStoreSpec",
    "SharedStorageOffloadingSpec",
    "XNvmeOffloadingSpec",
]
