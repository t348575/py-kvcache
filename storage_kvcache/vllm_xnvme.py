from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import torch
from vllm import _custom_ops as ops
from vllm.config import VllmConfig
from vllm.platforms import current_platform
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.kv_offload.abstract import (
    LoadStoreSpec,
    OffloadingManager,
    PrepareStoreOutput,
)
from vllm.v1.kv_offload.mediums import GPULoadStoreSpec
from vllm.v1.kv_offload.spec import OffloadingSpec
from vllm.v1.kv_offload.worker.worker import (
    OffloadingHandler,
    TransferResult,
    TransferSpec,
)

from .file_mapper import FileMapper
from .fs_config import SharedFileConfig
from .xnvme_file import XNvmeFileStore

logger = logging.getLogger(__name__)


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
) -> torch.Tensor:
    src_sub_block_count = src_blocks.size * src_block_size_factor
    dst_sub_block_count = dst_blocks.size * dst_block_size_factor
    src_sub_blocks_to_skip = -dst_blocks.size % src_block_size_factor

    if dst_sub_block_count != src_sub_block_count - src_sub_blocks_to_skip:
        raise ValueError("source and destination block counts do not line up")

    src_to_dst = np.empty((dst_sub_block_count, 2), dtype=np.int64)
    _expand_block_ids(
        src_blocks,
        src_block_size_factor,
        src_to_dst[:, 0],
        skip_count=src_sub_blocks_to_skip,
    )
    _expand_block_ids(dst_blocks, dst_block_size_factor, src_to_dst[:, 1])
    return torch.from_numpy(src_to_dst)


class SharedStorageLoadStoreSpec(LoadStoreSpec):
    def __init__(self, block_hashes: Iterable[BlockHash]):
        self.block_hashes = list(block_hashes)

    @staticmethod
    def medium() -> str:
        return "SHARED_STORAGE"


class SharedStorageOffloadingManager(OffloadingManager):
    def __init__(self, file_mapper: FileMapper):
        self.file_mapper = file_mapper

    def lookup(self, block_hashes: Iterable[BlockHash]) -> int:
        hit_count = 0
        for block_hash in block_hashes:
            if not os.path.exists(self.file_mapper.get_file_name(block_hash)):
                break
            hit_count += 1
        return hit_count

    def prepare_load(self, block_hashes: Iterable[BlockHash]) -> LoadStoreSpec:
        return SharedStorageLoadStoreSpec(block_hashes)

    def touch(self, block_hashes: Iterable[BlockHash]):
        del block_hashes
        return

    def complete_load(self, block_hashes: Iterable[BlockHash]):
        del block_hashes
        return

    def prepare_store(
        self, block_hashes: Iterable[BlockHash]
    ) -> PrepareStoreOutput | None:
        missing_block_hashes = [
            block_hash
            for block_hash in block_hashes
            if not os.path.exists(self.file_mapper.get_file_name(block_hash))
        ]
        return PrepareStoreOutput(
            block_hashes_to_store=missing_block_hashes,
            store_spec=SharedStorageLoadStoreSpec(missing_block_hashes),
            block_hashes_evicted=[],
        )

    def complete_store(self, block_hashes: Iterable[BlockHash], success: bool = True):
        del block_hashes, success
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
    def from_gpu_caches(
        cls,
        *,
        gpu_block_size: int,
        storage_block_size: int,
        gpu_caches: dict[str, torch.Tensor],
        attn_backends: dict[str, type[AttentionBackend]],
    ) -> "ParsedKvLayout":
        kernel_block_size: int | None = None
        parsed_gpu_tensors: list[torch.Tensor] = []

        for layer_name, gpu_tensor in gpu_caches.items():
            gpu_shape = gpu_tensor.shape
            attn_backend = attn_backends[layer_name]
            test_shape = attn_backend.get_kv_cache_shape(
                num_blocks=1234, block_size=16, num_kv_heads=8, head_size=256
            )

            has_layers_dim = False
            split_k_and_v = False
            if len(gpu_shape) != len(test_shape):
                if len(gpu_shape) != len(test_shape) + 1:
                    raise ValueError("unexpected cross-layer KV cache shape")
                has_layers_dim = True
                test_shape = (80,) + test_shape
            elif test_shape[0] != 1234:
                if test_shape[0] != 2 or test_shape[1] != 1234 or gpu_shape[0] != 2:
                    raise ValueError("unexpected split-KV cache shape")
                split_k_and_v = True

            if has_layers_dim:
                try:
                    kv_cache_stride_order = attn_backend.get_kv_cache_stride_order(
                        include_num_layers_dimension=True
                    )
                except (AttributeError, NotImplementedError):
                    kv_cache_stride_order = tuple(range(len(gpu_shape)))
                test_shape = tuple(test_shape[index] for index in kv_cache_stride_order)

            block_size_idx = test_shape.index(16)
            detected_kernel_block_size = gpu_shape[block_size_idx]
            if kernel_block_size is None:
                kernel_block_size = detected_kernel_block_size
            elif kernel_block_size != detected_kernel_block_size:
                raise ValueError("all KV tensors must share the same kernel block size")

            parsed_gpu_tensors.extend(
                gpu_tensor.unbind(0) if split_k_and_v else [gpu_tensor]
            )

        if kernel_block_size is None:
            raise ValueError("at least one KV cache tensor must be registered")
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

    def pack_storage_slot(
        self, staging_tensors: list[torch.Tensor], slot_index: int
    ) -> bytes:
        start = slot_index * self.storage_block_size_factor
        end = start + self.storage_block_size_factor
        parts: list[bytes] = []
        for tensor in staging_tensors:
            parts.append(self._byte_array_view(tensor[start:end]).tobytes(order="C"))
        return b"".join(parts)

    def unpack_storage_slot(
        self,
        payload: bytes,
        staging_tensors: list[torch.Tensor],
        slot_index: int,
    ) -> None:
        if len(payload) != self.storage_block_bytes:
            raise ValueError(
                f"payload size {len(payload)} does not match storage block size {self.storage_block_bytes}"
            )

        start = slot_index * self.storage_block_size_factor
        end = start + self.storage_block_size_factor
        payload_view = memoryview(payload)
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
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="kv-xnvme"
        )
        self._futures: dict[int, Future[TransferResult]] = {}
        self._lock = threading.Lock()
        self._staging_tensors = self.layout.allocate_staging_tensors(
            staging_slot_capacity
        )

    def transfer_async(
        self,
        job_id: int,
        spec: TransferSpec,
        profile_tid: str = "kv_transfer",
        req_id: str = "",
    ) -> bool:
        del profile_tid, req_id
        with self._lock:
            if job_id in self._futures:
                raise ValueError(f"job {job_id} is already active")
            self._futures[job_id] = self._executor.submit(
                self._run_transfer, job_id, spec
            )
        return True

    def _run_transfer(self, job_id: int, spec: TransferSpec) -> TransferResult:
        started = time.perf_counter()
        src_spec, dst_spec = spec
        if isinstance(src_spec, GPULoadStoreSpec) and isinstance(
            dst_spec, SharedStorageLoadStoreSpec
        ):
            transfer_size = self._store(src_spec, dst_spec)
            transfer_type = (
                GPULoadStoreSpec.medium(),
                SharedStorageLoadStoreSpec.medium(),
            )
        elif isinstance(src_spec, SharedStorageLoadStoreSpec) and isinstance(
            dst_spec, GPULoadStoreSpec
        ):
            transfer_size = self._load(src_spec, dst_spec)
            transfer_type = (
                SharedStorageLoadStoreSpec.medium(),
                GPULoadStoreSpec.medium(),
            )
        else:
            raise TypeError(
                f"unsupported transfer spec: {type(src_spec)!r} -> {type(dst_spec)!r}"
            )

        return TransferResult(
            job_id=job_id,
            success=True,
            transfer_size=transfer_size,
            transfer_time=time.perf_counter() - started,
            transfer_type=transfer_type,
        )

    def _swap_blocks(
        self,
        src_tensors: list[torch.Tensor],
        dst_tensors: list[torch.Tensor],
        src_to_dst: torch.Tensor,
    ) -> None:
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            for src_tensor, dst_tensor, block_bytes in zip(
                src_tensors,
                dst_tensors,
                self.layout.bytes_per_kernel_block,
            ):
                ops.swap_blocks(src_tensor, dst_tensor, block_bytes, src_to_dst)
            end_event = torch.cuda.Event(enable_timing=False)
            end_event.record(stream)
        end_event.synchronize()

    def _store(
        self,
        src_spec: GPULoadStoreSpec,
        dst_spec: SharedStorageLoadStoreSpec,
    ) -> int:
        total_file_count = len(dst_spec.block_hashes)
        gpu_blocks_per_storage_block = self.layout.gpu_blocks_per_storage_block

        for start in range(0, total_file_count, self.staging_slot_capacity):
            end = min(start + self.staging_slot_capacity, total_file_count)
            chunk_file_count = end - start
            local_slots = np.arange(chunk_file_count, dtype=np.int64)
            src_chunk = src_spec.block_ids[
                start * gpu_blocks_per_storage_block : end
                * gpu_blocks_per_storage_block
            ]
            dst_hashes = dst_spec.block_hashes[start:end]

            src_to_dst = _build_block_mapping(
                src_chunk,
                self.layout.gpu_block_size_factor,
                local_slots,
                self.layout.storage_block_size_factor,
            )
            self._swap_blocks(
                self.layout.gpu_tensors, self._staging_tensors, src_to_dst
            )

            for local_slot, block_hash in enumerate(dst_hashes):
                payload = self.layout.pack_storage_slot(
                    self._staging_tensors, local_slot
                )
                self.file_store.store_if_missing(
                    self.file_mapper.get_file_name(block_hash),
                    payload,
                )

        return total_file_count * self.layout.storage_block_bytes

    def _load(
        self,
        src_spec: SharedStorageLoadStoreSpec,
        dst_spec: GPULoadStoreSpec,
    ) -> int:
        total_file_count = len(src_spec.block_hashes)
        gpu_blocks_per_storage_block = self.layout.gpu_blocks_per_storage_block

        for start in range(0, total_file_count, self.staging_slot_capacity):
            end = min(start + self.staging_slot_capacity, total_file_count)
            chunk_file_count = end - start
            src_hashes = src_spec.block_hashes[start:end]
            dst_chunk = dst_spec.block_ids[
                start * gpu_blocks_per_storage_block : end
                * gpu_blocks_per_storage_block
            ]

            for local_slot, block_hash in enumerate(src_hashes):
                payload = self.file_store.read_file(
                    self.file_mapper.get_file_name(block_hash)
                )
                self.layout.unpack_storage_slot(
                    payload, self._staging_tensors, local_slot
                )

            local_slots = np.arange(chunk_file_count, dtype=np.int64)
            src_to_dst = _build_block_mapping(
                local_slots,
                self.layout.storage_block_size_factor,
                dst_chunk,
                self.layout.gpu_block_size_factor,
            )
            self._swap_blocks(
                self._staging_tensors, self.layout.gpu_tensors, src_to_dst
            )

        return total_file_count * self.layout.storage_block_bytes

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
                results.append(future.result())
            except Exception:
                logger.exception("shared KV transfer job %s failed", job_id)
                results.append(TransferResult(job_id=job_id, success=False))
            finally:
                with self._lock:
                    self._futures.pop(job_id, None)
        return results

    def wait(self, job_ids: set[int]) -> None:
        with self._lock:
            futures = [
                self._futures[job_id] for job_id in job_ids if job_id in self._futures
            ]
        for future in futures:
            future.result()

    def close(self) -> None:
        self._executor.shutdown(wait=True)


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
        self,
        kv_caches: dict[str, torch.Tensor],
        attn_backends: dict[str, type[AttentionBackend]],
    ) -> Iterator[tuple[type[LoadStoreSpec], type[LoadStoreSpec], OffloadingHandler]]:
        if self._handler is None:
            if not current_platform.is_cuda_alike():
                raise RuntimeError(
                    "xNVMe KV offloading is currently only supported on CUDA-like GPUs"
                )

            gpu_block_size = self.gpu_block_size[0]
            layout = ParsedKvLayout.from_gpu_caches(
                gpu_block_size=gpu_block_size,
                storage_block_size=gpu_block_size * self.block_size_factor,
                gpu_caches=kv_caches,
                attn_backends=attn_backends,
            )
            staging_slot_capacity = self._resolve_staging_slot_capacity(layout)
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
