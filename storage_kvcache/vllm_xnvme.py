from __future__ import annotations

import os
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import numpy as np
import torch
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
from vllm.v1.kv_offload.worker.worker import OffloadingHandler

from simple_profiler import profiler

from .file_mapper import FileMapper
from .fs_config import SharedFileConfig
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


class XNvmeOffloadingSpec(OffloadingSpec):
    def __init__(self, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig):
        super().__init__(vllm_config, kv_cache_config)

        self.shared_file_config = SharedFileConfig.from_extra_config(self.extra_config)
        self._manager: OffloadingManager | None = None
        self._handler: OffloadingHandler | None = None

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
                "Initializing xNVMe KV offloading: root_dir=%s model=%s dtype=%s gpu_block_size=%d storage_block_size=%d storage_block_bytes=%d block_size_factor=%d staging_slot_capacity=%d engine=%s direct_io=%s sync_on_store=%s profiler_active=%s file_mapper_base=%s",
                self.shared_file_config.root_dir,
                self.vllm_config.model_config.model,
                str(self.vllm_config.cache_config.cache_dtype).replace("torch.", ""),
                gpu_block_size,
                gpu_block_size * self.block_size_factor,
                layout.storage_block_bytes,
                self.block_size_factor,
                staging_slot_capacity,
                self.shared_file_config.engine,
                self.shared_file_config.direct,
                self.shared_file_config.sync_on_store,
                getattr(profiler, "_active", False),
                self.file_mapper.base_path,
            )
            from .xnvme_engines import (
                create_xnvme_offloading_handler,
                normalize_xnvme_engine_name,
            )

            engine_name = normalize_xnvme_engine_name(self.shared_file_config.engine)
            if engine_name in {"liburing_single_threaded", "liburing_multithreaded"}:
                from .liburing_file import LiburingFileStore

                file_store = LiburingFileStore(
                    self.shared_file_config,
                    payload_size=layout.storage_block_bytes,
                )
            else:
                file_store = XNvmeFileStore(
                    self.shared_file_config,
                    payload_size=layout.storage_block_bytes,
                )

            self._handler = create_xnvme_offloading_handler(
                engine=self.shared_file_config.engine,
                layout=layout,
                file_mapper=self.file_mapper,
                file_store=file_store,
                staging_slot_capacity=staging_slot_capacity,
            )

        yield GPULoadStoreSpec, SharedStorageLoadStoreSpec, self._handler
        yield SharedStorageLoadStoreSpec, GPULoadStoreSpec, self._handler

    def _resolve_staging_slot_capacity(self, layout: ParsedKvLayout) -> int:
        configured_num_blocks = self.extra_config.get("staging_num_blocks")
        if configured_num_blocks is not None:
            slot_capacity = int(configured_num_blocks)
        else:
            configured_gb = self.extra_config.get("staging_mem")
            staging_bytes_to_use = (
                int(configured_gb) * (1 << 30) if configured_gb is not None else 1 << 30
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
