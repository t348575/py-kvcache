from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .profiling import add_event, now_ns

try:
    import torch
    from vllm import _custom_ops as ops
    from vllm.utils.platform_utils import is_pin_memory_available

    TORCH_COPY_AVAILABLE = True
except Exception:
    torch = Any
    ops = None
    TORCH_COPY_AVAILABLE = False

    def is_pin_memory_available() -> bool:
        return False


Sample = tuple[int, int, int]


@dataclass(frozen=True)
class _TransferProfile:
    job_id: int
    direction: str
    profile_tid: str
    req_id: str
    start_ns: int


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


def build_block_mapping(
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
        dst_blocks,
        dst_block_size_factor,
        output[:dst_sub_block_count, 1],
    )
    return dst_sub_block_count


@dataclass
class ParsedKvLayout:
    gpu_tensors: list[Any]
    bytes_per_kernel_block: list[int]
    storage_block_size_factor: int
    pin_memory: bool

    @property
    def storage_block_bytes(self) -> int:
        return self.storage_block_size_factor * sum(self.bytes_per_kernel_block)

    @property
    def gpu_blocks_per_storage_block(self) -> int:
        return self.storage_block_size_factor

    @classmethod
    def from_canonical_kv_caches(
        cls,
        *,
        gpu_block_size: int,
        storage_block_size: int,
        kv_caches: Any,
    ) -> "ParsedKvLayout":
        if not TORCH_COPY_AVAILABLE:
            raise RuntimeError("vLLM and torch are required for GPU copy support")

        group_data_refs = getattr(kv_caches, "group_data_refs", None)
        if group_data_refs is not None and len(group_data_refs) != 1:
            raise ValueError(
                "py-kvcache shared-storage offload currently supports exactly one KV cache group"
            )

        if not getattr(kv_caches, "tensors", None):
            raise ValueError("at least one KV cache tensor must be registered")

        parsed_gpu_tensors = [kv_cache.tensor for kv_cache in kv_caches.tensors]
        if len(parsed_gpu_tensors) != 1:
            raise ValueError(
                "py-kvcache requires exactly one canonical KV cache tensor "
                "(direct-staging only); got "
                f"{len(parsed_gpu_tensors)}"
            )
        for tensor in parsed_gpu_tensors:
            if tensor.dtype != torch.int8:
                raise ValueError("canonical KV cache tensors must use int8 dtype")
            if tensor.ndim != 2:
                raise ValueError(
                    "canonical KV cache tensors must have shape (num_blocks, page_size_bytes)"
                )
            if not tensor.is_cuda:
                raise ValueError("canonical KV cache tensors must reside on CUDA")

        if storage_block_size % gpu_block_size != 0:
            raise ValueError("storage block size must be a multiple of the gpu block size")

        return cls(
            gpu_tensors=parsed_gpu_tensors,
            bytes_per_kernel_block=[
                int(tensor.element_size()) * int(tensor.stride(0)) for tensor in parsed_gpu_tensors
            ],
            storage_block_size_factor=storage_block_size // gpu_block_size,
            pin_memory=is_pin_memory_available(),
        )

    def allocate_staging_tensors(self, slot_count: int) -> list[Any]:
        if not TORCH_COPY_AVAILABLE:
            raise RuntimeError("vLLM and torch are required for GPU copy support")
        num_kernel_blocks = slot_count * self.storage_block_size_factor
        staging_tensors: list[Any] = []
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
    def _byte_array_view(tensor: Any) -> np.ndarray:
        array = tensor.view(torch.uint8).numpy()
        if not array.flags.c_contiguous:
            raise ValueError("staging tensor slice must be contiguous")
        return array.reshape(-1)

    def storage_slot_view(
        self,
        staging_tensors: list[Any],
        slot_index: int,
    ) -> np.ndarray[Any, np.dtype[np.uint8]]:
        start = slot_index * self.storage_block_size_factor
        end = start + self.storage_block_size_factor
        view = self._byte_array_view(staging_tensors[0][start:end])
        if view.nbytes != self.storage_block_bytes:
            raise ValueError(
                f"staging slot view size {view.nbytes} does not match storage "
                f"block size {self.storage_block_bytes}"
            )
        return view


def block_ids_of(spec: Any) -> np.ndarray:
    block_ids = getattr(spec, "block_ids", None)
    if block_ids is None:
        raise ValueError("GPU transfer spec is missing block_ids")
    return np.asarray(block_ids, dtype=np.int64)


def safe_num_blocks(spec: Any) -> int:
    block_ids = getattr(spec, "block_ids", None)
    if block_ids is None:
        return 0
    try:
        return int(len(block_ids))
    except TypeError:
        return 0


def first_group_block_index(dst_spec: Any) -> int:
    group_sizes = getattr(dst_spec, "group_sizes", None)
    if group_sizes is not None and len(group_sizes) != 1:
        raise ValueError(
            "py-kvcache shared-storage offload currently supports exactly one KV cache group"
        )
    block_indices = getattr(dst_spec, "block_indices", None)
    if block_indices is None:
        return 0
    if len(block_indices) != 1:
        raise ValueError(
            "py-kvcache shared-storage offload currently supports exactly one KV cache group"
        )
    return int(block_indices[0])


def split_block_ids_for_files(
    layout: ParsedKvLayout,
    block_ids: np.ndarray,
    file_count: int,
    first_block_index: int = 0,
) -> list[np.ndarray]:
    if file_count == 0:
        return []

    blocks_per_file = layout.gpu_blocks_per_storage_block
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


def _merged_duration_ns(samples: list[Sample]) -> int:
    intervals = sorted((sample[0], sample[0] + sample[1]) for sample in samples if sample[1] > 0)
    if not intervals:
        return 0
    merged_start, merged_end = intervals[0]
    total = 0
    for start_ns, end_ns in intervals[1:]:
        if start_ns <= merged_end:
            merged_end = max(merged_end, end_ns)
        else:
            total += merged_end - merged_start
            merged_start, merged_end = start_ns, end_ns
    return total + (merged_end - merged_start)


def _bandwidth(samples: list[Sample]) -> float:
    duration_ns = _merged_duration_ns(samples)
    nbytes = sum(sample[2] for sample in samples)
    return round((nbytes / duration_ns) if duration_ns > 0 else 0.0, 3)


def emit_transfer_events(
    layout: ParsedKvLayout,
    profile: _TransferProfile,
    *,
    success: bool,
    transfer_size: int,
    file_samples: list[Sample],
    cuda_samples: list[Sample],
    num_files: int,
    num_blocks: int,
) -> None:
    end_ns = now_ns()
    add_event(
        "py_kvcache.transfer",
        "kv_offload",
        profile.start_ns,
        end_ns - profile.start_ns,
        tid=profile.profile_tid,
        args={
            "job_id": profile.job_id,
            "req_id": profile.req_id,
            "direction": profile.direction,
            "success": success,
            "num_bytes": transfer_size,
            "num_files": num_files,
            "num_blocks": num_blocks,
            "file_io_bw_GBps": _bandwidth(file_samples),
        },
    )
    file_event = (
        "py_kvcache.file_write" if profile.direction == "gpu_to_storage" else "py_kvcache.file_read"
    )
    for sample_idx, sample in enumerate(file_samples):
        start_ns, duration_ns, nbytes = sample[0], sample[1], sample[2]
        add_event(
            file_event,
            "fs",
            start_ns,
            duration_ns,
            tid=profile.profile_tid,
            args={
                "job_id": profile.job_id,
                "req_id": profile.req_id,
                "sample": sample_idx,
                "num_bytes": nbytes,
            },
        )
    for sample_idx, sample in enumerate(cuda_samples):
        start_ns, duration_ns, nbytes = sample[0], sample[1], sample[2]
        add_event(
            "py_kvcache.cuda_staging",
            "cuda",
            start_ns,
            duration_ns,
            tid=profile.profile_tid,
            args={
                "job_id": profile.job_id,
                "req_id": profile.req_id,
                "sample": sample_idx,
                "direction": profile.direction,
                "num_bytes": nbytes,
                "batch_size": max(1, nbytes // layout.storage_block_bytes),
            },
        )


__all__ = [
    "ParsedKvLayout",
    "Sample",
    "TORCH_COPY_AVAILABLE",
    "block_ids_of",
    "build_block_mapping",
    "emit_transfer_events",
    "first_group_block_index",
    "safe_num_blocks",
    "split_block_ids_for_files",
]
