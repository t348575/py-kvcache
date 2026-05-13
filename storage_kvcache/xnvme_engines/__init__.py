from __future__ import annotations

from vllm.v1.kv_offload.worker.worker import OffloadingHandler

from ..file_mapper import FileMapper
from ..kvikio_file import KvikioFileStore
from ..vllm_xnvme import ParsedKvLayout
from ..xnvme_file import XNvmeFileStore
from ..liburing_file import LiburingFileStore
from .kvikio import KvikioDirectOffloadingHandler
from .liburing_single_threaded import LiburingSingleThreadedOffloadingHandler
from .multithreaded import MultithreadedXNvmeOffloadingHandler
from .single_threaded import SingleThreadedXNvmeOffloadingHandler


def normalize_xnvme_engine_name(engine: str | None) -> str:
    normalized = (engine or "multithreaded").strip().lower().replace("-", "_")
    if normalized in {"threaded", "multi_threaded", "multithreaded"}:
        return "multithreaded"
    if normalized in {"single", "single_thread", "single_threaded"}:
        return "single_threaded"
    if normalized in {
        "uring",
        "liburing",
        "io_uring",
        "liburing_single",
        "liburing_single_thread",
        "liburing_single_threaded",
    }:
        return "liburing_single_threaded"
    if normalized in {
        "uring_threaded",
        "uring_multithreaded",
        "uring_multi_threaded",
        "liburing_threaded",
        "liburing_multithreaded",
        "liburing_multi_threaded",
        "io_uring_threaded",
        "io_uring_multithreaded",
    }:
        return "liburing_multithreaded"
    if normalized in {"kvikio", "gds", "gpu_direct_storage"}:
        return "kvikio"
    raise ValueError(
        "xnvme_engine must be one of: multithreaded, single_threaded, liburing_single_threaded, liburing_multithreaded, kvikio"
    )


def create_xnvme_offloading_handler(
    *,
    engine: str | None,
    layout: ParsedKvLayout,
    file_mapper: FileMapper,
    file_store: XNvmeFileStore | LiburingFileStore | KvikioFileStore,
    staging_slot_capacity: int,
) -> OffloadingHandler:
    engine_name = normalize_xnvme_engine_name(engine)
    handler_cls = {
        "multithreaded": MultithreadedXNvmeOffloadingHandler,
        "single_threaded": SingleThreadedXNvmeOffloadingHandler,
        "liburing_single_threaded": LiburingSingleThreadedOffloadingHandler,
        "liburing_multithreaded": MultithreadedXNvmeOffloadingHandler,
        "kvikio": KvikioDirectOffloadingHandler,
    }[engine_name]
    return handler_cls(
        layout=layout,
        file_mapper=file_mapper,
        file_store=file_store,
        staging_slot_capacity=staging_slot_capacity,
    )


__all__ = ["create_xnvme_offloading_handler", "normalize_xnvme_engine_name"]
