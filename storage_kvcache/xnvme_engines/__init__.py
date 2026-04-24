from __future__ import annotations

from vllm.v1.kv_offload.worker.worker import OffloadingHandler

from ..file_mapper import FileMapper
from ..vllm_xnvme import ParsedKvLayout
from ..xnvme_file import XNvmeFileStore
from .multithreaded import MultithreadedXNvmeOffloadingHandler
from .single_threaded import SingleThreadedXNvmeOffloadingHandler


def normalize_xnvme_engine_name(engine: str | None) -> str:
    normalized = (engine or "multithreaded").strip().lower().replace("-", "_")
    if normalized in {"threaded", "multi_threaded", "multithreaded"}:
        return "multithreaded"
    if normalized in {"single", "single_thread", "single_threaded"}:
        return "single_threaded"
    raise ValueError(
        "xnvme_engine must be one of: multithreaded, single_threaded"
    )


def create_xnvme_offloading_handler(
    *,
    engine: str | None,
    layout: ParsedKvLayout,
    file_mapper: FileMapper,
    file_store: XNvmeFileStore,
    staging_slot_capacity: int,
) -> OffloadingHandler:
    engine_name = normalize_xnvme_engine_name(engine)
    handler_cls = {
        "multithreaded": MultithreadedXNvmeOffloadingHandler,
        "single_threaded": SingleThreadedXNvmeOffloadingHandler,
    }[engine_name]
    return handler_cls(
        layout=layout,
        file_mapper=file_mapper,
        file_store=file_store,
        staging_slot_capacity=staging_slot_capacity,
    )


__all__ = ["create_xnvme_offloading_handler", "normalize_xnvme_engine_name"]
