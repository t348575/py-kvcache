from __future__ import annotations

from vllm.v1.kv_offload.worker.worker import (
    OffloadingHandler,
    TransferResult,
    TransferSpec,
)

from ..file_mapper import FileMapper
from ..vllm_xnvme import ParsedKvLayout
from ..xnvme_file import XNvmeFileStore


class SingleThreadedXNvmeOffloadingHandler(OffloadingHandler):
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

    def transfer_async(
        self,
        job_id: int,
        spec: TransferSpec,
        profile_tid: str = "kv_transfer",
        req_id: str = "",
    ) -> bool:
        raise NotImplementedError("single-threaded xNVMe engine is not implemented yet")

    def get_finished(self) -> list[TransferResult]:
        return []

    def wait(self, job_ids: set[int]) -> None:
        if job_ids:
            raise NotImplementedError(
                "single-threaded xNVMe engine is not implemented yet"
            )

    def shutdown(self) -> None:
        self.file_store.close()


__all__ = ["SingleThreadedXNvmeOffloadingHandler"]
