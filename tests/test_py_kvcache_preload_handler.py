from __future__ import annotations

import unittest
from concurrent.futures import Future
from typing import Any

from py_kvcache.reactor import LoadDeclined
from py_kvcache.vllm import (
    GPULoadStoreSpec,
    NoopSharedStorageOffloadingHandler,
    SharedStorageLoadStoreSpec,
)


class FakeCoordinator:
    def __init__(self) -> None:
        self.preloads: list[tuple[list[bytes], str, str, str]] = []
        self.loads: list[tuple[Any, Any, int, str, str]] = []
        self.shutdown_called = False

    def submit_preload(
        self,
        block_hashes: list[bytes],
        *,
        preload_id: str = "",
        req_id: str = "",
        profile_tid: str = "kv_preload",
    ) -> None:
        self.preloads.append((block_hashes, preload_id, req_id, profile_tid))

    def submit_load(
        self,
        src_spec: Any,
        dst_spec: Any,
        *,
        job_id: int,
        profile_tid: str,
        req_id: str,
    ) -> Future[int]:
        self.loads.append((src_spec, dst_spec, job_id, profile_tid, req_id))
        future: Future[int] = Future()
        future.set_result(123)
        return future

    def shutdown(self) -> None:
        self.shutdown_called = True


def make_gpu_spec() -> Any:
    try:
        return GPULoadStoreSpec([1], group_sizes=[1], block_indices=[0])
    except TypeError:
        return GPULoadStoreSpec()


class PreloadHandlerTest(unittest.TestCase):
    def test_preload_async_is_idempotent_by_preload_id(self) -> None:
        coordinator = FakeCoordinator()
        handler = NoopSharedStorageOffloadingHandler(coordinator=coordinator)
        spec = SharedStorageLoadStoreSpec([b"a", b"b"])

        self.assertTrue(handler.preload_async("p0", spec, profile_tid="req_0", req_id="r0"))
        self.assertFalse(handler.preload_async("p0", spec, profile_tid="req_0", req_id="r0"))
        self.assertEqual(coordinator.preloads, [([b"a", b"b"], "p0", "r0", "req_0")])

    def test_load_from_preload_async_submits_real_load(self) -> None:
        coordinator = FakeCoordinator()
        handler = NoopSharedStorageOffloadingHandler(coordinator=coordinator)
        src_spec = SharedStorageLoadStoreSpec([b"a"])
        dst_spec = make_gpu_spec()

        self.assertTrue(
            handler.load_from_preload_async(
                7,
                "p0",
                src_spec,
                dst_spec,
                profile_tid="req_0",
                req_id="r0",
            )
        )

        self.assertEqual(len(coordinator.loads), 1)
        self.assertEqual(coordinator.loads[0][2:], (7, "req_0", "r0"))
        finished = handler.get_finished()
        self.assertEqual(len(finished), 1)
        self.assertEqual(finished[0].job_id, 7)
        self.assertTrue(finished[0].success)


class DecliningCoordinator(FakeCoordinator):
    def __init__(self, block_ids: list[int]) -> None:
        super().__init__()
        self._block_ids = block_ids

    def submit_load(
        self,
        src_spec: Any,
        dst_spec: Any,
        *,
        job_id: int,
        profile_tid: str,
        req_id: str,
    ) -> Future[int]:
        self.loads.append((src_spec, dst_spec, job_id, profile_tid, req_id))
        future: Future[int] = Future()
        future.set_exception(LoadDeclined(list(self._block_ids)))
        return future


class DeclineHandlerTest(unittest.TestCase):
    def test_declined_load_completes_and_surfaces_block_ids(self) -> None:
        coordinator = DecliningCoordinator([5, 6])
        handler = NoopSharedStorageOffloadingHandler(coordinator=coordinator)
        src_spec = SharedStorageLoadStoreSpec([b"a"])
        dst_spec = make_gpu_spec()

        self.assertTrue(
            handler.transfer_async(9, (src_spec, dst_spec), profile_tid="req_0", req_id="r0")
        )
        finished = handler.get_finished()
        self.assertEqual(len(finished), 1)
        self.assertEqual(finished[0].job_id, 9)
        # Declined is not a failure: the job completes so the request resumes.
        self.assertTrue(finished[0].success)
        # Block ids are surfaced once, then drained.
        self.assertEqual(handler.get_declined_block_ids(), {5, 6})
        self.assertEqual(handler.get_declined_block_ids(), set())


if __name__ == "__main__":
    unittest.main()
