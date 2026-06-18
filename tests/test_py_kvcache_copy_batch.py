"""Unit tests for the reactor's fused load-copy path (_flush_copy_batch /
_drain_cuda_copies).

The real path launches ``swap_blocks_batch`` on a CUDA stream; here a fake CUDA
surface (events / streams / the op) lets the test run on a CPU-only workspace.
The test checks the pure coalescing logic: N queued load copies collapse to ONE
launch, the fused src/dst/size pointer arrays come out correct, and on
completion the reactor releases every slot once and resolves every covered
file's job future.
"""

from __future__ import annotations

import contextlib
import unittest
from concurrent.futures import Future
from types import SimpleNamespace

try:
    import numpy as np
    import torch as real_torch

    from py_kvcache import reactor as reactor_mod
    from py_kvcache.reactor import IoReactor, _ReactorJob, _ReadyCopy
    from py_kvcache.transfer import _TransferProfile

    HAVE_DEPS = True
except Exception:  # pragma: no cover - minimal workspace
    HAVE_DEPS = False


class _FakeEvent:
    def __init__(self, enable_timing: bool = False) -> None:
        self.recorded = False

    def record(self, stream) -> None:
        self.recorded = True

    def query(self) -> bool:
        return True

    def elapsed_time(self, other: "_FakeEvent") -> float:
        return 1.0  # ms -> 1_000_000 ns


class _FakeCuda:
    Event = _FakeEvent

    @staticmethod
    def stream(stream):
        return contextlib.nullcontext()


class _FakeTorch:
    from_numpy = staticmethod(real_torch.from_numpy) if HAVE_DEPS else None
    cuda = _FakeCuda


class _RecordingOps:
    def __init__(self) -> None:
        self.calls: list[tuple[list[int], list[int], list[int]]] = []

    def swap_blocks_batch(self, src_t, dst_t, sizes_t) -> None:
        self.calls.append((src_t.tolist(), dst_t.tolist(), sizes_t.tolist()))


class _Pool:
    def __init__(self) -> None:
        self.released: list[int] = []

    def release(self, idx: int) -> None:
        self.released.append(idx)


def _bare_reactor(*, block_bytes: int, storage_block_bytes: int, slots: int):
    r = IoReactor.__new__(IoReactor)
    r.layout = SimpleNamespace(
        gpu_tensors=[SimpleNamespace(data_ptr=lambda: 1_000_000)],
        bytes_per_kernel_block=[block_bytes],
        storage_block_bytes=storage_block_bytes,
        storage_block_size_factor=storage_block_bytes // block_bytes,
        staging_layer_offsets=[0],
    )
    r._staging_base_ptr = 2_000_000
    r._mapping_buffers = [np.zeros((4, 2), dtype=np.int64) for _ in range(slots)]
    r._copy_ready = []
    r._pending_copies = []
    r._copy_stream = object()
    r._fused_capacity = 0
    r._fused_src_np = None
    r._fused_dst_np = None
    r._fused_sizes_np = None
    r._fused_src_t = None
    r._fused_dst_t = None
    r._fused_sizes_t = None
    r.staging_pool = _Pool()
    r._staging_cache = None
    return r


def _job(job_id: int, total_files: int) -> _ReactorJob:
    return _ReactorJob(
        job_id=job_id,
        is_store=False,
        block_hashes=[b"h"] * total_files,
        block_chunks=[np.asarray([0], dtype=np.int64)] * total_files,
        profile=_TransferProfile(
            job_id=job_id,
            direction="storage_to_gpu",
            profile_tid="kv_load",
            req_id=f"req_{job_id}",
            start_ns=0,
        ),
        future=Future(),
        total_files=total_files,
        transfer_size=total_files,
        num_blocks=total_files,
    )


@unittest.skipUnless(HAVE_DEPS, "numpy / torch / py_kvcache not importable")
class FusedCopyTest(unittest.TestCase):
    def setUp(self) -> None:
        self._orig_torch = reactor_mod.torch
        self._orig_ops = reactor_mod.ops
        self.ops = _RecordingOps()
        reactor_mod.torch = _FakeTorch
        reactor_mod.ops = self.ops

    def tearDown(self) -> None:
        reactor_mod.torch = self._orig_torch
        reactor_mod.ops = self._orig_ops

    def test_flush_coalesces_into_one_launch(self):
        block_bytes = 16
        r = _bare_reactor(block_bytes=block_bytes, storage_block_bytes=64, slots=4)
        job = _job(1, total_files=3)
        job.inflight_files = 3
        for slot in range(3):
            r._mapping_buffers[slot][:2] = [
                [slot * 4 + 0, 10 + slot * 2 + 0],
                [slot * 4 + 1, 10 + slot * 2 + 1],
            ]
            r._copy_ready.append(
                _ReadyCopy(
                    job=job,
                    file_index=slot,
                    slot_index=slot,
                    mapping_len=2,
                    nbytes=64,
                    mapping=r._mapping_buffers[slot][:2].copy(),
                )
            )

        made = r._flush_copy_batch()
        self.assertTrue(made)
        self.assertEqual(r._copy_ready, [])
        # Exactly one fused launch covering all 3 * 2 = 6 entries.
        self.assertEqual(len(self.ops.calls), 1)
        src, dst, sizes = self.ops.calls[0]
        self.assertEqual(len(src), 6)
        self.assertEqual(sizes, [block_bytes] * 6)
        # Pointer math: src = staging_base + map[:,0]*block_bytes,
        #               dst = gpu_base    + map[:,1]*block_bytes.
        expect_src, expect_dst = [], []
        for slot in range(3):
            for row in range(2):
                m = r._mapping_buffers[slot][row]
                expect_src.append(2_000_000 + int(m[0]) * block_bytes)
                expect_dst.append(1_000_000 + int(m[1]) * block_bytes)
        self.assertEqual(src, expect_src)
        self.assertEqual(dst, expect_dst)
        # One batched _CopyOp parked with all members, no per-file copies.
        self.assertEqual(len(r._pending_copies), 1)
        op = r._pending_copies[0]
        self.assertFalse(op.is_store)
        self.assertEqual(
            op.members,
            [(0, job, 0, None, None), (1, job, 1, None, None), (2, job, 2, None, None)],
        )

    def test_drain_releases_all_slots_and_finishes_job(self):
        r = _bare_reactor(block_bytes=16, storage_block_bytes=64, slots=4)
        job = _job(2, total_files=2)
        job.inflight_files = 2
        for slot in range(2):
            r._mapping_buffers[slot][:1] = [[slot, slot]]
            r._copy_ready.append(
                _ReadyCopy(
                    job=job,
                    file_index=slot,
                    slot_index=slot,
                    mapping_len=1,
                    nbytes=64,
                    mapping=r._mapping_buffers[slot][:1].copy(),
                )
            )
        r._flush_copy_batch()
        self.assertFalse(job.future.done())

        made = r._drain_cuda_copies()
        self.assertTrue(made)
        # Both slots released exactly once.
        self.assertEqual(sorted(r.staging_pool.released), [0, 1])
        # Job future resolved with its transfer size.
        self.assertTrue(job.future.done())
        self.assertEqual(job.future.result(), job.transfer_size)
        self.assertEqual(job.done_files, 2)
        # One cuda sample for the job, batch_size encoded via nbytes.
        self.assertEqual(len(job.cuda_samples), 1)
        _start, _dur, nbytes, _ = job.cuda_samples[0]
        self.assertEqual(nbytes, 2 * r.layout.storage_block_bytes)

    def test_flush_spanning_two_jobs(self):
        r = _bare_reactor(block_bytes=8, storage_block_bytes=32, slots=4)
        job_a = _job(3, total_files=1)
        job_b = _job(4, total_files=1)
        job_a.inflight_files = 1
        job_b.inflight_files = 1
        r._mapping_buffers[0][:1] = [[0, 0]]
        r._mapping_buffers[1][:1] = [[1, 1]]
        r._copy_ready.append(
            _ReadyCopy(
                job=job_a,
                file_index=0,
                slot_index=0,
                mapping_len=1,
                nbytes=32,
                mapping=r._mapping_buffers[0][:1].copy(),
            )
        )
        r._copy_ready.append(
            _ReadyCopy(
                job=job_b,
                file_index=0,
                slot_index=1,
                mapping_len=1,
                nbytes=32,
                mapping=r._mapping_buffers[1][:1].copy(),
            )
        )
        r._flush_copy_batch()
        r._drain_cuda_copies()
        self.assertTrue(job_a.future.done())
        self.assertTrue(job_b.future.done())
        self.assertEqual(sorted(r.staging_pool.released), [0, 1])
        # One cuda sample per job.
        self.assertEqual(len(job_a.cuda_samples), 1)
        self.assertEqual(len(job_b.cuda_samples), 1)

    def test_flush_empty_is_noop(self):
        r = _bare_reactor(block_bytes=8, storage_block_bytes=32, slots=2)
        self.assertFalse(r._flush_copy_batch())
        self.assertEqual(self.ops.calls, [])


if __name__ == "__main__":
    unittest.main()
