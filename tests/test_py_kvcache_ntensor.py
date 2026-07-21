"""Pure-Python coverage for the N per-layer canonical tensor path (Case A).

py-kvcache accepts a single KV cache group whose layers are *separate* tensors
(the per-layer layout vLLM emits when it does not fuse layers into one
cross-layer tensor). Staging is one flat buffer per slot; each layer's region
lives at ``staging_layer_offsets`` within the slot and the copy path addresses
it by raw byte pointer. No CUDA here: ``_fill_swap_ptrs`` builds plain numpy
pointer arrays, so a numpy memcpy stands in for ``swap_blocks_batch`` to prove a
store -> staging -> load round trip reconstructs the original per-layer bytes.
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

try:
    import numpy as np

    from py_kvcache.reactor import IoReactor
    from py_kvcache import transfer
    from py_kvcache.liburing_file import DIRECT_IO_ALIGNMENT
    from py_kvcache.transfer import ParsedKvLayout

    HAVE_DEPS = True
except Exception:  # pragma: no cover - minimal workspace
    HAVE_DEPS = False


@unittest.skipUnless(HAVE_DEPS, "numpy / py_kvcache not importable")
class StagingLayerOffsetsTest(unittest.TestCase):
    def _layout(self, block_bytes, factor):
        return ParsedKvLayout(
            gpu_tensors=[None] * len(block_bytes),
            bytes_per_kernel_block=list(block_bytes),
            storage_block_size_factor=factor,
            pin_memory=False,
        )

    def test_single_tensor_collapses(self):
        layout = self._layout([16], factor=4)
        self.assertEqual(layout.staging_layer_offsets, [0])
        self.assertEqual(layout.storage_block_bytes, 64)

    def test_uniform_layers(self):
        layout = self._layout([16, 16], factor=2)
        self.assertEqual(layout.staging_layer_offsets, [0, 32])
        self.assertEqual(layout.storage_block_bytes, 2 * (16 + 16))

    def test_ragged_layers(self):
        layout = self._layout([16, 8], factor=2)
        self.assertEqual(layout.staging_layer_offsets, [0, 32])
        self.assertEqual(layout.storage_block_bytes, 2 * (16 + 8))


@unittest.skipUnless(HAVE_DEPS, "numpy / py_kvcache not importable")
class StagingBufferAlignmentTest(unittest.TestCase):
    def test_all_slots_are_direct_io_aligned(self):
        layout = ParsedKvLayout(
            gpu_tensors=[None],
            bytes_per_kernel_block=[DIRECT_IO_ALIGNMENT],
            storage_block_size_factor=1,
            pin_memory=False,
        )
        real_empty = transfer.torch.empty

        def misaligned_empty(shape, **kwargs):
            numel = int(np.prod(shape))
            backing = real_empty((numel + 1,), **kwargs)
            return backing.narrow(0, 1, numel).view(shape)

        with patch.object(transfer.torch, "empty", side_effect=misaligned_empty):
            buffer = layout.allocate_staging_buffer(slot_count=3)

        self.assertEqual(buffer.data_ptr() % DIRECT_IO_ALIGNMENT, 0)
        for slot in buffer:
            self.assertEqual(slot.data_ptr() % DIRECT_IO_ALIGNMENT, 0)

    def test_rejects_unaligned_slot_stride(self):
        layout = ParsedKvLayout(
            gpu_tensors=[None],
            bytes_per_kernel_block=[DIRECT_IO_ALIGNMENT - 1],
            storage_block_size_factor=1,
            pin_memory=False,
        )

        with self.assertRaisesRegex(ValueError, "not divisible"):
            layout.allocate_staging_buffer(slot_count=1)


@unittest.skipUnless(HAVE_DEPS, "numpy / py_kvcache not importable")
class LayeredPointerFillTest(unittest.TestCase):
    """Drive _fill_swap_ptrs over fake buffers and emulate the swap with memcpy."""

    STAGING_BASE = 1 << 40
    GPU_BASE = [1 << 41, 1 << 42]

    def _reactor(self, block_bytes, factor, num_gpu_blocks, slot_count):
        layout = ParsedKvLayout(
            gpu_tensors=[
                SimpleNamespace(data_ptr=lambda b=base: b) for base in self.GPU_BASE[: len(block_bytes)]
            ],
            bytes_per_kernel_block=list(block_bytes),
            storage_block_size_factor=factor,
            pin_memory=False,
        )
        r = IoReactor.__new__(IoReactor)
        r.layout = layout
        r._staging_base_ptr = self.STAGING_BASE
        # Backing byte buffers the emulated swap reads/writes.
        self.staging = bytearray(slot_count * layout.storage_block_bytes)
        self.gpu = [bytearray(num_gpu_blocks * bb) for bb in block_bytes]
        return r

    def _resolve(self, ptr):
        if self.STAGING_BASE <= ptr < self.STAGING_BASE + len(self.staging):
            return self.staging, ptr - self.STAGING_BASE
        for layer, base in enumerate(self.GPU_BASE[: len(self.gpu)]):
            if base <= ptr < base + len(self.gpu[layer]):
                return self.gpu[layer], ptr - base
        raise AssertionError(f"pointer {ptr} resolves to no buffer")

    def _emulate(self, src, dst, sizes):
        for s_ptr, d_ptr, n in zip(src.tolist(), dst.tolist(), sizes.tolist()):
            s_buf, s_off = self._resolve(int(s_ptr))
            d_buf, d_off = self._resolve(int(d_ptr))
            d_buf[d_off : d_off + int(n)] = s_buf[s_off : s_off + int(n)]

    def test_store_then_load_round_trip(self):
        block_bytes = [16, 16]
        factor = 2
        slot = 1
        gpu_blocks = np.asarray([5, 6], dtype=np.int64)
        r = self._reactor(block_bytes, factor, num_gpu_blocks=8, slot_count=3)

        # Distinct pattern per (layer, gpu block) so a mis-offset is visible.
        for layer, bb in enumerate(block_bytes):
            for b in gpu_blocks.tolist():
                self.gpu[layer][b * bb : (b + 1) * bb] = bytes(
                    [(layer * 100 + b * 7 + k) & 0xFF for k in range(bb)]
                )
        original = [bytes(buf) for buf in self.gpu]

        cap = 64
        src = np.zeros(cap, dtype=np.int64)
        dst = np.zeros(cap, dtype=np.int64)
        sizes = np.zeros(cap, dtype=np.int64)

        # STORE: gpu -> staging. mapping col0 = gpu block, col1 = staging global.
        staging_global = slot * factor + np.arange(len(gpu_blocks), dtype=np.int64)
        store_map = np.stack([gpu_blocks, staging_global], axis=1)
        n = r._fill_swap_ptrs(store_map, slot, False, src, dst, sizes, 0)
        self.assertEqual(n, len(gpu_blocks) * len(block_bytes))
        self._emulate(src[:n], dst[:n], sizes[:n])

        # LOAD back: staging -> gpu. col0 = staging global, col1 = gpu.
        for buf in self.gpu:
            for i in range(len(buf)):
                buf[i] = 0
        load_map = np.stack([staging_global, gpu_blocks], axis=1)
        n2 = r._fill_swap_ptrs(load_map, slot, True, src, dst, sizes, 0)
        self.assertEqual(n2, n)
        self._emulate(src[:n2], dst[:n2], sizes[:n2])

        for layer in range(len(block_bytes)):
            self.assertEqual(bytes(self.gpu[layer]), original[layer])

    def test_layers_land_in_distinct_staging_regions(self):
        block_bytes = [16, 16]
        factor = 2
        slot = 0
        gpu_blocks = np.asarray([0, 1], dtype=np.int64)
        r = self._reactor(block_bytes, factor, num_gpu_blocks=4, slot_count=1)
        for layer, bb in enumerate(block_bytes):
            for b in gpu_blocks.tolist():
                self.gpu[layer][b * bb : (b + 1) * bb] = bytes([(layer + 1)] * bb)

        cap = 32
        src, dst, sizes = (np.zeros(cap, dtype=np.int64) for _ in range(3))
        staging_global = slot * factor + np.arange(len(gpu_blocks), dtype=np.int64)
        store_map = np.stack([gpu_blocks, staging_global], axis=1)
        n = r._fill_swap_ptrs(store_map, slot, False, src, dst, sizes, 0)
        self._emulate(src[:n], dst[:n], sizes[:n])

        # Layer 0 occupies [0,32), layer 1 [32,64) of the slot's 64-byte region.
        self.assertEqual(set(self.staging[0:32]), {1})
        self.assertEqual(set(self.staging[32:64]), {2})


if __name__ == "__main__":
    unittest.main()
