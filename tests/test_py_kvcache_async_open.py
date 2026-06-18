"""Tests for the py_kvcache async-open (io_uring OPENAT/CLOSE) path.

Pure config-coercion tests run anywhere. The ring tests need ``numpy`` and a
kernel with working ``io_uring`` + ``IORING_OP_OPENAT``; they skip when either
is missing (e.g. the minimal pure-Python workspace).
"""

from __future__ import annotations

import ctypes
import os
import tempfile
import unittest

try:
    import numpy  # noqa: F401

    from py_kvcache import liburing_file as lf
    from py_kvcache.fs_config import SharedFileConfig

    HAVE_DEPS = True
except Exception:  # pragma: no cover - minimal workspace
    HAVE_DEPS = False


@unittest.skipUnless(HAVE_DEPS, "numpy / py_kvcache not importable")
class OpenLookaheadConfigTest(unittest.TestCase):
    BASE = {"shared_storage_path": "/tmp/x"}

    def test_default_is_none(self):
        self.assertIsNone(SharedFileConfig.from_extra_config(self.BASE).open_lookahead)

    def test_parses_int_and_str(self):
        self.assertEqual(
            SharedFileConfig.from_extra_config({**self.BASE, "open_lookahead": 32}).open_lookahead,
            32,
        )
        self.assertEqual(
            SharedFileConfig.from_extra_config({**self.BASE, "open_lookahead": "8"}).open_lookahead,
            8,
        )

    def test_non_positive_rejected(self):
        for bad in (0, -4):
            with self.assertRaises(ValueError):
                SharedFileConfig.from_extra_config({**self.BASE, "open_lookahead": bad})


def _try_ring(depth: int = 16):
    try:
        return lf.LiburingRing(depth, ring_id=0)
    except OSError as exc:  # io_uring unavailable
        raise unittest.SkipTest(f"io_uring unavailable: {exc}")


@unittest.skipUnless(HAVE_DEPS, "numpy / py_kvcache not importable")
class RingOpEncodingTest(unittest.TestCase):
    def test_openat_and_close_sqe_fields(self):
        ring = _try_ring()
        try:
            path = ctypes.create_string_buffer(b"/tmp/foo.bin\0")
            tail = int(ring._sq_tail[0])
            ring.queue_openat(
                user_data=0xABCD,
                path_ptr=ctypes.addressof(path),
                open_flags=os.O_RDONLY | os.O_DIRECT,
                mode=0,
            )
            sqe = ring._sqes[tail & int(ring._sq_mask[0])]
            self.assertEqual(sqe.opcode, lf._IORING_OP_OPENAT)
            self.assertEqual(sqe.fd, lf._AT_FDCWD)
            self.assertEqual(sqe.addr, ctypes.addressof(path))
            self.assertEqual(sqe.rw_flags, os.O_RDONLY | os.O_DIRECT)
            self.assertEqual(sqe.user_data, 0xABCD)

            tail2 = int(ring._sq_tail[0])
            ring.queue_close(user_data=0x1234, fd=7)
            sqe2 = ring._sqes[tail2 & int(ring._sq_mask[0])]
            self.assertEqual(sqe2.opcode, lf._IORING_OP_CLOSE)
            self.assertEqual(sqe2.fd, 7)
        finally:
            ring.close()

    def test_queue_auto_flushes_when_sq_full(self):
        # A single pump can queue more SQEs than the ring holds; _begin_sqe must
        # flush to the kernel and keep going, not raise (the reactor-freeze bug).
        ring = _try_ring(depth=8)
        devnull = os.open(os.devnull, os.O_RDONLY)
        try:
            sq_entries = int(ring._params.sq_entries)
            cq_entries = int(ring._params.cq_entries)
            # More than the SQ holds (forces auto-flush) but within the CQ so no
            # completion is dropped before we reap.
            n = min(sq_entries * 2, cq_entries - 1)
            self.assertGreater(n, sq_entries)
            for i in range(n):
                # Unique valid fds so each close has real work to complete.
                ring.queue_close(user_data=i, fd=os.dup(devnull))
            ring.submit_pending()
            reaped = 0
            for _ in range(1000):
                reaped += len(ring.poll_all())
                if reaped >= n:
                    break
            self.assertEqual(reaped, n)
        finally:
            os.close(devnull)
            ring.close()


@unittest.skipUnless(HAVE_DEPS, "numpy / py_kvcache not importable")
class AsyncOpenEndToEndTest(unittest.TestCase):
    ALIGN = 4096

    def _aligned(self):
        libc = ctypes.CDLL(None, use_errno=True)
        buf = ctypes.c_void_p()
        rc = libc.posix_memalign(
            ctypes.byref(buf), ctypes.c_size_t(self.ALIGN), ctypes.c_size_t(self.ALIGN)
        )
        self.assertEqual(rc, 0)
        return buf

    def _poll(self, ring):
        for _ in range(1_000_000):
            done = ring.poll_all()
            if done:
                return done
        self.fail("completion never arrived")

    def test_openat_read_close_roundtrip(self):
        ring = _try_ring()
        path = tempfile.mktemp(suffix=".bin")
        payload = bytes((i % 251) for i in range(self.ALIGN))
        try:
            wbuf = self._aligned()
            ctypes.memmove(wbuf, payload, self.ALIGN)
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_DIRECT, 0o644)
            os.pwrite(fd, (ctypes.c_char * self.ALIGN).from_address(wbuf.value), 0)
            os.close(fd)

            pbuf = ctypes.create_string_buffer(os.fsencode(path) + b"\0")
            ring.queue_openat(
                user_data=1,
                path_ptr=ctypes.addressof(pbuf),
                open_flags=os.O_RDONLY | os.O_DIRECT,
                mode=0,
            )
            ring.submit_pending()
            ud, res = self._poll(ring)[0]
            self.assertEqual(ud, 1)
            self.assertGreaterEqual(res, 0, f"openat failed res={res}")
            opened_fd = res

            rbuf = self._aligned()
            ctypes.memset(rbuf, 0, self.ALIGN)
            ring.queue_rw(user_data=2, fd=opened_fd, ptr=rbuf, nbytes=self.ALIGN, write=False)
            ring.submit_pending()
            ud, res = self._poll(ring)[0]
            self.assertEqual((ud, res), (2, self.ALIGN))
            readback = bytes((ctypes.c_char * self.ALIGN).from_address(rbuf.value))
            self.assertEqual(readback, payload)

            ring.queue_close(user_data=3, fd=opened_fd)
            ring.submit_pending()
            ud, res = self._poll(ring)[0]
            self.assertEqual(ud, 3)
            self.assertEqual(res, 0)
        finally:
            ring.close()
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass


if __name__ == "__main__":
    unittest.main()
