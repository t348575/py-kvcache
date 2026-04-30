from __future__ import annotations

import ctypes
import errno
import os
import platform
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from simple_profiler import profile_scope, profiler
from vllm.logger import init_logger

from .fs_config import SharedFileConfig

logger = init_logger("vllm.storage_kvcache.liburing_file")

_DEFAULT_URING_DEPTH = 256

_LIBC = ctypes.CDLL(None, use_errno=True)
_LIBC.syscall.restype = ctypes.c_long
_LIBC.mmap.restype = ctypes.c_void_p
_LIBC.mmap.argtypes = [
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_long,
]
_LIBC.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
_LIBC.munmap.restype = ctypes.c_int
_LIBC.posix_memalign.argtypes = [
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.c_size_t,
    ctypes.c_size_t,
]
_LIBC.posix_memalign.restype = ctypes.c_int
_LIBC.free.argtypes = [ctypes.c_void_p]
_LIBC.free.restype = None

_PROT_READ = 0x1
_PROT_WRITE = 0x2
_MAP_SHARED = 0x01
_MAP_POPULATE = 0x8000
_MAP_FAILED = ctypes.c_void_p(-1).value

_IORING_OFF_SQ_RING = 0
_IORING_OFF_CQ_RING = 0x8000000
_IORING_OFF_SQES = 0x10000000

_IORING_OP_READ = 22
_IORING_OP_WRITE = 23


def _syscall_numbers() -> tuple[int, int]:
    machine = platform.machine().lower()
    if machine in {"x86_64", "amd64", "aarch64", "arm64", "riscv64"}:
        return 425, 426
    raise RuntimeError(f"unsupported architecture for raw io_uring syscalls: {machine}")


_NR_IO_URING_SETUP, _NR_IO_URING_ENTER = _syscall_numbers()


class _IoSqringOffsets(ctypes.Structure):
    _fields_ = [
        ("head", ctypes.c_uint32),
        ("tail", ctypes.c_uint32),
        ("ring_mask", ctypes.c_uint32),
        ("ring_entries", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("dropped", ctypes.c_uint32),
        ("array", ctypes.c_uint32),
        ("resv1", ctypes.c_uint32),
        ("user_addr", ctypes.c_uint64),
    ]


class _IoCqringOffsets(ctypes.Structure):
    _fields_ = [
        ("head", ctypes.c_uint32),
        ("tail", ctypes.c_uint32),
        ("ring_mask", ctypes.c_uint32),
        ("ring_entries", ctypes.c_uint32),
        ("overflow", ctypes.c_uint32),
        ("cqes", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("resv1", ctypes.c_uint32),
        ("user_addr", ctypes.c_uint64),
    ]


class _IoUringParams(ctypes.Structure):
    _fields_ = [
        ("sq_entries", ctypes.c_uint32),
        ("cq_entries", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("sq_thread_cpu", ctypes.c_uint32),
        ("sq_thread_idle", ctypes.c_uint32),
        ("features", ctypes.c_uint32),
        ("wq_fd", ctypes.c_uint32),
        ("resv", ctypes.c_uint32 * 3),
        ("sq_off", _IoSqringOffsets),
        ("cq_off", _IoCqringOffsets),
    ]


class _IoUringSqe(ctypes.Structure):
    _fields_ = [
        ("opcode", ctypes.c_uint8),
        ("flags", ctypes.c_uint8),
        ("ioprio", ctypes.c_uint16),
        ("fd", ctypes.c_int32),
        ("off", ctypes.c_uint64),
        ("addr", ctypes.c_uint64),
        ("len", ctypes.c_uint32),
        ("rw_flags", ctypes.c_uint32),
        ("user_data", ctypes.c_uint64),
        ("buf_index", ctypes.c_uint16),
        ("personality", ctypes.c_uint16),
        ("splice_fd_in", ctypes.c_int32),
        ("addr3", ctypes.c_uint64),
        ("resv", ctypes.c_uint64),
    ]


class _IoUringCqe(ctypes.Structure):
    _fields_ = [
        ("user_data", ctypes.c_uint64),
        ("res", ctypes.c_int32),
        ("flags", ctypes.c_uint32),
    ]


@dataclass
class _AlignedIoBuffer:
    ptr: ctypes.c_void_p
    array: np.ndarray[Any, np.dtype[np.uint8]]


@dataclass(frozen=True)
class _TimedIoResult:
    start_ns: int
    duration_ns: int


@dataclass(frozen=True)
class _TimedStoreResult:
    stored: bool
    start_ns: int
    duration_ns: int
    parent_path: str | None = None


def _align_up(value: int, alignment: int) -> int:
    if alignment <= 0:
        raise ValueError("alignment must be positive")
    return ((value + alignment - 1) // alignment) * alignment


class LiburingRing:
    def __init__(self, depth: int):
        if depth <= 0:
            raise ValueError("uring_depth must be positive")
        self.depth = depth
        self._params = _IoUringParams()
        fd = _LIBC.syscall(
            ctypes.c_long(_NR_IO_URING_SETUP),
            ctypes.c_uint32(depth),
            ctypes.byref(self._params),
        )
        if fd < 0:
            err = ctypes.get_errno()
            raise OSError(err, f"io_uring_setup(depth={depth}) failed")
        self.fd = int(fd)

        sq_ring_size = self._params.sq_off.array + self._params.sq_entries * ctypes.sizeof(ctypes.c_uint32)
        cq_ring_size = self._params.cq_off.cqes + self._params.cq_entries * ctypes.sizeof(_IoUringCqe)
        self._sq_ring_size = sq_ring_size
        self._cq_ring_size = cq_ring_size
        self._sqes_size = self._params.sq_entries * ctypes.sizeof(_IoUringSqe)

        self._sq_ring = self._mmap(sq_ring_size, _IORING_OFF_SQ_RING)
        self._cq_ring = self._mmap(cq_ring_size, _IORING_OFF_CQ_RING)
        self._sqes_map = self._mmap(self._sqes_size, _IORING_OFF_SQES)

        self._sq_head = self._u32_ptr(self._sq_ring, self._params.sq_off.head)
        self._sq_tail = self._u32_ptr(self._sq_ring, self._params.sq_off.tail)
        self._sq_mask = self._u32_ptr(self._sq_ring, self._params.sq_off.ring_mask)
        self._sq_array = self._u32_array(self._sq_ring, self._params.sq_off.array, self._params.sq_entries)
        self._sqes = ctypes.cast(self._sqes_map, ctypes.POINTER(_IoUringSqe))

        self._cq_head = self._u32_ptr(self._cq_ring, self._params.cq_off.head)
        self._cq_tail = self._u32_ptr(self._cq_ring, self._params.cq_off.tail)
        self._cq_mask = self._u32_ptr(self._cq_ring, self._params.cq_off.ring_mask)
        self._cqes = ctypes.cast(
            ctypes.c_void_p(self._cq_ring.value + self._params.cq_off.cqes),
            ctypes.POINTER(_IoUringCqe),
        )
        self._completed: dict[int, int] = {}
        self._pending_submit = 0

    def queue_rw(
        self,
        *,
        user_data: int,
        fd: int,
        ptr: ctypes.c_void_p,
        nbytes: int,
        write: bool,
    ) -> None:
        head = int(self._sq_head[0])
        tail = int(self._sq_tail[0])
        if tail - head >= self._params.sq_entries:
            raise BlockingIOError(errno.EAGAIN, "io_uring submission queue is full")

        index = tail & int(self._sq_mask[0])
        sqe = self._sqes[index]
        ctypes.memset(ctypes.byref(sqe), 0, ctypes.sizeof(_IoUringSqe))
        sqe.opcode = _IORING_OP_WRITE if write else _IORING_OP_READ
        sqe.fd = fd
        sqe.off = 0
        sqe.addr = int(ptr.value or 0)
        sqe.len = nbytes
        sqe.user_data = user_data
        self._sq_array[index] = index
        self._sq_tail[0] = tail + 1
        self._pending_submit += 1

    def submit_pending(self) -> None:
        if self._pending_submit == 0:
            return

        submit_start_ns = time.perf_counter_ns()
        pending_submit = self._pending_submit
        submitted = _LIBC.syscall(
            ctypes.c_long(_NR_IO_URING_ENTER),
            ctypes.c_int(self.fd),
            ctypes.c_uint32(self._pending_submit),
            ctypes.c_uint32(0),
            ctypes.c_uint32(0),
            ctypes.c_void_p(0),
            ctypes.c_size_t(0),
        )
        if submitted < 0:
            err = ctypes.get_errno()
            raise OSError(err, "io_uring_enter(submit) failed")
        if profiler._active:
            profiler.add_event(
                "storage_kvcache.liburing.enter_submit",
                "fs",
                submit_start_ns,
                time.perf_counter_ns() - submit_start_ns,
                args={"pending": pending_submit, "submitted": int(submitted)},
            )
        self._pending_submit = max(0, self._pending_submit - int(submitted))

    def poll_any(self) -> tuple[int, int] | None:
        poll_start_ns = time.perf_counter_ns()
        if self._completed:
            user_data, result = self._completed.popitem()
            if profiler._active:
                profiler.add_event(
                    "storage_kvcache.liburing.poll_any_cached",
                    "fs",
                    poll_start_ns,
                    time.perf_counter_ns() - poll_start_ns,
                )
            return user_data, result

        if int(self._cq_head[0]) == int(self._cq_tail[0]):
            if profiler._active:
                profiler.add_event(
                    "storage_kvcache.liburing.poll_any_empty",
                    "fs",
                    poll_start_ns,
                    time.perf_counter_ns() - poll_start_ns,
                )
            return None

        head = int(self._cq_head[0])
        index = head & int(self._cq_mask[0])
        cqe = self._cqes[index]
        user_data = int(cqe.user_data)
        result = int(cqe.res)
        self._cq_head[0] = head + 1
        if profiler._active:
            profiler.add_event(
                "storage_kvcache.liburing.poll_any_complete",
                "fs",
                poll_start_ns,
                time.perf_counter_ns() - poll_start_ns,
            )
        return user_data, result

    def submit_rw(self, **kwargs: object) -> None:
        self.queue_rw(**kwargs)
        self.submit_pending()

    def run_rw(
        self,
        *,
        user_data: int,
        fd: int,
        ptr: ctypes.c_void_p,
        nbytes: int,
        write: bool,
    ) -> int:
        self.submit_rw(
            user_data=user_data,
            fd=fd,
            ptr=ptr,
            nbytes=nbytes,
            write=write,
        )
        while True:
            result = self.poll(user_data)
            if result is not None:
                return result

    def poll(self, user_data: int) -> int | None:
        poll_start_ns = time.perf_counter_ns()
        cached = self._completed.pop(user_data, None)
        if cached is not None:
            if profiler._active:
                profiler.add_event(
                    "storage_kvcache.liburing.poll_cached",
                    "fs",
                    poll_start_ns,
                    time.perf_counter_ns() - poll_start_ns,
                )
            return cached

        while int(self._cq_head[0]) != int(self._cq_tail[0]):
            head = int(self._cq_head[0])
            index = head & int(self._cq_mask[0])
            cqe = self._cqes[index]
            completed_user_data = int(cqe.user_data)
            result = int(cqe.res)
            self._cq_head[0] = head + 1
            if completed_user_data == user_data:
                if profiler._active:
                    profiler.add_event(
                        "storage_kvcache.liburing.poll_complete",
                        "fs",
                        poll_start_ns,
                        time.perf_counter_ns() - poll_start_ns,
                    )
                return result
            self._completed[completed_user_data] = result

        if profiler._active:
            profiler.add_event(
                "storage_kvcache.liburing.poll_empty",
                "fs",
                poll_start_ns,
                time.perf_counter_ns() - poll_start_ns,
            )
        return None

    def close(self) -> None:
        if getattr(self, "fd", -1) < 0:
            return
        _LIBC.munmap(self._sq_ring, self._sq_ring_size)
        _LIBC.munmap(self._cq_ring, self._cq_ring_size)
        _LIBC.munmap(self._sqes_map, self._sqes_size)
        os.close(self.fd)
        self.fd = -1

    def _mmap(self, size: int, offset: int) -> ctypes.c_void_p:
        ptr = _LIBC.mmap(
            ctypes.c_void_p(0),
            ctypes.c_size_t(size),
            _PROT_READ | _PROT_WRITE,
            _MAP_SHARED | _MAP_POPULATE,
            self.fd,
            offset,
        )
        if ptr == _MAP_FAILED:
            err = ctypes.get_errno()
            raise OSError(err, f"mmap(io_uring offset={offset}) failed")
        return ctypes.c_void_p(ptr)

    @staticmethod
    def _u32_ptr(base: ctypes.c_void_p, offset: int):
        return ctypes.cast(
            ctypes.c_void_p(base.value + offset), ctypes.POINTER(ctypes.c_uint32)
        )

    @staticmethod
    def _u32_array(base: ctypes.c_void_p, offset: int, count: int):
        return (ctypes.c_uint32 * count).from_address(base.value + offset)


class LiburingFileStore:
    def __init__(self, config: SharedFileConfig, *, payload_size: int):
        if payload_size <= 0:
            raise ValueError("payload_size must be positive")
        self.config = config
        self.payload_size = payload_size
        self.io_size = _align_up(payload_size, config.io_alignment)
        configured_depth = config.uring_depth or config.parallel_files or _DEFAULT_URING_DEPTH
        self.uring_depth = max(1, configured_depth)
        self.parallel_files = self.uring_depth
        self._buffers_by_thread: dict[int, _AlignedIoBuffer] = {}
        self._rings_by_thread: dict[int, LiburingRing] = {}
        self._thread_user_data: dict[int, int] = {}
        self._buffers_lock = threading.Lock()
        logger.info(
            "LiburingFileStore initialized: root_dir=%s payload_size=%d io_size=%d io_alignment=%d direct=%s sync_on_store=%s uring_depth=%d",
            self.config.root_dir,
            self.payload_size,
            self.io_size,
            self.config.io_alignment,
            self.config.direct,
            self.config.sync_on_store,
            self.uring_depth,
        )

    def allocate_buffers(self, count: int) -> list[_AlignedIoBuffer]:
        with profile_scope(
            "storage_kvcache.liburing.allocate_buffers",
            "fs",
            args={"count": count, "io_size": self.io_size},
        ):
            return [self._allocate_buffer() for _ in range(count)]

    def free_buffers(self, buffers: list[_AlignedIoBuffer]) -> None:
        for buffer in buffers:
            _LIBC.free(buffer.ptr)

    def close(self) -> None:
        with self._buffers_lock:
            buffers = list(self._buffers_by_thread.values())
            rings = list(self._rings_by_thread.values())
            self._buffers_by_thread.clear()
            self._rings_by_thread.clear()
            self._thread_user_data.clear()
        for buffer in buffers:
            _LIBC.free(buffer.ptr)
        for ring in rings:
            ring.close()

    def fsync_dirs(self, paths: list[str]) -> None:
        if not self.config.sync_on_store:
            return
        for path in set(paths):
            with profile_scope("storage_kvcache.liburing.fsync_dir", "fs"):
                self._fsync_dir(path)

    def _allocate_buffer(self) -> _AlignedIoBuffer:
        alignment = max(self.config.io_alignment, ctypes.sizeof(ctypes.c_void_p))
        ptr = ctypes.c_void_p()
        with profile_scope(
            "storage_kvcache.liburing.posix_memalign",
            "fs",
            args={"alignment": alignment, "io_size": self.io_size},
        ):
            alloc_err = _LIBC.posix_memalign(ctypes.byref(ptr), alignment, self.io_size)
        if alloc_err != 0 or not ptr.value:
            raise RuntimeError(f"posix_memalign() failed: {alloc_err}")
        array = np.ctypeslib.as_array(
            ctypes.cast(ptr, ctypes.POINTER(ctypes.c_uint8)),
            shape=(self.io_size,),
        )
        return _AlignedIoBuffer(ptr=ptr, array=array)

    def _temp_path(self, final_path: str) -> str:
        unique_suffix = f"{os.getpid()}-{uuid.uuid4().hex}"
        return f"{final_path}.tmp-{unique_suffix}"

    def open_fd(self, path: str, *, write: bool, create: bool, truncate: bool) -> int:
        flags = os.O_WRONLY if write else os.O_RDONLY
        if create:
            flags |= os.O_CREAT
        if truncate:
            flags |= os.O_TRUNC
        if self.config.direct:
            flags |= getattr(os, "O_DIRECT", 0)
        mode = self.config.create_mode if self.config.create_mode is not None else 0o666
        open_start_ns = time.perf_counter_ns()
        fd = os.open(path, flags, mode)
        if profiler._active:
            profiler.add_event(
                "storage_kvcache.liburing.open_fd",
                "fs",
                open_start_ns,
                time.perf_counter_ns() - open_start_ns,
                args={"write": write, "create": create, "truncate": truncate},
            )
        return fd

    def read_path_into(
        self,
        path: str,
        slot_index: int,
        consumer: Callable[[int, np.ndarray[Any, np.dtype[np.uint8]]], None],
    ) -> _TimedIoResult:
        start_ns = time.perf_counter_ns()
        with profile_scope("storage_kvcache.liburing.get_thread_buffer", "fs"):
            buffer = self._get_thread_buffer()
        with profile_scope(
            "storage_kvcache.liburing.read_file_total",
            "fs",
            args={"io_size": self.io_size, "payload_size": self.payload_size},
        ):
            self._read_file(path, buffer)
        duration_ns = time.perf_counter_ns() - start_ns
        with profile_scope(
            "storage_kvcache.liburing.read_consumer_unpack",
            "fs",
            args={"payload_size": self.payload_size},
        ):
            consumer(slot_index, buffer.array[: self.payload_size])
        return _TimedIoResult(start_ns=start_ns, duration_ns=duration_ns)

    def store_slot_if_missing(
        self,
        path: str,
        slot_index: int,
        producer: Callable[[int, np.ndarray[Any, np.dtype[np.uint8]]], None],
    ) -> _TimedStoreResult:
        parent_path: str | None = None
        start_ns = time.perf_counter_ns()
        exists_start_ns = time.perf_counter_ns()
        exists = os.path.exists(path)
        if profiler._active:
            profiler.add_event(
                "storage_kvcache.liburing.store_exists",
                "fs",
                exists_start_ns,
                time.perf_counter_ns() - exists_start_ns,
                args={"exists": exists},
            )
        if exists:
            return _TimedStoreResult(
                stored=False,
                start_ns=start_ns,
                duration_ns=time.perf_counter_ns() - start_ns,
            )

        parent = os.path.dirname(path)
        with profile_scope("storage_kvcache.liburing.makedirs", "fs"):
            os.makedirs(parent, exist_ok=True)
        temp_path = self._temp_path(path)
        with profile_scope("storage_kvcache.liburing.get_thread_buffer", "fs"):
            buffer = self._get_thread_buffer()
        with profile_scope(
            "storage_kvcache.liburing.zero_buffer",
            "fs",
            args={"io_size": self.io_size},
        ):
            buffer.array.fill(0)
        with profile_scope(
            "storage_kvcache.liburing.store_producer_pack",
            "fs",
            args={"payload_size": self.payload_size},
        ):
            producer(slot_index, buffer.array[: self.payload_size])

        try:
            fd = self.open_fd(temp_path, write=True, create=True, truncate=True)
            try:
                with profile_scope(
                    "storage_kvcache.liburing.file_write",
                    "fs",
                    args={"io_size": self.io_size},
                ):
                    self._run_file_io(fd, buffer, path=temp_path, write=True)
                if self.config.sync_on_store:
                    with profile_scope("storage_kvcache.liburing.file_sync", "fs"):
                        os.fsync(fd)
            finally:
                with profile_scope("storage_kvcache.liburing.close_fd", "fs"):
                    os.close(fd)

            try:
                with profile_scope("storage_kvcache.liburing.link_temp", "fs"):
                    os.link(temp_path, path)
                if self.config.sync_on_store:
                    parent_path = parent
                return _TimedStoreResult(
                    stored=True,
                    start_ns=start_ns,
                    duration_ns=time.perf_counter_ns() - start_ns,
                    parent_path=parent_path,
                )
            except FileExistsError:
                return _TimedStoreResult(
                    stored=False,
                    start_ns=start_ns,
                    duration_ns=time.perf_counter_ns() - start_ns,
                )
        finally:
            try:
                with profile_scope("storage_kvcache.liburing.unlink_temp", "fs"):
                    os.unlink(temp_path)
            except FileNotFoundError:
                pass

    def _read_file(self, path: str, buffer: _AlignedIoBuffer) -> None:
        fd = self.open_fd(path, write=False, create=False, truncate=False)
        try:
            with profile_scope(
                "storage_kvcache.liburing.file_read",
                "fs",
                args={"io_size": self.io_size},
            ):
                self._run_file_io(fd, buffer, path=path, write=False)
        finally:
            with profile_scope("storage_kvcache.liburing.close_fd", "fs"):
                os.close(fd)

    def _run_file_io(
        self, fd: int, buffer: _AlignedIoBuffer, *, path: str, write: bool
    ) -> None:
        payload = memoryview(buffer.array)
        if write:
            io_start_ns = time.perf_counter_ns()
            result_nbytes = os.pwritev(fd, [payload], 0)
        else:
            io_start_ns = time.perf_counter_ns()
            result_nbytes = os.preadv(fd, [payload], 0)
        if profiler._active:
            profiler.add_event(
                "storage_kvcache.liburing.pwritev" if write else "storage_kvcache.liburing.preadv",
                "fs",
                io_start_ns,
                time.perf_counter_ns() - io_start_ns,
                args={"io_size": self.io_size},
            )
        if result_nbytes < 0:
            raise OSError(
                -result_nbytes, f"preadv/pwritev {'write' if write else 'read'} failed for {path}"
            )
        if result_nbytes != self.io_size:
            op_name = "write" if write else "read"
            raise IOError(
                f"preadv/pwritev {op_name} transferred {result_nbytes} bytes for {path}, expected {self.io_size}"
            )

    def _get_thread_buffer(self) -> _AlignedIoBuffer:
        thread_id = threading.get_ident()
        with self._buffers_lock:
            buffer = self._buffers_by_thread.get(thread_id)
            if buffer is not None:
                return buffer

        buffer = self._allocate_buffer()
        with self._buffers_lock:
            existing = self._buffers_by_thread.setdefault(thread_id, buffer)
        if existing is not buffer:
            _LIBC.free(buffer.ptr)
            return existing
        return buffer

    def _get_thread_ring(self) -> LiburingRing:
        thread_id = threading.get_ident()
        with self._buffers_lock:
            ring = self._rings_by_thread.get(thread_id)
            if ring is not None:
                return ring

        ring = LiburingRing(1)
        with self._buffers_lock:
            existing = self._rings_by_thread.setdefault(thread_id, ring)
        if existing is not ring:
            ring.close()
            return existing
        return ring

    def _next_thread_user_data(self) -> int:
        thread_id = threading.get_ident()
        with self._buffers_lock:
            user_data = self._thread_user_data.get(thread_id, 0) + 1
            self._thread_user_data[thread_id] = user_data
        return user_data

    def _fsync_dir(self, path: str) -> None:
        try:
            fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        except OSError:
            return
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


__all__ = [
    "LiburingFileStore",
    "LiburingRing",
    "_AlignedIoBuffer",
    "_TimedIoResult",
    "_TimedStoreResult",
]
