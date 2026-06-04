from __future__ import annotations

import ctypes
import errno
import os
import platform
import uuid

import numpy as np

from .fs_config import SharedFileConfig
from .profiling import add_event, now_ns

DIRECT_IO_ALIGNMENT = 4096
DEFAULT_IODEPTH = 256
DEFAULT_CREATE_MODE = 0o666

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
_IORING_OP_OPENAT = 18
_IORING_OP_CLOSE = 19
_IOSQE_ASYNC = 1 << 4
_IORING_ENTER_GETEVENTS = 1 << 0

# dirfd sentinel for *at syscalls; ignored since the file mapper emits absolute paths.
_AT_FDCWD = -100


def align_up(value: int, alignment: int = DIRECT_IO_ALIGNMENT) -> int:
    if alignment <= 0:
        raise ValueError("alignment must be positive")
    return ((value + alignment - 1) // alignment) * alignment


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


class LiburingRing:
    def __init__(self, depth: int, *, ring_id: int = -1):
        if depth <= 0:
            raise ValueError("iodepth must be positive")
        self.depth = depth
        self.ring_id = ring_id
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
        self._pending_submit = 0

        self._sq_ring_size = self._params.sq_off.array + self._params.sq_entries * ctypes.sizeof(
            ctypes.c_uint32
        )
        self._cq_ring_size = self._params.cq_off.cqes + self._params.cq_entries * ctypes.sizeof(
            _IoUringCqe
        )
        self._sqes_size = self._params.sq_entries * ctypes.sizeof(_IoUringSqe)

        self._sq_ring = self._mmap(self._sq_ring_size, _IORING_OFF_SQ_RING)
        self._cq_ring = self._mmap(self._cq_ring_size, _IORING_OFF_CQ_RING)
        self._sqes_map = self._mmap(self._sqes_size, _IORING_OFF_SQES)

        self._sq_head = self._u32_ptr(self._sq_ring, self._params.sq_off.head)
        self._sq_tail = self._u32_ptr(self._sq_ring, self._params.sq_off.tail)
        self._sq_mask = self._u32_ptr(self._sq_ring, self._params.sq_off.ring_mask)
        self._sq_array = self._u32_array(
            self._sq_ring, self._params.sq_off.array, self._params.sq_entries
        )
        self._sqes = ctypes.cast(self._sqes_map, ctypes.POINTER(_IoUringSqe))

        self._cq_head = self._u32_ptr(self._cq_ring, self._params.cq_off.head)
        self._cq_tail = self._u32_ptr(self._cq_ring, self._params.cq_off.tail)
        self._cq_mask = self._u32_ptr(self._cq_ring, self._params.cq_off.ring_mask)
        self._cqes = ctypes.cast(
            ctypes.c_void_p(self._cq_ring.value + self._params.cq_off.cqes),
            ctypes.POINTER(_IoUringCqe),
        )

    @property
    def pending_submit(self) -> int:
        return self._pending_submit

    def _begin_sqe(self) -> tuple[int, int, "_IoUringSqe"]:
        head = int(self._sq_head[0])
        tail = int(self._sq_tail[0])
        if tail - head >= self._params.sq_entries:
            raise BlockingIOError(errno.EAGAIN, "io_uring submission queue is full")
        index = tail & int(self._sq_mask[0])
        sqe = self._sqes[index]
        ctypes.memset(ctypes.byref(sqe), 0, ctypes.sizeof(_IoUringSqe))
        return index, tail, sqe

    def _commit_sqe(self, index: int, tail: int) -> None:
        self._sq_array[index] = index
        self._sq_tail[0] = tail + 1
        self._pending_submit += 1

    def queue_rw(
        self,
        *,
        user_data: int,
        fd: int,
        ptr: ctypes.c_void_p,
        nbytes: int,
        write: bool,
    ) -> None:
        index, tail, sqe = self._begin_sqe()
        sqe.opcode = _IORING_OP_WRITE if write else _IORING_OP_READ
        sqe.flags = _IOSQE_ASYNC
        sqe.fd = fd
        sqe.off = 0
        sqe.addr = int(ptr.value or 0)
        sqe.len = nbytes
        sqe.user_data = user_data
        self._commit_sqe(index, tail)

    def queue_openat(
        self,
        *,
        user_data: int,
        path_ptr: int,
        open_flags: int,
        mode: int = 0,
        dirfd: int = _AT_FDCWD,
    ) -> None:
        # path_ptr must remain alive until the open CQE is reaped.
        index, tail, sqe = self._begin_sqe()
        sqe.opcode = _IORING_OP_OPENAT
        sqe.flags = _IOSQE_ASYNC
        sqe.fd = dirfd
        sqe.off = 0
        sqe.addr = int(path_ptr)
        sqe.len = mode
        # rw_flags aliases open_flags in the kernel sqe union.
        sqe.rw_flags = open_flags
        sqe.user_data = user_data
        self._commit_sqe(index, tail)

    def queue_close(self, *, user_data: int, fd: int) -> None:
        index, tail, sqe = self._begin_sqe()
        sqe.opcode = _IORING_OP_CLOSE
        sqe.fd = fd
        sqe.user_data = user_data
        self._commit_sqe(index, tail)

    def submit_pending(self) -> None:
        if self._pending_submit == 0:
            return
        self._enter(self._pending_submit, 0)

    def _enter(self, to_submit: int, min_complete: int) -> None:
        flags = _IORING_ENTER_GETEVENTS if min_complete > 0 else 0
        submitted = _LIBC.syscall(
            ctypes.c_long(_NR_IO_URING_ENTER),
            ctypes.c_int(self.fd),
            ctypes.c_uint32(to_submit),
            ctypes.c_uint32(min_complete),
            ctypes.c_uint32(flags),
            ctypes.c_void_p(0),
            ctypes.c_size_t(0),
        )
        if submitted < 0:
            err = ctypes.get_errno()
            raise OSError(err, "io_uring_enter failed")
        self._pending_submit = max(0, self._pending_submit - int(submitted))

    def poll_all(self) -> list[tuple[int, int]]:
        head = int(self._cq_head[0])
        tail = int(self._cq_tail[0])
        if head == tail:
            return []
        mask = int(self._cq_mask[0])
        completions: list[tuple[int, int]] = []
        while head != tail:
            index = head & mask
            cqe = self._cqes[index]
            completions.append((int(cqe.user_data), int(cqe.res)))
            head += 1
        self._cq_head[0] = head
        return completions

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
        return ctypes.cast(ctypes.c_void_p(base.value + offset), ctypes.POINTER(ctypes.c_uint32))

    @staticmethod
    def _u32_array(base: ctypes.c_void_p, offset: int, count: int):
        return (ctypes.c_uint32 * count).from_address(base.value + offset)


class DirectIoFileStore:
    def __init__(self, config: SharedFileConfig, *, payload_size: int):
        if payload_size <= 0:
            raise ValueError("payload_size must be positive")
        self.config = config
        self.payload_size = payload_size
        self.io_size = align_up(payload_size)
        self.iodepth = max(1, config.iodepth or DEFAULT_IODEPTH)
        if not hasattr(os, "O_DIRECT"):
            raise RuntimeError("direct I/O requires os.O_DIRECT support")

    def queue_open_read(self, ring: LiburingRing, *, user_data: int, path: str) -> ctypes.Array:
        # Caller must keep the returned buffer alive until the open CQE is reaped.
        path_buf = ctypes.create_string_buffer(os.fsencode(path) + b"\0")
        ring.queue_openat(
            user_data=user_data,
            path_ptr=ctypes.addressof(path_buf),
            open_flags=os.O_RDONLY | os.O_DIRECT,
            mode=0,
        )
        return path_buf

    def queue_close(self, ring: LiburingRing, *, user_data: int, fd: int) -> None:
        ring.queue_close(user_data=user_data, fd=fd)

    def open_temp_write(self, final_path: str) -> tuple[int, str]:
        parent = os.path.dirname(final_path)
        os.makedirs(parent, exist_ok=True)
        temp_path = self._temp_path(final_path)
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_DIRECT
        fd = os.open(temp_path, flags, DEFAULT_CREATE_MODE)
        return fd, temp_path

    def queue_read(
        self,
        ring: LiburingRing,
        *,
        user_data: int,
        fd: int,
        io_array: np.ndarray,
    ) -> None:
        self._validate_io_array(io_array)
        ring.queue_rw(
            user_data=user_data,
            fd=fd,
            ptr=ctypes.c_void_p(io_array.ctypes.data),
            nbytes=self.io_size,
            write=False,
        )

    def queue_write(
        self,
        ring: LiburingRing,
        *,
        user_data: int,
        fd: int,
        io_array: np.ndarray,
    ) -> None:
        self._validate_io_array(io_array)
        ring.queue_rw(
            user_data=user_data,
            fd=fd,
            ptr=ctypes.c_void_p(io_array.ctypes.data),
            nbytes=self.io_size,
            write=True,
        )

    def finish_write(self, *, fd: int, temp_path: str, final_path: str) -> bool:
        if self.config.sync_on_store:
            sync_start_ns = now_ns()
            os.fsync(fd)
            add_event(
                "py_kvcache.file.file_sync",
                "fs",
                sync_start_ns,
                now_ns() - sync_start_ns,
            )
        os.close(fd)
        stored = False
        link_error: BaseException | None = None
        try:
            os.link(temp_path, final_path)
            stored = True
        except FileExistsError:
            stored = False
        except BaseException as exc:
            link_error = exc
        try:
            os.unlink(temp_path)
        except FileNotFoundError:
            pass
        if link_error is not None:
            raise link_error
        if self.config.sync_on_store:
            self._fsync_dir(os.path.dirname(final_path))
        return stored

    def cleanup_temp(self, *, fd: int | None, temp_path: str | None) -> None:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        if temp_path is not None:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass

    def _validate_io_array(self, array: np.ndarray) -> None:
        if array.nbytes < self.io_size:
            raise ValueError("I/O array is smaller than aligned I/O size")
        if array.ctypes.data % DIRECT_IO_ALIGNMENT != 0:
            raise ValueError("I/O array is not aligned for direct I/O")
        if not array.flags.c_contiguous:
            raise ValueError("I/O array must be contiguous")

    def _temp_path(self, final_path: str) -> str:
        unique_suffix = f"{os.getpid()}-{uuid.uuid4().hex}"
        return f"{final_path}.tmp-{unique_suffix}"

    @staticmethod
    def _fsync_dir(path: str) -> None:
        try:
            fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        except OSError:
            return
        try:
            sync_start_ns = now_ns()
            os.fsync(fd)
            add_event(
                "py_kvcache.file.fsync_dir",
                "fs",
                sync_start_ns,
                now_ns() - sync_start_ns,
            )
        finally:
            os.close(fd)


__all__ = [
    "DEFAULT_IODEPTH",
    "DIRECT_IO_ALIGNMENT",
    "DirectIoFileStore",
    "LiburingRing",
    "align_up",
]
