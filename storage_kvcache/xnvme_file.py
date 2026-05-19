from __future__ import annotations

import ctypes
import os
import sys
import threading
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from simple_profiler import profile_scope, profiler
from vllm.logger import init_logger
import xnvme.ctypes_bindings as xnvme_bindings
import xnvme.ctypes_bindings.api as xnvme

from .fs_config import SharedFileConfig

logger = init_logger("vllm.storage_kvcache.xnvme_file")

_ASYNC_BACKEND_NAMES = {
    "emu",
    "io_uring",
    "io_uring_cmd",
    "libaio",
    "nil",
    "posix",
    "thrpool",
}

_DEFAULT_PARALLEL_FILES = 16

_LIBC = ctypes.CDLL(None)
_LIBC.posix_memalign.argtypes = [
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.c_size_t,
    ctypes.c_size_t,
]
_LIBC.posix_memalign.restype = ctypes.c_int
_LIBC.free.argtypes = [ctypes.c_void_p]
_LIBC.free.restype = None


@dataclass(frozen=True)
class _TimedStoreResult:
    stored: bool
    start_ns: int
    duration_ns: int
    parent_path: str | None = None


@dataclass(frozen=True)
class _TimedIoResult:
    start_ns: int
    duration_ns: int


@dataclass
class _AlignedIoBuffer:
    ptr: ctypes.c_void_p
    array: np.ndarray[Any, np.dtype[np.uint8]]


def _xnvme_cmd_ctx_failed(ctx: Any) -> bool:
    return bool(ctx.cpl.status.sc or ctx.cpl.status.sct)


def _xnvme_cmd_ctx_result(ctx: Any) -> int:
    return int(ctx.cpl.result)


def _align_up(value: int, alignment: int) -> int:
    if alignment <= 0:
        raise ValueError("alignment must be positive")
    return ((value + alignment - 1) // alignment) * alignment


class XNvmeFileStore:
    def __init__(self, config: SharedFileConfig, *, payload_size: int):
        if not getattr(xnvme_bindings, "is_loaded", True):
            raise RuntimeError(
                "xNVMe Python bindings are importable, but libxnvme could not be loaded"
            )
        if payload_size <= 0:
            raise ValueError("payload_size must be positive")

        self.config = config
        self.payload_size = payload_size
        self.io_size = _align_up(payload_size, config.io_alignment)
        configured_parallel_files = config.parallel_files or _DEFAULT_PARALLEL_FILES
        self.parallel_files = max(
            1,
            min(configured_parallel_files, os.cpu_count() or configured_parallel_files),
        )
        self._buffers_by_thread: dict[int, _AlignedIoBuffer] = {}
        self._buffers_lock = threading.Lock()
        self.be, self.mem, self.sync, self.async_backend = (
            self._resolve_file_backend_config()
        )
        logger.info(
            "XNvmeFileStore initialized: root_dir=%s payload_size=%d io_size=%d io_alignment=%d padding=%d direct=%s sync_on_store=%s be=%s sync=%s async=%s mem=%s parallel_files=%d",
            self.config.root_dir,
            self.payload_size,
            self.io_size,
            self.config.io_alignment,
            self.io_size - self.payload_size,
            self.config.direct,
            self.config.sync_on_store,
            self.be,
            self.sync,
            self.async_backend,
            self.mem,
            self.parallel_files,
        )

    def read_path_into(
        self,
        path: str,
        slot_index: int,
        consumer: Callable[[int, np.ndarray[Any, np.dtype[np.uint8]]], None],
    ) -> _TimedIoResult:
        start_ns = time.perf_counter_ns()
        with profile_scope("storage_kvcache.xnvme.get_thread_buffer", "fs"):
            buffer = self._get_thread_buffer()
        with profile_scope(
            "storage_kvcache.xnvme.read_file_total",
            "fs",
            args={"io_size": self.io_size, "payload_size": self.payload_size},
        ):
            self._read_file(path, buffer)
        duration_ns = time.perf_counter_ns() - start_ns
        with profile_scope(
            "storage_kvcache.xnvme.read_consumer_unpack",
            "fs",
            args={"payload_size": self.payload_size},
        ):
            consumer(slot_index, buffer.array[: self.payload_size])
        return _TimedIoResult(start_ns=start_ns, duration_ns=duration_ns)

    def read_path_into_buffer(
        self,
        path: str,
        buffer: _AlignedIoBuffer,
    ) -> _TimedIoResult:
        start_ns = time.perf_counter_ns()
        with profile_scope(
            "storage_kvcache.xnvme.read_file_total",
            "fs",
            args={"io_size": self.io_size, "payload_size": self.payload_size},
        ):
            self._read_file(path, buffer)
        return _TimedIoResult(
            start_ns=start_ns,
            duration_ns=time.perf_counter_ns() - start_ns,
        )

    def allocate_buffers(self, count: int) -> list[_AlignedIoBuffer]:
        with profile_scope(
            "storage_kvcache.xnvme.allocate_buffers",
            "fs",
            args={"count": count, "io_size": self.io_size},
        ):
            return [self._allocate_buffer() for _ in range(count)]

    def free_buffers(self, buffers: Sequence[_AlignedIoBuffer]) -> None:
        for buffer in buffers:
            _LIBC.free(buffer.ptr)

    def close(self) -> None:
        with self._buffers_lock:
            buffers = list(self._buffers_by_thread.values())
            self._buffers_by_thread.clear()
        for buffer in buffers:
            _LIBC.free(buffer.ptr)

    def fsync_dirs(self, paths: Sequence[str]) -> None:
        if not self.config.sync_on_store:
            return
        for path in set(paths):
            with profile_scope("storage_kvcache.xnvme.fsync_dir", "fs"):
                self._fsync_dir(path)

    def _read_file(self, path: str, buffer: _AlignedIoBuffer) -> None:
        with profile_scope("storage_kvcache.xnvme.open_read", "fs"):
            handle_context = self._open_file(
                path, create=False, truncate=False, write=False
            )
        with handle_context as handle:
            with profile_scope(
                "storage_kvcache.xnvme.queued_read",
                "fs",
                args={"io_size": self.io_size},
            ):
                result_nbytes = self._run_queued_file_io(
                    handle,
                    buffer,
                    path=path,
                    write=False,
                )
            if result_nbytes != self.io_size:
                raise IOError(
                    f"xnvme_file_pread() transferred {result_nbytes} bytes for {path}, expected {self.io_size}"
                )

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
                "storage_kvcache.xnvme.store_exists",
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
        with profile_scope("storage_kvcache.xnvme.makedirs", "fs"):
            os.makedirs(parent, exist_ok=True)
        temp_path = self._temp_path(path)
        with profile_scope("storage_kvcache.xnvme.get_thread_buffer", "fs"):
            buffer = self._get_thread_buffer()
        with profile_scope(
            "storage_kvcache.xnvme.zero_buffer",
            "fs",
            args={"io_size": self.io_size},
        ):
            buffer.array.fill(0)
        with profile_scope(
            "storage_kvcache.xnvme.store_producer_pack",
            "fs",
            args={"payload_size": self.payload_size},
        ):
            producer(slot_index, buffer.array[: self.payload_size])

        try:
            with profile_scope("storage_kvcache.xnvme.open_write", "fs"):
                handle_context = self._open_file(
                    temp_path, create=True, truncate=True, write=True
                )
            with handle_context as handle:
                with profile_scope(
                    "storage_kvcache.xnvme.queued_write",
                    "fs",
                    args={"io_size": self.io_size},
                ):
                    result_nbytes = self._run_queued_file_io(
                        handle,
                        buffer,
                        path=temp_path,
                        write=True,
                    )
                if result_nbytes != self.io_size:
                    raise IOError(
                        f"xnvme_file_pwrite() transferred {result_nbytes} bytes for {temp_path}, expected {self.io_size}"
                    )
                if self.config.sync_on_store:
                    sync_start_ns = time.perf_counter_ns()
                    sync_err = xnvme.xnvme_file_sync(handle)
                    if profiler._active:
                        profiler.add_event(
                            "storage_kvcache.xnvme.file_sync",
                            "fs",
                            sync_start_ns,
                            time.perf_counter_ns() - sync_start_ns,
                        )
                    if sync_err:
                        raise IOError(f"xnvme_file_sync() failed for {temp_path}")

            try:
                with profile_scope("storage_kvcache.xnvme.link_temp", "fs"):
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
                with profile_scope("storage_kvcache.xnvme.unlink_temp", "fs"):
                    os.unlink(temp_path)
            except FileNotFoundError:
                pass

    def _run_queued_file_io(
        self,
        handle: Any,
        buffer: _AlignedIoBuffer,
        *,
        path: str,
        write: bool,
    ) -> int:
        queue = ctypes.POINTER(xnvme.xnvme_queue)()
        with profile_scope("storage_kvcache.xnvme.queue_init", "fs"):
            queue_err = xnvme.xnvme_queue_init(handle, 1, 0, ctypes.byref(queue))
        if queue_err or not queue:
            op = "write" if write else "read"
            raise IOError(f"xnvme_queue_init() failed for {op} of {path}: {queue_err}")

        ctx = None
        try:
            with profile_scope("storage_kvcache.xnvme.queue_get_ctx", "fs"):
                ctx = xnvme.xnvme_queue_get_cmd_ctx(queue)
            if not ctx:
                raise IOError(f"xnvme_queue_get_cmd_ctx() failed for {path}")

            ptr = ctypes.cast(buffer.ptr, ctypes.POINTER(None))
            submit_start_ns = time.perf_counter_ns()
            if write:
                io_err = xnvme.xnvme_file_pwrite(ctx, ptr, self.io_size, 0)
                op_name = "xnvme_file_pwrite"
            else:
                io_err = xnvme.xnvme_file_pread(ctx, ptr, self.io_size, 0)
                op_name = "xnvme_file_pread"
            if profiler._active:
                profiler.add_event(
                    f"storage_kvcache.xnvme.{op_name}_submit",
                    "fs",
                    submit_start_ns,
                    time.perf_counter_ns() - submit_start_ns,
                    args={"io_size": self.io_size},
                )
            if io_err:
                raise IOError(f"{op_name}() submit failed for {path}: {io_err}")

            poke_start_ns = time.perf_counter_ns()
            poke_count = 0
            while xnvme.xnvme_queue_get_outstanding(queue):
                poke_count += 1
                poke_err = xnvme.xnvme_queue_poke(queue, 0)
                if poke_err < 0:
                    raise IOError(f"xnvme_queue_poke() failed for {path}: {poke_err}")
            if profiler._active:
                profiler.add_event(
                    "storage_kvcache.xnvme.queue_poke_until_done",
                    "fs",
                    poke_start_ns,
                    time.perf_counter_ns() - poke_start_ns,
                    args={"poke_count": poke_count, "io_size": self.io_size},
                )

            result = ctx.contents
            if _xnvme_cmd_ctx_failed(result):
                raise IOError(f"{op_name}() failed for {path}")
            return _xnvme_cmd_ctx_result(result)
        finally:
            if ctx:
                with profile_scope("storage_kvcache.xnvme.queue_put_ctx", "fs"):
                    xnvme.xnvme_queue_put_cmd_ctx(queue, ctx)
            with profile_scope("storage_kvcache.xnvme.queue_term", "fs"):
                xnvme.xnvme_queue_term(queue)

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

    def _allocate_buffer(self) -> _AlignedIoBuffer:
        alignment = max(self.config.io_alignment, ctypes.sizeof(ctypes.c_void_p))
        ptr = ctypes.c_void_p()
        with profile_scope(
            "storage_kvcache.xnvme.posix_memalign",
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

    def _make_opts(self, *, create: bool, truncate: bool, write: bool):
        string_refs: list[ctypes.Array[ctypes.c_char]] = []
        opts = xnvme.xnvme_opts()
        xnvme.xnvme_opts_set_defaults(ctypes.byref(opts))

        self._set_opt_str(opts, string_refs, "be", self.be)
        self._set_opt_str(opts, string_refs, "mem", self.mem)
        self._set_opt_str(opts, string_refs, "sync", self.sync)
        self._set_opt_str(opts, string_refs, "async", self.async_backend)

        opts.rdonly = 0 if write else 1
        opts.wronly = 1 if write else 0
        opts.rdwr = 0
        opts.direct = 1 if self.config.direct else 0
        opts.create = 1 if create else 0
        opts.truncate = 1 if truncate else 0
        if self.config.create_mode is not None:
            opts.create_mode = self.config.create_mode
        return opts, string_refs, self.be, self.mem, self.sync, self.async_backend

    def _resolve_file_backend_config(
        self,
    ) -> tuple[str | None, str | None, str | None, str | None]:
        be = self.config.be
        mem = self.config.mem
        sync = self.config.sync
        async_backend = self.config.async_backend

        if be in _ASYNC_BACKEND_NAMES:
            if async_backend is None:
                async_backend = be
            logger.warning(
                "xnvme_be=%s is an async backend name, not a device backend; using be=linux and xnvme_async=%s",
                be,
                async_backend,
            )
            be = None

        if sys.platform.startswith("linux"):
            if be is None:
                be = "linux"
            elif be != "linux":
                logger.warning(
                    "xnvme_be=%s is not valid for shared filesystem paths on Linux; using be=linux",
                    be,
                )
                be = "linux"

            if sync is None:
                sync = "psync"
            elif sync != "psync":
                logger.warning(
                    "xnvme_sync=%s targets device I/O, not regular files; using sync=psync",
                    sync,
                )
                sync = "psync"

            if async_backend is None:
                async_backend = "io_uring"

        if self.config.admin is not None:
            logger.warning(
                "xnvme_admin=%s is ignored for shared filesystem paths",
                self.config.admin,
            )

        return be, mem, sync, async_backend

    def _open_file(
        self,
        path: str,
        *,
        create: bool,
        truncate: bool,
        write: bool,
    ):
        opts, string_refs, be, mem, sync, async_backend = self._make_opts(
            create=create, truncate=truncate, write=write
        )
        uri_ref = ctypes.create_string_buffer(path.encode("utf-8"))
        string_refs.append(uri_ref)
        open_start_ns = time.perf_counter_ns()
        handle = xnvme.xnvme_file_open(
            ctypes.cast(uri_ref, ctypes.POINTER(ctypes.c_char)), ctypes.byref(opts)
        )
        if profiler._active:
            profiler.add_event(
                "storage_kvcache.xnvme.file_open",
                "fs",
                open_start_ns,
                time.perf_counter_ns() - open_start_ns,
                args={"write": write, "create": create, "truncate": truncate},
            )
        if not handle:
            raise RuntimeError(
                "xnvme_file_open() failed for "
                f"{path} with be={be!r}, mem={mem!r}, sync={sync!r}, async={async_backend!r}, "
                f"direct={bool(opts.direct)}, create={bool(opts.create)}, "
                f"truncate={bool(opts.truncate)}, rdonly={bool(opts.rdonly)}, "
                f"wronly={bool(opts.wronly)}"
            )

        class _HandleContext:
            def __enter__(self_nonlocal):
                return handle

            def __exit__(self_nonlocal, exc_type, exc, tb):
                close_start_ns = time.perf_counter_ns()
                xnvme.xnvme_file_close(handle)
                if profiler._active:
                    profiler.add_event(
                        "storage_kvcache.xnvme.file_close",
                        "fs",
                        close_start_ns,
                        time.perf_counter_ns() - close_start_ns,
                    )
                return False

        return _HandleContext()

    def _set_opt_str(
        self,
        opts: Any,
        string_refs: list[ctypes.Array[ctypes.c_char]],
        field_name: str,
        value: str | None,
    ) -> None:
        if value is None:
            return
        ref = ctypes.create_string_buffer(value.encode("utf-8"))
        string_refs.append(ref)
        setattr(opts, field_name, ctypes.cast(ref, ctypes.POINTER(ctypes.c_char)))

    def _fsync_dir(self, path: str) -> None:
        try:
            fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        except OSError:
            return
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


__all__ = ["XNvmeFileStore"]
