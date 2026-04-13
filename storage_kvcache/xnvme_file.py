from __future__ import annotations

import ctypes
import os
import uuid
from contextlib import contextmanager
from typing import Any, Iterator

import numpy as np
import xnvme.ctypes_bindings as xnvme_bindings
import xnvme.ctypes_bindings.api as xnvme

from .fs_config import SharedFileConfig


def _xnvme_cmd_ctx_failed(ctx: Any) -> bool:
    return bool(ctx.cpl.status.sc or ctx.cpl.status.sct)


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

    def read_file(self, path: str) -> bytes:
        stat_result = os.stat(path)
        if stat_result.st_size < self.io_size:
            raise IOError(
                f"shared KV file {path} is too small: {stat_result.st_size} < {self.io_size}"
            )

        with self._open_file(path, create=False, truncate=False) as handle:
            ctx = xnvme.xnvme_file_get_cmd_ctx(handle)
            buf = xnvme.xnvme_buf_alloc(handle, self.io_size)
            if not buf:
                raise RuntimeError("xnvme_buf_alloc() failed")
            try:
                err = xnvme.xnvme_file_pread(
                    ctypes.byref(ctx),
                    buf,
                    self.io_size,
                    0,
                )
                if err or _xnvme_cmd_ctx_failed(ctx):
                    raise IOError(f"xnvme_file_pread() failed for {path}")

                array = np.ctypeslib.as_array(
                    ctypes.cast(buf, ctypes.POINTER(ctypes.c_uint8)),
                    shape=(self.io_size,),
                )
                return array[: self.payload_size].tobytes()
            finally:
                xnvme.xnvme_buf_free(handle, buf)

    def store_if_missing(self, path: str, payload: bytes) -> bool:
        if len(payload) != self.payload_size:
            raise ValueError(
                f"payload size {len(payload)} does not match expected {self.payload_size}"
            )
        if os.path.exists(path):
            return False

        parent = os.path.dirname(path)
        os.makedirs(parent, exist_ok=True)
        temp_path = self._temp_path(path)

        padded_payload = bytearray(self.io_size)
        padded_payload[: self.payload_size] = payload

        try:
            with self._open_file(temp_path, create=True, truncate=True) as handle:
                buf = xnvme.xnvme_buf_alloc(handle, self.io_size)
                if not buf:
                    raise RuntimeError("xnvme_buf_alloc() failed")
                try:
                    array = np.ctypeslib.as_array(
                        ctypes.cast(buf, ctypes.POINTER(ctypes.c_uint8)),
                        shape=(self.io_size,),
                    )
                    array[:] = np.frombuffer(
                        padded_payload, dtype=np.uint8, count=self.io_size
                    )

                    ctx = xnvme.xnvme_file_get_cmd_ctx(handle)
                    err = xnvme.xnvme_file_pwrite(
                        ctypes.byref(ctx),
                        buf,
                        self.io_size,
                        0,
                    )
                    if err or _xnvme_cmd_ctx_failed(ctx):
                        raise IOError(f"xnvme_file_pwrite() failed for {temp_path}")
                    sync_err = xnvme.xnvme_file_sync(handle)
                    if sync_err:
                        raise IOError(f"xnvme_file_sync() failed for {temp_path}")
                finally:
                    xnvme.xnvme_buf_free(handle, buf)

            try:
                os.link(temp_path, path)
                self._fsync_dir(parent)
                return True
            except FileExistsError:
                return False
        finally:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass

    def _temp_path(self, final_path: str) -> str:
        unique_suffix = f"{os.getpid()}-{uuid.uuid4().hex}"
        return f"{final_path}.tmp-{unique_suffix}"

    def _make_opts(self, *, create: bool, truncate: bool):
        string_refs: list[ctypes.Array[ctypes.c_char]] = []
        opts = xnvme.xnvme_opts()
        xnvme.xnvme_opts_set_defaults(ctypes.byref(opts))

        self._set_opt_str(opts, string_refs, "be", self.config.be)
        self._set_opt_str(opts, string_refs, "mem", self.config.mem)
        self._set_opt_str(opts, string_refs, "sync", self.config.sync)
        self._set_opt_str(opts, string_refs, "async", self.config.async_backend)
        self._set_opt_str(opts, string_refs, "admin", self.config.admin)

        opts.rdwr = 1
        opts.direct = 1 if self.config.direct else 0
        opts.create = 1 if create else 0
        opts.truncate = 1 if truncate else 0
        if self.config.create_mode is not None:
            opts.create_mode = self.config.create_mode
        return opts, string_refs

    @contextmanager
    def _open_file(
        self,
        path: str,
        *,
        create: bool,
        truncate: bool,
    ) -> Iterator[Any]:
        opts, string_refs = self._make_opts(create=create, truncate=truncate)
        uri_ref = ctypes.create_string_buffer(path.encode("utf-8"))
        string_refs.append(uri_ref)
        handle = xnvme.xnvme_file_open(
            ctypes.cast(uri_ref, ctypes.c_char_p), ctypes.byref(opts)
        )
        if not handle:
            raise RuntimeError(f"xnvme_file_open() failed for {path}")
        try:
            yield handle
        finally:
            xnvme.xnvme_file_close(handle)

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
        setattr(opts, field_name, ctypes.cast(ref, ctypes.c_char_p))

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
