from __future__ import annotations

import os
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any

import torch
from vllm.logger import init_logger

from .fs_config import SharedFileConfig

logger = init_logger("vllm.storage_kvcache.kvikio_file")

_DEFAULT_PARALLEL_FILES = 16


@dataclass(frozen=True)
class _TimedKvikioResult:
    start_ns: int
    duration_ns: int
    parent_path: str | None = None
    stored: bool = True


@dataclass
class _PendingKvikioIo:
    future: Any
    buffer: "_TorchCudaArrayInterface"
    expected_nbytes: int


class _TorchCudaArrayInterface:
    def __init__(self, tensor: torch.Tensor):
        if not tensor.is_cuda:
            raise ValueError("KvikIO direct I/O requires CUDA tensors")
        if not tensor.is_contiguous():
            raise ValueError("KvikIO direct I/O requires contiguous tensors")
        self._tensor = tensor
        self._nbytes = tensor.numel() * tensor.element_size()

    @property
    def __cuda_array_interface__(self) -> dict[str, Any]:
        return {
            "data": (self._tensor.data_ptr(), False),
            "shape": (self._nbytes,),
            "strides": None,
            "typestr": "|u1",
            "version": 3,
        }


class KvikioFileStore:
    def __init__(self, config: SharedFileConfig, *, payload_size: int):
        if payload_size <= 0:
            raise ValueError("payload_size must be positive")

        try:
            import kvikio  # type: ignore
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "KvikIO backend requires the kvikio Python package"
            ) from exc

        self._kvikio = kvikio
        self.config = config
        self.payload_size = payload_size
        self._streams_by_thread: dict[tuple[int, int], torch.cuda.Stream] = {}
        self._streams_lock = threading.Lock()
        configured_parallel_files = config.parallel_files or _DEFAULT_PARALLEL_FILES
        self.parallel_files = max(
            1,
            min(configured_parallel_files, os.cpu_count() or configured_parallel_files),
        )
        logger.info(
            "KvikioFileStore initialized: root_dir=%s payload_size=%d sync_on_store=%s parallel_files=%d",
            self.config.root_dir,
            self.payload_size,
            self.config.sync_on_store,
            self.parallel_files,
        )

    def close(self) -> None:
        return

    def fsync_dirs(self, paths: list[str]) -> None:
        if not self.config.sync_on_store:
            return
        for path in set(paths):
            self._fsync_dir(path)

    def read_tensor_slice(
        self,
        handle: Any,
        tensor: torch.Tensor,
        *,
        size: int,
        file_offset: int,
        tensor_offset: int,
    ) -> int:
        torch.cuda.set_device(tensor.device)
        return int(
            handle.raw_read(
                _TorchCudaArrayInterface(tensor),
                size=size,
                file_offset=file_offset,
                dev_offset=tensor_offset,
            )
        )

    def submit_read_tensor_slice(
        self,
        handle: Any,
        tensor: torch.Tensor,
        *,
        size: int,
        file_offset: int,
        tensor_offset: int,
    ) -> _PendingKvikioIo:
        stream = self._get_thread_stream(tensor)
        buffer = _TorchCudaArrayInterface(tensor)
        future = handle.raw_read_async(
            buffer,
            int(stream.cuda_stream),
            size=size,
            file_offset=file_offset,
            dev_offset=tensor_offset,
        )
        return _PendingKvikioIo(future=future, buffer=buffer, expected_nbytes=size)

    def write_tensor_slice(
        self,
        handle: Any,
        tensor: torch.Tensor,
        *,
        size: int,
        file_offset: int,
        tensor_offset: int,
    ) -> int:
        torch.cuda.set_device(tensor.device)
        return int(
            handle.raw_write(
                _TorchCudaArrayInterface(tensor),
                size=size,
                file_offset=file_offset,
                dev_offset=tensor_offset,
            )
        )

    def submit_write_tensor_slice(
        self,
        handle: Any,
        tensor: torch.Tensor,
        *,
        size: int,
        file_offset: int,
        tensor_offset: int,
    ) -> _PendingKvikioIo:
        stream = self._get_thread_stream(tensor)
        buffer = _TorchCudaArrayInterface(tensor)
        future = handle.raw_write_async(
            buffer,
            int(stream.cuda_stream),
            size=size,
            file_offset=file_offset,
            dev_offset=tensor_offset,
        )
        return _PendingKvikioIo(future=future, buffer=buffer, expected_nbytes=size)

    def wait_kvikio_io(self, pending: _PendingKvikioIo) -> int:
        return int(pending.future.check_bytes_done())

    def open_read(self, path: str):
        return self._kvikio.CuFile(path, "r")

    def open_write_temp(self, final_path: str):
        parent = os.path.dirname(final_path)
        os.makedirs(parent, exist_ok=True)
        temp_path = self._temp_path(final_path)
        handle = self._kvikio.CuFile(temp_path, "w")
        os.ftruncate(handle.fileno(), self.payload_size)
        return temp_path, handle

    def publish_temp(self, temp_path: str, final_path: str) -> str | None:
        parent_path: str | None = None
        try:
            os.link(temp_path, final_path)
            if self.config.sync_on_store:
                parent_path = os.path.dirname(final_path)
            return parent_path
        except FileExistsError:
            return None
        finally:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass

    def sync_file(self, path: str) -> None:
        if not self.config.sync_on_store:
            return
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def _temp_path(self, final_path: str) -> str:
        unique_suffix = f"{os.getpid()}-{uuid.uuid4().hex}"
        return f"{final_path}.tmp-{unique_suffix}"

    def _get_thread_stream(self, tensor: torch.Tensor) -> torch.cuda.Stream:
        torch.cuda.set_device(tensor.device)
        key = (threading.get_ident(), tensor.device.index or 0)
        with self._streams_lock:
            stream = self._streams_by_thread.get(key)
            if stream is None:
                stream = torch.cuda.Stream(device=tensor.device)
                self._streams_by_thread[key] = stream
            return stream

    def _fsync_dir(self, path: str) -> None:
        try:
            fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        except OSError:
            return
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


__all__ = ["KvikioFileStore", "_PendingKvikioIo", "_TimedKvikioResult"]
