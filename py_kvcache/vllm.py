from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from concurrent.futures import Future, wait as futures_wait
from dataclasses import dataclass
from typing import Any

from .file_mapper import FileMapper
from .fs_config import SharedFileConfig
from .profiling import add_event, now_ns
from .reactor import TransferCoordinator
from .transfer import ParsedKvLayout

logger = logging.getLogger(__name__)

try:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.kv_offload.abstract import (
        LoadStoreSpec,
        OffloadingManager,
        OffloadKey,
        PrepareStoreOutput,
        get_offload_block_hash,
        get_offload_group_idx,
    )
    from vllm.v1.kv_offload.mediums import GPULoadStoreSpec
    from vllm.v1.kv_offload.spec import CanonicalKVCaches, OffloadingSpec
    from vllm.v1.kv_offload.worker.worker import (
        OffloadingHandler,
        TransferResult,
        TransferSpec,
    )
    VLLM_AVAILABLE = True
except Exception:
    VLLM_AVAILABLE = False
    VllmConfig = Any
    KVCacheConfig = Any
    OffloadKey = Any
    CanonicalKVCaches = Any
    TransferSpec = tuple[Any, Any]

    class LoadStoreSpec:
        @staticmethod
        def medium() -> str:
            return "NOOP"

    class OffloadingManager:
        pass

    @dataclass(frozen=True)
    class PrepareStoreOutput:
        keys_to_store: list[Any]
        store_spec: LoadStoreSpec
        evicted_keys: list[Any]

    def get_offload_block_hash(key: Any) -> bytes:
        raise RuntimeError("vLLM is required for scheduler offload-key parsing")

    def get_offload_group_idx(key: Any) -> int:
        raise RuntimeError("vLLM is required for scheduler offload-key parsing")

    class GPULoadStoreSpec(LoadStoreSpec):
        @staticmethod
        def medium() -> str:
            return "GPU"

    class OffloadingSpec:
        def __init__(self, vllm_config: Any, kv_cache_config: Any) -> None:
            self.vllm_config = vllm_config
            self.kv_cache_config = kv_cache_config
            self.extra_config = _read_extra_config(vllm_config, kv_cache_config)
            self.gpu_block_size = [
                int(self.extra_config.get("gpu_block_size", 1))
            ]
            self.block_size_factor = int(
                self.extra_config.get("block_size_factor", 1)
            )

    class OffloadingHandler:
        pass

    @dataclass(frozen=True)
    class TransferResult:
        job_id: int
        success: bool = True


def _read_extra_config(
    vllm_config: Any, kv_cache_config: Any
) -> Mapping[str, object]:
    for owner in (kv_cache_config, vllm_config):
        for attr_name in ("kv_connector_extra_config", "extra_config"):
            value = getattr(owner, attr_name, None)
            if isinstance(value, Mapping):
                return value
    kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)
    value = getattr(kv_transfer_config, "kv_connector_extra_config", None)
    if isinstance(value, Mapping):
        return value
    return {}


def _get_nested_attr(root: Any, path: tuple[str, ...], default: Any) -> Any:
    value = root
    for attr_name in path:
        value = getattr(value, attr_name, None)
        if value is None:
            return default
    return value


def _first_int(value: Any, default: int) -> int:
    if isinstance(value, (list, tuple)):
        if not value:
            return default
        value = value[0]
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


class SharedStorageLoadStoreSpec(LoadStoreSpec):
    def __init__(self, block_hashes: Iterable[bytes] = ()):
        self.block_hashes = list(block_hashes)

    @staticmethod
    def medium() -> str:
        return "SHARED_STORAGE"


@dataclass(frozen=True)
class SharedStoragePreloadMessage:
    block_hashes: tuple[bytes, ...]


class SharedStorageOffloadingManager(OffloadingManager):
    """Scheduler-side manager for immutable shared-storage blocks."""

    def __init__(
        self,
        file_mapper: FileMapper,
        *,
        preload_batch_size: int = 1,
        enable_preload: bool = True,
    ):
        self.file_mapper = file_mapper
        self.preload_batch_size = max(1, preload_batch_size)
        self.enable_preload = enable_preload
        self._worker_message_sender: Callable[[Any], None] | None = None

    def set_worker_message_sender(self, sender: Callable[[Any], None] | None) -> None:
        self._worker_message_sender = sender

    @staticmethod
    def _get_block_hash(key: OffloadKey) -> bytes:
        group_idx = get_offload_group_idx(key)
        if group_idx != 0:
            raise ValueError(
                "py-kvcache shared-storage offload currently supports exactly one KV cache group"
            )
        return get_offload_block_hash(key)

    def lookup(self, keys: Iterable[OffloadKey]) -> int:
        start_ns = now_ns()
        hit_count = 0
        checked = 0
        preload_batch: list[bytes] = []

        def flush_preload_batch() -> None:
            if not preload_batch or not self.enable_preload:
                preload_batch.clear()
                return
            if self._worker_message_sender is not None:
                self._worker_message_sender(
                    SharedStoragePreloadMessage(tuple(preload_batch))
                )
            preload_batch.clear()

        for key in keys:
            checked += 1
            block_hash = self._get_block_hash(key)
            file_name = self.file_mapper.get_file_name(block_hash)
            exists = os.path.exists(file_name)
            if not exists:
                break
            hit_count += 1
            if self.enable_preload:
                preload_batch.append(block_hash)
                if len(preload_batch) >= self.preload_batch_size:
                    flush_preload_batch()

        flush_preload_batch()
        add_event(
            "py_kvcache.manager.lookup",
            "kv_offload",
            start_ns,
            now_ns() - start_ns,
            args={"checked_keys": checked, "hit_count": hit_count},
        )
        return hit_count

    def prepare_load(self, keys: Iterable[OffloadKey]) -> LoadStoreSpec:
        block_hashes = [self._get_block_hash(key) for key in keys]
        return SharedStorageLoadStoreSpec(block_hashes)

    def touch(self, keys: Iterable[OffloadKey]) -> None:
        return None

    def complete_load(self, keys: Iterable[OffloadKey]) -> None:
        return None

    def prepare_store(self, keys: Iterable[OffloadKey]) -> PrepareStoreOutput | None:
        start_ns = now_ns()
        keys_to_store: list[OffloadKey] = []
        block_hashes_to_store: list[bytes] = []
        checked = 0

        for key in keys:
            checked += 1
            block_hash = self._get_block_hash(key)
            file_name = self.file_mapper.get_file_name(block_hash)
            exists = os.path.exists(file_name)
            if exists:
                continue
            keys_to_store.append(key)
            block_hashes_to_store.append(block_hash)

        add_event(
            "py_kvcache.manager.prepare_store",
            "kv_offload",
            start_ns,
            now_ns() - start_ns,
            args={
                "checked_keys": checked,
                "keys_to_store": len(keys_to_store),
            },
        )
        return PrepareStoreOutput(
            keys_to_store=keys_to_store,
            store_spec=SharedStorageLoadStoreSpec(block_hashes_to_store),
            evicted_keys=[],
        )

    def complete_store(
        self, keys: Iterable[OffloadKey], success: bool = True
    ) -> None:
        return None


class NoopSharedStorageOffloadingHandler(OffloadingHandler):
    """Worker-side handler for py-kvcache load/store transfers."""

    def __init__(
        self,
        *,
        coordinator: TransferCoordinator,
    ) -> None:
        self.coordinator = coordinator
        self._active: dict[int, tuple[Future[int], float, tuple[str, str]]] = {}
        self._finished: dict[int, TransferResult] = {}
        self.is_shutdown = False

    def handle_worker_message(self, message: Any) -> bool:
        if isinstance(message, SharedStoragePreloadMessage):
            self.coordinator.submit_preload(list(message.block_hashes))
            return True
        return False

    def transfer_async(
        self,
        job_id: int,
        spec: TransferSpec,
        profile_tid: str = "kv_transfer",
        req_id: str = "",
    ) -> bool:
        if self.is_shutdown:
            raise RuntimeError("py-kvcache handler is shut down")
        if job_id in self._active or job_id in self._finished:
            raise ValueError(f"job {job_id} is already finished but not collected")

        src_spec, dst_spec = spec
        started = time.perf_counter()
        if isinstance(src_spec, GPULoadStoreSpec) and isinstance(
            dst_spec, SharedStorageLoadStoreSpec
        ):
            future = self.coordinator.submit_store(
                src_spec,
                dst_spec,
                job_id=job_id,
                profile_tid=profile_tid,
                req_id=req_id,
            )
            transfer_type = (
                GPULoadStoreSpec.medium(),
                SharedStorageLoadStoreSpec.medium(),
            )
            direction = "gpu_to_storage"
            num_hashes = len(dst_spec.block_hashes)
        elif isinstance(src_spec, SharedStorageLoadStoreSpec) and isinstance(
            dst_spec, GPULoadStoreSpec
        ):
            future = self.coordinator.submit_load(
                src_spec,
                dst_spec,
                job_id=job_id,
                profile_tid=profile_tid,
                req_id=req_id,
            )
            transfer_type = (
                SharedStorageLoadStoreSpec.medium(),
                GPULoadStoreSpec.medium(),
            )
            direction = "storage_to_gpu"
            num_hashes = len(src_spec.block_hashes)
        else:
            raise TypeError(
                f"unsupported transfer spec: {type(src_spec)!r} -> {type(dst_spec)!r}"
            )

        self._active[job_id] = (future, started, transfer_type)
        return True

    def get_finished(self) -> list[TransferResult]:
        for job_id, (future, started, transfer_type) in list(self._active.items()):
            if not future.done():
                continue
            self._finished[job_id] = self._future_to_transfer_result(
                job_id,
                future,
                started,
                transfer_type,
            )
            self._active.pop(job_id, None)
        results = list(self._finished.values())
        self._finished.clear()
        return results

    def wait(self, job_ids: set[int]) -> None:
        start_ns = now_ns()
        futures = [
            active[0]
            for job_id in job_ids
            if (active := self._active.get(job_id)) is not None
        ]
        if futures:
            futures_wait(futures)
        add_event(
            "py_kvcache.handler.wait",
            "kv_offload",
            start_ns,
            now_ns() - start_ns,
            args={"requested_jobs": len(job_ids), "waited_jobs": len(futures)},
        )

    def shutdown(self) -> None:
        self.is_shutdown = True
        for job_id, (future, started, transfer_type) in list(self._active.items()):
            if future.done():
                self._finished[job_id] = self._future_to_transfer_result(
                    job_id,
                    future,
                    started,
                    transfer_type,
                )
        self._finished.clear()
        self._active.clear()
        self.coordinator.shutdown()

    @staticmethod
    def _future_to_transfer_result(
        job_id: int,
        future: Future[int],
        started: float,
        transfer_type: tuple[str, str],
    ) -> TransferResult:
        try:
            transfer_size = future.result()
            return _make_transfer_result(
                job_id=job_id,
                success=True,
                transfer_size=transfer_size,
                transfer_time=time.perf_counter() - started,
                transfer_type=transfer_type,
            )
        except BaseException as exc:
            logger.exception("py-kvcache transfer job %s failed", job_id)
            fail_start_ns = now_ns()
            add_event(
                "py_kvcache.handler.transfer_failure",
                "kv_offload",
                fail_start_ns,
                0,
                args={"job_id": job_id, "exception": type(exc).__name__},
            )
            return _make_transfer_result(
                job_id=job_id,
                success=False,
                transfer_size=0,
                transfer_time=time.perf_counter() - started,
                transfer_type=transfer_type,
                message=str(exc),
            )


class PyKvCacheOffloadingSpec(OffloadingSpec):
    def __init__(self, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig):
        super().__init__(vllm_config, kv_cache_config)

        self.shared_file_config = SharedFileConfig.from_extra_config(
            self.extra_config
        )
        self._manager: SharedStorageOffloadingManager | None = None
        self._handler: NoopSharedStorageOffloadingHandler | None = None

        parallel_config = getattr(vllm_config, "parallel_config", None)
        model_name = str(
            _get_nested_attr(vllm_config, ("model_config", "model"), "unknown-model")
        )
        dtype = str(
            _get_nested_attr(vllm_config, ("cache_config", "cache_dtype"), "auto")
        ).replace("torch.", "")
        gpu_block_size = _first_int(getattr(self, "gpu_block_size", None), 1)
        block_size_factor = _first_int(getattr(self, "block_size_factor", None), 1)

        self.file_mapper = FileMapper(
            root_dir=self.shared_file_config.root_dir,
            model_name=model_name,
            gpu_block_size=gpu_block_size,
            gpu_blocks_per_file=block_size_factor,
            tp_size=_first_int(
                getattr(parallel_config, "tensor_parallel_size", None), 1
            ),
            pp_size=_first_int(
                getattr(parallel_config, "pipeline_parallel_size", None), 1
            ),
            pcp_size=_first_int(
                getattr(parallel_config, "prefill_context_parallel_size", None), 1
            ),
            rank=_first_int(getattr(parallel_config, "rank", None), 0),
            dtype=dtype,
        )
        os.makedirs(self.file_mapper.base_path, exist_ok=True)

    def get_manager(self) -> OffloadingManager:
        if self._manager is None:
            self._manager = SharedStorageOffloadingManager(
                self.file_mapper,
                preload_batch_size=self.shared_file_config.preload_batch_size,
                enable_preload=self.shared_file_config.enable_preload,
            )
        return self._manager

    def get_handlers(
        self, kv_caches: CanonicalKVCaches
    ) -> Iterator[tuple[type[LoadStoreSpec], type[LoadStoreSpec], OffloadingHandler]]:
        if self._handler is None:
            gpu_block_size = _first_int(getattr(self, "gpu_block_size", None), 1)
            block_size_factor = _first_int(
                getattr(self, "block_size_factor", None),
                1,
            )
            storage_block_size = gpu_block_size * block_size_factor
            layout = ParsedKvLayout.from_canonical_kv_caches(
                gpu_block_size=gpu_block_size,
                storage_block_size=storage_block_size,
                kv_caches=kv_caches,
            )
            coordinator = TransferCoordinator(
                config=self.shared_file_config,
                file_mapper=self.file_mapper,
                layout=layout,
            )
            self._handler = NoopSharedStorageOffloadingHandler(
                coordinator=coordinator,
            )

        yield GPULoadStoreSpec, SharedStorageLoadStoreSpec, self._handler
        yield SharedStorageLoadStoreSpec, GPULoadStoreSpec, self._handler


PyKvcacheOffloadingSpec = PyKvCacheOffloadingSpec
SharedStorageOffloadingSpec = PyKvCacheOffloadingSpec


def _make_transfer_result(
    *,
    job_id: int,
    success: bool,
    transfer_size: int = 0,
    transfer_time: float = 0.0,
    transfer_type: tuple[str, str] | None = None,
    message: str | None = None,
) -> TransferResult:
    if transfer_type is None:
        transfer_type = (
            SharedStorageLoadStoreSpec.medium(),
            GPULoadStoreSpec.medium(),
        )
    try:
        return TransferResult(
            job_id=job_id,
            success=success,
            transfer_size=transfer_size,
            transfer_time=transfer_time,
            transfer_type=transfer_type,
            error_msg=message,
        )
    except TypeError:
        try:
            return TransferResult(job_id=job_id, success=success)
        except TypeError:
            return TransferResult(job_id, success)

__all__ = [
    "NoopSharedStorageOffloadingHandler",
    "PyKvCacheOffloadingSpec",
    "PyKvcacheOffloadingSpec",
    "SharedStorageLoadStoreSpec",
    "SharedStorageOffloadingManager",
    "SharedStorageOffloadingSpec",
    "SharedStoragePreloadMessage",
    "VLLM_AVAILABLE",
]
