from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


def _coerce_str(value: object | None, *, field_name: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    raise TypeError(f"{field_name} must be a string when provided")


def _coerce_int(value: object | None, *, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be an integer when provided")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return int(value)
    raise TypeError(f"{field_name} must be an integer when provided")


def _coerce_bool(value: object | None, *, field_name: str) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise TypeError(f"{field_name} must be a boolean when provided")


@dataclass(frozen=True)
class SharedFileConfig:
    root_dir: str
    direct: bool = False
    sync_on_store: bool = False
    parallel_files: int | None = None
    create_mode: int | None = None
    io_alignment: int = 4096
    be: str | None = None
    mem: str | None = None
    sync: str | None = None
    async_backend: str | None = None
    admin: str | None = None

    @classmethod
    def from_extra_config(
        cls, extra_config: Mapping[str, object]
    ) -> "SharedFileConfig":
        root_dir = _coerce_str(
            extra_config.get("shared_storage_path", extra_config.get("device_uri")),
            field_name="shared_storage_path",
        )
        if not root_dir:
            raise ValueError(
                "shared_storage_path must be provided in kv_connector_extra_config"
            )

        nested = extra_config.get("xnvme_opts")
        if nested is None:
            nested_opts: Mapping[str, object] = {}
        elif isinstance(nested, Mapping):
            nested_opts = nested
        else:
            raise TypeError("xnvme_opts must be a mapping when provided")

        def resolve(name: str) -> object | None:
            if name in nested_opts:
                return nested_opts[name]
            return extra_config.get(f"xnvme_{name}")

        direct = _coerce_bool(
            extra_config.get("direct_io", resolve("direct")),
            field_name="direct_io",
        )
        sync_on_store = _coerce_bool(
            extra_config.get("sync_on_store", extra_config.get("xnvme_sync_on_store")),
            field_name="sync_on_store",
        )
        io_alignment = _coerce_int(
            extra_config.get("io_alignment", extra_config.get("direct_io_alignment")),
            field_name="io_alignment",
        )
        parallel_files = _coerce_int(
            extra_config.get("parallel_files", extra_config.get("xnvme_parallel_files")),
            field_name="parallel_files",
        )

        return cls(
            root_dir=root_dir,
            direct=bool(direct) if direct is not None else False,
            sync_on_store=False if sync_on_store is None else sync_on_store,
            parallel_files=parallel_files,
            create_mode=_coerce_int(
                resolve("create_mode"), field_name="xnvme_create_mode"
            ),
            io_alignment=io_alignment if io_alignment is not None else 4096,
            be=_coerce_str(resolve("be"), field_name="xnvme_be"),
            mem=_coerce_str(resolve("mem"), field_name="xnvme_mem"),
            sync=_coerce_str(resolve("sync"), field_name="xnvme_sync"),
            async_backend=_coerce_str(resolve("async"), field_name="xnvme_async"),
            admin=_coerce_str(resolve("admin"), field_name="xnvme_admin"),
        )


__all__ = ["SharedFileConfig"]
