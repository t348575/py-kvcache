from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


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
        return int(value, 0)
    raise TypeError(f"{field_name} must be an integer when provided")


def _coerce_float(value: object | None, *, field_name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be a number when provided")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise TypeError(f"{field_name} must be a number when provided")


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


def _validate_positive(value: int | None, *, field_name: str) -> None:
    if value is not None and value <= 0:
        raise ValueError(f"{field_name} must be positive when provided")


@dataclass(frozen=True)
class SharedFileConfig:
    root_dir: str
    sync_on_store: bool = False
    iodepth: int | None = None
    staging_mem: float | None = None
    enable_preload: bool = True
    # When on, identical-prefix preload candidates share one disk read + one staging
    # slot, released by a reference counter once all demanders are served.
    preload_share_staging: bool = True
    # Max in-flight async openat ops + opened-but-unread fds. Defaults to iodepth.
    open_lookahead: int | None = None

    @classmethod
    def from_extra_config(cls, extra_config: Mapping[str, object]) -> "SharedFileConfig":
        root_dir = _coerce_str(
            extra_config.get("shared_storage_path"),
            field_name="shared_storage_path",
        )
        if not root_dir:
            raise ValueError("shared_storage_path must be provided in kv_connector_extra_config")

        sync_on_store = _coerce_bool(
            extra_config.get("sync_on_store"),
            field_name="sync_on_store",
        )
        iodepth = _coerce_int(
            extra_config.get("iodepth"),
            field_name="iodepth",
        )
        staging_mem = _coerce_float(
            extra_config.get("staging_mem"),
            field_name="staging_mem",
        )
        enable_preload = _coerce_bool(
            extra_config.get("enable_preload"),
            field_name="enable_preload",
        )
        preload_share_staging = _coerce_bool(
            extra_config.get("preload_share_staging"),
            field_name="preload_share_staging",
        )
        open_lookahead = _coerce_int(
            extra_config.get("open_lookahead"),
            field_name="open_lookahead",
        )

        _validate_positive(iodepth, field_name="iodepth")
        _validate_positive(open_lookahead, field_name="open_lookahead")
        if staging_mem is not None and staging_mem <= 0:
            raise ValueError("staging_mem must be positive when provided")

        return cls(
            root_dir=root_dir,
            sync_on_store=False if sync_on_store is None else sync_on_store,
            iodepth=iodepth,
            staging_mem=staging_mem,
            enable_preload=True if enable_preload is None else enable_preload,
            preload_share_staging=(
                True if preload_share_staging is None else preload_share_staging
            ),
            open_lookahead=open_lookahead,
        )


__all__ = ["SharedFileConfig"]
