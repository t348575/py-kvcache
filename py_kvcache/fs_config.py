from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

DEFAULT_IODEPTH = 16
DEFAULT_STAGING_MEM_GIB = 1.0
# Multiple of a request's own predicted storage wait before it stops deferring
# and reads foreground.
DEFAULT_LOAD_PLANNER_DEFER_TOLERANCE = 2.0
# Hard cap on any single deferral, regardless of tolerance.
DEFAULT_LOAD_PLANNER_DEFER_DEADLINE_MAX_S = 2.0


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
    iodepth: int = DEFAULT_IODEPTH
    staging_mem: float = DEFAULT_STAGING_MEM_GIB
    enable_preload: bool = False
    # Identical-prefix preload candidates share one disk read + staging slot,
    # refcounted.
    preload_share_staging: bool = True
    # Max in-flight async openat ops + opened-but-unread fds. Defaults to iodepth.
    open_lookahead: int | None = None
    # "off" (default) | "lru" | "arc". Store buffers are never cached.
    staging_cache: str = "off"
    # Path to a per-(GPU, model, SSD, dtype) break-even data file. Absent disables
    # load-side break-even gating.
    prefix_cache_break_even_path: str | None = None
    # Plan width comes from preload_lookahead_requests, not a knob of its own.
    load_planner: str = "off"
    # Multiple of predicted storage wait before a deferred candidate falls back
    # to a foreground load (never a decline).
    load_planner_defer_tolerance: float = DEFAULT_LOAD_PLANNER_DEFER_TOLERANCE
    load_planner_defer_deadline_max_s: float = DEFAULT_LOAD_PLANNER_DEFER_DEADLINE_MAX_S

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
        staging_cache = _coerce_str(
            extra_config.get("staging_cache"),
            field_name="staging_cache",
        )
        staging_cache = "off" if staging_cache is None else staging_cache.strip().lower()
        if staging_cache not in {"off", "lru", "arc"}:
            raise ValueError("staging_cache must be one of: off, lru, arc")

        prefix_cache_break_even_path = _coerce_str(
            extra_config.get("prefix_cache_break_even_path"),
            field_name="prefix_cache_break_even_path",
        )

        load_planner = _coerce_str(
            extra_config.get("load_planner"),
            field_name="load_planner",
        )
        load_planner = "off" if load_planner is None else load_planner.strip().lower()
        if load_planner not in {"off", "on"}:
            raise ValueError("load_planner must be one of: off, on")

        if extra_config.get("load_planner_defer_deadline_steps") is not None:
            # Raise instead of ignoring it: silently dropping the setting would
            # change behavior for anyone who had tuned it.
            raise ValueError(
                "load_planner_defer_deadline_steps has been removed; the defer "
                "deadline is now wall-clock and derived per request. Use "
                "load_planner_defer_tolerance instead"
            )

        load_planner_defer_tolerance = _coerce_float(
            extra_config.get("load_planner_defer_tolerance"),
            field_name="load_planner_defer_tolerance",
        )
        load_planner_defer_deadline_max_s = _coerce_float(
            extra_config.get("load_planner_defer_deadline_max_s"),
            field_name="load_planner_defer_deadline_max_s",
        )

        _validate_positive(iodepth, field_name="iodepth")
        _validate_positive(open_lookahead, field_name="open_lookahead")
        if load_planner_defer_tolerance is not None and load_planner_defer_tolerance <= 0:
            raise ValueError("load_planner_defer_tolerance must be positive when provided")
        if (
            load_planner_defer_deadline_max_s is not None
            and load_planner_defer_deadline_max_s <= 0
        ):
            raise ValueError(
                "load_planner_defer_deadline_max_s must be positive when provided"
            )
        if staging_mem is not None and staging_mem <= 0:
            raise ValueError("staging_mem must be positive when provided")

        return cls(
            root_dir=root_dir,
            sync_on_store=False if sync_on_store is None else sync_on_store,
            iodepth=DEFAULT_IODEPTH if iodepth is None else iodepth,
            staging_mem=DEFAULT_STAGING_MEM_GIB if staging_mem is None else staging_mem,
            enable_preload=False if enable_preload is None else enable_preload,
            preload_share_staging=(
                True if preload_share_staging is None else preload_share_staging
            ),
            open_lookahead=open_lookahead,
            staging_cache=staging_cache,
            prefix_cache_break_even_path=prefix_cache_break_even_path,
            load_planner=load_planner,
            load_planner_defer_tolerance=(
                DEFAULT_LOAD_PLANNER_DEFER_TOLERANCE
                if load_planner_defer_tolerance is None
                else load_planner_defer_tolerance
            ),
            load_planner_defer_deadline_max_s=(
                DEFAULT_LOAD_PLANNER_DEFER_DEADLINE_MAX_S
                if load_planner_defer_deadline_max_s is None
                else load_planner_defer_deadline_max_s
            ),
        )


__all__ = ["SharedFileConfig"]
