from __future__ import annotations


def resolve_preload_target_io_depth(
    configured_depth: int | None, default_depth: int
) -> int:
    if default_depth < 0:
        raise ValueError("default preload target IO depth must be non-negative")
    if configured_depth is None:
        return default_depth
    if configured_depth < 0:
        raise ValueError("preload_target_io_depth must be non-negative")
    return configured_depth


def preload_submission_capacity(
    *,
    target_io_depth: int,
    reserved_io_depth: int,
    pending_io_depth_reservations: int,
    requested_blocks: int,
) -> int:
    if target_io_depth < 0:
        raise ValueError("target_io_depth must be non-negative")
    if reserved_io_depth < 0:
        raise ValueError("reserved_io_depth must be non-negative")
    if pending_io_depth_reservations < 0:
        raise ValueError("pending_io_depth_reservations must be non-negative")
    if requested_blocks <= 0:
        return 0

    system_io_depth = reserved_io_depth + pending_io_depth_reservations
    if system_io_depth + requested_blocks > target_io_depth:
        return 0
    return requested_blocks


__all__ = ["preload_submission_capacity", "resolve_preload_target_io_depth"]
