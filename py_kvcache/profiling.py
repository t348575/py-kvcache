from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Iterator

try:
    from simple_profiler import profile_scope as _profile_scope
    from simple_profiler import profiler as _profiler
except Exception:
    _profile_scope = None

    class _NoopProfiler:
        _active = False

        def add_event(self, *args: Any, **kwargs: Any) -> None:
            return None

    _profiler = _NoopProfiler()


profiler = _profiler


def is_active() -> bool:
    return bool(getattr(profiler, "_active", False))


def now_ns() -> int:
    return time.perf_counter_ns()


@contextmanager
def profile_scope(name: str, category: str, **kwargs: Any) -> Iterator[None]:
    if _profile_scope is None:
        yield
        return
    with _profile_scope(name, category, **kwargs):
        yield


def add_event(
    name: str,
    category: str,
    start_ns: int,
    duration_ns: int,
    *,
    tid: str | None = None,
    args: dict[str, Any] | None = None,
) -> None:
    if not is_active():
        return
    kwargs: dict[str, Any] = {}
    if tid is not None:
        kwargs["tid"] = tid
    if args is not None:
        kwargs["args"] = args
    profiler.add_event(name, category, start_ns, duration_ns, **kwargs)


__all__ = ["add_event", "is_active", "now_ns", "profile_scope", "profiler"]
