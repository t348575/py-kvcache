from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BreakEvenThresholds:
    """Medium-aware break-even prefix lengths (effective tokens, margin folded in).

    A load is worth doing when its prefix is at least the threshold for the medium
    that will serve it. RAM-served loads use `mem_tokens` (typically ~0, so they
    are effectively never gated); SSD-served loads use `ssd_tokens`.
    """

    ssd_tokens: int = 0
    mem_tokens: int = 0

    @property
    def enabled(self) -> bool:
        return self.ssd_tokens > 0 or self.mem_tokens > 0


def break_even_threshold(thresholds: BreakEvenThresholds, *, ram_resident: bool) -> int:
    return thresholds.mem_tokens if ram_resident else thresholds.ssd_tokens


def should_load(
    prefix_tokens: int,
    thresholds: BreakEvenThresholds,
    *,
    ram_resident: bool,
) -> bool:
    """Whether a load of `prefix_tokens` should run, or be declined for recompute.

    Reference implementation of the worker load gate; pure so the boundary is
    unit-testable without torch / io_uring / vLLM. A threshold of 0 disables
    gating for that medium.
    """
    threshold = break_even_threshold(thresholds, ram_resident=ram_resident)
    return not (threshold > 0 and 0 < prefix_tokens < threshold)


def _non_negative_int(value: object, *, path: str, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(
            f"break-even file {path!r}: {field} must be a non-negative integer"
        )
    return value


def load_break_even(
    path: str | None,
    *,
    model_name: str,
    kv_dtype: str,
) -> BreakEvenThresholds:
    """Resolve a break-even data file to medium-aware token thresholds.

    Returns disabled thresholds (0, 0) when no path is configured. Raises on a
    missing/malformed file or a model-name mismatch so a misconfigured deployment
    fails loudly rather than gating on a wrong `P*`.
    """
    if not path:
        return BreakEvenThresholds()

    try:
        with open(path) as f:
            data = json.load(f)
    except OSError as exc:
        raise ValueError(
            f"prefix_cache_break_even_path {path!r} cannot be read: {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"prefix_cache_break_even_path {path!r} is not valid JSON: {exc}"
        ) from exc

    if not isinstance(data, Mapping):
        raise ValueError(
            f"prefix_cache_break_even_path {path!r} must contain a JSON object"
        )

    file_model = data.get("model_name")
    if file_model is not None and file_model != model_name:
        raise ValueError(
            f"break-even file {path!r} is for model {file_model!r}, "
            f"but the running model is {model_name!r}"
        )

    file_dtype = data.get("kv_dtype")
    if file_dtype is not None and file_dtype != kv_dtype:
        # dtype strings can differ harmlessly (e.g. "auto" resolution), so warn
        # rather than fail — P* is a perf threshold, not a correctness gate.
        logger.warning(
            "break-even file %r kv_dtype %r != running kv_dtype %r; using it anyway",
            path,
            file_dtype,
            kv_dtype,
        )

    margin = _non_negative_int(
        data.get("safety_margin_tokens", 0),
        path=path,
        field="safety_margin_tokens",
    )
    ssd = _non_negative_int(
        data.get("break_even_ssd_tokens", 0),
        path=path,
        field="break_even_ssd_tokens",
    )
    mem = _non_negative_int(
        data.get("break_even_mem_tokens", 0),
        path=path,
        field="break_even_mem_tokens",
    )
    return BreakEvenThresholds(
        ssd_tokens=ssd + margin if ssd else 0,
        mem_tokens=mem + margin if mem else 0,
    )


__all__ = [
    "BreakEvenThresholds",
    "break_even_threshold",
    "load_break_even",
    "should_load",
]
