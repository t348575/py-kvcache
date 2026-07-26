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


@dataclass(frozen=True)
class Curve:
    floor: float
    knots: tuple[tuple[int, float], ...]  # sorted by token count, ascending


@dataclass(frozen=True)
class GoldenPoint:
    tokens: int
    f: float
    g_ssd: float
    g_mem: float


@dataclass(frozen=True)
class CurveData:
    f: Curve
    g_ssd: Curve
    g_mem: Curve
    golden: tuple[GoldenPoint, ...]
    kv_bytes_per_token: int | None


_CURVE_NAMES = ("f", "g_ssd", "g_mem")


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


def _read_and_validate(path: str, *, model_name: str, kv_dtype: str) -> Mapping[str, object]:
    """Read + parse `path`, validate model/dtype. Shared by `load_break_even`
    and `load_curves` so both raise/warn identically on the same file.
    """
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
        # Warn, don't fail -- P* is a perf threshold, not a correctness gate.
        logger.warning(
            "break-even file %r kv_dtype %r != running kv_dtype %r; using it anyway",
            path,
            file_dtype,
            kv_dtype,
        )

    return data


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

    data = _read_and_validate(path, model_name=model_name, kv_dtype=kv_dtype)

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


def _parse_curve(raw: object, *, path: str, name: str) -> Curve:
    if not isinstance(raw, Mapping):
        raise ValueError(f"break-even file {path!r}: curve {name!r} must be a JSON object")

    floor = raw.get("floor")
    if isinstance(floor, bool) or not isinstance(floor, (int, float)):
        raise ValueError(f"break-even file {path!r}: curve {name!r} floor must be numeric")

    knots_raw = raw.get("knots")
    if not isinstance(knots_raw, Mapping) or not knots_raw:
        raise ValueError(
            f"break-even file {path!r}: curve {name!r} knots must be a non-empty object"
        )

    parsed: list[tuple[int, float]] = []
    for key, value in knots_raw.items():
        try:
            token = int(key)
        except (TypeError, ValueError):
            raise ValueError(
                f"break-even file {path!r}: curve {name!r} knot key {key!r} is not an integer"
            ) from None
        if token < 0:
            raise ValueError(
                f"break-even file {path!r}: curve {name!r} knot key {key!r} must be non-negative"
            )
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"break-even file {path!r}: curve {name!r} knot {key!r} value must be numeric"
            )
        parsed.append((token, float(value)))

    parsed.sort(key=lambda kv: kv[0])
    tokens_seen = [token for token, _ in parsed]
    if len(set(tokens_seen)) != len(tokens_seen):
        raise ValueError(
            f"break-even file {path!r}: curve {name!r} has duplicate/non-monotonic "
            "knot token counts"
        )
    # A single knot at token 0 gets no (0, floor) anchor prepended, so it never
    # reaches two interpolation points.
    if len(parsed) < 2 and parsed[0][0] == 0:
        raise ValueError(
            f"break-even file {path!r}: curve {name!r} needs at least two knots"
        )

    return Curve(floor=float(floor), knots=tuple(parsed))


def _parse_golden_point(raw: object, *, path: str, index: int) -> GoldenPoint:
    if not isinstance(raw, Mapping):
        raise ValueError(f"break-even file {path!r}: golden[{index}] must be a JSON object")

    tokens = raw.get("tokens")
    if isinstance(tokens, bool) or not isinstance(tokens, int):
        raise ValueError(f"break-even file {path!r}: golden[{index}].tokens must be an integer")

    values: dict[str, float] = {}
    for name in _CURVE_NAMES:
        value = raw.get(name)
        if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"break-even file {path!r}: golden[{index}] missing numeric {name!r}"
            )
        values[name] = float(value)

    return GoldenPoint(tokens=tokens, f=values["f"], g_ssd=values["g_ssd"], g_mem=values["g_mem"])


def load_curves(
    path: str | None,
    *,
    model_name: str,
    kv_dtype: str,
) -> CurveData | None:
    """Parse the v2 cost-curve section of a break-even file, if present.

    Returns `None` when no path is configured, or when the file is a v1 file
    with no `curves` key — that's not an error, it just means the caller
    decides whether the cost-model planner can run.
    """
    if not path:
        return None

    data = _read_and_validate(path, model_name=model_name, kv_dtype=kv_dtype)

    curves_raw = data.get("curves")
    if curves_raw is None:
        return None
    if not isinstance(curves_raw, Mapping):
        raise ValueError(f"break-even file {path!r}: curves must be a JSON object")

    if set(curves_raw.keys()) != set(_CURVE_NAMES):
        raise ValueError(
            f"break-even file {path!r}: curves must have exactly {_CURVE_NAMES}, "
            f"got {sorted(curves_raw.keys())}"
        )

    curves = {
        name: _parse_curve(curves_raw[name], path=path, name=name) for name in _CURVE_NAMES
    }

    golden_raw = data.get("golden")
    if golden_raw is None:
        golden_raw = []
    elif not isinstance(golden_raw, list):
        raise ValueError(f"break-even file {path!r}: golden must be a list")
    golden = tuple(
        _parse_golden_point(entry, path=path, index=i) for i, entry in enumerate(golden_raw)
    )

    kv_bytes = data.get("kv_bytes_per_token")
    kv_bytes_invalid = (
        isinstance(kv_bytes, bool) or not isinstance(kv_bytes, int) or kv_bytes <= 0
    )
    if kv_bytes is not None and kv_bytes_invalid:
        raise ValueError(
            f"break-even file {path!r}: kv_bytes_per_token must be a positive integer"
        )

    return CurveData(
        f=curves["f"],
        g_ssd=curves["g_ssd"],
        g_mem=curves["g_mem"],
        golden=golden,
        kv_bytes_per_token=kv_bytes,
    )


__all__ = [
    "BreakEvenThresholds",
    "Curve",
    "CurveData",
    "GoldenPoint",
    "break_even_threshold",
    "load_break_even",
    "load_curves",
    "should_load",
]
