from __future__ import annotations

import logging
from dataclasses import dataclass

from py_kvcache._pchip import Pchip
from py_kvcache.break_even import Curve, CurveData

logger = logging.getLogger(__name__)

_GOLDEN_ABS_TOL = 1e-6
_GOLDEN_REL_TOL = 1e-3


@dataclass(frozen=True)
class CostTables:
    """Block-granular cost tables baked from `CurveData` at construction time.

    All accessors index a precomputed tuple; nothing is recomputed per call.
    """

    block_tokens: int
    f_s: tuple[float, ...]
    g_ssd_s: tuple[float, ...]
    g_mem_s: tuple[float, ...]
    service_s: tuple[float, ...]
    max_blocks: int

    def _clamp(self, nblocks: int) -> int:
        if nblocks < 0:
            return 0
        if nblocks > self.max_blocks:
            return self.max_blocks
        return nblocks

    def recompute_s(self, nblocks: int) -> float:
        return self.f_s[self._clamp(nblocks)]

    def load_ssd_s(self, nblocks: int) -> float:
        return self.g_ssd_s[self._clamp(nblocks)]

    def load_mem_s(self, nblocks: int) -> float:
        return self.g_mem_s[self._clamp(nblocks)]

    def service_s_of(self, nblocks: int) -> float:
        return self.service_s[self._clamp(nblocks)]


def _build_interpolator(curve: Curve) -> Pchip:
    xs = [token for token, _ in curve.knots]
    ys = [value for _, value in curve.knots]
    return Pchip(xs, ys, floor=curve.floor)


def build_cost_tables(
    curves: CurveData,
    *,
    block_tokens: int,
    max_model_len: int,
) -> CostTables:
    if block_tokens <= 0:
        raise ValueError(f"block_tokens must be positive, got {block_tokens}")
    if max_model_len < 0:
        raise ValueError(f"max_model_len must be non-negative, got {max_model_len}")

    interpolators = {
        "f": _build_interpolator(curves.f),
        "g_ssd": _build_interpolator(curves.g_ssd),
        "g_mem": _build_interpolator(curves.g_mem),
    }

    for point in curves.golden:
        for name in ("f", "g_ssd", "g_mem"):
            expected = getattr(point, name)
            produced = interpolators[name](point.tokens)
            if not _golden_ok(expected, produced):
                raise ValueError(
                    f"cost model golden self-check failed for curve {name!r} at "
                    f"{point.tokens} tokens: expected {expected}, produced {produced}"
                )

    last_knot = min(curve.knots[-1][0] for curve in (curves.f, curves.g_ssd, curves.g_mem))
    if max_model_len > last_knot:
        # Golden points never cover above the last knot, so the self-check above doesn't either.
        logger.warning(
            "cost model extrapolates: max_model_len=%d exceeds the last measured "
            "curve knot at %d tokens; every table entry above it is linear "
            "extrapolation from the final segment slope, unverified by the golden "
            "self-check, and under-prices superlinear prefill",
            max_model_len,
            last_knot,
        )

    max_blocks = max_model_len // block_tokens
    f_s: list[float] = []
    g_ssd_s: list[float] = []
    g_mem_s: list[float] = []
    service_s: list[float] = []
    for n in range(max_blocks + 1):
        tok = n * block_tokens
        f_v = max(0.0, interpolators["f"](tok))
        ssd_v = max(0.0, interpolators["g_ssd"](tok))
        mem_v = max(0.0, interpolators["g_mem"](tok))
        f_s.append(f_v)
        g_ssd_s.append(ssd_v)
        g_mem_s.append(mem_v)
        service_s.append(max(0.0, ssd_v - mem_v))

    return CostTables(
        block_tokens=block_tokens,
        f_s=tuple(f_s),
        g_ssd_s=tuple(g_ssd_s),
        g_mem_s=tuple(g_mem_s),
        service_s=tuple(service_s),
        max_blocks=max_blocks,
    )


def _golden_ok(expected: float, produced: float) -> bool:
    diff = abs(expected - produced)
    if diff <= _GOLDEN_ABS_TOL:
        return True
    if expected == 0:
        return False
    return diff / abs(expected) <= _GOLDEN_REL_TOL


__all__ = ["CostTables", "build_cost_tables"]
