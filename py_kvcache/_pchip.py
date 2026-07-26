from __future__ import annotations

import bisect


def _sign(v: float) -> int:
    return (v > 0) - (v < 0)


def _edge_derivative(h0: float, h1: float, m0: float, m1: float) -> float:
    """scipy's one-sided three-point endpoint estimate (`_edge_case`)."""
    d = ((2 * h0 + h1) * m0 - h0 * m1) / (h0 + h1)
    if _sign(d) != _sign(m0):
        return 0.0
    if _sign(m0) != _sign(m1) and abs(d) > abs(3 * m0):
        return 3 * m0
    return d


def _derivatives(xs: list[float], ys: list[float]) -> list[float]:
    n = len(xs)
    if n < 2:
        raise ValueError("Pchip needs at least two knots")
    h = [xs[i + 1] - xs[i] for i in range(n - 1)]
    m = [(ys[i + 1] - ys[i]) / h[i] for i in range(n - 1)]
    if n == 2:
        return [m[0], m[0]]

    d = [0.0] * n
    for k in range(1, n - 1):
        m_prev, m_next = m[k - 1], m[k]
        if m_prev == 0.0 or m_next == 0.0 or _sign(m_prev) != _sign(m_next):
            continue
        w1 = 2 * h[k] + h[k - 1]
        w2 = h[k] + 2 * h[k - 1]
        d[k] = (w1 + w2) / (w1 / m_prev + w2 / m_next)

    d[0] = _edge_derivative(h[0], h[1], m[0], m[1])
    d[-1] = _edge_derivative(h[-1], h[-2], m[-1], m[-2])
    return d


class Pchip:
    """Monotone cubic Hermite interpolant (Fritsch-Carlson), matching
    `scipy.interpolate.PchipInterpolator`'s derivative rule exactly. Unlike
    scipy, extrapolation past the knot range is linear from the boundary
    segment's end slope, not cubic.
    """

    def __init__(self, xs: list[float], ys: list[float], *, floor: float = 0.0) -> None:
        xs = list(xs)
        ys = list(ys)
        if not xs:
            raise ValueError("Pchip requires at least two knots, got 0")
        if xs[0] != 0:
            xs = [0.0] + xs
            ys = [floor] + ys
        if len(xs) < 2:
            raise ValueError(f"Pchip requires at least two knots, got {len(xs)}")
        if any(xs[i] >= xs[i + 1] for i in range(len(xs) - 1)):
            raise ValueError("Pchip requires strictly increasing xs")
        self._xs = xs
        self._ys = ys
        self._ds = _derivatives(xs, ys)

    def __call__(self, x: float) -> float:
        xs, ys, ds = self._xs, self._ys, self._ds
        if x <= xs[0]:
            return ys[0] + ds[0] * (x - xs[0])
        if x >= xs[-1]:
            return ys[-1] + ds[-1] * (x - xs[-1])

        i = bisect.bisect_right(xs, x) - 1
        h = xs[i + 1] - xs[i]
        t = (x - xs[i]) / h
        h00 = 2 * t**3 - 3 * t**2 + 1
        h10 = t**3 - 2 * t**2 + t
        h01 = -2 * t**3 + 3 * t**2
        h11 = t**3 - t**2
        return h00 * ys[i] + h10 * h * ds[i] + h01 * ys[i + 1] + h11 * h * ds[i + 1]


__all__ = ["Pchip"]
