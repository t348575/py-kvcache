from __future__ import annotations

import logging
import unittest
from unittest import mock

from py_kvcache._pchip import Pchip
from py_kvcache.break_even import Curve, CurveData, GoldenPoint
from py_kvcache.cost_model import build_cost_tables

try:
    from scipy.interpolate import PchipInterpolator

    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False


F_KNOTS = {0: 0.0, 1024: 0.090, 4096: 0.34, 16384: 1.3}
G_SSD_KNOTS = {0: 0.031, 1024: 0.031, 4096: 0.050, 16384: 0.12}
G_MEM_KNOTS = {0: 0.003, 1024: 0.003, 4096: 0.006, 16384: 0.02}

# First knot != 0 to exercise the anchor-prepend branch; last knot stays past
# the largest sample probe (16384) to avoid the linear-extrapolation region.
NONZERO_START_KNOTS = {512: 0.10, 2048: 0.40, 8192: 1.00, 20000: 1.50}
NONZERO_START_FLOOR = 0.05


def _curve(knots: dict[int, float], floor: float) -> Curve:
    return Curve(floor=floor, knots=tuple(sorted(knots.items())))


def _curve_data(golden: tuple[GoldenPoint, ...] = ()) -> CurveData:
    return CurveData(
        f=_curve(F_KNOTS, 0.0),
        g_ssd=_curve(G_SSD_KNOTS, 0.031),
        g_mem=_curve(G_MEM_KNOTS, 0.003),
        golden=golden,
        kv_bytes_per_token=131072,
    )


# Golden points pinned at knot token counts are self-consistent by
# construction (a monotone cubic Hermite spline reproduces its own knots).
GOLDEN = (
    GoldenPoint(tokens=1024, f=F_KNOTS[1024], g_ssd=G_SSD_KNOTS[1024], g_mem=G_MEM_KNOTS[1024]),
    GoldenPoint(tokens=4096, f=F_KNOTS[4096], g_ssd=G_SSD_KNOTS[4096], g_mem=G_MEM_KNOTS[4096]),
    GoldenPoint(tokens=16384, f=F_KNOTS[16384], g_ssd=G_SSD_KNOTS[16384], g_mem=G_MEM_KNOTS[16384]),
)


class PchipVsScipyTests(unittest.TestCase):
    @unittest.skipUnless(_HAVE_SCIPY, "scipy not installed")
    def test_matches_scipy_across_and_past_knots(self) -> None:
        for knots, floor in (
            (F_KNOTS, 0.0),
            (G_SSD_KNOTS, 0.031),
            (G_MEM_KNOTS, 0.003),
            (NONZERO_START_KNOTS, NONZERO_START_FLOOR),
        ):
            sizes = sorted(knots)
            vals = [knots[s] for s in sizes]
            if sizes[0] != 0:
                sizes = [0] + sizes
                vals = [floor] + vals
            scipy_interp = PchipInterpolator(sizes, vals, extrapolate=True)
            ours = Pchip(sorted(knots), [knots[s] for s in sorted(knots)], floor=floor)

            for x in [0, 1, 500, 1024, 2000, 4096, 8000, 16384]:
                self.assertAlmostEqual(ours(x), float(scipy_interp(x)), places=9)

            # Past the last knot we're deliberately linear, not scipy's cubic;
            # only the boundary slope (scipy's own right-edge derivative) must match.
            end_slope = float(scipy_interp.derivative(1)(sizes[-1]))
            for x in [sizes[-1] + 1, sizes[-1] + 5000]:
                expected = vals[-1] + end_slope * (x - sizes[-1])
                self.assertAlmostEqual(ours(x), expected, places=9)


class PchipAnchorAndEdgeCaseTests(unittest.TestCase):
    def test_anchor_prepends_zero_at_floor(self) -> None:
        p = Pchip([512, 2048, 8192], [0.10, 0.40, 1.00], floor=0.05)
        self.assertEqual(p(0), 0.05)

    def test_two_point_input_is_linear(self) -> None:
        p = Pchip([0, 100], [0.0, 10.0])
        self.assertAlmostEqual(p(50), 5.0)
        self.assertAlmostEqual(p(150), 15.0)

    def test_edge_case_zero_when_sign_disagrees_with_secant(self) -> None:
        # h0=h1=1, m0=1, m1=5 -> d=(3*m0-m1)/2=-1, sign(d) != sign(m0) -> d[0]=0.
        p = Pchip([0, 1, 2], [0.0, 1.0, 6.0])
        self.assertAlmostEqual(p(-1), 0.0)

    def test_edge_case_clamped_to_three_times_secant(self) -> None:
        # h0=h1=1, m0=1, m1=-10 -> d=6.5, same sign as m0 but sign(m0)!=sign(m1)
        # and abs(d) > abs(3*m0) -> d[0] clamps to 3*m0=3.
        p = Pchip([0, 1, 2], [0.0, 1.0, -9.0])
        self.assertAlmostEqual(p(-1), -3.0)

    def test_single_knot_raises_value_error(self) -> None:
        # Pins the `len(xs) < 2` guard, not `_derivatives`'s internal one.
        with self.assertRaises(ValueError) as ctx:
            Pchip([0], [0.0])
        self.assertIn("got 1", str(ctx.exception))

    def test_non_monotonic_xs_raises_value_error_not_zero_division(self) -> None:
        # Repeated x would divide by zero inside _derivatives without this guard.
        with self.assertRaises(ValueError) as ctx:
            Pchip([0, 5, 5], [0.0, 1.0, 2.0])
        self.assertIn("strictly increasing", str(ctx.exception))


class GoldenSelfCheckTests(unittest.TestCase):
    def test_passes_on_consistent_file(self) -> None:
        tables = build_cost_tables(
            _curve_data(GOLDEN), block_tokens=16, max_model_len=32768
        )
        self.assertIsNotNone(tables)

    def test_raises_on_tampered_golden(self) -> None:
        tampered_f = 999.0
        tampered = (
            GoldenPoint(tokens=1024, f=tampered_f, g_ssd=G_SSD_KNOTS[1024], g_mem=G_MEM_KNOTS[1024]),
        ) + GOLDEN[1:]
        with self.assertRaises(ValueError) as ctx:
            build_cost_tables(
                _curve_data(tampered), block_tokens=16, max_model_len=32768
            )
        message = str(ctx.exception)
        f_interp = Pchip(sorted(F_KNOTS), [F_KNOTS[s] for s in sorted(F_KNOTS)], floor=0.0)
        produced = f_interp(1024)
        self.assertIn("'f'", message)
        self.assertIn(str(1024), message)
        self.assertIn(str(tampered_f), message)
        self.assertIn(str(produced), message)


class CostTablesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.block_tokens = 16
        self.max_model_len = 32768
        self.curves = _curve_data(GOLDEN)
        self.tables = build_cost_tables(
            self.curves, block_tokens=self.block_tokens, max_model_len=self.max_model_len
        )
        self.f_interp = Pchip(sorted(F_KNOTS), [F_KNOTS[s] for s in sorted(F_KNOTS)], floor=0.0)
        self.ssd_interp = Pchip(
            sorted(G_SSD_KNOTS), [G_SSD_KNOTS[s] for s in sorted(G_SSD_KNOTS)], floor=0.031
        )
        self.mem_interp = Pchip(
            sorted(G_MEM_KNOTS), [G_MEM_KNOTS[s] for s in sorted(G_MEM_KNOTS)], floor=0.003
        )

    def test_exact_at_block_multiples(self) -> None:
        for n in range(self.tables.max_blocks + 1):
            tok = n * self.block_tokens
            expected = max(0.0, self.f_interp(tok))
            self.assertEqual(self.tables.recompute_s(n), expected)

    def test_negative_values_clipped_to_zero(self) -> None:
        # f's floor is 0 and rises monotonically here, so force a curve with
        # a dip below zero to exercise the clip.
        dipping = _curve({0: 0.0, 100: -5.0, 200: 10.0}, 0.0)
        curves = CurveData(
            f=dipping, g_ssd=self.curves.g_ssd, g_mem=self.curves.g_mem, golden=(), kv_bytes_per_token=None
        )
        tables = build_cost_tables(curves, block_tokens=100, max_model_len=200)
        self.assertEqual(tables.recompute_s(1), 0.0)

    def test_linear_past_last_knot(self) -> None:
        last_tok = 16384
        step = self.block_tokens
        n_last = last_tok // step
        diffs = set()
        for n in range(n_last, n_last + 5):
            a = self.tables.recompute_s(n)
            b = self.tables.recompute_s(n + 1)
            diffs.add(round(b - a, 9))
        self.assertEqual(len(diffs), 1)

    def test_service_s_is_max_zero_ssd_minus_mem(self) -> None:
        # Derived from independent interpolators, not the baked g_ssd_s/g_mem_s.
        for n in range(self.tables.max_blocks + 1):
            tok = n * self.block_tokens
            expected = max(
                0.0, max(0.0, self.ssd_interp(tok)) - max(0.0, self.mem_interp(tok))
            )
            self.assertEqual(self.tables.service_s_of(n), expected)

    def test_service_s_clips_inputs_before_subtracting(self) -> None:
        # Pins clip-then-subtract order: clipping only the difference would
        # yield a larger service here since g_mem dips negative.
        dipping_mem = _curve({0: 0.010, 100: -0.010, 200: 0.010}, 0.010)
        curves = CurveData(
            f=self.curves.f,
            g_ssd=_curve({0: 0.05, 100: 0.05, 200: 0.05}, 0.05),
            g_mem=dipping_mem,
            golden=(),
            kv_bytes_per_token=None,
        )
        tables = build_cost_tables(curves, block_tokens=100, max_model_len=200)

        self.assertEqual(tables.g_mem_s[1], 0.0)
        self.assertAlmostEqual(tables.service_s_of(1), 0.05)

    def test_accessor_clamping(self) -> None:
        self.assertEqual(self.tables.recompute_s(-5), self.tables.recompute_s(0))
        self.assertEqual(
            self.tables.recompute_s(self.tables.max_blocks + 1000),
            self.tables.recompute_s(self.tables.max_blocks),
        )


class ExtrapolationWarningTests(unittest.TestCase):
    def test_warns_naming_max_model_len_and_last_knot(self) -> None:
        with self.assertLogs("py_kvcache.cost_model", level="WARNING") as logs:
            build_cost_tables(_curve_data(GOLDEN), block_tokens=16, max_model_len=32768)
        (message,) = logs.output
        self.assertIn("32768", message)
        self.assertIn("16384", message)

    def test_silent_when_the_table_stays_inside_the_measured_range(self) -> None:
        logger = logging.getLogger("py_kvcache.cost_model")
        with mock.patch.object(logger, "warning") as warn:
            build_cost_tables(_curve_data(GOLDEN), block_tokens=16, max_model_len=16384)
        warn.assert_not_called()

    def test_shortest_curve_sets_the_reported_knot(self) -> None:
        # A curve is extrapolating as soon as *its own* knots run out, so the
        # threshold is the smallest last knot across the three, not the largest.
        curves = CurveData(
            f=_curve(F_KNOTS, 0.0),
            g_ssd=_curve({0: 0.031, 4096: 0.05}, 0.031),
            g_mem=_curve(G_MEM_KNOTS, 0.003),
            golden=(),
            kv_bytes_per_token=None,
        )
        with self.assertLogs("py_kvcache.cost_model", level="WARNING") as logs:
            build_cost_tables(curves, block_tokens=16, max_model_len=8192)
        self.assertIn("4096", logs.output[0])


class BuildCostTablesValidationTests(unittest.TestCase):
    def test_non_positive_block_tokens_raises(self) -> None:
        with self.assertRaises(ValueError):
            build_cost_tables(_curve_data(), block_tokens=0, max_model_len=1024)

    def test_negative_max_model_len_raises(self) -> None:
        with self.assertRaises(ValueError):
            build_cost_tables(_curve_data(), block_tokens=16, max_model_len=-1)


if __name__ == "__main__":
    unittest.main()
