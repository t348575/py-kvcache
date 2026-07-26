import json
import os
import tempfile
import unittest

from py_kvcache.break_even import (
    BreakEvenThresholds,
    CurveData,
    break_even_threshold,
    load_break_even,
    load_curves,
    should_load,
)
from py_kvcache.fs_config import SharedFileConfig

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DTYPE = "auto"

V2_PAYLOAD = {
    "schema_version": 2,
    "gpu_model": "NVIDIA H100 80GB HBM3",
    "ssd_model": "Kioxia CM7-R",
    "model_name": MODEL,
    "kv_dtype": DTYPE,
    "kv_bytes_per_token": 131072,
    "curves": {
        "f": {"floor": 0.0, "knots": {"0": 0.0, "1024": 0.090, "4096": 0.34, "16384": 1.3}},
        "g_ssd": {
            "floor": 0.031,
            "knots": {"0": 0.031, "1024": 0.031, "4096": 0.050, "16384": 0.12},
        },
        "g_mem": {
            "floor": 0.003,
            "knots": {"0": 0.003, "1024": 0.003, "4096": 0.006, "16384": 0.02},
        },
    },
    "golden": [
        {"tokens": 700, "f": 0.0631, "g_ssd": 0.0309, "g_mem": 0.0028},
        {"tokens": 3000, "f": 0.2612, "g_ssd": 0.0431, "g_mem": 0.0047},
        {"tokens": 30000, "f": 2.3140, "g_ssd": 0.2011, "g_mem": 0.0332},
    ],
    "break_even_ssd_tokens": 6057,
    "break_even_mem_tokens": 736,
    "safety_margin_tokens": 0,
    "provenance": {
        "source_csv": "...",
        "generated": "2026-07-21T12:00:00Z",
        "ssd_bandwidth_gbps": 13.0,
        "cache_server_config": "...",
    },
}


def _v2_payload(**overrides: object) -> dict:
    payload = json.loads(json.dumps(V2_PAYLOAD))
    payload.update(overrides)
    return payload


def _write_file(payload: object) -> str:
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(payload, f)
    return path


class ShouldLoadTests(unittest.TestCase):
    def test_ssd_branch_gates_below_threshold(self) -> None:
        thr = BreakEvenThresholds(ssd_tokens=4096, mem_tokens=0)
        self.assertFalse(should_load(2048, thr, ram_resident=False))
        self.assertTrue(should_load(4096, thr, ram_resident=False))
        self.assertTrue(should_load(8192, thr, ram_resident=False))

    def test_ram_branch_uses_mem_threshold(self) -> None:
        # P*_mem ~ 0: a RAM-resident reuse always loads, even far below P*_ssd.
        thr = BreakEvenThresholds(ssd_tokens=4096, mem_tokens=0)
        self.assertTrue(should_load(128, thr, ram_resident=True))

    def test_ram_branch_with_nonzero_mem_threshold(self) -> None:
        thr = BreakEvenThresholds(ssd_tokens=4096, mem_tokens=256)
        self.assertFalse(should_load(128, thr, ram_resident=True))
        self.assertTrue(should_load(256, thr, ram_resident=True))

    def test_disabled_thresholds_always_load(self) -> None:
        thr = BreakEvenThresholds()
        self.assertFalse(thr.enabled)
        self.assertTrue(should_load(1, thr, ram_resident=False))
        self.assertTrue(should_load(1, thr, ram_resident=True))

    def test_zero_prefix_is_not_gated(self) -> None:
        thr = BreakEvenThresholds(ssd_tokens=4096)
        self.assertTrue(should_load(0, thr, ram_resident=False))

    def test_threshold_selects_medium(self) -> None:
        thr = BreakEvenThresholds(ssd_tokens=4096, mem_tokens=64)
        self.assertEqual(break_even_threshold(thr, ram_resident=False), 4096)
        self.assertEqual(break_even_threshold(thr, ram_resident=True), 64)


class LoadBreakEvenTests(unittest.TestCase):
    def test_absent_path_disabled(self) -> None:
        thr = load_break_even(None, model_name=MODEL, kv_dtype=DTYPE)
        self.assertEqual(thr, BreakEvenThresholds(0, 0))
        self.assertFalse(thr.enabled)

    def test_parses_thresholds_with_margin(self) -> None:
        path = _write_file(
            {
                "model_name": MODEL,
                "kv_dtype": DTYPE,
                "break_even_ssd_tokens": 4096,
                "break_even_mem_tokens": 0,
                "safety_margin_tokens": 128,
            }
        )
        try:
            thr = load_break_even(path, model_name=MODEL, kv_dtype=DTYPE)
        finally:
            os.unlink(path)
        # Margin must not turn an "off" (0) medium into an active gate.
        self.assertEqual(thr.ssd_tokens, 4096 + 128)
        self.assertEqual(thr.mem_tokens, 0)

    def test_model_mismatch_raises(self) -> None:
        path = _write_file(
            {"model_name": "other/Model", "break_even_ssd_tokens": 4096}
        )
        try:
            with self.assertRaises(ValueError):
                load_break_even(path, model_name=MODEL, kv_dtype=DTYPE)
        finally:
            os.unlink(path)

    def test_dtype_mismatch_warns_not_raises(self) -> None:
        path = _write_file(
            {"model_name": MODEL, "kv_dtype": "fp8", "break_even_ssd_tokens": 4096}
        )
        try:
            with self.assertLogs("py_kvcache.break_even", level="WARNING"):
                thr = load_break_even(path, model_name=MODEL, kv_dtype=DTYPE)
        finally:
            os.unlink(path)
        self.assertEqual(thr.ssd_tokens, 4096)

    def test_missing_configured_file_raises(self) -> None:
        with self.assertRaises(ValueError):
            load_break_even(
                "/nonexistent/break_even.json", model_name=MODEL, kv_dtype=DTYPE
            )

    def test_negative_threshold_raises(self) -> None:
        path = _write_file({"break_even_ssd_tokens": -1})
        try:
            with self.assertRaises(ValueError):
                load_break_even(path, model_name=MODEL, kv_dtype=DTYPE)
        finally:
            os.unlink(path)


class ConfigPathTests(unittest.TestCase):
    def test_break_even_path_defaults_none(self) -> None:
        config = SharedFileConfig.from_extra_config(
            {"shared_storage_path": "/mnt/shared-kv"}
        )
        self.assertIsNone(config.prefix_cache_break_even_path)

    def test_parses_break_even_path(self) -> None:
        config = SharedFileConfig.from_extra_config(
            {
                "shared_storage_path": "/mnt/shared-kv",
                "prefix_cache_break_even_path": "/etc/kv/break_even.json",
            }
        )
        self.assertEqual(
            config.prefix_cache_break_even_path, "/etc/kv/break_even.json"
        )


class LoadCurvesTests(unittest.TestCase):
    def test_v2_file_parses_into_curve_data(self) -> None:
        path = _write_file(_v2_payload())
        try:
            curves = load_curves(path, model_name=MODEL, kv_dtype=DTYPE)
        finally:
            os.unlink(path)
        self.assertIsInstance(curves, CurveData)
        self.assertEqual(
            curves.f.knots, ((0, 0.0), (1024, 0.090), (4096, 0.34), (16384, 1.3))
        )
        self.assertEqual(curves.f.floor, 0.0)
        self.assertEqual(len(curves.golden), 3)
        self.assertEqual(curves.golden[0].tokens, 700)
        self.assertEqual(curves.kv_bytes_per_token, 131072)

    def test_knots_are_sorted_even_if_out_of_order_in_file(self) -> None:
        payload = _v2_payload()
        payload["curves"]["f"]["knots"] = {
            "4096": 0.34,
            "0": 0.0,
            "16384": 1.3,
            "1024": 0.090,
        }
        path = _write_file(payload)
        try:
            curves = load_curves(path, model_name=MODEL, kv_dtype=DTYPE)
        finally:
            os.unlink(path)
        tokens = [t for t, _ in curves.f.knots]
        self.assertEqual(tokens, sorted(tokens))

    def test_v1_file_returns_none(self) -> None:
        path = _write_file(
            {
                "model_name": MODEL,
                "kv_dtype": DTYPE,
                "break_even_ssd_tokens": 4096,
                "break_even_mem_tokens": 0,
            }
        )
        try:
            self.assertIsNone(load_curves(path, model_name=MODEL, kv_dtype=DTYPE))
            # The same file's scalar view is unaffected by the v2 addition.
            thr = load_break_even(path, model_name=MODEL, kv_dtype=DTYPE)
        finally:
            os.unlink(path)
        self.assertEqual(thr, BreakEvenThresholds(ssd_tokens=4096, mem_tokens=0))

    def test_v2_file_load_break_even_still_returns_scalars(self) -> None:
        path = _write_file(_v2_payload())
        try:
            thr = load_break_even(path, model_name=MODEL, kv_dtype=DTYPE)
        finally:
            os.unlink(path)
        self.assertEqual(thr, BreakEvenThresholds(ssd_tokens=6057, mem_tokens=736))

    def test_absent_path_returns_none(self) -> None:
        self.assertIsNone(load_curves(None, model_name=MODEL, kv_dtype=DTYPE))

    def test_missing_curve_name_raises(self) -> None:
        payload = _v2_payload()
        del payload["curves"]["g_mem"]
        path = _write_file(payload)
        try:
            with self.assertRaises(ValueError):
                load_curves(path, model_name=MODEL, kv_dtype=DTYPE)
        finally:
            os.unlink(path)

    def test_extra_curve_name_raises(self) -> None:
        payload = _v2_payload()
        payload["curves"]["extra"] = {"floor": 0.0, "knots": {"0": 0.0}}
        path = _write_file(payload)
        try:
            with self.assertRaises(ValueError):
                load_curves(path, model_name=MODEL, kv_dtype=DTYPE)
        finally:
            os.unlink(path)

    def test_bad_knot_key_raises(self) -> None:
        payload = _v2_payload()
        payload["curves"]["f"]["knots"] = {"0": 0.0, "not-a-number": 0.5}
        path = _write_file(payload)
        try:
            with self.assertRaises(ValueError):
                load_curves(path, model_name=MODEL, kv_dtype=DTYPE)
        finally:
            os.unlink(path)

    def test_bad_knot_value_raises(self) -> None:
        payload = _v2_payload()
        payload["curves"]["f"]["knots"] = {"0": 0.0, "1024": "not-a-number"}
        path = _write_file(payload)
        try:
            with self.assertRaises(ValueError):
                load_curves(path, model_name=MODEL, kv_dtype=DTYPE)
        finally:
            os.unlink(path)

    def test_empty_knots_raises(self) -> None:
        payload = _v2_payload()
        payload["curves"]["f"]["knots"] = {}
        path = _write_file(payload)
        try:
            with self.assertRaises(ValueError):
                load_curves(path, model_name=MODEL, kv_dtype=DTYPE)
        finally:
            os.unlink(path)

    def test_negative_knot_key_raises(self) -> None:
        payload = _v2_payload()
        payload["curves"]["f"]["knots"] = {"-512": 0.5, "1024": 0.09, "4096": 0.34}
        path = _write_file(payload)
        try:
            with self.assertRaises(ValueError):
                load_curves(path, model_name=MODEL, kv_dtype=DTYPE)
        finally:
            os.unlink(path)

    def test_single_knot_at_zero_raises(self) -> None:
        payload = _v2_payload()
        payload["curves"]["f"]["knots"] = {"0": 0.0}
        path = _write_file(payload)
        try:
            with self.assertRaises(ValueError):
                load_curves(path, model_name=MODEL, kv_dtype=DTYPE)
        finally:
            os.unlink(path)

    def test_non_list_golden_raises(self) -> None:
        for bad_golden in ({}, 0, ""):
            payload = _v2_payload()
            payload["golden"] = bad_golden
            path = _write_file(payload)
            try:
                with self.assertRaises(ValueError):
                    load_curves(path, model_name=MODEL, kv_dtype=DTYPE)
            finally:
                os.unlink(path)

    def test_duplicate_knot_token_raises(self) -> None:
        payload = _v2_payload()
        # "1024" and "01024" both parse to the int token 1024.
        payload["curves"]["f"]["knots"] = {"0": 0.0, "1024": 0.09, "01024": 0.10}
        path = _write_file(payload)
        try:
            with self.assertRaises(ValueError):
                load_curves(path, model_name=MODEL, kv_dtype=DTYPE)
        finally:
            os.unlink(path)

    def test_malformed_golden_entry_raises(self) -> None:
        payload = _v2_payload()
        payload["golden"][0] = {"tokens": 700, "f": 0.06, "g_ssd": 0.03}  # missing g_mem
        path = _write_file(payload)
        try:
            with self.assertRaises(ValueError):
                load_curves(path, model_name=MODEL, kv_dtype=DTYPE)
        finally:
            os.unlink(path)

    def test_absent_golden_defaults_empty(self) -> None:
        payload = _v2_payload()
        del payload["golden"]
        path = _write_file(payload)
        try:
            curves = load_curves(path, model_name=MODEL, kv_dtype=DTYPE)
        finally:
            os.unlink(path)
        self.assertEqual(curves.golden, ())

    def test_model_mismatch_raises(self) -> None:
        payload = _v2_payload(model_name="other/Model")
        path = _write_file(payload)
        try:
            with self.assertRaises(ValueError):
                load_curves(path, model_name=MODEL, kv_dtype=DTYPE)
        finally:
            os.unlink(path)

    def test_dtype_mismatch_warns_not_raises(self) -> None:
        payload = _v2_payload(kv_dtype="fp8")
        path = _write_file(payload)
        try:
            with self.assertLogs("py_kvcache.break_even", level="WARNING"):
                curves = load_curves(path, model_name=MODEL, kv_dtype=DTYPE)
        finally:
            os.unlink(path)
        self.assertIsInstance(curves, CurveData)


if __name__ == "__main__":
    unittest.main()
