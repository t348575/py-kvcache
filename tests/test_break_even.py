import json
import os
import tempfile
import unittest

from py_kvcache.break_even import (
    BreakEvenThresholds,
    break_even_threshold,
    load_break_even,
    should_load,
)
from py_kvcache.fs_config import SharedFileConfig

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DTYPE = "auto"


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
        # Margin folds into a live threshold but must not turn an "off" (0)
        # medium into an active gate.
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


if __name__ == "__main__":
    unittest.main()
