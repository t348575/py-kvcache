from __future__ import annotations

import unittest
from dataclasses import dataclass
from unittest.mock import patch

import py_kvcache.vllm as vmod


@dataclass
class _CurrentTransferResult:
    job_id: int
    success: bool
    transfer_size: int | None = None
    transfer_time: float | None = None
    transfer_type: tuple[str, str] | None = None


class TransferResultCompatibilityTests(unittest.TestCase):
    def test_current_vllm_result_preserves_transfer_stats(self) -> None:
        with patch.object(vmod, "TransferResult", _CurrentTransferResult):
            result = vmod._make_transfer_result(
                job_id=7,
                success=True,
                transfer_size=4096,
                transfer_time=0.25,
                transfer_type=("GPU", "SHARED_STORAGE"),
            )

        self.assertEqual(result.transfer_size, 4096)
        self.assertEqual(result.transfer_time, 0.25)
        self.assertEqual(result.transfer_type, ("GPU", "SHARED_STORAGE"))

    def test_failure_message_does_not_discard_transfer_stats(self) -> None:
        with patch.object(vmod, "TransferResult", _CurrentTransferResult):
            result = vmod._make_transfer_result(
                job_id=8,
                success=False,
                transfer_size=0,
                transfer_time=0.5,
                transfer_type=("SHARED_STORAGE", "GPU"),
                message="boom",
            )

        self.assertFalse(result.success)
        self.assertEqual(result.transfer_time, 0.5)
        self.assertEqual(result.transfer_type, ("SHARED_STORAGE", "GPU"))


if __name__ == "__main__":
    unittest.main()
