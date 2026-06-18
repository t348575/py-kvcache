"""End-to-end KV cache store + reload test.

Starts a real vLLM server, sends an ~80k-token prompt twice with greedy
decoding, and verifies that the completion is identical on both passes.
Pass 1 computes KV from scratch and writes blocks to storage. Pass 2 finds the
blocks on storage via ``manager.lookup`` and loads them; if the loaded bytes are
corrupt the attention output differs and the assertion fails.

Requirements
------------
- CUDA GPU
- vLLM installed (``pip install vllm``)
- Hugging Face model accessible (set ``HUGGING_FACE_HUB_TOKEN`` if needed)

Enable
------
    RUN_E2E_TESTS=1 pytest tests/test_e2e_kvcache.py -v -s

Overrides
---------
    E2E_MODEL   Hugging Face model id  (default: meta-llama/Llama-3.2-3B-Instruct)
    E2E_PORT    vLLM listen port       (default: 8051)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request

_REQUIRES_E2E = unittest.skipUnless(
    os.environ.get("RUN_E2E_TESTS"),
    "set RUN_E2E_TESTS=1 to run (requires GPU + vLLM)",
)

_MODEL = os.environ.get("E2E_MODEL", "meta-llama/Llama-3.2-3B-Instruct")
_PORT = int(os.environ.get("E2E_PORT", "8051"))
_BASE = f"http://localhost:{_PORT}"
_STARTUP_TIMEOUT = 300  # seconds

# ~80 k tokens for LLaMA-family tokenizers (~10 tokens per sentence, 8000 reps).
# Stays well under --max-model-len 92000 even with 32 completion tokens.
_LONG_PROMPT = "The quick brown fox jumps over the lazy dog. " * 8_000


def _wait_healthy(timeout: float, base: str = _BASE) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{base}/health", timeout=5) as r:
                if r.status == 200:
                    return
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(3)
    raise TimeoutError(f"vLLM at {base} not ready after {timeout:.0f}s")


def _complete(prompt: str, *, max_tokens: int = 32, base: str = _BASE) -> str:
    body = json.dumps(
        {
            "model": _MODEL,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
        }
    ).encode()
    req = urllib.request.Request(
        f"{base}/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())["choices"][0]["text"]


@_REQUIRES_E2E
class KvCacheE2eTest(unittest.TestCase):
    _proc: subprocess.Popen | None = None
    _tmpdir: tempfile.TemporaryDirectory | None = None
    _pass1_out: str = ""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmpdir = tempfile.TemporaryDirectory(prefix="py_kvcache_e2e_")
        storage = cls._tmpdir.name

        kv_cfg = {
            "kv_connector": "OffloadingConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
                "spec_name": "PyKvCacheOffloadingSpec",
                "spec_module_path": "py_kvcache.vllm",
                "shared_storage_path": storage,
                "block_size": 256,
                "sync_on_store": False,
                "staging_mem": 32,
                "iodepth": 16,
                "enable_preload": True,
                "preload_lookahead_requests": 10,
            },
        }
        cmd = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            _MODEL,
            "--port",
            str(_PORT),
            "--no-enable-prefix-caching",
            "--max-model-len",
            "92000",
            "--gpu-memory-utilization",
            "0.9",
            "--kv-transfer-config",
            json.dumps(kv_cfg),
        ]
        cls._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            _wait_healthy(_STARTUP_TIMEOUT)
        except Exception:
            cls._shutdown_server()
            raise

        # Pass 1: cold path; KV computed from scratch, blocks stored to disk.
        cls._pass1_out = _complete(_LONG_PROMPT)
        # Let the io_uring ring flush all queued writes before tests inspect files.
        time.sleep(3)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._shutdown_server()
        if cls._tmpdir is not None:
            cls._tmpdir.cleanup()
            cls._tmpdir = None

    @classmethod
    def _shutdown_server(cls) -> None:
        if cls._proc is not None:
            cls._proc.terminate()
            try:
                cls._proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                cls._proc.kill()
                cls._proc.wait()
            cls._proc = None

    def _bin_files(self) -> list[str]:
        assert self._tmpdir is not None
        return [
            os.path.join(dirpath, fname)
            for dirpath, _, fnames in os.walk(self._tmpdir.name)
            for fname in fnames
            if fname.endswith(".bin")
        ]

    def test_kv_files_written(self) -> None:
        """Pass 1 must have written at least one .bin file to storage."""
        files = self._bin_files()
        self.assertGreater(
            len(files),
            0,
            "no .bin files found under storage path — KV store did not run",
        )

    def test_kv_files_uniform_aligned_size(self) -> None:
        """All .bin files must be the same size and 4096-byte aligned (direct I/O)."""
        files = self._bin_files()
        if not files:
            self.skipTest("no .bin files — test_kv_files_written should have caught this")
        sizes = {os.path.getsize(f) for f in files}
        self.assertEqual(
            len(sizes),
            1,
            f"non-uniform .bin file sizes: {sorted(sizes)}",
        )
        (size,) = sizes
        self.assertEqual(
            size % 4096,
            0,
            f".bin file size {size} is not 4096-aligned (direct I/O requires alignment)",
        )

    def test_reload_produces_identical_completion(self) -> None:
        """Pass 2 must produce a bit-identical completion to pass 1.

        With --no-enable-prefix-caching the GPU KV pool is free after pass 1.
        The scheduler's manager.lookup finds the hash files on disk and routes
        pass 2 through the load path.  If the loaded bytes are corrupt, the
        attention scores differ and the greedy completion diverges.
        """
        pass2_out = _complete(_LONG_PROMPT)
        self.assertEqual(
            self._pass1_out,
            pass2_out,
            (
                "completions differ after KV reload — loaded bytes may be corrupt\n"
                f"  pass1={self._pass1_out!r}\n"
                f"  pass2={pass2_out!r}"
            ),
        )


@_REQUIRES_E2E
class BreakEvenDeclineE2eTest(unittest.TestCase):
    """Force every load below break-even so the worker declines it.

    A break-even file with both thresholds set far above --max-model-len makes
    the reactor decline every load (RAM and SSD branches alike). Pass 2 must then
    recompute the whole prefix via invalid_block_ids and still produce a
    bit-identical completion, proving the decline -> recompute path is correct
    (no corruption, no hang) end to end. Stores still happen, so .bin files are
    written on pass 1 just as without gating.
    """

    _proc: subprocess.Popen | None = None
    _tmpdir: tempfile.TemporaryDirectory | None = None
    _pass1_out: str = ""
    _port: int = _PORT + 1

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmpdir = tempfile.TemporaryDirectory(prefix="py_kvcache_be_e2e_")
        storage = cls._tmpdir.name
        break_even_path = os.path.join(storage, "break_even.json")
        with open(break_even_path, "w") as f:
            json.dump(
                {
                    "schema_version": 1,
                    "model_name": _MODEL,
                    # Both thresholds far above max-model-len => decline every load.
                    "break_even_ssd_tokens": 10_000_000,
                    "break_even_mem_tokens": 10_000_000,
                },
                f,
            )

        # No kv_load_failure_policy here on purpose: OffloadingConnector forces
        # "recompute" itself, so declined loads recompute without extra config.
        kv_cfg = {
            "kv_connector": "OffloadingConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
                "spec_name": "PyKvCacheOffloadingSpec",
                "spec_module_path": "py_kvcache.vllm",
                "shared_storage_path": storage,
                "block_size": 256,
                "sync_on_store": False,
                "staging_mem": 32,
                "iodepth": 16,
                "enable_preload": True,
                "preload_lookahead_requests": 10,
                "prefix_cache_break_even_path": break_even_path,
            },
        }
        base = f"http://localhost:{cls._port}"
        cmd = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            _MODEL,
            "--port",
            str(cls._port),
            "--no-enable-prefix-caching",
            "--max-model-len",
            "92000",
            "--gpu-memory-utilization",
            "0.9",
            "--kv-transfer-config",
            json.dumps(kv_cfg),
        ]
        cls._proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        try:
            _wait_healthy(_STARTUP_TIMEOUT, base=base)
        except Exception:
            cls._shutdown_server()
            raise
        cls._pass1_out = _complete(_LONG_PROMPT, base=base)
        time.sleep(3)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._shutdown_server()
        if cls._tmpdir is not None:
            cls._tmpdir.cleanup()
            cls._tmpdir = None

    @classmethod
    def _shutdown_server(cls) -> None:
        if cls._proc is not None:
            cls._proc.terminate()
            try:
                cls._proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                cls._proc.kill()
                cls._proc.wait()
            cls._proc = None

    def _bin_files(self) -> list[str]:
        assert self._tmpdir is not None
        return [
            os.path.join(dirpath, fname)
            for dirpath, _, fnames in os.walk(self._tmpdir.name)
            for fname in fnames
            if fname.endswith(".bin")
        ]

    def test_stores_still_happen(self) -> None:
        """Break-even gates loads only; pass 1 must still write .bin files."""
        self.assertGreater(
            len(self._bin_files()),
            0,
            "no .bin files — stores must still run under load-only break-even gating",
        )

    def test_declined_load_recomputes_identically(self) -> None:
        """Pass 2 reuse is declined -> recomputed; output must match pass 1."""
        base = f"http://localhost:{self._port}"
        pass2_out = _complete(_LONG_PROMPT, base=base)
        self.assertEqual(
            self._pass1_out,
            pass2_out,
            (
                "completions differ after break-even decline+recompute\n"
                f"  pass1={self._pass1_out!r}\n"
                f"  pass2={pass2_out!r}"
            ),
        )


if __name__ == "__main__":
    unittest.main()
