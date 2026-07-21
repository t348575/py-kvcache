import unittest

from py_kvcache.file_mapper import FileMapper
from py_kvcache.fs_config import SharedFileConfig


class SharedFileConfigTests(unittest.TestCase):
    def test_requires_shared_storage_path(self) -> None:
        with self.assertRaises(ValueError):
            SharedFileConfig.from_extra_config({})

    def test_parses_minimal_config(self) -> None:
        config = SharedFileConfig.from_extra_config(
            {"shared_storage_path": "/mnt/shared-kv"}
        )
        self.assertEqual(config.root_dir, "/mnt/shared-kv")
        self.assertFalse(config.sync_on_store)
        self.assertEqual(config.iodepth, 16)
        self.assertAlmostEqual(config.staging_mem, 1.0)
        self.assertFalse(config.enable_preload)
        self.assertIsNone(config.open_lookahead)

    def test_parses_sync_on_store(self) -> None:
        config = SharedFileConfig.from_extra_config(
            {"shared_storage_path": "/mnt/shared-kv", "sync_on_store": "true"}
        )
        self.assertTrue(config.sync_on_store)

    def test_parses_iodepth(self) -> None:
        config = SharedFileConfig.from_extra_config(
            {"shared_storage_path": "/mnt/shared-kv", "iodepth": "16"}
        )
        self.assertEqual(config.iodepth, 16)

    def test_rejects_non_positive_iodepth(self) -> None:
        with self.assertRaises(ValueError):
            SharedFileConfig.from_extra_config(
                {"shared_storage_path": "/mnt/shared-kv", "iodepth": "0"}
            )

    def test_parses_enable_preload(self) -> None:
        config = SharedFileConfig.from_extra_config(
            {"shared_storage_path": "/mnt/shared-kv", "enable_preload": "false"}
        )
        self.assertFalse(config.enable_preload)

    def test_parses_open_lookahead(self) -> None:
        config = SharedFileConfig.from_extra_config(
            {"shared_storage_path": "/mnt/shared-kv", "open_lookahead": 32}
        )
        self.assertEqual(config.open_lookahead, 32)

    def test_parses_staging_mem(self) -> None:
        config = SharedFileConfig.from_extra_config(
            {"shared_storage_path": "/mnt/shared-kv", "staging_mem": "4.0"}
        )
        self.assertAlmostEqual(config.staging_mem, 4.0)

    def test_staging_cache_defaults_off(self) -> None:
        config = SharedFileConfig.from_extra_config({"shared_storage_path": "/mnt/shared-kv"})
        self.assertEqual(config.staging_cache, "off")

    def test_parses_staging_cache(self) -> None:
        for value, expected in (("LRU", "lru"), ("arc", "arc"), ("off", "off")):
            config = SharedFileConfig.from_extra_config(
                {"shared_storage_path": "/mnt/shared-kv", "staging_cache": value}
            )
            self.assertEqual(config.staging_cache, expected)

    def test_rejects_unknown_staging_cache(self) -> None:
        with self.assertRaises(ValueError):
            SharedFileConfig.from_extra_config(
                {"shared_storage_path": "/mnt/shared-kv", "staging_cache": "fifo"}
            )


class FileMapperTests(unittest.TestCase):
    def test_rejects_absolute_model_name(self) -> None:
        with self.assertRaisesRegex(ValueError, "model_name must be relative"):
            FileMapper(
                root_dir="/cache",
                model_name="/models/llama",
                gpu_block_size=16,
                gpu_blocks_per_file=1,
                tp_size=1,
                pp_size=1,
                pcp_size=1,
                rank=0,
                dtype="float16",
            )

    def test_rejects_model_name_that_escapes_root(self) -> None:
        with self.assertRaisesRegex(ValueError, "model_name must not escape"):
            FileMapper(
                root_dir="/cache",
                model_name="../models/llama",
                gpu_block_size=16,
                gpu_blocks_per_file=1,
                tp_size=1,
                pp_size=1,
                pcp_size=1,
                rank=0,
                dtype="float16",
            )

    def test_uses_llmd_style_directory_layout(self) -> None:
        mapper = FileMapper(
            root_dir="/mnt/files-storage/kv-cache/",
            model_name="meta-llama/Llama-3.1-8B-Instruct",
            gpu_block_size=16,
            gpu_blocks_per_file=4,
            tp_size=2,
            pp_size=1,
            pcp_size=1,
            rank=0,
            dtype="bfloat16",
        )

        file_name = mapper.get_file_name(bytes.fromhex("0123456789abcdef" * 4))

        self.assertEqual(
            file_name,
            "/mnt/files-storage/kv-cache"
            "/meta-llama/Llama-3.1-8B-Instruct"
            "/block_size_16_blocks_per_file_4"
            "/tp_2_pp_size_1_pcp_size_1"
            "/rank_0"
            "/bfloat16"
            "/012/34"
            "/0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef.bin",
        )

    def test_supports_integer_hashes(self) -> None:
        mapper = FileMapper(
            root_dir="/cache",
            model_name="model",
            gpu_block_size=16,
            gpu_blocks_per_file=1,
            tp_size=1,
            pp_size=1,
            pcp_size=1,
            rank=3,
            dtype="float16",
        )

        file_name = mapper.get_file_name(0x1234ABCD)

        self.assertTrue(file_name.endswith("/000/00/000000001234abcd.bin"))


if __name__ == "__main__":
    unittest.main()
