from __future__ import annotations


class FileMapper:
    """Maps block hashes to shared filesystem paths."""

    def __init__(
        self,
        *,
        root_dir: str,
        model_name: str,
        gpu_block_size: int,
        gpu_blocks_per_file: int,
        tp_size: int,
        pp_size: int,
        pcp_size: int,
        rank: int,
        dtype: str,
    ):
        self.base_path = (
            f"{root_dir}"
            f"/{model_name}"
            f"/block_size_{gpu_block_size}_blocks_per_file_{gpu_blocks_per_file}"
            f"/tp_{tp_size}_pp_size_{pp_size}_pcp_size_{pcp_size}"
            f"/rank_{rank}"
            f"/{dtype}"
        )

    def get_file_name(self, block_hash: bytes | int) -> str:
        block_hash_hex = self._to_hex(block_hash)
        subfolder1, subfolder2 = block_hash_hex[:3], block_hash_hex[3:5]
        return f"{self.base_path}/{subfolder1}/{subfolder2}/{block_hash_hex}.bin"

    @staticmethod
    def _to_hex(block_hash: bytes | int) -> str:
        if isinstance(block_hash, bytes):
            return block_hash.hex()
        if isinstance(block_hash, int):
            return f"{block_hash & ((1 << 64) - 1):016x}"
        raise TypeError(f"unsupported block hash type: {type(block_hash)!r}")


__all__ = ["FileMapper"]
