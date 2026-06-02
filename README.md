# py-kvcache

`py-kvcache` is a Python KV-cache offloading engine for vLLM, designed to be used with a storage interface supporting direct I/O with `io_uring`. It implements vLLM's `OffloadingConnector` and stores KV blocks as immutable, hash-addressed files on a filesystem shared by the participating workers.

**The project is meant for experimentation and profiling**.

Only one KV cache group is currently supported. i.e. the cache can serve only a single model at a time

## What it does

- Checks whether offloaded KV blocks already exist in shared storage.
- Stores missing KV blocks as files named by the vLLM block hash.
- Loads stored blocks back into GPU KV cache tensors.
- Uses pinned CPU staging tensors between GPU memory and file I/O.
- Runs storage transfers through one Python reactor thread with one `io_uring` ring.
- Optionally pre-opens/preloads block files after the scheduler prepares a load.

There is cache index, cleanup, garbage collection, or eviction policy.
Existing hash files are reused.

## Install

```bash
python3 -m pip install -e .
```

## Use with vLLM

Configure vLLM's `OffloadingConnector` to load the spec from `py_kvcache.vllm`:

```python
from vllm.config import KVTransferConfig

kv_transfer_config = KVTransferConfig(
    kv_connector="OffloadingConnector",
    kv_role="kv_both",
    kv_connector_extra_config={
        "spec_module_path": "py_kvcache.vllm",
        "spec_name": "PyKvCacheOffloadingSpec",
        "shared_storage_path": "/mnt/kv-cache",
        "iodepth": 16,
        "staging_mem": 4.0,
        "sync_on_store": False,
        "enable_preload": True,
        "open_lookahead": 16,
    },
)
```

`SharedStorageOffloadingSpec` is an alias for the same spec.

## Configuration

All options are passed through `kv_connector_extra_config`.

| Key | Required | Default | Meaning |
| --- | --- | --- | --- |
| `shared_storage_path` | yes | none | Root directory for stored KV block files. |
| `iodepth` | no | internal default | Max in-flight file reads and writes. |
| `staging_mem` | no | internal default | CPU staging memory budget in GiB. |
| `sync_on_store` | no | `false` | `fsync` temp files and parent dirs before publishing stored blocks. |
| `enable_preload` | no | `true` | Send preload messages when loads are prepared. |
| `open_lookahead` | no | `iodepth` | Max async opens plus opened-but-unread file descriptors. |

## Filesystem layout

Each storage block maps to one immutable file:

```text
<shared_storage_path>/<model_name>/
  block_size_<gpu_block_size>_blocks_per_file_<gpu_blocks_per_file>/
  tp_<tp>_pp_size_<pp>_pcp_size_<pcp>/
  rank_<rank>/<dtype>/<hhh>/<hh>/<full_hash>.bin
```

The `<hhh>/<hh>` directories are derived from the hash prefix to avoid putting
all files in one directory.

## Architecture

Scheduler side:

1. `PyKvCacheOffloadingSpec` builds a `FileMapper` for the current model,
   parallel rank, dtype, and block sizing.
2. `SharedStorageOffloadingManager.lookup()` checks whether a hash file exists.
3. `prepare_store()` skips hashes already present on disk.
4. `prepare_load()` returns the hashes to load and may emit preload messages.

Worker side:

1. `NoopSharedStorageOffloadingHandler` receives vLLM transfer requests.
2. `TransferCoordinator` splits GPU block ids into per-file reactor jobs.
3. `IoReactor` owns the `io_uring` ring, staging slots, CUDA streams, and active
   job table.
4. Stores flow as:

```text
GPU KV cache -> vLLM swap_blocks -> CPU staging -> direct-I/O write
  -> temp file -> hard-link final hash file
```

5. Loads flow as:

```text
hash file -> async open/read -> CPU staging -> vLLM swap_blocks -> GPU KV cache
```

Reads and writes are bounded by `iodepth`. Async opens are bounded separately by
`open_lookahead`, so metadata latency can overlap active file I/O without tying
up staging slots.
