<h1 align="center">py-kvcache</h1>
<p align="center">
  <img alt="Views" src="https://lambda.348575.xyz/repo-view-counter?repo=py-kvcache"/>
</p>

<p align="center">
  py-kvcache is a python KV cache offloading engine for vLLM, designed to be used with a storage interface supporting direct I/O with iouring. It implements vLLM's OffloadingConnector and stores KV blocks as immutable, hash-addressed files on a filesystem shared by the participating workers.
</p>


**Important notes:**
* The cache can serve only a single model at a time.
* Requires various changes in vllm kv offload API as well as the scheduler, available in my vllm fork [t348575/vllm](https://github.com/t348575/vllm).
* Requires the profiler [t348575/simple-profiler](https://github.com/t348575/simple-profiler/).

## What it does & Features

- Checks whether offloaded KV blocks already exist in shared storage.
- Stores missing KV blocks as files named by the vLLM block hash.
- Loads stored blocks back into GPU KV cache tensors.
- Uses pinned CPU staging tensors between GPU memory and file I/O.
- Runs storage transfers through one Python reactor thread with one `io_uring` ring.
- Can preload KV for upcoming requests. The scheduler looks ahead at the next waiting requests and the reactor pre-reads their blocks into CPU staging during background stores or idle time.
- Support for running with a DRAM cache with LRU or ARC (write-through).
- Supports gating kvcache load & store ops using a calculated break-even curve. When a prefix is too short to be worth it, a disk or DRAM read is declined and recomputed on the GPU instead. The break-even curve can be calculated using the `pareto_measure` script in [t348575/kvcache-experiments](https://github.com/t348575/kvcache-experiments#scriptspareto_measurepy).

There is no cache index, cleanup, garbage collection, or eviction policy for the files.

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
        "preload_lookahead_requests": 16,
        "preload_share_staging": True,
        "open_lookahead": 16,
        "staging_cache": "off",
    },
)
```

`SharedStorageOffloadingSpec` is an alias for the same spec.

## Configuration

All options are passed through `kv_connector_extra_config`.

| Key | Required | Default | Meaning |
| --- | --- | --- | --- |
| `shared_storage_path` | yes | none | Root directory for stored KV block files. |
| `iodepth` | no | 16 | Max in-flight file reads and writes. |
| `staging_mem` | no | 1GB | CPU staging memory budget in GiB. |
| `sync_on_store` | no | `false` | `fsync` temp files and parent dirs before publishing stored blocks. Useful when using shared storage. |
| `enable_preload` | no | `false` | Kv offload connector, scheduler, and vLLM scheduler work to preemptively load entries from disk to CPU DRAM before they actually need to run. |
| `preload_lookahead_requests` | no | 0 | Number of upcoming requests the scheduler will try to preload at each step. Ensure this is > 0 when setting `enable_preload`. Ensure `staging_mem` is large enough to preload many requests. |
| `preload_share_staging` | no | `true` | Share one disk read and one staging slot across multiple preload candidates. This is intended for experiments, leave on. |
| `open_lookahead` | no | `iodepth` | Used to reduce latency from open to read, by opening files before they are actually read. |
| `staging_cache` | no | `off` | Write-through DRAM cache: `off`, `lru`, or `arc`. Functions as a regular cache for load operations, in a write-through manner. Evictions are removed, not flushed to disk. |
| `prefix_cache_break_even_path` | no | none | Path to a JSON file with break-even data (setup specific). Ops below the threshold are denied, and are computed instead. |

## Filesystem layout

Each storage block maps to one immutable file:

```text
<shared_storage_path>/<model_name>/block_size_<gpu_block_size>_blocks_per_file_<gpu_blocks_per_file>/tp_<tp>_pp_size_<pp>_pcp_size_<pcp>/rank_<rank>/<dtype>/<hhh>/<hh>/<full_hash>.bin
```

The `<hhh>/<hh>` directories are derived from the hash prefix to avoid putting all files in one directory.

## Architecture

Scheduler side:

1. `PyKvCacheOffloadingSpec` builds a `FileMapper` for the current model, parallel rank, dtype, and block sizing.
2. `SharedStorageOffloadingManager.lookup()` returns `True` if the hash file exists, `False` if absent, and `None` (defer) while the block is being written, so the scheduler waits for the store instead of recomputing a reusable prefix from scratch.
3. `prepare_store()` skips hashes already present on disk and marks the rest in flight.
4. `prepare_load()` returns the hashes to load. Preload is driven entirely by the scheduler lookahead below, not by `prepare_load`.

Worker side:

1. `NoopSharedStorageOffloadingHandler` receives vLLM transfer requests.
2. `TransferCoordinator` splits GPU block ids into per-file reactor jobs.
3. `IoReactor` owns the `io_uring` ring, staging slots, CUDA streams, and active job table.
4. Store flow:

```text
GPU KV cache -> vLLM swap_blocks -> CPU staging -> direct-I/O write -> temp file -> hard-link final hash file
```

1. Load flow:

```text
hash file -> async open/read -> CPU staging -> vLLM swap_blocks -> GPU KV cache
```

Reads and writes are bounded by `iodepth`. Async opens are bounded separately by `open_lookahead`, so metadata latency can overlap active file I/O without tying up staging slots.

## vLLM changes

The stock vLLM `OffloadingConnector` only stores and loads KV on demand. Preload and break-even gating need the scheduler and connector to do things the upstream API does not expose, they are added to the vLLM fork [t348575/vllm](https://github.com/t348575/vllm). py-kvcache only performs the I/O and cache work for these features.

### Preload lookahead

The goal is to let the storage backend start reading a waiting request's KV *before* that request is scheduled, so its load is already in CPU memory when it runs, and will only require a much faster CPU->GPU copy.

1. At each scheduling step the scheduler sends the next N requests to the offloading connector (`_notify_preload_candidates` / `_get_preload_candidate_requests` in the v1 scheduler).
2. The offloading connector then calculates if the request has a stored prefix (and how much), and forwards it to the vLLM worker as a preload hint using the `reqs_to_preload` field.
3. The vLLM worker then forwards the hint to the external kvcache calling `preload_async(preload_id, ...)` to start the speculative read. When the actual load for the request happens i.e. when the scheduler starts executing it, the worker calls `load_from_preload_async(...)` passing `preload_id`.

The connector does not check whether the blocks exist on disk (or cache) before starting a preload, this is up to py-kvcache to check. py-kvcache only performs a preload if no regular load is ongoing, i.e. it is idle or performing store ops. It also does not perform preloads if enough staging memory is not available.

### Break-even gating

Through testing, a clear break-even point exists for kv caching, when the cost of loading is lower than re-computing the prefix. This break-even point is setup specific (GPU, LLM model, SSD). The vLLM scheduler uses the break-even data to determine when a load from either CPU DRAM or disk is worth it. If the prefix size is less than the break-even, then the load & store ops are declined.

These changes are added to the vLLM fork [t348575/vllm](https://github.com/t348575/vllm).

`scripts/pareto_measure.py` in [t348575/kvcache-experiments](https://github.com/t348575/kvcache-experiments) can be used to generate a pareto plot, and the break-even point for your setup.

`scripts/emit_break_even.py` can be used to generate the json break even data for vLLM to use.