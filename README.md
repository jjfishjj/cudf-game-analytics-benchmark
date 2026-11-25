# cuDF Game Analytics Benchmark

A reproducible comparison of common game-analytics operations on pandas (CPU) and cuDF (NVIDIA GPU). The benchmark measures sorting, multi-aggregation groupby, and a left hash join with deterministic inputs, explicit warm-ups, repeated samples, CUDA synchronization, and JSON output.

## Reproducible environment

The benchmark pilot uses CPython 3.12, NumPy 2.3.4, pandas 3.0.0, and cuDF 26.10.2 for CUDA 12. The complete resolved Python dependency graph and wheel hashes are recorded in [`uv.lock`](uv.lock); the shared benchmark contract is documented in [`BENCHMARK_STANDARD.md`](BENCHMARK_STANDARD.md), and output is validated against [`benchmark-results.schema.json`](benchmark-results.schema.json).

The GPU path requires a Linux x86_64 host with an NVIDIA GPU of compute capability 7.0 or higher and a compatible CUDA 12 driver/toolkit. NVIDIA's current installation guide lists CUDA 12 with driver 525.60.13 or newer as a general minimum; check the guide for the exact system requirements for your platform. See [NVIDIA's installation guide](https://docs.rapids.ai/install/) and [cuDF 26.10.2 package metadata](https://pypi.org/project/cudf-cu12/26.10.2/).

## Run the benchmark

Install the CPU baseline environment, or the same environment with cuDF for GPU measurements:

```bash
uv sync --extra dev                 # pandas CPU baseline
uv sync --extra gpu --extra dev     # cuDF GPU run
```

Run CPU and GPU in separate fresh processes using identical arguments and seed. The default workload has 5 million fact rows, a 1 million-row dimension table, one warm-up, and five recorded samples:

```bash
uv run python benchmarks/run_benchmark.py --backend cpu --output results/cpu.json
uv run --extra gpu python benchmarks/run_benchmark.py --backend gpu --output results/gpu.json
```

For a quick CPU smoke test, reduce the workload and repetitions:

```bash
uv run python benchmarks/run_benchmark.py --backend cpu --rows 20000 --dimension-rows 1000 --warmups 0 --repetitions 1 --output /tmp/cudf-cpu-smoke.json
```

The runner reports data construction/setup separately from operation timings. The cuDF path synchronizes CUDA around each sample. The result JSON includes raw timing samples, summary statistics, input parameters, Git state, installed software versions, CUDA details, and GPU metadata.

## Colab notebook

Open [`notebooks/game_analytics_benchmark.ipynb`](notebooks/game_analytics_benchmark.ipynb) in Colab with a compatible GPU. It defaults to `main` after this change is merged; while reviewing an unmerged PR, set `BENCHMARK_REF` in the first cell to the PR branch. The exact checked-out commit is recorded in each JSON result.

## Results status

This repository previously contained illustrative Colab timing values and blank 50-million-row fields, but did not preserve raw samples or a pinned runtime. Those figures are not presented as a reproducible benchmark here. The standardized runner has been added and can be smoke-tested on CPU; this execution environment has no NVIDIA GPU or CUDA tools, so no GPU result is claimed. Run the GPU command on a compatible NVIDIA machine to produce a real, schema-conforming cuDF result.

The original RFM, retention, and funnel exploration is preserved in [`notebooks/game_analytics_legacy.ipynb`](notebooks/game_analytics_legacy.ipynb) as historical, unpinned work; its hard-coded figures are not reproducible results.

## Shared format and validation

Each benchmark repository will carry the same `benchmark-results.schema.json` and use the same environment/result rules. Validate results with the project tests:

```bash
uv run --extra dev pytest
```
