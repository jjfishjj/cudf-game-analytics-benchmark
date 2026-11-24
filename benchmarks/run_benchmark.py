#!/usr/bin/env python3
"""Deterministic CPU/cuDF benchmark using the shared result schema."""
from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

BENCHMARK_ID = "cudf.game-analytics.core-ops.v1"
SOFTWARE_DISTRIBUTIONS = (
    "numpy",
    "pandas",
    "cudf-cu12",
    "cupy-cuda12x",
    "cuda-bindings",
    "pyarrow",
)


def distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def git_metadata() -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True
            ).strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": True}


def cuda_version_string(value: int | None) -> str | None:
    if value is None or value <= 0:
        return None
    # CUDA runtime/driver API encodes 12.9 as 12090.
    return f"{value // 1000}.{(value % 1000) // 10}"


def gpu_metadata(cupy: Any | None) -> tuple[list[dict[str, Any]], str | None, str | None]:
    if cupy is None:
        return [], None, None
    devices: list[dict[str, Any]] = []
    count = int(cupy.cuda.runtime.getDeviceCount())
    for index in range(count):
        props = cupy.cuda.runtime.getDeviceProperties(index)
        name = props.get("name") if isinstance(props, dict) else None
        if isinstance(name, bytes):
            name = name.decode("utf-8", errors="replace")
        major = props.get("major") if isinstance(props, dict) else None
        minor = props.get("minor") if isinstance(props, dict) else None
        memory = props.get("totalGlobalMem") if isinstance(props, dict) else None
        devices.append(
            {
                "index": index,
                "name": name,
                "compute_capability": f"{major}.{minor}" if major is not None and minor is not None else None,
                "memory_bytes": int(memory) if memory is not None else None,
            }
        )
    runtime = cuda_version_string(int(cupy.cuda.runtime.runtimeGetVersion()))
    driver = cuda_version_string(int(cupy.cuda.runtime.driverGetVersion()))
    return devices, runtime, driver


def load_backend(name: str) -> tuple[Any, Any | None, str]:
    if name == "cpu":
        return pd, None, "pandas-cpu"
    try:
        import cupy
        import cudf
    except ImportError as exc:
        raise SystemExit(
            "GPU backend needs cudf-cu12 and cupy-cuda12x. Install with `uv sync --extra gpu --extra dev`."
        ) from exc
    try:
        count = int(cupy.cuda.runtime.getDeviceCount())
    except Exception as exc:
        raise SystemExit(f"GPU backend could not initialize CUDA: {exc}") from exc
    if count < 1:
        raise SystemExit("GPU backend requested but no CUDA-visible NVIDIA GPU was found.")
    return cudf, cupy, "cudf-gpu"


def make_frames(rows: int, dimension_rows: int, seed: int, backend: Any) -> tuple[Any, Any, float]:
    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    keys = rng.integers(0, dimension_rows, size=rows, dtype=np.int32)
    values = rng.random(rows, dtype=np.float32)
    quantities = rng.integers(1, 100, size=rows, dtype=np.int32)
    tiers = rng.integers(0, 5, size=dimension_rows, dtype=np.int32)
    facts_host = pd.DataFrame({"key": keys, "val": values, "qty": quantities})
    dim_host = pd.DataFrame({"key": np.arange(dimension_rows, dtype=np.int32), "tier": tiers})
    if backend is pd:
        facts, dimension = facts_host, dim_host
    else:
        facts = backend.from_pandas(facts_host)
        dimension = backend.from_pandas(dim_host)
    return facts, dimension, time.perf_counter() - started


def summarize(samples: list[float]) -> dict[str, float | None]:
    if not samples:
        return {"median": None, "mean": None, "stdev": None, "min": None, "max": None}
    return {
        "median": statistics.median(samples),
        "mean": statistics.mean(samples),
        "stdev": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "min": min(samples),
        "max": max(samples),
    }


def run_case(
    case_id: str,
    operation: Callable[[], Any],
    *,
    input_rows: int,
    warmups: int,
    repetitions: int,
    cupy: Any | None,
) -> dict[str, Any]:
    def synchronize() -> None:
        if cupy is not None:
            cupy.cuda.runtime.deviceSynchronize()

    output_rows: int | None = None
    for _ in range(warmups):
        result = operation()
        synchronize()
        output_rows = len(result)
        del result
        gc.collect()

    samples: list[float] = []
    for _ in range(repetitions):
        synchronize()
        started = time.perf_counter()
        result = operation()
        synchronize()
        samples.append(time.perf_counter() - started)
        output_rows = len(result)
        del result
        gc.collect()

    return {
        "case_id": case_id,
        "backend": "cudf-gpu" if cupy is not None else "pandas-cpu",
        "status": "completed",
        "unit": "seconds",
        "input_rows": input_rows,
        "output_rows": output_rows,
        "warmup_iterations": warmups,
        "samples": samples,
        "statistics": summarize(samples),
        "notes": "Input generation and host-to-device setup are excluded from operation timings; CUDA is synchronized around each GPU sample.",
    }


def build_result(args: argparse.Namespace) -> dict[str, Any]:
    backend, cupy, backend_name = load_backend(args.backend)
    facts, dimension, setup_seconds = make_frames(args.rows, args.dimension_rows, args.seed, backend)
    if backend_name == "cudf-gpu":
        devices, cuda_runtime, cuda_driver = gpu_metadata(cupy)
    else:
        devices, cuda_runtime, cuda_driver = [], None, None

    cases: list[tuple[str, Callable[[], Any]]] = [
        ("sort-values", lambda: facts.sort_values("val")),
        (
            "groupby-multi-aggregation",
            lambda: facts.groupby("key").agg(
                total=("val", "sum"),
                average=("val", "mean"),
                count=("qty", "count"),
                maximum=("val", "max"),
                minimum=("val", "min"),
            ),
        ),
        ("left-hash-join", lambda: facts.merge(dimension, on="key", how="left", sort=False)),
    ]
    measurements = [
        run_case(
            case_id,
            operation,
            input_rows=args.rows,
            warmups=args.warmups,
            repetitions=args.repetitions,
            cupy=cupy,
        )
        for case_id, operation in cases
    ]
    for measurement in measurements:
        if measurement["case_id"] in {"sort-values", "left-hash-join"} and measurement["output_rows"] != args.rows:
            raise RuntimeError(f"{measurement['case_id']} returned an unexpected row count")

    environment = {
        "os": platform.system(),
        "os_release": platform.release(),
        "architecture": platform.machine(),
        "python": platform.python_version(),
        "cuda_runtime": cuda_runtime,
        "cuda_driver": cuda_driver,
        "software": {name: distribution_version(name) for name in SOFTWARE_DISTRIBUTIONS},
        "gpus": devices,
    }
    return {
        "schema_version": "1.0.0",
        "benchmark_id": BENCHMARK_ID,
        "run_id": str(uuid.uuid4()),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git": git_metadata(),
        "environment": environment,
        "parameters": {
            "backend": backend_name,
            "rows": args.rows,
            "dimension_rows": args.dimension_rows,
            "seed": args.seed,
            "warmup_iterations": args.warmups,
            "repetitions": args.repetitions,
            "setup_seconds_excluded": setup_seconds,
            "cuda_synchronized": cupy is not None,
        },
        "measurements": measurements,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--rows", type=int, default=5_000_000)
    parser.add_argument("--dimension-rows", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("results/latest.json"))
    args = parser.parse_args()
    if args.rows < 1 or args.dimension_rows < 1:
        parser.error("--rows and --dimension-rows must be positive")
    if args.warmups < 0 or args.repetitions < 1:
        parser.error("--warmups must be non-negative and --repetitions must be at least one")
    return args


def main() -> None:
    args = parse_args()
    result = build_result(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"backend={result['parameters']['backend']} rows={args.rows:,} output={args.output}")
    for item in result["measurements"]:
        print(f"{item['case_id']:28s} median={item['statistics']['median']:.6f}s samples={item['samples']}")
    print(f"GPU devices recorded: {len(result['environment']['gpus'])}")


if __name__ == "__main__":
    main()
