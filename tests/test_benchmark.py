from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import jsonschema

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "benchmarks" / "run_benchmark.py"
SCHEMA = json.loads((ROOT / "benchmark-results.schema.json").read_text(encoding="utf-8"))


def test_cpu_smoke_result_matches_shared_schema(tmp_path: Path) -> None:
    output = tmp_path / "cpu-smoke.json"
    subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--backend",
            "cpu",
            "--rows",
            "2000",
            "--dimension-rows",
            "200",
            "--seed",
            "7",
            "--warmups",
            "0",
            "--repetitions",
            "2",
            "--output",
            str(output),
        ],
        check=True,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    result = json.loads(output.read_text(encoding="utf-8"))
    jsonschema.validate(result, SCHEMA)
    assert result["parameters"]["backend"] == "pandas-cpu"
    assert result["environment"]["gpus"] == []
    assert {item["case_id"] for item in result["measurements"]} == {
        "sort-values",
        "groupby-multi-aggregation",
        "left-hash-join",
    }
    assert all(len(item["samples"]) == 2 for item in result["measurements"])
    assert all(item["status"] == "completed" for item in result["measurements"])


def test_summarize_keeps_single_sample_stdev_defined() -> None:
    sys.path.insert(0, str(ROOT))
    from benchmarks.run_benchmark import summarize

    assert summarize([0.25]) == {
        "median": 0.25,
        "mean": 0.25,
        "stdev": 0.0,
        "min": 0.25,
        "max": 0.25,
    }
