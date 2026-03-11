"""
Catalog metadata overhead benchmark.

Measures the time spent on catalog queries vs actual data reading.
This isolates the metadata bottleneck that batch optimization targets.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time

import duckdb
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ducklake_polars import (
    create_ducklake_table,
    read_ducklake,
    scan_ducklake,
    write_ducklake,
)


def setup_catalog(tmpdir: str, n_files: int, rows_per_file: int) -> str:
    """Create a DuckLake catalog with N files."""
    meta = os.path.join(tmpdir, "bench.ducklake")
    data = os.path.join(tmpdir, "bench_data")
    os.makedirs(data, exist_ok=True)

    con = duckdb.connect()
    con.install_extension("ducklake")
    con.load_extension("ducklake")
    con.execute(
        f"ATTACH 'ducklake:sqlite:{meta}' AS ducklake "
        f"(DATA_PATH '{data}', DATA_INLINING_ROW_LIMIT 0)"
    )
    con.execute(
        f"CREATE TABLE ducklake.events AS "
        f"SELECT i AS id, CAST(i AS DOUBLE) AS value, "
        f"'region_' || (i % 20) AS region "
        f"FROM range({rows_per_file}) t(i)"
    )
    con.close()

    for i in range(1, n_files):
        df = pl.DataFrame({
            "id": list(range(i * rows_per_file, (i + 1) * rows_per_file)),
            "value": [float(x) for x in range(i * rows_per_file, (i + 1) * rows_per_file)],
            "region": [f"region_{x % 20}" for x in range(rows_per_file)],
        })
        write_ducklake(df, meta, "events", mode="append")

    return meta


def bench_read_cold(meta: str, n_iters: int = 5) -> dict:
    """Time cold reads (new reader each time)."""
    times = []
    for _ in range(n_iters):
        start = time.perf_counter()
        df = read_ducklake(meta, "events")
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    return {
        "operation": "read_cold",
        "mean_s": sum(times) / len(times),
        "min_s": min(times),
        "max_s": max(times),
        "rows": df.shape[0],
    }


def bench_scan_cold(meta: str, n_iters: int = 5) -> dict:
    """Time cold scans with filter (new reader each time)."""
    times = []
    for _ in range(n_iters):
        start = time.perf_counter()
        df = scan_ducklake(meta, "events").filter(
            pl.col("region") == "region_0"
        ).collect()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    return {
        "operation": "scan_filter_cold",
        "mean_s": sum(times) / len(times),
        "min_s": min(times),
        "max_s": max(times),
        "rows": df.shape[0],
    }


def bench_scan_no_filter_cold(meta: str, n_iters: int = 5) -> dict:
    """Time cold scans without filter."""
    times = []
    for _ in range(n_iters):
        start = time.perf_counter()
        df = scan_ducklake(meta, "events").collect()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    return {
        "operation": "scan_no_filter_cold",
        "mean_s": sum(times) / len(times),
        "min_s": min(times),
        "max_s": max(times),
        "rows": df.shape[0],
    }


def bench_write_append(meta: str, rows: int = 10000, n_iters: int = 5) -> dict:
    """Time append writes."""
    times = []
    for i in range(n_iters):
        df = pl.DataFrame({
            "id": list(range(i * rows, (i + 1) * rows)),
            "value": [float(x) for x in range(i * rows, (i + 1) * rows)],
            "region": [f"region_{x % 20}" for x in range(rows)],
        })
        start = time.perf_counter()
        write_ducklake(df, meta, "events", mode="append")
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    return {
        "operation": "write_append",
        "mean_s": sum(times) / len(times),
        "min_s": min(times),
        "max_s": max(times),
        "rows": rows,
    }


def run_benchmark(n_files: int = 50, rows_per_file: int = 1000, output: str | None = None):
    total_rows = n_files * rows_per_file
    print(f"\n{'='*70}")
    print(f"Catalog Metadata Overhead Benchmark")
    print(f"  {n_files} files, {rows_per_file} rows/file, {total_rows:,} total rows")
    print(f"{'='*70}\n")

    tmpdir = tempfile.mkdtemp(prefix="catalog_overhead_")
    results = []

    try:
        print("Setting up catalog...")
        meta = setup_catalog(tmpdir, n_files, rows_per_file)
        print(f"  Created {n_files} files\n")

        # Benchmark reads
        r = bench_read_cold(meta)
        results.append(r)
        print(f"read_cold:            {r['mean_s']:.4f}s (min={r['min_s']:.4f}s, max={r['max_s']:.4f}s) [{r['rows']} rows]")

        r = bench_scan_no_filter_cold(meta)
        results.append(r)
        print(f"scan_no_filter_cold:  {r['mean_s']:.4f}s (min={r['min_s']:.4f}s, max={r['max_s']:.4f}s) [{r['rows']} rows]")

        r = bench_scan_cold(meta)
        results.append(r)
        print(f"scan_filter_cold:     {r['mean_s']:.4f}s (min={r['min_s']:.4f}s, max={r['max_s']:.4f}s) [{r['rows']} rows]")

        # Write benchmark (uses a separate catalog to not affect read benchmarks)
        meta_w = setup_catalog(os.path.join(tmpdir, "write"), 5, rows_per_file)
        r = bench_write_append(meta_w)
        results.append(r)
        print(f"write_append:         {r['mean_s']:.4f}s (min={r['min_s']:.4f}s, max={r['max_s']:.4f}s) [{r['rows']} rows]")

        if output:
            with open(output, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {output}")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", type=int, default=50)
    parser.add_argument("--rows-per-file", type=int, default=1000)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    run_benchmark(args.files, args.rows_per_file, args.output)
