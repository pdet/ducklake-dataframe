"""
DuckLake Catalog Backend Benchmark — SQLite vs DuckDB vs PostgreSQL

Compares catalog performance across different metadata backends.
All backends store the same DuckLake metadata; only the catalog
database engine differs. Data files are always local Parquet.

Measures both the DuckDB extension path (writing via DuckDB SQL)
and the ducklake-dataframe path (reading/writing via Python).

Scenarios:
  1. Catalog Creation + Table Create + Insert (cold start)
  2. Sequential Writes (N appends)
  3. Read Performance (after N appends)
  4. Schema Evolution (add/rename column)
  5. Scan + Filter (predicate pushdown)
  6. Snapshot History (list after N writes)
  7. Mixed Workload (interleaved reads and writes)

Usage:
    python benchmarks/bench_backends.py [--rows 100000] [--appends 50]
    python benchmarks/bench_backends.py --pg-dsn "postgresql://user:pass@localhost/testdb"
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from typing import Callable

import duckdb
import polars as pl
import pyarrow as pa

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ducklake_polars import (
    DuckLakeCatalog,
    alter_ducklake_add_column,
    alter_ducklake_rename_column,
    read_ducklake,
    scan_ducklake,
    write_ducklake,
)


@dataclass
class BenchResult:
    name: str
    backend: str  # "sqlite", "duckdb", or "postgres"
    operation: str
    total_rows: int
    elapsed_s: float
    rows_per_sec: float = 0.0
    notes: str = ""

    def __post_init__(self):
        if self.elapsed_s > 0 and self.total_rows > 0:
            self.rows_per_sec = self.total_rows / self.elapsed_s


class BackendSetup:
    """Manages catalog creation for each backend."""

    def __init__(self, base_dir: str, pg_dsn: str | None = None):
        self.base_dir = base_dir
        self.pg_dsn = pg_dsn
        self._pg_counter = 0

    def backends(self) -> list[str]:
        backends = ["sqlite", "duckdb"]
        if self.pg_dsn:
            backends.append("postgres")
        return backends

    def make_catalog(self, backend: str, name: str) -> tuple[str, str]:
        """Create a DuckLake catalog, return (meta_path, data_dir).

        For postgres, meta_path is the DSN with a unique schema.
        """
        data_dir = os.path.join(self.base_dir, f"{name}_{backend}_data")
        os.makedirs(data_dir, exist_ok=True)

        if backend == "sqlite":
            meta = os.path.join(self.base_dir, f"{name}_{backend}.ducklake")
            attach = f"ducklake:sqlite:{meta}"
        elif backend == "duckdb":
            meta = os.path.join(self.base_dir, f"{name}_{backend}.duckdb")
            attach = f"ducklake:duckdb:{meta}"
        elif backend == "postgres":
            self._pg_counter += 1
            schema = f"bench_{name}_{self._pg_counter}"
            meta = f"{self.pg_dsn}?schema={schema}"
            attach = f"ducklake:postgres:{meta}"
        else:
            raise ValueError(f"Unknown backend: {backend}")

        con = duckdb.connect()
        con.install_extension("ducklake")
        con.load_extension("ducklake")
        con.execute(
            f"ATTACH '{attach}' AS ducklake "
            f"(DATA_PATH '{data_dir}', DATA_INLINING_ROW_LIMIT 0)"
        )
        con.close()
        return meta, data_dir

    def ducklake_path(self, backend: str, meta: str) -> str:
        """Return the path ducklake-dataframe uses to open the catalog."""
        if backend == "postgres":
            return meta  # DSN
        return meta  # file path


def _make_batch(offset: int, size: int) -> pl.DataFrame:
    return pl.DataFrame({
        "id": list(range(offset, offset + size)),
        "category": [f"cat_{(offset + i) % 50}" for i in range(size)],
        "value": [float(offset + i) * 1.1 for i in range(size)],
        "status": [
            "active" if (offset + i) % 10 != 0 else "inactive"
            for i in range(size)
        ],
    })


# ===================================================================
# Scenario 1: Cold Start
# ===================================================================


def bench_cold_start(setup: BackendSetup, backend: str, rows: int) -> BenchResult:
    data_dir = os.path.join(setup.base_dir, f"cold_{backend}_data")
    os.makedirs(data_dir, exist_ok=True)

    if backend == "sqlite":
        meta = os.path.join(setup.base_dir, f"cold_{backend}.ducklake")
        attach = f"ducklake:sqlite:{meta}"
    elif backend == "duckdb":
        meta = os.path.join(setup.base_dir, f"cold_{backend}.duckdb")
        attach = f"ducklake:duckdb:{meta}"
    elif backend == "postgres":
        setup._pg_counter += 1
        schema = f"bench_cold_{setup._pg_counter}"
        meta = f"{setup.pg_dsn}?schema={schema}"
        attach = f"ducklake:postgres:{meta}"

    start = time.perf_counter()
    con = duckdb.connect()
    con.install_extension("ducklake")
    con.load_extension("ducklake")
    con.execute(
        f"ATTACH '{attach}' AS ducklake "
        f"(DATA_PATH '{data_dir}', DATA_INLINING_ROW_LIMIT 0)"
    )
    con.execute(
        f"CREATE TABLE ducklake.bench AS "
        f"SELECT i AS id, 'cat_' || (i % 50) AS category, "
        f"CAST(i AS DOUBLE) * 1.1 AS value, "
        f"CASE WHEN i % 10 = 0 THEN 'inactive' ELSE 'active' END AS status "
        f"FROM range({rows}) t(i)"
    )
    con.close()
    df = read_ducklake(meta, "bench")
    elapsed = time.perf_counter() - start

    return BenchResult(
        "cold_start", backend, "end_to_end", df.shape[0], elapsed,
    )


# ===================================================================
# Scenario 2: Sequential Writes
# ===================================================================


def bench_sequential_writes(
    setup: BackendSetup, backend: str, rows: int, appends: int
) -> BenchResult:
    meta, _ = setup.make_catalog(backend, "writes")
    batch_size = rows // appends

    start = time.perf_counter()
    for i in range(appends):
        df = _make_batch(i * batch_size, batch_size)
        mode = "error" if i == 0 else "append"
        write_ducklake(df, meta, "bench", mode=mode)
    elapsed = time.perf_counter() - start

    return BenchResult(
        "sequential_writes", backend, "write", rows, elapsed,
        notes=f"{appends} appends",
    )


# ===================================================================
# Scenario 3: Read Performance (after appends)
# ===================================================================


def bench_read_after_writes(
    setup: BackendSetup, backend: str, meta: str
) -> BenchResult:
    start = time.perf_counter()
    df = read_ducklake(meta, "bench")
    elapsed = time.perf_counter() - start

    return BenchResult(
        "read_after_writes", backend, "read", df.shape[0], elapsed,
    )


# ===================================================================
# Scenario 4: Schema Evolution
# ===================================================================


def bench_schema_evolution(
    setup: BackendSetup, backend: str, rows: int, n_ops: int
) -> BenchResult:
    meta, _ = setup.make_catalog(backend, "schema")
    df = _make_batch(0, rows)
    write_ducklake(df, meta, "bench", mode="error")

    start = time.perf_counter()
    for i in range(n_ops):
        alter_ducklake_add_column(meta, "bench", f"col_{i}", pl.Float64)
    for i in range(n_ops):
        alter_ducklake_rename_column(meta, "bench", f"col_{i}", f"renamed_{i}")
    elapsed = time.perf_counter() - start

    return BenchResult(
        "schema_evolution", backend, "ddl", n_ops * 2, elapsed,
        notes=f"{n_ops} adds + {n_ops} renames",
    )


# ===================================================================
# Scenario 5: Scan + Filter
# ===================================================================


def bench_scan_filter(
    setup: BackendSetup, backend: str, rows: int
) -> BenchResult:
    meta, _ = setup.make_catalog(backend, "scan")
    # Write in multiple files for realistic filter test
    batch_size = rows // 5
    for i in range(5):
        df = _make_batch(i * batch_size, batch_size)
        mode = "error" if i == 0 else "append"
        write_ducklake(df, meta, "bench", mode=mode)

    start = time.perf_counter()
    result = (
        scan_ducklake(meta, "bench")
        .filter(pl.col("status") == "inactive")
        .select("id", "value")
        .collect()
    )
    elapsed = time.perf_counter() - start

    return BenchResult(
        "scan_filter", backend, "scan", rows, elapsed,
        notes=f"matched={result.shape[0]}",
    )


# ===================================================================
# Scenario 6: Snapshot History
# ===================================================================


def bench_snapshot_history(
    setup: BackendSetup, backend: str, meta: str
) -> BenchResult:
    start = time.perf_counter()
    catalog = DuckLakeCatalog(meta)
    snapshots = catalog.snapshots()
    elapsed = time.perf_counter() - start

    return BenchResult(
        "snapshot_history", backend, "metadata",
        snapshots.shape[0], elapsed,
        notes=f"{snapshots.shape[0]} snapshots",
    )


# ===================================================================
# Scenario 7: Mixed Workload
# ===================================================================


def bench_mixed_workload(
    setup: BackendSetup, backend: str, rows: int, rounds: int
) -> BenchResult:
    meta, _ = setup.make_catalog(backend, "mixed")
    batch_size = rows // rounds

    start = time.perf_counter()
    for i in range(rounds):
        df = _make_batch(i * batch_size, batch_size)
        mode = "error" if i == 0 else "append"
        write_ducklake(df, meta, "bench", mode=mode)
        # Read after every write
        result = read_ducklake(meta, "bench")
    elapsed = time.perf_counter() - start

    return BenchResult(
        "mixed_workload", backend, "mixed", rows, elapsed,
        notes=f"{rounds} write+read cycles",
    )


# ===================================================================
# Runner
# ===================================================================


def _print_group(name: str, results: list[BenchResult]):
    print(f"\n--- {name} ---")
    valid = [r for r in results if r.elapsed_s > 0]
    if not valid:
        return
    fastest = min(r.elapsed_s for r in valid)
    for r in results:
        if r.elapsed_s <= 0:
            print(f"  {r.backend:>10s}: {'N/A':>10s}  ({r.notes})")
            continue
        ratio = r.elapsed_s / fastest if fastest > 0 else 0
        tag = "" if ratio < 1.05 else f"  ({ratio:.1f}x slower)"
        notes = f"  [{r.notes}]" if r.notes else ""
        print(
            f"  {r.backend:>10s}: {r.elapsed_s:8.3f}s "
            f"({r.rows_per_sec:>12,.0f} rows/s){tag}{notes}"
        )


def run_benchmark(
    rows: int, appends: int, pg_dsn: str | None, output: str | None
):
    tmpdir = tempfile.mkdtemp(prefix="backend_bench_")
    setup = BackendSetup(tmpdir, pg_dsn)
    backends = setup.backends()

    print(f"\n{'='*70}")
    print(f"DuckLake Catalog Backend Benchmark")
    print(f"  Backends: {', '.join(backends)}")
    print(f"  Rows: {rows:,}  Appends: {appends}")
    print(f"  polars {pl.__version__}  duckdb {duckdb.__version__}")
    print(f"{'='*70}")

    all_results: list[BenchResult] = []

    try:
        # 1. Cold start
        group = [bench_cold_start(setup, b, rows) for b in backends]
        all_results.extend(group)
        _print_group("Cold Start (create + insert + read)", group)

        # 2. Sequential writes
        group = [bench_sequential_writes(setup, b, rows, appends) for b in backends]
        all_results.extend(group)
        _print_group(f"Sequential Writes ({appends} appends)", group)

        # 3. Read after writes (reuse the catalogs from scenario 2)
        read_group = []
        for b in backends:
            if b == "sqlite":
                meta = os.path.join(tmpdir, f"writes_{b}.ducklake")
            elif b == "duckdb":
                meta = os.path.join(tmpdir, f"writes_{b}.duckdb")
            else:
                # find the postgres meta from the writes catalog
                meta = [r for r in group if r.backend == b][0].notes
                continue  # skip for now if we can't easily recover meta
            r = bench_read_after_writes(setup, b, meta)
            read_group.append(r)
        all_results.extend(read_group)
        _print_group(f"Read After {appends} Appends", read_group)

        # 4. Schema evolution
        n_schema_ops = min(appends, 30)
        group = [bench_schema_evolution(setup, b, rows, n_schema_ops) for b in backends]
        all_results.extend(group)
        _print_group(f"Schema Evolution ({n_schema_ops} adds + {n_schema_ops} renames)", group)

        # 5. Scan + Filter
        group = [bench_scan_filter(setup, b, rows) for b in backends]
        all_results.extend(group)
        _print_group("Scan + Filter (10% selectivity, 5 files)", group)

        # 6. Snapshot history (reuse writes catalogs)
        snap_group = []
        for b in backends:
            if b == "sqlite":
                meta = os.path.join(tmpdir, f"writes_{b}.ducklake")
            elif b == "duckdb":
                meta = os.path.join(tmpdir, f"writes_{b}.duckdb")
            else:
                continue
            r = bench_snapshot_history(setup, b, meta)
            snap_group.append(r)
        all_results.extend(snap_group)
        _print_group(f"Snapshot History ({appends} snapshots)", snap_group)

        # 7. Mixed workload
        mix_rounds = min(appends, 20)
        group = [bench_mixed_workload(setup, b, rows, mix_rounds) for b in backends]
        all_results.extend(group)
        _print_group(f"Mixed Workload ({mix_rounds} write+read cycles)", group)

        # Summary
        print(f"\n{'='*70}")
        print(f"{'Name':<25s} {'Backend':<10s} {'Time(s)':<10s} {'Rows/s':<15s}")
        print("-" * 60)
        for r in all_results:
            if r.elapsed_s > 0:
                print(
                    f"{r.name:<25s} {r.backend:<10s} "
                    f"{r.elapsed_s:<10.3f} {r.rows_per_sec:<15,.0f}"
                )

        if output:
            with open(output, "w") as f:
                json.dump([asdict(r) for r in all_results], f, indent=2)
            print(f"\nResults saved to {output}")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(
        description="DuckLake Catalog Backend Benchmark"
    )
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--appends", type=int, default=50)
    parser.add_argument("--pg-dsn", type=str, default=None,
                        help="PostgreSQL DSN (e.g. postgresql://user:pass@localhost/testdb)")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    run_benchmark(args.rows, args.appends, args.pg_dsn, args.output)


if __name__ == "__main__":
    main()
