"""
DuckLake vs Iceberg — Schema Evolution Benchmark

Measures the cost of schema changes and read performance after
many schema evolution steps. Both DuckLake and PyIceberg support
schema evolution, but the catalog overhead and read-time reconciliation
differ significantly.

Scenarios:
  1. Add Column (N times) — catalog DDL cost
  2. Read After Schema Evolution — read with many historical columns
  3. Rename Column (N times) — column rename DDL cost
  4. Read After Renames — reader must reconcile old Parquet files
  5. Drop + Add Cycle — repeated drop/add to simulate schema churn
  6. Wide Table Projection — read 5 columns from a 200-column table

Usage:
    python benchmarks/bench_schema_evolution.py [--evolutions 50] [--rows 100000]
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

import duckdb
import polars as pl
import pyarrow as pa

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ducklake_polars import (
    alter_ducklake_add_column,
    alter_ducklake_drop_column,
    alter_ducklake_rename_column,
    read_ducklake,
    scan_ducklake,
    write_ducklake,
)


@dataclass
class BenchResult:
    name: str
    system: str
    operation: str
    total_rows: int
    elapsed_s: float
    rows_per_sec: float = 0.0
    notes: str = ""

    def __post_init__(self):
        if self.elapsed_s > 0 and self.total_rows > 0:
            self.rows_per_sec = self.total_rows / self.elapsed_s


def _seed_ducklake(base_dir: str, rows: int) -> str:
    """Create a DuckLake catalog with a base table."""
    meta = os.path.join(base_dir, "ducklake.ducklake")
    data = os.path.join(base_dir, "ducklake_data")
    os.makedirs(data, exist_ok=True)
    con = duckdb.connect()
    con.install_extension("ducklake")
    con.load_extension("ducklake")
    con.execute(
        f"ATTACH 'ducklake:sqlite:{meta}' AS ducklake "
        f"(DATA_PATH '{data}', DATA_INLINING_ROW_LIMIT 0)"
    )
    con.execute(
        "CREATE TABLE ducklake.bench AS "
        f"SELECT i AS id, 'row_' || i AS name, CAST(i AS DOUBLE) * 1.1 AS value "
        f"FROM range({rows}) t(i)"
    )
    con.close()
    return meta


def _seed_iceberg(base_dir: str, rows: int):
    """Create an Iceberg table with the same base data."""
    from pyiceberg.catalog.sql import SqlCatalog
    from pyiceberg.schema import Schema
    from pyiceberg.types import DoubleType, LongType, NestedField, StringType

    warehouse = os.path.join(base_dir, "iceberg_warehouse")
    os.makedirs(warehouse, exist_ok=True)
    catalog = SqlCatalog(
        "bench",
        **{
            "uri": f"sqlite:///{base_dir}/iceberg.db",
            "warehouse": f"file://{warehouse}",
        },
    )
    catalog.create_namespace("default")
    schema = Schema(
        NestedField(1, "id", LongType(), required=False),
        NestedField(2, "name", StringType(), required=False),
        NestedField(3, "value", DoubleType(), required=False),
    )
    table = catalog.create_table("default.bench", schema=schema)
    arrow_data = pa.table(
        {
            "id": pa.array(range(rows), type=pa.int64()),
            "name": pa.array([f"row_{i}" for i in range(rows)], type=pa.string()),
            "value": pa.array(
                [float(i) * 1.1 for i in range(rows)], type=pa.float64()
            ),
        }
    )
    table.append(arrow_data)
    return catalog, table


# ===================================================================
# Scenario 1: Add Column (N times)
# ===================================================================


def bench_add_columns_ducklake(meta: str, n: int) -> BenchResult:
    start = time.perf_counter()
    for i in range(n):
        alter_ducklake_add_column(meta, "bench", f"extra_{i}", pl.Float64)
    elapsed = time.perf_counter() - start
    return BenchResult(
        "add_column", "ducklake", "ddl", n, elapsed, notes=f"{n} columns added"
    )


def bench_add_columns_iceberg(catalog, table, n: int) -> BenchResult:
    from pyiceberg.types import DoubleType

    start = time.perf_counter()
    for i in range(n):
        with table.update_schema() as schema_update:
            schema_update.add_column(f"extra_{i}", DoubleType())
        table = catalog.load_table("default.bench")
    elapsed = time.perf_counter() - start
    return BenchResult(
        "add_column", "iceberg", "ddl", n, elapsed, notes=f"{n} columns added"
    )


# ===================================================================
# Scenario 2: Read after schema evolution
# ===================================================================


def bench_read_after_evolution_ducklake(meta: str, rows: int) -> BenchResult:
    start = time.perf_counter()
    df = read_ducklake(meta, "bench")
    elapsed = time.perf_counter() - start
    return BenchResult(
        "read_after_evolution",
        "ducklake",
        "read",
        df.shape[0],
        elapsed,
        notes=f"{df.shape[1]} cols",
    )


def bench_read_after_evolution_iceberg(table, rows: int) -> BenchResult:
    start = time.perf_counter()
    scan = table.scan()
    arrow_table = scan.to_arrow()
    df = pl.from_arrow(arrow_table)
    elapsed = time.perf_counter() - start
    return BenchResult(
        "read_after_evolution",
        "iceberg",
        "read",
        df.shape[0],
        elapsed,
        notes=f"{df.shape[1]} cols",
    )


# ===================================================================
# Scenario 3: Rename Column (N times)
# ===================================================================


def bench_rename_columns_ducklake(meta: str, n: int) -> BenchResult:
    start = time.perf_counter()
    for i in range(n):
        alter_ducklake_rename_column(meta, "bench", f"extra_{i}", f"renamed_{i}")
    elapsed = time.perf_counter() - start
    return BenchResult(
        "rename_column", "ducklake", "ddl", n, elapsed, notes=f"{n} renames"
    )


def bench_rename_columns_iceberg(catalog, table, n: int) -> BenchResult:
    start = time.perf_counter()
    for i in range(n):
        with table.update_schema() as schema_update:
            schema_update.rename_column(f"extra_{i}", f"renamed_{i}")
        table = catalog.load_table("default.bench")
    elapsed = time.perf_counter() - start
    return BenchResult(
        "rename_column", "iceberg", "ddl", n, elapsed, notes=f"{n} renames"
    )


# ===================================================================
# Scenario 4: Read after renames
# ===================================================================


def bench_read_after_renames_ducklake(meta: str) -> BenchResult:
    start = time.perf_counter()
    df = read_ducklake(meta, "bench")
    elapsed = time.perf_counter() - start
    return BenchResult(
        "read_after_renames", "ducklake", "read", df.shape[0], elapsed,
        notes=f"{df.shape[1]} cols",
    )


def bench_read_after_renames_iceberg(table) -> BenchResult:
    start = time.perf_counter()
    scan = table.scan()
    arrow_table = scan.to_arrow()
    df = pl.from_arrow(arrow_table)
    elapsed = time.perf_counter() - start
    return BenchResult(
        "read_after_renames", "iceberg", "read", df.shape[0], elapsed,
        notes=f"{df.shape[1]} cols",
    )


# ===================================================================
# Scenario 5: Schema Churn (add + drop cycles)
# ===================================================================


def bench_schema_churn_ducklake(meta: str, cycles: int) -> BenchResult:
    start = time.perf_counter()
    for i in range(cycles):
        alter_ducklake_add_column(meta, "bench", f"churn_{i}", pl.Int64)
        alter_ducklake_drop_column(meta, "bench", f"churn_{i}")
    elapsed = time.perf_counter() - start
    return BenchResult(
        "schema_churn", "ducklake", "ddl", cycles * 2, elapsed,
        notes=f"{cycles} add+drop cycles",
    )


def bench_schema_churn_iceberg(catalog, table, cycles: int) -> BenchResult:
    from pyiceberg.types import LongType

    start = time.perf_counter()
    for i in range(cycles):
        with table.update_schema() as schema_update:
            schema_update.add_column(f"churn_{i}", LongType())
        table = catalog.load_table("default.bench")
        with table.update_schema(allow_incompatible_changes=True) as schema_update:
            schema_update.delete_column(f"churn_{i}")
        table = catalog.load_table("default.bench")
    elapsed = time.perf_counter() - start
    return BenchResult(
        "schema_churn", "iceberg", "ddl", cycles * 2, elapsed,
        notes=f"{cycles} add+drop cycles",
    )


# ===================================================================
# Scenario 6: Wide Table Projection
# ===================================================================


def bench_wide_projection_ducklake(base_dir: str, rows: int, ncols: int) -> BenchResult:
    meta = os.path.join(base_dir, "wide_dl.ducklake")
    data = os.path.join(base_dir, "wide_dl_data")
    os.makedirs(data, exist_ok=True)
    con = duckdb.connect()
    con.install_extension("ducklake")
    con.load_extension("ducklake")
    con.execute(
        f"ATTACH 'ducklake:sqlite:{meta}' AS ducklake "
        f"(DATA_PATH '{data}', DATA_INLINING_ROW_LIMIT 0)"
    )
    col_defs = ", ".join(f"c{i} BIGINT" for i in range(ncols))
    col_exprs = ", ".join(f"i + {i} AS c{i}" for i in range(ncols))
    con.execute(f"CREATE TABLE ducklake.wide ({col_defs})")
    con.execute(
        f"INSERT INTO ducklake.wide SELECT {col_exprs} FROM range({rows}) t(i)"
    )
    con.close()

    project = [f"c{i}" for i in range(0, ncols, ncols // 5)][:5]
    start = time.perf_counter()
    result = scan_ducklake(meta, "wide").select(project).collect()
    elapsed = time.perf_counter() - start
    return BenchResult(
        "wide_projection", "ducklake", "scan", rows, elapsed,
        notes=f"{ncols} cols, project {len(project)}",
    )


def bench_wide_projection_iceberg(base_dir: str, rows: int, ncols: int) -> BenchResult:
    from pyiceberg.catalog.sql import SqlCatalog
    from pyiceberg.schema import Schema
    from pyiceberg.types import LongType, NestedField

    warehouse = os.path.join(base_dir, "wide_ice_warehouse")
    os.makedirs(warehouse, exist_ok=True)
    catalog = SqlCatalog(
        "wide_bench",
        **{
            "uri": f"sqlite:///{base_dir}/wide_ice.db",
            "warehouse": f"file://{warehouse}",
        },
    )
    catalog.create_namespace("default")
    fields = [
        NestedField(i + 1, f"c{i}", LongType(), required=False) for i in range(ncols)
    ]
    table = catalog.create_table("default.wide", Schema(*fields))
    data = {f"c{i}": pa.array(range(rows), type=pa.int64()) for i in range(ncols)}
    table.append(pa.table(data))

    project = [f"c{i}" for i in range(0, ncols, ncols // 5)][:5]
    start = time.perf_counter()
    scan = table.scan(selected_fields=tuple(project))
    arrow_table = scan.to_arrow()
    result = pl.from_arrow(arrow_table)
    elapsed = time.perf_counter() - start
    return BenchResult(
        "wide_projection", "iceberg", "scan", rows, elapsed,
        notes=f"{ncols} cols, project {len(project)}",
    )


# ===================================================================
# Runner
# ===================================================================


def _print_group(name: str, results: list[BenchResult]):
    print(f"--- {name} ---")
    valid = [r for r in results if r.elapsed_s > 0]
    fastest = min(r.elapsed_s for r in valid) if valid else 1.0
    for r in results:
        if r.elapsed_s <= 0:
            print(f"  {r.system:>10s}: {'N/A':>10s}  ({r.notes})")
            continue
        ratio = r.elapsed_s / fastest if fastest > 0 else 0
        tag = "" if ratio < 1.05 else f"  ({ratio:.1f}x slower)"
        notes = f"  [{r.notes}]" if r.notes else ""
        print(
            f"  {r.system:>10s}: {r.elapsed_s:8.3f}s "
            f"({r.rows_per_sec:>12,.0f} ops/s){tag}{notes}"
        )
    print()


def run_benchmark(evolutions: int, rows: int, output: str | None = None):
    print(f"\n{'='*70}")
    print(f"DuckLake vs Iceberg — Schema Evolution Benchmark")
    print(f"  {evolutions} schema changes, {rows:,} rows")
    print(f"  polars {pl.__version__}  pyarrow {pa.__version__}")
    print(f"{'='*70}\n")

    tmpdir = tempfile.mkdtemp(prefix="schema_evo_bench_")
    results: list[BenchResult] = []

    try:
        dl_dir = os.path.join(tmpdir, "ducklake")
        os.makedirs(dl_dir)
        ice_dir = os.path.join(tmpdir, "iceberg")
        os.makedirs(ice_dir)

        dl_meta = _seed_ducklake(dl_dir, rows)
        ice_catalog, ice_table = _seed_iceberg(ice_dir, rows)

        # 1. Add columns
        r_dl = bench_add_columns_ducklake(dl_meta, evolutions)
        r_ice = bench_add_columns_iceberg(ice_catalog, ice_table, evolutions)
        ice_table = ice_catalog.load_table("default.bench")
        results.extend([r_dl, r_ice])
        _print_group(f"Add Column ({evolutions}x)", [r_dl, r_ice])

        # 2. Read after evolution
        r_dl = bench_read_after_evolution_ducklake(dl_meta, rows)
        r_ice = bench_read_after_evolution_iceberg(ice_table, rows)
        results.extend([r_dl, r_ice])
        _print_group("Read After Schema Evolution", [r_dl, r_ice])

        # 3. Rename columns
        r_dl = bench_rename_columns_ducklake(dl_meta, evolutions)
        r_ice = bench_rename_columns_iceberg(ice_catalog, ice_table, evolutions)
        ice_table = ice_catalog.load_table("default.bench")
        results.extend([r_dl, r_ice])
        _print_group(f"Rename Column ({evolutions}x)", [r_dl, r_ice])

        # 4. Read after renames
        r_dl = bench_read_after_renames_ducklake(dl_meta)
        r_ice = bench_read_after_renames_iceberg(ice_table)
        results.extend([r_dl, r_ice])
        _print_group("Read After Renames", [r_dl, r_ice])

        # 5. Schema churn (fresh catalogs)
        dl_churn_dir = os.path.join(tmpdir, "ducklake_churn")
        os.makedirs(dl_churn_dir)
        ice_churn_dir = os.path.join(tmpdir, "iceberg_churn")
        os.makedirs(ice_churn_dir)
        dl_churn_meta = _seed_ducklake(dl_churn_dir, rows)
        ice_churn_cat, ice_churn_tbl = _seed_iceberg(ice_churn_dir, rows)
        r_dl = bench_schema_churn_ducklake(dl_churn_meta, evolutions)
        r_ice = bench_schema_churn_iceberg(ice_churn_cat, ice_churn_tbl, evolutions)
        results.extend([r_dl, r_ice])
        _print_group(f"Schema Churn ({evolutions} add+drop cycles)", [r_dl, r_ice])

        # 6. Wide table projection
        r_dl = bench_wide_projection_ducklake(tmpdir, rows, 200)
        r_ice = bench_wide_projection_iceberg(tmpdir, rows, 200)
        results.extend([r_dl, r_ice])
        _print_group("Wide Table Projection (200 cols -> 5)", [r_dl, r_ice])

        # Summary
        print(f"\n{'='*70}")
        print(f"{'Name':<25s} {'System':<10s} {'Time(s)':<10s} {'Ops/s':<15s}")
        print("-" * 60)
        for r in results:
            if r.elapsed_s > 0:
                print(
                    f"{r.name:<25s} {r.system:<10s} "
                    f"{r.elapsed_s:<10.3f} {r.rows_per_sec:<15,.0f}"
                )

        if output:
            with open(output, "w") as f:
                json.dump([asdict(r) for r in results], f, indent=2)
            print(f"\nResults saved to {output}")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(
        description="DuckLake vs Iceberg — Schema Evolution Benchmark"
    )
    parser.add_argument("--evolutions", type=int, default=50)
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    run_benchmark(args.evolutions, args.rows, args.output)


if __name__ == "__main__":
    main()
