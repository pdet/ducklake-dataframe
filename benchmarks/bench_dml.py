"""
DuckLake vs Iceberg — DML Operations Benchmark

Compares delete, update, and upsert (merge) performance. DuckLake uses
position-delete files while Iceberg uses copy-on-write or merge-on-read
depending on configuration.

Scenarios:
  1. Selective Delete — delete 10% of rows by predicate
  2. Read After Deletes — read performance with accumulated delete files
  3. Bulk Update — update 20% of rows
  4. Upsert (Merge) — upsert 50% existing + 50% new rows
  5. Delete Cascade — many small deletes accumulating delete files
  6. Read Degradation — read perf after N delete rounds vs after compaction

Usage:
    python benchmarks/bench_dml.py [--rows 100000] [--delete-rounds 20]
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
    delete_ducklake,
    merge_ducklake,
    read_ducklake,
    rewrite_data_files_ducklake,
    scan_ducklake,
    update_ducklake,
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


def _make_data(rows: int) -> dict:
    return {
        "id": list(range(rows)),
        "category": [f"cat_{i % 50}" for i in range(rows)],
        "value": [float(i) * 1.1 for i in range(rows)],
        "status": ["active" if i % 10 != 0 else "inactive" for i in range(rows)],
    }


def _seed_ducklake(base_dir: str, rows: int, name: str = "bench") -> str:
    meta = os.path.join(base_dir, f"{name}.ducklake")
    data = os.path.join(base_dir, f"{name}_data")
    os.makedirs(data, exist_ok=True)
    con = duckdb.connect()
    con.install_extension("ducklake")
    con.load_extension("ducklake")
    con.execute(
        f"ATTACH 'ducklake:sqlite:{meta}' AS ducklake "
        f"(DATA_PATH '{data}', DATA_INLINING_ROW_LIMIT 0)"
    )
    con.execute(
        f"CREATE TABLE ducklake.{name} AS "
        f"SELECT i AS id, 'cat_' || (i % 50) AS category, "
        f"CAST(i AS DOUBLE) * 1.1 AS value, "
        f"CASE WHEN i % 10 = 0 THEN 'inactive' ELSE 'active' END AS status "
        f"FROM range({rows}) t(i)"
    )
    con.close()
    return meta


def _seed_iceberg(base_dir: str, rows: int, name: str = "bench"):
    from pyiceberg.catalog.sql import SqlCatalog
    from pyiceberg.schema import Schema
    from pyiceberg.types import DoubleType, LongType, NestedField, StringType

    warehouse = os.path.join(base_dir, f"{name}_warehouse")
    os.makedirs(warehouse, exist_ok=True)
    catalog = SqlCatalog(
        name,
        **{
            "uri": f"sqlite:///{base_dir}/{name}_iceberg.db",
            "warehouse": f"file://{warehouse}",
        },
    )
    catalog.create_namespace("default")
    schema = Schema(
        NestedField(1, "id", LongType(), required=False),
        NestedField(2, "category", StringType(), required=False),
        NestedField(3, "value", DoubleType(), required=False),
        NestedField(4, "status", StringType(), required=False),
    )
    table = catalog.create_table(f"default.{name}", schema=schema)
    d = _make_data(rows)
    arrow = pa.table({
        "id": pa.array(d["id"], type=pa.int64()),
        "category": pa.array(d["category"], type=pa.string()),
        "value": pa.array(d["value"], type=pa.float64()),
        "status": pa.array(d["status"], type=pa.string()),
    })
    table.append(arrow)
    return catalog, table


# ===================================================================
# Scenario 1: Selective Delete (10%)
# ===================================================================


def bench_delete_ducklake(meta: str, rows: int) -> BenchResult:
    start = time.perf_counter()
    deleted = delete_ducklake(meta, "bench", pl.col("status") == "inactive")
    elapsed = time.perf_counter() - start
    return BenchResult(
        "selective_delete", "ducklake", "delete", rows, elapsed,
        notes=f"deleted={deleted}",
    )


def bench_delete_iceberg(table, rows: int) -> BenchResult:
    start = time.perf_counter()
    table.delete("status = 'inactive'")
    elapsed = time.perf_counter() - start
    return BenchResult(
        "selective_delete", "iceberg", "delete", rows, elapsed,
        notes="status='inactive'",
    )


# ===================================================================
# Scenario 2: Read After Deletes
# ===================================================================


def bench_read_after_delete_ducklake(meta: str) -> BenchResult:
    start = time.perf_counter()
    df = read_ducklake(meta, "bench")
    elapsed = time.perf_counter() - start
    return BenchResult(
        "read_after_delete", "ducklake", "read", df.shape[0], elapsed,
    )


def bench_read_after_delete_iceberg(table) -> BenchResult:
    start = time.perf_counter()
    scan = table.scan()
    df = pl.from_arrow(scan.to_arrow())
    elapsed = time.perf_counter() - start
    return BenchResult(
        "read_after_delete", "iceberg", "read", df.shape[0], elapsed,
    )


# ===================================================================
# Scenario 3: Bulk Update (20%)
# ===================================================================


def bench_update_ducklake(base_dir: str, rows: int) -> BenchResult:
    meta = _seed_ducklake(base_dir, rows, "update_dl")
    start = time.perf_counter()
    updated = update_ducklake(
        meta, "update_dl",
        updates={"value": -1.0, "status": "updated"},
        predicate=pl.col("id") % 5 == 0,
    )
    elapsed = time.perf_counter() - start
    return BenchResult(
        "bulk_update", "ducklake", "update", rows, elapsed,
        notes=f"updated={updated}",
    )


def bench_update_iceberg(base_dir: str, rows: int) -> BenchResult:
    catalog, table = _seed_iceberg(base_dir, rows, "update_ice")
    # PyIceberg doesn't have a native update — need to overwrite
    start = time.perf_counter()
    scan = table.scan()
    arrow_table = scan.to_arrow()
    df = pl.from_arrow(arrow_table)
    mask = df["id"] % 5 == 0
    updated_count = mask.sum()
    df = df.with_columns([
        pl.when(pl.col("id") % 5 == 0).then(pl.lit(-1.0)).otherwise(pl.col("value")).alias("value"),
        pl.when(pl.col("id") % 5 == 0).then(pl.lit("updated")).otherwise(pl.col("status")).alias("status"),
    ])
    table.overwrite(df.to_arrow())
    elapsed = time.perf_counter() - start
    return BenchResult(
        "bulk_update", "iceberg", "update", rows, elapsed,
        notes=f"updated={updated_count} (read-modify-write)",
    )


# ===================================================================
# Scenario 4: Upsert (Merge)
# ===================================================================


def bench_merge_ducklake(base_dir: str, rows: int) -> BenchResult:
    meta = _seed_ducklake(base_dir, rows, "merge_dl")
    half = rows // 2
    source = pl.DataFrame({
        "id": list(range(half, half + rows)),  # 50% overlap
        "category": [f"cat_{i % 50}" for i in range(half, half + rows)],
        "value": [float(i) * 2.2 for i in range(half, half + rows)],
        "status": ["merged"] * rows,
    })
    start = time.perf_counter()
    updated, inserted = merge_ducklake(
        meta, "merge_dl", source, on="id",
        when_matched_update=True,
        when_not_matched_insert=True,
    )
    elapsed = time.perf_counter() - start
    return BenchResult(
        "upsert_merge", "ducklake", "merge", rows, elapsed,
        notes=f"updated={updated}, inserted={inserted}",
    )


def bench_merge_iceberg(base_dir: str, rows: int) -> BenchResult:
    catalog, table = _seed_iceberg(base_dir, rows, "merge_ice")
    half = rows // 2
    source = pa.table({
        "id": pa.array(list(range(half, half + rows)), type=pa.int64()),
        "category": pa.array([f"cat_{i % 50}" for i in range(half, half + rows)], type=pa.string()),
        "value": pa.array([float(i) * 2.2 for i in range(half, half + rows)], type=pa.float64()),
        "status": pa.array(["merged"] * rows, type=pa.string()),
    })
    start = time.perf_counter()
    result = table.upsert(source, join_cols=["id"])
    elapsed = time.perf_counter() - start
    return BenchResult(
        "upsert_merge", "iceberg", "merge", rows, elapsed,
        notes=f"rows_updated={result.rows_updated}, rows_inserted={result.rows_inserted}",
    )


# ===================================================================
# Scenario 5: Delete Cascade (many small deletes)
# ===================================================================


def bench_delete_cascade_ducklake(base_dir: str, rows: int, rounds: int) -> BenchResult:
    meta = _seed_ducklake(base_dir, rows, "cascade_dl")
    chunk = rows // (rounds * 2)
    start = time.perf_counter()
    total_deleted = 0
    for r in range(rounds):
        lo = r * chunk
        hi = lo + chunk
        deleted = delete_ducklake(
            meta, "cascade_dl",
            (pl.col("id") >= lo) & (pl.col("id") < hi),
        )
        total_deleted += deleted
    elapsed = time.perf_counter() - start
    return BenchResult(
        "delete_cascade", "ducklake", "delete", rows, elapsed,
        notes=f"{rounds} rounds, deleted={total_deleted}",
    )


def bench_delete_cascade_iceberg(base_dir: str, rows: int, rounds: int) -> BenchResult:
    catalog, table = _seed_iceberg(base_dir, rows, "cascade_ice")
    chunk = rows // (rounds * 2)
    start = time.perf_counter()
    total_deleted = 0
    for r in range(rounds):
        lo = r * chunk
        hi = lo + chunk
        before = table.scan().to_arrow().num_rows
        table.delete(f"id >= {lo} and id < {hi}")
        after = table.scan().to_arrow().num_rows
        total_deleted += before - after
    elapsed = time.perf_counter() - start
    return BenchResult(
        "delete_cascade", "iceberg", "delete", rows, elapsed,
        notes=f"{rounds} rounds, deleted={total_deleted}",
    )


# ===================================================================
# Scenario 6: Read degradation after deletes vs after compaction
# ===================================================================


def bench_read_degradation_ducklake(base_dir: str, rows: int, rounds: int) -> list[BenchResult]:
    meta = _seed_ducklake(base_dir, rows, "degrad_dl")
    # Accumulate deletes
    chunk = rows // (rounds * 4)
    for r in range(rounds):
        lo = r * chunk
        hi = lo + chunk
        delete_ducklake(meta, "degrad_dl", (pl.col("id") >= lo) & (pl.col("id") < hi))

    # Read with accumulated delete files
    start = time.perf_counter()
    df = read_ducklake(meta, "degrad_dl")
    elapsed_before = time.perf_counter() - start
    r_before = BenchResult(
        "read_before_compact", "ducklake", "read", df.shape[0], elapsed_before,
        notes=f"after {rounds} delete rounds",
    )

    # Compact
    rewrite_data_files_ducklake(meta, "degrad_dl")

    # Read after compaction
    start = time.perf_counter()
    df = read_ducklake(meta, "degrad_dl")
    elapsed_after = time.perf_counter() - start
    r_after = BenchResult(
        "read_after_compact", "ducklake", "read", df.shape[0], elapsed_after,
        notes=f"compacted",
    )
    return [r_before, r_after]


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
            f"({r.rows_per_sec:>12,.0f} rows/s){tag}{notes}"
        )
    print()


def run_benchmark(rows: int, delete_rounds: int, output: str | None = None):
    print(f"\n{'='*70}")
    print(f"DuckLake vs Iceberg — DML Operations Benchmark")
    print(f"  {rows:,} rows, {delete_rounds} delete rounds")
    print(f"  polars {pl.__version__}  pyarrow {pa.__version__}")
    print(f"{'='*70}\n")

    tmpdir = tempfile.mkdtemp(prefix="dml_bench_")
    results: list[BenchResult] = []

    try:
        # Setup for delete + read-after-delete
        dl_dir = os.path.join(tmpdir, "ducklake")
        os.makedirs(dl_dir)
        ice_dir = os.path.join(tmpdir, "iceberg")
        os.makedirs(ice_dir)
        dl_meta = _seed_ducklake(dl_dir, rows)
        ice_catalog, ice_table = _seed_iceberg(ice_dir, rows)

        # 1. Selective delete
        r_dl = bench_delete_ducklake(dl_meta, rows)
        r_ice = bench_delete_iceberg(ice_table, rows)
        results.extend([r_dl, r_ice])
        _print_group("Selective Delete (10% of rows)", [r_dl, r_ice])

        # 2. Read after delete
        r_dl = bench_read_after_delete_ducklake(dl_meta)
        ice_table = ice_catalog.load_table("default.bench")
        r_ice = bench_read_after_delete_iceberg(ice_table)
        results.extend([r_dl, r_ice])
        _print_group("Read After Delete", [r_dl, r_ice])

        # 3. Bulk update
        r_dl = bench_update_ducklake(tmpdir, rows)
        r_ice = bench_update_iceberg(tmpdir, rows)
        results.extend([r_dl, r_ice])
        _print_group("Bulk Update (20% of rows)", [r_dl, r_ice])

        # 4. Upsert merge
        r_dl = bench_merge_ducklake(tmpdir, rows)
        r_ice = bench_merge_iceberg(tmpdir, rows)
        results.extend([r_dl, r_ice])
        _print_group("Upsert Merge (50% overlap)", [r_dl, r_ice])

        # 5. Delete cascade
        r_dl = bench_delete_cascade_ducklake(tmpdir, rows, delete_rounds)
        r_ice = bench_delete_cascade_iceberg(tmpdir, rows, delete_rounds)
        results.extend([r_dl, r_ice])
        _print_group(f"Delete Cascade ({delete_rounds} rounds)", [r_dl, r_ice])

        # 6. Read degradation (DuckLake only — shows compaction benefit)
        degrad_results = bench_read_degradation_ducklake(tmpdir, rows, delete_rounds)
        results.extend(degrad_results)
        _print_group("Read Degradation (DuckLake: before vs after compaction)", degrad_results)

        # Summary
        print(f"\n{'='*70}")
        print(f"{'Name':<25s} {'System':<10s} {'Time(s)':<10s} {'Rows/s':<15s}")
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
        description="DuckLake vs Iceberg — DML Operations Benchmark"
    )
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--delete-rounds", type=int, default=20)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    run_benchmark(args.rows, args.delete_rounds, args.output)


if __name__ == "__main__":
    main()
