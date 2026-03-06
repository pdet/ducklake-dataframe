"""
DuckLake vs Iceberg — Catalog & Metadata Benchmark

Measures pure metadata operations: catalog open time, table listing,
snapshot history, time travel lookups, and cold-start read latency.
These are the operations that hit the catalog database, not data files.

Scenarios:
  1. Cold Start — time from zero to first query result
  2. Catalog Open — open catalog and list tables
  3. Snapshot History — list N snapshots after N writes
  4. Time Travel Lookup — read at snapshot version k out of N
  5. Multi-Table Catalog — create 100 tables, list them
  6. Partition Pruning — scan a partitioned table with selective filter

Usage:
    python benchmarks/bench_catalog.py [--snapshots 100] [--rows 50000]
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
    DuckLakeCatalog,
    alter_ducklake_set_partitioned_by,
    create_ducklake_table,
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


def _make_ducklake(base_dir: str, name: str = "cat") -> str:
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
    con.close()
    return meta


def _make_iceberg(base_dir: str, name: str = "cat"):
    from pyiceberg.catalog.sql import SqlCatalog

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
    return catalog


# ===================================================================
# Scenario 1: Cold Start (create catalog + table + insert + read)
# ===================================================================


def bench_cold_start_ducklake(base_dir: str, rows: int) -> BenchResult:
    d = os.path.join(base_dir, "cold_dl")
    os.makedirs(d)
    start = time.perf_counter()
    meta = os.path.join(d, "cold.ducklake")
    data = os.path.join(d, "cold_data")
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
        f"SELECT i AS id, CAST(i AS DOUBLE) AS value FROM range({rows}) t(i)"
    )
    con.close()
    df = read_ducklake(meta, "events")
    elapsed = time.perf_counter() - start
    return BenchResult(
        "cold_start", "ducklake", "end_to_end", df.shape[0], elapsed,
    )


def bench_cold_start_iceberg(base_dir: str, rows: int) -> BenchResult:
    from pyiceberg.catalog.sql import SqlCatalog
    from pyiceberg.schema import Schema
    from pyiceberg.types import DoubleType, LongType, NestedField

    d = os.path.join(base_dir, "cold_ice")
    os.makedirs(d)
    start = time.perf_counter()
    warehouse = os.path.join(d, "warehouse")
    os.makedirs(warehouse, exist_ok=True)
    catalog = SqlCatalog(
        "cold",
        **{
            "uri": f"sqlite:///{d}/cold.db",
            "warehouse": f"file://{warehouse}",
        },
    )
    catalog.create_namespace("default")
    schema = Schema(
        NestedField(1, "id", LongType(), required=False),
        NestedField(2, "value", DoubleType(), required=False),
    )
    table = catalog.create_table("default.events", schema=schema)
    arrow_data = pa.table({
        "id": pa.array(range(rows), type=pa.int64()),
        "value": pa.array([float(i) for i in range(rows)], type=pa.float64()),
    })
    table.append(arrow_data)
    df = pl.from_arrow(table.scan().to_arrow())
    elapsed = time.perf_counter() - start
    return BenchResult(
        "cold_start", "iceberg", "end_to_end", df.shape[0], elapsed,
    )


# ===================================================================
# Scenario 2: Snapshot History after N writes
# ===================================================================


def bench_snapshot_history_ducklake(base_dir: str, rows: int, n: int) -> BenchResult:
    meta = _make_ducklake(base_dir, "snap_dl")
    batch = rows // n if n > 0 else rows
    for i in range(n):
        df = pl.DataFrame({"id": list(range(i * batch, (i + 1) * batch)),
                           "value": [float(x) for x in range(i * batch, (i + 1) * batch)]})
        write_ducklake(df, meta, "events", mode="append" if i > 0 else "error")

    start = time.perf_counter()
    catalog = DuckLakeCatalog(meta)
    snapshots = catalog.snapshots()
    elapsed = time.perf_counter() - start
    return BenchResult(
        "snapshot_history", "ducklake", "metadata", n, elapsed,
        notes=f"{snapshots.shape[0]} snapshots",
    )


def bench_snapshot_history_iceberg(base_dir: str, rows: int, n: int) -> BenchResult:
    from pyiceberg.schema import Schema
    from pyiceberg.types import DoubleType, LongType, NestedField

    catalog = _make_iceberg(base_dir, "snap_ice")
    schema = Schema(
        NestedField(1, "id", LongType(), required=False),
        NestedField(2, "value", DoubleType(), required=False),
    )
    table = catalog.create_table("default.events", schema=schema)
    batch = rows // n if n > 0 else rows
    for i in range(n):
        arrow = pa.table({
            "id": pa.array(range(i * batch, (i + 1) * batch), type=pa.int64()),
            "value": pa.array([float(x) for x in range(i * batch, (i + 1) * batch)], type=pa.float64()),
        })
        table.append(arrow)

    start = time.perf_counter()
    table = catalog.load_table("default.events")
    snapshots = list(table.metadata.snapshots)
    elapsed = time.perf_counter() - start
    return BenchResult(
        "snapshot_history", "iceberg", "metadata", n, elapsed,
        notes=f"{len(snapshots)} snapshots",
    )


# ===================================================================
# Scenario 3: Multi-Table Catalog
# ===================================================================


def bench_multi_table_ducklake(base_dir: str, n_tables: int) -> BenchResult:
    meta = _make_ducklake(base_dir, "multi_dl")
    for i in range(n_tables):
        create_ducklake_table(meta, f"table_{i}", pl.Schema({"id": pl.Int64, "val": pl.Float64}))

    start = time.perf_counter()
    catalog = DuckLakeCatalog(meta)
    tables = catalog.list_tables()
    elapsed = time.perf_counter() - start
    return BenchResult(
        "multi_table_list", "ducklake", "metadata", n_tables, elapsed,
        notes=f"{tables.shape[0]} tables",
    )


def bench_multi_table_iceberg(base_dir: str, n_tables: int) -> BenchResult:
    from pyiceberg.schema import Schema
    from pyiceberg.types import DoubleType, LongType, NestedField

    catalog = _make_iceberg(base_dir, "multi_ice")
    schema = Schema(
        NestedField(1, "id", LongType(), required=False),
        NestedField(2, "val", DoubleType(), required=False),
    )
    for i in range(n_tables):
        catalog.create_table(f"default.table_{i}", schema=schema)

    start = time.perf_counter()
    tables = catalog.list_tables("default")
    elapsed = time.perf_counter() - start
    return BenchResult(
        "multi_table_list", "iceberg", "metadata", n_tables, elapsed,
        notes=f"{len(tables)} tables",
    )


# ===================================================================
# Scenario 4: Partition Pruning
# ===================================================================


def bench_partition_pruning_ducklake(base_dir: str, rows: int) -> BenchResult:
    meta = _make_ducklake(base_dir, "part_dl")
    df = pl.DataFrame({
        "id": list(range(rows)),
        "region": [f"region_{i % 20}" for i in range(rows)],
        "value": [float(i) for i in range(rows)],
    })
    write_ducklake(df, meta, "partitioned", mode="error")
    alter_ducklake_set_partitioned_by(meta, "partitioned", ["region"])
    # Re-write with partitioning active
    df2 = pl.DataFrame({
        "id": list(range(rows, rows * 2)),
        "region": [f"region_{i % 20}" for i in range(rows)],
        "value": [float(i) for i in range(rows)],
    })
    write_ducklake(df2, meta, "partitioned", mode="append")

    # Scan with filter that should prune partitions
    start = time.perf_counter()
    result = (
        scan_ducklake(meta, "partitioned")
        .filter(pl.col("region") == "region_0")
        .collect()
    )
    elapsed = time.perf_counter() - start
    return BenchResult(
        "partition_pruning", "ducklake", "scan", rows * 2, elapsed,
        notes=f"matched={result.shape[0]} (1/20 partitions)",
    )


def bench_partition_pruning_iceberg(base_dir: str, rows: int) -> BenchResult:
    from pyiceberg.catalog.sql import SqlCatalog
    from pyiceberg.schema import Schema
    from pyiceberg.types import DoubleType, LongType, NestedField, StringType
    from pyiceberg.partitioning import PartitionSpec, PartitionField
    from pyiceberg.transforms import IdentityTransform

    warehouse = os.path.join(base_dir, "part_ice_warehouse")
    os.makedirs(warehouse, exist_ok=True)
    catalog = SqlCatalog(
        "part_bench",
        **{
            "uri": f"sqlite:///{base_dir}/part_ice.db",
            "warehouse": f"file://{warehouse}",
        },
    )
    catalog.create_namespace("default")
    schema = Schema(
        NestedField(1, "id", LongType(), required=False),
        NestedField(2, "region", StringType(), required=False),
        NestedField(3, "value", DoubleType(), required=False),
    )
    spec = PartitionSpec(
        PartitionField(source_id=2, field_id=1000, transform=IdentityTransform(), name="region")
    )
    table = catalog.create_table("default.partitioned", schema=schema, partition_spec=spec)

    # Write data in batches to get multiple partition files
    for batch in range(4):
        offset = batch * (rows // 2)
        arrow = pa.table({
            "id": pa.array(range(offset, offset + rows // 2), type=pa.int64()),
            "region": pa.array([f"region_{i % 20}" for i in range(offset, offset + rows // 2)], type=pa.string()),
            "value": pa.array([float(i) for i in range(offset, offset + rows // 2)], type=pa.float64()),
        })
        table.append(arrow)

    start = time.perf_counter()
    scan = table.scan(row_filter="region = 'region_0'")
    result = pl.from_arrow(scan.to_arrow())
    elapsed = time.perf_counter() - start
    return BenchResult(
        "partition_pruning", "iceberg", "scan", rows * 2, elapsed,
        notes=f"matched={result.shape[0]} (1/20 partitions)",
    )


# ===================================================================
# Scenario 5: Time Travel Across Many Snapshots
# ===================================================================


def bench_time_travel_ducklake(meta: str, target_version: int) -> BenchResult:
    start = time.perf_counter()
    df = read_ducklake(meta, "events", snapshot_version=target_version)
    elapsed = time.perf_counter() - start
    return BenchResult(
        "time_travel", "ducklake", "read", df.shape[0], elapsed,
        notes=f"version={target_version}",
    )


def bench_time_travel_iceberg(catalog, target_idx: int) -> BenchResult:
    table = catalog.load_table("default.events")
    snapshots = list(table.metadata.snapshots)
    if target_idx >= len(snapshots):
        target_idx = len(snapshots) - 1
    snap_id = snapshots[target_idx].snapshot_id
    start = time.perf_counter()
    scan = table.scan(snapshot_id=snap_id)
    df = pl.from_arrow(scan.to_arrow())
    elapsed = time.perf_counter() - start
    return BenchResult(
        "time_travel", "iceberg", "read", df.shape[0], elapsed,
        notes=f"snapshot_idx={target_idx}",
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


def run_benchmark(snapshots: int, rows: int, output: str | None = None):
    print(f"\n{'='*70}")
    print(f"DuckLake vs Iceberg — Catalog & Metadata Benchmark")
    print(f"  {snapshots} snapshots, {rows:,} rows")
    print(f"  polars {pl.__version__}  pyarrow {pa.__version__}")
    print(f"{'='*70}\n")

    tmpdir = tempfile.mkdtemp(prefix="catalog_bench_")
    results: list[BenchResult] = []

    try:
        # 1. Cold start
        r_dl = bench_cold_start_ducklake(tmpdir, rows)
        r_ice = bench_cold_start_iceberg(tmpdir, rows)
        results.extend([r_dl, r_ice])
        _print_group("Cold Start (create + insert + read)", [r_dl, r_ice])

        # 2. Snapshot history
        r_dl = bench_snapshot_history_ducklake(tmpdir, rows, snapshots)
        r_ice = bench_snapshot_history_iceberg(tmpdir, rows, snapshots)
        results.extend([r_dl, r_ice])
        _print_group(f"Snapshot History ({snapshots} snapshots)", [r_dl, r_ice])

        # 3. Multi-table catalog
        r_dl = bench_multi_table_ducklake(tmpdir, 100)
        r_ice = bench_multi_table_iceberg(tmpdir, 100)
        results.extend([r_dl, r_ice])
        _print_group("Multi-Table Catalog (100 tables)", [r_dl, r_ice])

        # 4. Partition pruning
        r_dl = bench_partition_pruning_ducklake(tmpdir, rows)
        r_ice = bench_partition_pruning_iceberg(tmpdir, rows)
        results.extend([r_dl, r_ice])
        _print_group("Partition Pruning (20 partitions, select 1)", [r_dl, r_ice])

        # 5. Time travel (use the snapshot catalogs from scenario 2)
        # Re-use the snap_dl catalog path
        snap_dl_meta = os.path.join(tmpdir, "snap_dl.ducklake")
        snap_ice_catalog = _make_iceberg(tmpdir, "snap_ice_tt")
        # Just use existing ones
        from pyiceberg.catalog.sql import SqlCatalog
        snap_ice_catalog = SqlCatalog(
            "snap_ice",
            **{
                "uri": f"sqlite:///{tmpdir}/snap_ice_iceberg.db",
                "warehouse": f"file://{tmpdir}/snap_ice_warehouse",
            },
        )
        mid = max(1, snapshots // 2)
        r_dl = bench_time_travel_ducklake(snap_dl_meta, mid)
        r_ice = bench_time_travel_iceberg(snap_ice_catalog, mid)
        results.extend([r_dl, r_ice])
        _print_group(f"Time Travel (snapshot {mid}/{snapshots})", [r_dl, r_ice])

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
        description="DuckLake vs Iceberg — Catalog & Metadata Benchmark"
    )
    parser.add_argument("--snapshots", type=int, default=100)
    parser.add_argument("--rows", type=int, default=50_000)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    run_benchmark(args.snapshots, args.rows, args.output)


if __name__ == "__main__":
    main()
