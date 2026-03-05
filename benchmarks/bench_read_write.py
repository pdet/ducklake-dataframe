"""
DuckLake-Polars Benchmarks: Read/Write/Scan Performance

Compare ducklake-dataframe vs DuckDB native ducklake across various
data sizes, column counts, and operation types.

Usage:
    python benchmarks/bench_read_write.py [--rows 100000] [--output results.json]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field

import duckdb
import polars as pl

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ducklake_polars import (
    DuckLakeCatalog,
    read_ducklake,
    scan_ducklake,
    write_ducklake,
)


@dataclass
class BenchResult:
    name: str
    system: str  # "polars" or "duckdb"
    operation: str  # "write", "read", "scan_filter", "scan_agg"
    rows: int
    columns: int
    elapsed_s: float
    rows_per_sec: float = 0.0
    mb_per_sec: float = 0.0
    notes: str = ""

    def __post_init__(self):
        if self.elapsed_s > 0:
            self.rows_per_sec = self.rows / self.elapsed_s


class DuckLakeBenchmark:
    """Benchmark harness for ducklake operations."""

    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.results: list[BenchResult] = []

    def _make_catalog(self, name: str) -> tuple[str, str]:
        meta = os.path.join(self.base_dir, f"{name}.ducklake")
        data = os.path.join(self.base_dir, f"{name}_data")
        os.makedirs(data, exist_ok=True)
        # Init via DuckDB
        con = duckdb.connect()
        con.install_extension("ducklake")
        con.load_extension("ducklake")
        con.execute(
            f"ATTACH 'ducklake:sqlite:{meta}' AS ducklake "
            f"(DATA_PATH '{data}', DATA_INLINING_ROW_LIMIT 0)"
        )
        con.close()
        return meta, data

    def _make_duckdb_con(self, meta: str, data: str) -> duckdb.DuckDBPyConnection:
        con = duckdb.connect()
        con.install_extension("ducklake")
        con.load_extension("ducklake")
        con.execute(
            f"ATTACH 'ducklake:sqlite:{meta}' AS ducklake "
            f"(DATA_PATH '{data}', DATA_INLINING_ROW_LIMIT 0)"
        )
        return con

    def _generate_data(self, rows: int, cols: int) -> pl.DataFrame:
        data = {}
        for i in range(cols):
            ctype = i % 4
            if ctype == 0:
                data[f"int_{i}"] = list(range(rows))
            elif ctype == 1:
                data[f"float_{i}"] = [float(x) * 1.1 for x in range(rows)]
            elif ctype == 2:
                data[f"str_{i}"] = [f"value_{x % 1000}" for x in range(rows)]
            else:
                data[f"bool_{i}"] = [x % 2 == 0 for x in range(rows)]
        return pl.DataFrame(data)

    # ---------------------------------------------------------------
    # Write benchmarks
    # ---------------------------------------------------------------

    def bench_write_polars(self, rows: int, cols: int) -> BenchResult:
        meta, data = self._make_catalog(f"write_polars_{rows}_{cols}")
        df = self._generate_data(rows, cols)

        start = time.perf_counter()
        write_ducklake(df, meta, "bench", mode="error")
        elapsed = time.perf_counter() - start

        result = BenchResult(
            name=f"write_{rows}r_{cols}c",
            system="polars",
            operation="write",
            rows=rows,
            columns=cols,
            elapsed_s=elapsed,
        )
        self.results.append(result)
        return result

    def bench_write_duckdb(self, rows: int, cols: int) -> BenchResult:
        meta, data = self._make_catalog(f"write_duckdb_{rows}_{cols}")
        con = self._make_duckdb_con(meta, data)

        # Generate via DuckDB SQL
        col_defs = []
        col_exprs = []
        for i in range(cols):
            ctype = i % 4
            if ctype == 0:
                col_defs.append(f"int_{i} BIGINT")
                col_exprs.append(f"i AS int_{i}")
            elif ctype == 1:
                col_defs.append(f"float_{i} DOUBLE")
                col_exprs.append(f"CAST(i AS DOUBLE) * 1.1 AS float_{i}")
            elif ctype == 2:
                col_defs.append(f"str_{i} VARCHAR")
                col_exprs.append(f"'value_' || (i % 1000) AS str_{i}")
            else:
                col_defs.append(f"bool_{i} BOOLEAN")
                col_exprs.append(f"(i % 2 = 0) AS bool_{i}")

        start = time.perf_counter()
        con.execute(
            f"CREATE TABLE ducklake.bench ({', '.join(col_defs)})"
        )
        con.execute(
            f"INSERT INTO ducklake.bench SELECT {', '.join(col_exprs)} "
            f"FROM range({rows}) t(i)"
        )
        elapsed = time.perf_counter() - start
        con.close()

        result = BenchResult(
            name=f"write_{rows}r_{cols}c",
            system="duckdb",
            operation="write",
            rows=rows,
            columns=cols,
            elapsed_s=elapsed,
        )
        self.results.append(result)
        return result

    # ---------------------------------------------------------------
    # Read benchmarks
    # ---------------------------------------------------------------

    def bench_read_polars(self, rows: int, cols: int) -> BenchResult:
        # Write data first with DuckDB
        meta, data = self._make_catalog(f"read_polars_{rows}_{cols}")
        con = self._make_duckdb_con(meta, data)

        col_defs = []
        col_exprs = []
        for i in range(cols):
            ctype = i % 4
            if ctype == 0:
                col_defs.append(f"int_{i} BIGINT")
                col_exprs.append(f"i AS int_{i}")
            elif ctype == 1:
                col_defs.append(f"float_{i} DOUBLE")
                col_exprs.append(f"CAST(i AS DOUBLE) * 1.1 AS float_{i}")
            elif ctype == 2:
                col_defs.append(f"str_{i} VARCHAR")
                col_exprs.append(f"'value_' || (i % 1000) AS str_{i}")
            else:
                col_defs.append(f"bool_{i} BOOLEAN")
                col_exprs.append(f"(i % 2 = 0) AS bool_{i}")

        con.execute(f"CREATE TABLE ducklake.bench ({', '.join(col_defs)})")
        con.execute(
            f"INSERT INTO ducklake.bench SELECT {', '.join(col_exprs)} "
            f"FROM range({rows}) t(i)"
        )
        con.close()

        # Benchmark read
        start = time.perf_counter()
        df = read_ducklake(meta, "bench")
        elapsed = time.perf_counter() - start

        assert df.shape[0] == rows

        result = BenchResult(
            name=f"read_{rows}r_{cols}c",
            system="polars",
            operation="read",
            rows=rows,
            columns=cols,
            elapsed_s=elapsed,
        )
        self.results.append(result)
        return result

    def bench_read_duckdb(self, rows: int, cols: int) -> BenchResult:
        # Write data first with DuckDB
        meta, data = self._make_catalog(f"read_duckdb_{rows}_{cols}")
        con = self._make_duckdb_con(meta, data)

        col_defs = []
        col_exprs = []
        for i in range(cols):
            ctype = i % 4
            if ctype == 0:
                col_defs.append(f"int_{i} BIGINT")
                col_exprs.append(f"i AS int_{i}")
            elif ctype == 1:
                col_defs.append(f"float_{i} DOUBLE")
                col_exprs.append(f"CAST(i AS DOUBLE) * 1.1 AS float_{i}")
            elif ctype == 2:
                col_defs.append(f"str_{i} VARCHAR")
                col_exprs.append(f"'value_' || (i % 1000) AS str_{i}")
            else:
                col_defs.append(f"bool_{i} BOOLEAN")
                col_exprs.append(f"(i % 2 = 0) AS bool_{i}")

        con.execute(f"CREATE TABLE ducklake.bench ({', '.join(col_defs)})")
        con.execute(
            f"INSERT INTO ducklake.bench SELECT {', '.join(col_exprs)} "
            f"FROM range({rows}) t(i)"
        )
        con.close()

        con2 = self._make_duckdb_con(meta, data)
        start = time.perf_counter()
        result_rows = con2.execute("SELECT * FROM ducklake.bench").fetchall()
        elapsed = time.perf_counter() - start
        con2.close()

        assert len(result_rows) == rows

        result = BenchResult(
            name=f"read_{rows}r_{cols}c",
            system="duckdb",
            operation="read",
            rows=rows,
            columns=cols,
            elapsed_s=elapsed,
        )
        self.results.append(result)
        return result

    # ---------------------------------------------------------------
    # Scan + filter benchmarks
    # ---------------------------------------------------------------

    def bench_scan_filter_polars(self, rows: int, cols: int) -> BenchResult:
        meta, data = self._make_catalog(f"scan_polars_{rows}_{cols}")
        con = self._make_duckdb_con(meta, data)

        # Create multi-file table (5 batches)
        batch_size = rows // 5
        con.execute("CREATE TABLE ducklake.bench (id BIGINT, category VARCHAR, value DOUBLE)")
        for b in range(5):
            offset = b * batch_size
            con.execute(
                f"INSERT INTO ducklake.bench "
                f"SELECT i + {offset}, "
                f"CASE WHEN (i + {offset}) % 10 = 0 THEN 'target' ELSE 'other' END, "
                f"CAST(i + {offset} AS DOUBLE) "
                f"FROM range({batch_size}) t(i)"
            )
        con.close()

        start = time.perf_counter()
        lf = scan_ducklake(meta, "bench")
        result_df = lf.filter(pl.col("category") == "target").collect()
        elapsed = time.perf_counter() - start

        result = BenchResult(
            name=f"scan_filter_{rows}r",
            system="polars",
            operation="scan_filter",
            rows=rows,
            columns=3,
            elapsed_s=elapsed,
            notes=f"filtered_rows={result_df.shape[0]}",
        )
        self.results.append(result)
        return result

    def bench_scan_filter_duckdb(self, rows: int, cols: int) -> BenchResult:
        meta, data = self._make_catalog(f"scan_duckdb_{rows}_{cols}")
        con = self._make_duckdb_con(meta, data)

        batch_size = rows // 5
        con.execute("CREATE TABLE ducklake.bench (id BIGINT, category VARCHAR, value DOUBLE)")
        for b in range(5):
            offset = b * batch_size
            con.execute(
                f"INSERT INTO ducklake.bench "
                f"SELECT i + {offset}, "
                f"CASE WHEN (i + {offset}) % 10 = 0 THEN 'target' ELSE 'other' END, "
                f"CAST(i + {offset} AS DOUBLE) "
                f"FROM range({batch_size}) t(i)"
            )
        con.close()

        con2 = self._make_duckdb_con(meta, data)
        start = time.perf_counter()
        filtered = con2.execute(
            "SELECT * FROM ducklake.bench WHERE category = 'target'"
        ).fetchall()
        elapsed = time.perf_counter() - start
        con2.close()

        result = BenchResult(
            name=f"scan_filter_{rows}r",
            system="duckdb",
            operation="scan_filter",
            rows=rows,
            columns=3,
            elapsed_s=elapsed,
            notes=f"filtered_rows={len(filtered)}",
        )
        self.results.append(result)
        return result

    # ---------------------------------------------------------------
    # Aggregation benchmarks
    # ---------------------------------------------------------------

    def bench_scan_agg_polars(self, rows: int) -> BenchResult:
        meta, data = self._make_catalog(f"agg_polars_{rows}")
        con = self._make_duckdb_con(meta, data)
        con.execute(
            "CREATE TABLE ducklake.bench AS "
            f"SELECT i AS id, i % 100 AS group_id, CAST(i AS DOUBLE) * 1.5 AS value "
            f"FROM range({rows}) t(i)"
        )
        con.close()

        start = time.perf_counter()
        lf = scan_ducklake(meta, "bench")
        result_df = (
            lf.group_by("group_id")
            .agg(
                pl.col("value").sum().alias("total"),
                pl.col("value").mean().alias("avg"),
                pl.len().alias("cnt"),
            )
            .sort("group_id")
            .collect()
        )
        elapsed = time.perf_counter() - start

        result = BenchResult(
            name=f"scan_agg_{rows}r",
            system="polars",
            operation="scan_agg",
            rows=rows,
            columns=3,
            elapsed_s=elapsed,
        )
        self.results.append(result)
        return result

    def bench_scan_agg_duckdb(self, rows: int) -> BenchResult:
        meta, data = self._make_catalog(f"agg_duckdb_{rows}")
        con = self._make_duckdb_con(meta, data)
        con.execute(
            "CREATE TABLE ducklake.bench AS "
            f"SELECT i AS id, i % 100 AS group_id, CAST(i AS DOUBLE) * 1.5 AS value "
            f"FROM range({rows}) t(i)"
        )
        con.close()

        con2 = self._make_duckdb_con(meta, data)
        start = time.perf_counter()
        agg_rows = con2.execute(
            "SELECT group_id, SUM(value) AS total, AVG(value) AS avg, COUNT(*) AS cnt "
            "FROM ducklake.bench GROUP BY group_id ORDER BY group_id"
        ).fetchall()
        elapsed = time.perf_counter() - start
        con2.close()

        result = BenchResult(
            name=f"scan_agg_{rows}r",
            system="duckdb",
            operation="scan_agg",
            rows=rows,
            columns=3,
            elapsed_s=elapsed,
        )
        self.results.append(result)
        return result

    # ---------------------------------------------------------------
    # Multi-file read (time travel) benchmarks
    # ---------------------------------------------------------------

    def bench_multifile_read_polars(self, rows: int, num_files: int) -> BenchResult:
        meta, data = self._make_catalog(f"multifile_polars_{rows}_{num_files}")
        con = self._make_duckdb_con(meta, data)

        batch = rows // num_files
        con.execute("CREATE TABLE ducklake.bench (id BIGINT, value DOUBLE)")
        for f in range(num_files):
            con.execute(
                f"INSERT INTO ducklake.bench "
                f"SELECT i + {f * batch}, CAST(i + {f * batch} AS DOUBLE) "
                f"FROM range({batch}) t(i)"
            )
        con.close()

        start = time.perf_counter()
        df = read_ducklake(meta, "bench")
        elapsed = time.perf_counter() - start

        assert df.shape[0] == rows

        result = BenchResult(
            name=f"multifile_{rows}r_{num_files}f",
            system="polars",
            operation="read",
            rows=rows,
            columns=2,
            elapsed_s=elapsed,
            notes=f"files={num_files}",
        )
        self.results.append(result)
        return result

    def bench_multifile_read_duckdb(self, rows: int, num_files: int) -> BenchResult:
        meta, data = self._make_catalog(f"multifile_duckdb_{rows}_{num_files}")
        con = self._make_duckdb_con(meta, data)

        batch = rows // num_files
        con.execute("CREATE TABLE ducklake.bench (id BIGINT, value DOUBLE)")
        for f in range(num_files):
            con.execute(
                f"INSERT INTO ducklake.bench "
                f"SELECT i + {f * batch}, CAST(i + {f * batch} AS DOUBLE) "
                f"FROM range({batch}) t(i)"
            )
        con.close()

        con2 = self._make_duckdb_con(meta, data)
        start = time.perf_counter()
        result_rows = con2.execute("SELECT * FROM ducklake.bench").fetchall()
        elapsed = time.perf_counter() - start
        con2.close()

        assert len(result_rows) == rows

        result = BenchResult(
            name=f"multifile_{rows}r_{num_files}f",
            system="duckdb",
            operation="read",
            rows=rows,
            columns=2,
            elapsed_s=elapsed,
            notes=f"files={num_files}",
        )
        self.results.append(result)
        return result

    # ---------------------------------------------------------------
    # Run all benchmarks
    # ---------------------------------------------------------------

    def run_all(self, rows: int = 100_000, cols: int = 10):
        print(f"\n{'='*70}")
        print(f"DuckLake Benchmark Suite — {rows:,} rows, {cols} columns")
        print(f"{'='*70}\n")

        benchmarks = [
            ("Write", [
                ("polars", lambda: self.bench_write_polars(rows, cols)),
                ("duckdb", lambda: self.bench_write_duckdb(rows, cols)),
            ]),
            ("Read (full table)", [
                ("polars", lambda: self.bench_read_polars(rows, cols)),
                ("duckdb", lambda: self.bench_read_duckdb(rows, cols)),
            ]),
            ("Scan + Filter (5 files)", [
                ("polars", lambda: self.bench_scan_filter_polars(rows, cols)),
                ("duckdb", lambda: self.bench_scan_filter_duckdb(rows, cols)),
            ]),
            ("Scan + Aggregation", [
                ("polars", lambda: self.bench_scan_agg_polars(rows)),
                ("duckdb", lambda: self.bench_scan_agg_duckdb(rows)),
            ]),
            ("Multi-file Read (10 files)", [
                ("polars", lambda: self.bench_multifile_read_polars(rows, 10)),
                ("duckdb", lambda: self.bench_multifile_read_duckdb(rows, 10)),
            ]),
            ("Multi-file Read (50 files)", [
                ("polars", lambda: self.bench_multifile_read_polars(rows, 50)),
                ("duckdb", lambda: self.bench_multifile_read_duckdb(rows, 50)),
            ]),
        ]

        for group_name, bench_list in benchmarks:
            print(f"--- {group_name} ---")
            group_results = []
            for system, fn in bench_list:
                r = fn()
                group_results.append(r)
                print(
                    f"  {system:>8s}: {r.elapsed_s:8.3f}s "
                    f"({r.rows_per_sec:>12,.0f} rows/s)"
                )
            # Ratio
            if len(group_results) == 2:
                p, d = group_results
                if d.elapsed_s > 0:
                    ratio = p.elapsed_s / d.elapsed_s
                    faster = "duckdb" if ratio > 1 else "polars"
                    factor = max(ratio, 1 / ratio)
                    print(f"  {'ratio':>8s}: {faster} is {factor:.1f}x faster")
            print()

    def save_results(self, path: str):
        with open(path, "w") as f:
            json.dump([asdict(r) for r in self.results], f, indent=2)
        print(f"Results saved to {path}")

    def print_summary(self):
        print(f"\n{'='*70}")
        print("Summary")
        print(f"{'='*70}")
        print(f"{'Name':<35s} {'System':<8s} {'Time(s)':<10s} {'Rows/s':<15s}")
        print("-" * 70)
        for r in self.results:
            print(f"{r.name:<35s} {r.system:<8s} {r.elapsed_s:<10.3f} {r.rows_per_sec:<15,.0f}")


def main():
    parser = argparse.ArgumentParser(description="DuckLake Benchmark Suite")
    parser.add_argument("--rows", type=int, default=100_000, help="Number of rows")
    parser.add_argument("--cols", type=int, default=10, help="Number of columns")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    args = parser.parse_args()

    tmpdir = tempfile.mkdtemp(prefix="ducklake_bench_")
    try:
        bench = DuckLakeBenchmark(tmpdir)
        bench.run_all(rows=args.rows, cols=args.cols)
        bench.print_summary()
        if args.output:
            bench.save_results(args.output)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
