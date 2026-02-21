"""Statistics extraction for file pruning in scan_parquet."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import TYPE_CHECKING

import polars as pl

from ducklake_polars._schema import duckdb_type_to_polars

if TYPE_CHECKING:
    from ducklake_polars._catalog import ColumnInfo, ColumnStats, FileInfo


def _parse_stat_value(value: str | None, polars_type: pl.DataType) -> object:
    """Parse a DuckLake stat value string into a Python value."""
    if value is None or value == "NULL":
        return None

    # DuckLake stores stats as SQL string literals
    # Strip surrounding quotes if present (minimum 2 chars for valid quoting)
    if len(value) >= 2 and value.startswith("'") and value.endswith("'"):
        value = value[1:-1].replace("''", "'")

    try:
        base = polars_type.base_type()

        if base in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64):
            return int(value)
        if base in (pl.Float32, pl.Float64):
            return float(value)
        if base == pl.Boolean:
            low = value.lower()
            if low in ("true", "1"):
                return True
            if low in ("false", "0"):
                return False
            return None
        if base == pl.String:
            return value
        if base == pl.Date:
            return date.fromisoformat(value)
        if base == pl.Datetime:
            return datetime.fromisoformat(value)
        if base == pl.Decimal:
            return Decimal(value)
    except (ValueError, TypeError, ArithmeticError):
        return None

    # For unsupported types, return None (skip statistics)
    return None


def build_table_statistics(
    files: list[FileInfo],
    stats: list[ColumnStats],
    columns: list[ColumnInfo],
    filter_columns: list[str] | None = None,
) -> pl.DataFrame | None:
    """
    Build a _table_statistics DataFrame for scan_parquet.

    The format expected by Polars is:
    - 'len' column: UInt32 with record count per file
    - For each filter column: '{col}_nc' (null count), '{col}_min', '{col}_max'

    Files are in the same order as the paths passed to scan_parquet.
    """
    if not files or not stats:
        return None

    # Determine which columns to include stats for
    if filter_columns is not None:
        filter_set = set(filter_columns)
        target_cols = [c for c in columns if c.column_name in filter_set]
    else:
        target_cols = columns

    # Only include columns with simple types that support statistics.
    # Cache the polars type to avoid redundant parsing later.
    stat_cols: list[tuple[ColumnInfo, pl.DataType]] = []
    for col in target_cols:
        try:
            pt = duckdb_type_to_polars(col.column_type)
            base = pt.base_type()
            if base in (
                pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                pl.Float32, pl.Float64,
                pl.Boolean, pl.String, pl.Date, pl.Datetime, pl.Decimal,
            ):
                stat_cols.append((col, pt))
        except ValueError:
            continue

    if not stat_cols:
        return None

    n_files = len(files)

    # Index stats by (data_file_id, column_id)
    stats_lookup: dict[tuple[int, int], ColumnStats] = {}
    for s in stats:
        stats_lookup[(s.data_file_id, s.column_id)] = s

    # Build the statistics DataFrame
    series_list: list[pl.Series] = [
        pl.Series("len", [f.record_count for f in files], dtype=pl.UInt32),
    ]

    for col, polars_type in stat_cols:
        nc_values: list[object] = []
        min_values: list[object] = []
        max_values: list[object] = []

        for file in files:
            key = (file.data_file_id, col.column_id)
            cs = stats_lookup.get(key)
            if cs is not None:
                nc_values.append(cs.null_count if cs.null_count is not None else 0)
                min_values.append(_parse_stat_value(cs.min_value, polars_type))
                max_values.append(_parse_stat_value(cs.max_value, polars_type))
            else:
                nc_values.append(None)
                min_values.append(None)
                max_values.append(None)

        nc_name = f"{col.column_name}_nc"
        min_name = f"{col.column_name}_min"
        max_name = f"{col.column_name}_max"

        series_list.append(pl.Series(nc_name, nc_values, dtype=pl.UInt32))

        try:
            min_series = pl.Series(min_name, min_values, dtype=polars_type)
            max_series = pl.Series(max_name, max_values, dtype=polars_type)
            series_list.append(min_series)
            series_list.append(max_series)
        except (TypeError, ValueError):
            # If casting fails, use null series for both min and max
            series_list.append(
                pl.Series(min_name, [None] * n_files, dtype=polars_type)
            )
            series_list.append(
                pl.Series(max_name, [None] * n_files, dtype=polars_type)
            )

    return pl.DataFrame(series_list)
