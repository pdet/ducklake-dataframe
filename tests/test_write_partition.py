"""Tests for partitioned writes in ducklake-polars."""

from __future__ import annotations

import os
import sqlite3

import duckdb
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from ducklake_polars import (
    alter_ducklake_set_partitioned_by,
    create_ducklake_table,
    read_ducklake,
    scan_ducklake,
    write_ducklake,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_catalog(tmp_path):
    """Create a DuckLake catalog via DuckDB and return (metadata_path, data_path)."""
    metadata_path = str(tmp_path / "part_test.ducklake")
    data_path = str(tmp_path / "data")
    os.makedirs(data_path, exist_ok=True)

    con = duckdb.connect()
    con.install_extension("ducklake")
    con.load_extension("ducklake")
    con.execute(
        f"ATTACH 'ducklake:sqlite:{metadata_path}' AS ducklake "
        f"(DATA_PATH '{data_path}', DATA_INLINING_ROW_LIMIT 0)"
    )
    con.close()
    return metadata_path, data_path


def _read_with_duckdb(metadata_path, data_path, table_name):
    """Read a table with DuckDB's DuckLake extension."""
    con = duckdb.connect()
    con.install_extension("ducklake")
    con.load_extension("ducklake")
    con.execute(
        f"ATTACH 'ducklake:sqlite:{metadata_path}' AS ducklake "
        f"(DATA_PATH '{data_path}', DATA_INLINING_ROW_LIMIT 0)"
    )
    cursor = con.execute(f'SELECT * FROM ducklake."{table_name}"')
    columns = [desc[0] for desc in cursor.description]
    rows = cursor.fetchall()
    con.close()
    if not rows:
        return pl.DataFrame({c: [] for c in columns})
    data = {c: [r[i] for r in rows] for i, c in enumerate(columns)}
    return pl.DataFrame(data)


# ---------------------------------------------------------------------------
# ALTER TABLE SET PARTITIONED BY
# ---------------------------------------------------------------------------


class TestSetPartitionedBy:
    """Test alter_ducklake_set_partitioned_by."""

    def test_set_partitioned_by_single_column(self, tmp_path):
        """Set partitioning on a single column."""
        metadata_path, data_path = _make_catalog(tmp_path)
        create_ducklake_table(
            metadata_path, "test", {"a": pl.Int64(), "b": pl.String()},
        )

        alter_ducklake_set_partitioned_by(metadata_path, "test", ["b"])

        # Check metadata
        con = sqlite3.connect(metadata_path)
        pi = con.execute("SELECT partition_id, table_id FROM ducklake_partition_info").fetchone()
        assert pi is not None
        partition_id = pi[0]

        pc = con.execute(
            "SELECT partition_key_index, column_id, transform "
            "FROM ducklake_partition_column WHERE partition_id = ?",
            [partition_id],
        ).fetchall()
        assert len(pc) == 1
        assert pc[0][0] == 0  # partition_key_index
        assert pc[0][2] == "identity"

        # Schema version bumped
        change = con.execute(
            "SELECT changes_made FROM ducklake_snapshot_changes "
            "ORDER BY snapshot_id DESC LIMIT 1"
        ).fetchone()
        assert "altered_table" in change[0]
        con.close()

    def test_set_partitioned_by_multi_column(self, tmp_path):
        """Set partitioning on multiple columns."""
        metadata_path, data_path = _make_catalog(tmp_path)
        create_ducklake_table(
            metadata_path, "test",
            {"a": pl.Int64(), "b": pl.String(), "c": pl.Int64()},
        )

        alter_ducklake_set_partitioned_by(metadata_path, "test", ["b", "c"])

        con = sqlite3.connect(metadata_path)
        pcs = con.execute(
            "SELECT partition_key_index, transform "
            "FROM ducklake_partition_column ORDER BY partition_key_index"
        ).fetchall()
        assert len(pcs) == 2
        assert pcs[0][0] == 0
        assert pcs[1][0] == 1
        assert pcs[0][1] == "identity"
        assert pcs[1][1] == "identity"
        con.close()

    def test_set_partitioned_by_nonexistent_table(self, tmp_path):
        metadata_path, data_path = _make_catalog(tmp_path)
        with pytest.raises(ValueError, match="not found"):
            alter_ducklake_set_partitioned_by(metadata_path, "missing", ["a"])

    def test_set_partitioned_by_nonexistent_column(self, tmp_path):
        metadata_path, data_path = _make_catalog(tmp_path)
        create_ducklake_table(
            metadata_path, "test", {"a": pl.Int64()},
        )
        with pytest.raises(ValueError, match="not found"):
            alter_ducklake_set_partitioned_by(metadata_path, "test", ["missing"])

    def test_set_partitioned_by_duckdb_reads(self, tmp_path):
        """DuckDB can read the partition spec created by ducklake-polars."""
        metadata_path, data_path = _make_catalog(tmp_path)
        df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "x"]})
        write_ducklake(df, metadata_path, "test", mode="error")

        alter_ducklake_set_partitioned_by(metadata_path, "test", ["b"])

        # Insert after partitioning
        df2 = pl.DataFrame({"a": [4, 5], "b": ["z", "x"]})
        write_ducklake(df2, metadata_path, "test", mode="append")

        pdf = _read_with_duckdb(metadata_path, data_path, "test")
        assert len(pdf) == 5
        assert sorted(pdf["a"].to_list()) == [1, 2, 3, 4, 5]


# ---------------------------------------------------------------------------
# Partitioned INSERT
# ---------------------------------------------------------------------------


class TestPartitionedInsert:
    """Test writing data to partitioned tables."""

    def test_basic_partitioned_insert(self, tmp_path):
        """Insert into a partitioned table creates per-partition files."""
        metadata_path, data_path = _make_catalog(tmp_path)
        schema = {"a": pl.Int32(), "b": pl.String()}
        create_ducklake_table(
            metadata_path, "test", {"a": pl.Int64(), "b": pl.String()},
        )
        alter_ducklake_set_partitioned_by(metadata_path, "test", ["b"])

        df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "x"]})
        write_ducklake(df, metadata_path, "test", mode="append")

        result = read_ducklake(metadata_path, "test").sort("a")
        assert result["a"].to_list() == [1, 2, 3]
        assert result["b"].to_list() == ["x", "y", "x"]

    def test_partitioned_insert_hive_paths(self, tmp_path):
        """Verify Hive-style directory layout."""
        metadata_path, data_path = _make_catalog(tmp_path)
        create_ducklake_table(
            metadata_path, "test", {"a": pl.Int64(), "b": pl.String()},
        )
        alter_ducklake_set_partitioned_by(metadata_path, "test", ["b"])

        df = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        write_ducklake(df, metadata_path, "test", mode="append")

        # Check data file paths in catalog
        con = sqlite3.connect(metadata_path)
        paths = con.execute(
            "SELECT path FROM ducklake_data_file WHERE partition_id IS NOT NULL"
        ).fetchall()
        con.close()

        path_strs = [p[0] for p in paths]
        assert any("b=x/" in p for p in path_strs)
        assert any("b=y/" in p for p in path_strs)

    def test_partitioned_insert_multi_column(self, tmp_path):
        """Multi-column partition produces nested Hive paths."""
        metadata_path, data_path = _make_catalog(tmp_path)
        create_ducklake_table(
            metadata_path, "test",
            {"a": pl.Int64(), "b": pl.String(), "c": pl.Int64()},
        )
        alter_ducklake_set_partitioned_by(metadata_path, "test", ["b", "c"])

        df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "x"], "c": [10, 20, 10]})
        write_ducklake(df, metadata_path, "test", mode="append")

        result = read_ducklake(metadata_path, "test").sort("a")
        assert result["a"].to_list() == [1, 2, 3]
        assert result["b"].to_list() == ["x", "y", "x"]
        assert result["c"].to_list() == [10, 20, 10]

        # Check nested paths
        con = sqlite3.connect(metadata_path)
        paths = con.execute(
            "SELECT path FROM ducklake_data_file WHERE partition_id IS NOT NULL"
        ).fetchall()
        con.close()

        path_strs = [p[0] for p in paths]
        assert any("b=x/c=10/" in p for p in path_strs)
        assert any("b=y/c=20/" in p for p in path_strs)

    def test_partitioned_insert_partition_values_registered(self, tmp_path):
        """Partition values are correctly registered in ducklake_file_partition_value."""
        metadata_path, data_path = _make_catalog(tmp_path)
        create_ducklake_table(
            metadata_path, "test", {"a": pl.Int64(), "b": pl.String()},
        )
        alter_ducklake_set_partitioned_by(metadata_path, "test", ["b"])

        df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "x"]})
        write_ducklake(df, metadata_path, "test", mode="append")

        con = sqlite3.connect(metadata_path)
        pvs = con.execute(
            "SELECT data_file_id, partition_key_index, partition_value "
            "FROM ducklake_file_partition_value ORDER BY data_file_id"
        ).fetchall()
        con.close()

        # Should have one entry per partition file
        assert len(pvs) >= 2
        values = {pv[2] for pv in pvs}
        assert "x" in values
        assert "y" in values

    def test_partitioned_insert_partition_id_on_files(self, tmp_path):
        """Data files reference the correct partition_id."""
        metadata_path, data_path = _make_catalog(tmp_path)
        create_ducklake_table(
            metadata_path, "test", {"a": pl.Int64(), "b": pl.String()},
        )
        alter_ducklake_set_partitioned_by(metadata_path, "test", ["b"])

        df = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        write_ducklake(df, metadata_path, "test", mode="append")

        con = sqlite3.connect(metadata_path)
        pi = con.execute(
            "SELECT partition_id FROM ducklake_partition_info"
        ).fetchone()
        dfs = con.execute(
            "SELECT partition_id FROM ducklake_data_file WHERE partition_id IS NOT NULL"
        ).fetchall()
        con.close()

        # All partitioned files should reference the same partition_id
        for (part_id,) in dfs:
            assert part_id == pi[0]

    def test_partitioned_insert_column_stats(self, tmp_path):
        """Column statistics are computed per-partition file."""
        metadata_path, data_path = _make_catalog(tmp_path)
        create_ducklake_table(
            metadata_path, "test", {"a": pl.Int64(), "b": pl.String()},
        )
        alter_ducklake_set_partitioned_by(metadata_path, "test", ["b"])

        df = pl.DataFrame({"a": [1, 5, 3], "b": ["x", "x", "y"]})
        write_ducklake(df, metadata_path, "test", mode="append")

        con = sqlite3.connect(metadata_path)
        # Stats for partition b=x: min=1, max=5
        # Stats for partition b=y: min=3, max=3
        stats = con.execute(
            "SELECT data_file_id, column_id, min_value, max_value "
            "FROM ducklake_file_column_stats "
            "WHERE column_id = (SELECT column_id FROM ducklake_column WHERE column_name='a' LIMIT 1) "
            "ORDER BY data_file_id"
        ).fetchall()
        con.close()

        assert len(stats) >= 2
        mins = {s[2] for s in stats}
        maxs = {s[3] for s in stats}
        assert "1" in mins
        assert "3" in mins
        assert "5" in maxs
        assert "3" in maxs

    def test_partitioned_insert_multiple_batches(self, tmp_path):
        """Multiple inserts into a partitioned table append correctly."""
        metadata_path, data_path = _make_catalog(tmp_path)
        create_ducklake_table(
            metadata_path, "test", {"a": pl.Int64(), "b": pl.String()},
        )
        alter_ducklake_set_partitioned_by(metadata_path, "test", ["b"])

        write_ducklake(
            pl.DataFrame({"a": [1, 2], "b": ["x", "y"]}),
            metadata_path, "test", mode="append",
        )
        write_ducklake(
            pl.DataFrame({"a": [3, 4], "b": ["x", "z"]}),
            metadata_path, "test", mode="append",
        )

        result = read_ducklake(metadata_path, "test").sort("a")
        assert result["a"].to_list() == [1, 2, 3, 4]
        assert result["b"].to_list() == ["x", "y", "x", "z"]

    def test_partitioned_insert_integer_partition(self, tmp_path):
        """Partition on an integer column."""
        metadata_path, data_path = _make_catalog(tmp_path)
        create_ducklake_table(
            metadata_path, "test", {"a": pl.Int64(), "b": pl.Int64()},
        )
        alter_ducklake_set_partitioned_by(metadata_path, "test", ["b"])

        df = pl.DataFrame({"a": [1, 2, 3], "b": [100, 200, 100]})
        write_ducklake(df, metadata_path, "test", mode="append")

        result = read_ducklake(metadata_path, "test").sort("a")
        assert result["a"].to_list() == [1, 2, 3]
        assert result["b"].to_list() == [100, 200, 100]


# ---------------------------------------------------------------------------
# Partitioned INSERT with filter
# ---------------------------------------------------------------------------


class TestPartitionedFilter:
    """Test filtering on partition columns after partitioned writes."""

    def test_filter_on_partition_column(self, tmp_path):
        """Filter on partition column returns correct subset."""
        metadata_path, data_path = _make_catalog(tmp_path)
        create_ducklake_table(
            metadata_path, "test", {"a": pl.Int64(), "b": pl.String()},
        )
        alter_ducklake_set_partitioned_by(metadata_path, "test", ["b"])

        df = pl.DataFrame({"a": [1, 2, 3, 4], "b": ["x", "y", "x", "y"]})
        write_ducklake(df, metadata_path, "test", mode="append")

        lf = scan_ducklake(metadata_path, "test")
        result = lf.filter(pl.col("b") == "x").collect().sort("a")
        assert result["a"].to_list() == [1, 3]
        assert result["b"].to_list() == ["x", "x"]

    def test_filter_on_non_partition_column(self, tmp_path):
        """Filter on non-partition column in partitioned table."""
        metadata_path, data_path = _make_catalog(tmp_path)
        create_ducklake_table(
            metadata_path, "test", {"a": pl.Int64(), "b": pl.String()},
        )
        alter_ducklake_set_partitioned_by(metadata_path, "test", ["b"])

        df = pl.DataFrame({"a": [1, 5, 3, 2], "b": ["x", "x", "y", "y"]})
        write_ducklake(df, metadata_path, "test", mode="append")

        lf = scan_ducklake(metadata_path, "test")
        result = lf.filter(pl.col("a") > 2).collect().sort("a")
        assert result["a"].to_list() == [3, 5]


# ---------------------------------------------------------------------------
# Partitioned OVERWRITE
# ---------------------------------------------------------------------------


class TestPartitionedOverwrite:
    """Test overwrite on partitioned tables."""

    def test_overwrite_partitioned(self, tmp_path):
        """Overwrite partitioned table replaces all data."""
        metadata_path, data_path = _make_catalog(tmp_path)
        create_ducklake_table(
            metadata_path, "test", {"a": pl.Int64(), "b": pl.String()},
        )
        alter_ducklake_set_partitioned_by(metadata_path, "test", ["b"])

        write_ducklake(
            pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "x"]}),
            metadata_path, "test", mode="append",
        )
        write_ducklake(
            pl.DataFrame({"a": [10, 20], "b": ["z", "w"]}),
            metadata_path, "test", mode="overwrite",
        )

        result = read_ducklake(metadata_path, "test").sort("a")
        assert result["a"].to_list() == [10, 20]
        assert result["b"].to_list() == ["z", "w"]


# ---------------------------------------------------------------------------
# DuckDB interop
# ---------------------------------------------------------------------------


class TestPartitionedDuckDBInterop:
    """DuckDB can read partitioned data written by ducklake-polars."""

    def test_duckdb_reads_polars_partitioned_insert(self, tmp_path):
        """DuckDB reads data written by ducklake-polars into partitioned table."""
        metadata_path, data_path = _make_catalog(tmp_path)
        create_ducklake_table(
            metadata_path, "test", {"a": pl.Int64(), "b": pl.String()},
        )
        alter_ducklake_set_partitioned_by(metadata_path, "test", ["b"])

        df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "x"]})
        write_ducklake(df, metadata_path, "test", mode="append")

        pdf = _read_with_duckdb(metadata_path, data_path, "test")
        assert len(pdf) == 3
        assert sorted(pdf["a"].to_list()) == [1, 2, 3]
        assert sorted(pdf["b"].to_list()) == ["x", "x", "y"]

    def test_duckdb_reads_polars_multi_partition(self, tmp_path):
        """DuckDB reads multi-column partitioned data."""
        metadata_path, data_path = _make_catalog(tmp_path)
        create_ducklake_table(
            metadata_path, "test",
            {"a": pl.Int64(), "b": pl.String(), "c": pl.Int64()},
        )
        alter_ducklake_set_partitioned_by(metadata_path, "test", ["b", "c"])

        df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "x"], "c": [10, 20, 10]})
        write_ducklake(df, metadata_path, "test", mode="append")

        pdf = _read_with_duckdb(metadata_path, data_path, "test")
        assert len(pdf) == 3
        assert sorted(pdf["a"].to_list()) == [1, 2, 3]

    def test_duckdb_writes_polars_reads_partitioned(self, tmp_path):
        """DuckDB creates partitioned table, ducklake-polars writes to it."""
        metadata_path = str(tmp_path / "test.ducklake")
        data_path = str(tmp_path / "data")
        os.makedirs(data_path, exist_ok=True)

        con = duckdb.connect()
        con.install_extension("ducklake")
        con.load_extension("ducklake")
        con.execute(
            f"ATTACH 'ducklake:sqlite:{metadata_path}' AS ducklake "
            f"(DATA_PATH '{data_path}', DATA_INLINING_ROW_LIMIT 0)"
        )
        con.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        con.execute("ALTER TABLE ducklake.test SET PARTITIONED BY (b)")
        con.close()

        # Insert via ducklake-polars (cast to Int32 to match DuckDB's INTEGER)
        df = pl.DataFrame({"a": pl.Series([1, 2, 3], dtype=pl.Int32), "b": ["x", "y", "x"]})
        write_ducklake(df, metadata_path, "test", mode="append")

        result = read_ducklake(metadata_path, "test").sort("a")
        assert result["a"].to_list() == [1, 2, 3]
        assert result["b"].to_list() == ["x", "y", "x"]

    def test_duckdb_insert_then_polars_insert_partitioned(self, tmp_path):
        """DuckDB inserts data, then ducklake-polars appends to same partitioned table."""
        metadata_path = str(tmp_path / "test.ducklake")
        data_path = str(tmp_path / "data")
        os.makedirs(data_path, exist_ok=True)

        con = duckdb.connect()
        con.install_extension("ducklake")
        con.load_extension("ducklake")
        con.execute(
            f"ATTACH 'ducklake:sqlite:{metadata_path}' AS ducklake "
            f"(DATA_PATH '{data_path}', DATA_INLINING_ROW_LIMIT 0)"
        )
        con.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        con.execute("ALTER TABLE ducklake.test SET PARTITIONED BY (b)")
        con.execute("INSERT INTO ducklake.test VALUES (1, 'x'), (2, 'y')")
        con.close()

        # Append via ducklake-polars (cast to Int32 to match DuckDB's INTEGER)
        df = pl.DataFrame({"a": pl.Series([3, 4], dtype=pl.Int32), "b": ["x", "z"]})
        write_ducklake(df, metadata_path, "test", mode="append")

        result = read_ducklake(metadata_path, "test").sort("a")
        assert result["a"].to_list() == [1, 2, 3, 4]
        assert result["b"].to_list() == ["x", "y", "x", "z"]


# ---------------------------------------------------------------------------
# Round-trip tests
# ---------------------------------------------------------------------------


class TestPartitionedRoundTrip:
    """Write and read partitioned data with ducklake-polars."""

    def test_roundtrip_basic(self, tmp_path):
        metadata_path, data_path = _make_catalog(tmp_path)
        df = pl.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "category": ["a", "b", "a", "b", "c"],
            "value": [10.0, 20.0, 30.0, 40.0, 50.0],
        })

        write_ducklake(df, metadata_path, "test", mode="error")
        alter_ducklake_set_partitioned_by(metadata_path, "test", ["category"])

        # Insert more data
        df2 = pl.DataFrame({
            "id": [6, 7],
            "category": ["a", "d"],
            "value": [60.0, 70.0],
        })
        write_ducklake(df2, metadata_path, "test", mode="append")

        result = read_ducklake(metadata_path, "test").sort("id")
        assert result["id"].to_list() == [1, 2, 3, 4, 5, 6, 7]
        assert result["category"].to_list() == ["a", "b", "a", "b", "c", "a", "d"]
        assert result["value"].to_list() == [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]

    def test_roundtrip_time_travel(self, tmp_path):
        """Time travel works with partitioned writes."""
        metadata_path, data_path = _make_catalog(tmp_path)
        create_ducklake_table(
            metadata_path, "test", {"a": pl.Int64(), "b": pl.String()},
        )
        alter_ducklake_set_partitioned_by(metadata_path, "test", ["b"])

        write_ducklake(
            pl.DataFrame({"a": [1, 2], "b": ["x", "y"]}),
            metadata_path, "test", mode="append",
        )

        con = sqlite3.connect(metadata_path)
        snap_after_first = con.execute(
            "SELECT MAX(snapshot_id) FROM ducklake_snapshot"
        ).fetchone()[0]
        con.close()

        write_ducklake(
            pl.DataFrame({"a": [3, 4], "b": ["x", "z"]}),
            metadata_path, "test", mode="append",
        )

        # Latest: 4 rows
        result = read_ducklake(metadata_path, "test")
        assert result.shape[0] == 4

        # At first insert: 2 rows
        result_old = read_ducklake(
            metadata_path, "test", snapshot_version=snap_after_first
        )
        assert result_old.shape[0] == 2
        result_old = result_old.sort("a")
        assert result_old["a"].to_list() == [1, 2]


# ---------------------------------------------------------------------------
# DELETE on partitioned table (written by polars)
# ---------------------------------------------------------------------------


class TestPartitionedDelete:
    """Test DELETE on partitioned tables written by ducklake-polars."""

    def test_delete_from_partitioned(self, tmp_path):
        from ducklake_polars import delete_ducklake

        metadata_path, data_path = _make_catalog(tmp_path)
        create_ducklake_table(
            metadata_path, "test", {"a": pl.Int64(), "b": pl.String()},
        )
        alter_ducklake_set_partitioned_by(metadata_path, "test", ["b"])

        df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "x"]})
        write_ducklake(df, metadata_path, "test", mode="append")

        deleted = delete_ducklake(metadata_path, "test", pl.col("a") == 1)
        assert deleted == 1

        result = read_ducklake(metadata_path, "test").sort("a")
        assert result["a"].to_list() == [2, 3]
        assert result["b"].to_list() == ["y", "x"]


# ---------------------------------------------------------------------------
# UPDATE on partitioned table (written by polars)
# ---------------------------------------------------------------------------


class TestPartitionedUpdate:
    """Test UPDATE on partitioned tables written by ducklake-polars."""

    def test_update_non_partition_column(self, tmp_path):
        from ducklake_polars import update_ducklake

        metadata_path, data_path = _make_catalog(tmp_path)
        create_ducklake_table(
            metadata_path, "test", {"a": pl.Int64(), "b": pl.String()},
        )
        alter_ducklake_set_partitioned_by(metadata_path, "test", ["b"])

        df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "x"]})
        write_ducklake(df, metadata_path, "test", mode="append")

        updated = update_ducklake(
            metadata_path, "test", {"a": 10}, pl.col("a") == 1
        )
        assert updated == 1

        result = read_ducklake(metadata_path, "test").sort("a")
        assert result["a"].to_list() == [2, 3, 10]
        assert result["b"].to_list() == ["y", "x", "x"]


# ---------------------------------------------------------------------------
# write_ducklake mode='error' with partition
# ---------------------------------------------------------------------------


class TestWriteModeWithPartition:
    """Test write_ducklake modes combined with partitioning."""

    def test_mode_error_then_partition_then_append(self, tmp_path):
        """Create with mode='error', partition, then append."""
        metadata_path, data_path = _make_catalog(tmp_path)

        df1 = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        write_ducklake(df1, metadata_path, "test", mode="error")
        alter_ducklake_set_partitioned_by(metadata_path, "test", ["b"])

        df2 = pl.DataFrame({"a": [3, 4], "b": ["x", "z"]})
        write_ducklake(df2, metadata_path, "test", mode="append")

        result = read_ducklake(metadata_path, "test").sort("a")
        assert result["a"].to_list() == [1, 2, 3, 4]
        assert result["b"].to_list() == ["x", "y", "x", "z"]

    def test_mode_overwrite_on_partitioned(self, tmp_path):
        """mode='overwrite' on partitioned table."""
        metadata_path, data_path = _make_catalog(tmp_path)

        df1 = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "x"]})
        write_ducklake(df1, metadata_path, "test", mode="error")
        alter_ducklake_set_partitioned_by(metadata_path, "test", ["b"])

        # Overwrite with different partition values
        df2 = pl.DataFrame({"a": [10, 20], "b": ["p", "q"]})
        write_ducklake(df2, metadata_path, "test", mode="overwrite")

        result = read_ducklake(metadata_path, "test").sort("a")
        assert result["a"].to_list() == [10, 20]
        assert result["b"].to_list() == ["p", "q"]
