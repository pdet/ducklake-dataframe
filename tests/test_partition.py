"""Partition pruning tests for ducklake-polars."""

from __future__ import annotations

import polars as pl
import pytest

from ducklake_polars import read_ducklake, scan_ducklake


class TestPartition:
    """Test reading partitioned tables."""

    def test_read_partitioned_table(self, ducklake_catalog):
        """Basic read of a partitioned table."""
        cat = ducklake_catalog

        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        cat.execute("ALTER TABLE ducklake.test SET PARTITIONED BY (b)")
        cat.execute("INSERT INTO ducklake.test VALUES (1, 'x'), (2, 'y'), (3, 'x')")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape == (3, 2)
        result = result.sort("a")
        assert result["a"].to_list() == [1, 2, 3]
        assert result["b"].to_list() == ["x", "y", "x"]

    def test_partitioned_filter(self, ducklake_catalog):
        """Filter on partition column, verify correct data returned."""
        cat = ducklake_catalog

        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        cat.execute("ALTER TABLE ducklake.test SET PARTITIONED BY (b)")
        cat.execute("INSERT INTO ducklake.test VALUES (1, 'x'), (2, 'y'), (3, 'x')")
        cat.close()

        lf = scan_ducklake(cat.metadata_path, "test")
        result = lf.filter(pl.col("b") == "x").collect()
        assert result.shape[0] == 2
        result = result.sort("a")
        assert result["a"].to_list() == [1, 3]
        assert result["b"].to_list() == ["x", "x"]

    def test_partitioned_multi_column(self, ducklake_catalog):
        """Multiple partition columns."""
        cat = ducklake_catalog

        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR, c INTEGER)")
        cat.execute("ALTER TABLE ducklake.test SET PARTITIONED BY (b, c)")
        cat.execute(
            "INSERT INTO ducklake.test VALUES "
            "(1, 'x', 10), (2, 'y', 20), (3, 'x', 10), (4, 'y', 30)"
        )
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape == (4, 3)
        result = result.sort("a")
        assert result["a"].to_list() == [1, 2, 3, 4]
        assert result["b"].to_list() == ["x", "y", "x", "y"]
        assert result["c"].to_list() == [10, 20, 10, 30]

    def test_partitioned_with_stats(self, ducklake_catalog):
        """Verify statistics-based filtering works on partitioned data."""
        cat = ducklake_catalog

        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        cat.execute("ALTER TABLE ducklake.test SET PARTITIONED BY (b)")
        # Insert in separate batches to create multiple files
        cat.execute("INSERT INTO ducklake.test VALUES (1, 'x'), (2, 'x')")
        cat.execute("INSERT INTO ducklake.test VALUES (3, 'y'), (4, 'y')")
        cat.close()

        lf = scan_ducklake(cat.metadata_path, "test")
        result = lf.filter(pl.col("a") > 2).collect()
        result = result.sort("a")
        assert result["a"].to_list() == [3, 4]
        assert result["b"].to_list() == ["y", "y"]

    def test_partitioned_integer_partition_column(self, ducklake_catalog):
        """Partition on an integer column."""
        cat = ducklake_catalog

        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b INTEGER)")
        cat.execute("ALTER TABLE ducklake.test SET PARTITIONED BY (b)")
        cat.execute("INSERT INTO ducklake.test VALUES (1, 100), (2, 200), (3, 100)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape == (3, 2)
        result = result.sort("a")
        assert result["a"].to_list() == [1, 2, 3]
        assert result["b"].to_list() == [100, 200, 100]

    def test_empty_partitioned_table(self, ducklake_catalog):
        """Read an empty partitioned table."""
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        cat.execute("ALTER TABLE ducklake.test SET PARTITIONED BY (b)")
        cat.close()
        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape == (0, 2)
        assert result.columns == ["a", "b"]

    def test_partitioned_with_delete(self, ducklake_catalog):
        """Delete rows from a partitioned table."""
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        cat.execute("ALTER TABLE ducklake.test SET PARTITIONED BY (b)")
        cat.execute("INSERT INTO ducklake.test VALUES (1, 'x'), (2, 'y'), (3, 'x')")
        cat.execute("DELETE FROM ducklake.test WHERE a = 1")
        cat.close()
        result = read_ducklake(cat.metadata_path, "test")
        result = result.sort("a")
        assert result["a"].to_list() == [2, 3]
        assert result["b"].to_list() == ["y", "x"]
