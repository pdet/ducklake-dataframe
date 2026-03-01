"""Basic tests for ducklake-pandas read operations."""

from __future__ import annotations

import pandas as pd
import numpy as np
import pytest
from pandas.testing import assert_frame_equal

from ducklake_pandas import read_ducklake


class TestBasicRead:
    """Test basic read_ducklake functionality."""

    def test_read_returns_dataframe(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        cat.execute("INSERT INTO ducklake.test VALUES (1, 'hello'), (2, 'world')")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert isinstance(result, pd.DataFrame)

    def test_basic_read(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        cat.execute("INSERT INTO ducklake.test VALUES (1, 'hello'), (2, 'world')")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        result = result.sort_values("a").reset_index(drop=True)
        expected = pd.DataFrame({"a": pd.array([1, 2], dtype="int32"), "b": ["hello", "world"]})
        assert_frame_equal(result, expected, check_dtype=False)

    def test_read_with_column_selection(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR, c DOUBLE)")
        cat.execute("INSERT INTO ducklake.test VALUES (1, 'hello', 3.14)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test", columns=["a", "c"])
        assert list(result.columns) == ["a", "c"]
        assert result["a"].tolist() == [1]
        assert result["c"].tolist() == [3.14]

    def test_empty_table(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape == (0, 2)
        assert list(result.columns) == ["a", "b"]

    def test_table_not_found(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        cat.close()

        with pytest.raises(ValueError, match="not found"):
            read_ducklake(cat.metadata_path, "nonexistent")

    def test_multiple_inserts(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        cat.execute("INSERT INTO ducklake.test VALUES (1, 'first')")
        cat.execute("INSERT INTO ducklake.test VALUES (2, 'second')")
        cat.execute("INSERT INTO ducklake.test VALUES (3, 'third')")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape[0] == 3
        assert sorted(result["a"].tolist()) == [1, 2, 3]

    def test_create_table_as_select(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute(
            "CREATE TABLE ducklake.test AS SELECT i AS a, i * 2 AS b FROM range(10) t(i)"
        )
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape == (10, 2)
        result = result.sort_values("a").reset_index(drop=True)
        assert result["a"].tolist() == list(range(10))
        assert result["b"].tolist() == [i * 2 for i in range(10)]

    def test_null_values(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        cat.execute("INSERT INTO ducklake.test VALUES (1, NULL), (NULL, 'hello'), (NULL, NULL)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape == (3, 2)

    def test_large_insert(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute(
            "CREATE TABLE ducklake.test AS SELECT i AS a, 'val_' || i AS b FROM range(10000) t(i)"
        )
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape == (10000, 2)
        assert result["a"].min() == 0
        assert result["a"].max() == 9999


class TestInlinedData:
    """Test reading tables with inlined data."""

    def test_read_inlined_data(self, ducklake_catalog_inline):
        cat = ducklake_catalog_inline
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        cat.execute("INSERT INTO ducklake.test VALUES (1, 'hello'), (2, 'world')")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape[0] == 2
        result = result.sort_values("a").reset_index(drop=True)
        assert result["a"].tolist() == [1, 2]
        assert result["b"].tolist() == ["hello", "world"]

    def test_read_inlined_empty_table(self, ducklake_catalog_inline):
        cat = ducklake_catalog_inline
        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape == (0, 1)


class TestEdgeCases:
    """Test edge cases and validation."""

    def test_snapshot_mutual_exclusivity(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        cat.execute("INSERT INTO ducklake.test VALUES (1)")
        cat.close()

        with pytest.raises(ValueError, match="Cannot specify both"):
            read_ducklake(
                cat.metadata_path, "test",
                snapshot_version=1,
                snapshot_time="2025-01-01T00:00:00",
            )

    def test_pathlib_path(self, ducklake_catalog):
        from pathlib import Path

        cat = ducklake_catalog
        if cat.backend == "postgres":
            pytest.skip("Path objects not applicable for PostgreSQL connection strings")

        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        cat.execute("INSERT INTO ducklake.test VALUES (1)")
        cat.close()

        result = read_ducklake(Path(cat.metadata_path), "test")
        assert result.shape == (1, 1)
        assert result["a"].tolist() == [1]

    def test_data_path_override(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        cat.execute("INSERT INTO ducklake.test VALUES (42)")
        cat.close()

        # Reading with the original data_path should work
        result = read_ducklake(cat.metadata_path, "test", data_path=cat.data_path)
        assert result["a"].tolist() == [42]

    def test_data_path_override_wrong_path(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        cat.execute("INSERT INTO ducklake.test VALUES (1)")
        cat.close()

        with pytest.raises(Exception):
            read_ducklake(cat.metadata_path, "test", data_path="/nonexistent/path")

    def test_read_with_nonexistent_column(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        cat.execute("INSERT INTO ducklake.test VALUES (1)")
        cat.close()

        with pytest.raises(Exception):
            read_ducklake(cat.metadata_path, "test", columns=["nonexistent"])


class TestPredicate:
    """Test read_ducklake with predicate parameter."""

    def test_read_with_predicate(self, ducklake_catalog):
        """Basic predicate usage filters rows."""
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        cat.execute("INSERT INTO ducklake.test VALUES (1, 'x'), (2, 'y'), (3, 'x'), (4, 'z')")
        cat.close()

        result = read_ducklake(
            cat.metadata_path, "test",
            predicate=lambda df: df["a"] > 2,
        )
        result = result.sort_values("a").reset_index(drop=True)
        assert result["a"].tolist() == [3, 4]
        assert result["b"].tolist() == ["x", "z"]

    def test_read_with_predicate_string_filter(self, ducklake_catalog):
        """Predicate filtering on string column."""
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        cat.execute("INSERT INTO ducklake.test VALUES (1, 'hello'), (2, 'world'), (3, 'hello')")
        cat.close()

        result = read_ducklake(
            cat.metadata_path, "test",
            predicate=lambda df: df["b"] == "hello",
        )
        result = result.sort_values("a").reset_index(drop=True)
        assert result["a"].tolist() == [1, 3]
        assert result["b"].tolist() == ["hello", "hello"]

    def test_read_with_predicate_no_match(self, ducklake_catalog):
        """Predicate that matches nothing returns empty DataFrame."""
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        cat.execute("INSERT INTO ducklake.test VALUES (1), (2), (3)")
        cat.close()

        result = read_ducklake(
            cat.metadata_path, "test",
            predicate=lambda df: df["a"] > 100,
        )
        assert result.shape[0] == 0

    def test_read_with_predicate_and_columns(self, ducklake_catalog):
        """Predicate combined with column selection."""
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR, c DOUBLE)")
        cat.execute("INSERT INTO ducklake.test VALUES (1, 'x', 1.0), (2, 'y', 2.0), (3, 'x', 3.0)")
        cat.close()

        result = read_ducklake(
            cat.metadata_path, "test",
            predicate=lambda df: df["a"] >= 2,
            columns=["a", "c"],
        )
        result = result.sort_values("a").reset_index(drop=True)
        assert list(result.columns) == ["a", "c"]
        assert result["a"].tolist() == [2, 3]

    def test_read_with_predicate_none_is_noop(self, ducklake_catalog):
        """predicate=None returns all rows (default behavior)."""
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        cat.execute("INSERT INTO ducklake.test VALUES (1), (2), (3)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test", predicate=None)
        assert result.shape[0] == 3


class TestMultipleTables:
    """Test reading from multiple tables."""

    def test_read_different_tables(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.t1 (a INTEGER)")
        cat.execute("INSERT INTO ducklake.t1 VALUES (1), (2)")
        cat.execute("CREATE TABLE ducklake.t2 (x VARCHAR, y DOUBLE)")
        cat.execute("INSERT INTO ducklake.t2 VALUES ('hello', 3.14)")
        cat.close()

        r1 = read_ducklake(cat.metadata_path, "t1")
        r2 = read_ducklake(cat.metadata_path, "t2")

        assert r1.shape == (2, 1)
        assert r2.shape == (1, 2)
        assert sorted(r1["a"].tolist()) == [1, 2]
        assert r2["x"].tolist() == ["hello"]
