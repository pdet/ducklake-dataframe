"""Schema evolution tests for ducklake-polars."""

from __future__ import annotations

import polars as pl
import pytest

from ducklake_polars import read_ducklake, scan_ducklake


class TestAddColumn:
    """Test reading after columns are added."""

    def test_read_after_add_column(self, ducklake_catalog):
        cat = ducklake_catalog

        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        cat.execute("INSERT INTO ducklake.test VALUES (1), (2)")

        cat.execute("ALTER TABLE ducklake.test ADD COLUMN b VARCHAR")
        cat.execute("INSERT INTO ducklake.test VALUES (3, 'hello')")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape == (3, 2)
        assert result.schema == {"a": pl.Int32, "b": pl.String}
        # Old rows should have NULL for the new column
        result = result.sort("a")
        assert result.filter(pl.col("a") <= 2)["b"].to_list() == [None, None]
        # New row should have the value
        assert result.filter(pl.col("a") == 3)["b"].to_list() == ["hello"]

    def test_read_after_add_multiple_columns(self, ducklake_catalog):
        cat = ducklake_catalog

        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        cat.execute("INSERT INTO ducklake.test VALUES (1)")

        cat.execute("ALTER TABLE ducklake.test ADD COLUMN b VARCHAR")
        cat.execute("ALTER TABLE ducklake.test ADD COLUMN c DOUBLE")
        cat.execute("INSERT INTO ducklake.test VALUES (2, 'hello', 3.14)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape == (2, 3)
        assert result.schema == {"a": pl.Int32, "b": pl.String, "c": pl.Float64}
        result = result.sort("a")
        assert result["a"].to_list() == [1, 2]
        assert result["b"].to_list() == [None, "hello"]


class TestDropColumn:
    """Test reading after columns are dropped."""

    def test_read_after_drop_column(self, ducklake_catalog):
        cat = ducklake_catalog

        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR, c DOUBLE)")
        cat.execute("INSERT INTO ducklake.test VALUES (1, 'hello', 3.14)")

        cat.execute("ALTER TABLE ducklake.test DROP COLUMN b")
        cat.execute("INSERT INTO ducklake.test VALUES (2, 2.72)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.columns == ["a", "c"]
        assert result.shape == (2, 2)
        result = result.sort("a")
        assert result["a"].to_list() == [1, 2]
        assert result["c"].to_list() == [3.14, 2.72]


class TestRenameColumn:
    """Test reading after columns are renamed."""

    @pytest.mark.xfail(reason="Column mapping for renames not yet implemented (Phase 2)")
    def test_read_after_rename_column(self, ducklake_catalog):
        cat = ducklake_catalog

        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        cat.execute("INSERT INTO ducklake.test VALUES (1, 'hello')")

        cat.execute("ALTER TABLE ducklake.test RENAME COLUMN b TO name")
        cat.execute("INSERT INTO ducklake.test VALUES (2, 'world')")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.columns == ["a", "name"]
        assert result.shape == (2, 2)
        # Both old and new data should be accessible under the new name
        assert sorted(result["name"].to_list()) == ["hello", "world"]
