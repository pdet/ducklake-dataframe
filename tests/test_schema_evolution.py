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
        result = result.sort("a")
        assert result["name"].to_list() == ["hello", "world"]

    def test_read_after_multiple_renames(self, ducklake_catalog):
        """Rename b -> name -> full_name, verify all data accessible."""
        cat = ducklake_catalog

        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        cat.execute("INSERT INTO ducklake.test VALUES (1, 'alice')")

        cat.execute("ALTER TABLE ducklake.test RENAME COLUMN b TO name")
        cat.execute("INSERT INTO ducklake.test VALUES (2, 'bob')")

        cat.execute("ALTER TABLE ducklake.test RENAME COLUMN name TO full_name")
        cat.execute("INSERT INTO ducklake.test VALUES (3, 'charlie')")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.columns == ["a", "full_name"]
        assert result.shape == (3, 2)
        result = result.sort("a")
        assert result["full_name"].to_list() == ["alice", "bob", "charlie"]

    def test_rename_with_add_column(self, ducklake_catalog):
        """Rename + add column in the same table."""
        cat = ducklake_catalog

        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        cat.execute("INSERT INTO ducklake.test VALUES (1, 'hello')")

        cat.execute("ALTER TABLE ducklake.test RENAME COLUMN b TO name")
        cat.execute("ALTER TABLE ducklake.test ADD COLUMN c DOUBLE")
        cat.execute("INSERT INTO ducklake.test VALUES (2, 'world', 3.14)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.columns == ["a", "name", "c"]
        assert result.shape == (2, 3)
        result = result.sort("a")
        assert result["name"].to_list() == ["hello", "world"]
        assert result["c"].to_list() == [None, 3.14]

    def test_rename_with_filter(self, ducklake_catalog):
        """Verify filter pushdown works after rename."""
        cat = ducklake_catalog

        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        cat.execute("INSERT INTO ducklake.test VALUES (1, 'hello')")

        cat.execute("ALTER TABLE ducklake.test RENAME COLUMN b TO name")
        cat.execute("INSERT INTO ducklake.test VALUES (2, 'world')")
        cat.close()

        lf = scan_ducklake(cat.metadata_path, "test")
        result = lf.filter(pl.col("a") == 2).collect()
        assert result.shape == (1, 2)
        assert result.columns == ["a", "name"]
        assert result["a"].to_list() == [2]
        assert result["name"].to_list() == ["world"]

    def test_rename_with_delete(self, ducklake_catalog):
        """Verify delete files work correctly after rename."""
        cat = ducklake_catalog

        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        cat.execute("INSERT INTO ducklake.test VALUES (1, 'hello'), (2, 'world')")

        cat.execute("ALTER TABLE ducklake.test RENAME COLUMN b TO name")
        cat.execute("DELETE FROM ducklake.test WHERE a = 1")
        cat.execute("INSERT INTO ducklake.test VALUES (3, 'new')")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape == (2, 2)
        result = result.sort("a")
        assert result["a"].to_list() == [2, 3]
        assert result["name"].to_list() == ["world", "new"]

    def test_rename_time_travel(self, ducklake_catalog):
        """Read at snapshot before and after rename."""
        cat = ducklake_catalog

        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        cat.execute("INSERT INTO ducklake.test VALUES (1, 'hello')")

        # Get snapshot before rename
        snap_before = cat.fetchone(
            "SELECT * FROM ducklake_current_snapshot('ducklake')"
        )[0]

        cat.execute("ALTER TABLE ducklake.test RENAME COLUMN b TO name")
        cat.execute("INSERT INTO ducklake.test VALUES (2, 'world')")
        cat.close()

        # Read at snapshot before rename: should have old name
        result_before = read_ducklake(
            cat.metadata_path, "test", snapshot_version=snap_before
        )
        assert result_before.columns == ["a", "b"]
        assert result_before["b"].to_list() == ["hello"]

        # Read latest: should have new name with all data
        result_latest = read_ducklake(cat.metadata_path, "test")
        assert result_latest.columns == ["a", "name"]
        assert sorted(result_latest["name"].to_list()) == ["hello", "world"]

    def test_rename_back_to_original_name(self, ducklake_catalog):
        """Rename b -> name -> b (round-trip), verify no data loss."""
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        cat.execute("INSERT INTO ducklake.test VALUES (1, 'first')")
        cat.execute("ALTER TABLE ducklake.test RENAME COLUMN b TO name")
        cat.execute("INSERT INTO ducklake.test VALUES (2, 'second')")
        cat.execute("ALTER TABLE ducklake.test RENAME COLUMN name TO b")
        cat.execute("INSERT INTO ducklake.test VALUES (3, 'third')")
        cat.close()
        result = read_ducklake(cat.metadata_path, "test")
        assert result.columns == ["a", "b"]
        assert result.shape == (3, 2)
        result = result.sort("a")
        assert result["b"].to_list() == ["first", "second", "third"]

    def test_rename_and_drop_column(self, ducklake_catalog):
        """Rename one column while dropping another."""
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR, c DOUBLE)")
        cat.execute("INSERT INTO ducklake.test VALUES (1, 'hello', 3.14)")
        cat.execute("ALTER TABLE ducklake.test RENAME COLUMN b TO name")
        cat.execute("ALTER TABLE ducklake.test DROP COLUMN c")
        cat.execute("INSERT INTO ducklake.test VALUES (2, 'world')")
        cat.close()
        result = read_ducklake(cat.metadata_path, "test")
        assert result.columns == ["a", "name"]
        assert result.shape == (2, 2)
        result = result.sort("a")
        assert result["name"].to_list() == ["hello", "world"]

    def test_rename_with_filter_on_renamed_column(self, ducklake_catalog):
        """Filter on the renamed column itself."""
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        cat.execute("INSERT INTO ducklake.test VALUES (1, 'hello'), (2, 'world')")
        cat.execute("ALTER TABLE ducklake.test RENAME COLUMN b TO name")
        cat.execute("INSERT INTO ducklake.test VALUES (3, 'new')")
        cat.close()
        lf = scan_ducklake(cat.metadata_path, "test")
        result = lf.filter(pl.col("name") == "hello").collect()
        assert result.shape == (1, 2)
        assert result["a"].to_list() == [1]
        assert result["name"].to_list() == ["hello"]
