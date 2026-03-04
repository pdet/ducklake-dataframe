"""Schema evolution tests for ducklake-dataframe."""

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
        assert result["c"].to_list() == [None, 3.14]


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


class TestTypePromotion:
    """Test reading after column type promotions."""

    def test_tinyint_to_integer_promotion(self, ducklake_catalog):
        cat = ducklake_catalog

        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b TINYINT)")
        cat.execute("INSERT INTO ducklake.test VALUES (1, 25)")

        cat.execute(
            "ALTER TABLE ducklake.test ALTER COLUMN b SET DATA TYPE INTEGER"
        )
        cat.execute("INSERT INTO ducklake.test VALUES (2, 1000)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        result = result.sort("a")
        assert result.shape == (2, 2)
        assert result.schema["b"] == pl.Int32
        assert result["a"].to_list() == [1, 2]
        assert result["b"].to_list() == [25, 1000]

    def test_float_to_double_promotion(self, ducklake_catalog):
        cat = ducklake_catalog

        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b FLOAT)")
        cat.execute("INSERT INTO ducklake.test VALUES (1, 1.5)")

        cat.execute(
            "ALTER TABLE ducklake.test ALTER COLUMN b SET DATA TYPE DOUBLE"
        )
        cat.execute("INSERT INTO ducklake.test VALUES (2, 3.14)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        result = result.sort("a")
        assert result.shape == (2, 2)
        assert result.schema["b"] == pl.Float64
        assert result["a"].to_list() == [1, 2]
        assert result["b"][0] == pytest.approx(1.5)
        assert result["b"][1] == pytest.approx(3.14)

    def test_integer_to_bigint_promotion(self, ducklake_catalog):
        cat = ducklake_catalog

        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b INTEGER)")
        cat.execute("INSERT INTO ducklake.test VALUES (1, 42)")

        cat.execute(
            "ALTER TABLE ducklake.test ALTER COLUMN b SET DATA TYPE BIGINT"
        )
        # Value larger than 2^31 (2147483648)
        cat.execute("INSERT INTO ducklake.test VALUES (2, 3000000000)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        result = result.sort("a")
        assert result.shape == (2, 2)
        assert result.schema["b"] == pl.Int64
        assert result["a"].to_list() == [1, 2]
        assert result["b"].to_list() == [42, 3000000000]


class TestDefaultValues:
    """Test reading after columns are added with default values."""

    def test_add_column_with_default(self, ducklake_catalog):
        cat = ducklake_catalog

        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        cat.execute("INSERT INTO ducklake.test VALUES (1), (2)")

        cat.execute("ALTER TABLE ducklake.test ADD COLUMN b INTEGER DEFAULT 42")
        cat.execute("INSERT INTO ducklake.test VALUES (3, 100)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        result = result.sort("a")
        assert result.shape == (3, 2)
        assert result.schema == {"a": pl.Int32, "b": pl.Int32}
        assert result["a"].to_list() == [1, 2, 3]
        # Old Parquet files don't contain the new column, so old rows get NULL
        # (DuckLake does not backfill defaults into existing Parquet files)
        assert result["b"].to_list() == [None, None, 100]

    def test_add_column_with_string_default(self, ducklake_catalog):
        cat = ducklake_catalog

        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        cat.execute("INSERT INTO ducklake.test VALUES (1), (2)")

        cat.execute(
            "ALTER TABLE ducklake.test ADD COLUMN b VARCHAR DEFAULT 'hello'"
        )
        cat.execute("INSERT INTO ducklake.test VALUES (3, 'world')")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        result = result.sort("a")
        assert result.shape == (3, 2)
        assert result.schema == {"a": pl.Int32, "b": pl.String}
        assert result["a"].to_list() == [1, 2, 3]
        # Old Parquet files don't contain the new column, so old rows get NULL
        # (DuckLake does not backfill defaults into existing Parquet files)
        assert result["b"].to_list() == [None, None, "world"]

    def test_add_column_default_vs_null(self, ducklake_catalog):
        cat = ducklake_catalog

        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        cat.execute("INSERT INTO ducklake.test VALUES (1)")

        # Add column without default (should be NULL for old rows)
        cat.execute("ALTER TABLE ducklake.test ADD COLUMN b VARCHAR")
        # Add column with default (old rows still get NULL -- defaults not backfilled)
        cat.execute("ALTER TABLE ducklake.test ADD COLUMN c INTEGER DEFAULT 0")
        cat.execute("INSERT INTO ducklake.test VALUES (2, 'val', 5)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        result = result.sort("a")
        assert result.shape == (2, 3)
        assert result.schema == {"a": pl.Int32, "b": pl.String, "c": pl.Int32}
        assert result["a"].to_list() == [1, 2]
        # Old row: both b and c are NULL (DuckLake does not backfill defaults
        # into existing Parquet files; missing_columns="insert" fills with NULL)
        assert result["b"].to_list() == [None, "val"]
        assert result["c"].to_list() == [None, 5]


class TestMixedAlter:
    """Test reading after mixed ALTER TABLE operations."""

    def test_drop_and_readd_column_different_type(self, ducklake_catalog):
        cat = ducklake_catalog

        cat.execute(
            "CREATE TABLE ducklake.test (a INTEGER, b INTEGER, c VARCHAR)"
        )
        cat.execute("INSERT INTO ducklake.test VALUES (1, 10, 'first')")

        cat.execute("ALTER TABLE ducklake.test DROP COLUMN b")
        cat.execute("INSERT INTO ducklake.test VALUES (2, 'second')")

        cat.execute("ALTER TABLE ducklake.test ADD COLUMN b VARCHAR")
        cat.execute("INSERT INTO ducklake.test VALUES (3, 'third', 'new_b')")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        result = result.sort("a")
        assert result.shape == (3, 3)
        # b column should be String type (re-added as VARCHAR)
        assert result.schema["b"] == pl.String
        assert result["a"].to_list() == [1, 2, 3]
        assert result["c"].to_list() == ["first", "second", "third"]
        # Old rows (before b was re-added) should have NULL for b
        assert result["b"].to_list() == [None, None, "new_b"]

    def test_rename_and_drop(self, ducklake_catalog):
        """Rename one column and drop another, then insert and read."""
        cat = ducklake_catalog

        cat.execute(
            "CREATE TABLE ducklake.test (a INTEGER, b VARCHAR, c DOUBLE)"
        )
        cat.execute("INSERT INTO ducklake.test VALUES (1, 'hello', 3.14)")

        cat.execute("ALTER TABLE ducklake.test RENAME COLUMN b TO name")
        cat.execute("ALTER TABLE ducklake.test DROP COLUMN c")
        cat.execute("INSERT INTO ducklake.test VALUES (2, 'world')")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        result = result.sort("a")
        assert result.columns == ["a", "name"]
        assert result.shape == (2, 2)
        assert result["a"].to_list() == [1, 2]
        assert result["name"].to_list() == ["hello", "world"]


class TestTableRename:
    """Test reading after table rename."""

    def test_read_after_table_rename(self, ducklake_catalog):
        cat = ducklake_catalog

        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        cat.execute(
            "INSERT INTO ducklake.test VALUES (1, 'hello'), (2, 'world')"
        )

        cat.execute("ALTER TABLE ducklake.test RENAME TO test2")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test2")
        result = result.sort("a")
        assert result.shape == (2, 2)
        assert result.schema == {"a": pl.Int32, "b": pl.String}
        assert result["a"].to_list() == [1, 2]
        assert result["b"].to_list() == ["hello", "world"]

    def test_table_rename_old_name_fails(self, ducklake_catalog):
        cat = ducklake_catalog

        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        cat.execute("INSERT INTO ducklake.test VALUES (1, 'hello')")

        cat.execute("ALTER TABLE ducklake.test RENAME TO test2")
        cat.close()

        with pytest.raises(ValueError, match="test"):
            read_ducklake(cat.metadata_path, "test")


class TestStructEvolution:
    """Test reading after struct column evolution."""

    def test_struct_add_field(self, ducklake_catalog):
        cat = ducklake_catalog

        cat.execute(
            "CREATE TABLE ducklake.test "
            "(a INTEGER, col STRUCT(i INTEGER, j INTEGER))"
        )
        cat.execute(
            "INSERT INTO ducklake.test VALUES (1, {i: 10, j: 20})"
        )

        cat.execute(
            "ALTER TABLE ducklake.test ALTER COLUMN col "
            "SET DATA TYPE STRUCT(i INTEGER, j INTEGER, k INTEGER)"
        )
        cat.execute(
            "INSERT INTO ducklake.test VALUES (2, {i: 30, j: 40, k: 50})"
        )
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        result = result.sort("a")
        assert result.shape == (2, 2)
        structs = result["col"].to_list()
        # Old row should have k=None, new row should have k=50
        assert structs[0] == {"i": 10, "j": 20, "k": None}
        assert structs[1] == {"i": 30, "j": 40, "k": 50}

    def test_struct_drop_field(self, ducklake_catalog):
        cat = ducklake_catalog

        cat.execute(
            "CREATE TABLE ducklake.test "
            "(a INTEGER, col STRUCT(i INTEGER, j INTEGER, k INTEGER))"
        )
        cat.execute(
            "INSERT INTO ducklake.test VALUES (1, {i: 10, j: 20, k: 30})"
        )

        cat.execute(
            "ALTER TABLE ducklake.test ALTER COLUMN col "
            "SET DATA TYPE STRUCT(i INTEGER, k INTEGER)"
        )
        cat.execute(
            "INSERT INTO ducklake.test VALUES (2, {i: 40, k: 50})"
        )
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        result = result.sort("a")
        assert result.shape == (2, 2)
        structs = result["col"].to_list()
        # All rows should have only i and k fields
        assert structs[0] == {"i": 10, "k": 30}
        assert structs[1] == {"i": 40, "k": 50}

    def test_struct_rename_field(self, ducklake_catalog):
        cat = ducklake_catalog

        cat.execute(
            "CREATE TABLE ducklake.test "
            "(a INTEGER, col STRUCT(i INTEGER, j INTEGER))"
        )
        cat.execute(
            "INSERT INTO ducklake.test VALUES (1, {i: 10, j: 20})"
        )

        cat.execute(
            "ALTER TABLE ducklake.test ALTER COLUMN col "
            "SET DATA TYPE STRUCT(i INTEGER, val INTEGER)"
        )
        cat.execute(
            "INSERT INTO ducklake.test VALUES (2, {i: 30, val: 40})"
        )
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        result = result.sort("a")
        assert result.shape == (2, 2)
        structs = result["col"].to_list()
        # Both old and new data should be accessible under field name 'val'
        assert structs[0] == {"i": 10, "val": 20}
        assert structs[1] == {"i": 30, "val": 40}


# -----------------------------------------------------------------------
# Comprehensive schema evolution edge case tests
# -----------------------------------------------------------------------


class TestSchemaEvolutionEdgeCases:
    """Comprehensive edge-case tests for schema evolution using SQLite backend."""

    # 1. Add column with default, insert data, read — new column has defaults
    #    for old rows (DuckLake does NOT backfill defaults into existing Parquet)
    def test_add_column_with_default_old_rows_get_null(
        self, ducklake_catalog_sqlite
    ):
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.test (id INTEGER, val VARCHAR)")
        cat.execute(
            "INSERT INTO ducklake.test VALUES (1, 'a'), (2, 'b'), (3, 'c')"
        )
        cat.execute(
            "ALTER TABLE ducklake.test ADD COLUMN score INTEGER DEFAULT 99"
        )
        cat.execute("INSERT INTO ducklake.test VALUES (4, 'd', 100)")
        cat.execute("INSERT INTO ducklake.test VALUES (5, 'e', NULL)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        result = result.sort("id")
        assert result.shape == (5, 3)
        assert result.schema == {
            "id": pl.Int32,
            "val": pl.String,
            "score": pl.Int32,
        }
        # Old rows get NULL (not the default), new rows get their explicit values
        assert result["score"].to_list() == [None, None, None, 100, None]

    # 2. Drop column, insert data, read — dropped column gone
    def test_drop_column_completely_gone(self, ducklake_catalog_sqlite):
        cat = ducklake_catalog_sqlite
        cat.execute(
            "CREATE TABLE ducklake.test "
            "(id INTEGER, name VARCHAR, age INTEGER, city VARCHAR)"
        )
        cat.execute(
            "INSERT INTO ducklake.test VALUES "
            "(1, 'alice', 30, 'NYC'), (2, 'bob', 25, 'LA')"
        )
        cat.execute("ALTER TABLE ducklake.test DROP COLUMN age")
        cat.execute("ALTER TABLE ducklake.test DROP COLUMN city")
        cat.execute("INSERT INTO ducklake.test VALUES (3, 'charlie')")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.columns == ["id", "name"]
        assert result.shape == (3, 2)
        result = result.sort("id")
        assert result["id"].to_list() == [1, 2, 3]
        assert result["name"].to_list() == ["alice", "bob", "charlie"]

    # 3. Rename column, verify old data accessible under new name
    def test_rename_column_old_data_under_new_name(
        self, ducklake_catalog_sqlite
    ):
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.test (id INTEGER, old_name VARCHAR)")
        cat.execute(
            "INSERT INTO ducklake.test VALUES (1, 'before_rename_1')"
        )
        cat.execute(
            "INSERT INTO ducklake.test VALUES (2, 'before_rename_2')"
        )
        cat.execute(
            "ALTER TABLE ducklake.test RENAME COLUMN old_name TO new_name"
        )
        cat.execute(
            "INSERT INTO ducklake.test VALUES (3, 'after_rename')"
        )
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert "old_name" not in result.columns
        assert "new_name" in result.columns
        result = result.sort("id")
        assert result["new_name"].to_list() == [
            "before_rename_1",
            "before_rename_2",
            "after_rename",
        ]

    # 4. Change column type (INTEGER → BIGINT), verify data preserved
    def test_change_column_type_int_to_bigint(self, ducklake_catalog_sqlite):
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.test (id INTEGER, amount INTEGER)")
        cat.execute("INSERT INTO ducklake.test VALUES (1, 42), (2, 2147483647)")
        cat.execute(
            "ALTER TABLE ducklake.test ALTER COLUMN amount SET DATA TYPE BIGINT"
        )
        # Insert a value > INT32 max to prove BIGINT works
        cat.execute("INSERT INTO ducklake.test VALUES (3, 5000000000)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        result = result.sort("id")
        assert result.schema["amount"] == pl.Int64
        assert result["amount"].to_list() == [42, 2147483647, 5000000000]

    # 5. Add column, drop it, add column with same name but different type
    def test_add_drop_readd_same_name_different_type(
        self, ducklake_catalog_sqlite
    ):
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.test (id INTEGER)")
        cat.execute("INSERT INTO ducklake.test VALUES (1)")

        # Add 'extra' as INTEGER
        cat.execute("ALTER TABLE ducklake.test ADD COLUMN extra INTEGER")
        cat.execute("INSERT INTO ducklake.test VALUES (2, 42)")

        # Drop 'extra'
        cat.execute("ALTER TABLE ducklake.test DROP COLUMN extra")
        cat.execute("INSERT INTO ducklake.test VALUES (3)")

        # Re-add 'extra' as VARCHAR
        cat.execute("ALTER TABLE ducklake.test ADD COLUMN extra VARCHAR")
        cat.execute("INSERT INTO ducklake.test VALUES (4, 'hello')")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        result = result.sort("id")
        assert result.schema["extra"] == pl.String
        assert result.columns == ["id", "extra"]
        assert result["id"].to_list() == [1, 2, 3, 4]
        # Rows before re-add have NULL for the new 'extra'
        assert result["extra"].to_list() == [None, None, None, "hello"]

    # 6. Multiple ALTER operations in sequence: add A, rename A→B, add new A
    def test_multiple_alter_add_rename_add(self, ducklake_catalog_sqlite):
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.test (id INTEGER)")
        cat.execute("INSERT INTO ducklake.test VALUES (1)")

        # Add column A
        cat.execute("ALTER TABLE ducklake.test ADD COLUMN a VARCHAR")
        cat.execute("INSERT INTO ducklake.test VALUES (2, 'first_a')")

        # Rename A → B
        cat.execute("ALTER TABLE ducklake.test RENAME COLUMN a TO b")
        cat.execute("INSERT INTO ducklake.test VALUES (3, 'now_b')")

        # Add new column A (different from the original A which is now B)
        cat.execute("ALTER TABLE ducklake.test ADD COLUMN a INTEGER")
        cat.execute("INSERT INTO ducklake.test VALUES (4, 'still_b', 999)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        result = result.sort("id")
        assert set(result.columns) == {"id", "b", "a"}
        assert result.schema["b"] == pl.String
        assert result.schema["a"] == pl.Int32
        assert result["id"].to_list() == [1, 2, 3, 4]
        assert result["b"].to_list() == [None, "first_a", "now_b", "still_b"]
        assert result["a"].to_list() == [None, None, None, 999]

    # 7. Schema evolution + partition pruning still works
    def test_schema_evolution_with_partition_pruning(
        self, ducklake_catalog_sqlite
    ):
        cat = ducklake_catalog_sqlite
        cat.execute(
            "CREATE TABLE ducklake.test (id INTEGER, category VARCHAR)"
        )
        cat.execute("ALTER TABLE ducklake.test SET PARTITIONED BY (category)")
        cat.execute(
            "INSERT INTO ducklake.test VALUES "
            "(1, 'A'), (2, 'B'), (3, 'A')"
        )

        # Add a new column after partitioned data exists
        cat.execute("ALTER TABLE ducklake.test ADD COLUMN value DOUBLE")
        cat.execute(
            "INSERT INTO ducklake.test VALUES "
            "(4, 'A', 1.1), (5, 'B', 2.2), (6, 'C', 3.3)"
        )
        cat.close()

        # Filter on partition column — should still prune correctly
        lf = scan_ducklake(cat.metadata_path, "test")
        result_a = lf.filter(pl.col("category") == "A").collect().sort("id")
        assert result_a["id"].to_list() == [1, 3, 4]
        assert result_a["value"].to_list() == [None, None, 1.1]

        result_c = (
            scan_ducklake(cat.metadata_path, "test")
            .filter(pl.col("category") == "C")
            .collect()
            .sort("id")
        )
        assert result_c["id"].to_list() == [6]
        assert result_c["value"].to_list() == [3.3]

    # 8. Schema evolution + time travel (read at old snapshot sees old schema)
    def test_schema_evolution_time_travel(self, ducklake_catalog_sqlite):
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.test (id INTEGER, val VARCHAR)")
        cat.execute("INSERT INTO ducklake.test VALUES (1, 'original')")
        snap_v1 = cat.fetchone(
            "SELECT * FROM ducklake_current_snapshot('ducklake')"
        )[0]

        # Add column
        cat.execute("ALTER TABLE ducklake.test ADD COLUMN extra INTEGER")
        cat.execute("INSERT INTO ducklake.test VALUES (2, 'with_extra', 42)")
        snap_v2 = cat.fetchone(
            "SELECT * FROM ducklake_current_snapshot('ducklake')"
        )[0]

        # Rename column
        cat.execute("ALTER TABLE ducklake.test RENAME COLUMN val TO name")
        cat.execute("INSERT INTO ducklake.test VALUES (3, 'renamed', 99)")
        snap_v3 = cat.fetchone(
            "SELECT * FROM ducklake_current_snapshot('ducklake')"
        )[0]

        # Type change
        cat.execute(
            "ALTER TABLE ducklake.test ALTER COLUMN extra SET DATA TYPE BIGINT"
        )
        cat.execute(
            "INSERT INTO ducklake.test VALUES (4, 'bigint', 5000000000)"
        )
        cat.close()

        # v1: old schema {id, val}, 1 row
        r1 = read_ducklake(
            cat.metadata_path, "test", snapshot_version=snap_v1
        )
        assert r1.columns == ["id", "val"]
        assert r1.shape == (1, 2)
        assert r1["val"].to_list() == ["original"]

        # v2: schema {id, val, extra}, 2 rows
        r2 = read_ducklake(
            cat.metadata_path, "test", snapshot_version=snap_v2
        )
        assert r2.columns == ["id", "val", "extra"]
        assert r2.shape == (2, 3)
        r2 = r2.sort("id")
        assert r2["val"].to_list() == ["original", "with_extra"]
        assert r2["extra"].to_list() == [None, 42]

        # v3: schema {id, name, extra} (val renamed to name), 3 rows
        r3 = read_ducklake(
            cat.metadata_path, "test", snapshot_version=snap_v3
        )
        assert r3.columns == ["id", "name", "extra"]
        assert r3.shape == (3, 3)
        r3 = r3.sort("id")
        assert r3["name"].to_list() == ["original", "with_extra", "renamed"]

        # Latest: schema {id, name, extra(BIGINT)}, 4 rows
        r_latest = read_ducklake(cat.metadata_path, "test")
        assert r_latest.columns == ["id", "name", "extra"]
        assert r_latest.schema["extra"] == pl.Int64
        assert r_latest.shape == (4, 3)
        r_latest = r_latest.sort("id")
        assert r_latest["extra"].to_list() == [None, 42, 99, 5000000000]

    # 9. Schema evolution across multiple inserts (insert, alter, insert, read)
    def test_schema_evolution_across_multiple_inserts(
        self, ducklake_catalog_sqlite
    ):
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.test (id INTEGER)")

        # Phase 1: just id
        cat.execute("INSERT INTO ducklake.test VALUES (1), (2)")

        # Phase 2: add col_a
        cat.execute("ALTER TABLE ducklake.test ADD COLUMN col_a VARCHAR")
        cat.execute("INSERT INTO ducklake.test VALUES (3, 'a3'), (4, 'a4')")

        # Phase 3: add col_b, rename col_a → col_x
        cat.execute("ALTER TABLE ducklake.test ADD COLUMN col_b DOUBLE")
        cat.execute("ALTER TABLE ducklake.test RENAME COLUMN col_a TO col_x")
        cat.execute(
            "INSERT INTO ducklake.test VALUES (5, 'x5', 5.5), (6, 'x6', 6.6)"
        )

        # Phase 4: type promotion on col_b
        cat.execute(
            "ALTER TABLE ducklake.test "
            "ALTER COLUMN col_b SET DATA TYPE DOUBLE"
        )
        cat.execute("INSERT INTO ducklake.test VALUES (7, 'x7', 7.777)")

        # Phase 5: add col_c
        cat.execute("ALTER TABLE ducklake.test ADD COLUMN col_c BOOLEAN")
        cat.execute(
            "INSERT INTO ducklake.test VALUES (8, 'x8', 8.8, true)"
        )
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        result = result.sort("id")
        assert result.shape == (8, 4)
        assert result.columns == ["id", "col_x", "col_b", "col_c"]

        assert result["id"].to_list() == [1, 2, 3, 4, 5, 6, 7, 8]
        # Phase 1 rows: col_x=NULL, col_b=NULL, col_c=NULL
        assert result["col_x"][0] is None
        assert result["col_x"][1] is None
        # Phase 2 rows: col_x has values, col_b=NULL, col_c=NULL
        assert result["col_x"][2] == "a3"
        assert result["col_x"][3] == "a4"
        assert result["col_b"][2] is None
        # Phase 3 rows: col_x and col_b have values, col_c=NULL
        assert result["col_x"][4] == "x5"
        assert result["col_b"][4] == 5.5
        assert result["col_c"][4] is None
        # Phase 5 row: all columns populated
        assert result["col_x"][7] == "x8"
        assert result["col_b"][7] == 8.8
        assert result["col_c"][7] is True

    # 10. Add many columns (10+), verify wide table reads correctly
    def test_wide_table_many_columns(self, ducklake_catalog_sqlite):
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.test (id INTEGER)")
        cat.execute("INSERT INTO ducklake.test VALUES (1)")

        # Add 15 columns of varying types
        col_defs = [
            ("c_int", "INTEGER"),
            ("c_bigint", "BIGINT"),
            ("c_float", "FLOAT"),
            ("c_double", "DOUBLE"),
            ("c_bool", "BOOLEAN"),
            ("c_varchar", "VARCHAR"),
            ("c_smallint", "SMALLINT"),
            ("c_tinyint", "TINYINT"),
            ("c_date", "DATE"),
            ("c_int2", "INTEGER"),
            ("c_varchar2", "VARCHAR"),
            ("c_double2", "DOUBLE"),
            ("c_bool2", "BOOLEAN"),
            ("c_bigint2", "BIGINT"),
            ("c_varchar3", "VARCHAR"),
        ]
        for col_name, col_type in col_defs:
            cat.execute(
                f"ALTER TABLE ducklake.test ADD COLUMN {col_name} {col_type}"
            )

        # Insert a fully populated row
        cat.execute(
            "INSERT INTO ducklake.test VALUES ("
            "2, 10, 20, 1.5, 2.5, true, 'hello', 3, 4, "
            "'2025-01-15', 50, 'world', 9.9, false, 100, 'end'"
            ")"
        )

        # Insert row with NULLs in some columns
        cat.execute(
            "INSERT INTO ducklake.test VALUES ("
            "3, NULL, 30, NULL, 3.3, NULL, 'partial', NULL, NULL, "
            "NULL, 60, NULL, NULL, true, NULL, NULL"
            ")"
        )
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape == (3, 16)  # id + 15 columns
        assert len(result.columns) == 16

        result = result.sort("id")
        # Row 1 (pre-alter): all added columns are NULL
        row1 = result.row(0, named=True)
        assert row1["id"] == 1
        for col_name, _ in col_defs:
            assert row1[col_name] is None

        # Row 2: fully populated
        row2 = result.row(1, named=True)
        assert row2["id"] == 2
        assert row2["c_int"] == 10
        assert row2["c_bigint"] == 20
        assert row2["c_varchar"] == "hello"
        assert row2["c_bool"] is True
        assert row2["c_varchar3"] == "end"

        # Row 3: partial NULLs
        row3 = result.row(2, named=True)
        assert row3["id"] == 3
        assert row3["c_int"] is None
        assert row3["c_bigint"] == 30
        assert row3["c_varchar"] == "partial"
        assert row3["c_bool2"] is True

    # Extra edge case: drop column then filter on remaining columns
    def test_drop_column_filter_still_works(self, ducklake_catalog_sqlite):
        cat = ducklake_catalog_sqlite
        cat.execute(
            "CREATE TABLE ducklake.test "
            "(id INTEGER, drop_me VARCHAR, keep_me INTEGER)"
        )
        cat.execute(
            "INSERT INTO ducklake.test VALUES "
            "(1, 'gone', 10), (2, 'gone', 20), (3, 'gone', 30)"
        )
        cat.execute("ALTER TABLE ducklake.test DROP COLUMN drop_me")
        cat.execute("INSERT INTO ducklake.test VALUES (4, 40), (5, 50)")
        cat.close()

        lf = scan_ducklake(cat.metadata_path, "test")
        result = lf.filter(pl.col("keep_me") > 25).collect().sort("id")
        assert result.columns == ["id", "keep_me"]
        assert result["id"].to_list() == [3, 4, 5]
        assert result["keep_me"].to_list() == [30, 40, 50]

    # Extra edge case: type promotion + rename in same evolution chain
    def test_type_promotion_and_rename_combined(
        self, ducklake_catalog_sqlite
    ):
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.test (id INTEGER, val SMALLINT)")
        cat.execute("INSERT INTO ducklake.test VALUES (1, 100)")

        cat.execute(
            "ALTER TABLE ducklake.test ALTER COLUMN val SET DATA TYPE BIGINT"
        )
        cat.execute("ALTER TABLE ducklake.test RENAME COLUMN val TO big_val")
        cat.execute("INSERT INTO ducklake.test VALUES (2, 9999999999)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        result = result.sort("id")
        assert result.columns == ["id", "big_val"]
        assert result.schema["big_val"] == pl.Int64
        assert result["big_val"].to_list() == [100, 9999999999]

    # Extra edge case: partition pruning after column rename
    def test_partition_pruning_after_rename(self, ducklake_catalog_sqlite):
        cat = ducklake_catalog_sqlite
        cat.execute(
            "CREATE TABLE ducklake.test (id INTEGER, part_col VARCHAR, data INTEGER)"
        )
        cat.execute("ALTER TABLE ducklake.test SET PARTITIONED BY (part_col)")
        cat.execute(
            "INSERT INTO ducklake.test VALUES "
            "(1, 'X', 10), (2, 'Y', 20), (3, 'X', 30)"
        )
        cat.execute(
            "ALTER TABLE ducklake.test RENAME COLUMN part_col TO category"
        )
        cat.execute(
            "INSERT INTO ducklake.test VALUES (4, 'X', 40), (5, 'Z', 50)"
        )
        cat.close()

        lf = scan_ducklake(cat.metadata_path, "test")
        result = lf.filter(pl.col("category") == "X").collect().sort("id")
        assert result["id"].to_list() == [1, 3, 4]
        assert result["data"].to_list() == [10, 30, 40]
