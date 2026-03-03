"""Constraint tests — parity with ducklake-ref.

Tests NOT NULL constraints, read-only behavior, and general catalog operations.
"""
from __future__ import annotations

import polars as pl
import pytest

from ducklake_polars import read_ducklake


class TestNotNullConstraint:
    """Test NOT NULL constraint behavior."""

    def test_not_null_column_rejects_null_insert(self, ducklake_catalog_sqlite):
        """Insert NULL into NOT NULL column via DuckDB — should fail."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.t (a INTEGER NOT NULL, b VARCHAR)")
        cat.execute("INSERT INTO ducklake.t VALUES (1, 'x')")

        with pytest.raises(Exception):
            cat.execute("INSERT INTO ducklake.t VALUES (NULL, 'y')")
        cat.close()

        result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        assert result.shape[0] == 1

    def test_not_null_with_valid_data(self, ducklake_catalog_sqlite):
        """NOT NULL column with valid data should work fine."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.t (a INTEGER NOT NULL, b VARCHAR NOT NULL)")
        cat.execute("INSERT INTO ducklake.t VALUES (1, 'x'), (2, 'y'), (3, 'z')")
        cat.close()

        result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        assert result.shape[0] == 3
        assert result["a"].null_count() == 0
        assert result["b"].null_count() == 0

    def test_drop_not_null_column(self, ducklake_catalog_sqlite):
        """Drop a NOT NULL column — remaining columns unaffected."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.t (a INTEGER NOT NULL, b VARCHAR, c DOUBLE)")
        cat.execute("INSERT INTO ducklake.t VALUES (1, 'x', 1.0)")
        cat.execute("ALTER TABLE ducklake.t DROP COLUMN a")
        cat.execute("INSERT INTO ducklake.t VALUES ('y', 2.0)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        assert result.shape[0] == 2
        assert "a" not in result.columns


class TestGeneralCatalog:
    """General catalog operation tests — parity with ducklake-ref."""

    def test_multiple_schemas(self, ducklake_catalog_sqlite):
        """Create tables in different schemas."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE SCHEMA ducklake.s1")
        cat.execute("CREATE SCHEMA ducklake.s2")
        cat.execute("CREATE TABLE ducklake.s1.t (a INTEGER)")
        cat.execute("CREATE TABLE ducklake.s2.t (b VARCHAR)")
        cat.execute("INSERT INTO ducklake.s1.t VALUES (1)")
        cat.execute("INSERT INTO ducklake.s2.t VALUES ('hello')")
        cat.close()

        r1 = read_ducklake(cat.metadata_path, "t", schema="s1", data_path=cat.data_path)
        r2 = read_ducklake(cat.metadata_path, "t", schema="s2", data_path=cat.data_path)
        assert r1.shape[0] == 1
        assert r1.columns == ["a"]
        assert r2.shape[0] == 1
        assert r2.columns == ["b"]

    def test_quoted_identifiers(self, ducklake_catalog_sqlite):
        """Column names with special characters."""
        cat = ducklake_catalog_sqlite
        cat.execute('CREATE TABLE ducklake.t ("my col" INTEGER, "Hello World" VARCHAR)')
        cat.execute("INSERT INTO ducklake.t VALUES (1, 'x')")
        cat.close()

        result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        assert result.shape[0] == 1
        assert "my col" in result.columns
        assert "Hello World" in result.columns

    def test_empty_table_read(self, ducklake_catalog_sqlite):
        """Read an empty table — should return empty DataFrame with schema."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.t (a INTEGER, b VARCHAR, c DOUBLE)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        assert result.shape[0] == 0
        assert len(result.columns) == 3

    def test_large_column_count(self, ducklake_catalog_sqlite):
        """Table with many columns."""
        cat = ducklake_catalog_sqlite
        cols = ", ".join(f"c{i} INTEGER" for i in range(50))
        cat.execute(f"CREATE TABLE ducklake.t ({cols})")
        vals = ", ".join(str(i) for i in range(50))
        cat.execute(f"INSERT INTO ducklake.t VALUES ({vals})")
        cat.close()

        result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        assert result.shape == (1, 50)

    def test_drop_recreate_different_schema(self, ducklake_catalog_sqlite):
        """Drop table and recreate with completely different schema."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.t (a INTEGER, b VARCHAR)")
        cat.execute("INSERT INTO ducklake.t VALUES (1, 'x')")
        cat.execute("DROP TABLE ducklake.t")
        cat.execute("CREATE TABLE ducklake.t (x DOUBLE, y DOUBLE, z DOUBLE)")
        cat.execute("INSERT INTO ducklake.t VALUES (1.0, 2.0, 3.0)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        assert result.columns == ["x", "y", "z"]
        assert result.shape[0] == 1

    def test_table_name_stored_as_given(self, ducklake_catalog_sqlite):
        """DuckDB stores unquoted names lowercased; reader does exact match."""
        cat = ducklake_catalog_sqlite
        # DuckDB lowercases unquoted identifiers in SQL
        cat.execute("CREATE TABLE ducklake.my_table (a INTEGER)")
        cat.execute("INSERT INTO ducklake.my_table VALUES (1)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "my_table", data_path=cat.data_path)
        assert result.shape[0] == 1

    def test_projection_pushdown(self, ducklake_catalog_sqlite):
        """Read only specific columns."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.t (a INTEGER, b VARCHAR, c DOUBLE)")
        cat.execute("INSERT INTO ducklake.t VALUES (1, 'x', 1.5)")
        cat.close()

        result = read_ducklake(
            cat.metadata_path, "t", data_path=cat.data_path, columns=["a", "c"]
        )
        assert result.columns == ["a", "c"]
        assert result.shape[0] == 1
