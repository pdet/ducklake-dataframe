"""Tests for DuckLake schema listing."""

from __future__ import annotations

import pytest
import duckdb
import ducklake_polars
import ducklake_pandas


@pytest.fixture
def catalog_with_schemas(tmp_path):
    """Create a DuckLake catalog with multiple schemas."""
    metadata_path = str(tmp_path / "test.ducklake")
    data_path = str(tmp_path / "data")

    conn = duckdb.connect()
    conn.install_extension("ducklake")
    conn.load_extension("ducklake")
    conn.install_extension("sqlite_scanner")
    conn.load_extension("sqlite_scanner")
    conn.execute(
        f"ATTACH 'ducklake:sqlite:{metadata_path}' AS test (DATA_PATH '{data_path}')"
    )
    conn.execute("CREATE SCHEMA test.analytics")
    conn.execute("CREATE TABLE test.main.users (id INTEGER)")
    conn.execute("CREATE TABLE test.analytics.metrics (id INTEGER, value DOUBLE)")
    conn.close()
    return metadata_path


class TestListSchemas:
    def test_list_schemas_polars(self, catalog_with_schemas):
        schemas = ducklake_polars.list_schemas(catalog_with_schemas)
        assert "main" in schemas
        assert "analytics" in schemas

    def test_list_schemas_pandas(self, catalog_with_schemas):
        schemas = ducklake_pandas.list_schemas(catalog_with_schemas)
        assert "main" in schemas
        assert "analytics" in schemas

    def test_list_tables_per_schema(self, catalog_with_schemas):
        main_tables = ducklake_polars.list_tables(catalog_with_schemas, schema="main")
        assert "users" in main_tables
        analytics_tables = ducklake_polars.list_tables(catalog_with_schemas, schema="analytics")
        assert "metrics" in analytics_tables

    def test_default_schema_only(self, tmp_path):
        metadata_path = str(tmp_path / "simple.ducklake")
        data_path = str(tmp_path / "data2")
        conn = duckdb.connect()
        conn.install_extension("ducklake")
        conn.load_extension("ducklake")
        conn.install_extension("sqlite_scanner")
        conn.load_extension("sqlite_scanner")
        conn.execute(
            f"ATTACH 'ducklake:sqlite:{metadata_path}' AS test (DATA_PATH '{data_path}')"
        )
        conn.execute("CREATE TABLE test.main.t (id INTEGER)")
        conn.close()
        schemas = ducklake_polars.list_schemas(metadata_path)
        assert "main" in schemas
        assert len(schemas) == 1
