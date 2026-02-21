"""Tests for backend detection and adapter behaviour."""

from __future__ import annotations

import sqlite3
from unittest.mock import patch

import pytest

from ducklake_polars._backend import (
    PostgreSQLBackend,
    SQLiteBackend,
    create_backend,
)


# ---------------------------------------------------------------------------
# create_backend detection
# ---------------------------------------------------------------------------

class TestCreateBackendSQLite:
    def test_ducklake_file(self):
        b = create_backend("catalog.ducklake")
        assert isinstance(b, SQLiteBackend)
        assert b.path == "catalog.ducklake"

    def test_db_file(self):
        b = create_backend("/tmp/my_catalog.db")
        assert isinstance(b, SQLiteBackend)

    def test_relative_path(self):
        b = create_backend("./data/test.ducklake")
        assert isinstance(b, SQLiteBackend)

    def test_absolute_path(self):
        b = create_backend("/home/user/catalog.ducklake")
        assert isinstance(b, SQLiteBackend)


class TestCreateBackendPostgreSQL:
    def test_postgresql_uri(self):
        b = create_backend("postgresql://user:pass@localhost/mydb")
        assert isinstance(b, PostgreSQLBackend)
        assert b.connection_string == "postgresql://user:pass@localhost/mydb"

    def test_postgres_uri(self):
        b = create_backend("postgres://user:pass@localhost/mydb")
        assert isinstance(b, PostgreSQLBackend)

    def test_host_keyword(self):
        b = create_backend("host=localhost dbname=mydb user=test")
        assert isinstance(b, PostgreSQLBackend)

    def test_dbname_keyword(self):
        b = create_backend("dbname=mydb")
        assert isinstance(b, PostgreSQLBackend)

    def test_case_insensitive(self):
        b = create_backend("PostgreSQL://user:pass@localhost/mydb")
        assert isinstance(b, PostgreSQLBackend)

    def test_whitespace_stripped(self):
        b = create_backend("  postgresql://localhost/db  ")
        assert isinstance(b, PostgreSQLBackend)


# ---------------------------------------------------------------------------
# Placeholder values
# ---------------------------------------------------------------------------

class TestPlaceholders:
    def test_sqlite_placeholder(self):
        b = SQLiteBackend(path="test.db")
        assert b.placeholder == "?"

    def test_postgresql_placeholder(self):
        b = PostgreSQLBackend(connection_string="postgresql://localhost/db")
        assert b.placeholder == "%s"


# ---------------------------------------------------------------------------
# _sql() translation (via DuckLakeCatalogReader)
# ---------------------------------------------------------------------------

class TestSqlTranslation:
    def test_sqlite_noop(self):
        from ducklake_polars._catalog import DuckLakeCatalogReader

        reader = DuckLakeCatalogReader.__new__(DuckLakeCatalogReader)
        reader._backend = SQLiteBackend(path="test.db")
        assert reader._sql("SELECT * WHERE x = ?") == "SELECT * WHERE x = ?"

    def test_postgresql_translates(self):
        from ducklake_polars._catalog import DuckLakeCatalogReader

        reader = DuckLakeCatalogReader.__new__(DuckLakeCatalogReader)
        reader._backend = PostgreSQLBackend(connection_string="postgresql://localhost/db")
        assert reader._sql("SELECT * WHERE x = ? AND y = ?") == "SELECT * WHERE x = %s AND y = %s"


# ---------------------------------------------------------------------------
# SQLiteBackend.is_table_not_found
# ---------------------------------------------------------------------------

class TestSQLiteIsTableNotFound:
    def test_matching_error(self):
        b = SQLiteBackend(path="test.db")
        exc = sqlite3.OperationalError("no such table: ducklake_foo")
        assert b.is_table_not_found(exc) is True

    def test_other_operational_error(self):
        b = SQLiteBackend(path="test.db")
        exc = sqlite3.OperationalError("disk I/O error")
        assert b.is_table_not_found(exc) is False

    def test_non_operational_error(self):
        b = SQLiteBackend(path="test.db")
        assert b.is_table_not_found(ValueError("something")) is False


# ---------------------------------------------------------------------------
# PostgreSQLBackend.connect raises helpful error without psycopg2
# ---------------------------------------------------------------------------

class TestPostgreSQLConnectWithoutPsycopg2:
    def test_import_error_message(self):
        b = PostgreSQLBackend(connection_string="postgresql://localhost/db")
        with patch.dict("sys.modules", {"psycopg2": None}):
            with pytest.raises(ImportError, match="psycopg2 is required"):
                b.connect()
