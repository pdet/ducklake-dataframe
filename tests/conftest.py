"""Shared test fixtures for ducklake-polars tests."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import duckdb
import pytest


@dataclass
class DuckLakeTestCatalog:
    """
    Helper for creating and populating DuckLake catalogs in tests.

    Usage:
        catalog = DuckLakeTestCatalog(tmp_path)
        catalog.execute("CREATE TABLE ducklake.test (a INTEGER)")
        catalog.execute("INSERT INTO ducklake.test VALUES (1)")
        catalog.close()  # Release the file lock

        # Now read with ducklake-polars
        result = read_ducklake(catalog.metadata_path, "test")
    """

    metadata_path: str
    data_path: str
    _con: duckdb.DuckDBPyConnection = field(init=False, repr=False)
    _closed: bool = field(init=False, default=False)
    inline: bool = True

    def __post_init__(self) -> None:
        os.makedirs(self.data_path, exist_ok=True)
        self._con = duckdb.connect()
        self._con.install_extension("ducklake")
        self._con.load_extension("ducklake")
        self._con.install_extension("sqlite_scanner")
        self._con.load_extension("sqlite_scanner")

        inline_opt = "" if self.inline else ", DATA_INLINING_ROW_LIMIT 0"
        self._con.execute(
            f"""
            ATTACH 'ducklake:sqlite:{self.metadata_path}' AS ducklake
                (DATA_PATH '{self.data_path}'{inline_opt})
            """
        )

    def execute(self, sql: str, params: list[Any] | None = None) -> Any:
        if params is not None:
            return self._con.execute(sql, params)
        return self._con.execute(sql)

    def fetchone(self, sql: str) -> Any:
        return self._con.execute(sql).fetchone()

    def fetchall(self, sql: str) -> list[Any]:
        return self._con.execute(sql).fetchall()

    def close(self) -> None:
        """Close the DuckDB connection, releasing the file lock."""
        if not self._closed:
            self._con.close()
            self._closed = True

    def __enter__(self) -> "DuckLakeTestCatalog":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


@pytest.fixture
def ducklake_catalog(tmp_path):
    """
    Create a DuckLake catalog with data inlining disabled.

    Returns a DuckLakeTestCatalog instance. The caller MUST call .close()
    before reading with ducklake-polars to release the file lock.
    """
    metadata_path = str(tmp_path / "test.ducklake")
    data_path = str(tmp_path / "data")

    catalog = DuckLakeTestCatalog(
        metadata_path=metadata_path,
        data_path=data_path,
        inline=False,
    )

    yield catalog

    catalog.close()


@pytest.fixture
def ducklake_catalog_inline(tmp_path):
    """
    Create a DuckLake catalog with data inlining enabled (default).
    """
    metadata_path = str(tmp_path / "test.ducklake")
    data_path = str(tmp_path / "data")

    catalog = DuckLakeTestCatalog(
        metadata_path=metadata_path,
        data_path=data_path,
        inline=True,
    )

    yield catalog

    catalog.close()
