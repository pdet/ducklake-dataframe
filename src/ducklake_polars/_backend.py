"""Backend adapters for DuckLake metadata catalog connections."""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from typing import Any


@dataclass
class SQLiteBackend:
    """SQLite metadata backend."""

    path: str
    placeholder: str = "?"

    def connect(self) -> sqlite3.Connection:
        """Open a read-only SQLite connection."""
        abs_path = os.path.abspath(self.path)
        return sqlite3.connect(f"file:{abs_path}?mode=ro", uri=True)

    def is_table_not_found(self, exc: BaseException) -> bool:
        """Check if an exception indicates a missing table."""
        return isinstance(exc, sqlite3.OperationalError) and "no such table" in str(exc)


@dataclass
class PostgreSQLBackend:
    """PostgreSQL metadata backend."""

    connection_string: str
    placeholder: str = "%s"

    def connect(self) -> Any:
        """Open a read-only PostgreSQL connection via psycopg2."""
        try:
            import psycopg2
        except ImportError:
            msg = (
                "psycopg2 is required for PostgreSQL catalog backends. "
                "Install it with: pip install ducklake-polars[postgres]"
            )
            raise ImportError(msg) from None
        con = psycopg2.connect(self.connection_string)
        con.set_session(readonly=True, autocommit=True)
        return con

    def is_table_not_found(self, exc: BaseException) -> bool:
        """Check if an exception indicates a missing table (pgcode 42P01)."""
        try:
            import psycopg2
        except ImportError:
            return False
        return isinstance(exc, psycopg2.ProgrammingError) and getattr(exc, "pgcode", None) == "42P01"


def create_backend(path: str) -> SQLiteBackend | PostgreSQLBackend:
    """
    Auto-detect the backend type from the connection string.

    PostgreSQL is detected when the path starts with ``postgresql://``,
    ``postgres://``, or contains ``host=`` or ``dbname=`` (libpq key-value
    format).  Everything else is treated as a SQLite file path.
    """
    lower = path.strip().lower()
    if (
        lower.startswith("postgresql://")
        or lower.startswith("postgres://")
        or "host=" in lower
        or "dbname=" in lower
    ):
        return PostgreSQLBackend(connection_string=path)
    return SQLiteBackend(path=path)
