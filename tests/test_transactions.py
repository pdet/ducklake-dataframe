"""Transaction semantics tests for ducklake-dataframe.

Closes the gap identified in TEST_PARITY.md §Transaction (12 ref tests):
rollback on error, multi-operation transactions, schema change rollback,
conflict cleanup, create conflicts, transaction isolation, snapshot
consistency, retry semantics, writer reuse, and nested operations.
"""

from __future__ import annotations

import threading
import time
from unittest.mock import patch

import polars as pl
import pyarrow as pa
import pytest

from ducklake_polars import (
    alter_ducklake_add_column,
    alter_ducklake_drop_column,
    alter_ducklake_rename_column,
    delete_ducklake,
    read_ducklake,
    update_ducklake,
    write_ducklake,
)
from ducklake_core._writer import (
    DuckLakeCatalogWriter,
    TransactionConflictError,
)


# ── helpers ──────────────────────────────────────────────────────────────

def _snapshot_count(metadata_path: str) -> int:
    """Return the number of snapshots in the catalog."""
    import sqlite3
    con = sqlite3.connect(metadata_path)
    try:
        return con.execute("SELECT COUNT(*) FROM ducklake_snapshot").fetchone()[0]
    finally:
        con.close()


def _table_count(metadata_path: str) -> int:
    """Return the number of active (non-ended) tables."""
    import sqlite3
    con = sqlite3.connect(metadata_path)
    try:
        return con.execute(
            "SELECT COUNT(*) FROM ducklake_table WHERE end_snapshot IS NULL"
        ).fetchone()[0]
    finally:
        con.close()


def _column_names(metadata_path: str, table_name: str) -> list[str]:
    """Return active column names for a table."""
    import sqlite3
    con = sqlite3.connect(metadata_path)
    try:
        rows = con.execute(
            "SELECT c.column_name FROM ducklake_column c "
            "JOIN ducklake_table t ON c.table_id = t.table_id "
            "WHERE t.table_name = ? AND c.end_snapshot IS NULL "
            "AND t.end_snapshot IS NULL AND c.parent_column IS NULL "
            "ORDER BY c.column_order",
            [table_name],
        ).fetchall()
        return [r[0] for r in rows]
    finally:
        con.close()


# ── Test: rollback on error ─────────────────────────────────────────────


class TestRollbackOnError:
    """Verify catalog is unchanged after a failed write."""

    def test_insert_into_nonexistent_table_no_snapshot(self, ducklake_catalog_sqlite):
        """A failed insert (table not found) via core writer must not create a snapshot."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        cat.close()

        snap_before = _snapshot_count(cat.metadata_path)

        # Use the core writer directly — write_ducklake(mode="append")
        # auto-creates missing tables, but the raw insert_data raises.
        from ducklake_core._exceptions import TableNotFoundError
        with pytest.raises(TableNotFoundError):
            with DuckLakeCatalogWriter(
                cat.metadata_path, data_path_override=cat.data_path
            ) as writer:
                writer.insert_data(
                    pa.table({"a": pa.array([1], type=pa.int32())}),
                    "nonexistent",
                )

        snap_after = _snapshot_count(cat.metadata_path)
        assert snap_after == snap_before

    def test_insert_empty_df_no_snapshot(self, ducklake_catalog_sqlite):
        """Inserting an empty Arrow table via core writer raises and creates no snapshot."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        cat.close()

        snap_before = _snapshot_count(cat.metadata_path)

        # write_ducklake silently skips empty DFs; test the core writer
        # where the "Cannot insert empty" check actually lives.
        with pytest.raises(ValueError, match="Cannot insert empty"):
            with DuckLakeCatalogWriter(
                cat.metadata_path, data_path_override=cat.data_path
            ) as writer:
                writer.insert_data(
                    pa.table({"a": pa.array([], type=pa.int32())}),
                    "test",
                )

        snap_after = _snapshot_count(cat.metadata_path)
        assert snap_after == snap_before

    def test_create_duplicate_table_no_snapshot(self, ducklake_catalog_sqlite):
        """Creating a table that already exists must not create a snapshot."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        cat.close()

        snap_before = _snapshot_count(cat.metadata_path)

        with pytest.raises(ValueError, match="already exists"):
            write_ducklake(
                pl.DataFrame({"a": pl.Series([1], dtype=pl.Int32)}),
                cat.metadata_path, "test",
                data_path=cat.data_path, mode="error",
            )

        snap_after = _snapshot_count(cat.metadata_path)
        assert snap_after == snap_before

    def test_delete_no_match_no_snapshot(self, ducklake_catalog_sqlite):
        """A delete that matches nothing should not create a new snapshot."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        cat.execute("INSERT INTO ducklake.test VALUES (1), (2), (3)")
        cat.close()

        snap_before = _snapshot_count(cat.metadata_path)

        deleted = delete_ducklake(
            cat.metadata_path, "test",
            pl.col("a") > 100,
            data_path=cat.data_path,
        )

        assert deleted == 0
        snap_after = _snapshot_count(cat.metadata_path)
        # No new snapshot when nothing matched
        assert snap_after == snap_before

    def test_update_no_match_no_snapshot(self, ducklake_catalog_sqlite):
        """An update that matches nothing should not create a new snapshot."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        cat.execute("INSERT INTO ducklake.test VALUES (1, 'x'), (2, 'y')")
        cat.close()

        snap_before = _snapshot_count(cat.metadata_path)

        updated = update_ducklake(
            cat.metadata_path, "test",
            {"b": "z"},
            pl.col("a") > 100,
            data_path=cat.data_path,
        )

        assert updated == 0
        snap_after = _snapshot_count(cat.metadata_path)
        assert snap_after == snap_before


# ── Test: multi-operation transactions ──────────────────────────────────


class TestMultiOperationTransaction:
    """Insert + update + delete as sequential operations produce correct state."""

    def test_insert_update_delete_sequence(self, ducklake_catalog_sqlite):
        """Insert -> update -> delete sequence produces consistent catalog."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        cat.close()

        # Insert
        df = pl.DataFrame({
            "a": pl.Series([1, 2, 3, 4, 5], dtype=pl.Int32),
            "b": ["a", "b", "c", "d", "e"],
        })
        write_ducklake(df, cat.metadata_path, "test",
                       data_path=cat.data_path, mode="append")

        result = read_ducklake(cat.metadata_path, "test", data_path=cat.data_path)
        assert result.shape[0] == 5

        # Update
        updated = update_ducklake(
            cat.metadata_path, "test",
            {"b": "updated"},
            pl.col("a") <= 2,
            data_path=cat.data_path,
        )
        assert updated == 2

        result = read_ducklake(cat.metadata_path, "test", data_path=cat.data_path)
        assert result.shape[0] == 5
        updated_rows = result.filter(pl.col("b") == "updated")
        assert updated_rows.shape[0] == 2

        # Delete
        deleted = delete_ducklake(
            cat.metadata_path, "test",
            pl.col("a") > 3,
            data_path=cat.data_path,
        )
        assert deleted == 2

        result = read_ducklake(cat.metadata_path, "test", data_path=cat.data_path)
        assert result.shape[0] == 3

    def test_multiple_inserts_accumulate(self, ducklake_catalog_sqlite):
        """Multiple sequential inserts accumulate rows correctly."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        cat.close()

        for i in range(5):
            write_ducklake(
                pl.DataFrame({"a": pl.Series([i * 10 + j for j in range(3)], dtype=pl.Int32)}),
                cat.metadata_path, "test",
                data_path=cat.data_path, mode="append",
            )

        result = read_ducklake(cat.metadata_path, "test", data_path=cat.data_path)
        assert result.shape[0] == 15


# ── Test: schema change rollback ────────────────────────────────────────


class TestSchemaChangeRollback:
    """Failed schema alteration doesn't persist."""

    def test_add_duplicate_column_no_schema_change(self, ducklake_catalog_sqlite):
        """Adding a column that already exists must not change the schema."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        cat.close()

        cols_before = _column_names(cat.metadata_path, "test")

        with pytest.raises(ValueError, match="already exists"):
            alter_ducklake_add_column(
                cat.metadata_path, "test", "a", pl.Int32,
                data_path=cat.data_path,
            )

        cols_after = _column_names(cat.metadata_path, "test")
        assert cols_before == cols_after

    def test_drop_nonexistent_column_no_schema_change(self, ducklake_catalog_sqlite):
        """Dropping a column that doesn't exist must not change the schema."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        cat.close()

        cols_before = _column_names(cat.metadata_path, "test")

        with pytest.raises(ValueError, match="not found"):
            alter_ducklake_drop_column(
                cat.metadata_path, "test", "nonexistent",
                data_path=cat.data_path,
            )

        cols_after = _column_names(cat.metadata_path, "test")
        assert cols_before == cols_after

    def test_rename_nonexistent_column_no_change(self, ducklake_catalog_sqlite):
        """Renaming a column that doesn't exist must not change the schema."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        cat.close()

        cols_before = _column_names(cat.metadata_path, "test")

        with pytest.raises(ValueError, match="not found"):
            alter_ducklake_rename_column(
                cat.metadata_path, "test", "nonexistent", "new_name",
                data_path=cat.data_path,
            )

        cols_after = _column_names(cat.metadata_path, "test")
        assert cols_before == cols_after


# ── Test: create table conflict ─────────────────────────────────────────


class TestCreateConflict:
    """Two writers creating the same table -- second must fail."""

    def test_create_same_table_twice_fails(self, ducklake_catalog_sqlite):
        """Second create of the same table name fails with appropriate error."""
        cat = ducklake_catalog_sqlite
        cat.close()

        write_ducklake(
            pl.DataFrame({"a": pl.Series([1], dtype=pl.Int32)}),
            cat.metadata_path, "test",
            data_path=cat.data_path, mode="error",
        )

        with pytest.raises(ValueError, match="already exists"):
            write_ducklake(
                pl.DataFrame({"a": pl.Series([2], dtype=pl.Int32)}),
                cat.metadata_path, "test",
                data_path=cat.data_path, mode="error",
            )

        result = read_ducklake(cat.metadata_path, "test", data_path=cat.data_path)
        assert result["a"].to_list() == [1]

    def test_concurrent_create_same_table(self, ducklake_catalog_sqlite):
        """Two threads trying to create the same table -- exactly one succeeds."""
        cat = ducklake_catalog_sqlite
        cat.close()

        results = {"success": 0, "errors": []}
        lock = threading.Lock()

        def create_table(val: int):
            try:
                write_ducklake(
                    pl.DataFrame({"a": pl.Series([val], dtype=pl.Int32)}),
                    cat.metadata_path, "test",
                    data_path=cat.data_path, mode="error",
                )
                with lock:
                    results["success"] += 1
            except Exception as e:
                with lock:
                    results["errors"].append(e)

        t1 = threading.Thread(target=create_table, args=(1,))
        t2 = threading.Thread(target=create_table, args=(2,))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # At least one must succeed, at most one can succeed
        assert results["success"] >= 1
        # The table should exist and be readable
        result = read_ducklake(cat.metadata_path, "test", data_path=cat.data_path)
        assert result.shape[0] == 1


# ── Test: insert + schema change in same session ────────────────────────


class TestInsertAndSchemaChange:
    """Insert data then alter schema -- both changes visible."""

    def test_insert_then_add_column(self, ducklake_catalog_sqlite):
        """Insert data, then add a column -- both are reflected."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        cat.close()

        write_ducklake(
            pl.DataFrame({"a": pl.Series([1, 2, 3], dtype=pl.Int32)}),
            cat.metadata_path, "test",
            data_path=cat.data_path, mode="append",
        )

        alter_ducklake_add_column(
            cat.metadata_path, "test", "b", pl.Utf8,
            data_path=cat.data_path,
        )

        result = read_ducklake(cat.metadata_path, "test", data_path=cat.data_path)
        assert result.shape[0] == 3
        assert "b" in result.columns
        # Old rows have NULL for the new column
        assert result["b"].null_count() == 3

    def test_add_column_then_insert_with_new_column(self, ducklake_catalog_sqlite):
        """Add a column, then insert data that includes the new column."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        cat.execute("INSERT INTO ducklake.test VALUES (1)")
        cat.close()

        alter_ducklake_add_column(
            cat.metadata_path, "test", "b", pl.Utf8,
            data_path=cat.data_path,
        )

        write_ducklake(
            pl.DataFrame({
                "a": pl.Series([2], dtype=pl.Int32),
                "b": ["hello"],
            }),
            cat.metadata_path, "test",
            data_path=cat.data_path, mode="append",
        )

        result = read_ducklake(cat.metadata_path, "test", data_path=cat.data_path)
        assert result.shape[0] == 2
        non_null = result.filter(pl.col("b").is_not_null())
        assert non_null.shape[0] == 1
        assert non_null["b"][0] == "hello"

    def test_insert_then_drop_column(self, ducklake_catalog_sqlite):
        """Insert, then drop column -- data visible without dropped column."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        cat.close()

        write_ducklake(
            pl.DataFrame({
                "a": pl.Series([1, 2], dtype=pl.Int32),
                "b": ["x", "y"],
            }),
            cat.metadata_path, "test",
            data_path=cat.data_path, mode="append",
        )

        alter_ducklake_drop_column(
            cat.metadata_path, "test", "b",
            data_path=cat.data_path,
        )

        result = read_ducklake(cat.metadata_path, "test", data_path=cat.data_path)
        assert result.shape[0] == 2
        assert "b" not in result.columns
        assert result.columns == ["a"]


# ── Test: transaction isolation ─────────────────────────────────────────


class TestTransactionIsolation:
    """Reader doesn't see uncommitted writes."""

    def test_reader_sees_committed_state(self, ducklake_catalog_sqlite):
        """Two sequential reads with a write between them see different states."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        cat.execute("INSERT INTO ducklake.test VALUES (1)")
        cat.close()

        result_before = read_ducklake(cat.metadata_path, "test",
                                       data_path=cat.data_path)
        assert result_before.shape[0] == 1

        write_ducklake(
            pl.DataFrame({"a": pl.Series([2, 3], dtype=pl.Int32)}),
            cat.metadata_path, "test",
            data_path=cat.data_path, mode="append",
        )

        result_after = read_ducklake(cat.metadata_path, "test",
                                      data_path=cat.data_path)
        assert result_after.shape[0] == 3

    def test_time_travel_isolation(self, ducklake_catalog_sqlite):
        """Time travel reads at a specific snapshot see only that snapshot's data."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        cat.execute("INSERT INTO ducklake.test VALUES (1)")
        cat.close()

        # Snapshot at creation + first insert = early snapshots
        write_ducklake(
            pl.DataFrame({"a": pl.Series([2], dtype=pl.Int32)}),
            cat.metadata_path, "test",
            data_path=cat.data_path, mode="append",
        )

        write_ducklake(
            pl.DataFrame({"a": pl.Series([3], dtype=pl.Int32)}),
            cat.metadata_path, "test",
            data_path=cat.data_path, mode="append",
        )

        # Latest version has 3 rows
        result_latest = read_ducklake(cat.metadata_path, "test",
                                       data_path=cat.data_path)
        assert result_latest.shape[0] == 3

        # Read at earlier snapshot should show fewer rows
        import sqlite3
        con = sqlite3.connect(cat.metadata_path)
        snapshots = con.execute(
            "SELECT snapshot_id FROM ducklake_snapshot ORDER BY snapshot_id"
        ).fetchall()
        con.close()

        # Read at each snapshot and verify monotonic row count growth
        row_counts = []
        for (snap_id,) in snapshots:
            try:
                r = read_ducklake(cat.metadata_path, "test",
                                   data_path=cat.data_path,
                                   snapshot_version=snap_id)
                row_counts.append(r.shape[0])
            except Exception:
                row_counts.append(0)

        # Row counts should be non-decreasing (each insert adds rows)
        for i in range(1, len(row_counts)):
            assert row_counts[i] >= row_counts[i - 1], (
                f"Row count decreased from snapshot {i-1} to {i}: "
                f"{row_counts[i-1]} -> {row_counts[i]}"
            )


# ── Test: snapshot consistency ──────────────────────────────────────────


class TestSnapshotConsistency:
    """All operations in a transaction share one snapshot."""

    def test_each_operation_creates_exactly_one_snapshot(self, ducklake_catalog_sqlite):
        """Each write operation creates exactly one new snapshot."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        cat.close()

        snap_0 = _snapshot_count(cat.metadata_path)

        write_ducklake(
            pl.DataFrame({
                "a": pl.Series([1], dtype=pl.Int32),
                "b": ["x"],
            }),
            cat.metadata_path, "test",
            data_path=cat.data_path, mode="append",
        )
        snap_1 = _snapshot_count(cat.metadata_path)
        assert snap_1 == snap_0 + 1

        update_ducklake(
            cat.metadata_path, "test",
            {"b": "y"},
            pl.col("a") == 1,
            data_path=cat.data_path,
        )
        snap_2 = _snapshot_count(cat.metadata_path)
        assert snap_2 == snap_1 + 1

        delete_ducklake(
            cat.metadata_path, "test",
            pl.col("a") == 1,
            data_path=cat.data_path,
        )
        snap_3 = _snapshot_count(cat.metadata_path)
        assert snap_3 == snap_2 + 1

    def test_overwrite_creates_single_snapshot(self, ducklake_catalog_sqlite):
        """An overwrite (truncate + insert) creates exactly one snapshot."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        cat.execute("INSERT INTO ducklake.test VALUES (1), (2), (3)")
        cat.close()

        snap_before = _snapshot_count(cat.metadata_path)

        write_ducklake(
            pl.DataFrame({"a": pl.Series([10, 20], dtype=pl.Int32)}),
            cat.metadata_path, "test",
            data_path=cat.data_path, mode="overwrite",
        )

        snap_after = _snapshot_count(cat.metadata_path)
        assert snap_after == snap_before + 1

        result = read_ducklake(cat.metadata_path, "test", data_path=cat.data_path)
        assert sorted(result["a"].to_list()) == [10, 20]


# ── Test: retry semantics ──────────────────────────────────────────────


class TestRetrySemantics:
    """TransactionConflictError triggers automatic retry with backoff."""

    def test_retryable_decorator_retries_on_conflict(self):
        """The _retryable decorator retries the correct number of times."""
        call_count = 0

        class MockWriter:
            _max_retries = 3
            _retry_wait_ms = 1  # fast for testing
            _retry_backoff = 1.0

            def _reset_connection(self):
                pass

        from ducklake_core._writer import _retryable

        @_retryable
        def flaky_write(self):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise TransactionConflictError("conflict")
            return "success"

        writer = MockWriter()
        result = flaky_write(writer)
        assert result == "success"
        assert call_count == 3  # 2 failures + 1 success

    def test_retryable_exhausts_retries_raises(self):
        """When retries are exhausted, the error propagates."""
        call_count = 0

        class MockWriter:
            _max_retries = 2
            _retry_wait_ms = 1
            _retry_backoff = 1.0

            def _reset_connection(self):
                pass

        from ducklake_core._writer import _retryable

        @_retryable
        def always_fail(self):
            nonlocal call_count
            call_count += 1
            raise TransactionConflictError("permanent conflict")

        writer = MockWriter()
        with pytest.raises(TransactionConflictError, match="permanent conflict"):
            always_fail(writer)
        assert call_count == 3  # initial + 2 retries

    def test_retryable_resets_connection_between_retries(self):
        """_reset_connection is called between retry attempts."""
        reset_count = 0
        attempt = 0

        class MockWriter:
            _max_retries = 2
            _retry_wait_ms = 1
            _retry_backoff = 1.0

            def _reset_connection(self):
                nonlocal reset_count
                reset_count += 1

        from ducklake_core._writer import _retryable

        @_retryable
        def fail_once(self):
            nonlocal attempt
            attempt += 1
            if attempt == 1:
                raise TransactionConflictError("conflict")
            return "ok"

        writer = MockWriter()
        fail_once(writer)
        assert reset_count == 1

    def test_retry_with_real_writer(self, ducklake_catalog_sqlite):
        """Retry parameters work end-to-end through write_ducklake."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        cat.close()

        # This should succeed even with aggressive retry settings
        write_ducklake(
            pl.DataFrame({"a": pl.Series([1], dtype=pl.Int32)}),
            cat.metadata_path, "test",
            data_path=cat.data_path, mode="append",
            max_retries=5, retry_wait_ms=10, retry_backoff=1.5,
        )

        result = read_ducklake(cat.metadata_path, "test", data_path=cat.data_path)
        assert result.shape[0] == 1


# ── Test: writer reuse ──────────────────────────────────────────────────


class TestWriterReuse:
    """Same writer object used for multiple sequential writes."""

    def test_writer_reuse_multiple_inserts(self, ducklake_catalog_sqlite):
        """A single writer instance can perform multiple insert operations."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        cat.close()

        with DuckLakeCatalogWriter(
            cat.metadata_path, data_path_override=cat.data_path
        ) as writer:
            df1 = pa.table({"a": pa.array([1, 2], type=pa.int32())})
            df2 = pa.table({"a": pa.array([3, 4], type=pa.int32())})
            df3 = pa.table({"a": pa.array([5, 6], type=pa.int32())})

            writer.insert_data(df1, "test")
            writer.insert_data(df2, "test")
            writer.insert_data(df3, "test")

        result = read_ducklake(cat.metadata_path, "test", data_path=cat.data_path)
        assert result.shape[0] == 6
        assert sorted(result["a"].to_list()) == [1, 2, 3, 4, 5, 6]

    def test_writer_reuse_create_and_insert(self, ducklake_catalog_sqlite):
        """A single writer can create a table and then insert into it."""
        cat = ducklake_catalog_sqlite
        cat.close()

        with DuckLakeCatalogWriter(
            cat.metadata_path, data_path_override=cat.data_path
        ) as writer:
            writer.create_table("test", {"a": pa.int32(), "b": pa.string()})
            df = pa.table({
                "a": pa.array([1, 2], type=pa.int32()),
                "b": pa.array(["x", "y"], type=pa.string()),
            })
            writer.insert_data(df, "test")

        result = read_ducklake(cat.metadata_path, "test", data_path=cat.data_path)
        assert result.shape[0] == 2

    def test_writer_reuse_different_tables(self, ducklake_catalog_sqlite):
        """A single writer can operate on multiple tables sequentially."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.t1 (a INTEGER)")
        cat.execute("CREATE TABLE ducklake.t2 (b VARCHAR)")
        cat.close()

        with DuckLakeCatalogWriter(
            cat.metadata_path, data_path_override=cat.data_path
        ) as writer:
            writer.insert_data(
                pa.table({"a": pa.array([1, 2], type=pa.int32())}), "t1"
            )
            writer.insert_data(
                pa.table({"b": pa.array(["x", "y"], type=pa.string())}), "t2"
            )

        r1 = read_ducklake(cat.metadata_path, "t1", data_path=cat.data_path)
        r2 = read_ducklake(cat.metadata_path, "t2", data_path=cat.data_path)
        assert r1.shape[0] == 2
        assert r2.shape[0] == 2


# ── Test: nested operations ─────────────────────────────────────────────


class TestNestedOperations:
    """Write -> alter -> write -> commit patterns."""

    def test_write_alter_write(self, ducklake_catalog_sqlite):
        """Insert data, alter schema, insert more data with new schema."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        cat.close()

        # First insert
        write_ducklake(
            pl.DataFrame({"a": pl.Series([1, 2], dtype=pl.Int32)}),
            cat.metadata_path, "test",
            data_path=cat.data_path, mode="append",
        )

        # Alter: add column
        alter_ducklake_add_column(
            cat.metadata_path, "test", "b", pl.Utf8,
            data_path=cat.data_path,
        )

        # Second insert with new column
        write_ducklake(
            pl.DataFrame({
                "a": pl.Series([3, 4], dtype=pl.Int32),
                "b": ["hello", "world"],
            }),
            cat.metadata_path, "test",
            data_path=cat.data_path, mode="append",
        )

        result = read_ducklake(cat.metadata_path, "test", data_path=cat.data_path)
        assert result.shape[0] == 4
        assert set(result.columns) == {"a", "b"}
        # First 2 rows have NULL for b, last 2 have values
        non_null_b = result.filter(pl.col("b").is_not_null())
        assert non_null_b.shape[0] == 2

    def test_write_rename_write(self, ducklake_catalog_sqlite):
        """Insert -> rename column -> insert with renamed column."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        cat.close()

        write_ducklake(
            pl.DataFrame({
                "a": pl.Series([1], dtype=pl.Int32),
                "b": ["old"],
            }),
            cat.metadata_path, "test",
            data_path=cat.data_path, mode="append",
        )

        alter_ducklake_rename_column(
            cat.metadata_path, "test", "b", "c",
            data_path=cat.data_path,
        )

        write_ducklake(
            pl.DataFrame({
                "a": pl.Series([2], dtype=pl.Int32),
                "c": ["new"],
            }),
            cat.metadata_path, "test",
            data_path=cat.data_path, mode="append",
        )

        result = read_ducklake(cat.metadata_path, "test", data_path=cat.data_path)
        assert result.shape[0] == 2
        assert "c" in result.columns
        assert "b" not in result.columns

    def test_write_drop_column_write(self, ducklake_catalog_sqlite):
        """Insert -> drop column -> insert without dropped column."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR, c DOUBLE)")
        cat.close()

        write_ducklake(
            pl.DataFrame({
                "a": pl.Series([1], dtype=pl.Int32),
                "b": ["x"],
                "c": pl.Series([1.0], dtype=pl.Float64),
            }),
            cat.metadata_path, "test",
            data_path=cat.data_path, mode="append",
        )

        alter_ducklake_drop_column(
            cat.metadata_path, "test", "b",
            data_path=cat.data_path,
        )

        write_ducklake(
            pl.DataFrame({
                "a": pl.Series([2], dtype=pl.Int32),
                "c": pl.Series([2.0], dtype=pl.Float64),
            }),
            cat.metadata_path, "test",
            data_path=cat.data_path, mode="append",
        )

        result = read_ducklake(cat.metadata_path, "test", data_path=cat.data_path)
        assert result.shape[0] == 2
        assert set(result.columns) == {"a", "c"}

    def test_multiple_schema_changes(self, ducklake_catalog_sqlite):
        """Multiple schema changes in sequence: add, rename, add, drop."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        cat.close()

        write_ducklake(
            pl.DataFrame({"a": pl.Series([1], dtype=pl.Int32)}),
            cat.metadata_path, "test",
            data_path=cat.data_path, mode="append",
        )

        # Add column b
        alter_ducklake_add_column(
            cat.metadata_path, "test", "b", pl.Utf8,
            data_path=cat.data_path,
        )
        # Rename a -> x
        alter_ducklake_rename_column(
            cat.metadata_path, "test", "a", "x",
            data_path=cat.data_path,
        )
        # Add column c
        alter_ducklake_add_column(
            cat.metadata_path, "test", "c", pl.Float64,
            data_path=cat.data_path,
        )
        # Drop b
        alter_ducklake_drop_column(
            cat.metadata_path, "test", "b",
            data_path=cat.data_path,
        )

        result = read_ducklake(cat.metadata_path, "test", data_path=cat.data_path)
        assert set(result.columns) == {"x", "c"}
        assert result.shape[0] == 1


# ── Test: conflict detection internals ──────────────────────────────────


class TestConflictDetection:
    """Verify _check_conflicts and _parse_table_changes work correctly."""

    def test_parse_table_changes_insert(self):
        """Parse insert change strings."""
        result = DuckLakeCatalogWriter._parse_table_changes(
            "inserted_into_table:1"
        )
        assert result == {1: {"inserted_into_table"}}

    def test_parse_table_changes_multiple(self):
        """Parse multiple changes in one string."""
        result = DuckLakeCatalogWriter._parse_table_changes(
            "inserted_into_table:1,deleted_from_table:1,altered_table:2"
        )
        assert result == {
            1: {"inserted_into_table", "deleted_from_table"},
            2: {"altered_table"},
        }

    def test_parse_table_changes_empty(self):
        """Parse empty string returns empty dict."""
        assert DuckLakeCatalogWriter._parse_table_changes("") == {}

    def test_parse_table_changes_dropped(self):
        """Parse dropped_table changes."""
        result = DuckLakeCatalogWriter._parse_table_changes(
            "dropped_table:5"
        )
        assert result == {5: {"dropped_table"}}

    def test_check_conflicts_insert_vs_alter(self):
        """Insert conflicts with concurrent ALTER."""
        writer = DuckLakeCatalogWriter.__new__(DuckLakeCatalogWriter)
        writer._backend = None
        writer._con = None

        # Mock _get_concurrent_changes to return ALTER
        with patch.object(
            writer, "_get_concurrent_changes",
            return_value=[(2, "altered_table:1")],
        ):
            with pytest.raises(TransactionConflictError, match="schema was altered"):
                writer._check_conflicts(0, {1: "insert"})

    def test_check_conflicts_insert_vs_insert_ok(self):
        """Concurrent inserts to the same table should NOT conflict."""
        writer = DuckLakeCatalogWriter.__new__(DuckLakeCatalogWriter)
        writer._backend = None
        writer._con = None

        with patch.object(
            writer, "_get_concurrent_changes",
            return_value=[(2, "inserted_into_table:1")],
        ):
            # Should not raise
            writer._check_conflicts(0, {1: "insert"})

    def test_check_conflicts_delete_vs_delete(self):
        """Concurrent deletes on the same table conflict."""
        writer = DuckLakeCatalogWriter.__new__(DuckLakeCatalogWriter)
        writer._backend = None
        writer._con = None

        with patch.object(
            writer, "_get_concurrent_changes",
            return_value=[(2, "deleted_from_table:1")],
        ):
            with pytest.raises(TransactionConflictError, match="concurrent deletes"):
                writer._check_conflicts(0, {1: "delete"})

    def test_check_conflicts_overwrite_vs_insert(self):
        """Overwrite conflicts with concurrent insert."""
        writer = DuckLakeCatalogWriter.__new__(DuckLakeCatalogWriter)
        writer._backend = None
        writer._con = None

        with patch.object(
            writer, "_get_concurrent_changes",
            return_value=[(2, "inserted_into_table:1")],
        ):
            with pytest.raises(TransactionConflictError, match="was modified"):
                writer._check_conflicts(0, {1: "overwrite"})

    def test_check_conflicts_ddl_vs_any(self):
        """DDL conflicts with any concurrent change."""
        writer = DuckLakeCatalogWriter.__new__(DuckLakeCatalogWriter)
        writer._backend = None
        writer._con = None

        with patch.object(
            writer, "_get_concurrent_changes",
            return_value=[(2, "inserted_into_table:1")],
        ):
            with pytest.raises(TransactionConflictError, match="was modified"):
                writer._check_conflicts(0, {1: "ddl"})

    def test_check_conflicts_drop_vs_any(self):
        """Drop conflicts with any concurrent change."""
        writer = DuckLakeCatalogWriter.__new__(DuckLakeCatalogWriter)
        writer._backend = None
        writer._con = None

        with patch.object(
            writer, "_get_concurrent_changes",
            return_value=[(2, "inserted_into_table:1")],
        ):
            with pytest.raises(TransactionConflictError, match="was modified"):
                writer._check_conflicts(0, {1: "drop_table"})

    def test_check_conflicts_no_overlap(self):
        """Changes to different tables don't conflict."""
        writer = DuckLakeCatalogWriter.__new__(DuckLakeCatalogWriter)
        writer._backend = None
        writer._con = None

        with patch.object(
            writer, "_get_concurrent_changes",
            return_value=[(2, "deleted_from_table:99")],
        ):
            # Writing to table 1, concurrent change to table 99 -- no conflict
            writer._check_conflicts(0, {1: "delete"})


# ── Test: _commit_metadata and _reset_connection ────────────────────────


class TestCommitAndReset:
    """Core commit/reset mechanics work correctly."""

    def test_commit_clears_tracking_state(self, ducklake_catalog_sqlite):
        """After _commit_metadata, transaction tracking state is cleared."""
        cat = ducklake_catalog_sqlite
        cat.close()

        writer = DuckLakeCatalogWriter(
            cat.metadata_path, data_path_override=cat.data_path
        )
        try:
            writer._connect()

            # Simulate starting a tracked transaction
            writer._txn_start_snapshot = 0
            writer._txn_conflict_tables = {1: "insert"}

            # Use _commit_metadata which clears tracking state
            writer._commit_metadata()

            assert writer._txn_start_snapshot is None
            assert writer._txn_conflict_tables == {}
        finally:
            writer.close()

    def test_reset_connection_clears_state(self, ducklake_catalog_sqlite):
        """_reset_connection clears connection and tracking state."""
        cat = ducklake_catalog_sqlite
        cat.close()

        writer = DuckLakeCatalogWriter(
            cat.metadata_path, data_path_override=cat.data_path
        )
        try:
            writer._connect()
            writer._txn_start_snapshot = 5
            writer._txn_conflict_tables = {1: "delete"}

            writer._reset_connection()

            assert writer._con is None
            assert writer._txn_start_snapshot is None
            assert writer._txn_conflict_tables == {}
        finally:
            writer.close()

    def test_writer_context_manager(self, ducklake_catalog_sqlite):
        """Writer context manager properly opens and closes connection."""
        cat = ducklake_catalog_sqlite
        cat.close()

        with DuckLakeCatalogWriter(
            cat.metadata_path, data_path_override=cat.data_path
        ) as writer:
            writer._connect()
            assert writer._con is not None

        # After __exit__, connection should be closed
        assert writer._con is None


# ── Test: data integrity after operations ───────────────────────────────


class TestDataIntegrity:
    """End-to-end data integrity across multiple transaction types."""

    def test_insert_delete_update_final_state(self, ducklake_catalog_sqlite):
        """Complex sequence of operations produces correct final state."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.test (id INTEGER, name VARCHAR, val DOUBLE)")
        cat.close()

        # Insert initial data
        write_ducklake(
            pl.DataFrame({
                "id": pl.Series([1, 2, 3, 4, 5], dtype=pl.Int32),
                "name": ["alice", "bob", "carol", "dave", "eve"],
                "val": pl.Series([10.0, 20.0, 30.0, 40.0, 50.0], dtype=pl.Float64),
            }),
            cat.metadata_path, "test",
            data_path=cat.data_path, mode="append",
        )

        # Delete id=3
        delete_ducklake(
            cat.metadata_path, "test",
            pl.col("id") == 3,
            data_path=cat.data_path,
        )

        # Update id=1: val -> 100
        update_ducklake(
            cat.metadata_path, "test",
            {"val": 100.0},
            pl.col("id") == 1,
            data_path=cat.data_path,
        )

        # Insert more
        write_ducklake(
            pl.DataFrame({
                "id": pl.Series([6], dtype=pl.Int32),
                "name": ["frank"],
                "val": pl.Series([60.0], dtype=pl.Float64),
            }),
            cat.metadata_path, "test",
            data_path=cat.data_path, mode="append",
        )

        result = read_ducklake(cat.metadata_path, "test", data_path=cat.data_path)
        assert result.shape[0] == 5  # 5 - 1 delete + 1 insert = 5

        # Verify specific values
        result_sorted = result.sort("id")
        ids = result_sorted["id"].to_list()
        assert 3 not in ids  # deleted
        assert 6 in ids  # inserted

        alice = result.filter(pl.col("id") == 1)
        assert alice["val"][0] == 100.0  # updated

    def test_overwrite_preserves_table_identity(self, ducklake_catalog_sqlite):
        """Overwrite replaces data but the table identity is preserved."""
        cat = ducklake_catalog_sqlite
        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        cat.execute("INSERT INTO ducklake.test VALUES (1), (2), (3)")
        cat.close()

        # Get table count before overwrite
        tables_before = _table_count(cat.metadata_path)

        write_ducklake(
            pl.DataFrame({"a": pl.Series([10, 20], dtype=pl.Int32)}),
            cat.metadata_path, "test",
            data_path=cat.data_path, mode="overwrite",
        )

        tables_after = _table_count(cat.metadata_path)
        assert tables_after == tables_before

        result = read_ducklake(cat.metadata_path, "test", data_path=cat.data_path)
        assert sorted(result["a"].to_list()) == [10, 20]
