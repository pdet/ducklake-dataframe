"""Time travel tests for ducklake-polars."""

from __future__ import annotations

import polars as pl
import pytest

from ducklake_polars import read_ducklake, scan_ducklake


class TestTimeTravel:
    """Test reading tables at specific snapshot versions."""

    def test_read_at_version(self, ducklake_catalog):
        cat = ducklake_catalog

        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        cat.execute("INSERT INTO ducklake.test VALUES (1)")
        v1 = cat.fetchone("SELECT * FROM ducklake_current_snapshot('ducklake')")[0]

        cat.execute("INSERT INTO ducklake.test VALUES (2)")
        v2 = cat.fetchone("SELECT * FROM ducklake_current_snapshot('ducklake')")[0]

        cat.execute("INSERT INTO ducklake.test VALUES (3)")
        cat.close()

        # Read at latest - should have all 3 rows
        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape[0] == 3

        # Read at v1 - should have 1 row
        result_v1 = read_ducklake(cat.metadata_path, "test", snapshot_version=v1)
        assert result_v1.shape[0] == 1
        assert result_v1["a"].to_list() == [1]

        # Read at v2 - should have 2 rows
        result_v2 = read_ducklake(cat.metadata_path, "test", snapshot_version=v2)
        assert result_v2.shape[0] == 2
        assert sorted(result_v2["a"].to_list()) == [1, 2]

    def test_read_at_version_with_lazy(self, ducklake_catalog):
        cat = ducklake_catalog

        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        cat.execute("INSERT INTO ducklake.test VALUES (1)")
        v1 = cat.fetchone("SELECT * FROM ducklake_current_snapshot('ducklake')")[0]

        cat.execute("INSERT INTO ducklake.test VALUES (2), (3)")
        cat.close()

        result = (
            scan_ducklake(cat.metadata_path, "test", snapshot_version=v1)
            .filter(pl.col("a") == 1)
            .collect()
        )
        assert result.shape == (1, 1)

    def test_read_at_snapshot_time(self, ducklake_catalog):
        import sqlite3

        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        cat.execute("INSERT INTO ducklake.test VALUES (1)")
        # Get snapshot version after first insert
        v1 = cat.fetchone("SELECT * FROM ducklake_current_snapshot('ducklake')")[0]
        cat.execute("INSERT INTO ducklake.test VALUES (2)")
        cat.close()

        # Read the snapshot time for v1 from the SQLite catalog directly
        con = sqlite3.connect(cat.metadata_path)
        ts = con.execute(
            "SELECT snapshot_time FROM ducklake_snapshot WHERE snapshot_id = ?",
            [v1],
        ).fetchone()[0]
        con.close()

        # Read at the timestamp of v1 - should have 1 row
        result = read_ducklake(cat.metadata_path, "test", snapshot_time=ts)
        assert result.shape[0] == 1
        assert result["a"].to_list() == [1]

    def test_invalid_snapshot_version(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        cat.execute("INSERT INTO ducklake.test VALUES (1)")
        cat.close()

        with pytest.raises(ValueError, match="not found"):
            read_ducklake(cat.metadata_path, "test", snapshot_version=9999)

    def test_invalid_snapshot_time(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        cat.execute("INSERT INTO ducklake.test VALUES (1)")
        cat.close()

        with pytest.raises(ValueError, match="No snapshot found"):
            read_ducklake(cat.metadata_path, "test", snapshot_time="2000-01-01T00:00:00")
