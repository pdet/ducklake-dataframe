"""Delete handling tests for ducklake-polars."""

from __future__ import annotations

from ducklake_polars import read_ducklake


class TestDeleteFiles:
    """Test reading tables that have had rows deleted."""

    def test_read_after_delete(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute(
            "CREATE TABLE ducklake.test AS SELECT i AS a FROM range(10) t(i)"
        )
        cat.execute("DELETE FROM ducklake.test WHERE a >= 5")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert sorted(result["a"].to_list()) == [0, 1, 2, 3, 4]

    def test_read_after_partial_delete(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER, b VARCHAR)")
        cat.execute(
            "INSERT INTO ducklake.test VALUES (1, 'a'), (2, 'b'), (3, 'c'), (4, 'd')"
        )
        cat.execute("DELETE FROM ducklake.test WHERE a = 2")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape[0] == 3
        assert sorted(result["a"].to_list()) == [1, 3, 4]

    def test_read_after_multiple_deletes(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute(
            "CREATE TABLE ducklake.test AS SELECT i AS a FROM range(20) t(i)"
        )
        cat.execute("DELETE FROM ducklake.test WHERE a < 5")
        cat.execute("DELETE FROM ducklake.test WHERE a >= 15")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert sorted(result["a"].to_list()) == list(range(5, 15))

    def test_read_after_delete_all(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        cat.execute("INSERT INTO ducklake.test VALUES (1), (2), (3)")
        cat.execute("DELETE FROM ducklake.test")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape[0] == 0

    def test_time_travel_before_delete(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER)")
        cat.execute("INSERT INTO ducklake.test VALUES (1), (2), (3)")
        v1 = cat.fetchone("SELECT * FROM ducklake_current_snapshot('ducklake')")[0]

        cat.execute("DELETE FROM ducklake.test WHERE a = 2")
        cat.close()

        # Current should have 2 rows
        result_current = read_ducklake(cat.metadata_path, "test")
        assert sorted(result_current["a"].to_list()) == [1, 3]

        # Time travel to before delete should have 3 rows
        result_v1 = read_ducklake(cat.metadata_path, "test", snapshot_version=v1)
        assert sorted(result_v1["a"].to_list()) == [1, 2, 3]
