"""Pandas integration parity tests.

Ensures ducklake_pandas has feature parity with ducklake_polars
for core operations: read, write, delete, update, merge, rewrite.
"""

from __future__ import annotations

import os

import pandas as pd
import pytest

from ducklake_pandas import (
    read_ducklake,
    write_ducklake,
    delete_ducklake,
    update_ducklake,
    merge_ducklake,
    rewrite_data_files_ducklake,
    create_ducklake_table,
    drop_ducklake_table,
    alter_ducklake_add_column,
    alter_ducklake_drop_column,
    alter_ducklake_rename_column,
    expire_snapshots,
    vacuum_ducklake,
)


@pytest.fixture
def pandas_cat(tmp_path):
    """Create a DuckLake catalog for pandas tests."""
    import duckdb

    meta = str(tmp_path / "pandas_test.ducklake")
    data = str(tmp_path / "data")
    os.makedirs(data, exist_ok=True)

    con = duckdb.connect()
    con.install_extension("ducklake")
    con.load_extension("ducklake")
    con.execute(
        f"ATTACH 'ducklake:sqlite:{meta}' AS ducklake "
        f"(DATA_PATH '{data}', DATA_INLINING_ROW_LIMIT 0)"
    )
    con.close()

    class Cat:
        metadata_path = meta
        data_path = data
    return Cat()


class TestPandasReadWrite:
    """Basic read/write with Pandas."""

    def test_write_and_read(self, pandas_cat):
        cat = pandas_cat
        df = pd.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]})
        write_ducklake(df, cat.metadata_path, "t", data_path=cat.data_path)

        result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3

    def test_write_append(self, pandas_cat):
        cat = pandas_cat
        df1 = pd.DataFrame({"id": [1, 2]})
        write_ducklake(df1, cat.metadata_path, "t", data_path=cat.data_path)

        df2 = pd.DataFrame({"id": [3, 4]})
        write_ducklake(df2, cat.metadata_path, "t", mode="append",
                      data_path=cat.data_path)

        result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        assert len(result) == 4

    def test_write_overwrite(self, pandas_cat):
        cat = pandas_cat
        write_ducklake(pd.DataFrame({"id": [1, 2, 3]}),
                      cat.metadata_path, "t", data_path=cat.data_path)
        write_ducklake(pd.DataFrame({"id": [10]}),
                      cat.metadata_path, "t", mode="overwrite",
                      data_path=cat.data_path)

        result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        assert len(result) == 1
        assert result["id"].iloc[0] == 10


class TestPandasDML:
    """Delete, update, merge with Pandas."""

    def test_delete(self, pandas_cat):
        cat = pandas_cat
        write_ducklake(pd.DataFrame({"id": [1, 2, 3]}),
                      cat.metadata_path, "t", data_path=cat.data_path)
        delete_ducklake(cat.metadata_path, "t",
                       lambda df: df["id"] == 2,
                       data_path=cat.data_path)

        result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        assert set(result["id"].to_list()) == {1, 3}

    def test_update(self, pandas_cat):
        cat = pandas_cat
        write_ducklake(pd.DataFrame({"id": [1, 2], "name": ["a", "b"]}),
                      cat.metadata_path, "t", data_path=cat.data_path)
        update_ducklake(cat.metadata_path, "t", {"name": "X"},
                       lambda df: df["id"] == 1, data_path=cat.data_path)

        result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        row = result[result["id"] == 1]
        assert row["name"].iloc[0] == "X"


class TestPandasDDL:
    """Schema operations with Pandas."""

    def test_create_drop_table(self, pandas_cat):
        cat = pandas_cat
        create_ducklake_table(cat.metadata_path, "empty",
                             {"id": "BIGINT", "name": "VARCHAR"},
                             data_path=cat.data_path)
        drop_ducklake_table(cat.metadata_path, "empty",
                           data_path=cat.data_path)

    def test_add_drop_rename_column(self, pandas_cat):
        cat = pandas_cat
        write_ducklake(pd.DataFrame({"id": [1], "a": ["x"]}),
                      cat.metadata_path, "t", data_path=cat.data_path)

        alter_ducklake_add_column(cat.metadata_path, "t", "b", "BIGINT",
                                 data_path=cat.data_path)
        alter_ducklake_rename_column(cat.metadata_path, "t", "a", "name",
                                    data_path=cat.data_path)
        alter_ducklake_drop_column(cat.metadata_path, "t", "b",
                                  data_path=cat.data_path)

        result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        assert "name" in result.columns
        assert "b" not in result.columns


class TestPandasRewrite:
    """rewrite_data_files with Pandas."""

    def test_rewrite_merges_files(self, pandas_cat):
        cat = pandas_cat
        for i in range(3):
            df = pd.DataFrame({"id": [i]})
            write_ducklake(df, cat.metadata_path, "t", mode="append",
                          data_path=cat.data_path)

        snap = rewrite_data_files_ducklake(cat.metadata_path, "t",
                                           data_path=cat.data_path)
        assert snap > 0

        result = read_ducklake(cat.metadata_path, "t", data_path=cat.data_path)
        assert len(result) == 3

    def test_rewrite_noop(self, pandas_cat):
        cat = pandas_cat
        write_ducklake(pd.DataFrame({"id": [1, 2]}),
                      cat.metadata_path, "t", data_path=cat.data_path)

        snap = rewrite_data_files_ducklake(cat.metadata_path, "t",
                                           data_path=cat.data_path)
        assert snap == -1


class TestPandasMaintenance:
    """expire_snapshots and vacuum with Pandas."""

    def test_expire_and_vacuum(self, pandas_cat):
        cat = pandas_cat
        write_ducklake(pd.DataFrame({"id": [1]}),
                      cat.metadata_path, "t", data_path=cat.data_path)
        write_ducklake(pd.DataFrame({"id": [2]}),
                      cat.metadata_path, "t", mode="append",
                      data_path=cat.data_path)

        # These should not error
        expired = expire_snapshots(cat.metadata_path, keep_last_n=1,
                                  data_path=cat.data_path)
        assert expired >= 0

        removed = vacuum_ducklake(cat.metadata_path,
                                 data_path=cat.data_path)
        assert removed >= 0
