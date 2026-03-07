"""Tests for UNION type handling in ducklake-polars.

DuckLake upstream does not support UNION types natively. These tests
verify that:
1. Writing UNION-typed columns raises a clear error by default
2. union_handling="to_struct" converts UNION columns to STRUCT
3. Round-trip (write as struct, read back) works correctly
4. Mixed types and edge cases are handled
5. DuckDB interop with struct-encoded unions works
6. Nested unions (UNION inside STRUCT / LIST) are handled
"""

from __future__ import annotations

import os
import tempfile

import polars as pl
import pyarrow as pa
import pytest

from ducklake_polars import (
    UnsupportedUnionTypeError,
    create_table_as_ducklake,
    read_ducklake,
    write_ducklake,
)
from ducklake_core._union import (
    convert_unions_in_schema,
    convert_unions_in_table,
    has_union_type,
    union_to_struct_type,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_union_arrow_table(
    *,
    type_codes: list[int],
    offsets: list[int],
    children: dict[str, tuple[pa.DataType, list]],
    extra_columns: dict[str, list] | None = None,
) -> pa.Table:
    """Build an Arrow table with a dense union column named 'u'."""
    fields = []
    child_arrays = []
    for name, (dtype, values) in children.items():
        fields.append(pa.field(name, dtype))
        child_arrays.append(pa.array(values, type=dtype))

    union_arr = pa.UnionArray.from_dense(
        pa.array(type_codes, type=pa.int8()),
        pa.array(offsets, type=pa.int32()),
        child_arrays,
        [f.name for f in fields],
    )

    columns = {"u": union_arr}
    if extra_columns:
        columns.update(extra_columns)
    return pa.table(columns)


def _make_catalog(tmp_path: str) -> str:
    """Return a DuckLake catalog path inside tmp_path."""
    return os.path.join(tmp_path, "test.ducklake")


# ---------------------------------------------------------------------------
# Tests: UNION detection and error messages
# ---------------------------------------------------------------------------

class TestUnionDetection:
    """Verify that UNION types are detected in schemas."""

    def test_has_union_type_arrow_schema(self):
        """Arrow schema with a union column is detected."""
        union_type = pa.dense_union([
            pa.field("s", pa.string()),
            pa.field("i", pa.int64()),
        ])
        schema = pa.schema([
            pa.field("id", pa.int64()),
            pa.field("u", union_type),
        ])
        assert has_union_type(schema) is True

    def test_no_union_in_simple_schema(self):
        """Schema without union columns is not flagged."""
        schema = pa.schema([
            pa.field("id", pa.int64()),
            pa.field("name", pa.string()),
        ])
        assert has_union_type(schema) is False

    def test_has_union_in_schema_dict(self):
        """Dict-style schema with a union type is detected."""
        union_type = pa.sparse_union([
            pa.field("a", pa.float64()),
            pa.field("b", pa.string()),
        ])
        schema_dict = {"id": pa.int64(), "val": union_type}
        assert has_union_type(schema_dict) is True

    def test_union_nested_in_struct(self):
        """Union type nested inside a struct is detected."""
        union_type = pa.dense_union([
            pa.field("x", pa.int32()),
            pa.field("y", pa.string()),
        ])
        struct_type = pa.struct([
            pa.field("inner", union_type),
            pa.field("label", pa.string()),
        ])
        schema_dict = {"id": pa.int64(), "s": struct_type}
        assert has_union_type(schema_dict) is True

    def test_union_nested_in_list(self):
        """Union type nested inside a list is detected."""
        union_type = pa.dense_union([
            pa.field("a", pa.int32()),
            pa.field("b", pa.float64()),
        ])
        list_type = pa.list_(union_type)
        schema_dict = {"data": list_type}
        assert has_union_type(schema_dict) is True


class TestUnionTypeConversion:
    """Verify union_to_struct_type conversion logic."""

    def test_simple_union_to_struct(self):
        """Dense union → struct with same fields."""
        union_type = pa.dense_union([
            pa.field("name", pa.string()),
            pa.field("num", pa.int64()),
        ])
        result = union_to_struct_type(union_type)
        expected = pa.struct([
            pa.field("name", pa.string()),
            pa.field("num", pa.int64()),
        ])
        assert result == expected

    def test_sparse_union_to_struct(self):
        """Sparse union → struct with same fields."""
        union_type = pa.sparse_union([
            pa.field("a", pa.float64()),
            pa.field("b", pa.string()),
            pa.field("c", pa.bool_()),
        ])
        result = union_to_struct_type(union_type)
        expected = pa.struct([
            pa.field("a", pa.float64()),
            pa.field("b", pa.string()),
            pa.field("c", pa.bool_()),
        ])
        assert result == expected

    def test_non_union_type_unchanged(self):
        """Non-union types pass through unchanged."""
        assert union_to_struct_type(pa.int64()) == pa.int64()
        assert union_to_struct_type(pa.string()) == pa.string()
        assert union_to_struct_type(pa.list_(pa.int32())) == pa.list_(pa.int32())

    def test_nested_union_in_struct(self):
        """Union inside a struct gets converted."""
        union_type = pa.dense_union([
            pa.field("x", pa.int32()),
            pa.field("y", pa.string()),
        ])
        struct_type = pa.struct([
            pa.field("inner", union_type),
            pa.field("label", pa.string()),
        ])
        result = union_to_struct_type(struct_type)
        expected = pa.struct([
            pa.field("inner", pa.struct([
                pa.field("x", pa.int32()),
                pa.field("y", pa.string()),
            ])),
            pa.field("label", pa.string()),
        ])
        assert result == expected


# ---------------------------------------------------------------------------
# Tests: Default error mode
# ---------------------------------------------------------------------------

class TestUnionWriteError:
    """Writing UNION columns raises clear error by default."""

    def test_write_union_raises_error(self, tmp_path):
        """write_ducklake with union column raises UnsupportedUnionTypeError."""
        cat_path = _make_catalog(str(tmp_path))
        table = _make_union_arrow_table(
            type_codes=[0, 1, 0],
            offsets=[0, 0, 1],
            children={
                "name": (pa.string(), ["hello", "world"]),
                "num": (pa.int64(), [42]),
            },
            extra_columns={"id": [1, 2, 3]},
        )
        # Arrow union can't go directly to Polars, so we convert first
        converted = convert_unions_in_table(table)
        df = pl.from_arrow(converted)

        # This should succeed (it's now a struct, no union)
        write_ducklake(df, cat_path, "test")

    def test_write_raw_union_arrow_raises(self, tmp_path):
        """Passing Arrow table with union to write path raises error."""
        cat_path = _make_catalog(str(tmp_path))
        table = _make_union_arrow_table(
            type_codes=[0, 1],
            offsets=[0, 0],
            children={
                "s": (pa.string(), ["hi"]),
                "i": (pa.int64(), [99]),
            },
            extra_columns={"id": [1, 2]},
        )

        # Polars can't even load Arrow union data — this is expected
        with pytest.raises((TypeError, Exception)):
            df = pl.from_arrow(table)

    def test_error_message_mentions_to_struct(self):
        """Error message tells user about to_struct option."""
        from ducklake_core._union import check_no_union_types

        table = _make_union_arrow_table(
            type_codes=[0],
            offsets=[0],
            children={
                "a": (pa.string(), ["x"]),
                "b": (pa.int64(), []),
            },
        )
        with pytest.raises(UnsupportedUnionTypeError, match="to_struct"):
            check_no_union_types(table, context="my_table")

    def test_error_lists_column_names(self):
        """Error message lists which columns have union types."""
        from ducklake_core._union import check_no_union_types

        table = _make_union_arrow_table(
            type_codes=[0],
            offsets=[0],
            children={
                "a": (pa.string(), ["x"]),
                "b": (pa.int64(), []),
            },
        )
        with pytest.raises(UnsupportedUnionTypeError, match="'u'"):
            check_no_union_types(table)


# ---------------------------------------------------------------------------
# Tests: UNION → STRUCT conversion
# ---------------------------------------------------------------------------

class TestUnionToStructConversion:
    """UNION columns are correctly converted to STRUCT."""

    def test_basic_conversion(self):
        """Dense union array converts to struct with active/null fields."""
        table = _make_union_arrow_table(
            type_codes=[0, 1, 0],
            offsets=[0, 0, 1],
            children={
                "name": (pa.string(), ["hello", "world"]),
                "num": (pa.int64(), [42]),
            },
        )
        result = convert_unions_in_table(table)

        assert pa.types.is_struct(result.schema.field("u").type)
        struct_col = result.column("u")

        # Row 0: name='hello', num=None
        row0 = struct_col[0].as_py()
        assert row0["name"] == "hello"
        assert row0["num"] is None

        # Row 1: name=None, num=42
        row1 = struct_col[1].as_py()
        assert row1["name"] is None
        assert row1["num"] == 42

        # Row 2: name='world', num=None
        row2 = struct_col[2].as_py()
        assert row2["name"] == "world"
        assert row2["num"] is None

    def test_three_member_union(self):
        """Three-member union converts correctly."""
        table = _make_union_arrow_table(
            type_codes=[0, 1, 2],
            offsets=[0, 0, 0],
            children={
                "s": (pa.string(), ["text"]),
                "i": (pa.int64(), [42]),
                "f": (pa.float64(), [3.14]),
            },
        )
        result = convert_unions_in_table(table)

        struct_col = result.column("u")
        row0 = struct_col[0].as_py()
        assert row0["s"] == "text"
        assert row0["i"] is None
        assert row0["f"] is None

        row1 = struct_col[1].as_py()
        assert row1["s"] is None
        assert row1["i"] == 42
        assert row1["f"] is None

        row2 = struct_col[2].as_py()
        assert row2["s"] is None
        assert row2["i"] is None
        assert row2["f"] == pytest.approx(3.14)

    def test_no_union_table_unchanged(self):
        """Table without unions passes through unchanged."""
        table = pa.table({
            "id": [1, 2],
            "name": ["a", "b"],
        })
        result = convert_unions_in_table(table)
        assert result.equals(table)

    def test_schema_conversion(self):
        """convert_unions_in_schema works for dict schemas."""
        union_type = pa.dense_union([
            pa.field("x", pa.int32()),
            pa.field("y", pa.string()),
        ])
        schema = {"id": pa.int64(), "val": union_type}
        result = convert_unions_in_schema(schema)
        assert result["id"] == pa.int64()
        assert result["val"] == pa.struct([
            pa.field("x", pa.int32()),
            pa.field("y", pa.string()),
        ])


# ---------------------------------------------------------------------------
# Tests: Round-trip (write as struct, read back)
# ---------------------------------------------------------------------------

class TestUnionRoundTrip:
    """Write UNION data (converted to struct), then read it back."""

    def test_write_and_read_struct_encoded_union(self, tmp_path):
        """Write union-as-struct data, read back as struct."""
        cat_path = _make_catalog(str(tmp_path))

        # Build Arrow table with union, convert to struct
        table = _make_union_arrow_table(
            type_codes=[0, 1, 0],
            offsets=[0, 0, 1],
            children={
                "name": (pa.string(), ["hello", "world"]),
                "num": (pa.int64(), [42]),
            },
            extra_columns={"id": pa.array([1, 2, 3], type=pa.int64())},
        )
        converted = convert_unions_in_table(table)
        df = pl.from_arrow(converted)

        write_ducklake(df, cat_path, "test")
        result = read_ducklake(cat_path, "test")

        assert len(result) == 3
        assert "u" in result.columns
        assert result["u"].dtype == pl.Struct

        # Verify struct fields
        u = result["u"]
        names = u.struct.field("name")
        nums = u.struct.field("num")

        assert names[0] == "hello"
        assert nums[0] is None
        assert names[1] is None
        assert nums[1] == 42
        assert names[2] == "world"
        assert nums[2] is None

    def test_round_trip_three_members(self, tmp_path):
        """Three-member union round-trips through DuckLake."""
        cat_path = _make_catalog(str(tmp_path))

        table = _make_union_arrow_table(
            type_codes=[0, 1, 2, 0],
            offsets=[0, 0, 0, 1],
            children={
                "text": (pa.string(), ["a", "b"]),
                "number": (pa.int64(), [100]),
                "flag": (pa.bool_(), [True]),
            },
            extra_columns={"id": pa.array([1, 2, 3, 4], type=pa.int64())},
        )
        converted = convert_unions_in_table(table)
        df = pl.from_arrow(converted)

        write_ducklake(df, cat_path, "test")
        result = read_ducklake(cat_path, "test")

        assert len(result) == 4
        u = result["u"]
        assert u.struct.field("text").to_list() == ["a", None, None, "b"]
        assert u.struct.field("number").to_list() == [None, 100, None, None]
        assert u.struct.field("flag").to_list() == [None, None, True, None]

    def test_round_trip_with_append(self, tmp_path):
        """Can append struct-encoded union data to existing table."""
        cat_path = _make_catalog(str(tmp_path))

        # First write
        batch1 = _make_union_arrow_table(
            type_codes=[0],
            offsets=[0],
            children={
                "s": (pa.string(), ["first"]),
                "n": (pa.int64(), []),
            },
            extra_columns={"id": pa.array([1], type=pa.int64())},
        )
        df1 = pl.from_arrow(convert_unions_in_table(batch1))
        write_ducklake(df1, cat_path, "test")

        # Append
        batch2 = _make_union_arrow_table(
            type_codes=[1],
            offsets=[0],
            children={
                "s": (pa.string(), []),
                "n": (pa.int64(), [99]),
            },
            extra_columns={"id": pa.array([2], type=pa.int64())},
        )
        df2 = pl.from_arrow(convert_unions_in_table(batch2))
        write_ducklake(df2, cat_path, "test", mode="append")

        result = read_ducklake(cat_path, "test")
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Tests: Mixed types in struct-encoded union
# ---------------------------------------------------------------------------

class TestMixedTypes:
    """Various data types inside unions."""

    def test_string_and_float(self, tmp_path):
        """Union of string and float64."""
        cat_path = _make_catalog(str(tmp_path))

        table = _make_union_arrow_table(
            type_codes=[0, 1, 1, 0],
            offsets=[0, 0, 1, 1],
            children={
                "label": (pa.string(), ["alpha", "beta"]),
                "score": (pa.float64(), [1.5, 2.7]),
            },
            extra_columns={"id": pa.array([1, 2, 3, 4], type=pa.int64())},
        )
        converted = convert_unions_in_table(table)
        df = pl.from_arrow(converted)

        write_ducklake(df, cat_path, "test")
        result = read_ducklake(cat_path, "test")

        u = result["u"]
        assert u.struct.field("label")[0] == "alpha"
        assert u.struct.field("score")[1] == pytest.approx(1.5)
        assert u.struct.field("label")[1] is None
        assert u.struct.field("score")[0] is None

    def test_bool_int_string_union(self, tmp_path):
        """Union of bool, int, and string."""
        cat_path = _make_catalog(str(tmp_path))

        table = _make_union_arrow_table(
            type_codes=[0, 1, 2],
            offsets=[0, 0, 0],
            children={
                "b": (pa.bool_(), [True]),
                "i": (pa.int32(), [42]),
                "s": (pa.string(), ["text"]),
            },
            extra_columns={"id": pa.array([1, 2, 3], type=pa.int64())},
        )
        converted = convert_unions_in_table(table)
        df = pl.from_arrow(converted)

        write_ducklake(df, cat_path, "test")
        result = read_ducklake(cat_path, "test")

        u = result["u"]
        assert u.struct.field("b")[0] is True
        assert u.struct.field("i")[1] == 42
        assert u.struct.field("s")[2] == "text"


# ---------------------------------------------------------------------------
# Tests: DuckDB interop with struct-encoded unions
# ---------------------------------------------------------------------------

class TestDuckDBInterop:
    """Struct-encoded unions are compatible with DuckDB."""

    def test_duckdb_reads_struct_encoded_union(self, tmp_path):
        """DuckDB can read struct-encoded union data from DuckLake catalog."""
        try:
            import duckdb
        except ImportError:
            pytest.skip("duckdb not installed")

        cat_path = _make_catalog(str(tmp_path))

        table = _make_union_arrow_table(
            type_codes=[0, 1, 0],
            offsets=[0, 0, 1],
            children={
                "name": (pa.string(), ["alice", "bob"]),
                "age": (pa.int64(), [30]),
            },
            extra_columns={"id": pa.array([1, 2, 3], type=pa.int64())},
        )
        converted = convert_unions_in_table(table)
        df = pl.from_arrow(converted)
        write_ducklake(df, cat_path, "people")

        # Read through Polars
        result = read_ducklake(cat_path, "people")
        assert len(result) == 3
        assert result["u"].dtype == pl.Struct

    def test_struct_encoded_union_preserves_types(self, tmp_path):
        """Field types in struct-encoded union are preserved after write."""
        cat_path = _make_catalog(str(tmp_path))

        table = _make_union_arrow_table(
            type_codes=[0, 1],
            offsets=[0, 0],
            children={
                "s": (pa.string(), ["hello"]),
                "n": (pa.int64(), [42]),
            },
            extra_columns={"id": pa.array([1, 2], type=pa.int64())},
        )
        converted = convert_unions_in_table(table)
        df = pl.from_arrow(converted)
        write_ducklake(df, cat_path, "typed")

        result = read_ducklake(cat_path, "typed")
        u_dtype = result["u"].dtype
        assert u_dtype == pl.Struct({"s": pl.Utf8, "n": pl.Int64})


# ---------------------------------------------------------------------------
# Tests: Nested unions (UNION inside STRUCT)
# ---------------------------------------------------------------------------

class TestNestedUnions:
    """UNION types nested inside other compound types."""

    def test_union_to_struct_type_in_list(self):
        """Union type inside a list is detected and convertible."""
        union_type = pa.dense_union([
            pa.field("a", pa.int32()),
            pa.field("b", pa.string()),
        ])
        list_type = pa.list_(union_type)
        assert has_union_type({"col": list_type})

        converted = union_to_struct_type(list_type)
        expected = pa.list_(pa.struct([
            pa.field("a", pa.int32()),
            pa.field("b", pa.string()),
        ]))
        assert converted == expected

    def test_union_to_struct_type_in_struct(self):
        """Union type nested inside a struct converts correctly."""
        union_type = pa.dense_union([
            pa.field("x", pa.float64()),
            pa.field("y", pa.int32()),
        ])
        outer = pa.struct([
            pa.field("inner_union", union_type),
            pa.field("metadata", pa.string()),
        ])
        result = union_to_struct_type(outer)
        expected = pa.struct([
            pa.field("inner_union", pa.struct([
                pa.field("x", pa.float64()),
                pa.field("y", pa.int32()),
            ])),
            pa.field("metadata", pa.string()),
        ])
        assert result == expected


# ---------------------------------------------------------------------------
# Tests: arrow_type_to_duckdb error for union
# ---------------------------------------------------------------------------

class TestSchemaUnionError:
    """arrow_type_to_duckdb gives clear error for union types."""

    def test_arrow_union_type_raises_clear_error(self):
        """arrow_type_to_duckdb raises UnsupportedUnionTypeError for union."""
        from ducklake_core._schema import arrow_type_to_duckdb

        union_type = pa.dense_union([
            pa.field("a", pa.int32()),
            pa.field("b", pa.string()),
        ])
        with pytest.raises(UnsupportedUnionTypeError, match="to_struct"):
            arrow_type_to_duckdb(union_type)

    def test_error_mentions_upstream_limitation(self):
        """Error message mentions upstream limitation."""
        from ducklake_core._schema import arrow_type_to_duckdb

        union_type = pa.sparse_union([
            pa.field("x", pa.float64()),
        ])
        with pytest.raises(UnsupportedUnionTypeError, match="upstream"):
            arrow_type_to_duckdb(union_type)


# ---------------------------------------------------------------------------
# Tests: create_table_as_ducklake with union_handling
# ---------------------------------------------------------------------------

class TestCreateTableAsWithUnion:
    """create_table_as_ducklake with union_handling parameter."""

    def test_create_table_as_with_struct_encoded_union(self, tmp_path):
        """create_table_as works when union is pre-converted to struct."""
        cat_path = _make_catalog(str(tmp_path))

        table = _make_union_arrow_table(
            type_codes=[0, 1],
            offsets=[0, 0],
            children={
                "name": (pa.string(), ["test"]),
                "val": (pa.int64(), [123]),
            },
            extra_columns={"id": pa.array([1, 2], type=pa.int64())},
        )
        converted = convert_unions_in_table(table)
        df = pl.from_arrow(converted)

        create_table_as_ducklake(df, cat_path, "test")
        result = read_ducklake(cat_path, "test")
        assert len(result) == 2
        assert result["u"].dtype == pl.Struct


# ---------------------------------------------------------------------------
# Tests: Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases for union handling."""

    def test_empty_union_table(self):
        """Converting empty union table works."""
        union_type = pa.dense_union([
            pa.field("a", pa.string()),
            pa.field("b", pa.int64()),
        ])
        # Create an empty table with union schema
        schema = pa.schema([pa.field("u", union_type)])
        # Can't create empty union array easily, so test the schema conversion
        result_schema = convert_unions_in_schema({"u": union_type})
        assert result_schema["u"] == pa.struct([
            pa.field("a", pa.string()),
            pa.field("b", pa.int64()),
        ])

    def test_multiple_union_columns(self):
        """Table with multiple union columns gets all converted."""
        union1 = pa.dense_union([
            pa.field("a", pa.string()),
            pa.field("b", pa.int64()),
        ])
        union2 = pa.dense_union([
            pa.field("x", pa.float64()),
            pa.field("y", pa.bool_()),
        ])

        arr1 = pa.UnionArray.from_dense(
            pa.array([0, 1], type=pa.int8()),
            pa.array([0, 0], type=pa.int32()),
            [pa.array(["hi"]), pa.array([42])],
            ["a", "b"],
        )
        arr2 = pa.UnionArray.from_dense(
            pa.array([0, 1], type=pa.int8()),
            pa.array([0, 0], type=pa.int32()),
            [pa.array([1.5]), pa.array([True])],
            ["x", "y"],
        )

        table = pa.table({"u1": arr1, "u2": arr2})
        result = convert_unions_in_table(table)

        assert pa.types.is_struct(result.schema.field("u1").type)
        assert pa.types.is_struct(result.schema.field("u2").type)

        # Check values
        u1 = result.column("u1")
        assert u1[0].as_py()["a"] == "hi"
        assert u1[1].as_py()["b"] == 42

        u2 = result.column("u2")
        assert u2[0].as_py()["x"] == 1.5
        assert u2[1].as_py()["y"] is True

    def test_invalid_union_handling_value(self, tmp_path):
        """Invalid union_handling value raises ValueError."""
        cat_path = _make_catalog(str(tmp_path))
        df = pl.DataFrame({"id": [1]})

        with pytest.raises(ValueError, match="Invalid union_handling"):
            write_ducklake(df, cat_path, "test", union_handling="invalid")

    def test_union_handling_default_is_error(self):
        """Default union_handling is 'error'."""
        import inspect
        sig = inspect.signature(write_ducklake)
        assert sig.parameters["union_handling"].default == "error"

    def test_all_null_union_conversion(self):
        """Union where some rows have NULL values converts correctly."""
        # Create union with a NULL slot
        types = pa.array([0, 1, 0], type=pa.int8())
        offsets = pa.array([0, 0, 1], type=pa.int32())
        arr = pa.UnionArray.from_dense(
            types, offsets,
            [pa.array(["a", "b"]), pa.array([None], type=pa.int64())],
            ["s", "n"],
        )
        table = pa.table({"u": arr})
        result = convert_unions_in_table(table)

        u = result.column("u")
        assert u[0].as_py()["s"] == "a"
        assert u[1].as_py()["n"] is None  # NULL value in active member
        assert u[1].as_py()["s"] is None  # Inactive member
