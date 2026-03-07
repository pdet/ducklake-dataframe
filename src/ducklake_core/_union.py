"""UNION type detection and conversion utilities for DuckLake.

DuckLake does not natively support UNION types (upstream limitation).
This module provides utilities to detect UNION-typed columns in Arrow
tables and convert them to STRUCT representation, which Parquet can
store natively and Polars/DuckDB can read as struct columns.
"""

from __future__ import annotations

import pyarrow as pa

from ducklake_core._exceptions import UnsupportedUnionTypeError


def has_union_type(schema: pa.Schema | dict[str, pa.DataType]) -> bool:
    """Check whether any column in a schema contains a UNION type.

    Recursively inspects nested types (struct fields, list elements, etc.).
    """
    if isinstance(schema, pa.Schema):
        return any(_type_contains_union(field.type) for field in schema)
    return any(_type_contains_union(dtype) for dtype in schema.values())


def _type_contains_union(dtype: pa.DataType) -> bool:
    """Recursively check if a type is or contains a union type."""
    if pa.types.is_union(dtype):
        return True
    if pa.types.is_list(dtype) or pa.types.is_large_list(dtype):
        return _type_contains_union(dtype.value_type)
    if pa.types.is_struct(dtype):
        return any(_type_contains_union(field.type) for field in dtype)
    if pa.types.is_map(dtype):
        return (
            _type_contains_union(dtype.key_type)
            or _type_contains_union(dtype.item_type)
        )
    return False


def check_no_union_types(
    table: pa.Table,
    *,
    context: str = "",
) -> None:
    """Raise ``UnsupportedUnionTypeError`` if *table* has any UNION columns.

    Parameters
    ----------
    table
        Arrow table to check.
    context
        Optional context string included in the error message (e.g.
        table name).
    """
    union_cols = [
        field.name
        for field in table.schema
        if _type_contains_union(field.type)
    ]
    if union_cols:
        cols_str = ", ".join(repr(c) for c in union_cols)
        ctx = f" in table {context!r}" if context else ""
        raise UnsupportedUnionTypeError(
            f"UNION-typed column(s) {cols_str}{ctx} cannot be written to "
            f"DuckLake: the upstream catalog does not support UNION types. "
            f"Use union_handling='to_struct' to automatically convert UNION "
            f"columns to STRUCT before writing."
        )


def union_to_struct_type(dtype: pa.DataType) -> pa.DataType:
    """Convert a UNION type to an equivalent STRUCT type.

    For non-union types, returns the type unchanged. Recursively
    converts union types nested inside struct/list/map.
    """
    if pa.types.is_union(dtype):
        fields = []
        for i in range(dtype.num_fields):
            child = dtype[i]
            fields.append(
                pa.field(child.name, union_to_struct_type(child.type))
            )
        return pa.struct(fields)

    if pa.types.is_list(dtype) or pa.types.is_large_list(dtype):
        inner = union_to_struct_type(dtype.value_type)
        if inner != dtype.value_type:
            return pa.list_(inner)
        return dtype

    if pa.types.is_struct(dtype):
        new_fields = []
        changed = False
        for field in dtype:
            new_type = union_to_struct_type(field.type)
            if new_type != field.type:
                changed = True
            new_fields.append(pa.field(field.name, new_type))
        if changed:
            return pa.struct(new_fields)
        return dtype

    if pa.types.is_map(dtype):
        new_key = union_to_struct_type(dtype.key_type)
        new_val = union_to_struct_type(dtype.item_type)
        if new_key != dtype.key_type or new_val != dtype.item_type:
            return pa.map_(new_key, new_val)
        return dtype

    return dtype


def _convert_union_array(arr: pa.UnionArray) -> pa.StructArray:
    """Convert a single UnionArray to a StructArray.

    Each element gets the value in its active member field, with all
    other fields set to NULL (matching DuckDB's Parquet encoding of
    UNION types).
    """
    n = len(arr)
    union_type = arr.type
    num_fields = union_type.num_fields

    # Build per-field arrays: value when active, NULL otherwise
    field_values: dict[str, list] = {}
    field_types: dict[str, pa.DataType] = {}
    for fidx in range(num_fields):
        field = union_type[fidx]
        resolved = union_to_struct_type(field.type)
        field_values[field.name] = [None] * n
        field_types[field.name] = resolved

    type_codes = arr.type_codes
    # Build a validity mask (is_valid() returns a BooleanArray, not per-element)
    validity = arr.is_valid()

    for idx in range(n):
        if validity[idx].as_py():
            tc = type_codes[idx].as_py()
            field = union_type[tc]
            val = arr[idx].as_py()
            field_values[field.name][idx] = val

    struct_arrays = []
    struct_names = []
    for fidx in range(num_fields):
        field = union_type[fidx]
        struct_arrays.append(
            pa.array(field_values[field.name], type=field_types[field.name])
        )
        struct_names.append(field.name)

    return pa.StructArray.from_arrays(struct_arrays, names=struct_names)


def convert_unions_in_column(
    arr: pa.ChunkedArray, dtype: pa.DataType,
) -> pa.ChunkedArray:
    """Recursively convert union arrays to struct arrays in a column."""
    if pa.types.is_union(dtype):
        chunks = []
        for chunk in arr.chunks:
            chunks.append(_convert_union_array(chunk))
        new_type = union_to_struct_type(dtype)
        return pa.chunked_array(chunks, type=new_type)

    # For nested types containing unions, we need to rebuild
    if not _type_contains_union(dtype):
        return arr

    # For struct/list containing unions, convert via Python round-trip
    # (simpler and correct for the uncommon nested case)
    new_type = union_to_struct_type(dtype)
    py_vals = arr.to_pylist()
    return pa.chunked_array([pa.array(py_vals, type=new_type)])


def convert_unions_in_table(table: pa.Table) -> pa.Table:
    """Convert all UNION-typed columns in a table to STRUCT.

    Returns the table unchanged if no UNION columns are present.
    """
    if not has_union_type(table.schema):
        return table

    new_columns = []
    new_fields = []
    for i, field in enumerate(table.schema):
        col = table.column(i)
        if _type_contains_union(field.type):
            new_col = convert_unions_in_column(col, field.type)
            new_type = union_to_struct_type(field.type)
            new_fields.append(pa.field(field.name, new_type))
            new_columns.append(new_col)
        else:
            new_fields.append(field)
            new_columns.append(col)

    new_schema = pa.schema(new_fields, metadata=table.schema.metadata)
    return pa.table(
        {field.name: col for field, col in zip(new_fields, new_columns)},
        schema=new_schema,
    )


def convert_unions_in_schema(
    schema: dict[str, pa.DataType],
) -> dict[str, pa.DataType]:
    """Convert all UNION types in a schema dict to STRUCT types."""
    return {
        name: union_to_struct_type(dtype)
        for name, dtype in schema.items()
    }
