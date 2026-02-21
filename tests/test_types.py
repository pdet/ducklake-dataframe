"""Type mapping tests for ducklake-polars."""

from __future__ import annotations

import polars as pl
import pytest

from ducklake_polars import read_ducklake


class TestScalarTypes:
    """Test reading tables with various scalar DuckDB types."""

    def test_integer_types(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("""
            CREATE TABLE ducklake.test (
                a TINYINT,
                b SMALLINT,
                c INTEGER,
                d BIGINT
            )
        """)
        cat.execute("INSERT INTO ducklake.test VALUES (1, 100, 10000, 1000000)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.schema == {
            "a": pl.Int8,
            "b": pl.Int16,
            "c": pl.Int32,
            "d": pl.Int64,
        }
        assert result.row(0) == (1, 100, 10000, 1000000)

    def test_unsigned_integer_types(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("""
            CREATE TABLE ducklake.test (
                a UTINYINT,
                b USMALLINT,
                c UINTEGER,
                d UBIGINT
            )
        """)
        cat.execute("INSERT INTO ducklake.test VALUES (1, 100, 10000, 1000000)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.schema == {
            "a": pl.UInt8,
            "b": pl.UInt16,
            "c": pl.UInt32,
            "d": pl.UInt64,
        }

    @pytest.mark.xfail(reason="DuckDB writes HUGEINT as Float64 in Parquet, Polars reads Float64")
    def test_hugeint(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a HUGEINT)")
        cat.execute("INSERT INTO ducklake.test VALUES (123456789012345678901234)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.schema["a"].base_type() == pl.Int128

    @pytest.mark.xfail(reason="DuckDB writes UHUGEINT as Float64 in Parquet, Polars reads Float64")
    def test_uhugeint(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a UHUGEINT)")
        cat.execute("INSERT INTO ducklake.test VALUES (123456789012345678901234)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.schema["a"].base_type() == pl.UInt128

    def test_float_types(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a FLOAT, b DOUBLE)")
        cat.execute("INSERT INTO ducklake.test VALUES (3.14, 2.718281828)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.schema == {"a": pl.Float32, "b": pl.Float64}
        assert abs(result["a"][0] - 3.14) < 0.01
        assert abs(result["b"][0] - 2.718281828) < 1e-6

    def test_boolean(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a BOOLEAN)")
        cat.execute("INSERT INTO ducklake.test VALUES (true), (false), (NULL)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.schema == {"a": pl.Boolean}
        assert result["a"].to_list() == [True, False, None]

    def test_varchar(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a VARCHAR)")
        cat.execute("INSERT INTO ducklake.test VALUES ('hello'), ('world'), ('')")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.schema == {"a": pl.String}
        assert result["a"].to_list() == ["hello", "world", ""]

    def test_blob(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a BLOB)")
        cat.execute("INSERT INTO ducklake.test VALUES ('\\x01\\x02\\x03'::BLOB)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.schema == {"a": pl.Binary}
        assert result["a"][0] == b"\x01\x02\x03"

    def test_date(self, ducklake_catalog):
        import datetime

        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a DATE)")
        cat.execute("INSERT INTO ducklake.test VALUES ('2024-01-15'), ('2024-12-31')")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.schema == {"a": pl.Date}
        assert result["a"].to_list() == [
            datetime.date(2024, 1, 15),
            datetime.date(2024, 12, 31),
        ]

    def test_timestamp(self, ducklake_catalog):
        import datetime

        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a TIMESTAMP)")
        cat.execute("INSERT INTO ducklake.test VALUES ('2024-01-15 10:30:00')")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.schema["a"].base_type() == pl.Datetime
        assert result["a"][0] == datetime.datetime(2024, 1, 15, 10, 30, 0)

    def test_timestamp_s(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a TIMESTAMP_S)")
        cat.execute("INSERT INTO ducklake.test VALUES ('2024-01-15 10:30:00')")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.schema["a"].base_type() == pl.Datetime

    def test_timestamp_ms(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a TIMESTAMP_MS)")
        cat.execute("INSERT INTO ducklake.test VALUES ('2024-01-15 10:30:00.123')")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.schema["a"].base_type() == pl.Datetime

    def test_timestamp_ns(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a TIMESTAMP_NS)")
        cat.execute("INSERT INTO ducklake.test VALUES ('2024-01-15 10:30:00.123456789')")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.schema["a"].base_type() == pl.Datetime

    def test_timestamptz(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a TIMESTAMPTZ)")
        cat.execute("INSERT INTO ducklake.test VALUES ('2024-01-15 10:30:00+00')")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.schema["a"].base_type() == pl.Datetime

    def test_time(self, ducklake_catalog):
        import datetime

        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a TIME)")
        cat.execute("INSERT INTO ducklake.test VALUES ('10:30:00'), ('23:59:59')")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.schema == {"a": pl.Time}
        assert result["a"].to_list() == [
            datetime.time(10, 30, 0),
            datetime.time(23, 59, 59),
        ]

    @pytest.mark.xfail(reason="Polars cannot read Parquet month_day_millisecond_interval type natively")
    def test_interval(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTERVAL)")
        cat.execute("INSERT INTO ducklake.test VALUES (INTERVAL 1 HOUR), (INTERVAL 30 MINUTE)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.schema["a"].base_type() == pl.Duration
        assert result.shape == (2, 1)

    def test_uuid(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a UUID)")
        cat.execute("INSERT INTO ducklake.test VALUES ('550e8400-e29b-41d4-a716-446655440000')")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.schema == {"a": pl.Binary}
        # UUID stored as 16-byte binary in Parquet
        uuid_bytes = result["a"][0]
        assert uuid_bytes is not None
        assert len(uuid_bytes) == 16

    def test_json(self, ducklake_catalog):
        import json

        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a JSON)")
        cat.execute("""INSERT INTO ducklake.test VALUES ('{"key": "value"}'), ('[1, 2, 3]')""")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.schema == {"a": pl.Binary}
        assert result.shape == (2, 1)
        # JSON text is stored as binary in Parquet, can be cast to String
        texts = result["a"].cast(pl.String).to_list()
        parsed_0 = json.loads(texts[0])
        assert parsed_0 == {"key": "value"}
        parsed_1 = json.loads(texts[1])
        assert parsed_1 == [1, 2, 3]

    def test_decimal(self, ducklake_catalog):
        from decimal import Decimal

        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a DECIMAL(10, 2))")
        cat.execute("INSERT INTO ducklake.test VALUES (123.45), (678.90)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape == (2, 1)
        values = result["a"].to_list()
        assert values == [Decimal("123.45"), Decimal("678.90")]

    def test_decimal_various_precisions(self, ducklake_catalog):
        from decimal import Decimal

        cat = ducklake_catalog
        cat.execute("""
            CREATE TABLE ducklake.test (
                a DECIMAL(5, 2),
                b DECIMAL(18, 6),
                c DECIMAL(38, 10)
            )
        """)
        cat.execute("INSERT INTO ducklake.test VALUES (123.45, 123456.789012, 12345678901234567890.1234567890)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape == (1, 3)
        assert result["a"][0] == Decimal("123.45")
        assert result["b"][0] == Decimal("123456.789012")


class TestComplexTypes:
    """Test reading tables with complex/nested DuckDB types."""

    def test_list_type(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER[])")
        cat.execute("INSERT INTO ducklake.test VALUES ([1, 2, 3]), ([4, 5])")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape == (2, 1)
        assert result["a"].to_list() == [[1, 2, 3], [4, 5]]

    def test_list_of_varchar(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a VARCHAR[])")
        cat.execute("INSERT INTO ducklake.test VALUES (['hello', 'world']), (['foo'])")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape == (2, 1)
        assert result["a"].to_list() == [["hello", "world"], ["foo"]]

    def test_struct_type(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a STRUCT(x INTEGER, y VARCHAR))")
        cat.execute("INSERT INTO ducklake.test VALUES ({'x': 1, 'y': 'hello'})")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape == (1, 1)
        val = result["a"][0]
        assert val == {"x": 1, "y": "hello"}

    @pytest.mark.xfail(reason="MAP type reading from DuckDB Parquet is broken in Polars 1.36")
    def test_map_type(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a MAP(VARCHAR, INTEGER))")
        cat.execute("INSERT INTO ducklake.test VALUES (MAP {'a': 1, 'b': 2})")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape == (1, 1)

    def test_nested_list_of_structs(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a STRUCT(x INTEGER, y VARCHAR)[])")
        cat.execute(
            "INSERT INTO ducklake.test VALUES "
            "([{'x': 1, 'y': 'a'}, {'x': 2, 'y': 'b'}])"
        )
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape == (1, 1)
        assert result["a"].to_list() == [[{"x": 1, "y": "a"}, {"x": 2, "y": "b"}]]

    def test_list_of_lists(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a INTEGER[][])")
        cat.execute("INSERT INTO ducklake.test VALUES ([[1, 2], [3, 4]]), ([[5]])")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape == (2, 1)
        assert result["a"].to_list() == [[[1, 2], [3, 4]], [[5]]]

    def test_struct_with_list_field(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("CREATE TABLE ducklake.test (a STRUCT(name VARCHAR, scores INTEGER[]))")
        cat.execute("INSERT INTO ducklake.test VALUES ({'name': 'Alice', 'scores': [90, 85, 92]})")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape == (1, 1)
        val = result["a"][0]
        assert val["name"] == "Alice"
        assert val["scores"] == [90, 85, 92]


class TestMixedColumns:
    """Test tables with multiple column types."""

    def test_mixed_types(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("""
            CREATE TABLE ducklake.test (
                id INTEGER,
                name VARCHAR,
                score DOUBLE,
                active BOOLEAN,
                created DATE
            )
        """)
        cat.execute("""
            INSERT INTO ducklake.test VALUES
                (1, 'Alice', 95.5, true, '2024-01-01'),
                (2, 'Bob', 87.3, false, '2024-01-02'),
                (3, 'Charlie', NULL, true, NULL)
        """)
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape == (3, 5)
        assert result["id"].to_list() == [1, 2, 3]
        assert result["name"].to_list() == ["Alice", "Bob", "Charlie"]
        assert result["active"].to_list() == [True, False, True]

    def test_all_temporal_types(self, ducklake_catalog):
        cat = ducklake_catalog
        cat.execute("""
            CREATE TABLE ducklake.test (
                a DATE,
                b TIME,
                c TIMESTAMP
            )
        """)
        cat.execute("""
            INSERT INTO ducklake.test VALUES
                ('2024-01-15', '10:30:00', '2024-01-15 10:30:00')
        """)
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.shape == (1, 3)
        assert result.schema["a"] == pl.Date
        assert result.schema["b"] == pl.Time
        assert result.schema["c"].base_type() == pl.Datetime

    def test_all_integer_widths(self, ducklake_catalog):
        """Test all integer types in a single table."""
        cat = ducklake_catalog
        cat.execute("""
            CREATE TABLE ducklake.test (
                a TINYINT,
                b SMALLINT,
                c INTEGER,
                d BIGINT,
                e UTINYINT,
                f USMALLINT,
                g UINTEGER,
                h UBIGINT
            )
        """)
        cat.execute("INSERT INTO ducklake.test VALUES (1, 2, 3, 4, 5, 6, 7, 8)")
        cat.close()

        result = read_ducklake(cat.metadata_path, "test")
        assert result.schema == {
            "a": pl.Int8,
            "b": pl.Int16,
            "c": pl.Int32,
            "d": pl.Int64,
            "e": pl.UInt8,
            "f": pl.UInt16,
            "g": pl.UInt32,
            "h": pl.UInt64,
        }
        assert result.row(0) == (1, 2, 3, 4, 5, 6, 7, 8)
