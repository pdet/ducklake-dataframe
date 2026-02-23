# ducklake-polars

> **This project is a proof of concept. It was 100% written by [Claude Code](https://docs.anthropic.com/en/docs/build-with-claude/claude-code/overview) (Anthropic's AI coding agent). It is not intended for production use.**

A pure-Python [Polars](https://pola.rs/) integration for [DuckLake](https://ducklake.select/) catalogs — both read and write.

Reads and writes DuckLake metadata directly from SQLite or PostgreSQL and scans the underlying Parquet data files through Polars' native Parquet reader. **No DuckDB runtime dependency.** You get lazy evaluation, predicate pushdown, projection pushdown, file pruning, and all other Polars optimizations out of the box.

## Installation

```bash
pip install ducklake-polars

# With PostgreSQL catalog support
pip install ducklake-polars[postgres]
```

The only runtime dependency is `polars >= 1.0`. SQLite catalogs use Python's built-in `sqlite3`. PostgreSQL catalogs require the `postgres` extra (adds `psycopg2`).

## Quick start

### Reading data

```python
import polars as pl
from ducklake_polars import scan_ducklake, read_ducklake

# Eager read
df = read_ducklake("catalog.ducklake", "my_table")

# Lazy scan (recommended for large tables)
lf = scan_ducklake("catalog.ducklake", "my_table")
result = lf.filter(pl.col("x") > 100).select("x", "y").collect()

# Time travel
df = read_ducklake("catalog.ducklake", "my_table", snapshot_version=3)
df = read_ducklake("catalog.ducklake", "my_table", snapshot_time="2025-01-15T10:30:00")

# PostgreSQL-backed catalog
df = read_ducklake("postgresql://user:pass@localhost/mydb", "my_table")
```

### Writing data

```python
from ducklake_polars import write_ducklake

df = pl.DataFrame({"id": [1, 2, 3], "name": ["Alice", "Bob", "Carol"]})

# Create and populate a new table
write_ducklake(df, "catalog.ducklake", "users", mode="error")

# Append rows
write_ducklake(new_rows, "catalog.ducklake", "users", mode="append")

# Overwrite all data
write_ducklake(df, "catalog.ducklake", "users", mode="overwrite")
```

### DDL operations

```python
from ducklake_polars import (
    create_ducklake_table,
    alter_ducklake_add_column,
    alter_ducklake_drop_column,
    alter_ducklake_rename_column,
    alter_ducklake_set_partitioned_by,
    drop_ducklake_table,
    rename_ducklake_table,
    create_ducklake_schema,
    drop_ducklake_schema,
    create_ducklake_view,
    drop_ducklake_view,
)

# Schema management
create_ducklake_schema("catalog.ducklake", "analytics")
drop_ducklake_schema("catalog.ducklake", "analytics", cascade=True)

# Table management
create_ducklake_table("catalog.ducklake", "events", {"ts": pl.Datetime("us"), "value": pl.Float64})
rename_ducklake_table("catalog.ducklake", "events", "event_log")
drop_ducklake_table("catalog.ducklake", "event_log")

# Column management
alter_ducklake_add_column("catalog.ducklake", "users", "email", pl.String)
alter_ducklake_rename_column("catalog.ducklake", "users", "email", "contact_email")
alter_ducklake_drop_column("catalog.ducklake", "users", "contact_email")

# Partitioning
alter_ducklake_set_partitioned_by("catalog.ducklake", "events", ["region", "date"])

# Views
create_ducklake_view("catalog.ducklake", "active_users", "SELECT * FROM users WHERE active = true")
drop_ducklake_view("catalog.ducklake", "active_users")
```

### DML operations

```python
from ducklake_polars import delete_ducklake, update_ducklake, merge_ducklake

# Delete rows matching a predicate
deleted = delete_ducklake("catalog.ducklake", "users", pl.col("active") == False)

# Update rows
updated = update_ducklake(
    "catalog.ducklake", "users",
    updates={"status": "inactive"},
    predicate=pl.col("last_login") < "2024-01-01",
)

# Merge (upsert)
rows_updated, rows_inserted = merge_ducklake(
    "catalog.ducklake", "users", source_df, on="id",
    when_matched_update=True,
    when_not_matched_insert=True,
)
```

### Catalog inspection

```python
from ducklake_polars import DuckLakeCatalog

catalog = DuckLakeCatalog("catalog.ducklake")

catalog.snapshots()           # All snapshots
catalog.current_snapshot()    # Latest snapshot ID
catalog.list_schemas()        # All schemas
catalog.list_tables()         # Tables in a schema
catalog.table_info()          # Per-table storage metadata
catalog.list_files("users")   # Data files and delete files
catalog.options()             # Catalog key-value metadata
catalog.settings()            # Backend type and data path

# Change data feed
catalog.table_insertions("users", start_version=1, end_version=5)
catalog.table_deletions("users", start_version=1, end_version=5)
catalog.table_changes("users", start_version=1, end_version=5)
```

### Maintenance

```python
from ducklake_polars import expire_snapshots, vacuum_ducklake

# Expire old snapshots (metadata cleanup)
expired = expire_snapshots("catalog.ducklake", keep_last_n=10)

# Delete orphaned Parquet files (disk cleanup)
deleted = vacuum_ducklake("catalog.ducklake")
```

## Features

### Read path
- **Lazy and eager reads** via `scan_ducklake()` / `read_ducklake()`
- **Predicate and projection pushdown** through Polars' native optimizer
- **File pruning** via column-level min/max statistics and partition values
- **Time travel** by snapshot version or timestamp
- **Delete file handling** via Polars' Iceberg-compatible positional deletes
- **Schema evolution** — ADD COLUMN, DROP COLUMN, RENAME COLUMN all handled transparently
- **Inlined data** — small tables stored directly in catalog metadata
- **Partition pruning** for identity-transform partitions
- **Column renames** — old Parquet files with old names seamlessly reconciled

### Write path
- **INSERT** — append, overwrite, or error-on-exists modes
- **DELETE** — predicate-based row deletion with position-delete files
- **UPDATE** — atomic delete + insert in a single snapshot
- **MERGE** — upsert with configurable matched/unmatched behavior
- **CREATE TABLE AS** — single-snapshot table creation with data
- **Data inlining** — small inserts stored as rows in catalog metadata
- **Partitioned writes** — Hive-style directory layout per partition key

### DDL
- **CREATE/DROP TABLE** with full snapshot versioning
- **ADD/DROP/RENAME COLUMN** with schema evolution tracking
- **CREATE/DROP SCHEMA** with cascade support
- **RENAME TABLE** preserving table identity
- **SET PARTITIONED BY** for identity-transform partitioning
- **CREATE/DROP VIEW** with `OR REPLACE` support

### Catalog inspection
- Snapshot history and time travel metadata
- Per-table storage statistics (file counts, sizes)
- Data file and delete file listing
- Schema and table enumeration
- Key-value catalog options
- **Change data feed** — insertions, deletions, and update detection

### Maintenance
- **expire_snapshots** — remove old snapshot metadata
- **vacuum** — delete orphaned Parquet files

### Backend support
- **SQLite** — via Python stdlib `sqlite3` (zero-dependency)
- **PostgreSQL** — via `psycopg2` (optional extra)
- Full interoperability with DuckDB's DuckLake extension

## DuckDB interoperability

ducklake-polars produces catalogs that are fully interoperable with DuckDB's DuckLake extension. You can:

- Create catalogs with DuckDB, read/write with ducklake-polars
- Create catalogs with ducklake-polars, read/query with DuckDB
- Mix operations freely — both tools read the same metadata format

```python
# Create catalog with DuckDB
import duckdb
con = duckdb.connect()
con.execute("INSTALL ducklake; LOAD ducklake; INSTALL sqlite_scanner; LOAD sqlite_scanner")
con.execute("ATTACH 'ducklake:sqlite:catalog.ducklake' AS lake (DATA_PATH 'data/')")
con.execute("CREATE TABLE lake.users (id INTEGER, name VARCHAR)")
con.execute("INSERT INTO lake.users VALUES (1, 'Alice'), (2, 'Bob')")
con.close()

# Read with ducklake-polars
from ducklake_polars import read_ducklake
df = read_ducklake("catalog.ducklake", "users")
```

See the [DuckDB Interop Guide](https://github.com/pdet/ducklake-polars/wiki/DuckDB-Interop) for detailed interop patterns.

## Supported data types

| DuckLake / DuckDB type | Polars type | Notes |
|---|---|---|
| `TINYINT` / `int8` | `Int8` | |
| `SMALLINT` / `int16` | `Int16` | |
| `INTEGER` / `int32` | `Int32` | |
| `BIGINT` / `int64` | `Int64` | |
| `UTINYINT` / `uint8` | `UInt8` | |
| `USMALLINT` / `uint16` | `UInt16` | |
| `UINTEGER` / `uint32` | `UInt32` | |
| `UBIGINT` / `uint64` | `UInt64` | |
| `FLOAT` / `float32` | `Float32` | |
| `DOUBLE` / `float64` | `Float64` | |
| `BOOLEAN` | `Boolean` | |
| `VARCHAR` | `String` | |
| `BLOB` | `Binary` | |
| `DATE` | `Date` | |
| `TIME` / `time_ns` / `timetz` | `Time` | |
| `TIMESTAMP` / `timestamp_us` | `Datetime("us")` | |
| `TIMESTAMP_MS` | `Datetime("ms")` | |
| `TIMESTAMP_NS` | `Datetime("ns")` | |
| `TIMESTAMP_S` | `Datetime("us")` | DuckDB writes as microseconds in Parquet |
| `TIMESTAMPTZ` | `Datetime("us", "UTC")` | |
| `DECIMAL(p, s)` | `Decimal(p, s)` | |
| `UUID` | `Binary` | 16-byte binary in Parquet |
| `JSON` | `Binary` | Cast to `String` for text access |
| `HUGEINT` | `Int128` | Limited: DuckDB writes as Float64 in Parquet |
| `UHUGEINT` | `UInt128` | Limited: DuckDB writes as Float64 in Parquet |
| `INTERVAL` | `Duration("us")` | Limited: Polars Parquet reader limitation |
| `LIST(T)` | `List(T)` | Recursive nesting supported |
| `STRUCT(...)` | `Struct(...)` | Recursive nesting supported |
| `MAP(K, V)` | `List(Struct(key, value))` | Limited: Polars Parquet reader issue |
| `GEOMETRY` | `Binary` | |
| `VARIANT` | `String` | |

## Architecture

```
src/ducklake_polars/
    __init__.py       Public API (all functions and DuckLakeCatalog)
    _backend.py       Backend adapters (SQLite, PostgreSQL)
    _catalog.py       Metadata reader (snapshots, tables, columns, files, stats)
    _catalog_api.py   DuckLakeCatalog inspection class
    _dataset.py       Polars PythonDatasetProvider implementation
    _schema.py        DuckLake type -> Polars type mapping
    _stats.py         Column statistics for file pruning
    _writer.py        Catalog writer (tables, data, DDL, views, maintenance)
```

See the [Architecture Overview](https://github.com/pdet/ducklake-polars/wiki/Architecture) for a detailed deep-dive.

## Development

```bash
git clone https://github.com/pdet/ducklake-polars.git
cd ducklake-polars
pip install -e ".[dev]"
```

### Running tests

```bash
pytest                    # Full suite (SQLite backend)
pytest -n auto            # Parallel execution
pytest -k "test_views"    # Specific pattern

# With PostgreSQL backend
DUCKLAKE_PG_DSN="postgresql://user:pass@localhost/testdb" pytest
```

Test suite: **590 tests** (5 xfailed for known DuckDB/Polars limitations). Tests are parametrized over backends — SQLite always runs; PostgreSQL runs when `DUCKLAKE_PG_DSN` is set.

## Documentation

- [Architecture Overview](https://github.com/pdet/ducklake-polars/wiki/Architecture)
- [API Reference](https://github.com/pdet/ducklake-polars/wiki/API-Reference)
- [Configuration Guide](https://github.com/pdet/ducklake-polars/wiki/Configuration)
- [Feature Examples](https://github.com/pdet/ducklake-polars/wiki/Examples)
- [DuckDB Interop Guide](https://github.com/pdet/ducklake-polars/wiki/DuckDB-Interop)

## License

MIT
