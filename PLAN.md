# DuckLake-Polars: Implementation Plan

## Project Summary

`ducklake-polars` is a Python package that provides Polars integration for DuckLake catalogs. It enables reading DuckLake tables as Polars LazyFrames/DataFrames with full support for predicate pushdown, schema evolution, time travel, and partition pruning.

---

## Architecture

```
User API:
    scan_ducklake(path, table, ...) -> LazyFrame
    read_ducklake(path, table, ...) -> DataFrame

Internal:
    DuckLakeDataset (PythonDatasetProvider)
        |-- schema()           -> resolve column types from ducklake_column
        |-- to_dataset_scan()  -> resolve file list, stats, deletions
                |
                v
    DuckLakeCatalogReader (metadata access via duckdb python package)
        |-- connect to metadata catalog (.ducklake / .db file)
        |-- query ducklake_* tables for schema, files, stats, partitions
        |
        v
    polars.scan_parquet(
        sources=resolved_parquet_paths,
        _table_statistics=min_max_stats,
        _deletion_files=delete_file_paths,
        _column_mapping=field_id_mapping,
        hive_partitioning=True/False,
        ...
    )
```

### Dependencies

- `polars` (>= 1.0)
- `duckdb` (>= 1.3) - for reading the metadata catalog

---

## Package Structure

```
ducklake-polars/
    pyproject.toml
    README.md
    DISCOVERY.md
    PLAN.md
    src/
        ducklake_polars/
            __init__.py          - public API: scan_ducklake, read_ducklake
            _catalog.py          - DuckLakeCatalogReader: metadata queries
            _dataset.py          - DuckLakeDataset: PythonDatasetProvider impl
            _schema.py           - DuckLake -> Polars type mapping
            _stats.py            - Statistics extraction for file pruning
            _utils.py            - Helpers (path resolution, etc.)
    tests/
        conftest.py              - shared fixtures (create_ducklake_catalog, etc.)
        test_basic.py            - basic scan/read operations
        test_types.py            - all supported type round-trips
        test_schema_evolution.py - ADD/DROP/RENAME COLUMN handling
        test_partitioning.py     - partition pruning
        test_time_travel.py      - version/timestamp queries
        test_filter_pushdown.py  - statistics-based file skipping
        test_delete.py           - deletion file handling
        test_data_inlining.py    - inlined data reads
        test_table_changes.py    - change data feed
```

---

## Implementation Phases

### Phase 1: Core Read Path (MVP)

**Goal**: `scan_ducklake()` returns a LazyFrame that reads a DuckLake table.

#### 1.1 Project Setup
- [ ] Create `pyproject.toml` with dependencies (polars, duckdb)
- [ ] Set up package structure (`src/ducklake_polars/`)
- [ ] Set up test infrastructure with pytest
- [ ] Create shared test fixtures that use DuckDB+DuckLake extension to create test catalogs

#### 1.2 Metadata Catalog Reader (`_catalog.py`)
- [ ] Connect to DuckLake metadata database (`.ducklake` file via duckdb)
- [ ] Read `ducklake_metadata` for catalog version, data_path, encryption status
- [ ] Read `ducklake_schema` for schema enumeration
- [ ] Read `ducklake_table` for table enumeration and metadata
- [ ] Read `ducklake_column` for column definitions (name, type, field_id, default, not_null)
- [ ] Read `ducklake_data_file` for Parquet file list (path, record_count, file_size)
- [ ] Read `ducklake_file_column_stats` for per-file min/max/null_count statistics
- [ ] Read `ducklake_delete_file` for deletion tracking
- [ ] Read `ducklake_snapshot` for snapshot history
- [ ] Resolve the current snapshot (latest committed)

#### 1.3 Type Mapping (`_schema.py`)
- [ ] Map DuckLake/DuckDB SQL types to Polars types:
  - INTEGER/BIGINT/SMALLINT/TINYINT -> Int32/Int64/Int16/Int8
  - FLOAT/DOUBLE -> Float32/Float64
  - VARCHAR -> String (Utf8)
  - BOOLEAN -> Boolean
  - DATE -> Date
  - TIMESTAMP/TIMESTAMP_TZ -> Datetime
  - DECIMAL -> Decimal
  - BLOB -> Binary
  - UUID -> String
  - LIST -> List
  - STRUCT -> Struct
  - MAP -> List(Struct)
  - HUGEINT/UHUGEINT -> Int128/UInt128 (or String fallback)
  - INTERVAL, TIME -> appropriate Polars types

#### 1.4 Dataset Provider (`_dataset.py`)
- [ ] Implement `DuckLakeDataset` as a `@dataclass`
- [ ] Implement `schema()` method using catalog reader
- [ ] Implement `to_dataset_scan()`:
  - Resolve file list from metadata
  - Build absolute file paths from data_path + relative paths
  - Pass statistics via `_table_statistics`
  - Pass deletion files via `_deletion_files`
  - Handle column name mapping via `_column_mapping`
  - Return `(scan_parquet(...), version_key)`

#### 1.5 Public API (`__init__.py`)
- [ ] `scan_ducklake(path, table, *, schema=None, snapshot_version=None, snapshot_time=None) -> LazyFrame`
- [ ] `read_ducklake(path, table, **kwargs) -> DataFrame` (convenience: `scan_ducklake(...).collect()`)

#### 1.6 Basic Tests
- [ ] `test_basic.py`: Create table with DuckDB+DuckLake, read with ducklake-polars, verify data
- [ ] `test_types.py`: Round-trip all supported types
- [ ] Test re-read after additional inserts

---

### Phase 2: Advanced Read Features

#### 2.1 Time Travel
- [ ] Support `snapshot_version=N` parameter
- [ ] Support `snapshot_time="2024-01-01"` parameter
- [ ] Filter `ducklake_data_file` by snapshot version
- [ ] Handle deleted files at specific snapshots
- [ ] Tests: `test_time_travel.py`

#### 2.2 Schema Evolution
- [ ] Handle added columns (missing in old Parquet files -> NULL)
- [ ] Handle dropped columns (extra in old Parquet files -> ignore)
- [ ] Handle renamed columns via `ducklake_column_mapping` / `ducklake_name_mapping`
- [ ] Handle type promotion (e.g., INT -> BIGINT)
- [ ] Use `missing_columns="insert"` and `extra_columns="ignore"` in scan_parquet
- [ ] Tests: `test_schema_evolution.py`

#### 2.3 Partitioning
- [ ] Read partition info from `ducklake_partition_info` / `ducklake_partition_column`
- [ ] Read partition values from `ducklake_file_partition_value`
- [ ] Enable Hive-style partition pruning in scan_parquet
- [ ] Tests: `test_partitioning.py`

#### 2.4 Filter Pushdown via Statistics
- [ ] Extract min/max stats from `ducklake_file_column_stats`
- [ ] Format as `_table_statistics` for scan_parquet
- [ ] Verify file skipping with appropriate tests
- [ ] Tests: `test_filter_pushdown.py`

#### 2.5 Deletion File Handling
- [ ] Read `ducklake_delete_file` entries
- [ ] Pass as `_deletion_files` with format `"iceberg-position-delete"` (DuckLake uses same position-delete Parquet format)
- [ ] Tests: `test_delete.py`

#### 2.6 Data Inlining Support
- [ ] Detect inlined data in `ducklake_inlined_data_*` tables
- [ ] Read inlined data directly from metadata catalog
- [ ] Combine with Parquet file data
- [ ] Handle inlined deletions from `ducklake_inlined_delete_*`
- [ ] Tests: `test_data_inlining.py`

---

### Phase 3: Write Path

#### 3.1 Basic Write
- [ ] `write_ducklake(df, path, table, *, mode="append"|"overwrite")`
- [ ] Write DataFrame as Parquet file(s) to data_path
- [ ] Register files in metadata catalog (ducklake_data_file)
- [ ] Create new snapshot entry
- [ ] Update column statistics

#### 3.2 Partitioned Write
- [ ] Write partitioned Parquet files based on partition spec
- [ ] Register partition values in metadata

#### 3.3 Schema Management
- [ ] `create_table(path, table, schema)` - create new DuckLake table
- [ ] Handle schema inference from DataFrame

---

### Phase 4: Catalog API

#### 4.1 Catalog Browser
- [ ] `DuckLakeCatalog` class for browsing metadata
- [ ] `list_schemas()`, `list_tables(schema)`, `get_table_info(schema, table)`
- [ ] `list_snapshots()`, `get_snapshot(version)`
- [ ] `table_changes(table, start_version, end_version)` -> DataFrame

---

## Test Strategy

### Test Fixture Pattern

Tests use DuckDB with the DuckLake extension to **create** test catalogs, then use `ducklake-polars` to **read** them:

```python
import duckdb
import polars as pl
from ducklake_polars import scan_ducklake

@pytest.fixture
def ducklake_catalog(tmp_path):
    """Create a DuckLake catalog with test data using DuckDB."""
    metadata_path = str(tmp_path / "test.ducklake")
    data_path = str(tmp_path / "data")

    con = duckdb.connect()
    con.install_extension("ducklake")
    con.load_extension("ducklake")
    con.execute(f"""
        ATTACH 'ducklake:{metadata_path}' AS test_lake
            (DATA_PATH '{data_path}')
    """)
    return con, metadata_path, data_path
```

### Test Categories (Priority Order)

| # | Category | Source DuckLake Tests | Priority |
|---|----------|---------------------|----------|
| 1 | Basic CRUD read | `ducklake_basic.test` | P0 - MVP |
| 2 | All types | `all_types.test` | P0 - MVP |
| 3 | Filter pushdown | `filter_pushdown.test` | P1 |
| 4 | Time travel | `basic_time_travel.test` | P1 |
| 5 | Schema evolution | `add_column.test`, `struct_evolution.test` | P1 |
| 6 | Partitioning | `basic_partitioning.test` | P1 |
| 7 | Delete handling | `basic_delete.test` | P1 |
| 8 | Data inlining | `basic_data_inlining.test` | P2 |
| 9 | Table changes | `ducklake_table_changes.test` | P2 |
| 10 | Compaction effects | `small_insert_compaction.test` | P2 |
| 11 | Multiple schemas | `schema.test` | P3 |
| 12 | Snapshots listing | `ducklake_snapshots.test` | P3 |

### What We DON'T Port

These DuckLake tests are not relevant for ducklake-polars:
- **Write-path tests** (INSERT, UPDATE, DELETE, MERGE) - Phase 3
- **Concurrent transaction tests** - DuckDB-specific
- **Compaction/rewrite operations** - DuckDB admin functions
- **Encryption** - requires DuckLake extension internals
- **Views/Macros** - DuckDB SQL constructs
- **Migration tests** - DuckDB extension upgrade path
- **SQLite/Postgres backend tests** - backend-specific (we read the metadata DB directly)

---

## Key Design Questions & Decisions

### Q1: How do we read the metadata catalog?

**Decision**: Use the `duckdb` Python package to connect to the `.ducklake` file directly and query `ducklake_*` tables.

**Alternative considered**: Use the DuckLake C++ extension via `duckdb` to ATTACH and query. This would be simpler but creates a tight coupling to the extension and doesn't work for pure read scenarios.

**Update**: We should actually try BOTH approaches:
- **Primary**: Load the DuckLake extension in DuckDB Python, ATTACH the catalog, and use DuckDB SQL to query tables. This handles all the complexity (inlined data, schema evolution, snapshots) for us.
- **Secondary/Advanced**: Direct metadata queries for statistics extraction and file list resolution (for passing to scan_parquet).

### Q2: How do we handle deletion files?

**Decision**: DuckLake's deletion files are Parquet files containing `(file_path, row_id_start, row_id_end)` or similar position-based deletes. We need to investigate the exact format and determine if Polars' `_deletion_files` parameter (designed for Iceberg position deletes) is compatible.

If not compatible, we fall back to filtering via the DuckDB+DuckLake extension for correctness, and optimize later.

### Q3: How do we handle inlined data?

**Decision**: Inlined data lives in `ducklake_inlined_data_*` tables in the metadata catalog. For Phase 1, we query these via DuckDB and combine with Parquet scan results using `pl.concat()`. This is a DuckLake-specific feature not present in Iceberg/Delta.

### Q4: What about write support?

**Decision**: Defer to Phase 3. The write path requires creating snapshots, managing file metadata, handling transactions, and updating statistics - significant complexity. For writes, users can use DuckDB+DuckLake directly and then read with ducklake-polars.

---

## Milestones

| Milestone | Deliverable | Estimated Scope |
|-----------|-------------|-----------------|
| M1 | Project skeleton + basic scan | Package setup, type mapping, basic scan_ducklake() |
| M2 | Statistics + file pruning | _table_statistics support, filter pushdown tests |
| M3 | Time travel + schema evolution | Snapshot version/time support, column mapping |
| M4 | Partitioning + deletions | Hive partition pruning, deletion file handling |
| M5 | Data inlining | Read inlined data from metadata catalog |
| M6 | Write path MVP | write_ducklake() with append/overwrite |
| M7 | Catalog API | DuckLakeCatalog browser class |

---

## Open Questions

1. **Deletion file format**: What exactly is in DuckLake's delete Parquet files? Are they position-based (row numbers) like Iceberg, or something different? This determines if we can use Polars' native `_deletion_files`.

2. **Column mapping format**: DuckLake uses field IDs in Parquet files. Does this align with Polars' `_column_mapping=("iceberg-column-mapping", ...)` format, or do we need a different approach?

3. **Inlined data types**: The `ducklake_inlined_data_*` tables store data with DuckDB types. Are there any type mismatches when converting to Polars?

4. **Performance**: For large catalogs with many files, is querying the metadata catalog via DuckDB Python fast enough, or do we need caching?

5. **Cloud storage**: Should Phase 1 support S3/MinIO paths, or only local filesystem?
