# Changelog

## v0.4.0

### Highlights

- **Catalog inspection API** -- `list_tables`, `list_views`, `list_schemas`, `list_snapshots`, `catalog_info`, `table_info`
- **View support** -- `get_view` reads view definitions (SQL, dialect, column aliases)
- **Snapshot audit log** -- `snapshot_changes` queries per-snapshot change history

### Features

- `list_schemas(path)` -- list all schema names at the current snapshot
- `list_tables(path, schema)` -- list table names in a schema
- `list_views(path, schema)` -- list view names in a schema
- `get_view(path, name, schema)` -- get view definition (SQL, dialect, aliases)
- `list_snapshots(path, limit)` -- list recent snapshots
- `snapshot_changes(path, snapshot_id)` -- audit log of catalog changes
- `catalog_info(path)` -- catalog summary (version, table count, snapshot count)
- `table_info(path, table, schema)` -- column names, types, and nullability

### Testing

- Type coverage tests for all DuckDB types
- Catalog inspection, view, schema, and snapshot change tests
- 3000+ total tests

## v0.3.0

### Highlights

- **Unified Arrow core with Polars + pandas wrappers** — shared PyArrow-based engine powering both DataFrame flavors
- **Package renamed to `ducklake-dataframe`** — reflects multi-framework support beyond Polars
- **Object storage support (S3/GCS/Azure via fsspec)** — read and write Parquet files directly on cloud storage
- **DuckDB catalog backend** — use a local DuckDB database as the DuckLake metadata catalog

### Features

- Concurrent write safety with optimistic concurrency control
- Field-id based column mapping for robust schema evolution
- Case-insensitive table name lookup
- Add files and rewrite data files support

### Improvements

- Performance optimized catalog queries
- Schema evolution edge case handling

### Testing

- Comprehensive test suite (1500+ tests)
