# DuckDB v1.5 / DuckLake v0.4 Migration Plan

## Schema Changes (v0.3 → v0.4)

### New Tables
1. **ducklake_macro** — `(schema_id, macro_id, macro_name, begin_snapshot, end_snapshot)` — stored procedures/macros
2. **ducklake_macro_impl** — `(macro_id, impl_id, dialect, sql, type)` — macro implementations
3. **ducklake_macro_parameters** — `(macro_id, impl_id, column_id, parameter_name, parameter_type, default_value, default_value_type)` — macro parameters
4. **ducklake_file_variant_stats** — `(data_file_id, table_id, column_id, variant_path, shredded_type, column_size_bytes, value_count, null_count, min_value, max_value, contains_nan, extra_stats)` — VARIANT type statistics

### Altered Tables
5. **ducklake_column** — 2 new columns:
   - `default_value_type VARCHAR DEFAULT 'literal'`
   - `default_value_dialect VARCHAR DEFAULT NULL`
6. **ducklake_schema_versions** — 1 new column:
   - `table_id BIGINT` (makes schema versions per-table instead of global)
7. **ducklake_data_file** — 1 new column, 1 removed:
   - Added: `partial_max BIGINT`
   - Removed: `partial_file_info` (data migrated to `partial_max`)
8. **ducklake_delete_file** — 1 new column:
   - Added: `partial_max BIGINT`
9. **ducklake_name_mapping** — 1 new column:
   - Added: `is_partition BOOLEAN`

### New Features
10. **VARIANT type support** — first-class variant/shredded column stats
11. **Per-table schema versions** — `ducklake_schema_versions.table_id`
12. **Macro/stored procedure support** — CREATE MACRO
13. **Per-catalog data inlining settings** — inlining on each catalog
14. **Geometry/Variant inlining skip** — skip inlining for unsupported types

### Version
- Metadata version: `0.3` → `0.4`

## Implementation Plan

### Commit 1: Support v0.4 catalog version
- Update `SUPPORTED_DUCKLAKE_VERSIONS` to include `"0.4"`
- Update `_catalog.py` to handle new columns gracefully (default values for missing columns)
- Handle `ducklake_column.default_value_type` and `default_value_dialect`
- Handle `ducklake_data_file.partial_max` (new column)
- Handle `ducklake_delete_file.partial_max` (new column)
- Handle `ducklake_schema_versions.table_id` (new column)
- Handle `ducklake_name_mapping.is_partition` (new column)

### Commit 2: Update writer for v0.4 schema
- Writer creates v0.4 catalogs (set version to 0.4)
- Write `default_value_type` and `default_value_dialect` when creating columns
- Write `partial_max` for data files and delete files
- Write `table_id` in schema_versions
- Write `is_partition` in name_mapping

### Commit 3: VARIANT type support
- Add VARIANT to schema type mapping (→ pa.string() or pa.binary())
- Support ducklake_file_variant_stats in catalog reader
- Add variant_stats to catalog API

### Commit 4: Macro support (if practical)
- Assess complexity — may skip for now if macros are DuckDB-engine-specific

### Commit 5: Migration support
- Add migrate_v03_to_v04() function
- Support reading v0.3 catalogs and auto-migrating

### Commit 6: Test updates
- Update all tests to work with v0.4 catalogs
- Add v0.4-specific tests (variant stats, per-table schema versions)
- Verify DuckDB interop with v1.5
