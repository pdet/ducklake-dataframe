# %% [markdown]
# # ducklake-dataframe Tutorial
#
# This notebook walks through all the main features of `ducklake-dataframe` — a pure-Python Polars integration for DuckLake catalogs.
#
# We'll:
# 1. Create a DuckLake catalog with DuckDB and populate it
# 2. Read it with ducklake-dataframe (lazy scans, time travel, schema evolution)
# 3. Write data back with ducklake-dataframe (INSERT, DELETE, UPDATE, MERGE)
# 4. Verify everything from DuckDB
# 5. Explore DDL, partitioning, data inlining, views, and catalog maintenance

# %%
# Install/upgrade to latest versions (uncomment when running as a notebook):
# !pip install -U ducklake-dataframe[polars]
# !pip install duckdb==1.5.2

# %%
import os
import shutil
import tempfile

import duckdb
import polars as pl

# Work in a temp directory so we don't pollute the repo
WORKDIR = tempfile.mkdtemp(prefix="ducklake_tutorial_")
CATALOG = os.path.join(WORKDIR, "catalog.ducklake")
DATA = os.path.join(WORKDIR, "data")
os.makedirs(DATA, exist_ok=True)

print(f"Working directory: {WORKDIR}")
print(f"Catalog: {CATALOG}")
print(f"Data path: {DATA}")

# %% [markdown]
# ## 1. Initialize the Catalog with DuckDB
#
# DuckLake catalogs must be initialized by DuckDB's DuckLake extension. This creates the metadata schema (snapshot tables, column tables, etc.) that ducklake-dataframe reads and writes.

# %%
con = duckdb.connect()
con.execute("INSTALL ducklake; LOAD ducklake")
con.execute(f"ATTACH 'ducklake:sqlite:{CATALOG}' AS lake (DATA_PATH '{DATA}/')")

# Create a table and insert some data
con.execute("""
    CREATE TABLE lake.users (
        id INTEGER,
        name VARCHAR,
        email VARCHAR,
        score DOUBLE,
        active BOOLEAN
    )
""")

con.execute("""
    INSERT INTO lake.users VALUES
        (1, 'Alice',   'alice@example.com',   95.5, true),
        (2, 'Bob',     'bob@example.com',     87.3, true),
        (3, 'Carol',   'carol@example.com',   72.1, false),
        (4, 'Dave',    'dave@example.com',     91.0, true),
        (5, 'Eve',     'eve@example.com',      68.5, false)
""")

print("DuckDB: Created table and inserted 5 users")
con.execute("SELECT * FROM lake.users ORDER BY id").pl()

# %%
# Also create an events table for later
con.execute("""
    CREATE TABLE lake.events (
        event_id INTEGER,
        user_id INTEGER,
        region VARCHAR,
        amount DOUBLE,
        ts TIMESTAMP
    )
""")

con.execute("""
    INSERT INTO lake.events VALUES
        (1, 1, 'us',   100.0, '2025-01-15 10:00:00'),
        (2, 2, 'eu',   250.0, '2025-01-15 11:00:00'),
        (3, 1, 'us',   75.0,  '2025-01-16 09:00:00'),
        (4, 3, 'apac', 300.0, '2025-01-16 14:00:00'),
        (5, 4, 'eu',   150.0, '2025-01-17 08:00:00')
""")

# Close DuckDB so ducklake-dataframe can access the catalog
con.close()
print("DuckDB: Created events table with 5 rows. Connection closed.")

# %% [markdown]
# ## 2. Reading with ducklake-dataframe
#
# No DuckDB needed — reads metadata from SQLite directly and scans Parquet files through Polars.

# %%
from ducklake_polars import scan_ducklake, read_ducklake

# Eager read
df = read_ducklake(CATALOG, "users")
print("read_ducklake — all users:")
df

# %%
# Lazy scan with predicate + projection pushdown
result = (
    scan_ducklake(CATALOG, "users")
    .filter(pl.col("active") == True)
    .filter(pl.col("score") > 90)
    .select("name", "score")
    .sort("score", descending=True)
    .collect()
)
print("Active users with score > 90:")
result

# %%
# Select specific columns
df = read_ducklake(CATALOG, "users", columns=["id", "name"])
print("Projection — only id and name:")
df

# %% [markdown]
# ## 3. Catalog Inspection
#
# The `DuckLakeCatalog` class provides Python equivalents of DuckLake's utility functions.

# %%
from ducklake_polars import DuckLakeCatalog

catalog = DuckLakeCatalog(CATALOG)

print("=== Snapshots ===")
print(catalog.snapshots())

print("\n=== Current Snapshot ===")
print(catalog.current_snapshot())

print("\n=== Schemas ===")
print(catalog.list_schemas())

print("\n=== Tables ===")
print(catalog.list_tables())

# %%
print("=== Table Info ===")
print(catalog.table_info())

print("\n=== Files in 'users' ===")
print(catalog.list_files("users"))

print("\n=== Catalog Settings ===")
print(catalog.settings())

# %% [markdown]
# ## 4. Writing with ducklake-dataframe
#
# ### 4a. Append rows
#
# > **Note:** DuckDB created `id` as `INTEGER` (Int32). When appending with Polars, use matching types
# > (e.g. `pl.Int32`) to avoid schema mismatches.

# %%
from ducklake_polars import write_ducklake

new_users = pl.DataFrame({
    "id": pl.Series([6, 7], dtype=pl.Int32),
    "name": ["Frank", "Grace"],
    "email": ["frank@example.com", "grace@example.com"],
    "score": [88.0, 94.5],
    "active": [True, True],
})

write_ducklake(new_users, CATALOG, "users", mode="append",
               author="tutorial", commit_message="Add Frank and Grace")

print("After append — 7 users:")
read_ducklake(CATALOG, "users").sort("id")

# %% [markdown]
# ### 4b. Verify from DuckDB
#
# DuckDB can read what ducklake-dataframe wrote — full interop.

# %%
con = duckdb.connect()
con.execute("LOAD ducklake")
con.execute(f"ATTACH 'ducklake:sqlite:{CATALOG}' AS lake (DATA_PATH '{DATA}/')")

print("DuckDB reads ducklake-dataframe data:")
con.execute("SELECT * FROM lake.users ORDER BY id").pl()

# %%
con.close()  # release for ducklake-dataframe

# %% [markdown]
# ### 4c. Delete rows

# %%
from ducklake_polars import delete_ducklake

deleted = delete_ducklake(CATALOG, "users", pl.col("active") == False,
                          author="tutorial", commit_message="Remove inactive users")
print(f"Deleted {deleted} inactive users")

print("\nRemaining users:")
read_ducklake(CATALOG, "users").sort("id")

# %% [markdown]
# ### 4d. Update rows

# %%
from ducklake_polars import update_ducklake

updated = update_ducklake(
    CATALOG, "users",
    updates={"score": 100.0, "email": "alice-updated@example.com"},
    predicate=pl.col("name") == "Alice",
    author="tutorial", commit_message="Perfect score for Alice"
)
print(f"Updated {updated} rows")

print("\nAlice's new data:")
read_ducklake(CATALOG, "users").filter(pl.col("name") == "Alice")

# %% [markdown]
# ### 4e. Merge (upsert)

# %%
from ducklake_polars import merge_ducklake

source = pl.DataFrame({
    "id": pl.Series([1, 2, 8], dtype=pl.Int32),
    "name": ["Alice", "Bob", "Heidi"],
    "email": ["alice@new.com", "bob@new.com", "heidi@example.com"],
    "score": [98.0, 92.0, 89.0],
    "active": [True, True, True],
})

rows_updated, rows_inserted = merge_ducklake(
    CATALOG, "users", source, on="id",
    when_matched_update=True,
    when_not_matched_insert=True,
    author="tutorial", commit_message="Merge user updates"
)
print(f"Updated: {rows_updated}, Inserted: {rows_inserted}")

print("\nAll users after merge:")
read_ducklake(CATALOG, "users").sort("id")

# %% [markdown]
# ### 4f. Verify from DuckDB again

# %%
con = duckdb.connect()
con.execute("LOAD ducklake")
con.execute(f"ATTACH 'ducklake:sqlite:{CATALOG}' AS lake (DATA_PATH '{DATA}/')")

print("DuckDB sees all ducklake-dataframe changes:")
con.execute("SELECT * FROM lake.users ORDER BY id").pl()

# %%
con.close()

# %% [markdown]
# ## 5. Time Travel
#
# Every write creates a new snapshot. We can read the table at any historical point.

# %%
# See all snapshots
catalog = DuckLakeCatalog(CATALOG)
snapshots = catalog.snapshots()
print("All snapshots:")
print(snapshots)

# Find the first snapshot that has user data
for v in sorted(snapshots["snapshot_id"].to_list()):
    try:
        df = read_ducklake(CATALOG, "users", snapshot_version=v)
        if len(df) > 0:
            print(f"\nSnapshot {v} — original data ({len(df)} rows):")
            print(df.sort("id"))
            break
    except Exception:
        continue

print(f"\nLatest snapshot — current data:")
print(read_ducklake(CATALOG, "users").sort("id"))

# %% [markdown]
# ## 6. Change Data Feed
#
# Track what changed between snapshots.

# %%
snap_ids = sorted(catalog.snapshots()["snapshot_id"].to_list())
start_v = snap_ids[0]
end_v = snap_ids[-1]

print(f"Changes between snapshot {start_v} and {end_v}:")
changes = catalog.table_changes("users", start_version=start_v, end_version=end_v)
changes

# %% [markdown]
# Insertions, deletions, and full change-data-feed are also exposed as separate APIs. Updates surface as paired `update_preimage` / `update_postimage` rows.

# %%
# Just inserts in the range
insertions = catalog.table_insertions("users", start_version=start_v, end_version=end_v)
print(f"Insertions: {insertions.shape[0]} rows")

# Just deletions in the range
deletions = catalog.table_deletions("users", start_version=start_v, end_version=end_v)
print(f"Deletions: {deletions.shape[0]} rows")

# Lazy CDC scan (predicate/projection pushdown applies here too)
from ducklake_polars import scan_ducklake_changes
scan_ducklake_changes(CATALOG, "users", start_v, end_v).filter(
    pl.col("change_type").is_in(["update_preimage", "update_postimage"])
).collect()

# %% [markdown]
# ## 7. Schema Evolution (DDL)
#
# ### Add, rename, and drop columns

# %%
from ducklake_polars import (
    alter_ducklake_add_column,
    alter_ducklake_rename_column,
    alter_ducklake_drop_column,
)

# Add a column — existing rows get NULL
alter_ducklake_add_column(CATALOG, "users", "department", pl.String,
                          author="tutorial", commit_message="Add department column")

print("After ADD COLUMN 'department':")
print(read_ducklake(CATALOG, "users").sort("id"))

# %%
# Rename a column
alter_ducklake_rename_column(CATALOG, "users", "email", "contact_email",
                              author="tutorial", commit_message="Rename email → contact_email")

print("After RENAME COLUMN 'email' → 'contact_email':")
print(read_ducklake(CATALOG, "users").sort("id"))

# %%
# Drop a column
alter_ducklake_drop_column(CATALOG, "users", "department",
                            author="tutorial", commit_message="Drop department column")

print("After DROP COLUMN 'department':")
print(read_ducklake(CATALOG, "users").sort("id"))

# %%
# DuckDB sees the schema changes
con = duckdb.connect()
con.execute("LOAD ducklake")
con.execute(f"ATTACH 'ducklake:sqlite:{CATALOG}' AS lake (DATA_PATH '{DATA}/')")

print("DuckDB DESCRIBE after schema evolution:")
con.execute("DESCRIBE lake.users").pl()

# %%
con.close()

# %% [markdown]
# ## 8. Partitioned Writes
#
# Set partitioning on a table, then inserts automatically create Hive-style directory layouts.

# %%
from ducklake_polars import alter_ducklake_set_partitioned_by

# Partition the events table by region
alter_ducklake_set_partitioned_by(CATALOG, "events", ["region"],
                                   author="tutorial", commit_message="Partition events by region")

# Insert more events — they'll be written as partitioned Parquet files
# Note: DuckDB created event_id/user_id as INTEGER (Int32)
new_events = pl.DataFrame({
    "event_id": pl.Series([6, 7, 8, 9], dtype=pl.Int32),
    "user_id": pl.Series([1, 2, 6, 7], dtype=pl.Int32),
    "region": ["us", "eu", "us", "apac"],
    "amount": [200.0, 175.0, 50.0, 400.0],
    "ts": ["2025-02-01 10:00:00", "2025-02-01 11:00:00",
           "2025-02-02 09:00:00", "2025-02-02 14:00:00"],
}).with_columns(pl.col("ts").str.to_datetime())

write_ducklake(new_events, CATALOG, "events", mode="append",
               author="tutorial", commit_message="Partitioned event insert")

print("All events:")
read_ducklake(CATALOG, "events").sort("event_id")

# %%
# Partition pruning — only reads 'us' partition files
us_events = (
    scan_ducklake(CATALOG, "events")
    .filter(pl.col("region") == "us")
    .collect()
)
print("US events only (partition-pruned):")
us_events.sort("event_id")

# %%
# Show the partitioned file structure
print("Data files on disk:")
for root, dirs, files in os.walk(DATA):
    for f in files:
        path = os.path.join(root, f)
        rel = os.path.relpath(path, DATA)
        size = os.path.getsize(path)
        print(f"  {rel}  ({size:,} bytes)")

# %% [markdown]
# ## 9. Data Inlining
#
# Small inserts can be stored directly in the catalog database instead of creating tiny Parquet files.

# %%
from ducklake_polars import create_ducklake_table

# Create a table for small writes
create_ducklake_table(CATALOG, "metrics", {
    "ts": pl.Datetime("us"),
    "sensor": pl.String(),
    "value": pl.Float64(),
})

# Small inserts get inlined (stored in SQLite, not Parquet)
for i in range(5):
    row = pl.DataFrame({
        "ts": [f"2025-03-01 {10+i}:00:00"],
        "sensor": [f"sensor_{i}"],
        "value": [20.0 + i * 1.5],
    }).with_columns(pl.col("ts").str.to_datetime())
    
    write_ducklake(row, CATALOG, "metrics", mode="append",
                   data_inlining_row_limit=10)

print("Metrics (5 inlined rows, no Parquet files):")
print(read_ducklake(CATALOG, "metrics").sort("ts"))

# Check: no Parquet files for metrics yet
catalog = DuckLakeCatalog(CATALOG)
print("\nFiles for 'metrics':")
print(catalog.list_files("metrics"))

# %%
# A larger insert triggers flush to Parquet
big_batch = pl.DataFrame({
    "ts": [f"2025-03-02 {h:02d}:00:00" for h in range(24)],
    "sensor": [f"sensor_{i % 5}" for i in range(24)],
    "value": [15.0 + i * 0.5 for i in range(24)],
}).with_columns(pl.col("ts").str.to_datetime())

write_ducklake(big_batch, CATALOG, "metrics", mode="append",
               data_inlining_row_limit=10)

print(f"Metrics after big insert: {len(read_ducklake(CATALOG, 'metrics'))} rows")
print("\nFiles for 'metrics' (now has Parquet):")
catalog = DuckLakeCatalog(CATALOG)
print(catalog.list_files("metrics"))

# %% [markdown]
# ## 10. CREATE TABLE AS
#
# Create a new table with data in a single atomic snapshot.

# %%
from ducklake_polars import create_table_as_ducklake

# Build a summary from existing data
summary = (
    scan_ducklake(CATALOG, "events")
    .group_by("region")
    .agg(
        pl.col("amount").sum().alias("total_amount"),
        pl.col("event_id").count().alias("event_count"),
    )
    .sort("region")
    .collect()
)

create_table_as_ducklake(summary, CATALOG, "region_summary",
                          author="tutorial", commit_message="Regional summary CTAS")

print("Created 'region_summary' via CTAS:")
read_ducklake(CATALOG, "region_summary")

# %% [markdown]
# ## 11. Views

# %%
from ducklake_polars import create_ducklake_view

# Create a view
create_ducklake_view(
    CATALOG, "active_users",
    "SELECT id, name, score FROM users WHERE active = true",
    author="tutorial", commit_message="Create active_users view"
)
print("Created view 'active_users'")

# DuckDB can query it
con = duckdb.connect()
con.execute("LOAD ducklake")
con.execute(f"ATTACH 'ducklake:sqlite:{CATALOG}' AS lake (DATA_PATH '{DATA}/')")

print("\nDuckDB querying the view:")
con.execute("SELECT * FROM lake.active_users ORDER BY id").pl()
con.close()  # release for ducklake-dataframe

# %%
# Replace view
create_ducklake_view(
    CATALOG, "active_users",
    "SELECT id, name, score FROM users WHERE active = true AND score > 90",
    or_replace=True,
    author="tutorial", commit_message="Update view: score > 90 filter"
)

# Re-attach so DuckDB sees the updated view
con = duckdb.connect()
con.execute("LOAD ducklake")
con.execute(f"ATTACH 'ducklake:sqlite:{CATALOG}' AS lake (DATA_PATH '{DATA}/')")

print("Updated view (score > 90 only):")
con.execute("SELECT * FROM lake.active_users ORDER BY id").pl()

# %%
con.close()

# %% [markdown]
# ## 11.4. Catalog Migration
#
# Older catalogs (DuckLake v0.3, v0.4) can be brought up to v1.0 in place by calling `migrate_catalog`. The function is idempotent — calling it on a v1.0 catalog is a no-op. Migration is **opt-in**; reads against v0.3/v0.4 catalogs work without it, but v1.0-only writer features (macros, `merge_adjacent_files`, expression sort keys, custom column tag keys) require migration first.

# %%
from ducklake_polars import migrate_catalog

# Make a sibling catalog by copying the current one — we'll migrate the
# copy so the main CATALOG stays at its current version (older DuckDB
# builds can't read v1.0 catalogs, and the rest of this notebook still
# uses DuckDB to verify writes).
side_catalog = os.path.join(WORKDIR, "copy.ducklake")
shutil.copyfile(CATALOG, side_catalog)

before = DuckLakeCatalog(side_catalog).options()
before_version = before.filter(pl.col("key") == "version")["value"][0]
print(f"sibling catalog version before: {before_version}")

new_v = migrate_catalog(side_catalog)
print(f"sibling catalog version after:  {new_v}")
print("(idempotent — calling again is a no-op)")
print(f"second call returns: {migrate_catalog(side_catalog)!r}")

# %% [markdown]
# ## 11.5. Tags & Macros
#
# Tags attach key-value metadata to tables and columns; macros register user-defined SQL functions in the catalog. With DuckDB 1.5+ both round-trip cleanly through DuckDB's ducklake extension.

# %%
from ducklake_polars import (
    set_ducklake_table_tag,
    set_ducklake_column_tag,
    delete_ducklake_table_tag,
    create_ducklake_macro,
    drop_ducklake_macro,
)

# Set tags
set_ducklake_table_tag(CATALOG, "users", "owner", "analytics-team")
set_ducklake_table_tag(CATALOG, "users", "pii", "true")
# DuckDB 1.5+ supports arbitrary column tag keys end-to-end.
# DuckDB's ducklake extension currently only round-trips the `comment`
# key for columns (table tags do support arbitrary keys on 1.5+).
set_ducklake_column_tag(CATALOG, "users", "contact_email", "comment", "PII: email address")
# Inspect via catalog
cat = DuckLakeCatalog(CATALOG)
print("Table tags on 'users':")
print(cat.table_tags("users"))

# Remove a tag
delete_ducklake_table_tag(CATALOG, "users", "pii")
print("\nAfter deleting 'pii' tag:")
print(cat.table_tags("users"))

# %%
from ducklake_polars import create_ducklake_macro, drop_ducklake_macro

create_ducklake_macro(
    CATALOG, "add_one", "a + 1",
    parameters=[{"name": "a", "type": "integer"}],
)
print("Registered macros:")
print(cat.list_macros())

# DuckDB calls the macro
con = duckdb.connect()
con.execute("LOAD ducklake")
con.execute(f"ATTACH 'ducklake:sqlite:{CATALOG}' AS lake (DATA_PATH '{DATA}/')")
print("add_one(41) =", con.execute("SELECT lake.add_one(41)").fetchone()[0])
con.close()

drop_ducklake_macro(CATALOG, "add_one")

# %% [markdown]
# ## 12. Schema and Table Management

# %%
from ducklake_polars import (
    create_ducklake_schema,
    drop_ducklake_schema,
    rename_ducklake_table,
    drop_ducklake_table,
)

# Create a new schema
create_ducklake_schema(CATALOG, "staging")

# Create a table in it
create_ducklake_table(CATALOG, "raw_data", {
    "id": pl.Int64(),
    "payload": pl.String(),
}, schema="staging")

print("Schemas:", DuckLakeCatalog(CATALOG).list_schemas()["schema_name"].to_list())
print("Tables in 'staging':", DuckLakeCatalog(CATALOG).list_tables(schema="staging")["table_name"].to_list())

# Rename the table
rename_ducklake_table(CATALOG, "raw_data", "raw_events", schema="staging")
print("After rename:", DuckLakeCatalog(CATALOG).list_tables(schema="staging")["table_name"].to_list())

# Drop schema with cascade
drop_ducklake_schema(CATALOG, "staging", cascade=True)
print("After drop cascade:", DuckLakeCatalog(CATALOG).list_schemas()["schema_name"].to_list())

# %% [markdown]
# ## 12.5. Streaming Ingestion
#
# `DuckLakeStreamWriter` buffers micro-batches and auto-flushes when the buffer hits the threshold. On clean exit it can also auto-compact the produced files into one. Exiting the context with an exception drops any unflushed rows; already-flushed batches remain visible.

# %%
from ducklake_polars import DuckLakeStreamWriter

with DuckLakeStreamWriter(
    CATALOG, "events", flush_threshold=2, compact_on_close=False,
) as writer:
    writer.append(pl.DataFrame({
        "event_id": pl.Series([100, 101], dtype=pl.Int32),
        "user_id":  pl.Series([1, 2],     dtype=pl.Int32),
        "region":   ["us", "eu"],
        "amount":   [50.0, 75.0],
        "ts":       ["2025-04-01 09:00:00", "2025-04-01 09:05:00"],
    }).with_columns(pl.col("ts").str.to_datetime()))  # auto-flushes (threshold=2)
    writer.append(pl.DataFrame({
        "event_id": pl.Series([102], dtype=pl.Int32),
        "user_id":  pl.Series([1],   dtype=pl.Int32),
        "region":   ["us"],
        "amount":   [25.0],
        "ts":       ["2025-04-01 09:10:00"],
    }).with_columns(pl.col("ts").str.to_datetime()))  # buffered → flushes on close

print(f"Wrote {writer.total_rows} rows in {writer.flush_count} flushes")

# %% [markdown]
# ## 12.6. Registering External Parquet Files
#
# `add_files_ducklake` registers an existing Parquet file into the catalog without copying it.

# %%
from ducklake_polars import add_files_ducklake

# Write a parquet file outside the catalog
ext_path = os.path.join(WORKDIR, "external.parquet")
pl.DataFrame({
    "event_id": pl.Series([200, 201], dtype=pl.Int32),
    "user_id":  pl.Series([1, 2],     dtype=pl.Int32),
    "region":   ["us", "eu"],
    "amount":   [10.0, 20.0],
    "ts":       ["2025-05-01 10:00:00", "2025-05-01 10:05:00"],
}).with_columns(pl.col("ts").str.to_datetime()).write_parquet(ext_path)

added = add_files_ducklake(CATALOG, "events", [ext_path])
print(f"Registered {added} external file(s)")
print(f"Total events now: {len(read_ducklake(CATALOG, 'events'))}")

# %% [markdown]
# ## 12.7. Compaction
#
# After many small appends, compact the table to combine small files. `merge_adjacent_files_ducklake` (DuckLake v1.0+) is a lightweight option that merges adjacent small files within a partition. The source files are queued in `ducklake_files_scheduled_for_deletion`; `cleanup_old_files_ducklake` physically deletes them once they are older than the cutoff. `rewrite_data_files_ducklake` is the heavier alternative that fully rewrites everything (and applies pending positional deletes).

# %%
from datetime import datetime, timedelta, timezone
from ducklake_polars import (
    merge_adjacent_files_ducklake,
    cleanup_old_files_ducklake,
)

n_before = len(DuckLakeCatalog(CATALOG).list_files("events"))
merge_adjacent_files_ducklake(CATALOG, "events", min_file_size=1, max_file_size=10_000_000)
n_after = len(DuckLakeCatalog(CATALOG).list_files("events"))
print(f"events: {n_before} files → {n_after} files")

# Drain the deletion queue (physically removes the retired source files)
future = datetime.now(timezone.utc) + timedelta(days=1)
removed = cleanup_old_files_ducklake(CATALOG, older_than=future)
print(f"Cleaned up {len(removed)} retired data files")

# %% [markdown]
# ## 13. Catalog Maintenance
#
# Clean up old snapshots and orphaned files.

# %%
from ducklake_polars import expire_snapshots, vacuum_ducklake

catalog = DuckLakeCatalog(CATALOG)
print(f"Snapshots before cleanup: {len(catalog.snapshots())}")

# Keep only the last 3 snapshots
expired = expire_snapshots(CATALOG, keep_last_n=3)
print(f"Expired {expired} snapshots")

catalog = DuckLakeCatalog(CATALOG)
print(f"Snapshots after cleanup: {len(catalog.snapshots())}")

# Delete orphaned Parquet files
deleted = vacuum_ducklake(CATALOG)
print(f"Deleted {deleted} orphaned Parquet files")

# %% [markdown]
# ## 14. Final Verification: Full Roundtrip
#
# Let's verify the complete state from both DuckDB and ducklake-dataframe.

# %%
# ducklake-dataframe view
print("=== ducklake-dataframe ===")
for table in ["users", "events", "metrics", "region_summary"]:
    df = read_ducklake(CATALOG, table)
    print(f"\n{table}: {len(df)} rows")
    print(df.head(3))

# %%
# DuckDB view
con = duckdb.connect()
con.execute("LOAD ducklake")
con.execute(f"ATTACH 'ducklake:sqlite:{CATALOG}' AS lake (DATA_PATH '{DATA}/')")

print("=== DuckDB ===")
for table in ["users", "events", "metrics", "region_summary"]:
    result = con.execute(f"SELECT COUNT(*) FROM lake.{table}").fetchone()
    print(f"{table}: {result[0]} rows")

print("\nDuckDB — users:")
con.execute("SELECT * FROM lake.users ORDER BY id").pl()

# %%
con.close()

# %% [markdown]
# ## Cleanup

# %%
# Remove the temp directory
shutil.rmtree(WORKDIR)
print(f"Cleaned up {WORKDIR}")

# %% [markdown]
# ---
#
# ## Summary
#
# This tutorial covered:
#
# | Feature | Functions Used |
# |---------|---------------|
# | **Reading** | `scan_ducklake`, `read_ducklake` |
# | **Writing** | `write_ducklake` (error/append/overwrite) |
# | **Deletes** | `delete_ducklake` |
# | **Updates** | `update_ducklake` |
# | **Merge** | `merge_ducklake` |
# | **CTAS** | `create_table_as_ducklake` |
# | **DDL** | `create_ducklake_table`, `alter_ducklake_add/drop/rename_column` |
# | **Partitioning** | `alter_ducklake_set_partitioned_by` |
# | **Data inlining** | `write_ducklake(..., data_inlining_row_limit=N)` |
# | **Views** | `create_ducklake_view`, `drop_ducklake_view` |
# | **Schemas** | `create_ducklake_schema`, `drop_ducklake_schema` |
# | **Time travel** | `read_ducklake(..., snapshot_version=N)` |
# | **Change data feed** | `DuckLakeCatalog.table_changes()` |
# | **Catalog inspection** | `DuckLakeCatalog.snapshots/table_info/list_files/...` |
# | **Maintenance** | `expire_snapshots`, `vacuum_ducklake` |
# | **DuckDB interop** | ✅ Full bidirectional |
#
# For the complete API reference, see the [wiki](https://github.com/pdet/ducklake-dataframe/wiki).

