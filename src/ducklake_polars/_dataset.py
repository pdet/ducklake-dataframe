"""DuckLake dataset provider for Polars."""

from __future__ import annotations

import atexit
import os
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import polars as pl

from ducklake_polars._catalog import DuckLakeCatalogReader
from ducklake_polars._schema import resolve_column_type
from ducklake_polars._stats import build_table_statistics

if TYPE_CHECKING:
    from datetime import datetime

    from polars import LazyFrame
    from polars.schema import Schema

    from ducklake_polars._catalog import (
        ColumnHistoryEntry,
        ColumnInfo,
        DeleteFileInfo,
        FileInfo,
        FilePartitionValue,
        PartitionColumnDef,
        TableInfo,
    )


def _safe_unlink(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


def _has_renames(
    history: list[ColumnHistoryEntry],
    current_columns: list[ColumnInfo],
) -> bool:
    """Check if any column has been renamed by looking at the history."""
    current_names = {c.column_id: c.column_name for c in current_columns}
    for entry in history:
        if entry.column_id in current_names and entry.column_name != current_names[entry.column_id]:
            return True
    return False


def _get_physical_name(
    column_id: int,
    file_begin_snapshot: int,
    history: list[ColumnHistoryEntry],
) -> str | None:
    """Get the physical column name that was active when the file was written."""
    for entry in history:
        if entry.column_id != column_id:
            continue
        # The entry is active at file_begin_snapshot if:
        # begin_snapshot <= file_begin_snapshot AND (end_snapshot IS NULL OR end_snapshot > file_begin_snapshot)
        if entry.begin_snapshot <= file_begin_snapshot and (
            entry.end_snapshot is None or entry.end_snapshot > file_begin_snapshot
        ):
            return entry.column_name
    return None


def _get_rename_map(
    file_begin_snapshot: int,
    history: list[ColumnHistoryEntry],
    current_columns: list[ColumnInfo],
) -> dict[str, str]:
    """Get {physical_name -> current_name} for columns that differ."""
    rename_map: dict[str, str] = {}
    for col in current_columns:
        physical = _get_physical_name(col.column_id, file_begin_snapshot, history)
        if physical is not None and physical != col.column_name:
            rename_map[physical] = col.column_name
    return rename_map


def _group_files_by_rename_map(
    files: list[FileInfo],
    history: list[ColumnHistoryEntry],
    current_columns: list[ColumnInfo],
) -> list[tuple[dict[str, str], list[FileInfo]]]:
    """Group files by their rename map (files with identical mappings together)."""
    # Use a dict keyed by frozenset of rename_map items
    groups: dict[frozenset[tuple[str, str]], list[FileInfo]] = defaultdict(list)
    rename_maps: dict[frozenset[tuple[str, str]], dict[str, str]] = {}

    for f in files:
        rmap = _get_rename_map(f.begin_snapshot, history, current_columns)
        key = frozenset(rmap.items())
        groups[key].append(f)
        if key not in rename_maps:
            rename_maps[key] = rmap

    return [(rename_maps[k], group) for k, group in groups.items()]


def _build_partition_values_for_stats(
    partition_columns: list[PartitionColumnDef],
    file_partition_values: list[FilePartitionValue],
) -> dict[int, dict[int, str | None]]:
    """Build a lookup of {data_file_id: {column_id: partition_value}}.

    Only includes partition columns that use the identity transform.
    """
    # Map partition_key_index -> column_id
    identity_key_to_col: dict[int, int] = {}
    for pc in partition_columns:
        if pc.transform == "identity":
            identity_key_to_col[pc.partition_key_index] = pc.column_id

    if not identity_key_to_col:
        return {}

    result: dict[int, dict[int, str | None]] = {}
    for fpv in file_partition_values:
        if fpv.partition_key_index in identity_key_to_col:
            col_id = identity_key_to_col[fpv.partition_key_index]
            result.setdefault(fpv.data_file_id, {})[col_id] = fpv.partition_value

    return result


@dataclass
class DuckLakeDataset:
    """
    Dataset provider for DuckLake tables.

    Implements the PythonDatasetProvider interface expected by
    Polars' PyLazyFrame.new_from_dataset_object().
    """

    metadata_path: str
    table_name: str
    schema_name: str
    snapshot_version: int | None = None
    snapshot_time: datetime | str | None = None
    data_path_override: str | None = None

    def __post_init__(self) -> None:
        if self.snapshot_version is not None and self.snapshot_time is not None:
            msg = "Cannot specify both snapshot_version and snapshot_time"
            raise ValueError(msg)

    def _get_reader(self) -> DuckLakeCatalogReader:
        return DuckLakeCatalogReader(
            self.metadata_path,
            data_path_override=self.data_path_override,
        )

    def _resolve_snapshot(self, reader: DuckLakeCatalogReader) -> Any:
        if self.snapshot_version is not None:
            return reader.get_snapshot_at_version(self.snapshot_version)
        if self.snapshot_time is not None:
            return reader.get_snapshot_at_time(self.snapshot_time)
        return reader.get_current_snapshot()

    #
    # PythonDatasetProvider interface
    #

    @staticmethod
    def _build_schema_from_columns(all_columns: list[ColumnInfo]) -> dict[str, pl.DataType]:
        """Build the Polars schema dict from the column hierarchy."""
        top_level = [c for c in all_columns if c.parent_column is None]

        schema_dict: dict[str, pl.DataType] = {}
        for col in top_level:
            schema_dict[col.column_name] = resolve_column_type(
                col.column_id, col.column_type, all_columns
            )
        return schema_dict

    def schema(self) -> Schema:
        """Return the table schema as a Polars Schema."""
        with self._get_reader() as reader:
            snapshot = self._resolve_snapshot(reader)
            table = reader.get_table(self.table_name, self.schema_name, snapshot.snapshot_id)
            all_columns = reader.get_all_columns(table.table_id, snapshot.snapshot_id)
            return pl.Schema(self._build_schema_from_columns(all_columns))

    @staticmethod
    def _build_scan_kwargs(
        group_files: list[FileInfo],
        all_data_files: list[FileInfo],
        delete_files: list[DeleteFileInfo],
        reader: DuckLakeCatalogReader,
        table: TableInfo,
        columns: list[ColumnInfo],
        filter_columns: list[str] | None,
        partition_values: dict[int, dict[int, str | None]] | None,
    ) -> dict[str, Any]:
        """Build scan_parquet kwargs for a group of files."""
        kwargs: dict[str, Any] = {
            "missing_columns": "insert",
            "extra_columns": "ignore",
        }

        # Build deletion files mapping for this group
        if delete_files:
            file_id_to_idx = {
                f.data_file_id: i for i, f in enumerate(group_files)
            }
            deletion_files_map: dict[int, list[str]] = {}
            for df in delete_files:
                idx = file_id_to_idx.get(df.data_file_id)
                if idx is not None:
                    path = reader.resolve_data_file_path(df.path, df.path_is_relative, table)
                    deletion_files_map.setdefault(idx, []).append(path)
            if deletion_files_map:
                kwargs["_deletion_files"] = (
                    "iceberg-position-delete",
                    deletion_files_map,
                )

        # Build table statistics for this group
        if filter_columns:
            file_ids = [f.data_file_id for f in group_files]
            col_ids = [
                c.column_id
                for c in columns
                if c.column_name in filter_columns
            ]
            stats = reader.get_column_stats(table.table_id, file_ids, col_ids)
            table_statistics = build_table_statistics(
                group_files, stats, columns, filter_columns,
                partition_values=partition_values,
            )
            if table_statistics is not None:
                kwargs["_table_statistics"] = table_statistics

        return kwargs

    def to_dataset_scan(
        self,
        *,
        existing_resolved_version_key: str | None = None,
        limit: int | None = None,
        projection: list[str] | None = None,
        filter_columns: list[str] | None = None,
        pyarrow_predicate: str | None = None,
    ) -> tuple[LazyFrame, str] | None:
        """
        Resolve metadata and construct a scan_parquet LazyFrame.

        Returns (LazyFrame, version_key) or None if the version hasn't changed.

        Note: ``limit``, ``projection``, and ``pyarrow_predicate`` are part of the
        PythonDatasetProvider interface but are not used here; Polars handles
        projection and filtering on the resulting LazyFrame.
        """
        from polars.io.parquet.functions import scan_parquet

        with self._get_reader() as reader:
            snapshot = self._resolve_snapshot(reader)
            version_key = str(snapshot.snapshot_id)

            # Short-circuit if version hasn't changed
            if (
                existing_resolved_version_key is not None
                and existing_resolved_version_key == version_key
            ):
                return None

            table = reader.get_table(
                self.table_name, self.schema_name, snapshot.snapshot_id
            )
            all_columns = reader.get_all_columns(table.table_id, snapshot.snapshot_id)
            columns = [c for c in all_columns if c.parent_column is None]
            column_names = [c.column_name for c in columns]

            # Get data files
            data_files = reader.get_data_files(table.table_id, snapshot.snapshot_id)

            if not data_files:
                # No data files - check for inlined data
                inlined = reader.read_inlined_data(
                    table.table_id,
                    snapshot.snapshot_id,
                    column_names,
                )
                if inlined is not None and not inlined.is_empty():
                    return inlined.lazy(), version_key
                # Empty table - return scan_parquet with empty list
                schema_dict = self._build_schema_from_columns(all_columns)
                return scan_parquet(
                    [],
                    schema=schema_dict,
                ), version_key

            # Get delete files
            delete_files = reader.get_delete_files(table.table_id, snapshot.snapshot_id)

            # Fetch partition values for statistics supplementation
            partition_values: dict[int, dict[int, str | None]] | None = None
            if filter_columns:
                try:
                    part_info = reader.get_partition_info(table.table_id, snapshot.snapshot_id)
                    if part_info is not None:
                        part_cols = reader.get_partition_columns(part_info.partition_id, table.table_id)
                        # Check if any partition columns overlap with filter columns
                        col_id_to_name = {c.column_id: c.column_name for c in columns}
                        part_col_names = {col_id_to_name[pc.column_id] for pc in part_cols if pc.column_id in col_id_to_name}
                        if part_col_names & set(filter_columns):
                            file_ids = [f.data_file_id for f in data_files]
                            fpvs = reader.get_file_partition_values(table.table_id, file_ids)
                            partition_values = _build_partition_values_for_stats(
                                part_cols, fpvs,
                            )
                except Exception as e:
                    if not reader._backend.is_table_not_found(e):
                        raise

            # Detect column renames
            history = reader.get_column_history(table.table_id)
            has_rename = _has_renames(history, columns)

            if not has_rename:
                # Fast path: no renames, existing code
                sources = [
                    reader.resolve_data_file_path(f.path, f.path_is_relative, table)
                    for f in data_files
                ]

                kwargs = self._build_scan_kwargs(
                    data_files, data_files, delete_files, reader, table,
                    columns, filter_columns, partition_values,
                )

                lf = scan_parquet(sources, **kwargs)
            else:
                # Rename path: the dataset scan resolver only accepts bare
                # Parquet SCAN nodes.  We collect each file group eagerly,
                # apply renames, write the combined result to a temporary
                # Parquet file, and return scan_parquet on that file.
                groups = _group_files_by_rename_map(data_files, history, columns)

                group_dfs: list[pl.DataFrame] = []
                for rename_map, group_files_list in groups:
                    sources = [
                        reader.resolve_data_file_path(f.path, f.path_is_relative, table)
                        for f in group_files_list
                    ]

                    kwargs = self._build_scan_kwargs(
                        group_files_list, data_files, delete_files, reader, table,
                        columns, filter_columns, partition_values,
                    )

                    df = scan_parquet(sources, **kwargs).collect()
                    if rename_map:
                        df = df.rename(rename_map)
                    group_dfs.append(df)

                if len(group_dfs) == 1:
                    combined = group_dfs[0]
                else:
                    combined = pl.concat(group_dfs, how="diagonal_relaxed")

                # Write to temp file and scan it (resolver needs Parquet SCAN)
                fd, tmp_path = tempfile.mkstemp(suffix=".parquet")
                os.close(fd)
                atexit.register(lambda p=tmp_path: _safe_unlink(p))
                combined.write_parquet(tmp_path)

                lf = scan_parquet(
                    tmp_path,
                    missing_columns="insert",
                    extra_columns="ignore",
                )

            # If there's inlined data, we need to combine it
            inlined = reader.read_inlined_data(
                table.table_id,
                snapshot.snapshot_id,
                column_names,
            )
            if inlined is not None and not inlined.is_empty():
                lf = pl.concat([lf, inlined.lazy()], how="diagonal_relaxed")

            return lf, version_key
