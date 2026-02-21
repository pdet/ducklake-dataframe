"""DuckLake dataset provider for Polars."""

from __future__ import annotations

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

    from ducklake_polars._catalog import ColumnInfo


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
                # This works because scan_parquet([]) produces a valid empty LazyFrame
                schema_dict = self._build_schema_from_columns(all_columns)
                return scan_parquet(
                    [],
                    schema=schema_dict,
                ), version_key

            # Resolve file paths
            sources = [
                reader.resolve_data_file_path(f.path, f.path_is_relative, table)
                for f in data_files
            ]

            # Get delete files
            delete_files = reader.get_delete_files(table.table_id, snapshot.snapshot_id)

            # Build deletion files mapping: {file_index: [delete_file_paths]}
            deletion_files_map: dict[int, list[str]] | None = None
            if delete_files:
                # Map data_file_id -> file index in sources list
                file_id_to_idx = {
                    f.data_file_id: i for i, f in enumerate(data_files)
                }
                deletion_files_map = {}
                for df in delete_files:
                    idx = file_id_to_idx.get(df.data_file_id)
                    if idx is not None:
                        path = reader.resolve_data_file_path(df.path, df.path_is_relative, table)
                        deletion_files_map.setdefault(idx, []).append(path)

            # Build table statistics for file pruning
            table_statistics = None
            if filter_columns:
                file_ids = [f.data_file_id for f in data_files]
                col_ids = [
                    c.column_id
                    for c in columns
                    if c.column_name in filter_columns
                ]
                stats = reader.get_column_stats(table.table_id, file_ids, col_ids)
                table_statistics = build_table_statistics(
                    data_files, stats, columns, filter_columns
                )

            # Build scan_parquet kwargs
            kwargs: dict[str, Any] = {
                "missing_columns": "insert",
                "extra_columns": "ignore",
            }

            if table_statistics is not None:
                kwargs["_table_statistics"] = table_statistics

            if deletion_files_map:
                kwargs["_deletion_files"] = (
                    "iceberg-position-delete",
                    deletion_files_map,
                )

            lf = scan_parquet(sources, **kwargs)

            # If there's inlined data, we need to combine it
            inlined = reader.read_inlined_data(
                table.table_id,
                snapshot.snapshot_id,
                column_names,
            )
            if inlined is not None and not inlined.is_empty():
                lf = pl.concat([lf, inlined.lazy()], how="diagonal_relaxed")

            return lf, version_key
