"""Loaders for the Alibaba cluster-trace-gpu-v2020 tables stored as Parquet.

The package does not know where the data lives; callers pass the folder
(e.g. the Drive data/interim folder in Colab).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import pyarrow.parquet as pq

# Short table name -> Parquet file name in the interim folder
TABLE_FILES = {
    "job": "pai_job_table.parquet",
    "task": "pai_task_table.parquet",
    "instance": "pai_instance_table.parquet",
    "sensor": "pai_sensor_table.parquet",
    "group_tag": "pai_group_tag_table.parquet",
    "machine_spec": "pai_machine_spec.parquet",
    "machine_metric": "pai_machine_metric.parquet",
}


def table_path(name: str, data_dir: str | Path) -> Path:
    """Return the Parquet path of a table after checking the name and the file."""
    if name not in TABLE_FILES:
        valid = ", ".join(TABLE_FILES)
        raise ValueError(f"Unknown table '{name}'. Valid names: {valid}")
    path = Path(data_dir) / TABLE_FILES[name]
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")
    return path


def table_columns(name: str, data_dir: str | Path) -> list[str]:
    """Return the column names of a table without loading its data."""
    return pq.read_schema(table_path(name, data_dir)).names


def load_table(
    name: str,
    data_dir: str | Path,
    columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Load one table; pass ``columns`` to read only what you need (saves RAM)."""
    path = table_path(name, data_dir)
    cols = list(columns) if columns is not None else None
    return pd.read_parquet(path, columns=cols)