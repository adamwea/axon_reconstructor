from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExecutionTarget:
    dataset_index: int
    dataset_id: str
    h5_path: Path
    stream_id: str
    mea_output_root: Path


@dataclass(frozen=True)
class StageParallelism:
    max_workers: int
    max_stage_workers: int
    well_workers: int
    unit_workers: int
