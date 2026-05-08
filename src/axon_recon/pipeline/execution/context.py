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
    final_output_root: Path | None = None
    scratch_output_root: Path | None = None
    artifact_lookup_roots: tuple[Path, ...] = ()
    source_h5_path: Path | None = None

    @property
    def active_output_root(self) -> Path:
        return self.scratch_output_root if self.scratch_output_root is not None else self.mea_output_root


@dataclass(frozen=True)
class StageParallelism:
    max_workers: int
    max_stage_workers: int
    well_workers: int
    unit_workers: int
    unit_workers_source: str = "derived"
    max_simultaneous_well_reads_per_dataset: int | None = None
    divide_stage_workers_by_wells: bool = True
    task_allocation_plan: object | None = None
    set_thread_env: bool = False
    nested_thread_policy: str = "preserve_existing"
    use_hyperthreads: bool = False
