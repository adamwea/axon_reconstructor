from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class DebugDatasetConfig:
    """Configuration used by debug harnesses.

    This intentionally mirrors the env vars used by the private rerun bundle
    scripts in `dev/projects/260117_debugging_sorting_axontracking/_env.sh`.

    The goal is: a debug script can simply call `load_debug_dataset_config()`
    and then step through the pipeline code.
    """

    raw_h5: Path
    out_root: Path
    mea_repo: Optional[Path]
    axon_repo: Optional[Path]

    sorter: str
    n_jobs: int
    stream_id: str

    # Optional NERSC GPU sorting context (not required for preprocessing)
    account: Optional[str] = None
    shifter_image: Optional[str] = None


def _get_env(name: str) -> Optional[str]:
    value = os.environ.get(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def load_debug_dataset_config() -> DebugDatasetConfig:
    """Load debug configuration from environment variables.

    Required:
    - RAW_H5
    - OUT_ROOT
    - STREAM_ID

    Optional:
    - MEA_REPO, AXON_REPO, SORTER, N_JOBS, GPU_SMOKE_SALLOC_ACCOUNT, SHIFTER_IMAGE

    If you `source` the private bundle's `_env.sh` in your VS Code launch config
    (or terminal), these will already be set.
    """

    raw_h5 = _get_env("RAW_H5")
    out_root = _get_env("OUT_ROOT")
    stream_id = _get_env("STREAM_ID")

    missing = [k for k, v in [("RAW_H5", raw_h5), ("OUT_ROOT", out_root), ("STREAM_ID", stream_id)] if not v]
    if missing:
        raise RuntimeError(
            "Missing required debug env vars: "
            + ", ".join(missing)
            + ". "
            #+ "Tip: source dev/projects/260117_debugging_sorting_axontracking/_env.sh and set STREAM_ID."
        )

    mea_repo = _get_env("MEA_REPO")
    axon_repo = _get_env("AXON_REPO")

    sorter = _get_env("SORTER") or "kilosort4"

    n_jobs_str = _get_env("N_JOBS") or "8"
    try:
        n_jobs = int(n_jobs_str)
    except ValueError as e:
        raise RuntimeError(f"Invalid N_JOBS={n_jobs_str!r}; expected int") from e

    account = _get_env("GPU_SMOKE_SALLOC_ACCOUNT")
    shifter_image = _get_env("SHIFTER_IMAGE")

    return DebugDatasetConfig(
        raw_h5=Path(raw_h5).expanduser().resolve(),
        out_root=Path(out_root).expanduser().resolve(),
        mea_repo=Path(mea_repo).expanduser().resolve() if mea_repo else None,
        axon_repo=Path(axon_repo).expanduser().resolve() if axon_repo else None,
        sorter=sorter,
        n_jobs=n_jobs,
        stream_id=str(stream_id),
        account=account,
        shifter_image=shifter_image,
    )
