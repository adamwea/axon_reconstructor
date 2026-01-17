from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence


_INSTALLED_PATH_CONTRACT = importlib.import_module("IPNAnalysis.path_contract")


def compute_mea_relative_pattern(
    data_file: os.PathLike[str] | str,
) -> str:
    """Compute the MEA_Analysis relative_pattern for a data file.

    Source of truth is MEA_Analysis' dependency-free module: `IPNAnalysis.path_contract`.
    We deliberately avoid re-implementing this logic in axon_reconstructor.

    This requires MEA_Analysis to be installed (editable is fine) such that
    `IPNAnalysis.path_contract` is importable.
    """

    return str(_INSTALLED_PATH_CONTRACT.compute_relative_pattern(data_file))


def compute_mea_output_dir(
    *,
    output_root: os.PathLike[str] | str,
    data_file: os.PathLike[str] | str,
    well: str,
) -> Path:
    """Compute MEA_Analysis' per-well output directory under output_root."""

    output_root = Path(output_root).resolve()
    relative_pattern = compute_mea_relative_pattern(data_file)
    return output_root / relative_pattern / well


def compute_sorter_output_dir(
    *,
    output_root: os.PathLike[str] | str,
    data_file: os.PathLike[str] | str,
    well: str,
) -> Path:
    return (
        compute_mea_output_dir(
            output_root=output_root,
            data_file=data_file,
            well=well,
        )
        / "sorter_output"
    )


def validate_sorter_output_dir(sorter_output_dir: os.PathLike[str] | str) -> bool:
    """Lightweight check that a sorter_output folder exists and is plausibly non-empty.

    This intentionally avoids importing spikeinterface.
    """

    folder = Path(sorter_output_dir)
    if not folder.exists() or not folder.is_dir():
        return False

    # Common Kilosort/SpikeInterface artifacts (not guaranteed, but a good sanity check)
    expected_any = {
        "params.py",
        "spike_times.npy",
        "spike_clusters.npy",
        "ops.npy",
        "channel_map.npy",
        "cluster_info.tsv",
    }

    try:
        entries = {p.name for p in folder.iterdir() if p.is_file()}
    except OSError:
        return False

    if entries & expected_any:
        return True

    # Fallback: non-empty directory
    try:
        next(folder.iterdir())
        return True
    except StopIteration:
        return False


@dataclass(frozen=True)
class MEAAnalysisRunSpec:
    mea_analysis_repo_root: Path
    path: Path
    output_dir: Path
    sorter: str = "kilosort4"
    reference: Optional[Path] = None
    assay_types: Sequence[str] = ("network today", "network today/best")
    params: Optional[str] = None  # JSON file path or inline JSON string
    docker: Optional[str] = None
    skip_spikesorting: bool = False
    force_restart: bool = False
    debug: bool = False
    clean_up: bool = False
    export_to_phy: bool = False
    no_curation: bool = False
    checkpoint_dir: Optional[Path] = None

    # GPU/HPC carveouts
    cuda_visible_devices: Optional[str] = None
    require_gpu: bool = False
    n_jobs: Optional[int] = None
    chunk_duration: Optional[str] = None
    scratch_dir: Optional[Path] = None
    stage_back: str = "sorter"  # none|sorter|all
    stage_back_mode: str = "copy"  # copy|move


def build_run_pipeline_driver_cmd(spec: MEAAnalysisRunSpec) -> list[str]:
    """Build a subprocess argv list to run MEA_Analysis' IPNAnalysis driver.

    This is intentionally pure string/Path handling so it can be unit-tested without
    external runtime dependencies.
    """

    driver = spec.mea_analysis_repo_root / "IPNAnalysis" / "run_pipeline_driver.py"
    argv: list[str] = [
        "python3",
        str(driver),
        str(spec.path),
        "--output-dir",
        str(spec.output_dir),
        "--sorter",
        spec.sorter,
    ]

    if spec.reference is not None:
        argv += ["--reference", str(spec.reference)]

    if spec.assay_types:
        argv += ["--type", *list(spec.assay_types)]

    if spec.params is not None:
        argv += ["--params", spec.params]

    if spec.docker is not None:
        argv += ["--docker", spec.docker]

    if spec.checkpoint_dir is not None:
        argv += ["--checkpoint-dir", str(spec.checkpoint_dir)]

    if spec.skip_spikesorting:
        argv += ["--skip-spikesorting"]

    if spec.force_restart:
        argv += ["--force-restart"]

    if spec.debug:
        argv += ["--debug"]

    if spec.clean_up:
        argv += ["--clean-up"]

    if spec.export_to_phy:
        argv += ["--export-to-phy"]

    if spec.no_curation:
        argv += ["--no-curation"]

    # GPU/HPC carveouts
    if spec.cuda_visible_devices is not None:
        argv += ["--cuda-visible-devices", spec.cuda_visible_devices]

    if spec.require_gpu:
        argv += ["--require-gpu"]

    if spec.n_jobs is not None:
        argv += ["--n-jobs", str(int(spec.n_jobs))]

    if spec.chunk_duration is not None:
        argv += ["--chunk-duration", spec.chunk_duration]

    if spec.scratch_dir is not None:
        argv += ["--scratch-dir", str(spec.scratch_dir)]
        argv += ["--stage-back", spec.stage_back]
        argv += ["--stage-back-mode", spec.stage_back_mode]

    return argv
