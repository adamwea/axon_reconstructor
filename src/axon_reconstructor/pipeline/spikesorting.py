from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class SpikeSortRequest:
    """Inputs needed to run or locate spikesorting outputs."""

    data_file: Path
    mea_output_root: Path
    sorter: str = "kilosort4"
    well: Optional[str] = None


def resolve_mea_sorter_output_dir(req: SpikeSortRequest) -> Path:
    """Resolve MEA_Analysis' expected sorter_output directory for a given well.

    This uses `axon_reconstructor.integrations.mea_analysis.compute_sorter_output_dir`,
    which relies on MEA_Analysis' `IPNAnalysis.path_contract`.

    If MEA_Analysis is not importable, raise a clear error.
    """

    try:
        from axon_reconstructor.integrations.mea_analysis import compute_sorter_output_dir
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Resolving sorter output dirs requires MEA_Analysis installed (import IPNAnalysis.path_contract)."
        ) from e

    if not req.well:
        raise ValueError("well is required to resolve a per-well sorter_output directory")

    return compute_sorter_output_dir(
        output_root=req.mea_output_root,
        data_file=req.data_file,
        well=req.well,
    )


def validate_sorter_output(sorter_output_dir: Path) -> bool:
    from axon_reconstructor.integrations.mea_analysis import validate_sorter_output_dir

    return validate_sorter_output_dir(sorter_output_dir)


def build_mea_analysis_driver_cmd(
    *,
    mea_repo_root: Path,
    data_file: Path,
    output_root: Path,
    sorter: str = "kilosort4",
    require_gpu: bool = False,
    cuda_visible_devices: Optional[str] = None,
    n_jobs: Optional[int] = None,
    chunk_duration: Optional[str] = None,
    scratch_dir: Optional[Path] = None,
    stage_back: str = "sorter",
    stage_back_mode: str = "copy",
) -> list[str]:
    """Build the MEA_Analysis run_pipeline_driver.py argv.

    This is a thin wrapper around `integrations.mea_analysis.MEAAnalysisRunSpec`.
    """

    from axon_reconstructor.integrations.mea_analysis import MEAAnalysisRunSpec, build_run_pipeline_driver_cmd

    spec = MEAAnalysisRunSpec(
        mea_analysis_repo_root=Path(mea_repo_root),
        path=Path(data_file),
        output_dir=Path(output_root),
        sorter=sorter,
        require_gpu=bool(require_gpu),
        cuda_visible_devices=cuda_visible_devices,
        n_jobs=n_jobs,
        chunk_duration=chunk_duration,
        scratch_dir=scratch_dir,
        stage_back=stage_back,
        stage_back_mode=stage_back_mode,
    )

    return build_run_pipeline_driver_cmd(spec)
