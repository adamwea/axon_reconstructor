from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from axon_reconstructor.pipeline import raw_preprocessing


def _compute_mea_analysis_output_dir(
    *,
    output_root: Path,
    data_file: Path,
    well: str,
) -> Path:
    """Compute MEA_Analysis-style per-well output directory.

    Uses the dependency-free contract in MEA_Analysis when available.
    Falls back to an identical local implementation if MEA_Analysis is not importable.
    """

    output_root = Path(output_root).expanduser().resolve()
    data_file = Path(data_file).expanduser().resolve()

    try:
        # Prefer the authoritative contract from MEA_Analysis.
        from MEA_Analysis.IPNAnalysis.path_contract import compute_output_dir  # type: ignore

        return compute_output_dir(output_root=output_root, data_file=data_file, well=str(well))
    except Exception:
        # Fallback: mirror MEA_Analysis.IPNAnalysis.path_contract exactly.
        import os

        try:
            relative_pattern = f"{data_file.parent.parent.name}/{data_file.parent.name}/{data_file.name}"
        except Exception:
            relative_pattern = str(data_file.name)

        parts = str(data_file).split(os.sep)
        if len(parts) > 5:
            relative_pattern = os.path.join(*parts[-6:-1])

        return output_root / relative_pattern / str(well)


@dataclass
class PipelinePaths:
    recordings_dir: Path = Path("./data/temp_data/recordings")
    sortings_dir: Path = Path("./data/temp_data/sortings")
    waveforms_dir: Path = Path("./data/temp_data/waveforms")
    templates_dir: Path = Path("./data/temp_data/templates")
    recon_dir: Path = Path("./data/reconstructions")


class AxonReconstructor:
    """Rebuilt pipeline driver (WIP).

    This is the new home for the axon recon pipeline orchestration.

    Milestone (current): rebuild + improve the preprocessing and spikesorting
    steps that prepare data for MEA_Analysis.
    """

    def __init__(
        self,
        h5_parent_dirs: list[str] | list[Path],
        *,
        mea_environment: str = "nersc",
        mea_analysis_output_root: Optional[str] = None,
        mea_analysis_repo_root: Optional[str] = None,
        mea_analysis_docker_image: Optional[str] = None,
        mea_auto_run_driver: bool = False,
        paths: Optional[PipelinePaths] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.h5_parent_dirs = [Path(p) for p in h5_parent_dirs]

        self.mea_environment = mea_environment
        if mea_analysis_output_root:
            self.mea_analysis_output_root = Path(mea_analysis_output_root)
        else:
            env_out = os.environ.get("MEA_ANALYSIS_OUTPUT_DIR")
            self.mea_analysis_output_root = Path(env_out) if env_out else None
        self.mea_analysis_repo_root = Path(mea_analysis_repo_root) if mea_analysis_repo_root else None
        self.mea_analysis_docker_image = mea_analysis_docker_image
        self.mea_auto_run_driver = bool(mea_auto_run_driver)

        self.paths = paths or PipelinePaths()

        self.logger = logger or logging.getLogger("axon_reconstructor")
        if not self.logger.handlers:
            logging.basicConfig(level=logging.INFO)

    def iter_raw_h5_files(self) -> list[Path]:
        files: list[Path] = []
        for root in self.h5_parent_dirs:
            if root.is_file() and root.suffix == ".h5":
                files.append(root)
                continue
            if root.is_dir():
                files.extend(sorted(root.rglob("*.h5")))
        return files

    def preprocess_for_spikesorting(
        self,
        *,
        h5_path: Path,
        stream_id: str,
        n_jobs: int = 8,
        plot_layouts: bool = True,
    ):
        """Build a concatenated SpikeInterface recording for a given h5 + stream."""

        plan = raw_preprocessing.build_preprocess_plan(h5_path=h5_path, stream_id=stream_id)
        if plan.cfg_files:
            self.logger.info("Discovered %d cfg files next to %s", len(plan.cfg_files), plan.h5_path)
        else:
            self.logger.info("No .cfg files discovered next to %s; using contact_vector electrodes", plan.h5_path)

        plot_dir = None
        if plot_layouts:
            if self.mea_analysis_output_root is None:
                self.logger.warning(
                    "plot_layouts=True but mea_analysis_output_root is not set; skipping plots"
                )
            else:
                well_out_dir = _compute_mea_analysis_output_dir(
                    output_root=self.mea_analysis_output_root,
                    data_file=h5_path,
                    well=stream_id,
                )
                plot_dir = well_out_dir / "axon_reconstructor" / "preprocess"
                plot_dir.mkdir(parents=True, exist_ok=True)
                self.logger.info("Preprocess diagnostics output: %s", plot_dir)

        multirec, common_el = raw_preprocessing.build_concatenated_recording(
            h5_path=plan.h5_path,
            stream_id=plan.stream_id,
            n_jobs=n_jobs,
            plot_output_dir=plot_dir,
        )
        self.logger.info("Concatenated recording built; common electrodes=%d", len(common_el))
        return multirec, common_el

    def run_pipeline(
        self,
        *,
        concatenate_switch: bool = True,
        sort_switch: bool = True,
        waveform_switch: bool = True,
        template_switch: bool = True,
        recon_switch: bool = True,
        only_load_sortings: bool = False,
    ) -> None:
        """Entry point kept for CLI compatibility.

        For now, this driver focuses on preprocessing + spikesorting milestones.
        Downstream waveform/template/reconstruction steps will be reintroduced
        incrementally.
        """

        self.logger.info(
            "run_pipeline: concatenate=%s sort=%s waveforms=%s templates=%s recon=%s",
            concatenate_switch,
            sort_switch,
            waveform_switch,
            template_switch,
            recon_switch,
        )

        raw_files = self.iter_raw_h5_files()
        if not raw_files:
            self.logger.warning("No .h5 files found under: %s", self.h5_parent_dirs)
            return

        # NOTE: We are not yet running spikesorting from here; NERSC sorting
        # remains driven via `axon-reconstructor gpu-interact` or external runs.
        # This method will be expanded as the preprocessing/spikesorting APIs
        # stabilize.
        if concatenate_switch:
            self.logger.info("Preprocessing milestone: concatenate recordings (per stream)")
            # Placeholder: user will guide stream selection / mapping.
            # We intentionally avoid scanning all wells here until we define the
            # correct dataset semantics.

        if sort_switch and only_load_sortings:
            self.logger.info("Sorting milestone: will only load/validate existing sorter outputs")

        if waveform_switch or template_switch or recon_switch:
            self.logger.info("Downstream steps are currently WIP in pipeline_driver")

        return
