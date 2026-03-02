from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from . import raw_preprocessing
from .checkpointing import (
    ProcessingStage,
    compute_checkpoint_file,
    exception_to_error_dict,
    load_checkpoint,
    save_checkpoint,
)
from .pipeline_logging import compute_pipeline_log_file, setup_pipeline_logger


PREPROCESS_OUTPUTS_DIRNAME = "preprocess_outputs"


def _compute_mea_analysis_output_dir(
    *,
    output_root: Path,
    data_file: Path,
    well: str,
) -> Path:
    """Compute MEA_Analysis-style per-well output directory.

    This mirrors the effective contract used by
    MEA_Analysis/IPNAnalysis/mea_analysis_routine.py (MEAPipeline._parse_metadata + output_dir).
    We intentionally do not import from MEA_Analysis here, because that repo is actively
    evolving and we want axon_reconstructor to be robust to internal refactors.
    """

    # Intentionally avoid `.resolve()` on output_root: users commonly provide a
    # symlinked path (e.g. /home/.../symlinks/local_RBS_data/outputs) and expect
    # outputs to appear under that same path, not the resolved target.
    output_root = Path(output_root).expanduser()

    # Keep data_file resolved to ensure stable metadata parsing even if it is a
    # symlink.
    data_file = Path(data_file).expanduser().resolve()

    # MEA_Analysis's MEAPipeline._parse_metadata() builds a relative_pattern like:
    #   os.path.join(*parts[-6:-1])
    # then sets:
    #   output_dir = Path(output_root) / relative_pattern / stream_id
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
        force_restart: bool = False,
        enable_checkpointing: bool = True,
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

        self.force_restart = bool(force_restart)
        self.enable_checkpointing = bool(enable_checkpointing)

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
        temporal_resample_factor: Optional[int] = None,
        temporal_resample_rate_hz: Optional[int] = None,
        temporal_resample_margin_ms: float = 100.0,
        temporal_resample_dtype: Optional[str] = None,
        save_recording: bool = True,
        overwrite_saved_recording: bool = True,
    ):
        """Build a concatenated SpikeInterface recording for a given h5 + stream."""

        plan = raw_preprocessing.build_preprocess_plan(h5_path=h5_path, stream_id=stream_id)

        # If we can compute the per-well output dir, also set up a Mandar-style log file.
        logger = self.logger
        well_out_dir = None
        if self.mea_analysis_output_root is not None and (plot_layouts or save_recording or self.enable_checkpointing):
            well_out_dir = _compute_mea_analysis_output_dir(
                output_root=self.mea_analysis_output_root,
                data_file=h5_path,
                well=stream_id,
            )
            try:
                log_file = compute_pipeline_log_file(
                    well_out_dir=well_out_dir,
                    data_file=h5_path,
                    stream_id=stream_id,
                )
                logger = setup_pipeline_logger(
                    log_file=log_file,
                    logger_name=f"axon_reconstructor.{log_file.stem}",
                    verbose=True,
                )
            except Exception:
                # Never let logging setup break preprocessing.
                logger = self.logger

        if plan.cfg_files:
            logger.info("Discovered %d cfg files next to %s", len(plan.cfg_files), plan.h5_path)
        else:
            logger.info("No .cfg files discovered next to %s; using contact_vector electrodes", plan.h5_path)

        plot_dir = None
        if plot_layouts:
            if well_out_dir is None:
                logger.warning("plot_layouts=True but mea_analysis_output_root is not set; skipping plots")
            else:
                plot_dir = well_out_dir / PREPROCESS_OUTPUTS_DIRNAME
                plot_dir.mkdir(parents=True, exist_ok=True)
                logger.info("Preprocess diagnostics output: %s", plot_dir)

        if save_recording and well_out_dir is None:
            logger.warning(
                "save_recording=True but mea_analysis_output_root is not set; skipping recording save"
            )

        checkpoint_file = None
        checkpoint_state = None
        preprocess_dir = None
        recording_dir = None
        common_el_path = None
        epoch_maxwell_path = None
        epoch_concat_path = None

        if self.enable_checkpointing and well_out_dir is not None:
            checkpoint_file = compute_checkpoint_file(
                output_dir=well_out_dir,
                file_path=h5_path,
                stream_id=stream_id,
            )
            checkpoint_state = load_checkpoint(
                checkpoint_file=checkpoint_file,
                force_restart=self.force_restart,
                output_dir=well_out_dir,
                file_path=h5_path,
                stream_id=stream_id,
            )

        if well_out_dir is not None:
            preprocess_dir = well_out_dir / PREPROCESS_OUTPUTS_DIRNAME
            recording_dir = preprocess_dir / "preprocessed_recording"
            common_el_path = preprocess_dir / "common_electrodes.npy"
            epoch_maxwell_path = preprocess_dir / f"maxwell_contiguous_epochs_{stream_id}.json"
            epoch_concat_path = preprocess_dir / f"concatenation_stitch_epochs_{stream_id}.json"

        preprocess_cfg_path = (preprocess_dir / "preprocess_config.json") if preprocess_dir is not None else None
        requested_cfg = {
            "temporal_resample_factor": (int(temporal_resample_factor) if temporal_resample_factor is not None else None),
            "temporal_resample_rate_hz": (int(temporal_resample_rate_hz) if temporal_resample_rate_hz is not None else None),
            "temporal_resample_margin_ms": float(temporal_resample_margin_ms),
            "temporal_resample_dtype": (str(temporal_resample_dtype) if temporal_resample_dtype is not None else None),
        }

        # Resume shortcut: if preprocessing is complete and the caller doesn't want to overwrite.
        if (
            save_recording
            and not overwrite_saved_recording
            and checkpoint_state is not None
            and checkpoint_state.stage >= ProcessingStage.PREPROCESSING_COMPLETE.value
            and recording_dir is not None
            and common_el_path is not None
            and recording_dir.exists()
            and common_el_path.exists()
        ):
            try:
                import numpy as np  # type: ignore[import-not-found]
                import spikeinterface.full as si  # type: ignore[import-not-found]

                # If a config is present and doesn't match, do not resume.
                if preprocess_cfg_path is not None and preprocess_cfg_path.exists():
                    try:
                        import json

                        saved_cfg = json.loads(preprocess_cfg_path.read_text(errors="replace"))
                        if isinstance(saved_cfg, dict) and saved_cfg.get("requested_cfg") != requested_cfg:
                            raise RuntimeError(
                                f"Saved preprocess_config.json does not match requested options; re-running preprocessing. "
                                f"(saved at {preprocess_cfg_path})"
                            )
                    except Exception as e:
                        raise
                elif temporal_resample_rate_hz is not None or temporal_resample_factor is not None:
                    raise RuntimeError("Temporal resampling requested but no preprocess_config.json found; re-running")

                try:
                    multirec = si.load(recording_dir)
                except Exception:
                    multirec = si.load_extractor(recording_dir)

                common_el = np.load(common_el_path).tolist()
                logger.info("Resuming: loaded preprocessed recording from %s", recording_dir)
                return multirec, common_el
            except Exception as e:
                logger.warning("Failed to resume from saved preprocessed recording (%s); re-running", e)

        if checkpoint_file is not None and checkpoint_state is not None:
            checkpoint_state = save_checkpoint(
                checkpoint_file=checkpoint_file,
                state=checkpoint_state,
                stage=ProcessingStage.PREPROCESSING,
                failed_stage=None,
                error=None,
                extra_fields={
                    "preprocess_outputs_dir": str(preprocess_dir) if preprocess_dir else None,
                },
            )

        try:
            multirec, common_el = raw_preprocessing.build_concatenated_recording(
                h5_path=plan.h5_path,
                stream_id=plan.stream_id,
                n_jobs=n_jobs,
                plot_output_dir=plot_dir,
                epoch_markers_output_dir=(preprocess_dir if preprocess_dir is not None else plot_dir),
                temporal_resample_factor=(int(temporal_resample_factor) if temporal_resample_factor is not None else None),
                temporal_resample_rate_hz=(int(temporal_resample_rate_hz) if temporal_resample_rate_hz is not None else None),
                temporal_resample_margin_ms=float(temporal_resample_margin_ms),
                temporal_resample_dtype=(str(temporal_resample_dtype) if temporal_resample_dtype is not None else None),
            )
            logger.info("Preprocessed recording built; common electrodes=%d", len(common_el))
            if epoch_maxwell_path is not None and epoch_concat_path is not None:
                if epoch_maxwell_path.exists():
                    logger.info("Epoch markers (Maxwell): %s", epoch_maxwell_path)
                if epoch_concat_path.exists():
                    logger.info("Epoch markers (Concat stitches): %s", epoch_concat_path)
        except Exception as e:
            if checkpoint_file is not None and checkpoint_state is not None:
                save_checkpoint(
                    checkpoint_file=checkpoint_file,
                    state=checkpoint_state,
                    stage=ProcessingStage.NOT_STARTED,
                    failed_stage=ProcessingStage.PREPROCESSING.name,
                    error=exception_to_error_dict(e),
                )
            raise

        if save_recording and well_out_dir is not None:
            # Persist the preprocessed recording so later stages can be debugged independently.
            assert preprocess_dir is not None
            assert recording_dir is not None
            assert common_el_path is not None
            preprocess_dir.mkdir(parents=True, exist_ok=True)

            if preprocess_cfg_path is not None:
                try:
                    import json
                    import datetime as dt

                    preprocess_cfg_path.write_text(
                        json.dumps(
                            {
                                "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                                "requested_cfg": requested_cfg,
                            },
                            indent=2,
                            sort_keys=True,
                        )
                        + "\n"
                    )
                except Exception as e:
                    logger.warning("Failed to write preprocess_config.json: %s", e)
                try:
                    import numpy as np  # type: ignore[import-not-found]
                    import spikeinterface.full as si  # type: ignore[import-not-found]

                    if recording_dir.exists() and overwrite_saved_recording:
                        # `Recording.save(..., overwrite=True)` isn't consistent across all SI versions
                        # for all formats, so we proactively clean the folder.
                        import shutil

                        shutil.rmtree(recording_dir)

                    if (not recording_dir.exists()) or overwrite_saved_recording:
                        logger.info("Saving preprocessed recording to %s", recording_dir)
                        multirec.save(
                            folder=recording_dir,
                            format="binary",
                            overwrite=True,
                            n_jobs=n_jobs,
                            chunk_duration="1s",
                            progress_bar=False,
                        )
                    else:
                        logger.info("Preprocessed recording already exists at %s; not overwriting", recording_dir)

                    np.save(common_el_path, np.asarray(common_el, dtype=np.int64))
                except Exception as e:
                    logger.error("Failed to save preprocessed recording artifacts: %s", e)
                    if checkpoint_file is not None and checkpoint_state is not None:
                        save_checkpoint(
                            checkpoint_file=checkpoint_file,
                            state=checkpoint_state,
                            stage=ProcessingStage.NOT_STARTED,
                            failed_stage=ProcessingStage.PREPROCESSING.name,
                            error={
                                "type": type(e).__name__,
                                "message": str(e),
                            },
                        )
                    raise RuntimeError(f"Failed to save preprocessed recording artifacts: {e}") from e

        if checkpoint_file is not None and checkpoint_state is not None:
            preprocessed_recording_dir = str(recording_dir) if recording_dir and recording_dir.exists() else None
            common_electrodes_path = str(common_el_path) if common_el_path and common_el_path.exists() else None
            maxwell_epochs_path = str(epoch_maxwell_path) if epoch_maxwell_path and epoch_maxwell_path.exists() else None
            concat_epochs_path = str(epoch_concat_path) if epoch_concat_path and epoch_concat_path.exists() else None
            checkpoint_state = save_checkpoint(
                checkpoint_file=checkpoint_file,
                state=checkpoint_state,
                stage=ProcessingStage.PREPROCESSING_COMPLETE,
                failed_stage=None,
                error=None,
                extra_fields={
                    "preprocessed_recording_dir": preprocessed_recording_dir,
                    "common_electrodes_path": common_electrodes_path,
                    "n_common_electrodes": len(common_el),
                    "maxwell_epochs_path": maxwell_epochs_path,
                    "concat_epochs_path": concat_epochs_path,
                },
            )

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
