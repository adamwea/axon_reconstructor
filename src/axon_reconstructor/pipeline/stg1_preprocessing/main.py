"""Raw preprocessing public API.

The heavy concatenation implementation lives in .runner to keep this module orchestration-only.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from ..checkpointing import (
    ProcessingStage,
    compute_checkpoint_file,
    exception_to_error_dict,
    load_checkpoint,
    save_checkpoint,
)
from ..output_paths import compute_mea_analysis_output_dir
from ..pipeline_logging import compute_pipeline_log_file, setup_pipeline_logger
from .constants import PREPROCESS_OUTPUTS_DIRNAME

from .concatenation import find_common_electrodes_from_segments
from .planning import RawPreprocessPlan, build_preprocess_plan, discover_cfg_files, parse_cfg_channel_locations
from .runner import build_concatenated_recording


def run_preprocess_stage(
    *,
    h5_path: Path,
    stream_id: str,
    mea_output_root: Optional[Path],
    force_restart: bool,
    enable_checkpointing: bool = True,
    n_jobs: int = 8,
    plot_layouts: bool = True,
    temporal_resample_factor: Optional[int] = None,
    temporal_resample_rate_hz: Optional[int] = None,
    temporal_resample_margin_ms: float = 100.0,
    temporal_resample_dtype: Optional[str] = None,
    save_recording: bool = True,
    overwrite_saved_recording: bool = True,
    logger: Optional[logging.Logger] = None,
) -> tuple[object, list[int]]:
    plan = build_preprocess_plan(h5_path=h5_path, stream_id=stream_id)

    logger = logger or logging.getLogger("axon_reconstructor.preprocess")
    well_out_dir = None
    if mea_output_root is not None and (plot_layouts or save_recording or enable_checkpointing):
        well_out_dir = compute_mea_analysis_output_dir(
            output_root=mea_output_root,
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
            pass

    if plan.cfg_files:
        logger.info("Discovered %d cfg files next to %s", len(plan.cfg_files), plan.h5_path)
    else:
        logger.info("No .cfg files discovered next to %s; using contact_vector electrodes", plan.h5_path)

    plot_dir = None
    if plot_layouts:
        if well_out_dir is None:
            logger.warning("plot_layouts=True but mea_output_root is not set; skipping plots")
        else:
            plot_dir = well_out_dir / PREPROCESS_OUTPUTS_DIRNAME
            plot_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Preprocess diagnostics output: %s", plot_dir)

    if save_recording and well_out_dir is None:
        logger.warning("save_recording=True but mea_output_root is not set; skipping recording save")

    checkpoint_file = None
    checkpoint_state = None
    preprocess_dir = None
    recording_dir = None
    common_el_path = None
    epoch_maxwell_path = None
    epoch_concat_path = None

    if enable_checkpointing and well_out_dir is not None:
        checkpoint_file = compute_checkpoint_file(
            output_dir=well_out_dir,
            file_path=h5_path,
            stream_id=stream_id,
        )
        checkpoint_state = load_checkpoint(
            checkpoint_file=checkpoint_file,
            force_restart=force_restart,
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

            if preprocess_cfg_path is not None and preprocess_cfg_path.exists():
                import json

                saved_cfg = json.loads(preprocess_cfg_path.read_text(errors="replace"))
                if isinstance(saved_cfg, dict) and saved_cfg.get("requested_cfg") != requested_cfg:
                    raise RuntimeError(
                        f"Saved preprocess_config.json does not match requested options; re-running preprocessing. "
                        f"(saved at {preprocess_cfg_path})"
                    )
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
                "stg1_preprocess_outputs_dir": str(preprocess_dir) if preprocess_dir else None,
            },
        )

    try:
        multirec, common_el = build_concatenated_recording(
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
        assert preprocess_dir is not None
        assert recording_dir is not None
        assert common_el_path is not None
        preprocess_dir.mkdir(parents=True, exist_ok=True)

        if preprocess_cfg_path is not None:
            try:
                import datetime as dt
                import json

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

                if recording_dir.exists() and overwrite_saved_recording:
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
        _ = save_checkpoint(
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

__all__ = [
    "RawPreprocessPlan",
    "discover_cfg_files",
    "parse_cfg_channel_locations",
    "build_preprocess_plan",
    "find_common_electrodes_from_segments",
    "build_concatenated_recording",
    "run_preprocess_stage",
]
