"""Preprocessing debug harness with validations.

Designed to be imported and called from a *very simple* project script launched
under the VS Code debugger.

This module is a developer utility; it is not part of the main pipeline API.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass(frozen=True)
class PreprocessInputs:
    h5_path: Path
    stream_id: str
    n_jobs: int = 8
    mea_output_root: Optional[Path] = None
    plot_layouts: bool = True
    force_restart: bool = False

    # Optional: temporal resampling (e.g. factor=10) to emulate higher sampling rate.
    temporal_resample_factor: Optional[int] = None
    temporal_resample_rate_hz: Optional[int] = None
    temporal_resample_margin_ms: float = 100.0
    temporal_resample_dtype: Optional[str] = None


@dataclass(frozen=True)
class PreprocessOutputs:
    multirec: Any
    common_electrodes: list[int]


def _ensure_maxwell_hdf5_plugin_env(logger: logging.Logger) -> None:
    """Ensure Maxwell HDF5 filter plugin env is set before h5py/spikeinterface import."""

    if os.environ.get("HDF5_PLUGIN_PATH"):
        logger.debug("HDF5_PLUGIN_PATH already set: %s", os.environ["HDF5_PLUGIN_PATH"])
        return

    try:
        from axon_reconstructor.pipeline.validation_helpers import ensure_maxwell_hdf5_plugin_env

        ensure_maxwell_hdf5_plugin_env(logger=logger, strict=False)
    except Exception as e:
        logger.warning("Could not configure Maxwell HDF5 plugin env: %s", e)


def _validate_inputs(inputs: PreprocessInputs) -> None:
    if not inputs.h5_path.exists():
        raise FileNotFoundError(f"Raw H5 not found: {inputs.h5_path}")
    if inputs.h5_path.suffix != ".h5":
        raise ValueError(f"Expected a .h5 file, got: {inputs.h5_path}")
    if not inputs.stream_id:
        raise ValueError("stream_id is empty")
    if inputs.n_jobs < 1:
        raise ValueError(f"n_jobs must be >= 1, got {inputs.n_jobs}")


def _log_recording_summary(logger: logging.Logger, rec: Any, common_el: list[int]) -> None:
    def _try(_label: str, fn):
        try:
            return fn()
        except Exception:
            return None

    nseg = _try("num_segments", lambda: int(rec.get_num_segments()))
    nch = _try("num_channels", lambda: int(rec.get_num_channels()))
    fs = _try("sampling_frequency", lambda: float(rec.get_sampling_frequency()))
    dtype = _try("dtype", lambda: str(rec.get_dtype()))

    logger.info(
        "Built recording: segments=%s channels=%s fs_hz=%s dtype=%s common_electrodes=%d",
        nseg,
        nch,
        fs,
        dtype,
        len(common_el),
    )

    ch_ids = _try("channel_ids", lambda: list(rec.get_channel_ids()))
    if ch_ids is not None:
        logger.debug("Channel ids preview: %s", [str(x) for x in ch_ids[:10]])

    locs = _try("channel_locations", lambda: rec.get_channel_locations())
    if locs is not None:
        try:
            import numpy as np  # type: ignore[import-not-found]

            locs = np.asarray(locs)
            logger.debug("Channel locations shape=%s dtype=%s", tuple(locs.shape), str(locs.dtype))
            if locs.size and not np.isfinite(locs).all():
                raise AssertionError("Non-finite values in channel locations")
        except ImportError:
            # If numpy isn't available in the environment, skip this check.
            pass


def validate_preprocessed_recording(*, multirec: Any, common_electrodes: list[int]) -> None:
    """Raise an exception if obvious invariants are violated."""

    if multirec is None:
        raise AssertionError("multirec is None")
    if not isinstance(common_electrodes, list):
        raise AssertionError(f"common_electrodes expected list, got {type(common_electrodes)}")

    try:
        nseg = int(multirec.get_num_segments())
        if nseg < 1:
            raise AssertionError(f"Expected >=1 segment, got {nseg}")
    except Exception:
        pass

    try:
        nch = int(multirec.get_num_channels())
        if nch < 1:
            raise AssertionError(f"Expected >=1 channel, got {nch}")
    except Exception:
        pass

    if len(common_electrodes) == 0:
        raise AssertionError("common_electrodes is empty; likely electrode intersection failed")


def run_preprocessing_with_validations(
    *,
    inputs: PreprocessInputs,
    logger: Optional[logging.Logger] = None,
) -> PreprocessOutputs:
    """Run preprocessing and perform verbose validations.

    Suggested breakpoints:
    - axon_reconstructor/pipeline/pipeline_driver.py : AxonReconstructor.preprocess_for_spikesorting
    - axon_reconstructor/pipeline/raw_preprocessing.py : build_preprocess_plan
    - axon_reconstructor/pipeline/raw_preprocessing.py : build_concatenated_recording
    """

    logger = logger or logging.getLogger("axon_reconstructor.devtools.preprocess")

    _validate_inputs(inputs)
    _ensure_maxwell_hdf5_plugin_env(logger)

    from axon_reconstructor.pipeline.pipeline_driver import AxonReconstructor
    from axon_reconstructor.pipeline import raw_preprocessing

    if inputs.mea_output_root is not None:
        inputs.mea_output_root.mkdir(parents=True, exist_ok=True)

    recon = AxonReconstructor(
        h5_parent_dirs=[inputs.h5_path],
        mea_analysis_output_root=str(inputs.mea_output_root) if inputs.mea_output_root else None,
        force_restart=bool(inputs.force_restart),
    )

    multirec, common_el = recon.preprocess_for_spikesorting(
        h5_path=inputs.h5_path,
        stream_id=str(inputs.stream_id),
        n_jobs=int(inputs.n_jobs),
        plot_layouts=bool(inputs.plot_layouts),
        temporal_resample_factor=(
            int(inputs.temporal_resample_factor) if inputs.temporal_resample_factor is not None else None
        ),
        temporal_resample_rate_hz=(
            int(inputs.temporal_resample_rate_hz) if inputs.temporal_resample_rate_hz is not None else None
        ),
        temporal_resample_margin_ms=float(inputs.temporal_resample_margin_ms),
        temporal_resample_dtype=(
            str(inputs.temporal_resample_dtype) if inputs.temporal_resample_dtype is not None else None
        ),
        overwrite_saved_recording=bool(inputs.force_restart),
    )

    _log_recording_summary(logger, multirec, common_el)
    validate_preprocessed_recording(multirec=multirec, common_electrodes=common_el)

    try:
        raw_preprocessing.print_time_between_segments(multirec)
    except Exception:
        pass

    return PreprocessOutputs(multirec=multirec, common_electrodes=common_el)
