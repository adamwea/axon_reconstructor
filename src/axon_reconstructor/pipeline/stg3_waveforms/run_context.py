from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ..checkpointing import (
    load_checkpoint,
)
from ..output_paths import compute_mea_analysis_output_dir
from ..stg1_preprocessing.constants import PREPROCESS_OUTPUTS_DIRNAME
from ..pipeline_logging import build_stage_logger
from ..checkpointing import compute_stage_checkpoint_file

from .constants import WAVEFORMS_OUTPUTS_DIRNAME
from .utils import _epochs_to_intervals, _infer_cutout_ms, _read_json


def _compute_waveforms_checkpoint_file(*, well_out_dir: Path, h5_path: Path, stream_id: str) -> Path:
    """Use a dedicated checkpoint file for waveforms.

    The MEA_Analysis-style stage machine (PREPROCESSING..REPORTS_COMPLETE) doesn't
    include waveforms, so storing waveforms progress in the main checkpoint can
    accidentally *regress* stage numbers (e.g. REPORTS_COMPLETE -> ANALYZER_COMPLETE).
    """

    return compute_stage_checkpoint_file(
        well_out_dir=well_out_dir,
        h5_path=h5_path,
        stream_id=stream_id,
        stage_name="waveforms",
    )


def _compute_waveforms_out_dir(*, output_root: Path, data_file: Path, well: str) -> Path:
    return compute_mea_analysis_output_dir(output_root=output_root, data_file=data_file, well=well) / WAVEFORMS_OUTPUTS_DIRNAME


@dataclass(frozen=True)
class _WaveformsRunContext:
    well_out_dir: Path
    logger: Any
    waveforms_out_dir: Path
    concat_waveforms_dir: Path
    segment_waveforms_dir: Optional[Path]
    params_json: Path
    filtering_json: Path
    ckpt_file: Path
    ckpt: dict[str, Any]


@dataclass(frozen=True)
class _WaveformWindow:
    fs_hz: float
    ms_before: float
    ms_after: float
    pre_samples: int
    post_samples: int


def _initialize_run_context(*, inputs, logger_name_prefix: str) -> _WaveformsRunContext:
    well_out_dir = compute_mea_analysis_output_dir(
        output_root=inputs.mea_output_root,
        data_file=inputs.h5_path,
        well=inputs.stream_id,
    )

    logger = build_stage_logger(
        well_out_dir=well_out_dir,
        data_file=inputs.h5_path,
        stream_id=inputs.stream_id,
        stage_name="waveforms",
        logger_name_prefix=logger_name_prefix,
        verbose=True,
    )

    waveforms_out_dir = _compute_waveforms_out_dir(
        output_root=inputs.mea_output_root,
        data_file=inputs.h5_path,
        well=inputs.stream_id,
    )
    waveforms_out_dir.mkdir(parents=True, exist_ok=True)

    concat_waveforms_dir = waveforms_out_dir / "concat_waveforms"
    segment_waveforms_dir = waveforms_out_dir / "segment_waveforms" if inputs.per_segment else None

    params_json = waveforms_out_dir / "waveform_extraction_params.json"
    filtering_json = waveforms_out_dir / "waveform_filtering_summary.json"

    ckpt_file = _compute_waveforms_checkpoint_file(well_out_dir=well_out_dir, h5_path=inputs.h5_path, stream_id=inputs.stream_id)
    ckpt = load_checkpoint(
        checkpoint_file=ckpt_file,
        force_restart=bool(inputs.force_restart),
        output_dir=well_out_dir,
        file_path=inputs.h5_path,
        stream_id=inputs.stream_id,
    )

    return _WaveformsRunContext(
        well_out_dir=well_out_dir,
        logger=logger,
        waveforms_out_dir=waveforms_out_dir,
        concat_waveforms_dir=concat_waveforms_dir,
        segment_waveforms_dir=segment_waveforms_dir,
        params_json=params_json,
        filtering_json=filtering_json,
        ckpt_file=ckpt_file,
        ckpt=ckpt,
    )


def _resolve_waveform_window(*, inputs, fs_hz: float) -> _WaveformWindow:
    ms_before = float(inputs.ms_before) if inputs.ms_before is not None else None
    ms_after = float(inputs.ms_after) if inputs.ms_after is not None else None
    if ms_before is None or ms_after is None:
        inferred_before, inferred_after = _infer_cutout_ms(h5_path=inputs.h5_path, stream_id=inputs.stream_id, fs_hz=fs_hz)
        ms_before = inferred_before if ms_before is None else ms_before
        ms_after = inferred_after if ms_after is None else ms_after

    assert ms_before is not None
    assert ms_after is not None

    pre_samples = int(math.ceil(ms_before * fs_hz / 1000.0))
    post_samples = int(math.ceil(ms_after * fs_hz / 1000.0))

    return _WaveformWindow(
        fs_hz=float(fs_hz),
        ms_before=float(ms_before),
        ms_after=float(ms_after),
        pre_samples=int(pre_samples),
        post_samples=int(post_samples),
    )


@dataclass(frozen=True)
class _EpochInputs:
    preprocess_dir: Path
    maxwell_epochs_path: Path
    concat_epochs_path: Path
    maxwell_epochs: list[dict]
    maxwell_intervals: list[tuple[int, int]]
    concat_epochs: list[dict]


def _load_epoch_markers(*, well_out_dir: Path, stream_id: str) -> _EpochInputs:
    preprocess_dir = well_out_dir / PREPROCESS_OUTPUTS_DIRNAME
    maxwell_epochs_path = preprocess_dir / f"maxwell_contiguous_epochs_{stream_id}.json"
    concat_epochs_path = preprocess_dir / f"concatenation_stitch_epochs_{stream_id}.json"

    maxwell_epochs: list[dict] = []
    maxwell_intervals: list[tuple[int, int]] = []
    concat_epochs: list[dict] = []

    if maxwell_epochs_path.exists():
        maxwell_epochs = list(_read_json(maxwell_epochs_path))
        maxwell_intervals = _epochs_to_intervals(maxwell_epochs)
    if concat_epochs_path.exists():
        concat_epochs = list(_read_json(concat_epochs_path))

    return _EpochInputs(
        preprocess_dir=preprocess_dir,
        maxwell_epochs_path=maxwell_epochs_path,
        concat_epochs_path=concat_epochs_path,
        maxwell_epochs=maxwell_epochs,
        maxwell_intervals=maxwell_intervals,
        concat_epochs=concat_epochs,
    )


__all__ = [
    "_EpochInputs",
    "_WaveformWindow",
    "_WaveformsRunContext",
    "_compute_waveforms_checkpoint_file",
    "_compute_waveforms_out_dir",
    "_initialize_run_context",
    "_load_epoch_markers",
    "_resolve_waveform_window",
]
