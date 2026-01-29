from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ..checkpointing import (
    ProcessingStage,
    compute_checkpoint_file,
    exception_to_error_dict,
    load_checkpoint,
    save_checkpoint,
)
from ..pipeline_logging import compute_pipeline_log_file, setup_pipeline_logger
from ..pipeline_driver import PREPROCESS_OUTPUTS_DIRNAME, _compute_mea_analysis_output_dir

from .curation import _run_mea_analysis_style_curation
from .curation import apply_mea_analysis_curation
from .metrics import (
    SegmentAnalyzerSource,
    load_and_compute_metrics,
    merge_quality_metrics,
    merge_template_metrics,
    recompute_merged_quality_metrics_from_deduplicated_spikes,
)
from .plotting import _write_waveforms_grid_pdf
from .reporting import _load_wf_rejection_log_unit_counts, _write_wf_rejection_log_xlsx
from .utils import (
    _epochs_to_intervals,
    _filter_spike_train_by_intervals,
    _infer_cutout_ms,
    _load_preprocessed_recording,
    _load_raw_segment_recording_full_channels,
    _load_sorting_from_sorter_output_dir,
    _maxwell_epochs_to_segment_local_intervals,
    _read_json,
    _resolve_mea_sorter_output_dir,
    _to_numpy_sorting,
    _write_json,
)


WAVEFORMS_OUTPUTS_DIRNAME = "waveforms_outputs"


def _compute_waveforms_checkpoint_file(*, well_out_dir: Path, h5_path: Path, stream_id: str) -> Path:
    """Use a dedicated checkpoint file for waveforms.

    The MEA_Analysis-style stage machine (PREPROCESSING..REPORTS_COMPLETE) doesn't
    include waveforms, so storing waveforms progress in the main checkpoint can
    accidentally *regress* stage numbers (e.g. REPORTS_COMPLETE -> ANALYZER_COMPLETE).
    """

    main_ckpt = compute_checkpoint_file(output_dir=well_out_dir, file_path=h5_path, stream_id=stream_id)
    name = main_ckpt.name
    if name.endswith("_checkpoint.json"):
        name = name[: -len("_checkpoint.json")] + "_waveforms_checkpoint.json"
    else:
        name = main_ckpt.stem + "_waveforms_checkpoint.json"
    return main_ckpt.with_name(name)


@dataclass(frozen=True)
class WaveformExtractInputs:
    h5_path: Path
    stream_id: str
    mea_output_root: Path
    sorter: str = "kilosort4"

    # Waveform window. If None, try to infer from trigger_pre/post.
    ms_before: Optional[float] = None
    ms_after: Optional[float] = None

    n_jobs: int = 8
    max_spikes_per_unit: Optional[int] = None

    # If True, also extract waveforms per concatenated segment.
    per_segment: bool = True

    # If True, per-segment waveforms are extracted only on channels that were
    # excluded during concatenation (i.e. not in the common-electrode intersection).
    # This avoids duplicating waveforms for the common channels already covered by
    # concat_waveforms.
    per_segment_only_additional_channels: bool = True

    # Resume/overwrite controls
    force_restart: bool = False

    # If True, drop spikes whose waveform window would cross Maxwell snippet boundaries.
    filter_by_maxwell_epochs: bool = True

    # Plotting
    plot_waveforms_grid_pdf: bool = True


@dataclass(frozen=True)
class WaveformExtractOutputs:
    well_out_dir: Path
    waveforms_out_dir: Path
    concat_waveforms_dir: Path
    segment_waveforms_dir: Optional[Path]
    params_json: Path
    filtering_json: Path
    waveforms_grid_pdf: Optional[Path]
    spikesorting_waveforms_grid_pdf: Optional[Path]


def _compute_waveforms_out_dir(*, output_root: Path, data_file: Path, well: str) -> Path:
    return _compute_mea_analysis_output_dir(output_root=output_root, data_file=data_file, well=well) / WAVEFORMS_OUTPUTS_DIRNAME


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


def _initialize_run_context(*, inputs: WaveformExtractInputs, logger_name_prefix: str) -> _WaveformsRunContext:
    well_out_dir = _compute_mea_analysis_output_dir(
        output_root=inputs.mea_output_root,
        data_file=inputs.h5_path,
        well=inputs.stream_id,
    )

    log_file = compute_pipeline_log_file(well_out_dir=well_out_dir, data_file=inputs.h5_path, stream_id=inputs.stream_id)
    logger = setup_pipeline_logger(
        log_file=log_file,
        logger_name=f"{logger_name_prefix}.{inputs.stream_id}",
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


def _resume_if_possible(*, inputs: WaveformExtractInputs, ctx: _WaveformsRunContext) -> Optional[WaveformExtractOutputs]:
    # Resume shortcut: trust existing artifacts if present.
    if not inputs.force_restart and ctx.concat_waveforms_dir.exists():
        ctx.logger.info("Resuming waveforms: existing outputs found at %s", ctx.concat_waveforms_dir)

        # Best-effort: populate optional PDF fields if they exist.
        waveforms_grid_pdf = ctx.waveforms_out_dir / "waveforms_grid_uncurated.pdf"
        waveforms_grid_pdf = waveforms_grid_pdf if waveforms_grid_pdf.exists() else None
        spikesorting_waveforms_grid_pdf = None

        return WaveformExtractOutputs(
            well_out_dir=ctx.well_out_dir,
            waveforms_out_dir=ctx.waveforms_out_dir,
            concat_waveforms_dir=ctx.concat_waveforms_dir,
            segment_waveforms_dir=ctx.segment_waveforms_dir,
            params_json=ctx.params_json,
            filtering_json=ctx.filtering_json,
            waveforms_grid_pdf=waveforms_grid_pdf,
            spikesorting_waveforms_grid_pdf=spikesorting_waveforms_grid_pdf,
        )
    return None


def _resolve_waveform_window(*, inputs: WaveformExtractInputs, fs_hz: float) -> _WaveformWindow:
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
    # Load epoch markers from preprocessing.
    preprocess_dir = well_out_dir / PREPROCESS_OUTPUTS_DIRNAME
    maxwell_epochs_path = preprocess_dir / f"maxwell_contiguous_epochs_{stream_id}.json"
    concat_epochs_path = preprocess_dir / f"concatenation_stitch_epochs_{stream_id}.json"

    maxwell_epochs: list[dict] = []
    maxwell_intervals: list[tuple[int, int]] = []
    concat_epochs: list[dict] = []

    if maxwell_epochs_path.exists():
        maxwell_epochs = list(_read_json(maxwell_epochs_path))
        # These are in concatenated sample coordinates.
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


def _init_filtering_summary(*, inputs: WaveformExtractInputs, window: _WaveformWindow, epochs: _EpochInputs) -> dict[str, Any]:
    return {
        "filter_by_maxwell_epochs": bool(inputs.filter_by_maxwell_epochs),
        "ms_before": float(window.ms_before),
        "ms_after": float(window.ms_after),
        "pre_samples": int(window.pre_samples),
        "post_samples": int(window.post_samples),
        "maxwell_epochs_path": str(epochs.maxwell_epochs_path) if epochs.maxwell_epochs_path.exists() else None,
        "concat_epochs_path": str(epochs.concat_epochs_path) if epochs.concat_epochs_path.exists() else None,
        "removed_spikes_total": 0,
        "kept_spikes_total": 0,
        # Concat-level breakdown:
        # - removed_by_maxwell_epoch_total: spikes outside contiguous Maxwell epochs
        # - removed_by_edge_total: spikes inside an epoch but too close to an epoch edge
        #   for the requested waveform window (pre/post)
        "removed_by_maxwell_epoch_total": 0,
        "removed_by_edge_total": 0,
        # Per-segment extraction filtering summary (filled in below when enabled).
        "per_segment": {
            "enabled": bool(inputs.per_segment),
            "removed_by_maxwell_epoch_total": 0,
            "removed_by_edge_total": 0,
            "kept_spikes_total": 0,
            "segments": [],
        },
    }


def _init_wf_rejection_log_fields(*, inputs: WaveformExtractInputs, window: _WaveformWindow) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    # Per-spike rejection rows for auditability / downstream alignment.
    # Written to waveforms_outputs/wf_rejection_log.xlsx at the end of the stage.
    wf_rejection_rows: list[dict[str, Any]] = []
    base_rej_fields: dict[str, Any] = {
        "stream_id": inputs.stream_id,
        "sorter": inputs.sorter,
        "h5_path": str(inputs.h5_path),
        "fs_hz": float(window.fs_hz),
        "ms_before": float(window.ms_before),
        "ms_after": float(window.ms_after),
        "pre_samples": int(window.pre_samples),
        "post_samples": int(window.post_samples),
    }
    return wf_rejection_rows, base_rej_fields


def _filter_sorting_by_maxwell_epochs(
    *,
    inputs: WaveformExtractInputs,
    sorting: Any,
    epochs: _EpochInputs,
    window: _WaveformWindow,
    logger: Any,
    filtering_summary: dict[str, Any],
    wf_rejection_rows: list[dict[str, Any]],
    base_rej_fields: dict[str, Any],
) -> Any:
    # Filter spikes by Maxwell epochs (avoid snippet boundary crossings).
    filtered_sorting = sorting
    if inputs.filter_by_maxwell_epochs and epochs.maxwell_intervals:
        unit_trains: dict[int, list[int]] = {}
        removed_total = 0
        removed_outside_total = 0
        removed_edge_total = 0
        kept_total = 0

        unit_ids = list(sorting.get_unit_ids())
        for u in unit_ids:
            st = sorting.get_unit_spike_train(u)
            st_list = [int(x) for x in st]
            st_list.sort()

            kept, removed_outside, removed_edge, removed_outside_spikes, removed_edge_spikes = (
                _filter_spike_train_by_intervals(
                spike_train=st_list,
                intervals=epochs.maxwell_intervals,
                pre_samples=window.pre_samples,
                post_samples=window.post_samples,
                )
            )
            unit_trains[int(u)] = kept
            removed_outside_total += int(removed_outside)
            removed_edge_total += int(removed_edge)
            removed_total += int(removed_outside) + int(removed_edge)
            kept_total += len(kept)

            # Record exact removed spikes for joinability.
            for t in removed_outside_spikes:
                wf_rejection_rows.append(
                    {
                        **base_rej_fields,
                        "scope": "concat",
                        "source_name": "concat",
                        "segment_index": None,
                        "rec_name": None,
                        "unit_id": int(u),
                        "spike_sample_local": None,
                        "spike_sample_concat": int(t),
                        "spike_time_s": float(t) / float(window.fs_hz),
                        "reason": "outside_maxwell_epoch",
                    }
                )
            for t in removed_edge_spikes:
                wf_rejection_rows.append(
                    {
                        **base_rej_fields,
                        "scope": "concat",
                        "source_name": "concat",
                        "segment_index": None,
                        "rec_name": None,
                        "unit_id": int(u),
                        "spike_sample_local": None,
                        "spike_sample_concat": int(t),
                        "spike_time_s": float(t) / float(window.fs_hz),
                        "reason": "waveform_window_crosses_epoch_edge",
                    }
                )

        filtered_sorting = _to_numpy_sorting(unit_trains=unit_trains, fs_hz=window.fs_hz)
        filtering_summary["removed_spikes_total"] = int(removed_total)
        filtering_summary["kept_spikes_total"] = int(kept_total)
        filtering_summary["removed_by_maxwell_epoch_total"] = int(removed_outside_total)
        filtering_summary["removed_by_edge_total"] = int(removed_edge_total)
        logger.info(
            "Filtered spikes by Maxwell epochs: kept=%d removed=%d (pre=%d post=%d samples)",
            kept_total,
            removed_total,
            window.pre_samples,
            window.post_samples,
        )
    else:
        # Best-effort counts.
        try:
            n_total = sum(len(sorting.get_unit_spike_train(u)) for u in sorting.get_unit_ids())
            filtering_summary["kept_spikes_total"] = int(n_total)
        except Exception:
            pass

    return filtered_sorting


def _write_waveform_extraction_params(
    *,
    params_json: Path,
    inputs: WaveformExtractInputs,
    sorter_output_dir: Path,
    window: _WaveformWindow,
) -> None:
    _write_json(params_json, {
        "h5_path": str(inputs.h5_path),
        "stream_id": inputs.stream_id,
        "sorter": inputs.sorter,
        "sorter_output_dir": str(sorter_output_dir),
        "ms_before": float(window.ms_before),
        "ms_after": float(window.ms_after),
        "n_jobs": int(inputs.n_jobs),
        "max_spikes_per_unit": inputs.max_spikes_per_unit,
        "per_segment": bool(inputs.per_segment),
        "per_segment_recording_source": "raw_maxwell_full_channels" if inputs.per_segment else None,
        "per_segment_only_additional_channels": bool(inputs.per_segment_only_additional_channels),
    })


def _extract_concat_waveforms(
    *,
    inputs: WaveformExtractInputs,
    filtered_sorting: Any,
    recording: Any,
    concat_waveforms_dir: Path,
    window: _WaveformWindow,
    quality_metrics_params: dict[str, Any],
    logger: Any,
) -> None:
    # Extract concatenated waveforms.
    import shutil
    import spikeinterface.full as si  # type: ignore[import-not-found]

    if concat_waveforms_dir.exists() and inputs.force_restart:
        shutil.rmtree(concat_waveforms_dir)

    logger.info("Extracting concat waveforms -> %s", concat_waveforms_dir)
    concat_analyzer = si.create_sorting_analyzer(
        filtered_sorting,
        recording,
        format="binary_folder",
        folder=concat_waveforms_dir,
        return_in_uV=True,
    )
    concat_analyzer.compute(
        ["random_spikes", "waveforms"],
        extension_params={
            "random_spikes": {
                "method": "uniform",
                "max_spikes_per_unit": int(inputs.max_spikes_per_unit),
                "seed": 0,
            },
            "waveforms": {"ms_before": float(window.ms_before), "ms_after": float(window.ms_after)},
        },
        verbose=False,
        n_jobs=int(inputs.n_jobs),
    )

    # Compute metrics at extraction time so downstream steps can simply load them
    # as analyzer extensions (no re-computation).
    concat_analyzer.compute(
        [
            "spike_amplitudes",
            "templates",
            "noise_levels",
            "quality_metrics",
            "template_metrics",
            "unit_locations",
        ],
        extension_params={
            "unit_locations": {"method": "monopolar_triangulation"},
            "quality_metrics": dict(quality_metrics_params),
        },
        verbose=False,
        n_jobs=int(inputs.n_jobs),
    )


@dataclass(frozen=True)
class _SegmentSpec:
    segment_index: int
    rec_name: str
    start_sample_concat: int
    end_sample_concat: int
    seg_dir: Path


def _parse_concat_epoch_segment(*, seg: dict[str, Any], segment_waveforms_dir: Path) -> Optional[_SegmentSpec]:
    try:
        seg_index = int(seg["segment_index"])
        rec_name = str(seg.get("rec_name", f"seg{seg_index}"))
        start = int(seg["start_sample"])
        end = int(seg["end_sample"])
    except Exception:
        return None

    seg_dir = segment_waveforms_dir / f"seg{seg_index:02d}_{rec_name}"
    return _SegmentSpec(
        segment_index=int(seg_index),
        rec_name=str(rec_name),
        start_sample_concat=int(start),
        end_sample_concat=int(end),
        seg_dir=seg_dir,
    )


def _count_spikes_in_concat_window(*, sorting: Any, start: int, end: int) -> int:
    """Count spikes in concat time in [start, end)."""
    total = 0
    for u in sorting.get_unit_ids():
        st = sorting.get_unit_spike_train(u)
        # Avoid materializing huge lists; iterate.
        for t in st:
            tt = int(t)
            if start <= tt < end:
                total += 1
    return int(total)


def _compute_segment_maxwell_intervals(
    *,
    inputs: WaveformExtractInputs,
    epochs: _EpochInputs,
    segment_index: int,
    seg_len: int,
    logger: Any,
    rec_name: str,
) -> list[tuple[int, int]]:
    # Compute Maxwell contiguous-epoch intervals in *segment-local* coordinates.
    # This ensures we don't extract waveforms whose windows cross snippet gaps
    # within this segment.
    seg_maxwell_intervals: list[tuple[int, int]] = []
    if inputs.filter_by_maxwell_epochs and epochs.maxwell_epochs:
        seg_maxwell_intervals = _maxwell_epochs_to_segment_local_intervals(
            maxwell_epochs=epochs.maxwell_epochs,
            segment_index=segment_index,
        )

    # Sanity check: segment-local intervals should lie within [0, seg_len].
    # This is a cheap guardrail against accidentally mixing concat/global and segment-local coordinates.
    for a, b in seg_maxwell_intervals[:10]:
        if a < 0 or b > int(seg_len):
            logger.warning(
                "Segment %s: segment-local Maxwell interval out of bounds: (%d, %d) with seg_len=%d",
                rec_name,
                int(a),
                int(b),
                int(seg_len),
            )
            break

    return seg_maxwell_intervals


def _append_segment_skip_summary(
    *,
    filtering_summary: dict[str, Any],
    seg_index: int,
    rec_name: str,
    raw_channels_total: Optional[int],
    excluded_common_channels_total: Optional[int],
    kept_additional_channels_total: Optional[int],
    skipped_reason: str,
) -> None:
    filtering_summary["per_segment"]["segments"].append(
        {
            "segment_index": int(seg_index),
            "rec_name": str(rec_name),
            "spikes_in_segment_total": 0,
            "removed_by_maxwell_epoch": 0,
            "removed_by_edge": 0,
            "kept_spikes_total": 0,
            "maxwell_intervals_in_segment": None,
            "raw_channels_total": raw_channels_total,
            "excluded_common_channels_total": excluded_common_channels_total,
            "kept_additional_channels_total": kept_additional_channels_total,
            "skipped_reason": str(skipped_reason),
        }
    )


def _append_segment_summary(
    *,
    filtering_summary: dict[str, Any],
    seg_index: int,
    rec_name: str,
    seg_spikes_total: int,
    seg_removed_epoch_total: int,
    seg_removed_edge_total: int,
    seg_kept_total: int,
    seg_maxwell_intervals_len: int,
    raw_channels_total: Optional[int],
    excluded_common_channels_total: Optional[int],
    kept_additional_channels_total: Optional[int],
) -> None:
    filtering_summary["per_segment"]["removed_by_maxwell_epoch_total"] += int(seg_removed_epoch_total)
    filtering_summary["per_segment"]["removed_by_edge_total"] += int(seg_removed_edge_total)
    filtering_summary["per_segment"]["kept_spikes_total"] += int(seg_kept_total)
    filtering_summary["per_segment"]["segments"].append(
        {
            "segment_index": int(seg_index),
            "rec_name": str(rec_name),
            "spikes_in_segment_total": int(seg_spikes_total),
            "removed_by_maxwell_epoch": int(seg_removed_epoch_total),
            "removed_by_edge": int(seg_removed_edge_total),
            "kept_spikes_total": int(seg_kept_total),
            "maxwell_intervals_in_segment": int(seg_maxwell_intervals_len),
            "raw_channels_total": raw_channels_total,
            "excluded_common_channels_total": excluded_common_channels_total,
            "kept_additional_channels_total": kept_additional_channels_total,
        }
    )


def _extract_per_segment_waveforms(
    *,
    inputs: WaveformExtractInputs,
    recording: Any,
    sorting_unfiltered: Any,
    filtered_sorting: Any,
    epochs: _EpochInputs,
    window: _WaveformWindow,
    segment_waveforms_dir: Path,
    common_channel_ids: set[int],
    filtering_summary: dict[str, Any],
    wf_rejection_rows: list[dict[str, Any]],
    base_rej_fields: dict[str, Any],
    quality_metrics_params: dict[str, Any],
    logger: Any,
) -> None:
    # Optional: per-segment extraction using concatenation epochs.
    if inputs.per_segment and epochs.concat_epochs:
        segment_waveforms_dir.mkdir(parents=True, exist_ok=True)

        import shutil
        import spikeinterface.full as si  # type: ignore[import-not-found]

        logger.info("Extracting per-segment waveforms -> %s", segment_waveforms_dir)
        for seg in epochs.concat_epochs:
            spec = _parse_concat_epoch_segment(seg=seg, segment_waveforms_dir=segment_waveforms_dir)
            if spec is None:
                continue

            seg_index = int(spec.segment_index)
            rec_name = str(spec.rec_name)
            start = int(spec.start_sample_concat)
            end = int(spec.end_sample_concat)
            seg_dir = spec.seg_dir

            if seg_dir.exists() and inputs.force_restart:
                shutil.rmtree(seg_dir)

            # STEP 1) Load the *raw* segment recording with its full channel set.
            #
            # This is the point of the per-segment extraction: the concatenated
            # recording (used for sorting) only contains electrodes shared across
            # all segments. Any electrodes present in *some* segments but not all
            # are intentionally dropped during concatenation.
            #
            # By extracting waveforms on the raw segment, we can recover waveforms
            # on those dropped electrodes for this segment.
            seg_rec = _load_raw_segment_recording_full_channels(
                h5_path=inputs.h5_path,
                stream_id=inputs.stream_id,
                rec_name=rec_name,
                center_chunk_size=10_000,
            )

            # Optional: avoid redundant waveforms.
            # If enabled, remove the concat/common electrodes from the raw segment
            # so the per-segment waveforms contain only the *additional* channels.
            raw_channels_total: Optional[int]
            excluded_common_channels_total: Optional[int]
            kept_additional_channels_total: Optional[int]
            raw_channels_total = None
            excluded_common_channels_total = None
            kept_additional_channels_total = None

            try:
                raw_channels_total = int(seg_rec.get_num_channels())
            except Exception:
                pass

            if inputs.per_segment_only_additional_channels and common_channel_ids:
                try:
                    import numpy as np  # type: ignore[import-not-found]

                    cv = seg_rec.get_property("contact_vector")
                    electrodes = np.asarray(cv["electrode"], dtype=int)
                    ch_ids = list(seg_rec.get_channel_ids())
                    if len(electrodes) == len(ch_ids):
                        keep_mask = [int(e) not in common_channel_ids for e in electrodes]
                        keep_channel_ids = [ch_ids[i] for i, keep in enumerate(keep_mask) if keep]
                        kept_electrodes = [int(electrodes[i]) for i, keep in enumerate(keep_mask) if keep]
                        seg_rec = seg_rec.select_channels(keep_channel_ids)
                        excluded_common_channels_total = int(len(ch_ids) - len(keep_channel_ids))
                        kept_additional_channels_total = int(len(keep_channel_ids))

                        # Ensure channel ids are electrode ids for stable identity.
                        # (Helpful if earlier rename failed.)
                        try:
                            if len(kept_electrodes) == int(seg_rec.get_num_channels()):
                                if int(np.unique(np.asarray(kept_electrodes)).size) == int(len(kept_electrodes)):
                                    seg_rec = seg_rec.rename_channels([int(e) for e in kept_electrodes])
                        except Exception:
                            pass
                except Exception:
                    # Best-effort: if we cannot compute the additional-channel set,
                    # keep full channels rather than failing.
                    pass

            if inputs.per_segment_only_additional_channels:
                try:
                    if int(seg_rec.get_num_channels()) == 0:
                        logger.info(
                            "Segment %s: no additional channels after excluding common set; skipping per-segment waveforms.",
                            rec_name,
                        )
                        _append_segment_skip_summary(
                            filtering_summary=filtering_summary,
                            seg_index=seg_index,
                            rec_name=rec_name,
                            raw_channels_total=raw_channels_total,
                            excluded_common_channels_total=excluded_common_channels_total,
                            kept_additional_channels_total=kept_additional_channels_total,
                            skipped_reason="no_additional_channels",
                        )
                        continue
                except Exception:
                    pass

            # Sanity: concat_epochs were computed from the preprocessed segments.
            # Channel slicing does not change sample count, so this should match.
            seg_len_expected = int(end - start)
            try:
                seg_len = int(seg_rec.get_num_samples())
                if seg_len != seg_len_expected:
                    logger.warning(
                        "Segment length mismatch for %s: raw=%d expected=%d (start=%d end=%d). Proceeding.",
                        rec_name,
                        seg_len,
                        seg_len_expected,
                        start,
                        end,
                    )
            except Exception:
                seg_len = seg_len_expected

            try:
                if inputs.per_segment_only_additional_channels:
                    logger.info(
                        "Segment %s: raw channels=%s, kept additional=%s (excluded common=%s, concat/common=%d)",
                        rec_name,
                        raw_channels_total,
                        kept_additional_channels_total,
                        excluded_common_channels_total,
                        int(recording.get_num_channels()),
                    )
                else:
                    logger.info(
                        "Segment %s: raw channels=%d (concat/common=%d)",
                        rec_name,
                        int(seg_rec.get_num_channels()),
                        int(recording.get_num_channels()),
                    )
            except Exception:
                pass

            # IMPORTANT CONCEPTUAL NOTE:
            #
            # This per-segment waveforms step does *not* discover new spikes.
            # It reuses spike times from the concatenated spikesorting result and
            # extracts waveforms for those spike times on the raw segment recording.
            #
            # The main purpose is to recover waveform snippets on segment-specific
            # electrodes that were dropped during concatenation (i.e. channels that
            # are not in the common-electrode intersection).
            #
            # If you want "new spikes" that only appear on additional channels,
            # that requires an additional detection/sorting/template-matching step.

            # For debugging/sanity: count spikes in this segment *before* and *after*
            # concat-level filtering.
            try:
                in_seg_unfiltered = _count_spikes_in_concat_window(sorting=sorting_unfiltered, start=start, end=end)
                in_seg_filtered = _count_spikes_in_concat_window(sorting=filtered_sorting, start=start, end=end)
                logger.info(
                    "Segment %s: spikes in concat window (unfiltered=%d, after-concat-filter=%d, excluded-at-concat=%d)",
                    rec_name,
                    int(in_seg_unfiltered),
                    int(in_seg_filtered),
                    int(in_seg_unfiltered - in_seg_filtered),
                )
            except Exception:
                pass

            seg_maxwell_intervals = _compute_segment_maxwell_intervals(
                inputs=inputs,
                epochs=epochs,
                segment_index=seg_index,
                seg_len=int(seg_len),
                logger=logger,
                rec_name=rec_name,
            )

            # STEP 2) Build a segment-local sorting from the *concatenated* sorting.
            #
            # Conceptually:
            # - `filtered_sorting` spike times are in the concatenated time base.
            # - `seg_rec` is a single raw recording segment with its own 0..N time base.
            # So we:
            #   - select spikes that fall inside this concatenated segment window [start, end)
            #   - shift them to segment-local coordinates by subtracting `start`
            #   - drop spikes too close to segment edges so waveform windows fit.
            unit_trains_seg: dict[int, list[int]] = {}

            seg_spikes_total = 0
            seg_removed_epoch_total = 0
            seg_removed_edge_total = 0
            seg_kept_total = 0

            for u in filtered_sorting.get_unit_ids():
                st = filtered_sorting.get_unit_spike_train(u)
                st_list = [int(x) for x in st]

                # Keep only spikes in the segment, shift to segment-local coordinates.
                # Also drop spikes too close to segment edges for waveform windows.
                local_all = [int(t - start) for t in st_list if start <= t < end]
                local_all.sort()
                seg_spikes_total += len(local_all)

                # IMPORTANT BEHAVIOR CHANGE:
                #
                # We *do not* remove spikes from the segment-local sorting based on
                # Maxwell-epoch filtering. Instead, we extract waveforms for all
                # spikes that are in this segment window and are "edge-safe" for
                # waveform extraction.
                #
                # Any spikes that fall outside Maxwell contiguous epochs (or whose
                # waveform windows cross epoch edges) are recorded in the rejection
                # log / exclusions file so downstream steps can ignore them.

                # Extra safety: enforce segment-edge constraints so waveform windows fit.
                # This prevents errors in waveform extraction near segment boundaries.
                kept_edges: list[int] = []
                for t_local in local_all:
                    if t_local - window.pre_samples < 0:
                        continue
                    if t_local + window.post_samples >= int(seg_len):
                        continue
                    kept_edges.append(int(t_local))

                # NOTE:
                # We deliberately do *not* filter the spike train by Maxwell intervals
                # here. Maxwell-epoch boundary conditions are handled *after* waveform
                # extraction by excluding the affected extracted waveforms.

                # Log spikes rejected due to segment-edge constraints.
                try:
                    source_name = f"seg{int(seg_index):02d}_{rec_name}"
                    kept_edge_set = set(int(x) for x in kept_edges)
                    for t_local in local_all:
                        if int(t_local) in kept_edge_set:
                            continue
                        if (int(t_local) - int(window.pre_samples) < 0) or (
                            int(t_local) + int(window.post_samples) >= int(seg_len)
                        ):
                            t_concat = int(t_local) + int(start)
                            wf_rejection_rows.append(
                                {
                                    **base_rej_fields,
                                    "scope": "segment",
                                    "source_name": str(source_name),
                                    "segment_index": int(seg_index),
                                    "rec_name": str(rec_name),
                                    "unit_id": int(u),
                                    "spike_sample_local": int(t_local),
                                    "spike_sample_concat": int(t_concat),
                                    "spike_time_s": float(t_concat) / float(window.fs_hz),
                                    "reason": "waveform_window_outside_segment_bounds",
                                }
                            )
                except Exception:
                    pass

                removed_edge = int(len(local_all) - len(kept_edges))
                seg_removed_edge_total += int(removed_edge)
                seg_kept_total += int(len(kept_edges))

                unit_trains_seg[int(u)] = kept_edges

            # Debug/sanity: how many spikes are passed into the segment-local sorting.
            # (Waveforms are computed on a random subset of these via `random_spikes`.)
            try:
                logger.info(
                    "Segment %s: per-segment spike train prepared=%d (in_window=%d, excluded_by_segment_edge=%d)",
                    rec_name,
                    int(seg_kept_total),
                    int(seg_spikes_total),
                    int(seg_removed_edge_total),
                )
            except Exception:
                pass

            # Convert the segment-local spike trains into a SortingExtractor.
            # This is the bridge that lets us use the holistic concat sorting
            # to drive waveform extraction on each raw segment.
            seg_sort = _to_numpy_sorting(unit_trains=unit_trains_seg, fs_hz=window.fs_hz)

            # STEP 3) Register the sorting to the raw segment recording.
            #
            # Not strictly required for `si.create_sorting_analyzer(sorting=..., recording=...)`,
            # but this matches the intent/semantics of the legacy pipeline and helps
            # downstream utilities that expect the sorting to “know” its recording.
            try:
                seg_sort.register_recording(seg_rec)
            except Exception:
                pass

            # STEP 4) Extract waveforms on the raw segment.
            #
            # Key outcome: these waveforms include channels that are absent from the
            # concatenated recording (i.e. the non-shared electrodes), which is the
            # whole reason we do this per-segment pass.
            seg_analyzer = si.create_sorting_analyzer(
                seg_sort,
                seg_rec,
                format="binary_folder",
                folder=seg_dir,
                return_in_uV=True,
            )
            seg_analyzer.compute(
                ["random_spikes", "waveforms"],
                extension_params={
                    "random_spikes": {
                        "method": "uniform",
                        "max_spikes_per_unit": int(inputs.max_spikes_per_unit),
                        "seed": 0,
                    },
                    "waveforms": {"ms_before": float(window.ms_before), "ms_after": float(window.ms_after)},
                },
                verbose=False,
                n_jobs=max(1, int(inputs.n_jobs)),
            )

            # Compute metrics at extraction time so downstream steps can load the
            # analyzer extensions directly (no re-computation).
            seg_analyzer.compute(
                [
                    "spike_amplitudes",
                    "templates",
                    "noise_levels",
                    "quality_metrics",
                    "template_metrics",
                    "unit_locations",
                ],
                extension_params={
                    "unit_locations": {"method": "monopolar_triangulation"},
                    "quality_metrics": dict(quality_metrics_params),
                },
                verbose=False,
                n_jobs=max(1, int(inputs.n_jobs)),
            )

            # STEP 5) Exclude extracted waveforms based on boundary conditions.
            #
            # IMPORTANT: We do not prune the spike train by Maxwell intervals before
            # extraction. Instead, we mark the extracted waveforms that violate
            # Maxwell epoch constraints so downstream steps can drop them.
            if inputs.filter_by_maxwell_epochs and seg_maxwell_intervals:
                try:
                    from .exclusions import _get_random_spike_samples  # type: ignore

                    source_name = f"seg{int(seg_index):02d}_{rec_name}"
                    removed_outside_total = 0
                    removed_edge_epoch_total = 0
                    selected_total = 0

                    intervals = list(seg_maxwell_intervals) if seg_maxwell_intervals else []

                    def _classify_maxwell_exclusion(t_local: int) -> Optional[str]:
                        if not intervals:
                            return None

                        # Find the interval containing the spike center.
                        containing: Optional[tuple[int, int]] = None
                        for a, b in intervals:
                            if int(a) <= int(t_local) < int(b):
                                containing = (int(a), int(b))
                                break

                        if containing is None:
                            return "outside_maxwell_epoch"

                        a, b = containing
                        left_bound = int(t_local) - int(window.pre_samples)
                        right_bound = int(t_local) + int(window.post_samples)
                        if left_bound < int(a) or right_bound >= int(b):
                            return "waveform_window_crosses_epoch_edge"
                        return None

                    for u in seg_sort.get_unit_ids():
                        samples = _get_random_spike_samples(analyzer=seg_analyzer, unit_id=u)
                        if samples is None:
                            continue
                        try:
                            samples_iter = list(samples)
                        except Exception:
                            continue

                        selected_total += int(len(samples_iter))
                        for t_local in samples_iter:
                            try:
                                t_local_i = int(t_local)
                            except Exception:
                                continue

                            reason = _classify_maxwell_exclusion(t_local_i)
                            if reason is None:
                                continue

                            if reason == "outside_maxwell_epoch":
                                removed_outside_total += 1
                            elif reason == "waveform_window_crosses_epoch_edge":
                                removed_edge_epoch_total += 1

                            t_concat = int(t_local_i) + int(start)
                            wf_rejection_rows.append(
                                {
                                    **base_rej_fields,
                                    "scope": "segment",
                                    "source_name": str(source_name),
                                    "segment_index": int(seg_index),
                                    "rec_name": str(rec_name),
                                    "unit_id": int(u),
                                    "spike_sample_local": int(t_local_i),
                                    "spike_sample_concat": int(t_concat),
                                    "spike_time_s": float(t_concat) / float(window.fs_hz),
                                    "reason": str(reason),
                                }
                            )

                    seg_removed_epoch_total = int(removed_outside_total + removed_edge_epoch_total)
                    logger.info(
                        "Segment %s: flagged extracted waveforms by maxwell epochs=%d (outside=%d edge=%d, selected_random_spikes=%d)",
                        rec_name,
                        int(seg_removed_epoch_total),
                        int(removed_outside_total),
                        int(removed_edge_epoch_total),
                        int(selected_total),
                    )
                except Exception:
                    # Best-effort: exclusions are an enhancement but should not fail extraction.
                    seg_removed_epoch_total = 0
            else:
                seg_removed_epoch_total = 0

            # Accumulate per-segment filtering stats.
            try:
                _append_segment_summary(
                    filtering_summary=filtering_summary,
                    seg_index=seg_index,
                    rec_name=rec_name,
                    seg_spikes_total=int(seg_spikes_total),
                    seg_removed_epoch_total=int(seg_removed_epoch_total),
                    seg_removed_edge_total=int(seg_removed_edge_total),
                    seg_kept_total=int(seg_kept_total),
                    seg_maxwell_intervals_len=int(len(seg_maxwell_intervals)),
                    raw_channels_total=raw_channels_total,
                    excluded_common_channels_total=excluded_common_channels_total,
                    kept_additional_channels_total=kept_additional_channels_total,
                )
            except Exception:
                pass


def _persist_filtering_and_exclusions(
    *,
    filtering_json: Path,
    waveforms_out_dir: Path,
    wf_rejection_rows: list[dict[str, Any]],
    filtering_summary: dict[str, Any],
    inputs: WaveformExtractInputs,
    logger: Any,
) -> None:
    # Persist filtering summary after all filtering passes have contributed their counts
    # (concat-level filtering + optional per-segment filtering).
    _write_json(filtering_json, filtering_summary)

    # Persist per-spike rejection log (best effort; safe to skip if pandas engine is missing).
    try:
        wf_rejection_log_xlsx = waveforms_out_dir / "wf_rejection_log.xlsx"
        _write_wf_rejection_log_xlsx(
            wf_rejection_log_xlsx=wf_rejection_log_xlsx,
            rows=wf_rejection_rows,
            force_restart=bool(inputs.force_restart),
            logger=logger,
        )
    except Exception as e:
        logger.warning("Failed to write wf_rejection_log.xlsx: %s", e)

    # Persist compact per-spike exclusions for fast downstream filtering.
    # This avoids having to load large Excel sheets in footprinting/templates.
    try:
        from .exclusions import WF_EXCLUSIONS_NPZ_NAME, write_wf_exclusions_npz

        write_wf_exclusions_npz(
            wf_exclusions_npz=waveforms_out_dir / WF_EXCLUSIONS_NPZ_NAME,
            rows=wf_rejection_rows,
            force_restart=bool(inputs.force_restart),
            logger=logger,
        )
    except Exception as e:
        logger.warning("Failed to write wf_exclusions.npz: %s", e)


def _plot_and_curate_if_requested(
    *,
    inputs: WaveformExtractInputs,
    waveforms_out_dir: Path,
    concat_waveforms_dir: Path,
    recording: Any,
    sorting: Any,
    window: _WaveformWindow,
    logger: Any,
) -> tuple[Optional[Path], Optional[Path]]:
    raise RuntimeError("_plot_and_curate_if_requested is deprecated; use _curate_then_plot")


def _compute_and_merge_waveforms_metrics(
    *,
    inputs: WaveformExtractInputs,
    waveforms_out_dir: Path,
    concat_waveforms_dir: Path,
    segment_waveforms_dir: Optional[Path],
    epochs: _EpochInputs,
    window: _WaveformWindow,
    logger: Any,
) -> tuple[Any, Any]:
    """Waveforms step 1/3: compute per-source metrics and merge."""

    # 1) Compute per-source metrics (concat + per segment) using the already-built analyzers.
    try:
        import pandas as pd  # type: ignore[import-not-found]
    except Exception:
        pd = None  # type: ignore

    metrics_sources_dir = waveforms_out_dir / "metrics_sources"
    metrics_sources_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Waveforms step 1/3: compute + merge metrics")

    concat_qm, concat_tm = load_and_compute_metrics(
        analyzer_dir=concat_waveforms_dir,
        ms_before=float(window.ms_before),
        ms_after=float(window.ms_after),
        n_jobs=int(inputs.n_jobs),
        logger=logger,
    )

    # Save concat metrics.
    try:
        concat_dir = metrics_sources_dir / "concat"
        concat_dir.mkdir(parents=True, exist_ok=True)
        concat_qm.to_excel(concat_dir / "qm_unfiltered.xlsx")
        concat_tm.to_excel(concat_dir / "tm_unfiltered.xlsx")

        # Apply curation for concat source.
        clean_concat, rej_concat = apply_mea_analysis_curation(q_metrics=concat_qm, user_thresholds=None)
        clean_concat.to_excel(concat_dir / "metrics_curated.xlsx")
        rej_concat.to_excel(concat_dir / "rejection_log.xlsx")
        try:
            concat_tm.loc[list(clean_concat.index.values)].to_excel(concat_dir / "tm_curated.xlsx")
        except Exception:
            pass
    except Exception as e:
        logger.warning("Failed to save concat metrics xlsx: %s", e)

    seg_qm_by_source: dict[str, Any] = {}
    seg_tm_by_source: dict[str, Any] = {}

    if inputs.per_segment and epochs.concat_epochs and segment_waveforms_dir is not None:
        for seg in epochs.concat_epochs:
            spec = _parse_concat_epoch_segment(seg=seg, segment_waveforms_dir=segment_waveforms_dir)
            if spec is None:
                continue
            if not spec.seg_dir.exists():
                continue

            src = f"seg{int(spec.segment_index):02d}_{spec.rec_name}"
            try:
                qm, tm = load_and_compute_metrics(
                    analyzer_dir=spec.seg_dir,
                    ms_before=float(window.ms_before),
                    ms_after=float(window.ms_after),
                    n_jobs=int(inputs.n_jobs),
                    logger=logger,
                )
                seg_qm_by_source[src] = qm
                seg_tm_by_source[src] = tm

                out_dir = metrics_sources_dir / src
                out_dir.mkdir(parents=True, exist_ok=True)
                qm.to_excel(out_dir / "qm_unfiltered.xlsx")
                tm.to_excel(out_dir / "tm_unfiltered.xlsx")
            except Exception as e:
                logger.warning("Failed metrics for %s: %s", src, e)

    # 2) Merge metrics across contexts.
    # Start with a general-purpose merge so we keep *all* metric columns,
    # then overwrite key curation-driving columns with a recomputation from
    # a deduplicated spike+amplitude representation.
    merged_qm = merge_quality_metrics(concat_qm=concat_qm, segment_qm_by_source=seg_qm_by_source)
    merged_tm = merge_template_metrics(concat_tm=concat_tm, segment_tm_by_source=seg_tm_by_source)

    # Recompute key quality metrics from a deduplicated train.
    try:
        segment_sources: list[SegmentAnalyzerSource] = []
        if inputs.per_segment and epochs.concat_epochs and segment_waveforms_dir is not None:
            for seg in epochs.concat_epochs:
                spec = _parse_concat_epoch_segment(seg=seg, segment_waveforms_dir=segment_waveforms_dir)
                if spec is None:
                    continue
                if not spec.seg_dir.exists():
                    continue
                segment_sources.append(
                    SegmentAnalyzerSource(
                        source_name=f"seg{int(spec.segment_index):02d}_{spec.rec_name}",
                        analyzer_dir=spec.seg_dir,
                        segment_index=int(spec.segment_index),
                        start_sample_concat=int(spec.start_sample_concat),
                        end_sample_concat=int(spec.end_sample_concat),
                    )
                )

        recomputed = recompute_merged_quality_metrics_from_deduplicated_spikes(
            concat_analyzer_dir=concat_waveforms_dir,
            segment_sources=segment_sources,
            logger=logger,
        )

        # Ensure union index so recomputed-only units are not dropped.
        try:
            union_index = sorted(set(merged_qm.index.values) | set(recomputed.index.values))
            merged_qm = merged_qm.reindex(union_index)
        except Exception:
            pass

        # Overwrite only for units actually present in recomputed.
        for col in recomputed.columns:
            if col not in merged_qm.columns:
                merged_qm[col] = None
            merged_qm.loc[list(recomputed.index.values), col] = recomputed[col]
    except Exception as e:
        logger.warning("Merged-metric recomputation failed; using merge-only metrics: %s", e)

    try:
        merged_dir = metrics_sources_dir / "merged"
        merged_dir.mkdir(parents=True, exist_ok=True)
        merged_qm.to_excel(merged_dir / "qm_merged.xlsx")
        merged_tm.to_excel(merged_dir / "tm_merged.xlsx")
    except Exception as e:
        logger.warning("Failed to save merged metrics xlsx: %s", e)

    # Also keep the historical root-level filenames, but now they represent the merged metrics.
    try:
        merged_qm.to_excel(waveforms_out_dir / "qm_merged_unfiltered.xlsx")
        merged_tm.to_excel(waveforms_out_dir / "tm_merged_unfiltered.xlsx")
    except Exception as e:
        logger.warning("Failed to save root metrics xlsx: %s", e)

    return merged_qm, merged_tm


def _apply_waveforms_curation(
    *,
    waveforms_out_dir: Path,
    merged_qm: Any,
    merged_tm: Any,
    logger: Any,
) -> Optional[list[Any]]:
    """Waveforms step 2/3: apply curation logic to merged metrics."""

    logger.info("Waveforms step 2/3: apply curation")
    curated_units_for_plot: Optional[list[Any]] = None
    try:
        clean_metrics, rejection_log = apply_mea_analysis_curation(q_metrics=merged_qm, user_thresholds=None)
        curated_units_for_plot = list(clean_metrics.index.values)
        clean_metrics.to_excel(waveforms_out_dir / "metrics_curated.xlsx")
        rejection_log.to_excel(waveforms_out_dir / "rejection_log.xlsx")
        try:
            merged_tm.loc[curated_units_for_plot].to_excel(waveforms_out_dir / "tm_curated.xlsx")
        except Exception:
            pass
    except Exception as e:
        logger.warning("Curation failed; skipping curated artifacts: %s", e)
    return curated_units_for_plot


def _plot_waveforms_outputs(
    *,
    inputs: WaveformExtractInputs,
    waveforms_out_dir: Path,
    concat_waveforms_dir: Path,
    segment_waveforms_dir: Optional[Path],
    epochs: _EpochInputs,
    curated_units_for_plot: Optional[list[Any]],
    logger: Any,
) -> tuple[Optional[Path], Optional[Path]]:
    """Waveforms step 3/3: plot uncurated + curated waveforms PDFs."""

    logger.info("Waveforms step 3/3: plot")
    waveforms_grid_pdf: Optional[Path] = None
    spikesorting_waveforms_grid_pdf: Optional[Path] = None

    if inputs.plot_waveforms_grid_pdf:
        segment_folders: Optional[list[Path]] = None
        try:
            if inputs.per_segment and epochs.concat_epochs and segment_waveforms_dir is not None:
                seg_dirs: list[Path] = []
                for seg in epochs.concat_epochs:
                    spec = _parse_concat_epoch_segment(seg=seg, segment_waveforms_dir=segment_waveforms_dir)
                    if spec is None:
                        continue
                    if spec.seg_dir.exists():
                        seg_dirs.append(spec.seg_dir)
                segment_folders = seg_dirs if seg_dirs else None
        except Exception:
            segment_folders = None

        # Extra debug output: concat-only (no segment overlays), uncurated.
        # This is useful to verify differences vs the combined concat+segments plot.
        concat_only_pdf = waveforms_out_dir / "waveforms_grid_concat_uncurated.pdf"
        if (not concat_only_pdf.exists()) or inputs.force_restart:
            logger.info("Writing concat-only waveforms grid PDF -> %s", concat_only_pdf)
            _write_waveforms_grid_pdf(
                waveforms_folder=concat_waveforms_dir,
                pdf_path=concat_only_pdf,
                segment_waveforms_folders=None,
                show_debug_annotation=False,
            )

        waveforms_grid_pdf = waveforms_out_dir / "waveforms_grid_uncurated.pdf"
        if (not waveforms_grid_pdf.exists()) or inputs.force_restart:
            logger.info("Writing waveforms grid PDF -> %s", waveforms_grid_pdf)
            _write_waveforms_grid_pdf(
                waveforms_folder=concat_waveforms_dir,
                pdf_path=waveforms_grid_pdf,
                segment_waveforms_folders=segment_folders,
                show_debug_annotation=False,
            )

        if curated_units_for_plot is not None:
            curated_pdf = waveforms_out_dir / "waveforms_grid_curated.pdf"
            if (not curated_pdf.exists()) or inputs.force_restart:
                logger.info("Writing curated waveforms grid PDF -> %s", curated_pdf)
                _write_waveforms_grid_pdf(
                    waveforms_folder=concat_waveforms_dir,
                    pdf_path=curated_pdf,
                    unit_ids=list(curated_units_for_plot),
                    segment_waveforms_folders=segment_folders,
                    show_debug_annotation=False,
                )

    return waveforms_grid_pdf, spikesorting_waveforms_grid_pdf


def _curate_then_plot(
    *,
    inputs: WaveformExtractInputs,
    waveforms_out_dir: Path,
    concat_waveforms_dir: Path,
    segment_waveforms_dir: Optional[Path],
    epochs: _EpochInputs,
    window: _WaveformWindow,
    logger: Any,
) -> tuple[Optional[Path], Optional[Path]]:
    """Compute metrics/curation first, then plot PDFs.

    This intentionally does *not* reuse spikesorting-step curation outputs.
    It recomputes quality + template metrics from the analyzers generated during
    waveform extraction.

    The logic is split into 3 runner steps:
      1) Compute + merge metrics
      2) Apply curation logic
      3) Plot
    """

    merged_qm, merged_tm = _compute_and_merge_waveforms_metrics(
        inputs=inputs,
        waveforms_out_dir=waveforms_out_dir,
        concat_waveforms_dir=concat_waveforms_dir,
        segment_waveforms_dir=segment_waveforms_dir,
        epochs=epochs,
        window=window,
        logger=logger,
    )
    curated_units_for_plot = _apply_waveforms_curation(
        waveforms_out_dir=waveforms_out_dir,
        merged_qm=merged_qm,
        merged_tm=merged_tm,
        logger=logger,
    )
    return _plot_waveforms_outputs(
        inputs=inputs,
        waveforms_out_dir=waveforms_out_dir,
        concat_waveforms_dir=concat_waveforms_dir,
        segment_waveforms_dir=segment_waveforms_dir,
        epochs=epochs,
        curated_units_for_plot=curated_units_for_plot,
        logger=logger,
    )


def extract_waveforms(
    *,
    inputs: WaveformExtractInputs,
    logger_name_prefix: str = "axon_reconstructor",
) -> WaveformExtractOutputs:
    """Extract waveforms from the preprocessed recording and sorter output.

    Produces:
      <well>/waveforms_outputs/concat_waveforms/
      <well>/waveforms_outputs/segment_waveforms/ (optional)
    <well>/waveforms_outputs/waveforms_grid_uncurated.pdf
    <well>/waveforms_outputs/waveforms_grid_curated.pdf (if curation succeeds)
        <well>/waveforms_outputs/qm_unfiltered.xlsx  (merged)
        <well>/waveforms_outputs/tm_unfiltered.xlsx  (merged)
    <well>/waveforms_outputs/metrics_curated.xlsx
    <well>/waveforms_outputs/tm_curated.xlsx
    <well>/waveforms_outputs/rejection_log.xlsx
    <well>/waveforms_outputs/wf_rejection_log.xlsx
        <well>/waveforms_outputs/metrics_sources/ (per-source concat/segment metrics + curation)
    plus JSON summaries.

    Uses existing epoch marker JSONs (from preprocessing) to avoid extracting
    waveforms that cross Maxwell snippet discontinuities.
    """

    ctx = _initialize_run_context(inputs=inputs, logger_name_prefix=logger_name_prefix)

    resumed = _resume_if_possible(inputs=inputs, ctx=ctx)
    if resumed is not None:
        return resumed

    ckpt = save_checkpoint(
        checkpoint_file=ctx.ckpt_file,
        state=ctx.ckpt,
        stage=ProcessingStage.ANALYZER,
        failed_stage=None,
        error=None,
        extra_fields={
            "waveforms_out_dir": str(ctx.waveforms_out_dir),
        },
    )

    ctx.logger.info("Waveform extraction starting: well_out_dir=%s", ctx.well_out_dir)

    try:
        recording = _load_preprocessed_recording(well_out_dir=ctx.well_out_dir)
        fs_hz = float(recording.get_sampling_frequency())

        window = _resolve_waveform_window(inputs=inputs, fs_hz=fs_hz)

        sorter_output_dir = _resolve_mea_sorter_output_dir(well_out_dir=ctx.well_out_dir)
        sorting = _load_sorting_from_sorter_output_dir(sorter_output_dir=sorter_output_dir, sorter=inputs.sorter)

        epochs = _load_epoch_markers(well_out_dir=ctx.well_out_dir, stream_id=inputs.stream_id)

        # Quality-metric params: keep consistent for concat + segments, and adapt
        # presence-ratio bin duration for short recordings/segments.
        from .qm_config import build_quality_metrics_extension_params

        # Use the shortest segment duration when segments are available; otherwise
        # fall back to the full concat duration.
        min_duration_s: float | None = None
        try:
            if inputs.per_segment and epochs.concat_epochs:
                durations = []
                for seg in epochs.concat_epochs:
                    try:
                        start = int(seg["start_sample"])
                        end = int(seg["end_sample"])
                        if end > start:
                            durations.append((end - start) / float(fs_hz))
                    except Exception:
                        continue
                if durations:
                    min_duration_s = float(min(durations))
            if min_duration_s is None:
                min_duration_s = float(recording.get_total_duration())
        except Exception:
            min_duration_s = None

        quality_metrics_params = build_quality_metrics_extension_params(min_duration_s=min_duration_s, logger=ctx.logger)
        filtering_summary = _init_filtering_summary(inputs=inputs, window=window, epochs=epochs)
        wf_rejection_rows, base_rej_fields = _init_wf_rejection_log_fields(inputs=inputs, window=window)

        filtered_sorting = _filter_sorting_by_maxwell_epochs(
            inputs=inputs,
            sorting=sorting,
            epochs=epochs,
            window=window,
            logger=ctx.logger,
            filtering_summary=filtering_summary,
            wf_rejection_rows=wf_rejection_rows,
            base_rej_fields=base_rej_fields,
        )

        _write_waveform_extraction_params(
            params_json=ctx.params_json,
            inputs=inputs,
            sorter_output_dir=sorter_output_dir,
            window=window,
        )


        # Used to avoid redundant per-segment waveforms on common channels.
        try:
            common_channel_ids = set(int(x) for x in recording.get_channel_ids())
        except Exception:
            common_channel_ids = set()

        _extract_concat_waveforms(
            inputs=inputs,
            filtered_sorting=filtered_sorting,
            recording=recording,
            concat_waveforms_dir=ctx.concat_waveforms_dir,
            window=window,
            quality_metrics_params=quality_metrics_params,
            logger=ctx.logger,
        )

        if inputs.per_segment and epochs.concat_epochs:
            assert ctx.segment_waveforms_dir is not None
            _extract_per_segment_waveforms(
                inputs=inputs,
                recording=recording,
                sorting_unfiltered=sorting,
                filtered_sorting=filtered_sorting,
                epochs=epochs,
                window=window,
                segment_waveforms_dir=ctx.segment_waveforms_dir,
                common_channel_ids=common_channel_ids,
                filtering_summary=filtering_summary,
                wf_rejection_rows=wf_rejection_rows,
                base_rej_fields=base_rej_fields,
                quality_metrics_params=quality_metrics_params,
                logger=ctx.logger,
            )

        _persist_filtering_and_exclusions(
            filtering_json=ctx.filtering_json,
            waveforms_out_dir=ctx.waveforms_out_dir,
            wf_rejection_rows=wf_rejection_rows,
            filtering_summary=filtering_summary,
            inputs=inputs,
            logger=ctx.logger,
        )

        merged_qm, merged_tm = _compute_and_merge_waveforms_metrics(
            inputs=inputs,
            waveforms_out_dir=ctx.waveforms_out_dir,
            concat_waveforms_dir=ctx.concat_waveforms_dir,
            segment_waveforms_dir=ctx.segment_waveforms_dir,
            epochs=epochs,
            window=window,
            logger=ctx.logger,
        )

        curated_units_for_plot = _apply_waveforms_curation(
            waveforms_out_dir=ctx.waveforms_out_dir,
            merged_qm=merged_qm,
            merged_tm=merged_tm,
            logger=ctx.logger,
        )

        waveforms_grid_pdf, spikesorting_waveforms_grid_pdf = _plot_waveforms_outputs(
            inputs=inputs,
            waveforms_out_dir=ctx.waveforms_out_dir,
            concat_waveforms_dir=ctx.concat_waveforms_dir,
            segment_waveforms_dir=ctx.segment_waveforms_dir,
            epochs=epochs,
            curated_units_for_plot=curated_units_for_plot,
            logger=ctx.logger,
        )

        ckpt = save_checkpoint(
            checkpoint_file=ctx.ckpt_file,
            state=ckpt,
            stage=ProcessingStage.ANALYZER_COMPLETE,
            failed_stage=None,
            error=None,
            extra_fields={
                "waveforms_out_dir": str(ctx.waveforms_out_dir),
                "concat_waveforms_dir": str(ctx.concat_waveforms_dir),
                "segment_waveforms_dir": str(ctx.segment_waveforms_dir) if ctx.segment_waveforms_dir else None,
                "waveforms_params_json": str(ctx.params_json),
                "waveforms_filtering_json": str(ctx.filtering_json),
                "waveforms_grid_pdf": str(waveforms_grid_pdf) if waveforms_grid_pdf else None,
                "spikesorting_waveforms_grid_pdf": None,
            },
        )

        ctx.logger.info("Waveform extraction complete")

        return WaveformExtractOutputs(
            well_out_dir=ctx.well_out_dir,
            waveforms_out_dir=ctx.waveforms_out_dir,
            concat_waveforms_dir=ctx.concat_waveforms_dir,
            segment_waveforms_dir=ctx.segment_waveforms_dir,
            params_json=ctx.params_json,
            filtering_json=ctx.filtering_json,
            waveforms_grid_pdf=waveforms_grid_pdf,
            spikesorting_waveforms_grid_pdf=spikesorting_waveforms_grid_pdf,
        )

    except Exception as e:
        save_checkpoint(
            checkpoint_file=ctx.ckpt_file,
            state=ckpt,
            stage=ProcessingStage.ANALYZER,
            failed_stage="WAVEFORMS",
            error=exception_to_error_dict(e),
        )
        ctx.logger.exception("Waveform extraction FAILED")
        raise
