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
    <well>/waveforms_outputs/qm_unfiltered.xlsx
    <well>/waveforms_outputs/tm_unfiltered.xlsx
    <well>/waveforms_outputs/metrics_curated.xlsx
    <well>/waveforms_outputs/tm_curated.xlsx
    <well>/waveforms_outputs/rejection_log.xlsx
    <well>/waveforms_outputs/wf_rejection_log.xlsx
    plus JSON summaries.

    Uses existing epoch marker JSONs (from preprocessing) to avoid extracting
    waveforms that cross Maxwell snippet discontinuities.
    """

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

    # Resume shortcut: trust existing artifacts if present.
    if not inputs.force_restart and concat_waveforms_dir.exists():
        logger.info("Resuming waveforms: existing outputs found at %s", concat_waveforms_dir)

        # Best-effort: populate optional PDF fields if they exist.
        waveforms_grid_pdf = waveforms_out_dir / "waveforms_grid_uncurated.pdf"
        waveforms_grid_pdf = waveforms_grid_pdf if waveforms_grid_pdf.exists() else None
        spikesorting_waveforms_grid_pdf = None

        return WaveformExtractOutputs(
            well_out_dir=well_out_dir,
            waveforms_out_dir=waveforms_out_dir,
            concat_waveforms_dir=concat_waveforms_dir,
            segment_waveforms_dir=segment_waveforms_dir,
            params_json=params_json,
            filtering_json=filtering_json,
            waveforms_grid_pdf=waveforms_grid_pdf,
            spikesorting_waveforms_grid_pdf=spikesorting_waveforms_grid_pdf,
        )

    ckpt = save_checkpoint(
        checkpoint_file=ckpt_file,
        state=ckpt,
        stage=ProcessingStage.ANALYZER,
        failed_stage=None,
        error=None,
        extra_fields={
            "waveforms_out_dir": str(waveforms_out_dir),
        },
    )

    logger.info("Waveform extraction starting: well_out_dir=%s", well_out_dir)

    try:
        recording = _load_preprocessed_recording(well_out_dir=well_out_dir)
        fs_hz = float(recording.get_sampling_frequency())

        ms_before = float(inputs.ms_before) if inputs.ms_before is not None else None
        ms_after = float(inputs.ms_after) if inputs.ms_after is not None else None
        if ms_before is None or ms_after is None:
            inferred_before, inferred_after = _infer_cutout_ms(h5_path=inputs.h5_path, stream_id=inputs.stream_id, fs_hz=fs_hz)
            ms_before = inferred_before if ms_before is None else ms_before
            ms_after = inferred_after if ms_after is None else ms_after

        pre_samples = int(math.ceil(ms_before * fs_hz / 1000.0))
        post_samples = int(math.ceil(ms_after * fs_hz / 1000.0))

        sorter_output_dir = _resolve_mea_sorter_output_dir(well_out_dir=well_out_dir)
        sorting = _load_sorting_from_sorter_output_dir(sorter_output_dir=sorter_output_dir, sorter=inputs.sorter)

        # Load epoch markers from preprocessing.
        preprocess_dir = well_out_dir / PREPROCESS_OUTPUTS_DIRNAME
        maxwell_epochs_path = preprocess_dir / f"maxwell_contiguous_epochs_{inputs.stream_id}.json"
        concat_epochs_path = preprocess_dir / f"concatenation_stitch_epochs_{inputs.stream_id}.json"

        maxwell_epochs: list[dict] = []
        maxwell_intervals: list[tuple[int, int]] = []
        concat_epochs: list[dict] = []

        if maxwell_epochs_path.exists():
            maxwell_epochs = list(_read_json(maxwell_epochs_path))
            # These are in concatenated sample coordinates.
            maxwell_intervals = _epochs_to_intervals(maxwell_epochs)
        if concat_epochs_path.exists():
            concat_epochs = list(_read_json(concat_epochs_path))

        # Filter spikes by Maxwell epochs (avoid snippet boundary crossings).
        filtered_sorting = sorting
        filtering_summary: dict[str, Any] = {
            "filter_by_maxwell_epochs": bool(inputs.filter_by_maxwell_epochs),
            "ms_before": ms_before,
            "ms_after": ms_after,
            "pre_samples": pre_samples,
            "post_samples": post_samples,
            "maxwell_epochs_path": str(maxwell_epochs_path) if maxwell_epochs_path.exists() else None,
            "concat_epochs_path": str(concat_epochs_path) if concat_epochs_path.exists() else None,
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

        # Per-spike rejection rows for auditability / downstream alignment.
        # Written to waveforms_outputs/wf_rejection_log.xlsx at the end of the stage.
        wf_rejection_rows: list[dict[str, Any]] = []
        base_rej_fields: dict[str, Any] = {
            "stream_id": inputs.stream_id,
            "sorter": inputs.sorter,
            "h5_path": str(inputs.h5_path),
            "fs_hz": float(fs_hz),
            "ms_before": float(ms_before),
            "ms_after": float(ms_after),
            "pre_samples": int(pre_samples),
            "post_samples": int(post_samples),
        }

        if inputs.filter_by_maxwell_epochs and maxwell_intervals:
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
                    intervals=maxwell_intervals,
                    pre_samples=pre_samples,
                    post_samples=post_samples,
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
                            "spike_time_s": float(t) / float(fs_hz),
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
                            "spike_time_s": float(t) / float(fs_hz),
                            "reason": "waveform_window_crosses_epoch_edge",
                        }
                    )

            filtered_sorting = _to_numpy_sorting(unit_trains=unit_trains, fs_hz=fs_hz)
            filtering_summary["removed_spikes_total"] = int(removed_total)
            filtering_summary["kept_spikes_total"] = int(kept_total)
            filtering_summary["removed_by_maxwell_epoch_total"] = int(removed_outside_total)
            filtering_summary["removed_by_edge_total"] = int(removed_edge_total)
            logger.info(
                "Filtered spikes by Maxwell epochs: kept=%d removed=%d (pre=%d post=%d samples)",
                kept_total,
                removed_total,
                pre_samples,
                post_samples,
            )
        else:
            # Best-effort counts.
            try:
                n_total = sum(len(sorting.get_unit_spike_train(u)) for u in sorting.get_unit_ids())
                filtering_summary["kept_spikes_total"] = int(n_total)
            except Exception:
                pass

        _write_json(params_json, {
            "h5_path": str(inputs.h5_path),
            "stream_id": inputs.stream_id,
            "sorter": inputs.sorter,
            "sorter_output_dir": str(sorter_output_dir),
            "ms_before": ms_before,
            "ms_after": ms_after,
            "n_jobs": int(inputs.n_jobs),
            "max_spikes_per_unit": inputs.max_spikes_per_unit,
            "per_segment": bool(inputs.per_segment),
            "per_segment_recording_source": "raw_maxwell_full_channels" if inputs.per_segment else None,
            "per_segment_only_additional_channels": bool(inputs.per_segment_only_additional_channels),
        })

        # Used to avoid redundant per-segment waveforms on common channels.
        try:
            common_channel_ids = set(int(x) for x in recording.get_channel_ids())
        except Exception:
            common_channel_ids = set()

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
                "waveforms": {"ms_before": float(ms_before), "ms_after": float(ms_after)},
            },
            verbose=False,
            n_jobs=int(inputs.n_jobs),
        )

        # Optional: per-segment extraction using concatenation epochs.
        if inputs.per_segment and concat_epochs:
            assert segment_waveforms_dir is not None
            segment_waveforms_dir.mkdir(parents=True, exist_ok=True)

            logger.info("Extracting per-segment waveforms -> %s", segment_waveforms_dir)
            for seg in concat_epochs:
                try:
                    seg_index = int(seg["segment_index"])
                    rec_name = str(seg.get("rec_name", f"seg{seg_index}"))
                    start = int(seg["start_sample"])
                    end = int(seg["end_sample"])
                except Exception:
                    continue

                seg_dir = segment_waveforms_dir / f"seg{seg_index:02d}_{rec_name}"
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
                                    "skipped_reason": "no_additional_channels",
                                }
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
                # Compute Maxwell contiguous-epoch intervals in *segment-local* coordinates.
                # This ensures we don't extract waveforms whose windows cross snippet gaps
                # within this segment.
                seg_maxwell_intervals: list[tuple[int, int]] = []
                if inputs.filter_by_maxwell_epochs and maxwell_epochs:
                    seg_maxwell_intervals = _maxwell_epochs_to_segment_local_intervals(
                        maxwell_epochs=maxwell_epochs,
                        segment_index=seg_index,
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

                    # Filter by Maxwell contiguous epochs *within this segment*.
                    local_epoch_filtered = local_all
                    removed_epoch = 0
                    if inputs.filter_by_maxwell_epochs and seg_maxwell_intervals:
                        (
                            local_epoch_filtered,
                            removed_outside,
                            removed_edge_epoch,
                            removed_outside_spikes,
                            removed_edge_spikes,
                        ) = _filter_spike_train_by_intervals(
                            spike_train=local_all,
                            intervals=seg_maxwell_intervals,
                            pre_samples=pre_samples,
                            post_samples=post_samples,
                        )
                        removed_epoch = int(removed_outside) + int(removed_edge_epoch)

                        source_name = f"seg{int(seg_index):02d}_{rec_name}"
                        for t_local in removed_outside_spikes:
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
                                    "spike_time_s": float(t_concat) / float(fs_hz),
                                    "reason": "outside_maxwell_epoch",
                                }
                            )
                        for t_local in removed_edge_spikes:
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
                                    "spike_time_s": float(t_concat) / float(fs_hz),
                                    "reason": "waveform_window_crosses_epoch_edge",
                                }
                            )

                    # Extra safety: enforce segment-edge constraints even if epoch markers
                    # are missing/unexpected.
                    kept_edges: list[int] = []
                    for t_local in local_epoch_filtered:
                        if t_local - pre_samples < 0:
                            continue
                        if t_local + post_samples >= int(seg_len):
                            continue
                        kept_edges.append(int(t_local))

                    # Log spikes rejected due to segment-edge constraints.
                    try:
                        source_name = f"seg{int(seg_index):02d}_{rec_name}"
                        kept_edge_set = set(int(x) for x in kept_edges)
                        for t_local in local_epoch_filtered:
                            if int(t_local) in kept_edge_set:
                                continue
                            if (int(t_local) - int(pre_samples) < 0) or (
                                int(t_local) + int(post_samples) >= int(seg_len)
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
                                        "spike_time_s": float(t_concat) / float(fs_hz),
                                        "reason": "waveform_window_outside_segment_bounds",
                                    }
                                )
                    except Exception:
                        pass

                    removed_edge = int(len(local_epoch_filtered) - len(kept_edges))
                    seg_removed_epoch_total += int(removed_epoch)
                    seg_removed_edge_total += int(removed_edge)
                    seg_kept_total += int(len(kept_edges))

                    unit_trains_seg[int(u)] = kept_edges

                # Convert the segment-local spike trains into a SortingExtractor.
                # This is the bridge that lets us use the holistic concat sorting
                # to drive waveform extraction on each raw segment.
                seg_sort = _to_numpy_sorting(unit_trains=unit_trains_seg, fs_hz=fs_hz)

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
                        "waveforms": {"ms_before": float(ms_before), "ms_after": float(ms_after)},
                    },
                    verbose=False,
                    n_jobs=max(1, int(inputs.n_jobs)),
                )

                # Accumulate per-segment filtering stats.
                try:
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
                            "maxwell_intervals_in_segment": int(len(seg_maxwell_intervals)),
                            "raw_channels_total": raw_channels_total,
                            "excluded_common_channels_total": excluded_common_channels_total,
                            "kept_additional_channels_total": kept_additional_channels_total,
                        }
                    )
                except Exception:
                    pass

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

        waveforms_grid_pdf: Optional[Path] = None
        spikesorting_waveforms_grid_pdf: Optional[Path] = None
        if inputs.plot_waveforms_grid_pdf:
            waveforms_grid_pdf = waveforms_out_dir / "waveforms_grid_uncurated.pdf"

            # Recreate only when force_restart or missing.
            if (not waveforms_grid_pdf.exists()) or inputs.force_restart:
                logger.info("Writing waveforms grid PDF -> %s", waveforms_grid_pdf)
                _write_waveforms_grid_pdf(
                    waveforms_folder=concat_waveforms_dir,
                    pdf_path=waveforms_grid_pdf,
                )

            # Curation outputs (MEA_Analysis-style) + curated grid PDF.
            try:
                curated_units, curation_paths = _run_mea_analysis_style_curation(
                    recording=recording,
                    sorting=sorting,
                    output_dir=waveforms_out_dir,
                    n_jobs=int(inputs.n_jobs),
                    ms_before=float(ms_before),
                    ms_after=float(ms_after),
                    force_restart=bool(inputs.force_restart),
                    logger=logger,
                )

                curated_pdf = waveforms_out_dir / "waveforms_grid_curated.pdf"
                if (not curated_pdf.exists()) or inputs.force_restart:
                    logger.info("Writing curated waveforms grid PDF -> %s", curated_pdf)
                    _write_waveforms_grid_pdf(
                        waveforms_folder=concat_waveforms_dir,
                        pdf_path=curated_pdf,
                        unit_ids=list(curated_units),
                    )
            except Exception as e:
                logger.warning("Curation failed; skipping curated metrics/grid: %s", e)

        ckpt = save_checkpoint(
            checkpoint_file=ckpt_file,
            state=ckpt,
            stage=ProcessingStage.ANALYZER_COMPLETE,
            failed_stage=None,
            error=None,
            extra_fields={
                "waveforms_out_dir": str(waveforms_out_dir),
                "concat_waveforms_dir": str(concat_waveforms_dir),
                "segment_waveforms_dir": str(segment_waveforms_dir) if segment_waveforms_dir else None,
                "waveforms_params_json": str(params_json),
                "waveforms_filtering_json": str(filtering_json),
                "waveforms_grid_pdf": str(waveforms_grid_pdf) if waveforms_grid_pdf else None,
                "spikesorting_waveforms_grid_pdf": None,
            },
        )

        logger.info("Waveform extraction complete")

        return WaveformExtractOutputs(
            well_out_dir=well_out_dir,
            waveforms_out_dir=waveforms_out_dir,
            concat_waveforms_dir=concat_waveforms_dir,
            segment_waveforms_dir=segment_waveforms_dir,
            params_json=params_json,
            filtering_json=filtering_json,
            waveforms_grid_pdf=waveforms_grid_pdf,
            spikesorting_waveforms_grid_pdf=spikesorting_waveforms_grid_pdf,
        )

    except Exception as e:
        save_checkpoint(
            checkpoint_file=ckpt_file,
            state=ckpt,
            stage=ProcessingStage.ANALYZER,
            failed_stage="WAVEFORMS",
            error=exception_to_error_dict(e),
        )
        logger.exception("Waveform extraction FAILED")
        raise
