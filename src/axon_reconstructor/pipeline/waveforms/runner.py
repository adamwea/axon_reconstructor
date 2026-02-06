from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..checkpointing import ProcessingStage, exception_to_error_dict, save_checkpoint

from .artifacts import _persist_channel_groups_json, _persist_filtering_and_exclusions, _write_waveform_extraction_params
from .extraction import _extract_concat_waveforms, _extract_per_segment_waveforms
from .filtering import _filter_sorting_by_maxwell_epochs, _init_filtering_summary, _init_wf_rejection_log_fields
from .run_context import _WaveformsRunContext, _initialize_run_context, _load_epoch_markers, _resolve_waveform_window
from .steps import _apply_waveforms_curation, _plot_waveforms_outputs
from .utils import _load_preprocessed_recording, _load_sorting_from_sorter_output_dir, _resolve_mea_sorter_output_dir


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

    # If True, apply MEA_Analysis-style preprocessing to each raw segment recording
    # before extracting per-segment waveforms.
    #
    # This is intentionally applied only to per-segment analyzers (not the concat
    # analyzer), because per-segment recordings are loaded directly from the raw
    # Maxwell .h5 and would otherwise not match the preprocessing used by the
    # MEA_Analysis sorting pipeline.
    per_segment_preprocess_like_mea_analysis: bool = True

    # If True, per-segment waveforms are extracted only on channels that were
    # excluded during concatenation (i.e. not in the common-electrode intersection).
    # This avoids duplicating waveforms for the common channels already covered by
    # concat_waveforms.
    per_segment_only_additional_channels: bool = True

    # Resume/overwrite controls
    force_restart: bool = False

    # If True, drop spikes whose waveform window would cross Maxwell snippet boundaries.
    filter_by_maxwell_epochs: bool = True

    # Deprecated (2026-01): historically we also *flagged* extracted per-segment random_spikes
    # against segment-local Maxwell intervals after waveforms were computed. This is redundant
    # when concat-time filtering is authoritative and does not mutate analyzers anyway.
    deprecated_flag_segment_random_spikes_by_epochs: bool = False

    # Deprecated (2026-01): spike-level exclusions are no longer persisted as wf_exclusions.npz.
    # Downstream stages should consume the waveforms analyzers directly.
    # This flag remains as an escape hatch for older workflows.
    write_wf_exclusions_npz: bool = False

    # Plotting
    plot_waveforms_grid_pdf: bool = True

    # Debug/perf controls (developer convenience): optionally limit work to the first N
    # units/segments to speed up interactive runs.
    debug_max_units: Optional[int] = None
    debug_max_segments: Optional[int] = None


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


def _resume_if_possible(*, inputs: WaveformExtractInputs, ctx: _WaveformsRunContext) -> Optional[WaveformExtractOutputs]:
    if not inputs.force_restart and ctx.concat_waveforms_dir.exists():
        ctx.logger.info("Resuming waveforms: existing outputs found at %s", ctx.concat_waveforms_dir)

        # Preferred (2026-02): grid outputs live under waveforms_outputs/grids.
        # Current convention: only "uncurated.pdf" (+ "curated.pdf" best-effort).
        waveforms_grid_pdf = ctx.waveforms_out_dir / "grids" / "uncurated.pdf"
        if not waveforms_grid_pdf.exists():
            # Backward-compat: older variants.
            legacy_variant = ctx.waveforms_out_dir / "grids" / "uncurated_new_best_chan.pdf"
            if legacy_variant.exists():
                waveforms_grid_pdf = legacy_variant
            else:
                legacy_root = ctx.waveforms_out_dir / "waveforms_grid_uncurated.pdf"
                waveforms_grid_pdf = legacy_root if legacy_root.exists() else None
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


def extract_waveforms(
    *,
    inputs: WaveformExtractInputs,
    logger_name_prefix: str = "axon_reconstructor",
) -> WaveformExtractOutputs:
    """Extract waveforms from the preprocessed recording and sorter output.

    Produces:
      <well>/waveforms_outputs/concat_waveforms/
      <well>/waveforms_outputs/segment_waveforms/ (optional)
            <well>/waveforms_outputs/grids/uncurated.pdf
            <well>/waveforms_outputs/grids/curated.pdf (if curation succeeds)
            <well>/waveforms_outputs/grids/segments/<seg_name>/curated_best_local.pdf (per segment)
            <well>/waveforms_outputs/panels/<name>/unit_<id>.svg (per unit)
            <well>/waveforms_outputs/wf_rejection_log.xlsx
            plus JSON summaries.

        Notes:
        - Waveforms stage does not recompute quality/template metrics.
        - When applying curation logic for plotting, this stage reads:
                <well>/spikesorting_outputs/qm_unfiltered.xlsx

    Uses existing epoch marker JSONs (from preprocessing) to avoid extracting
    waveforms that cross Maxwell snippet discontinuities.
    """

    ctx = _initialize_run_context(inputs=inputs, logger_name_prefix=logger_name_prefix)

    # Fast-path resume: if prior outputs exist and we're not forcing a restart,
    # return a best-effort outputs object without recomputing analyzers/metrics.
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
        # Load the *preprocessed* concatenated recording used for spikesorting.
        # All time/sample references in this stage are in this concatenated time base.
        recording = _load_preprocessed_recording(well_out_dir=ctx.well_out_dir)
        fs_hz = float(recording.get_sampling_frequency())

        # Determine waveform cutout window in samples.
        # If ms_before/ms_after are not explicitly provided, infer them from trigger metadata.
        window = _resolve_waveform_window(inputs=inputs, fs_hz=fs_hz)

        # Load spikesorting results. Spike times are in the concatenated time base.
        sorter_output_dir = _resolve_mea_sorter_output_dir(well_out_dir=ctx.well_out_dir)
        sorting = _load_sorting_from_sorter_output_dir(sorter_output_dir=sorter_output_dir, sorter=inputs.sorter)

        # Postprocessing parity with MEA_Analysis (after pairing sorting with the concat recording):
        # remove spikes that fall beyond the recording length, then drop any units that
        # became empty as a result. This prevents analyzer-time indexing errors and
        # keeps downstream metrics consistent.
        # NOTE: this step may not be necessery if the sorter output is already cleaned. aw 2026-02-01 21:47:35
        try:
            import spikeinterface.full as si  # type: ignore[import-not-found]

            sorting = si.remove_excess_spikes(sorting, recording)
            sorting = sorting.remove_empty_units()
        except Exception:
            ctx.logger.debug(
                "Sorting cleanup (remove_excess_spikes/remove_empty_units) failed; continuing without cleanup.",
                exc_info=True,
            )

        # Optional debug limit: restrict to a subset of units early to reduce compute.
        try:
            if inputs.debug_max_units is not None:
                max_units = int(inputs.debug_max_units)
                if max_units > 0:
                    try:
                        unit_ids = list(sorting.get_unit_ids())
                    except Exception:
                        unit_ids = list(getattr(sorting, "unit_ids", []))

                    if unit_ids:
                        try:
                            unit_ids_sorted = sorted(unit_ids)
                        except Exception:
                            unit_ids_sorted = sorted(unit_ids, key=lambda x: str(x))

                        keep = unit_ids_sorted[:max_units]
                        try:
                            sorting = sorting.select_units(unit_ids=keep)
                        except Exception:
                            # Some Sorting implementations use positional args.
                            try:
                                sorting = sorting.select_units(keep)
                            except Exception:
                                pass

                        ctx.logger.warning(
                            "DEBUG: limiting waveforms to first %d units (of %d)",
                            int(len(keep)),
                            int(len(unit_ids_sorted)),
                        )
        except Exception:
            pass

        # Load epoch marker JSONs produced during preprocessing.
        # Scientific rationale: Maxwell recordings can contain snippet discontinuities;
        # we use contiguous-epoch markers to avoid extracting waveforms whose window
        # would cross a boundary (which would mix unrelated signal segments).
        epochs = _load_epoch_markers(well_out_dir=ctx.well_out_dir, stream_id=inputs.stream_id)

        # Optional debug limit: restrict to the first N concat stitch segments.
        # This only affects per-segment extraction/plotting (concat analyzer still
        # spans the full concatenated recording).
        try:
            if inputs.debug_max_segments is not None and epochs.concat_epochs:
                from .run_context import _EpochInputs

                max_segments = int(inputs.debug_max_segments)
                if max_segments > 0:
                    limited = list(epochs.concat_epochs)[:max_segments]
                    if len(limited) < len(epochs.concat_epochs):
                        ctx.logger.warning(
                            "DEBUG: limiting per-segment waveforms to first %d segments (of %d)",
                            int(len(limited)),
                            int(len(epochs.concat_epochs)),
                        )

                    epochs = _EpochInputs(
                        preprocess_dir=epochs.preprocess_dir,
                        maxwell_epochs_path=epochs.maxwell_epochs_path,
                        concat_epochs_path=epochs.concat_epochs_path,
                        maxwell_epochs=epochs.maxwell_epochs,
                        maxwell_intervals=epochs.maxwell_intervals,
                        concat_epochs=limited,
                    )
        except Exception:
            pass

        # Initialize structured summaries that are persisted for auditability:
        # - filtering_summary: aggregate counts and per-segment breakdowns
        # - wf_rejection_rows: spike-level reasons for exclusion (joinable downstream)
        filtering_summary = _init_filtering_summary(inputs=inputs, window=window, epochs=epochs)
        wf_rejection_rows, base_rej_fields = _init_wf_rejection_log_fields(inputs=inputs, window=window)

        # Filter the sorting to remove spikes that are incompatible with waveform extraction.
        # Specifically, drop spikes whose waveform window would cross Maxwell snippet
        # boundaries (when markers exist and filter_by_maxwell_epochs=True).
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

        # Persist the parameters used so results are reproducible and debuggable.
        _write_waveform_extraction_params(
            params_json=ctx.params_json,
            inputs=inputs,
            sorter_output_dir=sorter_output_dir,
            window=window,
        )

        # Identify the electrode/channel set present in the concatenated recording.
        # Scientific rationale: concatenation often keeps only the shared electrode
        # intersection across segments. Per-segment extraction can optionally focus
        # on *non-common* segment channels that were dropped from the concat recording.
        #
        # IMPORTANT: for set operations we prefer electrode IDs from contact_vector
        # when available; otherwise fall back to recording channel IDs.
        common_channel_ids: set[int] = set()
        try:
            import numpy as np  # type: ignore[import-not-found]

            cv = recording.get_property("contact_vector")
            electrodes = np.asarray(cv["electrode"], dtype=int)
            if electrodes.size == int(recording.get_num_channels()):
                common_channel_ids = set(int(e) for e in electrodes.tolist())

                try:
                    ch_ids = [int(x) for x in recording.get_channel_ids()]
                    if set(ch_ids) != set(common_channel_ids):
                        ctx.logger.warning(
                            "Concat recording channel_ids differ from contact_vector electrode ids (n_ch=%d). "
                            "Using electrode ids for common_channel_ids.",
                            int(recording.get_num_channels()),
                        )
                except Exception:
                    pass
        except Exception:
            common_channel_ids = set()

        if not common_channel_ids:
            try:
                common_channel_ids = set(int(x) for x in recording.get_channel_ids())
            except Exception:
                common_channel_ids = set()

        # Channel-set bookkeeping for debugging/documentation. This is later enriched
        # by per-segment extraction (if enabled) and persisted as channel_groups.json.
        channel_groups: dict[str, object] = {
            "common_channel_ids": sorted(int(x) for x in common_channel_ids),
            "segments": {},
        }

        # 1) Extract waveforms (and compute analyzer extensions) on the concatenated recording.
        # Computational rationale: random spike sub-sampling bounds compute cost while still
        # providing representative waveform snippets for QC and template estimation.
        concat_best = _extract_concat_waveforms(
            inputs=inputs,
            filtered_sorting=filtered_sorting,
            recording=recording,
            concat_waveforms_dir=ctx.concat_waveforms_dir,
            window=window,
            logger=ctx.logger,
        )

        # 2) Optionally extract per-segment waveforms on the *raw* segment recordings.
        # Scientific rationale: this recovers waveforms on electrodes that were excluded
        # during concatenation (i.e. channels not present in all segments).
        # Note: we reuse spike times from the concatenated sorting; this step does not
        # perform new spike detection.
        seg_best: dict[object, tuple[float, object, str]] = {}
        if inputs.per_segment and epochs.concat_epochs:
            assert ctx.segment_waveforms_dir is not None
            seg_best = _extract_per_segment_waveforms(
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
                logger=ctx.logger,
                channel_groups=channel_groups,  # populated in-place
            )

        # Persist channel group info (best-effort, even if per-segment is disabled).
        try:
            _persist_channel_groups_json(
                channel_groups_json=ctx.waveforms_out_dir / "channel_groups.json",
                channel_groups=channel_groups,  # type: ignore[arg-type]
            )

            try:
                counts = channel_groups.get("counts") if isinstance(channel_groups, dict) else None
                if isinstance(counts, dict):
                    ctx.logger.info(
                        "Channel groups: common=%s, all_union=%s, non_unique_non_common=%s",
                        str(counts.get("common_channels_concat")),
                        str(counts.get("all_channels_union")),
                        str(counts.get("non_unique_non_common")),
                    )
                else:
                    ctx.logger.info(
                        "Channel groups persisted: common=%d (per-segment groups may be unavailable if per_segment=False)",
                        int(len(common_channel_ids)),
                    )
            except Exception:
                pass
        except Exception:
            pass

        # Cross-source QC: if any segment contains a stronger channel (PTP) for a unit
        # than concat, it's a sign the concat/common-electrode intersection may have
        # dropped the unit's true best electrode. This can make sorter timestamps a
        # weaker reference for the full channel set.
        try:
            if seg_best:
                ratio_thr = 1.10
                abs_thr_uv = 2.0
                warned = 0

                for u, (ptp_seg, ch_seg, src) in seg_best.items():
                    ptp_seg_f = float(ptp_seg)
                    ptp_concat_f = float(concat_best.get(u, (0.0, None))[0]) if concat_best else 0.0
                    ch_concat = concat_best.get(u, (0.0, None))[1] if concat_best else None

                    if not (ptp_seg_f > 0):
                        continue

                    better = (ptp_seg_f > ptp_concat_f + float(abs_thr_uv)) and (
                        ptp_concat_f <= 0 or (ptp_seg_f / max(ptp_concat_f, 1e-9)) >= float(ratio_thr)
                    )
                    if better:
                        warned += 1
                        ctx.logger.warning(
                            "Unit %s: segment has stronger best-channel PTP than concat (segment=%s ch=%s ptp=%.2f µV; concat ch=%s ptp=%.2f µV). "
                            "This suggests concat/common channel intersection may have dropped the unit's strongest electrode; consider alignment work for multi-source merging/propagation.",
                            u,
                            str(src),
                            str(ch_seg),
                            float(ptp_seg_f),
                            str(ch_concat),
                            float(ptp_concat_f),
                        )

                if warned == 0:
                    ctx.logger.info("Cross-source best-channel check: no units had a stronger segment channel than concat.")
        except Exception:
            pass

        # Persist a cross-source best-channel log so later stages (templates/reconstruction)
        # can audit which source appears to contain the strongest electrode for each unit.
        try:
            from .reporting import _write_best_channel_sources_xlsx

            _write_best_channel_sources_xlsx(
                best_channel_sources_xlsx=ctx.waveforms_out_dir / "best_channel_sources.xlsx",
                concat_waveforms_dir=ctx.concat_waveforms_dir,
                segment_waveforms_dir=ctx.segment_waveforms_dir,
                force_restart=bool(inputs.force_restart),
                logger=ctx.logger,
            )
        except Exception:
            pass

        # Persist filtering summaries and compact exclusion artifacts.
        # Scientific rationale: keeping both aggregate counts and spike-level rows enables
        # reproducible downstream template averaging (with consistent exclusions applied).
        _persist_filtering_and_exclusions(
            filtering_json=ctx.filtering_json,
            waveforms_out_dir=ctx.waveforms_out_dir,
            wf_rejection_rows=wf_rejection_rows,
            filtering_summary=filtering_summary,
            inputs=inputs,
            logger=ctx.logger,
        )

        # Apply curation thresholds (MEA_Analysis-style) based on spikesorting outputs.
        # If this fails, plotting proceeds with uncurated units.
        curated_units_for_plot = _apply_waveforms_curation(
            waveforms_out_dir=ctx.waveforms_out_dir,
            logger=ctx.logger,
        )

        # Plot waveforms grids for human QC (uncurated always; curated best-effort).
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
        # Preserve the waveforms-specific checkpoint state (separate from the main pipeline
        # checkpoint) so failures can be diagnosed and rerun without regressing stage numbers.
        save_checkpoint(
            checkpoint_file=ctx.ckpt_file,
            state=ckpt,
            stage=ProcessingStage.ANALYZER,
            failed_stage="WAVEFORMS",
            error=exception_to_error_dict(e),
        )
        ctx.logger.exception("Waveform extraction FAILED")
        raise
