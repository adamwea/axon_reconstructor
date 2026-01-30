from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .curation import apply_mea_analysis_curation
from .metrics import (
    SegmentAnalyzerSource,
    load_and_compute_metrics,
    merge_quality_metrics,
    merge_template_metrics,
    recompute_merged_quality_metrics_from_deduplicated_spikes,
)
from .plotting import _write_waveforms_grid_pdf
from .segments import _parse_concat_epoch_segment


def _compute_and_merge_waveforms_metrics(
    *,
    inputs,
    waveforms_out_dir: Path,
    concat_waveforms_dir: Path,
    segment_waveforms_dir: Optional[Path],
    epochs,
    window,
    logger: Any,
) -> tuple[Any, Any]:
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

    try:
        concat_dir = metrics_sources_dir / "concat"
        concat_dir.mkdir(parents=True, exist_ok=True)
        concat_qm.to_excel(concat_dir / "qm_unfiltered.xlsx")
        concat_tm.to_excel(concat_dir / "tm_unfiltered.xlsx")

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

    merged_qm = merge_quality_metrics(concat_qm=concat_qm, segment_qm_by_source=seg_qm_by_source)
    merged_tm = merge_template_metrics(concat_tm=concat_tm, segment_tm_by_source=seg_tm_by_source)

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

        try:
            union_index = sorted(set(merged_qm.index.values) | set(recomputed.index.values))
            merged_qm = merged_qm.reindex(union_index)
        except Exception:
            pass

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

    try:
        merged_qm.to_excel(waveforms_out_dir / "qm_merged_unfiltered.xlsx")
        merged_tm.to_excel(waveforms_out_dir / "tm_merged_unfiltered.xlsx")
    except Exception as e:
        logger.warning("Failed to save root metrics xlsx: %s", e)

    return merged_qm, merged_tm


def _apply_waveforms_curation(*, waveforms_out_dir: Path, merged_qm: Any, merged_tm: Any, logger: Any) -> Optional[list[Any]]:
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
    inputs,
    waveforms_out_dir: Path,
    concat_waveforms_dir: Path,
    segment_waveforms_dir: Optional[Path],
    epochs,
    curated_units_for_plot: Optional[list[Any]],
    logger: Any,
) -> tuple[Optional[Path], Optional[Path]]:
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

        concat_only_pdf = waveforms_out_dir / "waveforms_grid_concat_uncurated.pdf"
        if (not concat_only_pdf.exists()) or inputs.force_restart:
            logger.info("Writing concat-only waveforms grid PDF -> %s", concat_only_pdf)
            _write_waveforms_grid_pdf(
                waveforms_folder=concat_waveforms_dir,
                pdf_path=concat_only_pdf,
                segment_waveforms_folders=None,
                show_debug_annotation=False,
                panel_dir=(waveforms_out_dir / "waveforms_grid_concat_uncurated_panels"),
            )

        waveforms_grid_pdf = waveforms_out_dir / "waveforms_grid_uncurated.pdf"
        if (not waveforms_grid_pdf.exists()) or inputs.force_restart:
            logger.info("Writing waveforms grid PDF -> %s", waveforms_grid_pdf)
            _write_waveforms_grid_pdf(
                waveforms_folder=concat_waveforms_dir,
                pdf_path=waveforms_grid_pdf,
                segment_waveforms_folders=segment_folders,
                show_debug_annotation=False,
                panel_dir=(waveforms_out_dir / "waveforms_grid_uncurated_panels"),
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
                    panel_dir=(waveforms_out_dir / "waveforms_grid_curated_panels"),
                )

    return waveforms_grid_pdf, spikesorting_waveforms_grid_pdf


def _curate_then_plot(
    *,
    inputs,
    waveforms_out_dir: Path,
    concat_waveforms_dir: Path,
    segment_waveforms_dir: Optional[Path],
    epochs,
    window,
    logger: Any,
) -> tuple[Optional[Path], Optional[Path]]:
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


__all__ = [
    "_apply_waveforms_curation",
    "_compute_and_merge_waveforms_metrics",
    "_curate_then_plot",
    "_plot_waveforms_outputs",
]
