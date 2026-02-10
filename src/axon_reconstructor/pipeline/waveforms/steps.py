from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .curation import apply_mea_analysis_curation
from .plotting import _write_waveforms_grid_pdf
from .segments import _parse_concat_epoch_segment


def _load_spikesorting_quality_metrics(*, well_out_dir: Path, logger: Any) -> Optional[Any]:
    """Load quality metrics computed during spikesorting (MEA_Analysis).

    Waveforms stage intentionally does not recompute quality metrics because
    several metrics (e.g. presence_ratio) are sensitive to parameterization.
    """

    qm_xlsx = well_out_dir / "spikesorting_outputs" / "qm_unfiltered.xlsx"
    if not qm_xlsx.exists():
        logger.warning("Missing spikesorting qm_unfiltered.xlsx: %s", qm_xlsx)
        return None

    try:
        import pandas as pd  # type: ignore[import-not-found]
    except Exception:
        logger.warning("pandas is required to read %s", qm_xlsx)
        return None

    try:
        df = pd.read_excel(qm_xlsx, index_col=0)
    except Exception as e:
        logger.warning("Failed reading %s: %s", qm_xlsx, e)
        return None

    try:
        if (getattr(df, "index", None) is not None) and (str(df.index.name) == "unit_id"):
            return df
    except Exception:
        pass

    # Best-effort normalization: sometimes unit_id is a regular column.
    try:
        if "unit_id" in df.columns:
            df = df.set_index("unit_id", drop=True)
    except Exception:
        pass

    return df


def _apply_waveforms_curation(*, waveforms_out_dir: Path, logger: Any) -> Optional[list[Any]]:
    logger.info("Waveforms step 1/2: apply curation (from spikesorting metrics)")
    curated_units_for_plot: Optional[list[Any]] = None

    well_out_dir = waveforms_out_dir.parent
    qm = _load_spikesorting_quality_metrics(well_out_dir=well_out_dir, logger=logger)
    if qm is None:
        return None

    try:
        clean_metrics, _rejection_log = apply_mea_analysis_curation(q_metrics=qm, user_thresholds=None)
        curated_units_for_plot = list(clean_metrics.index.values)
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
    logger.info("Waveforms step 2/2: plot")
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

        grids_dir = waveforms_out_dir / "grids"
        panels_root = waveforms_out_dir / "panels"
        grids_dir.mkdir(parents=True, exist_ok=True)
        panels_root.mkdir(parents=True, exist_ok=True)

        # Best-effort cleanup: older versions of this step produced many grid/panel
        # permutations. Going forward we keep only uncurated + curated (plus the
        # per-segment curated_best_local outputs under grids/segments).
        try:
            import shutil

            legacy_stems = [
                "concat_uncurated",
                "concat_curated",
                "uncurated_old_bestchan",
                "uncurated_new_best_chan",
                "curated_old_bestchan",
                "curated_new_bestchan",
            ]

            for stem in legacy_stems:
                try:
                    (grids_dir / f"{stem}.pdf").unlink(missing_ok=True)  # type: ignore[arg-type]
                except Exception:
                    pass
                try:
                    shutil.rmtree(grids_dir / f"{stem}_pages", ignore_errors=True)
                except Exception:
                    pass
                try:
                    shutil.rmtree(panels_root / stem, ignore_errors=True)
                except Exception:
                    pass

            # Backward-compat: older root-level names.
            try:
                (waveforms_out_dir / "waveforms_grid_uncurated.pdf").unlink(missing_ok=True)  # type: ignore[arg-type]
            except Exception:
                pass
            try:
                (waveforms_out_dir / "waveforms_grid_curated.pdf").unlink(missing_ok=True)  # type: ignore[arg-type]
            except Exception:
                pass
        except Exception:
            pass

        def _write_grid(
            *,
            waveforms_folder: Path,
            name: str,
            unit_ids: Optional[list[Any]],
            segment_waveforms_folders: Optional[list[Path]],
            best_channel_mode: str,
            grids_subdir: Optional[Path] = None,
            panels_subdir: Optional[Path] = None,
        ) -> Path:
            base_grids_dir = grids_dir if grids_subdir is None else grids_subdir
            base_panels_dir = panels_root if panels_subdir is None else panels_subdir

            pdf_path = base_grids_dir / f"{name}.pdf"
            should_write = (not pdf_path.exists()) or inputs.force_restart or bool(getattr(inputs, "force_replot", False))
            if should_write:
                # Best-effort cleanup to avoid stale pages/panels lingering across replot runs.
                try:
                    import shutil

                    try:
                        pdf_path.unlink(missing_ok=True)  # type: ignore[arg-type]
                    except Exception:
                        pass
                    try:
                        shutil.rmtree(base_grids_dir / f"{name}_pages", ignore_errors=True)
                    except Exception:
                        pass
                    try:
                        shutil.rmtree(base_panels_dir / name, ignore_errors=True)
                    except Exception:
                        pass
                except Exception:
                    pass

                logger.info("Writing waveforms grid -> %s", pdf_path)
                _write_waveforms_grid_pdf(
                    waveforms_folder=waveforms_folder,
                    pdf_path=pdf_path,
                    unit_ids=unit_ids,
                    segment_waveforms_folders=segment_waveforms_folders,
                    show_debug_annotation=False,
                    panel_dir=(base_panels_dir / name),
                    pages_dir=(base_grids_dir / f"{name}_pages"),
                    write_page_png=True,
                    write_page_svg=True,
                    max_spikes_to_plot=50,
                    best_channel_mode=("old" if str(best_channel_mode) == "old" else "new"),
                )
            return pdf_path

        # Primary outputs: only uncurated + curated (best-effort) grids/panels.
        # These use concat waveforms as the anchor and choose the best channel from
        # the concat spikesorting waveform/template (best_channel_mode="old").
        waveforms_grid_pdf = _write_grid(
            waveforms_folder=concat_waveforms_dir,
            name="uncurated",
            unit_ids=None,
            segment_waveforms_folders=segment_folders,
            best_channel_mode="old",
        )

        if curated_units_for_plot is not None:
            _write_grid(
                waveforms_folder=concat_waveforms_dir,
                name="curated",
                unit_ids=list(curated_units_for_plot),
                segment_waveforms_folders=segment_folders,
                best_channel_mode="old",
            )

        # Novel outputs: per-segment curated grids using best-local channels.
        # Here we plot each segment in isolation (no cross-source channel selection).
        if curated_units_for_plot is not None and segment_folders:
            try:
                seg_grids_root = grids_dir / "segments"
                seg_panels_root = panels_root / "segments"
                seg_grids_root.mkdir(parents=True, exist_ok=True)
                seg_panels_root.mkdir(parents=True, exist_ok=True)

                for seg_dir in segment_folders:
                    seg_name = seg_dir.name
                    out_grids_dir = seg_grids_root / seg_name
                    out_panels_dir = seg_panels_root / seg_name
                    out_grids_dir.mkdir(parents=True, exist_ok=True)
                    out_panels_dir.mkdir(parents=True, exist_ok=True)

                    _write_grid(
                        waveforms_folder=seg_dir,
                        name="curated_best_local",
                        unit_ids=list(curated_units_for_plot),
                        segment_waveforms_folders=None,
                        best_channel_mode="old",
                        grids_subdir=out_grids_dir,
                        panels_subdir=out_panels_dir,
                    )
            except Exception:
                # Per-segment plotting is best-effort.
                pass

    return waveforms_grid_pdf, spikesorting_waveforms_grid_pdf
__all__ = [
    "_apply_waveforms_curation",
    "_plot_waveforms_outputs",
    "_load_spikesorting_quality_metrics",
]
