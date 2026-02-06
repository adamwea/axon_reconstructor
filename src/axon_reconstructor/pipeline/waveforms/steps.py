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

        def _write_perm(
            *,
            name: str,
            unit_ids: Optional[list[Any]],
            segment_waveforms_folders: Optional[list[Path]],
            best_channel_mode: str,
        ) -> Path:
            pdf_path = grids_dir / f"{name}.pdf"
            if (not pdf_path.exists()) or inputs.force_restart:
                logger.info("Writing waveforms grid -> %s", pdf_path)
                _write_waveforms_grid_pdf(
                    waveforms_folder=concat_waveforms_dir,
                    pdf_path=pdf_path,
                    unit_ids=unit_ids,
                    segment_waveforms_folders=segment_waveforms_folders,
                    show_debug_annotation=False,
                    panel_dir=(panels_root / name),
                    pages_dir=(grids_dir / f"{name}_pages"),
                    write_page_png=True,
                    write_page_svg=True,
                    best_channel_mode=("old" if str(best_channel_mode) == "old" else "new"),
                )
            return pdf_path

        # Requested permutations.
        # Note: concat-only grids have no segments; old/new best-channel modes are equivalent.
        _write_perm(
            name="concat_uncurated",
            unit_ids=None,
            segment_waveforms_folders=None,
            best_channel_mode="old",
        )

        if curated_units_for_plot is not None:
            _write_perm(
                name="concat_curated",
                unit_ids=list(curated_units_for_plot),
                segment_waveforms_folders=None,
                best_channel_mode="old",
            )

        # Combined grids (concat + segments), using old vs new best-channel selection.
        _write_perm(
            name="uncurated_old_bestchan",
            unit_ids=None,
            segment_waveforms_folders=segment_folders,
            best_channel_mode="old",
        )
        waveforms_grid_pdf = _write_perm(
            name="uncurated_new_best_chan",
            unit_ids=None,
            segment_waveforms_folders=segment_folders,
            best_channel_mode="new",
        )

        if curated_units_for_plot is not None:
            _write_perm(
                name="curated_old_bestchan",
                unit_ids=list(curated_units_for_plot),
                segment_waveforms_folders=segment_folders,
                best_channel_mode="old",
            )
            _write_perm(
                name="curated_new_bestchan",
                unit_ids=list(curated_units_for_plot),
                segment_waveforms_folders=segment_folders,
                best_channel_mode="new",
            )

    return waveforms_grid_pdf, spikesorting_waveforms_grid_pdf
__all__ = [
    "_apply_waveforms_curation",
    "_plot_waveforms_outputs",
    "_load_spikesorting_quality_metrics",
]
