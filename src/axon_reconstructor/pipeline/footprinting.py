from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .checkpointing import (
    ProcessingStage,
    compute_checkpoint_file,
    exception_to_error_dict,
    load_checkpoint,
    save_checkpoint,
)
from .pipeline_logging import compute_pipeline_log_file, setup_pipeline_logger
from .pipeline_driver import _compute_mea_analysis_output_dir


FOOTPRINTING_OUTPUTS_DIRNAME = "footprinting_outputs"


def _infer_location_tolerance(locations: "Any") -> float:
    """Infer a reasonable coordinate-space tolerance for location matching.

    Channel locations are typically in µm for MEA data, but can be mm or meters.
    We use a 0.5 µm tolerance (scaled to the inferred units).
    """

    import numpy as np  # type: ignore[import-not-found]

    locs = np.asarray(locations)
    if locs.size == 0:
        return 0.0
    max_coord = float(np.nanmax(np.abs(locs)))
    # Heuristic matching the one used for electrode square sizing.
    if max_coord > 100.0:
        return 0.5
    if max_coord > 1.0:
        return 0.5 / 1000.0
    return 0.5e-6


def _try_get_recording_property(recording: Any, key: str):
    try:
        if hasattr(recording, "get_property_keys"):
            keys = set(recording.get_property_keys())
            if key not in keys:
                return None
        if hasattr(recording, "get_property"):
            return recording.get_property(key)
    except Exception:
        return None
    return None


def _try_get_electrode_ids(recording: Any):
    """Best-effort extraction of an 'electrode id' per channel.

    SpikeInterface recordings can carry various per-channel properties.
    We check a few common candidates; if none exist, returns None.
    """

    for key in (
        "electrode_id",
        "electrode",
        "contact_id",
        "contact_ids",
        "contact",
        "site_id",
        "site",
    ):
        vals = _try_get_recording_property(recording, key)
        if vals is not None:
            return vals
    return None


def _build_union_source_for_unit(
    *,
    sources: list[dict[str, Any]],
    unit_id: Any,
    logger,
) -> Optional[dict[str, Any]]:
    """Union channels across per-source footprints for a unit.

    If overlaps are detected (by channel_id, electrode_id, or location within tolerance),
    we warn and keep the first occurrence.
    """

    import numpy as np  # type: ignore[import-not-found]

    if not sources:
        return None

    # Determine location tolerance from all locations.
    all_locs = [np.asarray(s.get("channel_locations")) for s in sources if s.get("channel_locations") is not None]
    if not all_locs:
        return None
    stacked = np.concatenate(all_locs, axis=0)
    tol = float(_infer_location_tolerance(stacked))
    if tol <= 0:
        tol = 0.0

    def loc_key(x: float, y: float) -> tuple[int, int]:
        if tol <= 0:
            return (int(round(x * 1e6)), int(round(y * 1e6)))
        return (int(round(x / tol)), int(round(y / tol)))

    union_locs: list[list[float]] = []
    union_amp: list[float] = []
    union_channel_ids: list[Any] = []
    union_electrode_ids: list[Any] = []

    seen_channel_ids: set[Any] = set()
    seen_electrode_ids: set[Any] = set()
    seen_loc_keys: set[tuple[int, int]] = set()

    overlap_counts = {"channel_id": 0, "electrode_id": 0, "location": 0}

    for src in sources:
        locs = np.asarray(src.get("channel_locations"))
        amp = np.asarray(src.get("amp"))
        ch_ids = src.get("channel_ids")
        el_ids = src.get("electrode_ids")

        if locs.size == 0 or amp.size == 0 or locs.shape[0] != amp.shape[0]:
            continue

        # Optional: warn about duplicates within source.
        try:
            if ch_ids is not None:
                ch_list = list(ch_ids)
                if len(set(ch_list)) != len(ch_list):
                    logger.warning("Duplicate channel_ids within source %s (unit %s)", src.get("name"), unit_id)
        except Exception:
            pass

        for i in range(locs.shape[0]):
            x = float(locs[i, 0])
            y = float(locs[i, 1])
            lk = loc_key(x, y)

            cid = None
            if ch_ids is not None:
                try:
                    cid = ch_ids[i]
                except Exception:
                    cid = None

            eid = None
            if el_ids is not None:
                try:
                    eid = el_ids[i]
                except Exception:
                    eid = None

            # Overlap checks: if any overlap, keep the first and skip.
            if cid is not None and cid in seen_channel_ids:
                overlap_counts["channel_id"] += 1
                continue
            if eid is not None and eid in seen_electrode_ids:
                overlap_counts["electrode_id"] += 1
                continue
            if lk in seen_loc_keys:
                overlap_counts["location"] += 1
                continue

            seen_loc_keys.add(lk)
            if cid is not None:
                seen_channel_ids.add(cid)
            if eid is not None:
                seen_electrode_ids.add(eid)

            union_locs.append([x, y])
            union_amp.append(float(amp[i]))
            union_channel_ids.append(cid)
            union_electrode_ids.append(eid)

    total_overlaps = sum(overlap_counts.values())
    if total_overlaps:
        logger.warning(
            "Merged footprint overlaps for unit %s: %s (kept first, skipped the rest)",
            unit_id,
            overlap_counts,
        )

    if not union_locs:
        return None

    union_locs_arr = np.asarray(union_locs)
    union_amp_arr = np.asarray(union_amp)

    return {
        "name": "merged_union",
        "channel_locations": union_locs_arr,
        "amp": union_amp_arr,
        "best_ch": int(np.argmax(union_amp_arr)) if union_amp_arr.size else 0,
        "n_channels": int(union_locs_arr.shape[0]),
        "channel_ids": union_channel_ids,
        "electrode_ids": union_electrode_ids,
        "merge": {
            "location_tolerance": tol,
            "overlap_counts": overlap_counts,
        },
    }


def _ensure_analyzer_extensions(*, analyzer, extension_names: list[str], logger, n_jobs: int) -> None:
    missing = [name for name in extension_names if not analyzer.has_extension(name)]
    if not missing:
        return
    logger.info("Computing extensions: %s", ", ".join(missing))
    analyzer.compute(missing, verbose=False, n_jobs=max(1, int(n_jobs)))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _compute_footprinting_checkpoint_file(*, well_out_dir: Path, h5_path: Path, stream_id: str) -> Path:
    main_ckpt = compute_checkpoint_file(output_dir=well_out_dir, file_path=h5_path, stream_id=stream_id)
    name = main_ckpt.name
    if name.endswith("_checkpoint.json"):
        name = name[: -len("_checkpoint.json")] + "_footprinting_checkpoint.json"
    else:
        name = main_ckpt.stem + "_footprinting_checkpoint.json"
    return main_ckpt.with_name(name)


def _load_waveforms_analyzers(
    *,
    well_out_dir: Path,
    include_concat: bool,
    include_segments: bool,
    logger,
):
    """Load analyzers produced by the waveforms stage.

    Returns a list of (source_name, analyzer).
    """

    import spikeinterface.full as si  # type: ignore[import-not-found]

    waveforms_out_dir = well_out_dir / "waveforms_outputs"
    concat_waveforms_dir = waveforms_out_dir / "concat_waveforms"
    segment_waveforms_dir = waveforms_out_dir / "segment_waveforms"

    analyzers: list[tuple[str, Any]] = []

    if include_concat:
        if not concat_waveforms_dir.exists():
            raise FileNotFoundError(f"Missing concat waveforms analyzer at {concat_waveforms_dir}")
        logger.info("Loading concat analyzer: %s", concat_waveforms_dir)
        analyzers.append(("concat", si.load_sorting_analyzer(concat_waveforms_dir)))

    if include_segments and segment_waveforms_dir.exists():
        seg_dirs = sorted([p for p in segment_waveforms_dir.iterdir() if p.is_dir()])
        logger.info("Found %d segment analyzers", len(seg_dirs))
        for p in seg_dirs:
            try:
                analyzers.append((p.name, si.load_sorting_analyzer(p)))
            except Exception:
                logger.warning("Skipping unreadable segment analyzer: %s", p)

    if not analyzers:
        raise RuntimeError("No analyzers available for footprinting")

    return analyzers


def _get_unit_template_from_extension(*, analyzer, templates_ext, unit_id: Any):
    """Compatibility helper for SpikeInterface templates extension."""

    # When comparing across sources (concat vs segments), some units may be
    # absent in a particular analyzer. SpikeInterface will raise in that case;
    # treat it as a missing template instead.
    try:
        if hasattr(analyzer, "sorting") and hasattr(analyzer.sorting, "id_to_index"):
            analyzer.sorting.id_to_index(unit_id)
    except Exception:
        return None

    if hasattr(templates_ext, "get_unit_template"):
        try:
            return templates_ext.get_unit_template(unit_id=unit_id)
        except Exception:
            return None

    if hasattr(templates_ext, "get_templates"):
        try:
            all_templates = templates_ext.get_templates()
            unit_index = list(analyzer.sorting.unit_ids).index(unit_id)
            return all_templates[unit_index]
        except Exception:
            return None

    return None


def _write_footprints_grid_pdf(*, analyzer_folder: Path, pdf_path: Path, unit_ids: Optional[list[Any]] = None) -> None:
    """Write a multi-page PDF of per-unit footprints (single source).

    A footprint is shown as electrode locations colored by template peak-to-peak.
    """

    try:
        import logging

        import numpy as np  # type: ignore[import-not-found]
        import spikeinterface.full as si  # type: ignore[import-not-found]

        logging.getLogger("matplotlib").setLevel(logging.WARNING)
        logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)

        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        import matplotlib.backends.backend_pdf as pdf
        from matplotlib.colors import LogNorm

        # Intentionally no scalebar here: it tends to add clutter and can make
        # small scatter plots feel visually cramped.
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Plotting footprints requires numpy/matplotlib") from e

    analyzer = si.load_sorting_analyzer(analyzer_folder)
    _ensure_analyzer_extensions(analyzer=analyzer, extension_names=["templates"], logger=logging.getLogger(__name__), n_jobs=1)
    templates_ext = analyzer.get_extension("templates")

    if unit_ids is None:
        unit_ids = list(analyzer.sorting.unit_ids)

    locs = np.asarray(analyzer.recording.get_channel_locations())
    xs_all = locs[:, 0]
    ys_all = locs[:, 1]

    def _electrode_square_side_in_data_units(channel_locations: "np.ndarray", *, side_um: float = 17.5) -> float:
        """Return an electrode square side length in the same units as channel_locations.

        Maxwell/SpikeInterface channel locations are typically in µm, but may be in mm or m
        depending on provenance. We infer units by coordinate magnitude.
        """

        max_coord = float(np.nanmax(np.abs(channel_locations)))
        if max_coord > 100.0:
            # Likely µm (e.g., ~0..4000)
            return float(side_um)
        if max_coord > 1.0:
            # Likely mm (e.g., ~0..4)
            return float(side_um) / 1000.0
        # Likely meters (e.g., ~0..0.004)
        return float(side_um) * 1e-6

    def _square_marker_area_points2(ax, *, side_len: float) -> float:
        """Convert a square side length in data units to scatter 's' (points^2)."""

        p0 = ax.transData.transform((0.0, 0.0))
        p1 = ax.transData.transform((float(side_len), 0.0))
        dx_pixels = abs(float(p1[0]) - float(p0[0]))
        side_points = dx_pixels * 72.0 / float(ax.figure.dpi)
        return float(side_points * side_points)

    pad = 20.0
    xlim = (float(np.min(xs_all)) - pad, float(np.max(xs_all)) + pad)
    ylim = (float(np.min(ys_all)) - pad, float(np.max(ys_all)) + pad)

    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    units_per_page = 12
    n_rows = 4
    n_cols = 3
    square_side = _electrode_square_side_in_data_units(locs, side_um=17.5)
    with pdf.PdfPages(pdf_path) as pdf_doc:
        for i in range(0, len(unit_ids), units_per_page):
            batch = unit_ids[i : i + units_per_page]
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 12))
            axes = axes.flatten()

            # Dark background per-plot only (keep the overall page light).
            dark_bg = "#0b0b0b"
            cmap_name = "turbo"
            fig.patch.set_facecolor("white")

            # Make space for a colorbar without shrinking subplots unpredictably.
            fig.subplots_adjust(left=0.04, right=0.88, bottom=0.04, top=0.93, wspace=0.05, hspace=0.12)

            last_mappable = None
            marker_area = None
            # Simple log scaling: compute from all units on the page.
            amps_for_page: list[np.ndarray] = []
            for uid in batch:
                tmpl = _get_unit_template_from_extension(analyzer=analyzer, templates_ext=templates_ext, unit_id=uid)
                if tmpl is None:
                    continue
                tmpl = np.asarray(tmpl)
                if tmpl.ndim != 2 or tmpl.size == 0:
                    continue
                amps_for_page.append(np.ptp(tmpl, axis=0))

            norm = None
            norm_vmin_for_zeros = None
            if amps_for_page:
                amp_all = np.concatenate(amps_for_page)
                vmax = float(np.nanmax(amp_all))
                pos = amp_all[amp_all > 0]
                if pos.size and vmax > 0:
                    # LogNorm cannot represent non-positive values; map zeros to vmin.
                    vmin = float(np.nanmin(pos))
                    vmin = max(1.0, vmin)
                    # Avoid pathological vmin==vmax.
                    if vmin >= vmax:
                        vmin = vmax / 10.0
                    norm = LogNorm(vmin=vmin, vmax=vmax)
                    norm_vmin_for_zeros = float(vmin)

            for ax, uid in zip(axes, batch, strict=False):
                ax.set_facecolor(dark_bg)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
                for spine in ax.spines.values():
                    spine.set_visible(False)

                ax.set_xlim(*xlim)
                ax.set_ylim(*ylim)
                ax.set_aspect("equal", adjustable="box")

                if marker_area is None:
                    # Same limits/aspect across axes => same marker area.
                    marker_area = _square_marker_area_points2(ax, side_len=square_side)

                tmpl = _get_unit_template_from_extension(analyzer=analyzer, templates_ext=templates_ext, unit_id=uid)
                if tmpl is None:
                    ax.axis("off")
                    continue
                tmpl = np.asarray(tmpl)
                if tmpl.ndim != 2 or tmpl.size == 0:
                    ax.axis("off")
                    continue

                amp = np.ptp(tmpl, axis=0)
                amp_for_color = amp
                if norm is not None and norm_vmin_for_zeros is not None:
                    # Ensure ALL channels render under LogNorm (avoid masking/dropping zeros).
                    amp_for_color = np.where(amp <= 0, norm_vmin_for_zeros, amp)

                last_mappable = ax.scatter(
                    locs[:, 0],
                    locs[:, 1],
                    c=amp_for_color,
                    s=float(marker_area or 1.0),
                    marker="s",
                    cmap=cmap_name,
                    norm=norm,
                    linewidths=0,
                    edgecolors="none",
                    alpha=1.0,
                )
                last_mappable.set_rasterized(True)

                # Title text renders on the white page background, not the dark axes.
                ax.set_title(f"Unit {uid}", fontsize=10, color="black")



            for j in range(len(batch), len(axes)):
                axes[j].axis("off")

            fig.suptitle("Footprints (template PTP)", fontsize=12, color="black")

            if last_mappable is not None:
                try:
                    cax = fig.add_axes([0.90, 0.15, 0.02, 0.70])
                    cax.set_facecolor("white")
                    cbar = fig.colorbar(last_mappable, cax=cax)
                    cbar.set_label("Template PTP (µV)", fontsize=9, color="black")
                    cbar.ax.tick_params(labelsize=8, colors="black")
                    try:
                        cbar.outline.set_edgecolor("black")
                    except Exception:
                        pass
                except Exception:
                    pass

            # Keep raster dpi modest to avoid heavy memory use (WSL-friendly).
            pdf_doc.savefig(fig, dpi=300)
            plt.close(fig)


def _write_unit_footprints_across_sources_pdf(
    *,
    sources: list[dict[str, Any]],
    unit_id: Any,
    pdf_path: Path,
    logger,
) -> None:
    """Write a per-unit multi-page PDF showing footprints across sources."""

    try:
        import logging

        import numpy as np  # type: ignore[import-not-found]

        logging.getLogger("matplotlib").setLevel(logging.WARNING)
        logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)

        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        import matplotlib.backends.backend_pdf as pdf
        from matplotlib.colors import LogNorm

        # Intentionally no scalebar here (see _write_footprints_grid_pdf).
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Plotting multi-source footprints requires numpy/matplotlib") from e

    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    all_locs = [np.asarray(s["channel_locations"]) for s in sources if s.get("channel_locations") is not None]
    if not all_locs:
        return

    stacked = np.concatenate(all_locs, axis=0)
    xs_all = stacked[:, 0]
    ys_all = stacked[:, 1]

    pad = 20.0
    xlim = (float(np.min(xs_all)) - pad, float(np.max(xs_all)) + pad)
    ylim = (float(np.min(ys_all)) - pad, float(np.max(ys_all)) + pad)

    def _electrode_square_side_in_data_units(channel_locations: "np.ndarray", *, side_um: float = 17.5) -> float:
        max_coord = float(np.nanmax(np.abs(channel_locations)))
        if max_coord > 100.0:
            return float(side_um)
        if max_coord > 1.0:
            return float(side_um) / 1000.0
        return float(side_um) * 1e-6

    def _square_marker_area_points2(ax, *, side_len: float) -> float:
        p0 = ax.transData.transform((0.0, 0.0))
        p1 = ax.transData.transform((float(side_len), 0.0))
        dx_pixels = abs(float(p1[0]) - float(p0[0]))
        side_points = dx_pixels * 72.0 / float(ax.figure.dpi)
        return float(side_points * side_points)

    # Simple shared log scaling across sources for this unit.
    norm = None
    norm_vmin_for_zeros = None
    try:
        amp_all = np.concatenate([np.asarray(s["amp"]) for s in sources if s.get("amp") is not None])
        vmax = float(np.nanmax(amp_all))
        pos = amp_all[amp_all > 0]
        if pos.size and vmax > 0:
            vmin = float(np.nanmin(pos))
            vmin = max(1.0, vmin)
            if vmin >= vmax:
                vmin = vmax / 10.0
            norm = LogNorm(vmin=vmin, vmax=vmax)
            norm_vmin_for_zeros = float(vmin)
    except Exception:
        norm = None
        norm_vmin_for_zeros = None

    panels_per_page = 12
    n_rows = 4
    n_cols = 3
    square_side = _electrode_square_side_in_data_units(stacked, side_um=17.5)
    with pdf.PdfPages(pdf_path) as pdf_doc:
        for i in range(0, len(sources), panels_per_page):
            batch = sources[i : i + panels_per_page]
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 12))
            axes = axes.flatten()

            dark_bg = "#0b0b0b"
            cmap_name = "turbo"
            fig.patch.set_facecolor("white")

            fig.subplots_adjust(left=0.04, right=0.88, bottom=0.04, top=0.91, wspace=0.05, hspace=0.12)

            last_mappable = None
            marker_area = None
            for ax, src in zip(axes, batch, strict=False):
                ax.set_facecolor(dark_bg)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
                for spine in ax.spines.values():
                    spine.set_visible(False)

                ax.set_xlim(*xlim)
                ax.set_ylim(*ylim)
                ax.set_aspect("equal", adjustable="box")

                if marker_area is None:
                    marker_area = _square_marker_area_points2(ax, side_len=square_side)

                locs = np.asarray(src["channel_locations"])
                amp = np.asarray(src["amp"])
                amp_for_color = amp
                if norm is not None and norm_vmin_for_zeros is not None:
                    amp_for_color = np.where(amp <= 0, norm_vmin_for_zeros, amp)

                last_mappable = ax.scatter(
                    locs[:, 0],
                    locs[:, 1],
                    c=amp_for_color,
                    s=float(marker_area or 1.0),
                    marker="s",
                    cmap=cmap_name,
                    norm=norm,
                    linewidths=0,
                    edgecolors="none",
                    alpha=1.0,
                )
                last_mappable.set_rasterized(True)

                ax.set_title(
                    f"{src['name']} | n={int(src.get('n_channels', locs.shape[0]))}",
                    fontsize=10,
                    color="black",
                )



            for j in range(len(batch), len(axes)):
                axes[j].axis("off")

            fig.suptitle(
                f"Unit {unit_id} | Footprints across sources (log color scale)",
                fontsize=12,
                color="black",
            )

            if last_mappable is not None:
                try:
                    cax = fig.add_axes([0.90, 0.15, 0.02, 0.70])
                    cax.set_facecolor("white")
                    cbar = fig.colorbar(last_mappable, cax=cax)
                    cbar.set_label("Template PTP (µV)", fontsize=9, color="black")
                    cbar.ax.tick_params(labelsize=8, colors="black")
                    try:
                        cbar.outline.set_edgecolor("black")
                    except Exception:
                        pass
                except Exception:
                    pass

            pdf_doc.savefig(fig, dpi=300)
            plt.close(fig)

    logger.info("Wrote multi-source footprints PDF: %s", pdf_path)


@dataclass(frozen=True)
class FootprintingInputs:
    h5_path: Path
    stream_id: str
    mea_output_root: Path

    # Sources
    include_concat: bool = True
    include_segments: bool = True

    # Plotting
    plot_concat_footprints_grid_pdf: bool = True
    plot_multi_source_footprints_pdf: bool = True

    # Controls
    n_jobs: int = 8
    unit_limit: Optional[int] = None
    unit_ids: Optional[list[Any]] = None
    force_restart: bool = False


@dataclass(frozen=True)
class FootprintingOutputs:
    well_out_dir: Path
    templates_out_dir: Path
    footprinting_out_dir: Path

    concat_footprints_grid_pdf: Optional[Path]

    multi_source_footprints_dir: Optional[Path]
    multi_source_footprints_summary_json: Optional[Path]

    footprinting_summary_json: Path


def run_footprinting(*, inputs: FootprintingInputs, logger_name_prefix: str = "axon_reconstructor") -> FootprintingOutputs:
    """Compute/plot footprints from existing waveforms analyzers.

    This step is intentionally independent of template extraction/merging.
    It reads:
      <well>/waveforms_outputs/concat_waveforms/
      <well>/waveforms_outputs/segment_waveforms/*/

        And writes under:
            <well>/footprinting_outputs/
    """

    import numpy as np  # type: ignore[import-not-found]

    well_out_dir = _compute_mea_analysis_output_dir(
        output_root=inputs.mea_output_root,
        data_file=inputs.h5_path,
        well=inputs.stream_id,
    )

    log_file = compute_pipeline_log_file(
        well_out_dir=well_out_dir,
        data_file=inputs.h5_path,
        stream_id=inputs.stream_id,
    )
    logger = setup_pipeline_logger(
        log_file=log_file,
        logger_name=f"{logger_name_prefix}.{inputs.stream_id}.footprinting",
        verbose=True,
    )

    templates_out_dir = well_out_dir / "templates_outputs"
    footprinting_out_dir = well_out_dir / FOOTPRINTING_OUTPUTS_DIRNAME

    concat_grid_pdf = footprinting_out_dir / "footprints_grid_concat.pdf"
    multi_source_dir = footprinting_out_dir / "footprints_by_source"
    multi_source_summary_json = multi_source_dir / "footprints_by_source_summary.json"
    summary_json = footprinting_out_dir / "footprinting_summary.json"

    ckpt_file = _compute_footprinting_checkpoint_file(
        well_out_dir=well_out_dir,
        h5_path=inputs.h5_path,
        stream_id=inputs.stream_id,
    )
    ckpt = load_checkpoint(
        checkpoint_file=ckpt_file,
        force_restart=bool(inputs.force_restart),
        output_dir=well_out_dir,
        file_path=inputs.h5_path,
        stream_id=inputs.stream_id,
    )

    resume_ok = summary_json.exists()
    if inputs.plot_concat_footprints_grid_pdf:
        resume_ok = resume_ok and concat_grid_pdf.exists()
    if inputs.plot_multi_source_footprints_pdf:
        resume_ok = resume_ok and multi_source_summary_json.exists()

    if not inputs.force_restart and resume_ok:
        logger.info("Resuming footprinting: existing outputs found at %s", footprinting_out_dir)
        return FootprintingOutputs(
            well_out_dir=well_out_dir,
            templates_out_dir=templates_out_dir,
            footprinting_out_dir=footprinting_out_dir,
            concat_footprints_grid_pdf=(concat_grid_pdf if inputs.plot_concat_footprints_grid_pdf else None),
            multi_source_footprints_dir=(multi_source_dir if inputs.plot_multi_source_footprints_pdf else None),
            multi_source_footprints_summary_json=(
                multi_source_summary_json if inputs.plot_multi_source_footprints_pdf else None
            ),
            footprinting_summary_json=summary_json,
        )

    ckpt = save_checkpoint(
        checkpoint_file=ckpt_file,
        state=ckpt,
        stage=ProcessingStage.ANALYZER,
        failed_stage=None,
        error=None,
        extra_fields={"footprinting_out_dir": str(footprinting_out_dir)},
    )

    try:
        analyzers = _load_waveforms_analyzers(
            well_out_dir=well_out_dir,
            include_concat=bool(inputs.include_concat),
            include_segments=bool(inputs.include_segments),
            logger=logger,
        )

        for _, an in analyzers:
            _ensure_analyzer_extensions(analyzer=an, extension_names=["templates"], logger=logger, n_jobs=int(inputs.n_jobs))

        # Determine unit list from concat if present, else from first source.
        unit_ids: list[Any]
        if inputs.unit_ids is not None:
            unit_ids = list(inputs.unit_ids)
        else:
            unit_ids = list(analyzers[0][1].sorting.unit_ids)

        footprinting_out_dir.mkdir(parents=True, exist_ok=True)

        multi_source_summary: dict[str, Any] = {
            "h5_path": str(inputs.h5_path),
            "stream_id": inputs.stream_id,
            "well_out_dir": str(well_out_dir),
            "sources": [name for name, _ in analyzers],
            "units": [],
        }

        processed = 0
        for uid in unit_ids:
            # Gather sources where this unit has a template.
            sources_for_unit: list[dict[str, Any]] = []
            for name, an in analyzers:
                t_ext = an.get_extension("templates")
                tmpl_src = _get_unit_template_from_extension(analyzer=an, templates_ext=t_ext, unit_id=uid)
                if tmpl_src is None:
                    continue
                tmpl_src = np.asarray(tmpl_src)
                if tmpl_src.ndim != 2 or tmpl_src.size == 0:
                    continue
                locs_src = np.asarray(an.recording.get_channel_locations())
                ch_ids_src = None
                try:
                    ch_ids_src = np.asarray(an.recording.get_channel_ids())
                except Exception:
                    ch_ids_src = None

                el_ids_src = _try_get_electrode_ids(an.recording)
                if el_ids_src is not None:
                    try:
                        el_ids_src = np.asarray(el_ids_src)
                    except Exception:
                        el_ids_src = None

                if tmpl_src.shape[1] != locs_src.shape[0]:
                    continue

                amp = np.ptp(tmpl_src, axis=0)
                best_ch = int(np.argmax(amp))
                sources_for_unit.append(
                    {
                        "name": name,
                        "channel_locations": locs_src,
                        "amp": amp,
                        "best_ch": best_ch,
                        "n_channels": int(locs_src.shape[0]),
                        "channel_ids": ch_ids_src,
                        "electrode_ids": el_ids_src,
                    }
                )

            merged_union_src = _build_union_source_for_unit(sources=sources_for_unit, unit_id=uid, logger=logger)
            merged_sources_for_unit = (
                ([merged_union_src] + sources_for_unit) if merged_union_src is not None else sources_for_unit
            )

            multi_source_dir.mkdir(parents=True, exist_ok=True)
            pdf_path = multi_source_dir / f"unit_{uid}_footprints.pdf"

            unit_entry: dict[str, Any] = {
                "unit_id": int(uid) if str(uid).isdigit() else str(uid),
                "num_sources": int(len(sources_for_unit)),
                "sources": [s["name"] for s in sources_for_unit],
                "merged_union": (merged_union_src.get("merge") if merged_union_src is not None else None),
                "pdf_path": str(pdf_path) if merged_sources_for_unit else None,
                "error": None,
            }

            if inputs.plot_multi_source_footprints_pdf and merged_sources_for_unit:
                try:
                    _write_unit_footprints_across_sources_pdf(
                        sources=merged_sources_for_unit,
                        unit_id=uid,
                        pdf_path=pdf_path,
                        logger=logger,
                    )
                except Exception as e:
                    unit_entry["error"] = str(e)
                    unit_entry["pdf_path"] = None

            multi_source_summary["units"].append(unit_entry)

            processed += 1
            if inputs.unit_limit is not None and processed >= int(inputs.unit_limit):
                break

        if inputs.plot_multi_source_footprints_pdf:
            _write_json(multi_source_summary_json, multi_source_summary)

        if inputs.plot_concat_footprints_grid_pdf:
            # Prefer the concat analyzer folder when available.
            concat_folder = well_out_dir / "waveforms_outputs" / "concat_waveforms"
            if concat_folder.exists():
                _write_footprints_grid_pdf(analyzer_folder=concat_folder, pdf_path=concat_grid_pdf)
            else:
                # Fallback to first analyzer folder (rare).
                _write_footprints_grid_pdf(
                    analyzer_folder=well_out_dir / "waveforms_outputs" / "segment_waveforms" / analyzers[0][0],
                    pdf_path=concat_grid_pdf,
                )

        _write_json(
            summary_json,
            {
                "h5_path": str(inputs.h5_path),
                "stream_id": inputs.stream_id,
                "well_out_dir": str(well_out_dir),
                "footprinting_out_dir": str(footprinting_out_dir),
                "sources": [name for name, _ in analyzers],
                "concat_footprints_grid_pdf": str(concat_grid_pdf) if inputs.plot_concat_footprints_grid_pdf else None,
                "multi_source_footprints_dir": str(multi_source_dir) if inputs.plot_multi_source_footprints_pdf else None,
                "multi_source_footprints_summary_json": str(multi_source_summary_json)
                if inputs.plot_multi_source_footprints_pdf
                else None,
            },
        )

        ckpt = save_checkpoint(
            checkpoint_file=ckpt_file,
            state=ckpt,
            stage=ProcessingStage.ANALYZER_COMPLETE,
            failed_stage=None,
            error=None,
            extra_fields={
                "footprinting_out_dir": str(footprinting_out_dir),
                "concat_footprints_grid_pdf": str(concat_grid_pdf) if inputs.plot_concat_footprints_grid_pdf else None,
                "multi_source_footprints_dir": str(multi_source_dir) if inputs.plot_multi_source_footprints_pdf else None,
                "multi_source_footprints_summary_json": str(multi_source_summary_json)
                if inputs.plot_multi_source_footprints_pdf
                else None,
                "summary_json": str(summary_json),
            },
        )

        return FootprintingOutputs(
            well_out_dir=well_out_dir,
            templates_out_dir=templates_out_dir,
            footprinting_out_dir=footprinting_out_dir,
            concat_footprints_grid_pdf=(concat_grid_pdf if inputs.plot_concat_footprints_grid_pdf else None),
            multi_source_footprints_dir=(multi_source_dir if inputs.plot_multi_source_footprints_pdf else None),
            multi_source_footprints_summary_json=(multi_source_summary_json if inputs.plot_multi_source_footprints_pdf else None),
            footprinting_summary_json=summary_json,
        )

    except Exception as e:
        save_checkpoint(
            checkpoint_file=ckpt_file,
            state=ckpt,
            stage=ProcessingStage.ANALYZER,
            failed_stage=ProcessingStage.ANALYZER.name,
            error=exception_to_error_dict(e),
            extra_fields={"footprinting_out_dir": str(footprinting_out_dir)},
        )
        raise
