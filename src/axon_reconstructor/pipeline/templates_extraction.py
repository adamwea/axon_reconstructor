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


TEMPLATES_OUTPUTS_DIRNAME = "templates_outputs"


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


def _compute_templates_extract_checkpoint_file(*, well_out_dir: Path, h5_path: Path, stream_id: str) -> Path:
    main_ckpt = compute_checkpoint_file(output_dir=well_out_dir, file_path=h5_path, stream_id=stream_id)
    name = main_ckpt.name
    if name.endswith("_checkpoint.json"):
        name = name[: -len("_checkpoint.json")] + "_templates_extract_checkpoint.json"
    else:
        name = main_ckpt.stem + "_templates_extract_checkpoint.json"
    return main_ckpt.with_name(name)


def _write_templates_grid_pdf(*, analyzer_folder: Path, pdf_path: Path, unit_ids: Optional[list[Any]] = None) -> None:
    """Write a multi-page PDF of per-unit templates.

    Style matches the waveforms grid QC (axes removed, scalebar in lower-right).
    Each panel shows the top-K channels (by peak-to-peak) in gray, with the best
    channel highlighted in red.
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

        try:
            from scalebury import add_scalebar  # type: ignore[import-not-found]
        except Exception:
            try:
                from MEA_Analysis.IPNAnalysis.scalebury import add_scalebar  # type: ignore[import-not-found]
            except Exception:
                add_scalebar = None
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Plotting templates grid requires numpy/matplotlib/spikeinterface") from e

    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    analyzer = si.load_sorting_analyzer(analyzer_folder)
    if not analyzer.has_extension("templates"):
        analyzer.compute(["templates"], n_jobs=1, verbose=False)
    t_ext = analyzer.get_extension("templates")

    if unit_ids is None:
        unit_ids = list(analyzer.sorting.unit_ids)
    if len(unit_ids) == 0:
        return

    fs = float(analyzer.sampling_frequency)
    channel_ids = list(analyzer.recording.get_channel_ids())

    with pdf.PdfPages(pdf_path) as pdf_doc:
        units_per_page = 12
        for i in range(0, len(unit_ids), units_per_page):
            batch = unit_ids[i : i + units_per_page]
            fig, axes = plt.subplots(3, 4, figsize=(12, 9))
            axes = axes.flatten()

            for ax, uid in zip(axes, batch, strict=False):
                try:
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
                    for spine in ax.spines.values():
                        spine.set_visible(False)

                    # SpikeInterface 0.103.x: ComputeTemplates provides get_unit_template().
                    # Older versions used different method names.
                    if hasattr(t_ext, "get_unit_template"):
                        tmpl = t_ext.get_unit_template(unit_id=uid)
                    else:
                        # Fallback: try vectorized templates array.
                        tmpl = None
                        if hasattr(t_ext, "get_templates"):
                            all_templates = t_ext.get_templates()
                            try:
                                unit_index = list(analyzer.sorting.unit_ids).index(uid)
                                tmpl = all_templates[unit_index]
                            except Exception:
                                tmpl = None
                    if tmpl is None:
                        ax.axis("off")
                        continue

                    tmpl = np.asarray(tmpl)
                    if tmpl.ndim != 2 or tmpl.shape[0] == 0 or tmpl.shape[1] == 0:
                        ax.axis("off")
                        continue

                    # Choose channels to show.
                    ptp = np.ptp(tmpl, axis=0)
                    best_ch = int(np.argmax(ptp))
                    top_k = 6
                    top_idx = np.argsort(ptp)[::-1][:top_k]

                    time_ms = np.arange(tmpl.shape[0]) / fs * 1000

                    # Plot top channels in gray (thin), best channel in red.
                    for ch in top_idx:
                        ax.plot(time_ms, tmpl[:, int(ch)], c="gray", lw=0.6, alpha=0.35)
                    ax.plot(time_ms, tmpl[:, best_ch], c="red", lw=1.5)

                    ch_label = best_ch
                    try:
                        if best_ch < len(channel_ids):
                            ch_label = int(channel_ids[best_ch])
                    except Exception:
                        pass

                    ax.set_title(f"Unit {uid} | Ch {ch_label}", fontsize=10)

                    if add_scalebar is not None:
                        try:
                            add_scalebar(
                                ax,
                                matchx=False,
                                matchy=False,
                                sizex=1.0,
                                labelx="1 ms",
                                sizey=50,
                                labely="50 µV",
                                loc=4,
                                hidex=True,
                                hidey=True,
                            )
                        except Exception:
                            pass
                except Exception:
                    ax.axis("off")

            for j in range(len(batch), len(axes)):
                axes[j].axis("off")

            pdf_doc.savefig(fig)
            plt.close(fig)


def _write_footprints_grid_pdf(
    *,
    analyzer_folder: Path,
    pdf_path: Path,
    unit_ids: Optional[list[Any]] = None,
) -> None:
    """Write a multi-page PDF of per-unit spatial footprints.

    For each unit, plot channel locations (x,y) colored by a simple amplitude
    summary derived from the unit template (peak-to-peak per channel).

    This is meant as a quick QC view *before* any merging step.
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

        try:
            from scalebury import add_scalebar  # type: ignore[import-not-found]
        except Exception:
            try:
                from MEA_Analysis.IPNAnalysis.scalebury import add_scalebar  # type: ignore[import-not-found]
            except Exception:
                add_scalebar = None
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Plotting footprints grid requires numpy/matplotlib/spikeinterface") from e

    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    analyzer = si.load_sorting_analyzer(analyzer_folder)
    if not analyzer.has_extension("templates"):
        analyzer.compute(["templates"], n_jobs=1, verbose=False)
    t_ext = analyzer.get_extension("templates")

    if unit_ids is None:
        unit_ids = list(analyzer.sorting.unit_ids)
    if len(unit_ids) == 0:
        return

    locs = np.asarray(analyzer.recording.get_channel_locations())
    if locs.ndim != 2 or locs.shape[1] < 2:
        raise ValueError(f"Unexpected channel_locations shape: {locs.shape}")
    xs = locs[:, 0]
    ys = locs[:, 1]

    # Stable bounds so panels are comparable.
    pad = 20.0
    xlim = (float(np.min(xs)) - pad, float(np.max(xs)) + pad)
    ylim = (float(np.min(ys)) - pad, float(np.max(ys)) + pad)

    with pdf.PdfPages(pdf_path) as pdf_doc:
        units_per_page = 12
        for i in range(0, len(unit_ids), units_per_page):
            batch = unit_ids[i : i + units_per_page]
            fig, axes = plt.subplots(3, 4, figsize=(12, 9))
            axes = axes.flatten()

            for ax, uid in zip(axes, batch, strict=False):
                try:
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
                    for spine in ax.spines.values():
                        spine.set_visible(False)

                    ax.set_xlim(*xlim)
                    ax.set_ylim(*ylim)
                    ax.set_aspect("equal", adjustable="box")

                    # Template -> per-channel amplitude summary.
                    if hasattr(t_ext, "get_unit_template"):
                        tmpl = t_ext.get_unit_template(unit_id=uid)
                    else:
                        tmpl = None
                        if hasattr(t_ext, "get_templates"):
                            all_templates = t_ext.get_templates()
                            try:
                                unit_index = list(analyzer.sorting.unit_ids).index(uid)
                                tmpl = all_templates[unit_index]
                            except Exception:
                                tmpl = None
                    if tmpl is None:
                        ax.axis("off")
                        continue

                    tmpl = np.asarray(tmpl)
                    if tmpl.ndim != 2 or tmpl.shape[1] != locs.shape[0]:
                        ax.axis("off")
                        continue

                    amp = np.ptp(tmpl, axis=0)
                    best_ch = int(np.argmax(amp))

                    sc = ax.scatter(
                        xs,
                        ys,
                        c=amp,
                        s=8,
                        cmap="viridis",
                        linewidths=0,
                        alpha=0.95,
                    )
                    ax.scatter(
                        [xs[best_ch]],
                        [ys[best_ch]],
                        s=40,
                        facecolors="none",
                        edgecolors="red",
                        linewidths=1.2,
                    )

                    ax.set_title(f"Unit {uid}", fontsize=10)

                    # Small colorbar per-panel is noisy; instead add a tiny one only on first panel each page.
                    if ax is axes[0]:
                        try:
                            cbar = fig.colorbar(sc, ax=axes.tolist(), shrink=0.6, pad=0.01)
                            cbar.set_label("Template PTP (µV)", fontsize=9)
                            cbar.ax.tick_params(labelsize=8)
                        except Exception:
                            pass

                    # Optional spatial scalebar.
                    if add_scalebar is not None:
                        try:
                            add_scalebar(
                                ax,
                                matchx=False,
                                matchy=False,
                                sizex=100.0,
                                labelx="100 µm",
                                sizey=100.0,
                                labely="100 µm",
                                loc=4,
                                hidex=True,
                                hidey=True,
                            )
                        except Exception:
                            pass
                except Exception:
                    ax.axis("off")

            for j in range(len(batch), len(axes)):
                axes[j].axis("off")

            pdf_doc.savefig(fig)
            plt.close(fig)


def _load_waveforms_analyzers(
    *,
    well_out_dir: Path,
    include_concat: bool,
    include_segments: bool,
    logger,
):
    """Load analyzers produced by the waveforms stage.

    Returns a list of (source_name, analyzer, templates_extension).
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
        raise RuntimeError("No analyzers available for template extraction")

    # Ensure templates extension exists everywhere.
    for name, an in analyzers:
        _ensure_analyzer_extensions(
            analyzer=an,
            extension_names=["templates"],
            logger=logger,
            n_jobs=1,
        )

    return analyzers


def _get_unit_template_from_extension(*, analyzer, templates_ext, unit_id: Any):
    """Compatibility helper for SpikeInterface templates extension."""

    if hasattr(templates_ext, "get_unit_template"):
        return templates_ext.get_unit_template(unit_id=unit_id)

    if hasattr(templates_ext, "get_templates"):
        try:
            all_templates = templates_ext.get_templates()
            unit_index = list(analyzer.sorting.unit_ids).index(unit_id)
            return all_templates[unit_index]
        except Exception:
            return None

    return None


def _write_unit_footprints_across_sources_pdf(
    *,
    sources: list[dict[str, Any]],
    unit_id: Any,
    pdf_path: Path,
    logger,
) -> None:
    """Write a per-unit multi-page PDF showing footprints across sources.

    `sources` entries contain:
      - name
      - channel_locations (n,2)
      - amp (n,) amplitude summary for that source
      - best_ch (int)
    """

    try:
        import logging

        import numpy as np  # type: ignore[import-not-found]

        logging.getLogger("matplotlib").setLevel(logging.WARNING)
        logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)

        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        import matplotlib.backends.backend_pdf as pdf

        try:
            from scalebury import add_scalebar  # type: ignore[import-not-found]
        except Exception:
            try:
                from MEA_Analysis.IPNAnalysis.scalebury import add_scalebar  # type: ignore[import-not-found]
            except Exception:
                add_scalebar = None
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Plotting multi-source footprints requires numpy/matplotlib") from e

    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    # Compute stable bounds from union of locations across sources.
    all_locs = [np.asarray(s["channel_locations"]) for s in sources if s.get("channel_locations") is not None]
    if not all_locs:
        return
    stacked = np.concatenate(all_locs, axis=0)
    xs_all = stacked[:, 0]
    ys_all = stacked[:, 1]

    pad = 20.0
    xlim = (float(np.min(xs_all)) - pad, float(np.max(xs_all)) + pad)
    ylim = (float(np.min(ys_all)) - pad, float(np.max(ys_all)) + pad)

    # Shared color scaling per unit.
    vmax = 0.0
    for s in sources:
        amp = s.get("amp")
        if amp is None:
            continue
        try:
            vmax = max(vmax, float(np.nanmax(np.asarray(amp))))
        except Exception:
            continue
    if vmax <= 0:
        vmax = None

    units_per_page = 12
    with pdf.PdfPages(pdf_path) as pdf_doc:
        for i in range(0, len(sources), units_per_page):
            batch = sources[i : i + units_per_page]
            fig, axes = plt.subplots(3, 4, figsize=(12, 9))
            axes = axes.flatten()

            last_sc = None
            for ax, src in zip(axes, batch, strict=False):
                try:
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
                    for spine in ax.spines.values():
                        spine.set_visible(False)

                    ax.set_xlim(*xlim)
                    ax.set_ylim(*ylim)
                    ax.set_aspect("equal", adjustable="box")

                    locs = np.asarray(src["channel_locations"])
                    xs = locs[:, 0]
                    ys = locs[:, 1]
                    amp = np.asarray(src["amp"])
                    best_ch = int(src["best_ch"])

                    last_sc = ax.scatter(
                        xs,
                        ys,
                        c=amp,
                        s=8,
                        cmap="viridis",
                        vmin=0.0,
                        vmax=vmax,
                        linewidths=0,
                        alpha=0.95,
                    )
                    if 0 <= best_ch < len(xs):
                        ax.scatter(
                            [xs[best_ch]],
                            [ys[best_ch]],
                            s=40,
                            facecolors="none",
                            edgecolors="red",
                            linewidths=1.2,
                        )

                    ax.set_title(f"{src['name']} | n={int(src.get('n_channels', len(xs)))}", fontsize=10)

                    if add_scalebar is not None:
                        try:
                            add_scalebar(
                                ax,
                                matchx=False,
                                matchy=False,
                                sizex=100.0,
                                labelx="100 µm",
                                sizey=100.0,
                                labely="100 µm",
                                loc=4,
                                hidex=True,
                                hidey=True,
                            )
                        except Exception:
                            pass
                except Exception:
                    ax.axis("off")

            for j in range(len(batch), len(axes)):
                axes[j].axis("off")

            fig.suptitle(f"Unit {unit_id} | Footprints across sources", fontsize=12)

            # One shared colorbar per page if possible.
            if last_sc is not None:
                try:
                    cbar = fig.colorbar(last_sc, ax=axes.tolist(), shrink=0.6, pad=0.01)
                    cbar.set_label("Template PTP (µV)", fontsize=9)
                    cbar.ax.tick_params(labelsize=8)
                except Exception:
                    pass

            pdf_doc.savefig(fig)
            plt.close(fig)

    logger.info("Wrote multi-source footprints PDF: %s", pdf_path)


@dataclass(frozen=True)
class TemplateExtractionInputs:
    h5_path: Path
    stream_id: str
    mea_output_root: Path

    # Source analyzer
    include_concat: bool = True
    include_segments: bool = True

    # Controls
    unit_ids: Optional[list[Any]] = None
    unit_limit: Optional[int] = None
    n_jobs: int = 8

    # Plotting
    plot_templates_grid_pdf: bool = True
    plot_footprints_grid_pdf: bool = True
    plot_multi_source_footprints_pdf: bool = True

    # Resume/overwrite
    force_restart: bool = False


@dataclass(frozen=True)
class TemplateExtractionOutputs:
    well_out_dir: Path
    templates_out_dir: Path
    extracted_templates_dir: Path
    summary_json: Path
    templates_grid_pdf: Optional[Path]
    footprints_grid_pdf: Optional[Path]
    multi_source_footprints_dir: Optional[Path]
    multi_source_footprints_summary_json: Optional[Path]


def extract_templates(
    *,
    inputs: TemplateExtractionInputs,
    logger_name_prefix: str = "axon_reconstructor",
) -> TemplateExtractionOutputs:
    """Extract templates for each unit (no merging).

    Loads the concat waveforms analyzer produced by the waveforms stage and saves:
      - templates_outputs/extracted_templates/<unit>.npy
      - templates_outputs/extracted_templates/<unit>_channel_ids.npy
      - templates_outputs/extracted_templates/<unit>_channel_locations.npy
      - templates_outputs/template_extraction_summary.json
      - templates_outputs/templates_grid_uncurated.pdf (optional)
    """

    import numpy as np  # type: ignore[import-not-found]
    import spikeinterface.full as si  # type: ignore[import-not-found]

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
        logger_name=f"{logger_name_prefix}.{inputs.stream_id}.templates_extract",
        verbose=True,
    )

    templates_out_dir = well_out_dir / TEMPLATES_OUTPUTS_DIRNAME
    extracted_templates_dir = templates_out_dir / "extracted_templates"
    summary_json = templates_out_dir / "template_extraction_summary.json"
    templates_grid_pdf = templates_out_dir / "templates_grid_uncurated.pdf"
    footprints_grid_pdf = templates_out_dir / "footprints_grid_uncurated.pdf"
    multi_source_footprints_dir = templates_out_dir / "footprints_by_source"
    multi_source_footprints_summary_json = multi_source_footprints_dir / "footprints_by_source_summary.json"

    ckpt_file = _compute_templates_extract_checkpoint_file(
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

    resume_ok = extracted_templates_dir.exists() and summary_json.exists()
    if inputs.plot_templates_grid_pdf:
        resume_ok = resume_ok and templates_grid_pdf.exists()
    if inputs.plot_footprints_grid_pdf:
        resume_ok = resume_ok and footprints_grid_pdf.exists()
    if inputs.plot_multi_source_footprints_pdf:
        resume_ok = resume_ok and multi_source_footprints_summary_json.exists()

    if not inputs.force_restart and resume_ok:
        logger.info("Resuming template extraction: existing outputs found at %s", extracted_templates_dir)
        return TemplateExtractionOutputs(
            well_out_dir=well_out_dir,
            templates_out_dir=templates_out_dir,
            extracted_templates_dir=extracted_templates_dir,
            summary_json=summary_json,
            templates_grid_pdf=(templates_grid_pdf if inputs.plot_templates_grid_pdf else None),
            footprints_grid_pdf=(footprints_grid_pdf if inputs.plot_footprints_grid_pdf else None),
            multi_source_footprints_dir=(multi_source_footprints_dir if inputs.plot_multi_source_footprints_pdf else None),
            multi_source_footprints_summary_json=(
                multi_source_footprints_summary_json if inputs.plot_multi_source_footprints_pdf else None
            ),
        )

    ckpt = save_checkpoint(
        checkpoint_file=ckpt_file,
        state=ckpt,
        stage=ProcessingStage.ANALYZER,
        failed_stage=None,
        error=None,
        extra_fields={"templates_out_dir": str(templates_out_dir)},
    )

    try:
        analyzers = _load_waveforms_analyzers(
            well_out_dir=well_out_dir,
            include_concat=bool(inputs.include_concat),
            include_segments=bool(inputs.include_segments),
            logger=logger,
        )

        # Use concat analyzer as the primary extraction source (for per-unit template files).
        concat_name, concat_analyzer = analyzers[0]
        if concat_name != "concat" and inputs.include_concat:
            logger.warning("Expected first analyzer to be concat; got %s", concat_name)

        concat_templates_ext = concat_analyzer.get_extension("templates")
        channel_ids = np.asarray(list(concat_analyzer.recording.get_channel_ids()))
        channel_locations = np.asarray(concat_analyzer.recording.get_channel_locations())

        if inputs.unit_ids is not None:
            unit_ids = list(inputs.unit_ids)
        else:
            unit_ids = list(concat_analyzer.sorting.unit_ids)

        extracted_templates_dir.mkdir(parents=True, exist_ok=True)

        summary: dict[str, Any] = {
            "h5_path": str(inputs.h5_path),
            "stream_id": inputs.stream_id,
            "well_out_dir": str(well_out_dir),
            "sources": [name for name, _ in analyzers],
            "num_units": int(len(unit_ids)),
            "sampling_frequency": float(concat_analyzer.sampling_frequency),
            "templates_grid_pdf": str(templates_grid_pdf) if inputs.plot_templates_grid_pdf else None,
            "footprints_grid_pdf": str(footprints_grid_pdf) if inputs.plot_footprints_grid_pdf else None,
            "multi_source_footprints_dir": str(multi_source_footprints_dir)
            if inputs.plot_multi_source_footprints_pdf
            else None,
            "units": [],
        }

        unit_count = 0
        multi_source_summary: dict[str, Any] = {
            "h5_path": str(inputs.h5_path),
            "stream_id": inputs.stream_id,
            "well_out_dir": str(well_out_dir),
            "sources": [name for name, _ in analyzers],
            "units": [],
        }

        for uid in unit_ids:
            tmpl = _get_unit_template_from_extension(
                analyzer=concat_analyzer,
                templates_ext=concat_templates_ext,
                unit_id=uid,
            )
            if tmpl is None:
                continue
            tmpl = np.asarray(tmpl)
            if tmpl.ndim != 2 or tmpl.size == 0:
                continue

            unit_template_file = extracted_templates_dir / f"{uid}.npy"
            unit_channel_ids_file = extracted_templates_dir / f"{uid}_channel_ids.npy"
            unit_channel_locs_file = extracted_templates_dir / f"{uid}_channel_locations.npy"

            np.save(unit_template_file, tmpl)
            np.save(unit_channel_ids_file, channel_ids)
            np.save(unit_channel_locs_file, channel_locations)

            ptp = np.ptp(tmpl, axis=0)
            best_ch = int(np.argmax(ptp))
            best_ch_label = None
            try:
                best_ch_label = int(channel_ids[best_ch])
            except Exception:
                best_ch_label = int(best_ch)

            summary["units"].append(
                {
                    "unit_id": int(uid) if str(uid).isdigit() else str(uid),
                    "template_path": str(unit_template_file),
                    "channel_ids_path": str(unit_channel_ids_file),
                    "channel_locations_path": str(unit_channel_locs_file),
                    "template_shape": [int(tmpl.shape[0]), int(tmpl.shape[1])],
                    "best_channel_index": int(best_ch),
                    "best_channel_id": best_ch_label,
                }
            )

            unit_count += 1
            # Multi-source footprints for this unit (concat + segments where present).
            if inputs.plot_multi_source_footprints_pdf:
                sources_for_unit: list[dict[str, Any]] = []
                for name, an in analyzers:
                    t_ext = an.get_extension("templates")
                    tmpl_src = _get_unit_template_from_extension(analyzer=an, templates_ext=t_ext, unit_id=uid)
                    if tmpl_src is None:
                        continue
                    tmpl_src = np.asarray(tmpl_src)
                    if tmpl_src.ndim != 2 or tmpl_src.size == 0:
                        continue
                    try:
                        locs_src = np.asarray(an.recording.get_channel_locations())
                    except Exception:
                        continue
                    if tmpl_src.shape[1] != locs_src.shape[0]:
                        # Skip mismatched shapes (shouldn't happen, but keep QC robust).
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
                        }
                    )

                multi_source_footprints_dir.mkdir(parents=True, exist_ok=True)
                pdf_path = multi_source_footprints_dir / f"unit_{uid}_footprints.pdf"

                unit_entry: dict[str, Any] = {
                    "unit_id": int(uid) if str(uid).isdigit() else str(uid),
                    "num_sources": int(len(sources_for_unit)),
                    "sources": [s["name"] for s in sources_for_unit],
                    "pdf_path": str(pdf_path),
                    "error": None,
                }

                if not sources_for_unit:
                    unit_entry["pdf_path"] = None
                    multi_source_summary["units"].append(unit_entry)
                else:
                    try:
                        _write_unit_footprints_across_sources_pdf(
                            sources=sources_for_unit,
                            unit_id=uid,
                            pdf_path=pdf_path,
                            logger=logger,
                        )
                    except Exception as e:
                        unit_entry["error"] = str(e)
                        unit_entry["pdf_path"] = None

                    multi_source_summary["units"].append(unit_entry)

            if inputs.unit_limit is not None and unit_count >= int(inputs.unit_limit):
                break

        _write_json(summary_json, summary)

        if inputs.plot_multi_source_footprints_pdf:
            multi_source_footprints_dir.mkdir(parents=True, exist_ok=True)

            # Fallback: if for any reason the in-memory list stayed empty, but PDFs
            # exist on disk, populate the summary from the filesystem.
            if not multi_source_summary.get("units"):
                pdfs = sorted(multi_source_footprints_dir.glob("unit_*_footprints.pdf"))
                if pdfs:
                    inferred_units: list[dict[str, Any]] = []
                    for p in pdfs:
                        name = p.stem
                        uid_part = name.removeprefix("unit_").removesuffix("_footprints")
                        try:
                            uid_val: Any = int(uid_part)
                        except Exception:
                            uid_val = uid_part
                        inferred_units.append(
                            {
                                "unit_id": uid_val,
                                "num_sources": None,
                                "sources": None,
                                "pdf_path": str(p),
                                "error": None,
                            }
                        )
                    multi_source_summary["units"] = inferred_units

            _write_json(multi_source_footprints_summary_json, multi_source_summary)

        pdf_out = None
        if inputs.plot_templates_grid_pdf:
            logger.info("Writing templates grid PDF: %s", templates_grid_pdf)
            # Use concat analyzer folder for templates QC grid.
            _write_templates_grid_pdf(
                analyzer_folder=well_out_dir / "waveforms_outputs" / "concat_waveforms",
                pdf_path=templates_grid_pdf,
            )
            pdf_out = templates_grid_pdf

        footprints_out = None
        if inputs.plot_footprints_grid_pdf:
            logger.info("Writing footprints grid PDF: %s", footprints_grid_pdf)
            _write_footprints_grid_pdf(
                analyzer_folder=well_out_dir / "waveforms_outputs" / "concat_waveforms",
                pdf_path=footprints_grid_pdf,
            )
            footprints_out = footprints_grid_pdf

        ckpt = save_checkpoint(
            checkpoint_file=ckpt_file,
            state=ckpt,
            stage=ProcessingStage.ANALYZER_COMPLETE,
            failed_stage=None,
            error=None,
            extra_fields={
                "templates_out_dir": str(templates_out_dir),
                "extracted_templates_dir": str(extracted_templates_dir),
                "summary_json": str(summary_json),
                "templates_grid_pdf": str(templates_grid_pdf) if pdf_out else None,
                "footprints_grid_pdf": str(footprints_grid_pdf) if footprints_out else None,
                "multi_source_footprints_dir": str(multi_source_footprints_dir)
                if inputs.plot_multi_source_footprints_pdf
                else None,
                "multi_source_footprints_summary_json": str(multi_source_footprints_summary_json)
                if inputs.plot_multi_source_footprints_pdf
                else None,
            },
        )

        return TemplateExtractionOutputs(
            well_out_dir=well_out_dir,
            templates_out_dir=templates_out_dir,
            extracted_templates_dir=extracted_templates_dir,
            summary_json=summary_json,
            templates_grid_pdf=pdf_out,
            footprints_grid_pdf=footprints_out,
            multi_source_footprints_dir=(multi_source_footprints_dir if inputs.plot_multi_source_footprints_pdf else None),
            multi_source_footprints_summary_json=(
                multi_source_footprints_summary_json if inputs.plot_multi_source_footprints_pdf else None
            ),
        )

    except Exception as e:
        save_checkpoint(
            checkpoint_file=ckpt_file,
            state=ckpt,
            stage=ProcessingStage.ANALYZER,
            failed_stage=ProcessingStage.ANALYZER.name,
            error=exception_to_error_dict(e),
            extra_fields=None,
        )
        raise
