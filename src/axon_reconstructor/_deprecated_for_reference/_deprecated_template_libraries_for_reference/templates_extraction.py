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


def _write_templates_grid_pdf(*, analyzer_folder: Path, pdf_path: Path, unit_ids: Optional[list[Any]] = None) -> None:
    """Write a multi-page PDF of per-unit templates.

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
        raise RuntimeError("Plotting templates requires numpy/matplotlib") from e

    analyzer = si.load_sorting_analyzer(analyzer_folder)
    _ensure_analyzer_extensions(analyzer=analyzer, extension_names=["templates"], logger=logging.getLogger(__name__), n_jobs=1)
    t_ext = analyzer.get_extension("templates")

    if unit_ids is None:
        unit_ids = list(analyzer.sorting.unit_ids)

    sampling_frequency = float(analyzer.sampling_frequency)
    dt_ms = 1000.0 / sampling_frequency

    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    units_per_page = 12
    top_k = 15
    with pdf.PdfPages(pdf_path) as pdf_doc:
        for i in range(0, len(unit_ids), units_per_page):
            batch = unit_ids[i : i + units_per_page]
            fig, axes = plt.subplots(3, 4, figsize=(12, 9))
            axes = axes.flatten()

            for ax, uid in zip(axes, batch, strict=False):
                ax.set_xticks([])
                ax.set_yticks([])
                ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
                for spine in ax.spines.values():
                    spine.set_visible(False)

                tmpl = _get_unit_template_from_extension(analyzer=analyzer, templates_ext=t_ext, unit_id=uid)
                if tmpl is None:
                    ax.axis("off")
                    continue
                tmpl = np.asarray(tmpl)
                if tmpl.ndim != 2 or tmpl.size == 0:
                    ax.axis("off")
                    continue

                n_samples, n_channels = tmpl.shape
                times_ms = np.arange(n_samples) * dt_ms

                ptp = np.ptp(tmpl, axis=0)
                best_ch = int(np.argmax(ptp))
                order = np.argsort(ptp)[::-1]
                show = order[: min(top_k, n_channels)]

                # Plot in µV scale (data is already in µV in this pipeline).
                for ch in show:
                    ax.plot(times_ms, tmpl[:, ch], color="0.7", linewidth=0.7)
                ax.plot(times_ms, tmpl[:, best_ch], color="red", linewidth=1.0)

                ax.set_title(f"Unit {uid}", fontsize=10)

                if add_scalebar is not None:
                    try:
                        add_scalebar(
                            ax,
                            matchx=False,
                            matchy=False,
                            sizex=1.0,
                            labelx="1 ms",
                            sizey=50.0,
                            labely="50 µV",
                            loc=4,
                            hidex=True,
                            hidey=True,
                        )
                    except Exception:
                        pass

            for j in range(len(batch), len(axes)):
                axes[j].axis("off")

            pdf_doc.savefig(fig)
            plt.close(fig)


@dataclass(frozen=True)
class TemplateExtractionInputs:
    h5_path: Path
    stream_id: str
    mea_output_root: Path

    # Source analyzer
    include_concat: bool = True

    # Controls
    unit_ids: Optional[list[Any]] = None
    unit_limit: Optional[int] = None
    n_jobs: int = 8

    # Plotting
    plot_templates_grid_pdf: bool = True

    # Resume/overwrite
    force_restart: bool = False


@dataclass(frozen=True)
class TemplateExtractionOutputs:
    well_out_dir: Path
    templates_out_dir: Path
    extracted_templates_dir: Path
    summary_json: Path
    templates_grid_pdf: Optional[Path]


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

    if not inputs.force_restart and resume_ok:
        logger.info("Resuming template extraction: existing outputs found at %s", extracted_templates_dir)
        return TemplateExtractionOutputs(
            well_out_dir=well_out_dir,
            templates_out_dir=templates_out_dir,
            extracted_templates_dir=extracted_templates_dir,
            summary_json=summary_json,
            templates_grid_pdf=(templates_grid_pdf if inputs.plot_templates_grid_pdf else None),
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
        waveforms_out_dir = well_out_dir / "waveforms_outputs"
        concat_waveforms_dir = waveforms_out_dir / "concat_waveforms"

        if inputs.include_concat and not concat_waveforms_dir.exists():
            raise FileNotFoundError(f"Missing concat waveforms analyzer at {concat_waveforms_dir}")

        logger.info("Loading concat analyzer: %s", concat_waveforms_dir)
        analyzer = si.load_sorting_analyzer(concat_waveforms_dir)

        _ensure_analyzer_extensions(
            analyzer=analyzer,
            extension_names=["templates"],
            logger=logger,
            n_jobs=int(inputs.n_jobs),
        )

        t_ext = analyzer.get_extension("templates")
        channel_ids = np.asarray(list(analyzer.recording.get_channel_ids()))
        channel_locations = np.asarray(analyzer.recording.get_channel_locations())

        if inputs.unit_ids is not None:
            unit_ids = list(inputs.unit_ids)
        else:
            unit_ids = list(analyzer.sorting.unit_ids)

        extracted_templates_dir.mkdir(parents=True, exist_ok=True)

        summary: dict[str, Any] = {
            "h5_path": str(inputs.h5_path),
            "stream_id": inputs.stream_id,
            "well_out_dir": str(well_out_dir),
            "num_units": int(len(unit_ids)),
            "sampling_frequency": float(analyzer.sampling_frequency),
            "templates_grid_pdf": str(templates_grid_pdf) if inputs.plot_templates_grid_pdf else None,
            "units": [],
        }

        unit_count = 0
        for uid in unit_ids:
            tmpl = _get_unit_template_from_extension(
                analyzer=analyzer,
                templates_ext=t_ext,
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
            if inputs.unit_limit is not None and unit_count >= int(inputs.unit_limit):
                break

        _write_json(summary_json, summary)

        pdf_out = None
        if inputs.plot_templates_grid_pdf:
            logger.info("Writing templates grid PDF: %s", templates_grid_pdf)
            _write_templates_grid_pdf(analyzer_folder=concat_waveforms_dir, pdf_path=templates_grid_pdf)
            pdf_out = templates_grid_pdf

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
            },
        )

        return TemplateExtractionOutputs(
            well_out_dir=well_out_dir,
            templates_out_dir=templates_out_dir,
            extracted_templates_dir=extracted_templates_dir,
            summary_json=summary_json,
            templates_grid_pdf=pdf_out,
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
