from __future__ import annotations

import io
import json
import logging
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
from ..pipeline_driver import _compute_mea_analysis_output_dir
from ..pipeline_logging import compute_pipeline_log_file, setup_pipeline_logger

from .constants import ANALYSIS_OUTPUTS_DIRNAME


def _read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _compute_analysis_checkpoint_file(*, well_out_dir: Path, h5_path: Path, stream_id: str) -> Path:
    main_ckpt = compute_checkpoint_file(output_dir=well_out_dir, file_path=h5_path, stream_id=stream_id)
    name = main_ckpt.name
    if name.endswith("_checkpoint.json"):
        name = name[: -len("_checkpoint.json")] + "_analysis_checkpoint.json"
    else:
        name = main_ckpt.stem + "_analysis_checkpoint.json"
    return main_ckpt.with_name(name)


def _try_load_image_rgba(*, path: Path, logger: logging.Logger) -> tuple[Optional[Any], Optional[str]]:
    """Load an image as an RGBA numpy array (best effort).

    Supports PNG/JPG via Pillow.
    Supports SVG via cairosvg -> Pillow if cairosvg is installed.

    Returns (rgba_array, error_message).
    """

    path = Path(path)
    if not path.exists():
        return None, f"missing: {path.name}"

    suffix = path.suffix.lower()

    try:
        import numpy as np  # type: ignore[import-not-found]
        from PIL import Image  # type: ignore[import-not-found]
    except Exception as e:
        return None, f"missing plotting deps (numpy/pillow): {e}"

    try:
        if suffix == ".svg":
            try:
                import cairosvg  # type: ignore[import-not-found]

                png_bytes = cairosvg.svg2png(url=str(path))
                img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
                return np.asarray(img), None
            except Exception as e:
                logger.debug("SVG load failed for %s: %s", path, e)
                return None, "svg load failed (install cairosvg to render svg)"

        img = Image.open(path).convert("RGBA")
        return np.asarray(img), None
    except Exception as e:
        logger.debug("Image load failed for %s: %s", path, e)
        return None, f"load failed: {e}"


def _try_load_image_pil(*, path: Path, logger: logging.Logger) -> tuple[Optional[Any], Optional[str]]:
    """Load an image as a PIL Image in RGBA mode (best effort)."""

    path = Path(path)
    if not path.exists():
        return None, f"missing: {path.name}"

    try:
        from PIL import Image  # type: ignore[import-not-found]
    except Exception as e:
        return None, f"missing plotting deps (pillow): {e}"

    suffix = path.suffix.lower()
    try:
        if suffix == ".svg":
            try:
                import cairosvg  # type: ignore[import-not-found]

                png_bytes = cairosvg.svg2png(url=str(path))
                img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
                return img, None
            except Exception as e:
                logger.debug("SVG load failed for %s: %s", path, e)
                return None, "svg load failed (install cairosvg to render svg)"

        img = Image.open(path).convert("RGBA")
        return img, None
    except Exception as e:
        logger.debug("Image load failed for %s: %s", path, e)
        return None, f"load failed: {e}"


def _render_unit_grid_pil(
    *,
    uid: Any,
    panels: list[dict[str, Any]],
    out_png: Path,
    out_pdf: Path,
    force_restart: bool,
    logger: logging.Logger,
) -> dict[str, Any]:
    """Pillow-based fallback for rendering the 2x4 grid when matplotlib isn't available."""

    out_png = Path(out_png)
    out_pdf = Path(out_pdf)

    if (not force_restart) and out_png.exists() and out_pdf.exists():
        return {"grid_png": str(out_png), "grid_pdf": str(out_pdf), "status": "ok", "error": None}

    try:
        from PIL import Image, ImageDraw, ImageFont  # type: ignore[import-not-found]
    except Exception as e:
        return {"grid_png": None, "grid_pdf": None, "status": "error", "error": f"pillow unavailable: {e}"}

    # Match the previous matplotlib canvas (~39x16.5 at dpi=200).
    canvas_w, canvas_h = 7800, 3300
    height_top = int(round(canvas_h * (2.4 / (2.4 + 1.0))))
    height_bottom = canvas_h - height_top
    col_w = canvas_w // 4

    # Slots: (x0, y0, w, h)
    slots: list[tuple[int, int, int, int]] = [
        (0, 0, col_w * 3, height_top),
        (col_w * 3, 0, col_w, height_top),
        (0, height_top, col_w, height_bottom),
        (col_w * 1, height_top, col_w, height_bottom),
        (col_w * 2, height_top, col_w, height_bottom),
        (col_w * 3, height_top, col_w, height_bottom),
    ]

    canvas = Image.new("RGBA", (canvas_w, canvas_h), (255, 255, 255, 255))
    draw = ImageDraw.Draw(canvas, "RGBA")
    font = ImageFont.load_default()

    for (x0, y0, w, h), p in zip(slots, panels):
        path = Path(p.get("path")) if p.get("path") else None

        img = None
        err = "no path"
        if path is not None:
            img, err = _try_load_image_pil(path=path, logger=logger)

        if img is None:
            placeholder = f"{path.name if path else ''}\n({err})".strip()
            draw.multiline_text(
                (x0 + w // 2, y0 + h // 2),
                placeholder,
                fill=(0, 0, 0, 255),
                anchor="mm",
                align="center",
                font=font,
            )
            continue

        # Scale to fit slot while preserving aspect ratio.
        try:
            resample = Image.Resampling.LANCZOS  # Pillow>=9
        except Exception:
            resample = Image.LANCZOS

        img2 = img.copy()
        img2.thumbnail((w, h), resample=resample)
        px = x0 + (w - img2.size[0]) // 2
        py = y0 + (h - img2.size[1]) // 2
        canvas.alpha_composite(img2, dest=(px, py))

    out_png.parent.mkdir(parents=True, exist_ok=True)
    try:
        canvas.save(out_png, format="PNG")
        canvas.convert("RGB").save(out_pdf, format="PDF", resolution=200.0)
    except Exception as e:
        return {"grid_png": None, "grid_pdf": None, "status": "error", "error": f"save failed: {e}"}

    return {"grid_png": str(out_png), "grid_pdf": str(out_pdf), "status": "ok", "error": None}


def _render_unit_grid(
    *,
    uid: Any,
    panels: list[dict[str, Any]],
    out_png: Path,
    out_pdf: Path,
    force_restart: bool,
    logger: logging.Logger,
) -> dict[str, Any]:
    """Render a 2x4 unit summary grid.

        Current layout (tight, inspection-friendly):
    - Top: templates topo footprint (spans 3 columns)
    - Top-right: templates zoomed contributing footprint (PTP)
    - Bottom-left: templates propagation plot
            - Bottom row: reconstruction plots (4 panels)

        Notes:
            - We intentionally hide axes/spines to avoid boxing each panel.
            - Missing panels render placeholder text in the corresponding slot.
    """

    out_png = Path(out_png)
    out_pdf = Path(out_pdf)

    if (not force_restart) and out_png.exists() and out_pdf.exists():
        return {"grid_png": str(out_png), "grid_pdf": str(out_pdf), "status": "ok", "error": None}

    try:
        import matplotlib  # type: ignore[import-not-found]

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt  # type: ignore[import-not-found]
    except Exception as e:
        logger.debug("matplotlib unavailable for unit %s (%s); falling back to Pillow montage", uid, e)
        return _render_unit_grid_pil(
            uid=uid,
            panels=panels,
            out_png=out_png,
            out_pdf=out_pdf,
            force_restart=force_restart,
            logger=logger,
        )

    # Larger canvas so each panel is easier to inspect.
    # ~50% larger than the previous 26x11 layout.
    fig = plt.figure(figsize=(39, 16.5))

    # Give the top row more space so the topo footprint is easier to inspect.
    gs = fig.add_gridspec(nrows=2, ncols=4, height_ratios=[2.4, 1.0])
    axes = {
        "topo": fig.add_subplot(gs[0, 0:3]),
        "fpzoom": fig.add_subplot(gs[0, 3]),
        "prop": fig.add_subplot(gs[1, 0]),
        "b0": fig.add_subplot(gs[1, 1]),
        "b1": fig.add_subplot(gs[1, 2]),
        "b2": fig.add_subplot(gs[1, 3]),
    }

    # Panel order is defined by analyze_units(). Expected 6 panels.
    panel_axes = [
        axes["topo"],
        axes["fpzoom"],
        axes["prop"],
        axes["b0"],
        axes["b1"],
        axes["b2"],
    ]

    for ax, p in zip(panel_axes, panels):
        title = str(p.get("title") or "")
        path = Path(p.get("path")) if p.get("path") else None

        rgba = None
        err = "no path"
        if path is not None:
            rgba, err = _try_load_image_rgba(path=path, logger=logger)

        if rgba is not None:
            ax.imshow(rgba)
        else:
            # No per-panel titles in the grid; keep a lightweight placeholder.
            placeholder = f"{path.name if path else ''}\n({err})".strip()
            ax.text(0.5, 0.5, placeholder, ha="center", va="center", fontsize=10)

        # Hide axes/spines to avoid boxing panels.
        ax.set_axis_off()
        for spine in ax.spines.values():
            spine.set_visible(False)

    # If fewer than 5 panels are provided, blank the rest.
    for ax in panel_axes[len(panels) :]:
        ax.set_axis_off()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Tighten subplot spacing (titles are removed; axes are off).
        # Full-bleed with essentially zero padding.
        fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0, wspace=0.0, hspace=0.0)
    except Exception:
        pass

    # pad_inches=0 removes the last bit of whitespace around the grid.
    fig.savefig(out_png, dpi=200, bbox_inches="tight", pad_inches=0.0)
    fig.savefig(out_pdf, dpi=200, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)

    return {"grid_png": str(out_png), "grid_pdf": str(out_pdf), "status": "ok", "error": None}


@dataclass(frozen=True)
class AnalysisInputs:
    h5_path: Path
    stream_id: str
    mea_output_root: Path

    unit_ids: Optional[list[Any]] = None
    unit_limit: Optional[int] = None

    # Prefer waveforms curated panels when available.
    prefer_curated_waveforms_panels: bool = True

    # Resume/overwrite controls
    force_restart: bool = False


@dataclass(frozen=True)
class AnalysisOutputs:
    well_out_dir: Path
    analysis_out_dir: Path
    summary_json: Path
    by_unit_dir: Path


def analyze_units(*, inputs: AnalysisInputs, logger_name_prefix: str = "axon_reconstructor") -> AnalysisOutputs:
    """Generate analysis summary artifacts (currently per-unit summary grids)."""

    well_out_dir = _compute_mea_analysis_output_dir(
        output_root=inputs.mea_output_root,
        data_file=inputs.h5_path,
        well=inputs.stream_id,
    )

    log_file = compute_pipeline_log_file(well_out_dir=well_out_dir, data_file=inputs.h5_path, stream_id=inputs.stream_id)
    logger = setup_pipeline_logger(
        log_file=log_file,
        logger_name=f"{logger_name_prefix}.{inputs.stream_id}.analysis",
        verbose=True,
    )

    analysis_out_dir = well_out_dir / ANALYSIS_OUTPUTS_DIRNAME
    by_unit_dir = analysis_out_dir / "by_unit"
    summary_json = analysis_out_dir / "analysis_summary.json"

    ckpt_file = _compute_analysis_checkpoint_file(well_out_dir=well_out_dir, h5_path=inputs.h5_path, stream_id=inputs.stream_id)
    ckpt = load_checkpoint(
        checkpoint_file=ckpt_file,
        force_restart=bool(inputs.force_restart),
        output_dir=well_out_dir,
        file_path=inputs.h5_path,
        stream_id=inputs.stream_id,
    )

    analysis_out_dir.mkdir(parents=True, exist_ok=True)
    by_unit_dir.mkdir(parents=True, exist_ok=True)

    ckpt = save_checkpoint(
        checkpoint_file=ckpt_file,
        state=ckpt,
        stage=ProcessingStage.REPORTS,
        failed_stage=None,
        error=None,
        extra_fields={"analysis_out_dir": str(analysis_out_dir)},
    )

    # Unit discovery: default to templates merged_units (stable contract).
    templates_out_dir = well_out_dir / "templates_outputs"
    merged_units_dir = templates_out_dir / "merged_units"
    discovered_unit_ids: list[Any] = []
    for p in sorted(merged_units_dir.glob("unit_*") if merged_units_dir.exists() else []):
        if not p.is_dir():
            continue
        try:
            discovered_unit_ids.append(int(p.name.split("unit_", 1)[1]))
        except Exception:
            discovered_unit_ids.append(p.name.split("unit_", 1)[1])

    if inputs.unit_ids is not None:
        unit_ids = list(inputs.unit_ids)
    else:
        unit_ids = list(discovered_unit_ids)

    if inputs.unit_limit is not None:
        unit_ids = unit_ids[: int(inputs.unit_limit)]

    # Common roots
    recon_by_unit_root = well_out_dir / "reconstruction_outputs" / "by_unit"

    summary: dict[str, Any] = {
        "h5_path": str(inputs.h5_path),
        "stream_id": inputs.stream_id,
        "well_out_dir": str(well_out_dir),
        "analysis_out_dir": str(analysis_out_dir),
        "by_unit_dir": str(by_unit_dir),
        "units": [],
    }

    for uid in unit_ids:
        out_unit_dir = by_unit_dir / f"unit_{uid}"
        out_unit_dir.mkdir(parents=True, exist_ok=True)

        # Tight 2x4 grid layout expects 6 panels:
        #   [0] topo footprint (spans 3 cols)
        #   [1] zoomed contributing-footprint PTP
        #   [2] propagation plot
        #   [3:6] reconstruction plots (bottom row, 3 panels)
        panels = [
            {
                "title": "Templates: topo unit footprint",
                "path": templates_out_dir / "topo_unit_footprints" / f"unit_{uid}.png",
            },
            {
                "title": "Templates: merged contributing footprint (zoom)",
                "path": templates_out_dir
                / "footprints_zoomed"
                / f"unit_{uid}_merged_contributing_footprint_ptp_linear_zoom.png",
            },
            {
                "title": "Templates: propagation plot",
                "path": templates_out_dir / "propagation_plots" / f"unit_{uid}.png",
            },
            {
                "title": "Reconstruction: branch velocities",
                "path": recon_by_unit_root / f"unit_{uid}" / "branch_velocities.png",
            },
            {
                "title": "Reconstruction: raw branches (zoom)",
                "path": recon_by_unit_root / f"unit_{uid}" / "branches_raw_zoom.png",
            },
            {
                "title": "Reconstruction: graph heuristics",
                "path": recon_by_unit_root / f"unit_{uid}" / "graph_heuristics.png",
            },
        ]

        grid_png = out_unit_dir / "unit_summary_grid.png"
        grid_pdf = out_unit_dir / "unit_summary_grid.pdf"

        unit_record: dict[str, Any] = {
            "unit_id": uid,
            "inputs": {
                "reconstruction_unit_dir": str(recon_by_unit_root / f"unit_{uid}"),
                "templates_out_dir": str(templates_out_dir),
            },
            "outputs": {},
            "status": "ok",
            "error": None,
        }

        try:
            render_out = _render_unit_grid(
                uid=uid,
                panels=panels,
                out_png=grid_png,
                out_pdf=grid_pdf,
                force_restart=bool(inputs.force_restart),
                logger=logger,
            )
            unit_record["outputs"].update(render_out)
            unit_record["status"] = render_out.get("status", "ok")
            unit_record["error"] = render_out.get("error")
        except Exception as e:
            unit_record["status"] = "error"
            unit_record["error"] = exception_to_error_dict(e)

        summary["units"].append(unit_record)

    _write_json(summary_json, summary)

    ckpt = save_checkpoint(
        checkpoint_file=ckpt_file,
        state=ckpt,
        stage=ProcessingStage.REPORTS_COMPLETE,
        failed_stage=None,
        error=None,
        extra_fields={
            "analysis_out_dir": str(analysis_out_dir),
            "analysis_summary_json": str(summary_json),
        },
    )

    return AnalysisOutputs(
        well_out_dir=well_out_dir,
        analysis_out_dir=analysis_out_dir,
        summary_json=summary_json,
        by_unit_dir=by_unit_dir,
    )
