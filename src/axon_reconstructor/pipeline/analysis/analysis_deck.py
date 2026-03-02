#!/usr/bin/env python3
"""Build a PowerPoint slide deck from analysis unit summary grids.

This script:
  1) Runs the analysis stage for all discovered units (or a subset)
  2) Filters to units that have *all required input panels present*
  3) Creates a .pptx deck with one slide per unit (grid PNG full-bleed)

Env controls
------------
- AXON_RECON_FORCE_RESTART=1   re-render analysis grids
- AXON_RECON_UNIT_LIMIT=10     limit to first N units ("none" => all)
- AXON_RECON_UNIT_IDS=1,2,26   run only selected units
- AXON_RECON_REQUIRE_COMPLETE=0 include units even if some panels are missing

Output
------
Writes to:
  <well_out_dir>/analysis_outputs/unit_summary_grids_complete.pptx

Notes
-----
"complete" means the required source images exist:
    - templates topo footprint
    - templates propagation plot
    - templates zoomed merged_contributing footprint (PTP, linear)
    - reconstruction branch_velocities.png
    - reconstruction branches_raw_zoom.png
    - reconstruction graph_heuristics.png

We embed the rendered grid PNG on each slide (not the individual panels).
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import textwrap
from pathlib import Path

from axon_reconstructor import env_utils


# Dataset configuration comes from debug.env (or CLI overrides).
H5_PATH: Path | None = None
STREAM_ID: str | None = None
MEA_OUTPUT_ROOT: Path | None = None

DEBUG = False

FORCE_RESTART = False
UNIT_LIMIT: int | None = None
UNIT_IDS: list[int] | None = None
REQUIRE_COMPLETE = True


def _parse_int_or_none(raw: str) -> int | None:
    v = str(raw).strip().lower()
    if v in {"none", "null", "all"}:
        return None
    return int(v)


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "y"}


def _env_int_or_none(name: str, default: int | None) -> int | None:
    if os.environ.get(name) is None:
        return default
    raw = os.environ[name].strip().lower()
    if raw in {"none", "null", "all"}:
        return None
    return int(raw)


def _env_int_list(name: str) -> list[int] | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _is_nonempty_file(path: Path, *, min_bytes: int = 1024) -> bool:
    try:
        return path.is_file() and (path.stat().st_size >= min_bytes)
    except Exception:
        return False


def _try_export_pptx_as_pdf(*, pptx_path: Path, out_dir: Path, logger: logging.Logger) -> Path | None:
    """Best-effort PPTX -> PDF conversion.

    Uses LibreOffice/soffice headless if available.
    Returns the produced PDF path if successful, else None.
    """

    pptx_path = Path(pptx_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if not soffice:
        logger.warning("PDF export skipped (missing libreoffice/soffice)")
        return None

    # LibreOffice writes <stem>.pdf in out_dir.
    expected_pdf = out_dir / f"{pptx_path.stem}.pdf"
    if expected_pdf.exists():
        try:
            expected_pdf.unlink()
        except Exception:
            pass

    cmd = [
        soffice,
        "--headless",
        "--nologo",
        "--nodefault",
        "--nolockcheck",
        "--norestore",
        "--convert-to",
        "pdf",
        "--outdir",
        str(out_dir),
        str(pptx_path),
    ]

    try:
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    except Exception as e:
        logger.warning("PDF export failed to run (%s): %s", soffice, e)
        return None

    if proc.returncode != 0:
        logger.warning(
            "PDF export failed (exit %s). stdout=%s stderr=%s",
            proc.returncode,
            (proc.stdout or "").strip()[:1000],
            (proc.stderr or "").strip()[:1000],
        )
        return None

    if expected_pdf.exists() and expected_pdf.stat().st_size > 1024:
        return expected_pdf

    # Some LO versions may output a slightly different name; fall back to searching.
    try:
        cands = sorted(out_dir.glob("*.pdf"), key=lambda p: p.stat().st_mtime, reverse=True)
        for c in cands:
            if c.stem == pptx_path.stem and c.stat().st_size > 1024:
                return c
    except Exception:
        pass

    logger.warning("PDF export finished but output PDF not found")
    return None


def _build_summary_lines(
    *,
    h5_path: Path,
    stream_id: str,
    well_out_dir: Path,
    analysis_out_dir: Path,
    eligible_units: list[int],
    discovered_units: int,
    require_complete: bool,
) -> list[str]:
    ctx = _parse_recording_context_from_path(h5_path)

    # File stats
    size_mb = None
    mtime = None
    try:
        st = h5_path.stat()
        size_mb = st.st_size / (1024 * 1024)
        import datetime as _dt

        mtime = _dt.datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds")
    except Exception:
        pass

    si_info, si_err = _try_get_spikeinterface_recording_info(h5_path=h5_path, stream_id=stream_id)

    title_bits = ["Recording Summary"]
    if ctx.get("dataset"):
        title_bits.append(ctx["dataset"])
    if ctx.get("run"):
        title_bits.append(f"run {ctx['run']}")
    title_bits.append(stream_id)
    title = " — ".join([title_bits[0], " / ".join(title_bits[1:])]) if len(title_bits) > 1 else title_bits[0]

    lines: list[str] = [title]

    lines.append(f"H5: {h5_path}")
    if size_mb is not None:
        lines.append(f"File size: {size_mb:.1f} MB")
    if mtime is not None:
        lines.append(f"Last modified: {mtime}")

    pretty_ctx = []
    for k in ["date", "plate", "assay", "run"]:
        if ctx.get(k):
            pretty_ctx.append(f"{k}={ctx[k]}")
    if pretty_ctx:
        lines.append("Parsed path: " + ", ".join(pretty_ctx))

    try:
        cfgs = sorted(h5_path.parent.glob("*.cfg"))
        lines.append(f"Neighbor .cfg files: {len(cfgs)}")
    except Exception:
        pass

    lines.append(f"Output well dir: {well_out_dir}")
    lines.append(f"Analysis dir: {analysis_out_dir}")
    lines.append(
        f"Units: discovered={discovered_units}  eligible_for_deck={len(eligible_units)}  require_complete={require_complete}"
    )

    if si_info:
        lines.append("Recording metadata (SpikeInterface):")
        for k in [
            "sampling_frequency_hz",
            "num_channels",
            "num_segments",
            "num_frames",
            "num_frames_total",
            "duration_s_total",
            "dtype",
            "channel_locations_shape",
        ]:
            if k in si_info:
                lines.append(f"  - {k}: {si_info[k]}")
    elif si_err is not None:
        lines.append("Recording metadata (SpikeInterface): unavailable")
        lines.append(f"  - {si_err}")

    return lines


def _try_write_pdf_deck_from_grids(
    *,
    pdf_path: Path,
    h5_path: Path,
    stream_id: str,
    well_out_dir: Path,
    analysis_out_dir: Path,
    eligible: list[tuple[int, Path]],
    discovered_units: int,
    require_complete: bool,
    logger: logging.Logger,
) -> Path | None:
    """Write a multi-page PDF deck directly from grid PNGs.

    Prefer img2pdf (streaming, no huge memory spikes). Fallback to Pillow if needed.
    """

    pdf_path = Path(pdf_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    # Create a summary page as a temporary PNG so we can use the same PDF path for either backend.
    summary_png = analysis_out_dir / "_unit_summary_grids_complete__summary.png"

    try:
        from PIL import Image, ImageDraw, ImageFont  # type: ignore[import-not-found]
    except Exception as e:
        logger.warning("PDF export skipped (missing pillow): %s", e)
        return None

    # Determine page size from first eligible grid (or fallback).
    page_w, page_h = 1920, 1080
    if eligible:
        try:
            with Image.open(eligible[0][1]) as im:
                page_w, page_h = im.size
        except Exception:
            pass

    lines = _build_summary_lines(
        h5_path=h5_path,
        stream_id=stream_id,
        well_out_dir=well_out_dir,
        analysis_out_dir=analysis_out_dir,
        eligible_units=[uid for uid, _p in eligible],
        discovered_units=int(discovered_units),
        require_complete=bool(require_complete),
    )

    # Render summary image.
    try:
        summary_img = Image.new("RGB", (int(page_w), int(page_h)), (255, 255, 255))
        draw = ImageDraw.Draw(summary_img)

        # Try a real TTF so the page is readable; fall back to default.
        title_font = None
        body_font = None
        try:
            # Common on Linux
            ttf = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
            title_font = ImageFont.truetype(ttf, size=max(28, int(page_h * 0.035)))
            body_font = ImageFont.truetype(ttf, size=max(16, int(page_h * 0.022)))
        except Exception:
            title_font = ImageFont.load_default()
            body_font = ImageFont.load_default()

        margin_x = int(page_w * 0.06)
        y = int(page_h * 0.06)
        max_text_w = int(page_w * 0.88)

        # Title
        draw.text((margin_x, y), lines[0], fill=(0, 0, 0), font=title_font)
        y += int(page_h * 0.07)

        # Body with crude wrapping.
        wrap_chars = 110
        for raw in lines[1:]:
            for ln in textwrap.wrap(raw, width=wrap_chars, subsequent_indent="    ") or [""]:
                draw.text((margin_x, y), ln, fill=(0, 0, 0), font=body_font)
                y += int(page_h * 0.03)
                if y > int(page_h * 0.95):
                    break
            if y > int(page_h * 0.95):
                break

        summary_img.save(summary_png, format="PNG")
    except Exception as e:
        logger.warning("Failed rendering PDF summary page: %s", e)
        summary_png = None  # type: ignore[assignment]

    input_pngs: list[Path] = []
    if summary_png is not None and isinstance(summary_png, Path) and summary_png.exists():
        input_pngs.append(summary_png)
    input_pngs.extend([p for _uid, p in eligible])

    # Backend 1: img2pdf (preferred)
    try:
        import img2pdf  # type: ignore[import-not-found]

        with open(pdf_path, "wb") as f:
            f.write(img2pdf.convert([str(p) for p in input_pngs]))
        if _is_nonempty_file(pdf_path, min_bytes=10_000):
            return pdf_path
    except Exception as e:
        logger.info("img2pdf not available (or failed): %s; falling back to Pillow PDF", e)

    # Backend 2: Pillow multipage PDF (may be memory heavy; downscale to keep it safe).
    try:
        pages = []
        max_w = 2200  # keep memory bounded
        for p in input_pngs:
            with Image.open(p) as im:
                im = im.convert("RGB")
                if im.size[0] > max_w:
                    scale = max_w / float(im.size[0])
                    new_sz = (int(im.size[0] * scale), int(im.size[1] * scale))
                    im = im.resize(new_sz)
                pages.append(im)

        if not pages:
            return None
        first, rest = pages[0], pages[1:]
        first.save(pdf_path, format="PDF", save_all=True, append_images=rest, resolution=200.0)
        if _is_nonempty_file(pdf_path, min_bytes=10_000):
            return pdf_path
    except Exception as e:
        logger.warning("Pillow PDF export failed: %s", e)

    return None


def _parse_recording_context_from_path(h5_path: Path) -> dict[str, str]:
    """Best-effort parse of dataset-ish metadata from the h5 path."""

    parts = [p for p in h5_path.parts if p]

    # Heuristic for Mandar/MEA_Analysis-style layout (example):
    #   .../raw_data/.../<dataset>/<YYMMDD>/<plate>/<assay>/<run>/data.raw.h5
    out: dict[str, str] = {}
    try:
        out["filename"] = h5_path.name
        out["parent_dir"] = h5_path.parent.name
        if len(parts) >= 6:
            out["dataset"] = parts[-6]
            out["date"] = parts[-5]
            out["plate"] = parts[-4]
            out["assay"] = parts[-3]
            out["run"] = parts[-2]
    except Exception:
        pass

    return out


def _try_get_spikeinterface_recording_info(*, h5_path: Path, stream_id: str) -> tuple[dict[str, str], str | None]:
    """Best-effort SpikeInterface recording metadata.

    Returns (info_dict, error_message).
    """

    try:
        import spikeinterface.extractors as se  # type: ignore[import-not-found]

        rec = se.read_maxwell(h5_path, stream_id=stream_id)

        info: dict[str, str] = {}
        try:
            info["sampling_frequency_hz"] = str(rec.get_sampling_frequency())
        except Exception:
            pass
        try:
            info["num_channels"] = str(rec.get_num_channels())
        except Exception:
            pass
        try:
            info["num_segments"] = str(rec.get_num_segments())
        except Exception:
            pass
        try:
            dtype = getattr(rec, "get_dtype", None)
            if callable(dtype):
                info["dtype"] = str(dtype())
        except Exception:
            pass
        try:
            # Duration: if multi-segment, report the first and total.
            fs = float(rec.get_sampling_frequency())
            nseg = int(rec.get_num_segments())
            seg_frames = []
            for si in range(nseg):
                seg_frames.append(int(rec.get_num_frames(segment_index=si)))
            if seg_frames and fs > 0:
                total_frames = sum(seg_frames)
                info["duration_s_total"] = f"{total_frames / fs:.2f}"
                if nseg == 1:
                    info["num_frames"] = str(seg_frames[0])
                else:
                    info["num_frames_total"] = str(total_frames)
                    info["num_frames_by_segment"] = ", ".join(str(x) for x in seg_frames)
        except Exception:
            pass

        try:
            # channel locations if available
            get_locs = getattr(rec, "get_channel_locations", None)
            if callable(get_locs):
                locs = get_locs()
                try:
                    info["channel_locations_shape"] = f"{locs.shape}"  # type: ignore[attr-defined]
                except Exception:
                    pass
        except Exception:
            pass

        return info, None
    except Exception as e:
        return {}, f"SpikeInterface read_maxwell failed: {e}"


def _add_summary_slide(
    *,
    prs: any,
    h5_path: Path,
    stream_id: str,
    well_out_dir: Path,
    analysis_out_dir: Path,
    eligible_units: list[int],
    discovered_units: int,
    require_complete: bool,
) -> None:
    from pptx.util import Inches, Pt

    blank_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(blank_layout)

    box = slide.shapes.add_textbox(Inches(0.6), Inches(0.4), Inches(12.2), Inches(6.8))
    tf = box.text_frame
    tf.word_wrap = True

    ctx = _parse_recording_context_from_path(h5_path)

    # File stats
    size_mb = None
    mtime = None
    try:
        st = h5_path.stat()
        size_mb = st.st_size / (1024 * 1024)
        import datetime as _dt

        mtime = _dt.datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds")
    except Exception:
        pass

    si_info, si_err = _try_get_spikeinterface_recording_info(h5_path=h5_path, stream_id=stream_id)

    # Title
    title_bits = ["Recording Summary"]
    if ctx.get("dataset"):
        title_bits.append(ctx["dataset"])
    if ctx.get("run"):
        title_bits.append(f"run {ctx['run']}")
    title_bits.append(stream_id)
    title = " — ".join([title_bits[0], " / ".join(title_bits[1:])]) if len(title_bits) > 1 else title_bits[0]

    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(28)
    p.font.bold = True

    def add_line(text: str) -> None:
        para = tf.add_paragraph()
        para.text = text
        para.font.size = Pt(14)

    add_line(f"H5: {h5_path}")
    if size_mb is not None:
        add_line(f"File size: {size_mb:.1f} MB")
    if mtime is not None:
        add_line(f"Last modified: {mtime}")

    # Parsed context
    pretty_ctx = []
    for k in ["date", "plate", "assay", "run"]:
        if ctx.get(k):
            pretty_ctx.append(f"{k}={ctx[k]}")
    if pretty_ctx:
        add_line("Parsed path: " + ", ".join(pretty_ctx))

    # Neighbor cfg files (like preprocessing step does)
    try:
        cfgs = sorted(h5_path.parent.glob("*.cfg"))
        add_line(f"Neighbor .cfg files: {len(cfgs)}")
    except Exception:
        pass

    add_line(f"Output well dir: {well_out_dir}")
    add_line(f"Analysis dir: {analysis_out_dir}")
    add_line(
        f"Units: discovered={discovered_units}  eligible_for_deck={len(eligible_units)}  require_complete={require_complete}"
    )

    # SpikeInterface metadata
    if si_info:
        add_line("Recording metadata (SpikeInterface):")
        # Keep concise and in a stable order.
        for k in [
            "sampling_frequency_hz",
            "num_channels",
            "num_segments",
            "num_frames",
            "num_frames_total",
            "duration_s_total",
            "dtype",
            "channel_locations_shape",
        ]:
            if k in si_info:
                add_line(f"  - {k}: {si_info[k]}")
    elif si_err is not None:
        add_line("Recording metadata (SpikeInterface): unavailable")
        add_line(f"  - {si_err}")


def run_with_args(args: object) -> int:

    debug_enabled = env_utils.env_bool("AXON_RECON_DEBUG", default=bool(DEBUG)) if args.debug is None else bool(args.debug)
    log_level = logging.DEBUG if debug_enabled else logging.INFO
    logging.basicConfig(level=log_level, format="[%(levelname)s] %(message)s", force=True)
    logger = logging.getLogger("projects.debug_analysis_deck")

    h5_path = Path(args.h5_path) if args.h5_path is not None else env_utils.env_required_path("AXON_RECON_H5_PATH")
    stream_id = str(args.stream_id) if args.stream_id is not None else env_utils.env_required_str("AXON_RECON_STREAM_ID")
    mea_output_root = (
        Path(args.mea_output_root)
        if args.mea_output_root is not None
        else env_utils.env_required_path("AXON_RECON_MEA_OUTPUT_ROOT")
    )

    from axon_reconstructor.pipeline.analysis import AnalysisInputs, analyze_units

    # Precedence: CLI > env > ALL-CAPS constants.
    if args.force_restart is not None:
        force_restart = bool(args.force_restart)
    else:
        force_restart = env_utils.env_bool("AXON_RECON_FORCE_RESTART", default=bool(FORCE_RESTART))

    unit_limit = _env_int_or_none("AXON_RECON_UNIT_LIMIT", default=UNIT_LIMIT)
    if args.unit_limit is not None:
        unit_limit = _parse_int_or_none(args.unit_limit)

    unit_ids = _env_int_list("AXON_RECON_UNIT_IDS")
    if args.unit_ids:
        unit_ids = [int(x) for x in args.unit_ids]
    elif UNIT_IDS is not None:
        unit_ids = list(UNIT_IDS)

    if args.require_complete is not None:
        require_complete = bool(args.require_complete)
    else:
        require_complete = env_utils.env_bool("AXON_RECON_REQUIRE_COMPLETE", default=bool(REQUIRE_COMPLETE))

    compute_botm_validation = env_utils.env_bool("AXON_RECON_ANALYSIS_BOTM_ENABLE", default=False)
    botm_n_events = int(env_utils.env_int("AXON_RECON_ANALYSIS_BOTM_N_SPIKE", default=200) or 200)
    botm_n_noise_windows = int(env_utils.env_int("AXON_RECON_ANALYSIS_BOTM_N_NOISE", default=2000) or 2000)
    botm_seed = int(env_utils.env_int("AXON_RECON_ANALYSIS_BOTM_SEED", default=0) or 0)
    botm_prior_signal = float(env_utils.env_float("AXON_RECON_ANALYSIS_BOTM_CHANNEL_MATCH_PRIOR_SIGNAL", default=0.5) or 0.5)
    botm_match_fraction_threshold = float(
        env_utils.env_float("AXON_RECON_ANALYSIS_BOTM_CHANNEL_MATCH_FRACTION_THRESHOLD", default=0.70) or 0.70
    )
    botm_sorter = str(env_utils.env_str("AXON_RECON_ANALYSIS_BOTM_SORTER", default="kilosort4") or "kilosort4")

    inputs = AnalysisInputs(
        h5_path=h5_path,
        stream_id=stream_id,
        mea_output_root=mea_output_root,
        unit_ids=unit_ids,
        unit_limit=unit_limit,
        compute_botm_validation=compute_botm_validation,
        botm_n_events=botm_n_events,
        botm_n_noise_windows=botm_n_noise_windows,
        botm_seed=botm_seed,
        botm_prior_signal=botm_prior_signal,
        botm_match_fraction_threshold=botm_match_fraction_threshold,
        botm_sorter=botm_sorter,
        force_restart=force_restart,
    )

    logger.info("Running analysis to ensure grids exist")
    out = analyze_units(inputs=inputs, logger_name_prefix="projects")

    # Required source images for a "complete" grid.
    def required_sources(uid: int) -> list[Path]:
        well_out_dir = out.well_out_dir
        return [
            well_out_dir / "templates_outputs" / "footprints" / "3D" / f"unit_{uid}.png",
            well_out_dir / "templates_outputs" / "propagation_plots" / f"unit_{uid}.png",
            well_out_dir
            / "templates_outputs"
            / "footprints"
            / "zoomed"
            / f"unit_{uid}_merged_contributing_footprint_ptp_linear_zoom.png",
            well_out_dir
            / "reconstruction_outputs"
            / "by_unit"
            / f"unit_{uid}"
            / "branches"
            / "clean"
            / "branch_velocities.png",
            well_out_dir
            / "reconstruction_outputs"
            / "by_unit"
            / f"unit_{uid}"
            / "branches"
            / "raw"
            / "branches_raw_zoom.png",
            well_out_dir / "reconstruction_outputs" / "by_unit" / f"unit_{uid}" / "heuristics" / "graph_heuristics.png",
        ]

    by_unit_dir = out.by_unit_dir
    unit_dirs = sorted([p for p in by_unit_dir.glob("unit_*") if p.is_dir()])

    eligible: list[tuple[int, Path]] = []
    for ud in unit_dirs:
        try:
            uid = int(ud.name.split("unit_", 1)[1])
        except Exception:
            continue

        grid_png = ud / "unit_summary_grid.png"
        if not _is_nonempty_file(grid_png, min_bytes=5_000):
            continue

        if require_complete:
            missing = [p for p in required_sources(uid) if not _is_nonempty_file(p, min_bytes=1_000)]
            if missing:
                continue

        eligible.append((uid, grid_png))

    if not eligible:
        logger.error("No eligible units found (require_complete=%s).", require_complete)
        return 2

    safe_stream = str(stream_id).replace(os.sep, "_").replace(" ", "_")
    deck_path = out.analysis_out_dir / f"unit_summary_grids_{safe_stream}_complete.pptx"
    pdf_path = out.analysis_out_dir / f"unit_summary_grids_{safe_stream}_complete.pdf"

    eligible_uids = [uid for uid, _ in eligible]

    def _convert_svg_to_png(*, svg_path: Path, png_path: Path, logger: any) -> bool:
        """Best-effort SVG->PNG conversion for PPTX embedding."""

        png_path.parent.mkdir(parents=True, exist_ok=True)

        # 1) Try cairosvg (pure python dependency)
        try:
            import cairosvg  # type: ignore[import-not-found]

            cairosvg.svg2png(url=str(svg_path), write_to=str(png_path))
            return png_path.exists()
        except Exception:
            pass

        # 2) Try external converters if available
        import shutil
        import subprocess

        for exe, cmd in [
            ("rsvg-convert", lambda: ["rsvg-convert", str(svg_path), "-o", str(png_path)]),
            (
                "inkscape",
                lambda: [
                    "inkscape",
                    str(svg_path),
                    "--export-type=png",
                    f"--export-filename={png_path}",
                ],
            ),
        ]:
            if shutil.which(exe):
                try:
                    subprocess.run(cmd(), check=False, capture_output=True, text=True)
                    if png_path.exists():
                        return True
                except Exception:
                    continue

        logger.warning("Could not convert SVG to PNG: %s", svg_path)
        return False

    def _add_all_units_morphology_slide(*, prs: any, stream_id: str, well_out_dir: Path, blank_layout: any) -> None:
        """Add the all-units morphology as slide #2 when available."""

        slide_w = int(prs.slide_width)
        slide_h = int(prs.slide_height)

        recon_dir = Path(well_out_dir) / "reconstruction_outputs"
        svg_path = recon_dir / "all_units_morphology.svg"
        png_path = recon_dir / "all_units_morphology.png"
        pdf_path = recon_dir / "all_units_morphology.pdf"

        img_path: Path | None = None
        if png_path.exists():
            img_path = png_path
        elif svg_path.exists():
            tmp_png = Path(well_out_dir) / "analysis_outputs" / "_all_units_morphology.png"
            if _convert_svg_to_png(svg_path=svg_path, png_path=tmp_png, logger=logger):
                img_path = tmp_png
        else:
            # No usable raster source.
            if pdf_path.exists():
                logger.warning("All-units morphology exists as PDF but PPTX needs PNG/SVG: %s", pdf_path)
            return

        if img_path is None or (not img_path.exists()):
            return

        slide = prs.slides.add_slide(blank_layout)
        _add_picture_fit(
            slide=slide,
            img_path=img_path,
            left=0,
            top=0,
            box_w=slide_w,
            box_h=slide_h,
        )
        try:
            slide.notes_slide.notes_text_frame.text = f"all_units_morphology ({stream_id})"
        except Exception:
            pass

    def _fit_image_into_box(*, img_path: Path, box_w: int, box_h: int) -> tuple[int, int]:
        """Return (pic_w, pic_h) that fits image into the given box preserving aspect ratio."""

        from PIL import Image  # type: ignore[import-not-found]

        with Image.open(img_path) as im:
            iw, ih = im.size
        if iw <= 0 or ih <= 0:
            return box_w, box_h
        img_ratio = iw / float(ih)
        box_ratio = box_w / float(box_h)
        if img_ratio >= box_ratio:
            pic_w = box_w
            pic_h = int(box_w / img_ratio)
        else:
            pic_h = box_h
            pic_w = int(box_h * img_ratio)
        return max(1, int(pic_w)), max(1, int(pic_h))

    def _add_picture_fit(
        *,
        slide: any,
        img_path: Path,
        left: int,
        top: int,
        box_w: int,
        box_h: int,
    ) -> None:
        """Add picture centered inside bounding box, preserving aspect ratio."""

        pic_w, pic_h = _fit_image_into_box(img_path=img_path, box_w=box_w, box_h=box_h)
        pic_left = int(left + (box_w - pic_w) / 2)
        pic_top = int(top + (box_h - pic_h) / 2)
        slide.shapes.add_picture(str(img_path), pic_left, pic_top, width=pic_w, height=pic_h)

    # Optional: build PPTX deck (requires python-pptx). If missing, still proceed with PDF.
    try:
        from pptx import Presentation  # type: ignore[import-not-found]
        from pptx.util import Inches
        from PIL import Image  # type: ignore[import-not-found]

        prs = Presentation()
        prs.slide_width = Inches(13.333)
        prs.slide_height = Inches(7.5)
        slide_w = int(prs.slide_width)
        slide_h = int(prs.slide_height)

        blank_layout = prs.slide_layouts[6]

        _add_summary_slide(
            prs=prs,
            h5_path=h5_path,
            stream_id=stream_id,
            well_out_dir=out.well_out_dir,
            analysis_out_dir=out.analysis_out_dir,
            eligible_units=eligible_uids,
            discovered_units=len(unit_dirs),
            require_complete=require_complete,
        )

        # Slide 2: all-units morphology from reconstruction_outputs.
        _add_all_units_morphology_slide(
            prs=prs,
            stream_id=stream_id,
            well_out_dir=out.well_out_dir,
            blank_layout=blank_layout,
        )
        logger.info("Building deck with %d unit slides (+1 summary)", len(eligible))

        # Layout:
        # - Left: unit_summary_grid.png takes ~60% width
        # - Right: stack template_movie.gif (top) and summary_raw.png (bottom)
        grid_frac = 0.60
        grid_box_w = int(slide_w * grid_frac)
        right_box_w = int(slide_w - grid_box_w)

        pad = int(slide_w * 0.01)  # ~1% slide width padding
        right_left = grid_box_w + pad
        right_w = max(1, right_box_w - 2 * pad)
        right_h_total = max(1, slide_h - 2 * pad)
        right_top = pad

        gap = int(slide_h * 0.02)
        right_h_each = max(1, int((right_h_total - gap) / 2))

        for uid, png_path in eligible:
            slide = prs.slides.add_slide(blank_layout)

            # Left: analysis summary grid
            _add_picture_fit(
                slide=slide,
                img_path=png_path,
                left=0,
                top=0,
                box_w=grid_box_w,
                box_h=slide_h,
            )

            # Right: template movie (top) and raw summary (bottom)
            recon_unit_dir = out.well_out_dir / "reconstruction_outputs" / "by_unit" / f"unit_{uid}"
            gif_path = recon_unit_dir / "template_movie.gif"
            raw_summary_path = recon_unit_dir / "summary_raw.png"

            top_box_top = right_top
            bot_box_top = right_top + right_h_each + gap

            if gif_path.exists():
                _add_picture_fit(
                    slide=slide,
                    img_path=gif_path,
                    left=right_left,
                    top=top_box_top,
                    box_w=right_w,
                    box_h=right_h_each,
                )

            if raw_summary_path.exists():
                _add_picture_fit(
                    slide=slide,
                    img_path=raw_summary_path,
                    left=right_left,
                    top=bot_box_top,
                    box_w=right_w,
                    box_h=right_h_each,
                )
            try:
                slide.notes_slide.notes_text_frame.text = f"unit {uid}"
            except Exception:
                pass

        prs.save(str(deck_path))
        logger.info("Wrote PPTX: %s", deck_path)

        exported = _try_export_pptx_as_pdf(pptx_path=deck_path, out_dir=out.analysis_out_dir, logger=logger)
        if exported is not None:
            logger.info("Wrote PDF (LibreOffice): %s", exported)
    except Exception as e:
        logger.warning("Skipping PPTX export (python-pptx unavailable or failed): %s", e)

    # Always write a PDF deck directly from the PNG grids.
    pdf_written = _try_write_pdf_deck_from_grids(
        pdf_path=pdf_path,
        h5_path=h5_path,
        stream_id=stream_id,
        well_out_dir=out.well_out_dir,
        analysis_out_dir=out.analysis_out_dir,
        eligible=eligible,
        discovered_units=len(unit_dirs),
        require_complete=require_complete,
        logger=logger,
    )
    if pdf_written is not None:
        logger.info("Wrote PDF (from grids): %s", pdf_written)
    else:
        logger.warning("Failed to write PDF deck")
    return 0
