from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .cross_well import CrossWellConfig

def _try_get_spikeinterface_recording_info(*, h5_path: Path, stream_id: str) -> tuple[dict[str, str], str | None]:
    """Best-effort SpikeInterface metadata extraction (kept intentionally small)."""

    try:
        import spikeinterface.extractors as se

        rec = se.read_maxwell(str(h5_path), stream_id=stream_id)
        info: dict[str, str] = {}
        try:
            info["sampling_frequency_hz"] = str(float(rec.get_sampling_frequency()))
        except Exception:
            pass
        try:
            info["num_channels"] = str(len(rec.get_channel_ids()))
        except Exception:
            pass
        try:
            nseg = int(rec.get_num_segments())
            info["num_segments"] = str(nseg)
        except Exception:
            nseg = 1
        try:
            fs = float(rec.get_sampling_frequency())
            seg_frames = []
            for si in range(nseg):
                seg_frames.append(int(rec.get_num_frames(segment_index=si)))
            if seg_frames and fs > 0:
                total_frames = sum(seg_frames)
                info["duration_s_total"] = f"{total_frames / fs:.2f}"
        except Exception:
            pass
        return info, None
    except Exception as e:
        return {}, f"SpikeInterface read_maxwell failed: {e}"


def _parse_env_assignments(path: Path) -> dict[str, str]:
    """Parse simple KEY=VALUE lines from a dotenv-like file."""

    out: dict[str, str] = {}
    if not path.exists():
        return out
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        out[key] = value
    return out


def _feature_key_from_plot_name(name: str) -> str | None:
    lower = name.lower()
    if (
        "n_branches" in lower
        or "n_units_detected" in lower
        or "n_units_reconstructed" in lower
        or "pct_reconstructed" in lower
    ):
        return "n_counts"
    if "area" in lower:
        return "area"
    if "velocity" in lower:
        return "velocity"
    if "length" in lower:
        return "length"
    return None


def _plot_source_from_name(name: str) -> str:
    lower = name.lower()
    if "branch_source-raw" in lower:
        return "raw"
    if "branch_source-clean" in lower:
        return "clean"
    if "n_counts__n_branches_raw" in lower:
        return "raw"
    if "n_counts__n_branches_clean" in lower:
        return "clean"
    if "_clean" in lower:
        return "clean"
    if "_raw" in lower:
        return "raw"
    if lower.startswith("unit__") and any(
        token in lower
        for token in [
            "mean_branch_velocity",
            "mean_branch_length_um",
            "total_branch_length_um",
            "n_branches_raw",
        ]
    ):
        return "raw"
    return "na"


def _plot_exclusion_level(name: str) -> int:
    lower = name.lower()
    level = 0
    if "positive_only" in lower:
        level += 1
    if "outliers_excluded" in lower:
        level += 1
    return level


def _plot_kind_rank(name: str) -> int:
    lower = name.lower()
    if lower.startswith("branch__"):
        return 0  # individual data-point level plots
    if "mean_" in lower:
        return 1
    if "total_" in lower:
        return 2
    return 3


def _sorted_feature_plots(*, plots_dir: Path, feature: str) -> list[Path]:
    all_pngs = sorted([p for p in Path(plots_dir).glob("*.png") if p.is_file()])
    feature_pngs = [p for p in all_pngs if _feature_key_from_plot_name(p.name) == feature]
    return sorted(
        feature_pngs,
        key=lambda p: (
            0
            if _plot_source_from_name(p.name) == "raw"
            else (1 if _plot_source_from_name(p.name) == "clean" else 2),
            _plot_kind_rank(p.name),
            _plot_exclusion_level(p.name),
            p.name,
        ),
    )


def _feature_av_param_keys(feature: str) -> list[str]:
    feature = str(feature)
    common_graph = [
        "AXON_RECON_AV_MAX_DISTANCE_FOR_EDGE",
        "AXON_RECON_AV_MAX_DISTANCE_TO_INIT",
        "AXON_RECON_AV_N_NEIGHBORS",
        "AXON_RECON_AV_DISTANCE_EXP",
    ]
    if feature == "length":
        return [
            "AXON_RECON_AV_MIN_PATH_LENGTH",
            "AXON_RECON_AV_MIN_PATH_POINTS",
            "AXON_RECON_AV_MIN_POINTS_AFTER_BRANCHING",
            "AXON_RECON_AV_SPLIT_PATHS",
            "AXON_RECON_AV_MAX_PEAK_LATENCY_FOR_SPLITTING",
            *common_graph,
        ]
    if feature == "velocity":
        return [
            "AXON_RECON_AV_R2_THRESHOLD",
            "AXON_RECON_AV_R2_THRESHOLD_FOR_OUTLIERS",
            "AXON_RECON_AV_THEILSEN_MAXITER",
            "AXON_RECON_AV_MAD_THRESHOLD",
            "AXON_RECON_AV_MIN_OUTLIER_TRACKING_ERROR",
            "AXON_RECON_AV_UPSAMPLE",
            "AXON_RECON_AV_MAX_PEAK_LATENCY_FOR_SPLITTING",
            *common_graph,
        ]
    if feature == "area":
        return [
            "AXON_RECON_AV_DETECT_THRESHOLD",
            "AXON_RECON_AV_DETECTION_TYPE",
            "AXON_RECON_AV_KURT_THRESHOLD",
            "AXON_RECON_AV_REMOVE_ISOLATED",
            "AXON_RECON_AV_MIN_SELECTED_POINTS",
            "AXON_RECON_AV_NEIGHBOR_RADIUS",
            *common_graph,
        ]
    if feature == "n_counts":
        return [
            "AXON_RECON_AV_MIN_POINTS_AFTER_BRANCHING",
            "AXON_RECON_AV_SPLIT_PATHS",
            "AXON_RECON_AV_MAX_PEAK_LATENCY_FOR_SPLITTING",
            "AXON_RECON_AV_MIN_PATH_POINTS",
            "AXON_RECON_AV_MIN_PATH_LENGTH",
            "AXON_RECON_AV_N_NEIGHBORS",
            "AXON_RECON_AV_DISTANCE_EXP",
        ]
    return []


def _av_default_values() -> dict[str, str]:
    return {
        "AXON_RECON_AV_UPSAMPLE": "1",
        "AXON_RECON_AV_INIT_DELAY": "0",
        "AXON_RECON_AV_DETECT_THRESHOLD": "0.01",
        "AXON_RECON_AV_DETECTION_TYPE": "relative",
        "AXON_RECON_AV_KURT_THRESHOLD": "0.3",
        "AXON_RECON_AV_PEAK_STD_THRESHOLD": "null",
        "AXON_RECON_AV_PEAK_STD_DISTANCE": "30",
        "AXON_RECON_AV_REMOVE_ISOLATED": "true",
        "AXON_RECON_AV_MIN_SELECTED_POINTS": "30",
        "AXON_RECON_AV_MIN_PATH_LENGTH": "100",
        "AXON_RECON_AV_MIN_PATH_POINTS": "5",
        "AXON_RECON_AV_MIN_POINTS_AFTER_BRANCHING": "3",
        "AXON_RECON_AV_MAX_DISTANCE_FOR_EDGE": "300",
        "AXON_RECON_AV_MAX_DISTANCE_TO_INIT": "200",
        "AXON_RECON_AV_N_NEIGHBORS": "3",
        "AXON_RECON_AV_R2_THRESHOLD": "0.9",
        "AXON_RECON_AV_MAD_THRESHOLD": "8",
        "AXON_RECON_AV_INIT_AMP_PEAK_RATIO": "0.2",
        "AXON_RECON_AV_EDGE_DIST_AMP_RATIO": "0.3",
        "AXON_RECON_AV_DISTANCE_EXP": "2",
        "AXON_RECON_AV_MAX_PEAK_LATENCY_FOR_SPLITTING": "0.5",
        "AXON_RECON_AV_R2_THRESHOLD_FOR_OUTLIERS": "0.98",
        "AXON_RECON_AV_MIN_OUTLIER_TRACKING_ERROR": "50",
        "AXON_RECON_AV_THEILSEN_MAXITER": "2000",
        "AXON_RECON_AV_NEIGHBOR_RADIUS": "100",
        "AXON_RECON_AV_SPLIT_PATHS": "true",
    }


def _format_av_value_with_default(*, key: str, values: dict[str, str]) -> str:
    if key in values:
        return str(values[key])
    defaults = _av_default_values()
    if key in defaults:
        return f"{defaults[key]} (default)"
    return "<unset>"


def _add_recording_info_slide(
    *,
    prs: Any,
    cfg: CrossWellConfig,
    config_path: Path,
    title: str,
) -> None:
    from pptx.util import Inches, Pt

    blank_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(blank_layout)
    box = slide.shapes.add_textbox(Inches(0.6), Inches(0.4), Inches(12.2), Inches(6.8))
    tf = box.text_frame
    tf.word_wrap = True

    p0 = tf.paragraphs[0]
    p0.text = title
    p0.font.size = Pt(28)
    p0.font.bold = True

    def add_line(text: str, *, size: int = 14, bold: bool = False) -> None:
        para = tf.add_paragraph()
        para.text = text
        para.font.size = Pt(size)
        para.font.bold = bool(bold)

    add_line(f"Config: {config_path}")
    add_line(f"Analysis name: {cfg.analysis_name}")
    add_line(f"Out dir: {cfg.out_dir}")
    add_line(f"Electrode pitch (um): {cfg.electrode_pitch_um}")
    if cfg.runtime_env_file is not None:
        add_line(f"Runtime env file: {cfg.runtime_env_file}")
    add_line(f"Generated: {datetime.now().isoformat(timespec='seconds')}")

    for di, d in enumerate(cfg.datasets, start=1):
        add_line(f"Dataset {di}:", bold=True)
        add_line(f"  raw_data_h5_path: {d.raw_data_h5_path}")
        add_line(f"  inferred_output_root: {d.dataset_output_root}")
        if d.wells:
            add_line("  wells (density order from config):")
            for w in d.wells:
                add_line(
                    f"    - {w.well_id}: {w.condition} (density={w.plating_density_nbp}, genotype={w.genotype})",
                    size=12,
                )
            stream_id = d.wells[0].well_id
            si_info, si_err = _try_get_spikeinterface_recording_info(h5_path=d.raw_data_h5_path, stream_id=stream_id)
            if si_info:
                add_line(f"  recording info (SpikeInterface, stream={stream_id}):")
                for k in ["sampling_frequency_hz", "num_channels", "num_segments", "duration_s_total"]:
                    if k in si_info:
                        add_line(f"    - {k}: {si_info[k]}", size=12)
            elif si_err is not None:
                add_line(f"  recording info unavailable ({stream_id}): {si_err}", size=12)


def _add_feature_av_params_slide(
    *,
    prs: Any,
    feature: str,
    env_path: Path | None,
) -> None:
    from pptx.util import Inches, Pt

    feature_title = feature.replace("_", " ")
    values = _parse_env_assignments(env_path) if env_path is not None else {}
    keys = _feature_av_param_keys(feature)

    blank_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(blank_layout)
    box = slide.shapes.add_textbox(Inches(0.6), Inches(0.4), Inches(12.2), Inches(6.8))
    tf = box.text_frame
    tf.word_wrap = True

    p0 = tf.paragraphs[0]
    p0.text = f"Axon velocity params for {feature_title}"
    p0.font.size = Pt(28)
    p0.font.bold = True

    def add_line(text: str, *, size: int = 14, bold: bool = False) -> None:
        para = tf.add_paragraph()
        para.text = text
        para.font.size = Pt(size)
        para.font.bold = bool(bold)

    add_line(f"Source env: {env_path if env_path is not None else '<none>'}")
    add_line("Relevant AXON_RECON_AV_* keys:", bold=True)
    if not keys:
        add_line("  - none configured for this feature", size=12)
    else:
        for key in keys:
            val = _format_av_value_with_default(key=key, values=values)
            add_line(f"  - {key} = {val}", size=12)


def _convert_pptx_to_pdf(*, deck_path: Path, out_dir: Path) -> Path | None:
    import shutil
    import subprocess

    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if not soffice:
        return None
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
        str(deck_path),
    ]
    try:
        subprocess.run(cmd, check=False, capture_output=True, text=True)
    except Exception:
        return None
    pdf_path = out_dir / f"{deck_path.stem}.pdf"
    return pdf_path if pdf_path.exists() else None


def write_feature_slide_decks(*, out_dir: Path, cfg: CrossWellConfig, plots_dir: Path, config_path: Path) -> list[Path]:
    """Write one deck per feature (length/velocity/area/n_branches), plus PDF conversions."""

    import logging

    logger = logging.getLogger("projects.cross_well_feature_decks")

    try:
        from pptx import Presentation  # type: ignore[import-not-found]
        from PIL import Image  # type: ignore[import-not-found]
    except Exception as e:
        logger.warning("Skipping feature deck export (missing deps): %s", e)
        return []

    features = ["length", "velocity", "area", "n_counts"]
    written: list[Path] = []

    for feature in features:
        feature_pngs = _sorted_feature_plots(plots_dir=plots_dir, feature=feature)
        if not feature_pngs:
            continue

        prs = Presentation()
        prs.slide_width = 9144000 * 4 // 3
        prs.slide_height = 6858000
        slide_w = int(prs.slide_width)
        slide_h = int(prs.slide_height)

        blank_layout = prs.slide_layouts[6]
        for png_path in feature_pngs:
            s = prs.slides.add_slide(blank_layout)
            with Image.open(png_path) as im:
                iw, ih = im.size
            img_ratio = iw / float(ih)
            slide_ratio = slide_w / float(slide_h)

            if img_ratio >= slide_ratio:
                pic_w = slide_w
                pic_h = int(slide_w / img_ratio)
                left = 0
                top = int((slide_h - pic_h) / 2)
            else:
                pic_h = slide_h
                pic_w = int(slide_h * img_ratio)
                top = 0
                left = int((slide_w - pic_w) / 2)

            s.shapes.add_picture(str(png_path), left, top, width=pic_w, height=pic_h)
            try:
                s.notes_slide.notes_text_frame.text = png_path.name
            except Exception:
                pass

        _add_recording_info_slide(
            prs=prs,
            cfg=cfg,
            config_path=config_path,
            title=f"{feature.replace('_', ' ').title()} summary",
        )
        _add_feature_av_params_slide(
            prs=prs,
            feature=feature,
            env_path=cfg.runtime_env_file,
        )

        deck_path = Path(out_dir) / f"cross_well_{feature}_summary.pptx"
        prs.save(str(deck_path))
        written.append(deck_path)

        pdf_path = _convert_pptx_to_pdf(deck_path=deck_path, out_dir=Path(out_dir))
        if pdf_path is not None:
            written.append(pdf_path)

    return written


def write_feature_pdf_decks(*, out_dir: Path, cfg: CrossWellConfig, plots_dir: Path, config_path: Path) -> list[Path]:
    """Write one PDF deck per feature without relying on LibreOffice."""

    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    features = ["length", "velocity", "area", "n_counts"]
    written: list[Path] = []

    for feature in features:
        feature_pngs = _sorted_feature_plots(plots_dir=plots_dir, feature=feature)
        if not feature_pngs:
            continue

        pdf_path = Path(out_dir) / f"cross_well_{feature}_summary.pdf"
        values = _parse_env_assignments(cfg.runtime_env_file) if cfg.runtime_env_file is not None else {}
        keys = _feature_av_param_keys(feature)

        with PdfPages(pdf_path) as pdf:
            # Plot pages first
            for png in feature_pngs:
                img = plt.imread(str(png))
                fig = plt.figure(figsize=(13.333, 7.5), dpi=150)
                fig.patch.set_facecolor("white")
                ax = fig.add_subplot(111)
                ax.imshow(img)
                ax.set_title(png.name, fontsize=10)
                ax.axis("off")
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

            # Recording/runtime info page at end
            fig = plt.figure(figsize=(13.333, 7.5), dpi=150)
            fig.patch.set_facecolor("white")
            ax = fig.add_subplot(111)
            ax.axis("off")

            lines: list[str] = [
                f"{feature.replace('_', ' ').title()} summary",
                "",
                f"Config: {config_path}",
                f"Analysis name: {cfg.analysis_name}",
                f"Out dir: {cfg.out_dir}",
                f"Electrode pitch (um): {cfg.electrode_pitch_um}",
                f"Runtime env file: {cfg.runtime_env_file if cfg.runtime_env_file is not None else '<none>'}",
                f"Generated: {datetime.now().isoformat(timespec='seconds')}",
                "",
            ]
            for di, d in enumerate(cfg.datasets, start=1):
                lines.append(f"Dataset {di}:")
                lines.append(f"  raw_data_h5_path: {d.raw_data_h5_path}")
                lines.append(f"  inferred_output_root: {d.dataset_output_root}")
                if d.wells:
                    lines.append("  wells:")
                    for w in d.wells:
                        lines.append(
                            f"    - {w.well_id}: {w.condition} (density={w.plating_density_nbp}, genotype={w.genotype})"
                        )
            ax.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=10, family="monospace")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            # Feature-relevant AV params page at end
            fig = plt.figure(figsize=(13.333, 7.5), dpi=150)
            fig.patch.set_facecolor("white")
            ax = fig.add_subplot(111)
            ax.axis("off")
            param_lines = [
                f"Axon velocity params for {feature.replace('_', ' ')}",
                "",
                f"Source env: {cfg.runtime_env_file if cfg.runtime_env_file is not None else '<none>'}",
                "",
                "Relevant AXON_RECON_AV_* keys:",
            ]
            if keys:
                for key in keys:
                    param_lines.append(f"  - {key} = {_format_av_value_with_default(key=key, values=values)}")
            else:
                param_lines.append("  - none configured for this feature")
            ax.text(0.02, 0.98, "\n".join(param_lines), va="top", ha="left", fontsize=11, family="monospace")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        written.append(pdf_path)

    return written


def write_cross_well_slide_deck(*, out_dir: Path, cfg: CrossWellConfig, plots_dir: Path, config_path: Path) -> Path | None:
    """Write a PPTX deck with a summary slide + one slide per plot PNG."""

    import logging
    import shutil
    import subprocess
    from datetime import datetime

    logger = logging.getLogger("projects.cross_well_deck")

    try:
        from pptx import Presentation  # type: ignore[import-not-found]
        from pptx.util import Inches, Pt
        from PIL import Image  # type: ignore[import-not-found]
    except Exception as e:
        logger.warning("Skipping slide deck export (missing deps): %s", e)
        return None

    deck_path = Path(out_dir) / "cross_well_summary.pptx"
    pdf_out_dir = Path(out_dir)

    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    slide_w = int(prs.slide_width)
    slide_h = int(prs.slide_height)

    # --- Summary slide ---
    _add_recording_info_slide(
        prs=prs,
        cfg=cfg,
        config_path=config_path,
        title="Cross-well summary",
    )

    # --- Plot slides ---
    blank_layout = prs.slide_layouts[6]
    plot_paths = sorted([p for p in Path(plots_dir).glob("*.png") if p.is_file()])
    for png_path in plot_paths:
        s = prs.slides.add_slide(blank_layout)
        with Image.open(png_path) as im:
            iw, ih = im.size
        img_ratio = iw / float(ih)
        slide_ratio = slide_w / float(slide_h)

        if img_ratio >= slide_ratio:
            pic_w = slide_w
            pic_h = int(slide_w / img_ratio)
            left = 0
            top = int((slide_h - pic_h) / 2)
        else:
            pic_h = slide_h
            pic_w = int(slide_h * img_ratio)
            top = 0
            left = int((slide_w - pic_w) / 2)

        s.shapes.add_picture(str(png_path), left, top, width=pic_w, height=pic_h)
        try:
            s.notes_slide.notes_text_frame.text = png_path.name
        except Exception:
            pass

    prs.save(str(deck_path))

    # Best-effort PDF export (LibreOffice)
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if soffice:
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
            str(pdf_out_dir),
            str(deck_path),
        ]
        try:
            subprocess.run(cmd, check=False, capture_output=True, text=True)
        except Exception:
            pass

    return deck_path
