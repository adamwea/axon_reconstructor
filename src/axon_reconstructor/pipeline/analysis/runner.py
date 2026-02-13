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


def _pick_first_existing(paths: list[Path]) -> Optional[Path]:
    for p in paths:
        if p and Path(p).exists():
            return Path(p)
    return None


def _svg_supported() -> bool:
    try:
        import cairosvg  # type: ignore[import-not-found]  # noqa: F401

        return True
    except Exception:
        return False


def _prefer_svg_if_available(*, path: Optional[Path], svg_supported: bool) -> Optional[Path]:
    if path is None:
        return None
    path = Path(path)
    if (not svg_supported) or (path.suffix.lower() == ".svg"):
        return path
    svg = path.with_suffix(".svg")
    if svg.exists():
        return svg
    return path


def _autocrop_white_rgba(*, img: Any, pad_frac: float = 0.06) -> Any:
    """Crop near-white borders from an RGBA PIL image (best effort)."""

    from PIL import Image  # type: ignore[import-not-found]
    import numpy as np  # type: ignore[import-not-found]

    if not isinstance(img, Image.Image):
        raise TypeError("expected PIL.Image")
    rgba = img.convert("RGBA")
    arr = np.asarray(rgba)
    if arr.ndim != 3 or arr.shape[2] != 4:
        return rgba

    rgb = arr[:, :, :3]
    a = arr[:, :, 3]
    # Non-background mask: opaque-ish and not near-white.
    mask = (a > 10) & (np.any(rgb < 245, axis=2))
    if not np.any(mask):
        # Fall back to alpha-only bbox.
        mask = a > 10
    if not np.any(mask):
        return rgba

    ys, xs = np.where(mask)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    h, w = arr.shape[0], arr.shape[1]
    pad_x = int(round((x1 - x0 + 1) * pad_frac))
    pad_y = int(round((y1 - y0 + 1) * pad_frac))
    x0 = max(0, x0 - pad_x)
    x1 = min(w - 1, x1 + pad_x)
    y0 = max(0, y0 - pad_y)
    y1 = min(h - 1, y1 + pad_y)
    return rgba.crop((x0, y0, x1 + 1, y1 + 1))


def _render_zoomed_copy(
    *,
    src: Path,
    dst: Path,
    force_restart: bool,
    logger: logging.Logger,
) -> tuple[Optional[Path], Optional[str]]:
    """Create a zoomed-in (auto-cropped) copy of an image under the analysis output tree."""

    dst = Path(dst)
    if (not force_restart) and dst.exists():
        return dst, None

    img, err = _try_load_image_pil(path=src, logger=logger)
    if img is None:
        return None, err

    try:
        zoomed = _autocrop_white_rgba(img=img)
        dst.parent.mkdir(parents=True, exist_ok=True)
        zoomed.save(dst, format="PNG")
        return dst, None
    except Exception as e:
        return None, f"zoom-crop failed: {e}"


def _placeholder_rgba(*, size: tuple[int, int], text: str) -> Any:
    from PIL import Image, ImageDraw, ImageFont  # type: ignore[import-not-found]

    w, h = size
    img = Image.new("RGBA", (w, h), (255, 255, 255, 255))
    draw = ImageDraw.Draw(img, "RGBA")
    font = ImageFont.load_default()
    draw.multiline_text((w // 2, h // 2), text, fill=(0, 0, 0, 255), anchor="mm", align="center", font=font)
    return img


def _compose_row_figure(
    *,
    paths: list[Optional[Path]],
    width_ratios: list[int],
    out_png: Path,
    height_px: int,
    force_restart: bool,
    logger: logging.Logger,
) -> tuple[Optional[Path], Optional[str]]:
    """Compose a 1-row multi-panel PNG with exact width ratios (aspect-preserving).

    Images are auto-cropped to remove near-white borders and then fit into their
    allocated slots without distortion.
    """

    out_png = Path(out_png)
    if (not force_restart) and out_png.exists():
        return out_png, None

    try:
        from PIL import Image  # type: ignore[import-not-found]
    except Exception as e:
        return None, f"pillow unavailable: {e}"

    if len(paths) != len(width_ratios):
        return None, "paths/width_ratios length mismatch"

    try:
        resample = Image.Resampling.LANCZOS  # Pillow>=9
    except Exception:
        resample = Image.LANCZOS

    # Load + autocrop panels first.
    loaded: list[tuple[Optional[Any], str]] = []
    for p in paths:
        if p is None:
            loaded.append((None, "missing panel"))
            continue
        img, err = _try_load_image_pil(path=Path(p), logger=logger)
        if img is None:
            loaded.append((None, f"{Path(p).name} ({err})"))
            continue
        try:
            img = _autocrop_white_rgba(img=img)
        except Exception:
            pass
        loaded.append((img, ""))

    # Determine slot widths from ratios, anchored to the first available panel width.
    # This preserves the intended 2:1:1 and 3:1:1 feel without stretching.
    base_idx = next((i for i, (im, _) in enumerate(loaded) if im is not None), 0)
    base_im = loaded[base_idx][0]
    if base_im is None:
        return None, "all panels missing"

    # Scale the base image to the target height.
    base_w = int(round(base_im.size[0] * (height_px / max(1, base_im.size[1]))))
    unit_w = max(1, int(round(base_w / max(1, width_ratios[base_idx]))))
    slot_ws = [int(r * unit_w) for r in width_ratios]
    total_w = int(sum(slot_ws))

    def _fit_into_box(im: Any, box_w: int, box_h: int) -> Any:
        # Preserve aspect ratio; allow downscale only.
        w, h = im.size
        if (w <= 0) or (h <= 0) or (box_w <= 0) or (box_h <= 0):
            return im
        scale = min(box_w / w, box_h / h, 1.0)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        return im.resize((new_w, new_h), resample=resample)

    panels: list[Any] = []
    for (im, msg), box_w in zip(loaded, slot_ws):
        if im is None:
            panels.append(_placeholder_rgba(size=(box_w, height_px), text=msg or "missing"))
            continue
        fitted = _fit_into_box(im, box_w, height_px)
        slot = Image.new("RGBA", (box_w, height_px), (255, 255, 255, 255))
        px = (box_w - fitted.size[0]) // 2
        py = (height_px - fitted.size[1]) // 2
        slot.alpha_composite(fitted, dest=(px, py))
        panels.append(slot)

    canvas = Image.new("RGBA", (total_w, height_px), (255, 255, 255, 255))
    x = 0
    for panel in panels:
        canvas.alpha_composite(panel, dest=(x, 0))
        x += int(panel.size[0])

    # Trim outer whitespace a bit.
    try:
        canvas = _autocrop_white_rgba(img=canvas, pad_frac=0.02)
    except Exception:
        pass

    out_png.parent.mkdir(parents=True, exist_ok=True)
    try:
        canvas.save(out_png, format="PNG")
    except Exception as e:
        return None, f"save failed: {e}"
    return out_png, None


def _compose_row_figure_natural(
    *,
    paths: list[Optional[Path]],
    out_png: Path,
    height_px: int,
    force_restart: bool,
    logger: logging.Logger,
) -> tuple[Optional[Path], Optional[str]]:
    """Compose a 1-row multi-panel PNG, scaling each panel to a common height.

    This respects each panel's native aspect ratio (no slot-width forcing).
    """

    out_png = Path(out_png)
    if (not force_restart) and out_png.exists():
        return out_png, None

    try:
        from PIL import Image  # type: ignore[import-not-found]
    except Exception as e:
        return None, f"pillow unavailable: {e}"

    try:
        resample = Image.Resampling.LANCZOS
    except Exception:
        resample = Image.LANCZOS

    ims: list[Any] = []
    for p in paths:
        if p is None:
            ims.append(_placeholder_rgba(size=(height_px, height_px), text="missing panel"))
            continue
        img, err = _try_load_image_pil(path=Path(p), logger=logger)
        if img is None:
            ims.append(_placeholder_rgba(size=(height_px, height_px), text=f"{Path(p).name}\n({err})"))
            continue
        try:
            img = _autocrop_white_rgba(img=img)
        except Exception:
            pass
        w, h = img.size
        new_w = max(1, int(round(w * (height_px / max(1, h)))))
        ims.append(img.resize((new_w, height_px), resample=resample))

    total_w = int(sum(im.size[0] for im in ims))
    canvas = Image.new("RGBA", (total_w, height_px), (255, 255, 255, 255))
    x = 0
    for im in ims:
        canvas.alpha_composite(im, dest=(x, 0))
        x += int(im.size[0])
    try:
        canvas = _autocrop_white_rgba(img=canvas, pad_frac=0.02)
    except Exception:
        pass

    out_png.parent.mkdir(parents=True, exist_ok=True)
    try:
        canvas.save(out_png, format="PNG")
    except Exception as e:
        return None, f"save failed: {e}"
    return out_png, None


def _compose_fig3_recon_and_velocity_two_row(
    *,
    raw_branches_png: Optional[Path],
    branch_velocities_png: Optional[Path],
    propagation_png: Optional[Path],
    out_png: Path,
    height_px: int,
    force_restart: bool,
    logger: logging.Logger,
) -> tuple[Optional[Path], Optional[str]]:
    """Compose Fig 3 as a 1x3 layout.

    Layout (1 row, 3 columns; col ratios [3,1,1]):
    - Left: raw branches (full height)
    - Middle: branch velocities (full height)
    - Right: propagation plot (full height)

    Note: channel selection is intentionally excluded.
    """

    out_png = Path(out_png)
    if (not force_restart) and out_png.exists():
        return out_png, None

    try:
        from PIL import Image  # type: ignore[import-not-found]
    except Exception as e:
        return None, f"pillow unavailable: {e}"

    try:
        resample = Image.Resampling.LANCZOS  # Pillow>=9
    except Exception:
        resample = Image.LANCZOS

    # Column ratios: [3,1,1]
    col_ratios = [3, 1, 1]

    def _load_autocrop(p: Optional[Path], *, missing_msg: str) -> Any:
        if p is None:
            return _placeholder_rgba(size=(height_px, height_px), text=missing_msg)
        img, err = _try_load_image_pil(path=Path(p), logger=logger)
        if img is None:
            return _placeholder_rgba(size=(height_px, height_px), text=f"{Path(p).name}\n({err})")
        try:
            img = _autocrop_white_rgba(img=img)
        except Exception:
            pass
        return img

    raw_im = _load_autocrop(raw_branches_png, missing_msg="missing raw branches")
    vel_im = _load_autocrop(branch_velocities_png, missing_msg="missing branch velocities")
    prop_im = _load_autocrop(propagation_png, missing_msg="missing propagation")

    # Compute column widths anchored to the raw branches panel (spans full height).
    # Scale raw to height_px; infer unit width from its col ratio (3).
    raw_w = int(round(raw_im.size[0] * (height_px / max(1, raw_im.size[1]))))
    unit_w = max(1, int(round(raw_w / col_ratios[0])))
    col_ws = [int(r * unit_w) for r in col_ratios]
    W = int(sum(col_ws))

    def _fit_into_box(im: Any, box_w: int, box_h: int) -> Any:
        w, h = im.size
        if (w <= 0) or (h <= 0) or (box_w <= 0) or (box_h <= 0):
            return im
        # Preserve aspect ratio; allow upscaling so thin/tall panels can
        # actually fill the allocated slots (especially when SVG sources
        # are available).
        scale = min(box_w / w, box_h / h)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        return im.resize((new_w, new_h), resample=resample)

    # Slots
    left_w = int(col_ws[0])
    mid_w = int(col_ws[1])
    right_w = int(col_ws[2])

    canvas = Image.new("RGBA", (W, height_px), (255, 255, 255, 255))

    # Left: raw branches (full height)
    raw_f = _fit_into_box(raw_im, left_w, height_px)
    slot_left = Image.new("RGBA", (left_w, height_px), (255, 255, 255, 255))
    slot_left.alpha_composite(
        raw_f,
        dest=((left_w - raw_f.size[0]) // 2, (height_px - raw_f.size[1]) // 2),
    )
    canvas.alpha_composite(slot_left, dest=(0, 0))

    # Middle: velocities (full height)
    vel_f = _fit_into_box(vel_im, mid_w, height_px)
    slot_mid = Image.new("RGBA", (mid_w, height_px), (255, 255, 255, 255))
    slot_mid.alpha_composite(
        vel_f,
        dest=((mid_w - vel_f.size[0]) // 2, (height_px - vel_f.size[1]) // 2),
    )
    canvas.alpha_composite(slot_mid, dest=(left_w, 0))

    # Right: propagation (full height)
    prop_f = _fit_into_box(prop_im, right_w, height_px)
    slot_right = Image.new("RGBA", (right_w, height_px), (255, 255, 255, 255))
    slot_right.alpha_composite(
        prop_f,
        dest=((right_w - prop_f.size[0]) // 2, (height_px - prop_f.size[1]) // 2),
    )
    canvas.alpha_composite(slot_right, dest=(left_w + mid_w, 0))

    try:
        canvas = _autocrop_white_rgba(img=canvas, pad_frac=0.02)
    except Exception:
        pass

    out_png.parent.mkdir(parents=True, exist_ok=True)
    try:
        canvas.save(out_png, format="PNG")
    except Exception as e:
        return None, f"save failed: {e}"

    return out_png, None


def _compose_unit_summary_from_figs(
    *,
    fig3_png: Path,
    fig2_png: Path,
    out_png: Path,
    out_pdf: Path,
    force_restart: bool,
    logger: logging.Logger,
) -> tuple[Optional[Path], Optional[Path], Optional[str]]:
    """Final summary layout:

    - Top row: Fig 3 spans full width
    - Bottom row: Fig 2 spans full width
    """

    out_png = Path(out_png)
    out_pdf = Path(out_pdf)
    if (not force_restart) and out_png.exists() and out_pdf.exists():
        return out_png, out_pdf, None

    try:
        from PIL import Image  # type: ignore[import-not-found]
    except Exception as e:
        return None, None, f"pillow unavailable: {e}"

    top, err_top = _try_load_image_pil(path=Path(fig3_png), logger=logger)
    right, err_right = _try_load_image_pil(path=Path(fig2_png), logger=logger)
    if top is None:
        return None, None, f"missing Fig3 ({err_top})"
    if right is None:
        return None, None, f"missing Fig2 ({err_right})"

    try:
        resample = Image.Resampling.LANCZOS
    except Exception:
        resample = Image.LANCZOS

    # Anchor overall width to Fig3, scaling Fig3 down if needed.
    W = int(top.size[0])
    top_h = int(top.size[1])

    def _resize_to_width(im: Any, new_w: int) -> Any:
        w, h = im.size
        if w <= 0 or h <= 0:
            return im
        if w == new_w:
            return im
        new_h = max(1, int(round(h * (new_w / w))))
        return im.resize((new_w, new_h), resample=resample)

    right_r = _resize_to_width(right, W)
    bottom_h = int(right_r.size[1])

    canvas = Image.new("RGBA", (W, top_h + bottom_h), (255, 255, 255, 255))
    canvas.alpha_composite(top, dest=(0, 0))
    canvas.alpha_composite(right_r, dest=(0, top_h))

    try:
        canvas = _autocrop_white_rgba(img=canvas, pad_frac=0.01)
    except Exception:
        pass

    out_png.parent.mkdir(parents=True, exist_ok=True)
    try:
        canvas.save(out_png, format="PNG")
        canvas.convert("RGB").save(out_pdf, format="PDF", resolution=200.0)
    except Exception as e:
        return None, None, f"save failed: {e}"

    return out_png, out_pdf, None


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

    # Optional BOTM validation (raw-snippet, analysis-stage only)
    compute_botm_validation: bool = False
    botm_n_events: int = 200
    botm_n_noise_windows: int = 2000
    botm_seed: int = 0
    botm_prior_signal: float = 0.5
    botm_match_fraction_threshold: float = 0.70
    botm_sorter: str = "kilosort4"

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
    templates_dir = templates_out_dir / "templates"
    merged_units_dir = templates_dir / "merged"
    if not merged_units_dir.exists():
        legacy = templates_out_dir / "merged_units"
        if legacy.exists():
            merged_units_dir = legacy
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

    # Optional BOTM validation run (writes separate artifacts; does not alter grid montage).
    botm_summary_json: Optional[Path] = None
    botm_out_dir: Optional[Path] = None
    if bool(getattr(inputs, "compute_botm_validation", False)):
        from .botm_validation import BotmValidationInputs, write_botm_validation_outputs

        botm_out_dir = analysis_out_dir / "botm_validation"
        botm_inputs = BotmValidationInputs(
            well_out_dir=well_out_dir,
            h5_path=Path(inputs.h5_path),
            stream_id=str(inputs.stream_id),
            unit_ids=list(unit_ids),
            n_events=int(getattr(inputs, "botm_n_events", 200)),
            n_noise_windows=int(getattr(inputs, "botm_n_noise_windows", 2000)),
            seed=int(getattr(inputs, "botm_seed", 0)),
            prior_signal=float(getattr(inputs, "botm_prior_signal", 0.5)),
            match_fraction_threshold=float(getattr(inputs, "botm_match_fraction_threshold", 0.70)),
            sorter=str(getattr(inputs, "botm_sorter", "kilosort4")),
            out_dir=botm_out_dir,
            force_restart=bool(inputs.force_restart),
        )

        logger.info(
            "BOTM validation enabled: out_dir=%s n_units=%d seed=%s n_events=%d n_noise_windows=%d",
            str(botm_out_dir),
            int(len(unit_ids)),
            str(getattr(inputs, "botm_seed", 0)),
            int(getattr(inputs, "botm_n_events", 200)),
            int(getattr(inputs, "botm_n_noise_windows", 2000)),
        )

        botm_summary = write_botm_validation_outputs(inputs=botm_inputs, logger=logger)
        try:
            botm_summary_json = Path(botm_summary.get("out_dir")) / "summary.json"
        except Exception:
            botm_summary_json = botm_out_dir / "summary.json"

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

    if botm_out_dir is not None:
        summary["botm_validation"] = {
            "out_dir": str(botm_out_dir),
            "summary_json": (str(botm_summary_json) if botm_summary_json is not None else None),
            "enabled": True,
        }
    else:
        summary["botm_validation"] = {
            "enabled": False,
        }

    for uid in unit_ids:
        out_unit_dir = by_unit_dir / f"unit_{uid}"
        out_unit_dir.mkdir(parents=True, exist_ok=True)

        # HARD REVISION: build a few multi-panel figures first, then compose the final
        # unit summary grid from those figures. This improves spacing control.
        footprints_root = templates_out_dir / "footprints"
        topo_dir = footprints_root / "3D"
        if not topo_dir.exists():
            legacy = templates_out_dir / "topo_unit_footprints"
            if legacy.exists():
                topo_dir = legacy

        footprints_zoomed_dir = footprints_root / "zoomed"
        if not footprints_zoomed_dir.exists():
            legacy = templates_out_dir / "footprints_zoomed"
            if legacy.exists():
                footprints_zoomed_dir = legacy

        # Reconstruction outputs were reorganized under per-unit subdirs.
        recon_unit_dir = recon_by_unit_root / f"unit_{uid}"
        recon_branches_clean = recon_unit_dir / "branches" / "clean"
        recon_branches_raw = recon_unit_dir / "branches" / "raw"

        panels_dir = out_unit_dir / "panels"
        panels_dir.mkdir(parents=True, exist_ok=True)

        # Clean up legacy analysis-generated artifacts.
        # Topo zoom is now rendered in templates stage as:
        #   templates_outputs/footprints/3D/unit_<id>_zoom.png
        if bool(inputs.force_restart):
            try:
                legacy = panels_dir / "topo3d_zoom.png"
                if legacy.exists():
                    legacy.unlink()
            except Exception:
                pass
            try:
                legacy = panels_dir / "fig1_template_views.png"
                if legacy.exists():
                    legacy.unlink()
            except Exception:
                pass
            try:
                legacy = panels_dir / "channel_selection_all_zoom.png"
                if legacy.exists():
                    legacy.unlink()
            except Exception:
                pass

        # Source images (best-effort fallbacks for minor naming drift)
        svg_supported = _svg_supported()
        fp_ptp_log = _pick_first_existing(
            [
                (footprints_root / "full") / f"unit_{uid}_merged_contributing_footprint_ptp_log.png",
                (footprints_root / "full") / f"unit_{uid}_merged_contributing_footprint_ptp_log_linear.png",
                footprints_zoomed_dir / f"unit_{uid}_merged_contributing_footprint_ptp_log.png",
                footprints_zoomed_dir / f"unit_{uid}_merged_contributing_footprint_ptp_linear_zoom.png",
                footprints_zoomed_dir / f"unit_{uid}_merged_contributing_footprint_ptp_linear.png",
            ]
        )
        fp_ptp_log = _prefer_svg_if_available(path=fp_ptp_log, svg_supported=svg_supported)

        # Fig1 is intentionally disabled for now.

        edges_png = _pick_first_existing(
            [
                recon_unit_dir / "maps" / "graph_edges.png",
                recon_unit_dir / "graph_edges.png",
            ]
        )
        edges_png = _prefer_svg_if_available(path=edges_png, svg_supported=svg_supported)

        chan_all_src = _pick_first_existing(
            [
                recon_unit_dir / "maps" / "channel_selection_all.png",
                recon_unit_dir / "channel_selection_all.png",
            ]
        )
        chan_all_src = _prefer_svg_if_available(path=chan_all_src, svg_supported=svg_supported)

        raw_zoom = _pick_first_existing(
            [
                recon_branches_raw / "branches_raw_zoom.png",
                recon_unit_dir / "branches_raw_zoom.png",
            ]
        )
        raw_zoom = _prefer_svg_if_available(path=raw_zoom, svg_supported=svg_supported)

        vel_overlay = _pick_first_existing(
            [
                recon_branches_raw / "branch_velocities_overlay.png",
                recon_branches_clean / "branch_velocities_overlay.png",
                recon_unit_dir / "branch_velocities_overlay.png",
            ]
        )
        vel_overlay = _prefer_svg_if_available(path=vel_overlay, svg_supported=svg_supported)

        propagation = _pick_first_existing(
            [
                templates_out_dir / "propagation_plots" / f"unit_{uid}.png",
                templates_out_dir / "propagation_plots" / f"unit_{uid}_propagation.png",
            ]
        )
        propagation = _prefer_svg_if_available(path=propagation, svg_supported=svg_supported)

        # Note: topo zooming/cropping is intentionally NOT done in analysis.
        # Templates stage is responsible for writing `unit_<id>_zoom.png`.

        # Fig 2 (Edges and Node Candidates): one row, respect native aspect ratios.
        fig2_png = panels_dir / "fig2_edges_and_nodes.png"
        _compose_row_figure_natural(
            # Swap requested: Fig2 now shows footprint (instead of channel selection).
            paths=[edges_png, fp_ptp_log],
            out_png=fig2_png,
            height_px=900,
            force_restart=bool(inputs.force_restart),
            logger=logger,
        )

        # Fig 3 (Reconstruction and Velocity): two rows.
        fig3_png = panels_dir / "fig3_reconstruction_and_velocity.png"
        _compose_fig3_recon_and_velocity_two_row(
            raw_branches_png=raw_zoom,
            branch_velocities_png=vel_overlay,
            propagation_png=propagation,
            out_png=fig3_png,
            height_px=900,
            force_restart=bool(inputs.force_restart),
            logger=logger,
        )

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
            out_png2, out_pdf2, err = _compose_unit_summary_from_figs(
                fig3_png=fig3_png,
                fig2_png=fig2_png,
                out_png=grid_png,
                out_pdf=grid_pdf,
                force_restart=bool(inputs.force_restart),
                logger=logger,
            )
            unit_record["outputs"].update(
                {
                    "fig2_png": str(fig2_png),
                    "fig3_png": str(fig3_png),
                    "grid_png": str(out_png2) if out_png2 else None,
                    "grid_pdf": str(out_pdf2) if out_pdf2 else None,
                }
            )
            if err:
                unit_record["status"] = "error"
                unit_record["error"] = err
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
