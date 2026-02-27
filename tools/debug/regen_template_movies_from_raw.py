#!/usr/bin/env python3
"""Regenerate per-unit template_movie.gif using *raw* branches.

This avoids rerunning full reconstruction plotting (maps, PDFs, etc.).

Inputs (per well):
  - templates_outputs/templates/full/unit_<id>/full_template.npy
  - templates_outputs/templates/full/unit_<id>/full_channel_locations_xy.npy
  - reconstruction_outputs/by_unit/unit_<id>/branches_raw.json

Outputs:
  - reconstruction_outputs/by_unit/unit_<id>/template_movie.gif

Notes:
  - The overlay branches are taken from branches_raw.json (not clean branches).
  - Uses the same env knobs as reconstruction plotting where applicable.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

import debug_env


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _env_flag(name: str, default: bool) -> bool:
    raw = str(os.getenv(name, "")).strip().lower()
    if raw == "":
        return bool(default)
    return raw in {"1", "true", "yes", "y", "on"}


def _env_float(name: str, default: float) -> float:
    raw = str(os.getenv(name, "")).strip()
    if raw == "":
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


def _compute_zoom_limits_from_xy(
    xy_points: list[list[float]],
    *,
    pad_frac: float = 0.08,
    pad_abs: float = 20.0,
) -> tuple[float, float, float, float]:
    xs = [p[0] for p in xy_points if (p is not None and len(p) >= 2)]
    ys = [p[1] for p in xy_points if (p is not None and len(p) >= 2)]
    if not xs or not ys:
        raise ValueError("No XY points")

    xmin, xmax = float(min(xs)), float(max(xs))
    ymin, ymax = float(min(ys)), float(max(ys))

    dx = max(xmax - xmin, 0.0)
    dy = max(ymax - ymin, 0.0)
    pad_x = max(pad_abs, pad_frac * dx)
    pad_y = max(pad_abs, pad_frac * dy)
    if dx == 0.0:
        pad_x = max(pad_x, pad_abs)
    if dy == 0.0:
        pad_y = max(pad_y, pad_abs)

    return xmin - pad_x, xmax + pad_x, ymin - pad_y, ymax + pad_y


def _iter_unit_ids_from_recon_by_unit(by_unit_dir: Path) -> list[int]:
    unit_ids: list[int] = []
    for p in sorted(by_unit_dir.glob("unit_*")):
        if not p.is_dir():
            continue
        try:
            unit_ids.append(int(p.name.split("unit_", 1)[1]))
        except Exception:
            continue
    return unit_ids


def _write_movie_gif_for_unit(
    *,
    uid: int,
    well_out_dir: Path,
    force: bool,
) -> bool:
    # Paths
    recon_unit_dir = well_out_dir / "reconstruction_outputs" / "by_unit" / f"unit_{uid}"
    branches_raw_json = recon_unit_dir / "branches_raw.json"
    out_gif = recon_unit_dir / "template_movie.gif"

    if (not force) and out_gif.exists():
        return True

    if not branches_raw_json.exists():
        return False

    payload = _read_json(branches_raw_json)
    branches_raw = payload.get("branches") or []
    if not isinstance(branches_raw, list) or len(branches_raw) == 0:
        return False

    tmpl_dir = well_out_dir / "templates_outputs" / "templates" / "full" / f"unit_{uid}"
    full_template_npy = tmpl_dir / "full_template.npy"
    full_locs_npy = tmpl_dir / "full_channel_locations_xy.npy"

    if (not full_template_npy.exists()) or (not full_locs_npy.exists()):
        return False

    tmpl = np.load(full_template_npy)
    locs_xy = np.load(full_locs_npy)
    if tmpl.ndim != 2:
        return False
    if locs_xy.ndim != 2 or locs_xy.shape[1] < 2:
        return False

    template_ch_by_t = np.asarray(tmpl).T
    locs_xy = np.asarray(locs_xy)[:, :2]

    # Env knobs (match reconstruction plotting defaults)
    crop = _env_flag("AXON_RECON_RECON_TEMPLATE_MOVIE_GIF_CROP", True)
    zoom = _env_flag("AXON_RECON_RECON_TEMPLATE_MOVIE_GIF_ZOOM", True)
    cmap = str(os.getenv("AXON_RECON_RECON_TEMPLATE_MOVIE_GIF_CMAP", "coolwarm") or "coolwarm")
    clip_q = _env_float("AXON_RECON_RECON_TEMPLATE_MOVIE_GIF_CLIP_QUANTILE", 0.995)
    add_cbar = _env_flag("AXON_RECON_RECON_TEMPLATE_MOVIE_GIF_COLORBAR", True)
    cbar_label = str(os.getenv("AXON_RECON_RECON_TEMPLATE_MOVIE_GIF_COLORBAR_LABEL", "") or "").strip()
    add_time = _env_flag("AXON_RECON_RECON_TEMPLATE_MOVIE_GIF_TIME_COUNTER", True)
    pad_frac = _env_float("AXON_RECON_RECON_TEMPLATE_MOVIE_GIF_ZOOM_PAD_FRAC", 0.08)
    pad_abs = _env_float("AXON_RECON_RECON_TEMPLATE_MOVIE_GIF_ZOOM_PAD_ABS", 20.0)

    # Build branch overlay + contributing channels
    contributing: set[int] = set()
    branch_xy_points: list[list[float]] = []
    cleaned_branches: list[dict[str, Any]] = []

    for br in branches_raw:
        if not isinstance(br, dict):
            continue
        chans = br.get("channels")
        if not isinstance(chans, list):
            continue
        ch_ints: list[int] = []
        for c in chans:
            try:
                ci = int(c)
            except Exception:
                continue
            if 0 <= ci < locs_xy.shape[0]:
                ch_ints.append(ci)
                contributing.add(ci)
                branch_xy_points.append([float(locs_xy[ci, 0]), float(locs_xy[ci, 1])])
        if len(ch_ints) >= 2:
            cleaned_branches.append({"channels": ch_ints})

    if not cleaned_branches:
        return False

    # Optional crop ROI (speed)
    template_for_movie = template_ch_by_t
    locs_for_movie = locs_xy
    gtr_for_movie: Any | None = SimpleNamespace(branches=cleaned_branches, locations=locs_xy)

    if crop and contributing:
        try:
            xy_contrib = [[float(locs_xy[ch, 0]), float(locs_xy[ch, 1])] for ch in sorted(contributing)]
            xmin, xmax, ymin, ymax = _compute_zoom_limits_from_xy(xy_contrib, pad_frac=pad_frac, pad_abs=pad_abs)
            # square
            cx = 0.5 * (xmin + xmax)
            cy = 0.5 * (ymin + ymax)
            w = float(max(xmax - xmin, ymax - ymin))
            xmin, xmax = cx - 0.5 * w, cx + 0.5 * w
            ymin, ymax = cy - 0.5 * w, cy + 0.5 * w

            xs = locs_xy[:, 0]
            ys = locs_xy[:, 1]
            mask = (xs >= xmin) & (xs <= xmax) & (ys >= ymin) & (ys <= ymax)
            crop_inds = np.where(mask)[0].astype(int).tolist()
            if len(crop_inds) < 8:
                crop_inds = sorted(contributing)

            if crop_inds:
                remap = {int(old): int(new) for new, old in enumerate(crop_inds)}
                locs_for_movie = locs_xy[crop_inds, :]
                template_for_movie = template_ch_by_t[crop_inds, :]
                branches_cropped: list[dict[str, Any]] = []
                for br in cleaned_branches:
                    br_ch = [int(c) for c in br.get("channels", []) if int(c) in remap]
                    if len(br_ch) >= 2:
                        branches_cropped.append({"channels": [remap[c] for c in br_ch]})
                gtr_for_movie = SimpleNamespace(branches=branches_cropped, locations=locs_for_movie) if branches_cropped else None
        except Exception:
            template_for_movie = template_ch_by_t
            locs_for_movie = locs_xy
            gtr_for_movie = SimpleNamespace(branches=cleaned_branches, locations=locs_xy)

    # Clip extreme amplitudes
    try:
        if 0.0 < float(clip_q) < 1.0:
            vals = np.asarray(template_for_movie)
            vmax = float(np.quantile(np.abs(vals), float(clip_q)))
            if vmax > 0.0:
                template_for_movie = np.clip(vals, -vmax, vmax)
    except Exception:
        pass

    # Render
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.animation import PillowWriter

    from axon_velocity.plotting import play_template_map as av_play_template_map  # type: ignore

    fig = plt.figure(figsize=(7.2, 6.2))
    ax = fig.add_subplot(111)

    skip_frames = 2
    ani = av_play_template_map(
        template_for_movie,
        locs_for_movie,
        gtr=gtr_for_movie,
        ax=ax,
        cmap=cmap,
        log=False,
        skip_frames=skip_frames,
        interval=40,
    )

    if add_cbar:
        try:
            images = getattr(ax, "images", None)
            if images:
                mappable = images[0]
                cbar = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.02)
                if cbar_label:
                    cbar.set_label(cbar_label)
        except Exception:
            pass

    if add_time:
        # Best-effort: use frame index only (we don't have fs here).
        try:
            framedata = getattr(ani, "_framedata", None)
        except Exception:
            framedata = None
        if framedata:
            for fi, artists in enumerate(framedata):
                label = f"frame {fi * skip_frames}"
                try:
                    txt = ax.text(
                        0.02,
                        0.98,
                        label,
                        transform=ax.transAxes,
                        ha="left",
                        va="top",
                        fontsize=10,
                        color="#222222",
                        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 2.0},
                    )
                    try:
                        artists.append(txt)
                    except Exception:
                        pass
                except Exception:
                    continue

    if (not crop) and zoom and branch_xy_points:
        try:
            xmin, xmax, ymin, ymax = _compute_zoom_limits_from_xy(branch_xy_points, pad_frac=pad_frac, pad_abs=pad_abs)
            cx = 0.5 * (xmin + xmax)
            cy = 0.5 * (ymin + ymax)
            w = float(max(xmax - xmin, ymax - ymin))
            ax.set_xlim(cx - 0.5 * w, cx + 0.5 * w)
            ax.set_ylim(cy - 0.5 * w, cy + 0.5 * w)
            ax.set_aspect("equal", adjustable="box")
        except Exception:
            pass

    out_gif.parent.mkdir(parents=True, exist_ok=True)
    ani.save(
        str(out_gif),
        writer=PillowWriter(fps=12),
        dpi=180,
        savefig_kwargs={"facecolor": "white"},
    )
    plt.close(fig)

    return out_gif.exists()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Regenerate template_movie.gif from branches_raw.json")
    p.add_argument("--env-file", type=Path, default=None, help="Path to debug.env")
    p.add_argument("--stream-id", type=str, required=True, help="Well/stream id (e.g., well001)")
    p.add_argument("--unit-ids", nargs="*", default=None, help="Optional unit IDs (space-separated)")
    p.add_argument("--force", action="store_true", help="Overwrite existing template_movie.gif")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.env_file is not None:
        env_files = [Path(args.env_file)]
    else:
        env_files = debug_env.default_env_paths(script_path=__file__)
    debug_env.load_env_files_into_os(env_files=env_files, override_existing=False)

    h5_path = debug_env.env_required_path("AXON_RECON_H5_PATH")
    stream_id = str(args.stream_id)
    mea_output_root = debug_env.env_required_path("AXON_RECON_MEA_OUTPUT_ROOT")

    from axon_reconstructor.pipeline.pipeline_driver import _compute_mea_analysis_output_dir

    well_out_dir = _compute_mea_analysis_output_dir(
        output_root=mea_output_root,
        data_file=h5_path,
        well=stream_id,
    )

    by_unit_dir = well_out_dir / "reconstruction_outputs" / "by_unit"
    if not by_unit_dir.exists():
        raise SystemExit(f"Missing reconstruction by_unit dir: {by_unit_dir}")

    if args.unit_ids:
        unit_ids = [int(x) for x in args.unit_ids]
    else:
        unit_ids = _iter_unit_ids_from_recon_by_unit(by_unit_dir)

    ok = 0
    skipped = 0
    failed = 0
    for i, uid in enumerate(unit_ids, start=1):
        try:
            wrote = _write_movie_gif_for_unit(uid=uid, well_out_dir=well_out_dir, force=bool(args.force))
            if wrote:
                ok += 1
            else:
                skipped += 1
        except Exception:
            failed += 1
        if (i % 10) == 0:
            print(f"[{stream_id}] {i}/{len(unit_ids)}: ok={ok} skipped={skipped} failed={failed}")

    print(f"Done {stream_id}: ok={ok} skipped={skipped} failed={failed}")
    print(f"Well dir: {well_out_dir}")


if __name__ == "__main__":
    main()
