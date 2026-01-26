from __future__ import annotations

import json
import logging
import sys
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


RECONSTRUCTION_OUTPUTS_DIRNAME = "reconstruction_outputs"


def _read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _jsonable(x: Any) -> Any:
    try:
        import numpy as np  # type: ignore[import-not-found]

        if isinstance(x, (np.integer, np.floating)):
            return x.item()
    except Exception:
        pass
    if isinstance(x, Path):
        return str(x)
    return x


def _as_list(x: Any) -> list[Any]:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    if isinstance(x, (str, bytes)):
        return [x]
    try:
        import numpy as np  # type: ignore[import-not-found]

        if isinstance(x, np.ndarray):
            return x.ravel().tolist()
    except Exception:
        pass
    try:
        return list(x)
    except Exception:
        return [x]


def _as_float_list(x: Any) -> list[float]:
    out: list[float] = []
    for v in _as_list(x):
        try:
            out.append(float(v))
        except Exception:
            continue
    return out


def _as_int_list(x: Any) -> list[int]:
    out: list[int] = []
    for v in _as_list(x):
        try:
            out.append(int(v))
        except Exception:
            continue
    return out


def _compute_zoom_limits_from_xy(xy_points: list[list[float]], *, pad_frac: float = 0.08, pad_abs: float = 20.0) -> tuple[float, float, float, float]:
    """Compute (xmin, xmax, ymin, ymax) with padding for a set of [x,y] points."""

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


def _compute_reconstruction_checkpoint_file(*, well_out_dir: Path, h5_path: Path, stream_id: str) -> Path:
    main_ckpt = compute_checkpoint_file(output_dir=well_out_dir, file_path=h5_path, stream_id=stream_id)
    name = main_ckpt.name
    if name.endswith("_checkpoint.json"):
        name = name[: -len("_checkpoint.json")] + "_reconstruction_checkpoint.json"
    else:
        name = main_ckpt.stem + "_reconstruction_checkpoint.json"
    return main_ckpt.with_name(name)


def _filter_kwargs_for_callable(fn: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Filter a dict to only kwargs accepted by `fn` (best effort)."""

    try:
        import inspect

        sig = inspect.signature(fn)
        allowed = set(sig.parameters.keys())
        return {k: v for k, v in kwargs.items() if k in allowed}
    except Exception:
        return dict(kwargs)


@dataclass(frozen=True)
class ReconstructionInputs:
    h5_path: Path
    stream_id: str
    mea_output_root: Path

    # Units to reconstruct. If None, reconstruct all units found under templates_outputs/merged_union_by_unit.
    unit_ids: Optional[list[Any]] = None
    unit_limit: Optional[int] = None

    # axon_velocity params (merged onto defaults). Only keys accepted by
    # axon_velocity.compute_graph_propagation_velocity are passed through.
    axon_velocity_params: Optional[dict[str, Any]] = None

    # Plotting / outputs
    write_unit_pdfs: bool = True
    write_all_units_overview_pdf: bool = True

    # Runtime
    verbose: bool = False

    # Resume controls
    force_restart: bool = False


@dataclass(frozen=True)
class ReconstructionOutputs:
    well_out_dir: Path
    reconstruction_out_dir: Path
    summary_json: Path
    by_unit_dir: Path
    all_units_overview_pdf: Optional[Path]


def reconstruct_from_templates(*, inputs: ReconstructionInputs, logger_name_prefix: str = "axon_reconstructor") -> ReconstructionOutputs:
    """Run axon reconstruction/velocity estimation using axon_velocity.

    This stage consumes the merged_union artifacts from templates extraction:
      <well>/templates_outputs/merged_union_by_unit/unit_<id>/merged_union_template.npy
      <well>/templates_outputs/merged_union_by_unit/unit_<id>/merged_union_channel_locations.npy
      <well>/templates_outputs/merged_union_by_unit/unit_<id>/merged_union_template_meta.json

    And produces:
      <well>/reconstruction_outputs/
        reconstruction_summary.json
        all_units_morphology.pdf (optional)
        by_unit/unit_<id>/ ... per-unit json + pdfs

    Focused subset (requested):
    - sparse morphology per unit (branches as polylines)
    - heuristics per unit (selected channels + node heuristics)
    - velocity plots per unit/branch
    - all-units morphology overview
    """

    well_out_dir = _compute_mea_analysis_output_dir(
        output_root=inputs.mea_output_root,
        data_file=inputs.h5_path,
        well=inputs.stream_id,
    )

    log_file = compute_pipeline_log_file(well_out_dir=well_out_dir, data_file=inputs.h5_path, stream_id=inputs.stream_id)
    logger = setup_pipeline_logger(
        log_file=log_file,
        logger_name=f"{logger_name_prefix}.{inputs.stream_id}.reconstruction",
        verbose=True,
    )

    recon_out_dir = well_out_dir / RECONSTRUCTION_OUTPUTS_DIRNAME
    by_unit_dir = recon_out_dir / "by_unit"
    summary_json = recon_out_dir / "reconstruction_summary.json"
    all_units_overview_pdf = recon_out_dir / "all_units_morphology.pdf" if inputs.write_all_units_overview_pdf else None

    ckpt_file = _compute_reconstruction_checkpoint_file(well_out_dir=well_out_dir, h5_path=inputs.h5_path, stream_id=inputs.stream_id)
    ckpt = load_checkpoint(
        checkpoint_file=ckpt_file,
        force_restart=bool(inputs.force_restart),
        output_dir=well_out_dir,
        file_path=inputs.h5_path,
        stream_id=inputs.stream_id,
    )

    # Resume shortcut.
    if (
        (not inputs.force_restart)
        and summary_json.exists()
        and ((all_units_overview_pdf is None) or all_units_overview_pdf.exists())
        and by_unit_dir.exists()
    ):
        logger.info("Resuming reconstruction: existing outputs found at %s", recon_out_dir)
        return ReconstructionOutputs(
            well_out_dir=well_out_dir,
            reconstruction_out_dir=recon_out_dir,
            summary_json=summary_json,
            by_unit_dir=by_unit_dir,
            all_units_overview_pdf=(all_units_overview_pdf if (all_units_overview_pdf and all_units_overview_pdf.exists()) else None),
        )

    ckpt = save_checkpoint(
        checkpoint_file=ckpt_file,
        state=ckpt,
        stage=ProcessingStage.ANALYZER,
        failed_stage=None,
        error=None,
        extra_fields={"reconstruction_out_dir": str(recon_out_dir)},
    )

    recon_out_dir.mkdir(parents=True, exist_ok=True)
    by_unit_dir.mkdir(parents=True, exist_ok=True)

    templates_out_dir = well_out_dir / "templates_outputs"
    merged_union_by_unit_dir = templates_out_dir / "merged_union_by_unit"
    if not merged_union_by_unit_dir.exists():
        raise FileNotFoundError(f"Missing merged_union templates at {merged_union_by_unit_dir}")

    try:
        import numpy as np  # type: ignore[import-not-found]
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Reconstruction requires numpy") from e

    try:
        import axon_velocity as av  # type: ignore[import-not-found]
    except Exception as e:  # pragma: no cover
        # Best-effort fallback: user often has axon_velocity as an editable checkout,
        # but the active interpreter may not have it installed.
        axon_velocity_repo = Path("/home/adamm/dev/pkgs/axon_velocity")
        if (axon_velocity_repo / "axon_velocity").exists():
            sys.path.insert(0, str(axon_velocity_repo))
            try:
                import axon_velocity as av  # type: ignore[import-not-found]
            except Exception as e2:
                msg = str(e2)
                if "No module named 'sklearn'" in msg or "No module named sklearn" in msg:
                    raise RuntimeError(
                        "axon_velocity imported, but its dependency scikit-learn is missing. Install it in the active env (e.g. `pip install scikit-learn` or `conda install scikit-learn`)."
                    ) from e2
                raise RuntimeError(
                    "Reconstruction requires axon_velocity; attempted sys.path fallback to /home/adamm/dev/pkgs/axon_velocity"
                ) from e2
        else:
            raise RuntimeError(
                "Reconstruction requires axon_velocity (expected editable install at /home/adamm/dev/pkgs/axon_velocity)"
            ) from e

    # Determine unit list.
    discovered_unit_ids: list[Any] = []
    for p in sorted(merged_union_by_unit_dir.glob("unit_*")):
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

    params = av.get_default_graph_velocity_params()
    if inputs.axon_velocity_params:
        params.update(dict(inputs.axon_velocity_params))
    params = _filter_kwargs_for_callable(av.compute_graph_propagation_velocity, params)
    params.setdefault("verbose", bool(inputs.verbose))

    summary: dict[str, Any] = {
        "h5_path": str(inputs.h5_path),
        "stream_id": inputs.stream_id,
        "well_out_dir": str(well_out_dir),
        "templates_merged_union_by_unit_dir": str(merged_union_by_unit_dir),
        "reconstruction_out_dir": str(recon_out_dir),
        "axon_velocity_params": {k: _jsonable(v) for k, v in params.items()},
        "units": [],
    }

    # For all-units overview.
    all_unit_polylines: list[dict[str, Any]] = []
    all_locations: list[Any] = []

    for uid in unit_ids:
        unit_dir = merged_union_by_unit_dir / f"unit_{uid}"
        tmpl_npy = unit_dir / "merged_union_template.npy"
        locs_npy = unit_dir / "merged_union_channel_locations.npy"
        meta_json = unit_dir / "merged_union_template_meta.json"

        out_unit_dir = by_unit_dir / f"unit_{uid}"
        out_unit_dir.mkdir(parents=True, exist_ok=True)

        unit_summary: dict[str, Any] = {
            "unit_id": _jsonable(uid),
            "inputs": {
                "template_npy": str(tmpl_npy),
                "channel_locations_npy": str(locs_npy),
                "template_meta_json": str(meta_json),
            },
            "outputs": {},
            "status": "ok",
            "error": None,
        }

        try:
            if not tmpl_npy.exists() or not locs_npy.exists():
                raise FileNotFoundError(f"Missing merged_union inputs for unit {uid}")

            tmpl = np.load(tmpl_npy)
            locs = np.load(locs_npy)
            if tmpl.ndim != 2:
                raise ValueError(f"Unexpected template shape for unit {uid}: {tmpl.shape}")
            if locs.ndim != 2 or locs.shape[1] < 2:
                raise ValueError(f"Unexpected locations shape for unit {uid}: {locs.shape}")

            # Our saved templates are (n_samples, n_channels); axon_velocity expects (n_channels, n_timepoints).
            tmpl_ch_by_t = np.asarray(tmpl).T
            locs_xy = np.asarray(locs)[:, :2]

            fs_hz = None
            try:
                meta = _read_json(meta_json)
                fs_hz = float(meta.get("sampling_frequency_hz")) if meta.get("sampling_frequency_hz") is not None else None
            except Exception:
                fs_hz = None
            if fs_hz is None:
                # Fallback: try templates_summary.json (best effort)
                fs_hz = 10_000.0

            gtr = av.compute_graph_propagation_velocity(tmpl_ch_by_t, locs_xy, float(fs_hz), **params)

            # --- Save sparse morphology + velocity per branch ---
            branches_out: list[dict[str, Any]] = []
            branches_val = getattr(gtr, "branches", None)
            branches_iter = _as_list(branches_val)
            for bi, br in enumerate(branches_iter):
                try:
                    chans = _as_int_list(br.get("channels"))
                except Exception:
                    chans = []
                xy = []
                try:
                    for ch in chans:
                        xy.append([float(locs_xy[ch, 0]), float(locs_xy[ch, 1])])
                except Exception:
                    xy = []

                br_out = {
                    "branch_index": int(bi),
                    "channels": chans,
                    "polyline_xy": xy,
                    "velocity": _jsonable(br.get("velocity")),
                    "offset": _jsonable(br.get("offset")),
                    "r2": _jsonable(br.get("r2")),
                    "pval": _jsonable(br.get("pval")),
                    "distances": _as_float_list(br.get("distances")),
                    "peak_times": _as_float_list(br.get("peak_times")),
                }
                branches_out.append(br_out)

                if xy:
                    all_unit_polylines.append({"unit_id": _jsonable(uid), "branch_index": int(bi), "polyline_xy": xy})

            all_locations.append(locs_xy)

            heuristics: dict[str, Any] = {
                "init_channel": int(getattr(gtr, "init_channel", 0)),
                "n_channels_total": int(locs_xy.shape[0]),
                "n_selected_channels": int(len(_as_list(getattr(gtr, "selected_channels", None)))),
                "selected_channels": _as_int_list(getattr(gtr, "selected_channels", None)),
                "node_heuristic": None,
            }
            try:
                node_h = getattr(gtr, "_node_heuristic", None)
                if node_h is not None:
                    heuristics["node_heuristic"] = [float(x) for x in list(node_h)]
            except Exception:
                pass

            branches_json = out_unit_dir / "branches.json"
            heuristics_json = out_unit_dir / "heuristics.json"
            _write_json(branches_json, {"unit_id": _jsonable(uid), "branches": branches_out})
            _write_json(heuristics_json, {"unit_id": _jsonable(uid), "heuristics": heuristics})

            unit_summary["outputs"].update(
                {
                    "branches_json": str(branches_json),
                    "heuristics_json": str(heuristics_json),
                }
            )

            # --- Plots ---
            if inputs.write_unit_pdfs:
                try:
                    import numpy as np  # type: ignore[import-not-found]
                    import matplotlib

                    matplotlib.use("Agg", force=True)
                    import matplotlib.pyplot as plt
                    import matplotlib.backends.backend_pdf as pdf

                    # Always write a simple morphology PDF (robust against axon_velocity plotting changes).
                    morphology_pdf = out_unit_dir / "morphology.pdf"
                    if (not morphology_pdf.exists()) or inputs.force_restart:
                        try:
                            fig = plt.figure(figsize=(8, 6))
                            ax = fig.add_subplot(111)
                            ax.plot(locs_xy[:, 0], locs_xy[:, 1], marker=".", ls="", color="0.8", alpha=0.6, ms=3)

                            branches_for_plot = _as_list(getattr(gtr, "branches", None))
                            cm = plt.get_cmap("tab10")
                            branch_xy_points: list[list[float]] = []
                            for bi, br in enumerate(branches_for_plot):
                                chans = _as_int_list(br.get("channels"))
                                if not chans:
                                    continue
                                xy = np.asarray([[locs_xy[ch, 0], locs_xy[ch, 1]] for ch in chans if 0 <= ch < locs_xy.shape[0]])
                                if xy.size == 0:
                                    continue
                                color = cm(bi % 10)
                                ax.plot(xy[:, 0], xy[:, 1], color=color, lw=2, alpha=0.9)
                                ax.plot(xy[:, 0], xy[:, 1], marker="o", ls="", color=color, ms=3, alpha=0.9)
                                for row in xy.tolist():
                                    branch_xy_points.append([float(row[0]), float(row[1])])

                            ax.set_title(f"unit {uid} morphology")
                            ax.set_aspect("equal", adjustable="box")
                            ax.set_xlabel("x")
                            ax.set_ylabel("y")
                            with pdf.PdfPages(morphology_pdf) as out:
                                out.savefig(fig, dpi=150)
                            plt.close(fig)
                        except Exception as e:
                            logger.warning("Morphology plotting failed for unit %s: %s", uid, e)

                    if morphology_pdf.exists():
                        unit_summary["outputs"].update({"morphology_pdf": str(morphology_pdf)})

                    # Zoomed-in morphology around the reconstruction.
                    morphology_zoom_pdf = out_unit_dir / "morphology_zoom.pdf"
                    if (not morphology_zoom_pdf.exists()) or inputs.force_restart:
                        try:
                            fig = plt.figure(figsize=(8, 6))
                            ax = fig.add_subplot(111)

                            branches_for_plot = _as_list(getattr(gtr, "branches", None))
                            cm = plt.get_cmap("tab10")
                            branch_xy_points: list[list[float]] = []
                            for bi, br in enumerate(branches_for_plot):
                                chans = _as_int_list(br.get("channels"))
                                if not chans:
                                    continue
                                xy = np.asarray([[locs_xy[ch, 0], locs_xy[ch, 1]] for ch in chans if 0 <= ch < locs_xy.shape[0]])
                                if xy.size == 0:
                                    continue
                                color = cm(bi % 10)
                                ax.plot(xy[:, 0], xy[:, 1], color=color, lw=2, alpha=0.95)
                                ax.plot(xy[:, 0], xy[:, 1], marker="o", ls="", color=color, ms=3, alpha=0.95)
                                for row in xy.tolist():
                                    branch_xy_points.append([float(row[0]), float(row[1])])

                            # Background electrodes: only those inside the zoom window (computed from branches).
                            xmin, xmax, ymin, ymax = _compute_zoom_limits_from_xy(branch_xy_points)
                            in_view = (locs_xy[:, 0] >= xmin) & (locs_xy[:, 0] <= xmax) & (locs_xy[:, 1] >= ymin) & (locs_xy[:, 1] <= ymax)
                            if np.any(in_view):
                                ax.plot(locs_xy[in_view, 0], locs_xy[in_view, 1], marker=".", ls="", color="0.85", alpha=0.6, ms=3)

                            ax.set_xlim(xmin, xmax)
                            ax.set_ylim(ymin, ymax)
                            ax.set_title(f"unit {uid} morphology (zoom)")
                            ax.set_aspect("equal", adjustable="box")
                            ax.set_xlabel("x")
                            ax.set_ylabel("y")
                            with pdf.PdfPages(morphology_zoom_pdf) as out:
                                out.savefig(fig, dpi=150)
                            plt.close(fig)
                        except Exception as e:
                            logger.warning("Zoom morphology plotting failed for unit %s: %s", uid, e)

                    if morphology_zoom_pdf.exists():
                        unit_summary["outputs"].update({"morphology_zoom_pdf": str(morphology_zoom_pdf)})

                    # Heuristics / channel selection plot (axon_velocity built-in).
                    heuristics_pdf = out_unit_dir / "heuristics.pdf"
                    if (not heuristics_pdf.exists()) or inputs.force_restart:
                        try:
                            plot_fn = getattr(gtr, "plot_channel_selection", None)
                            if callable(plot_fn):
                                fig = plot_fn()
                                with pdf.PdfPages(heuristics_pdf) as out:
                                    out.savefig(fig, dpi=150)
                                plt.close(fig)
                        except Exception as e:
                            logger.warning("Heuristics plotting failed for unit %s: %s", uid, e)

                    if heuristics_pdf.exists():
                        unit_summary["outputs"].update({"heuristics_pdf": str(heuristics_pdf)})

                    # per-branch velocity plots
                    per_branch_dir = out_unit_dir / "branches"
                    per_branch_dir.mkdir(parents=True, exist_ok=True)

                    # All-branches overlay velocity plot
                    overlay_pdf = per_branch_dir / "branch_velocities_overlay.pdf"
                    if (not overlay_pdf.exists()) or inputs.force_restart:
                        try:
                            branches_for_plot = _as_list(getattr(gtr, "branches", None))
                            if branches_for_plot:
                                fig = plt.figure(figsize=(7, 5))
                                ax = fig.add_subplot(111)
                                cm = plt.get_cmap("tab10")
                                any_plotted = False
                                for bi, br in enumerate(branches_for_plot):
                                    peak_times = _as_float_list(br.get("peak_times"))
                                    distances = _as_float_list(br.get("distances"))
                                    if (len(peak_times) < 2) or (len(distances) != len(peak_times)):
                                        continue
                                    color = cm(bi % 10)
                                    ax.scatter(peak_times, distances, s=12, alpha=0.7, color=color, label=f"b{bi}")
                                    any_plotted = True
                                    try:
                                        velocity = br.get("velocity")
                                        offset = br.get("offset")
                                        if (velocity is not None) and (offset is not None):
                                            v = float(velocity)
                                            b = float(offset)
                                            xs = np.linspace(min(peak_times), max(peak_times), 50)
                                            ys = v * xs + b
                                            ax.plot(xs, ys, lw=2, alpha=0.8, color=color)
                                    except Exception:
                                        pass

                                if any_plotted:
                                    ax.set_title(f"unit {uid} branch velocities (overlay)")
                                    ax.set_xlabel("peak_time")
                                    ax.set_ylabel("distance")
                                    ax.legend(loc="best", fontsize=8, frameon=False, ncol=2)
                                    with pdf.PdfPages(overlay_pdf) as out:
                                        out.savefig(fig, dpi=150)
                                plt.close(fig)
                        except Exception as e:
                            logger.warning("Overlay velocity plotting failed for unit %s: %s", uid, e)

                    for bi, br in enumerate(_as_list(getattr(gtr, "branches", None))):
                        br_pdf = per_branch_dir / f"branch_{bi:02d}_velocity.pdf"
                        if br_pdf.exists() and (not inputs.force_restart):
                            continue
                        try:
                            peak_times = _as_float_list(br.get("peak_times"))
                            distances = _as_float_list(br.get("distances"))
                            velocity = br.get("velocity")
                            offset = br.get("offset")
                            r2 = br.get("r2")

                            if (len(peak_times) < 2) or (len(distances) != len(peak_times)):
                                continue

                            fig = plt.figure(figsize=(6, 4))
                            ax = fig.add_subplot(111)
                            ax.scatter(peak_times, distances, s=12, alpha=0.8)
                            ax.set_xlabel("peak_time")
                            ax.set_ylabel("distance")

                            try:
                                if (velocity is not None) and (offset is not None):
                                    v = float(velocity)
                                    b = float(offset)
                                    xs = np.linspace(min(peak_times), max(peak_times), 50)
                                    ys = v * xs + b
                                    ax.plot(xs, ys, lw=2, alpha=0.8)
                            except Exception:
                                pass

                            title_bits = [f"unit {uid}", f"branch {bi}"]
                            try:
                                if velocity is not None:
                                    title_bits.append(f"v={float(velocity):.3g}")
                            except Exception:
                                pass
                            try:
                                if r2 is not None:
                                    title_bits.append(f"r2={float(r2):.3g}")
                            except Exception:
                                pass
                            ax.set_title("  ".join(title_bits))

                            with pdf.PdfPages(br_pdf) as out:
                                out.savefig(fig, dpi=150)
                            plt.close(fig)
                        except Exception:
                            continue
                except Exception as e:
                    logger.warning("Plotting failed for unit %s: %s", uid, e)

        except Exception as e:
            unit_summary["status"] = "error"
            unit_summary["error"] = exception_to_error_dict(e)
            logger.warning("Reconstruction failed for unit %s: %s", uid, e)

        summary["units"].append(unit_summary)

    # All-units overview morphology plot.
    if all_units_overview_pdf is not None:
        try:
            import numpy as np  # type: ignore[import-not-found]
            import matplotlib

            matplotlib.use("Agg", force=True)
            import matplotlib.pyplot as plt
            import matplotlib.backends.backend_pdf as pdf

            if (not all_units_overview_pdf.exists()) or inputs.force_restart:
                fig = plt.figure(figsize=(11, 8.5))
                ax = fig.add_subplot(111)

                # Background electrode cloud (union of all template channel locations).
                if all_locations:
                    locs_all = np.concatenate([np.asarray(x)[:, :2] for x in all_locations if np.asarray(x).size], axis=0)
                    ax.plot(locs_all[:, 0], locs_all[:, 1], marker=".", ls="", color="0.8", alpha=0.15, ms=2)

                # Overlay unit/branch polylines.
                cm = plt.get_cmap("tab20")
                for i, poly in enumerate(all_unit_polylines):
                    xy = poly.get("polyline_xy")
                    if not xy:
                        continue
                    xs = [p[0] for p in xy]
                    ys = [p[1] for p in xy]
                    ax.plot(xs, ys, lw=1.2, alpha=0.9, color=cm(i % 20))

                ax.set_aspect("equal", adjustable="box")
                ax.set_title(f"All units morphology (stream={inputs.stream_id})")
                ax.set_xlabel("x (um)")
                ax.set_ylabel("y (um)")

                with pdf.PdfPages(all_units_overview_pdf) as out:
                    out.savefig(fig, dpi=150)
                plt.close(fig)

            summary["all_units_overview_pdf"] = str(all_units_overview_pdf)
        except Exception as e:
            logger.warning("Failed writing all-units overview pdf: %s", e)

    _write_json(summary_json, summary)

    ckpt = save_checkpoint(
        checkpoint_file=ckpt_file,
        state=ckpt,
        stage=ProcessingStage.ANALYZER_COMPLETE,
        failed_stage=None,
        error=None,
        extra_fields={
            "reconstruction_out_dir": str(recon_out_dir),
            "reconstruction_summary_json": str(summary_json),
            "all_units_overview_pdf": str(all_units_overview_pdf) if all_units_overview_pdf else None,
        },
    )

    return ReconstructionOutputs(
        well_out_dir=well_out_dir,
        reconstruction_out_dir=recon_out_dir,
        summary_json=summary_json,
        by_unit_dir=by_unit_dir,
        all_units_overview_pdf=all_units_overview_pdf,
    )


__all__ = [
    "ReconstructionInputs",
    "ReconstructionOutputs",
    "reconstruct_from_templates",
]
