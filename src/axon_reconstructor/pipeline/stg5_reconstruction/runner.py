from __future__ import annotations

import concurrent.futures
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ..checkpointing import (
    ProcessingStage,
    exception_to_error_dict,
    load_checkpoint,
)
from ..output_paths import compute_mea_analysis_output_dir
from ..pipeline_logging import build_stage_logger, log_stage_complete, log_stage_failure, log_stage_start
from ..shared_io import as_float_list, as_int_list, as_list, jsonable, read_json, write_json
from ..checkpointing import compute_stage_checkpoint_file, save_stage_completed, save_stage_failed, save_stage_started

from .plotting import (
    write_all_units_overview_pdf,
    write_top_density_raw_branch_footprint_grid,
    write_unit_reconstruction_pdfs,
)

RECONSTRUCTION_OUTPUTS_DIRNAME = "stg5_reconstruction_outputs"


def _read_json(path: Path) -> Any:
    return read_json(path)


def _write_json(path: Path, payload: Any) -> None:
    write_json(path, payload)


def _jsonable(x: Any) -> Any:
    return jsonable(x)


def _as_list(x: Any) -> list[Any]:
    return as_list(x)


def _as_float_list(x: Any) -> list[float]:
    return as_float_list(x)


def _as_int_list(x: Any) -> list[int]:
    return as_int_list(x)


def _compute_raw_branches_fallback(*, uid: Any, gtr: Any) -> list[dict[str, Any]]:
    branches_out: list[dict[str, Any]] = []
    branches_val = getattr(gtr, "branches", None)
    branches_iter = _as_list(branches_val)
    for bi, br in enumerate(branches_iter):
        br_dict = br if isinstance(br, dict) else {}
        branches_out.append(
            {
                "unit_id": _jsonable(uid),
                "branch_index": int(br_dict.get("branch_index", bi)),
                "channels": _as_int_list(br_dict.get("channels")),
                "velocity": _jsonable(br_dict.get("velocity")),
                "offset": _jsonable(br_dict.get("offset")),
                "r2": _jsonable(br_dict.get("r2")),
                "pval": _jsonable(br_dict.get("pval")),
                "distances": _as_float_list(br_dict.get("distances")),
                "peak_times": _as_float_list(br_dict.get("peak_times")),
            }
        )
    return branches_out


def _write_branches_raw_json(*, uid: Any, gtr: Any, raw_branches_json: Path, logger: logging.Logger) -> tuple[str | None, str | None]:
    try:
        from .plotting import compute_raw_branches_for_summary

        raw_branches_out = compute_raw_branches_for_summary(uid=uid, gtr=gtr)
        _write_json(raw_branches_json, {"unit_id": _jsonable(uid), "branches": raw_branches_out})
        return str(raw_branches_json), None
    except Exception as e:
        try:
            fallback = _compute_raw_branches_fallback(uid=uid, gtr=gtr)
            _write_json(raw_branches_json, {"unit_id": _jsonable(uid), "branches": fallback})
            return str(raw_branches_json), f"raw-branches helper failed, used fallback: {e}"
        except Exception as e2:
            return None, f"failed writing branches_raw.json: {e2}"


def _compute_reconstruction_checkpoint_file(*, well_out_dir: Path, h5_path: Path, stream_id: str) -> Path:
    return compute_stage_checkpoint_file(
        well_out_dir=well_out_dir,
        h5_path=h5_path,
        stream_id=stream_id,
        stage_name="reconstruction",
    )


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

    # Optional variant routing.
    # - If `templates_variant_name` is set (e.g. "merged"), reconstruction reads
    #   templates from `<well>/stg4_templates_outputs_<variant>`.
    # - If `reconstruction_variant_name` is set, reconstruction writes to
    #   `<well>/stg5_reconstruction_outputs_<variant>` and uses a variant checkpoint.
    # - If `reconstruction_variant_name` is omitted, it defaults to
    #   `templates_variant_name` when provided.
    templates_variant_name: Optional[str] = None
    reconstruction_variant_name: Optional[str] = None

    # Units to reconstruct. If None, reconstruct all units found under stg4_templates_outputs/templates/merged.
    unit_ids: Optional[list[Any]] = None
    unit_limit: Optional[int] = None

    # Template source for axon_velocity.
    #
    # axon_velocity is most robust when given a dense full-channel template on a deterministic
    # geometry (e.g. Maxwell full chip). By default we therefore consume templates-stage outputs
    # under `<well>/stg4_templates_outputs/templates/full/`.
    #
    # Set `use_full_channels_templates=False` to fall back to the sparse merged-contributing
    # template under `<well>/stg4_templates_outputs/templates/merged/`.
    use_full_channels_templates: bool = True

    # If True, raise if full-channel templates are missing.
    require_full_channels_templates: bool = True

    # Number of units to include in the footprint-density ranking/grid artifact.
    # If None (or non-positive), include all available candidates.
    top_n_density_requested: Optional[int] = None

    # If True, write the raw-branch + log-footprint density ranking grid artifact.
    write_top_density_grid: bool = True

    # Grid output controls.
    grid_output_subdir: str = "grids"
    grid_write_pdf: bool = True
    grid_write_png: bool = True
    grid_write_ranking_json: bool = True
    grid_log_basename: str = "raw_branch_log_footprint_top_density_grid"
    grid_linear_basename: str = "raw_branch_linear_footprint_top_density_grid"
    grid_ranking_filename: str = "raw_branch_log_footprint_top_density_ranking.json"
    grid_ncols: int = 5
    grid_dpi: int = 220
    grid_draw_zoom_range_box: bool = False
    grid_panel_background_color: str = "black"
    grid_branch_color: str = "red"
    grid_branch_outline_color: str = "white"
    grid_node_radius_um: float = 5.0
    grid_soma_node_radius_um: float = 10.0
    grid_soma_node_color: str = "yellow"
    grid_sort_by: str = "density"
    grid_zoom_priority: str = "branches"
    grid_zoom_padding_percent: float = 20.0
    grid_force_soma_centering: bool = False
    grid_soma_xy_show: bool = False
    grid_soma_xy_color: str = "white"
    grid_soma_xy_fontsize: float = 5.0
    grid_soma_xy_location: str = "bottom left"
    grid_show_unit_id_in_plot: bool = True
    grid_unit_id_fontsize: float = 6.0
    grid_unit_id_color: str = "white"
    grid_show_minimap: bool = True
    grid_minimap_position: str = "bottomright"
    grid_minimap_size: float = 0.20
    grid_minimap_outline_color: str = "white"
    grid_minimap_chip_width_mm: float = 3.85
    grid_minimap_chip_height_mm: float = 2.10
    grid_minimap_inner_box_linestyle: str = "dotted"
    grid_minimap_inner_box_linewidth: float = 0.8
    grid_minimap_include_footprint: bool = False
    grid_minimap_prevent_occlusions: bool = False
    grid_legend_show: bool = False
    grid_legend_location: str = "first_empty_panel"
    grid_legend_fontsize: float = 6.0
    grid_legend_fontcolor: str = "white"
    grid_legend_marker_size: float = 3.0
    grid_legend_show_nodes_in_legend: bool = True
    grid_legend_show_footprint_in_legend: bool = True
    grid_emit_debug_logs: bool = False

    # If True, annotate raw detected amplitude min/max in the last subplot of the
    # top-density grids (debugging aid).
    show_density_scale_debug_text: bool = False

    # If True, annotate global amplitude statistics in a dedicated debug panel.
    show_density_scale_global_debug_text: bool = False

    # If True, annotate per-unit local amplitude statistics in each unit subplot.
    show_density_scale_local_debug_text: bool = False

    # If True, only replot/update the top-density grid from existing reconstruction
    # outputs/summary and return without running per-unit reconstruction.
    replot_top_density_grid_only: bool = False

    # axon_velocity params (merged onto defaults). Only keys accepted by
    # axon_velocity.compute_graph_propagation_velocity are passed through.
    axon_velocity_params: Optional[dict[str, Any]] = None

    # Optional checkout root for axon_velocity when it is not installed in this
    # environment (editable/dev workflow).
    axon_velocity_repo_root: Optional[Path] = None

    # Plotting / outputs
    write_unit_pdfs: bool = True
    write_all_units_overview_pdf: bool = True
    # If False, skip writing per-unit template_movie.gif files.
    write_template_movie_gif: bool = False

    # If True, skip axon_velocity tracking and only (re)render per-unit summary plots
    # from already-written reconstruction/templates artifacts on disk.
    replot_summaries_only: bool = False

    # If True, run axon_velocity tracking but only (re)write branches_raw.json (raw paths)
    # and skip all other per-unit plots/outputs.
    recompute_branches_raw_only: bool = False

    # Explicit controls for per-unit reconstruction artifacts.
    per_unit_write_branches_raw_json: bool = True
    per_unit_write_branches_json: bool = True
    per_unit_write_heuristics_json: bool = True

    # Runtime
    verbose: bool = False
    unit_workers: int = 1

    # Resume controls
    force_restart: bool = False


@dataclass(frozen=True)
class ReconstructionOutputs:
    well_out_dir: Path
    reconstruction_out_dir: Path
    summary_json: Path
    by_unit_dir: Path
    all_units_overview_pdf: Optional[Path]


def _import_axon_velocity(*, repo_root: Optional[Path] = None) -> Any:
    try:
        import axon_velocity as av  # type: ignore[import-not-found]

        return av
    except Exception as e:
        fallback = Path(repo_root) if repo_root is not None else Path("/home/adamm/dev/pkgs/axon_velocity")
        if (fallback / "axon_velocity").exists():
            if str(fallback) not in sys.path:
                sys.path.insert(0, str(fallback))
            try:
                import axon_velocity as av  # type: ignore[import-not-found]

                return av
            except Exception as e2:
                msg = str(e2)
                if "No module named 'sklearn'" in msg or "No module named sklearn" in msg:
                    raise RuntimeError(
                        "axon_velocity imported, but its dependency scikit-learn is missing. Install it in the active env (e.g. `pip install scikit-learn` or `conda install scikit-learn`)."
                    ) from e2
                raise RuntimeError(
                    f"Reconstruction requires axon_velocity; attempted sys.path fallback to {fallback}"
                ) from e2

        raise RuntimeError(
            f"Reconstruction requires axon_velocity (expected editable install at {fallback})"
        ) from e


def _run_single_unit_reconstruction(
    *,
    uid: Any,
    reconstruction_out_dir: Path,
    merged_units_dir: Path,
    full_channels_templates_dir: Path,
    use_full_channels_templates: bool,
    require_full_channels_templates: bool,
    axon_velocity_repo_root: Optional[Path],
    params: dict[str, Any],
    force_restart: bool,
    write_unit_pdfs: bool,
    write_template_movie_gif: bool,
    replot_summaries_only: bool,
    recompute_branches_raw_only: bool,
    per_unit_write_branches_raw_json: bool,
    per_unit_write_branches_json: bool,
    per_unit_write_heuristics_json: bool,
) -> dict[str, Any]:
    logger = logging.getLogger("axon_reconstructor.reconstruction.unit")

    unit_dir = merged_units_dir / f"unit_{uid}"
    merged_tmpl_npy = unit_dir / "merged_contributing_template.npy"
    merged_locs_npy = unit_dir / "merged_contributing_channel_locations.npy"
    merged_meta_json = unit_dir / "merged_contributing_template_meta.json"

    full_unit_dir = full_channels_templates_dir / f"unit_{uid}"
    full_tmpl_npy = full_unit_dir / "full_template.npy"
    full_locs_npy = full_unit_dir / "full_channel_locations_xy.npy"
    full_meta_json = full_unit_dir / "full_template_meta.json"

    out_unit_dir = Path(reconstruction_out_dir) / "by_unit" / f"unit_{uid}"
    out_unit_dir.mkdir(parents=True, exist_ok=True)
    unit_checkpoint_json = out_unit_dir / "unit_reconstruction_checkpoint.json"
    unit_summary_json = out_unit_dir / "unit_reconstruction_summary.json"

    def _load_existing_completed_unit_result() -> Optional[dict[str, Any]]:
        try:
            ckpt_payload = _read_json(unit_checkpoint_json)
            if not isinstance(ckpt_payload, dict):
                return None
            if str(ckpt_payload.get("status", "")).lower() != "completed":
                return None

            existing_summary = _read_json(unit_summary_json)
            if not isinstance(existing_summary, dict):
                return None
            if str(existing_summary.get("status", "")).lower() != "ok":
                return None

            existing_polylines: list[dict[str, Any]] = []
            branches_json_path = (existing_summary.get("outputs") or {}).get("branches_json")
            if branches_json_path:
                try:
                    branches_payload = _read_json(Path(str(branches_json_path)))
                    branches = list((branches_payload or {}).get("branches", []) or [])
                    for br in branches:
                        if not isinstance(br, dict):
                            continue
                        poly = br.get("polyline_xy")
                        if isinstance(poly, list) and len(poly) >= 2:
                            existing_polylines.append(
                                {
                                    "unit_id": _jsonable(uid),
                                    "branch_index": int(br.get("branch_index", 0)),
                                    "polyline_xy": poly,
                                }
                            )
                except Exception:
                    pass

            existing_locs_path: Optional[str] = None
            try:
                selected_src = str((existing_summary.get("inputs") or {}).get("selected_template_source", ""))
                merged_locs_path = str((existing_summary.get("inputs") or {}).get("merged_contributing", {}).get("channel_locations_npy", ""))
                full_locs_path = str((existing_summary.get("inputs") or {}).get("full_channels", {}).get("full_channel_locations_xy_npy", ""))
                if selected_src == "full_channels_templates" and full_locs_path:
                    existing_locs_path = full_locs_path
                elif merged_locs_path:
                    existing_locs_path = merged_locs_path
                elif full_locs_path:
                    existing_locs_path = full_locs_path
            except Exception:
                existing_locs_path = None

            existing_summary["resume_skipped_completed"] = True
            return {
                "unit_summary": existing_summary,
                "unit_polylines": existing_polylines,
                "overview_locations_npy": existing_locs_path,
            }
        except Exception:
            return None

    if not bool(force_restart):
        existing_result = _load_existing_completed_unit_result()
        if existing_result is not None:
            logger.info("Unit %s already completed; skipping via unit checkpoint", uid)
            return existing_result

    unit_summary: dict[str, Any] = {
        "unit_id": _jsonable(uid),
        "inputs": {
            "merged_contributing": {
                "template_npy": str(merged_tmpl_npy),
                "channel_locations_npy": str(merged_locs_npy),
                "template_meta_json": str(merged_meta_json),
            },
            "full_channels": {
                "full_template_npy": str(full_tmpl_npy),
                "full_channel_locations_xy_npy": str(full_locs_npy),
                "full_template_meta_json": str(full_meta_json),
            },
        },
        "outputs": {},
        "status": "ok",
        "error": None,
    }

    unit_polylines: list[dict[str, Any]] = []
    overview_locations_npy: Optional[str] = None

    try:
        try:
            import numpy as np  # type: ignore[import-not-found]
        except Exception as e:
            raise RuntimeError("Reconstruction requires numpy") from e

        av = _import_axon_velocity(repo_root=axon_velocity_repo_root)

        use_full = bool(use_full_channels_templates)
        if use_full:
            if not full_tmpl_npy.exists() or not full_locs_npy.exists():
                if bool(require_full_channels_templates):
                    raise FileNotFoundError(
                        f"Missing full-channel templates for unit {uid}: expected {full_tmpl_npy} and {full_locs_npy}"
                    )
                use_full = False

        if use_full:
            tmpl = np.load(full_tmpl_npy)
            locs = np.load(full_locs_npy)
            unit_summary["inputs"]["selected_template_source"] = "full_channels_templates"
            overview_locations_npy = str(full_locs_npy)
        else:
            if not merged_tmpl_npy.exists() or not merged_locs_npy.exists():
                raise FileNotFoundError(f"Missing merged template inputs for unit {uid}")
            tmpl = np.load(merged_tmpl_npy)
            locs = np.load(merged_locs_npy)
            unit_summary["inputs"]["selected_template_source"] = "merged_contributing"
            overview_locations_npy = str(merged_locs_npy)

        if tmpl.ndim != 2:
            raise ValueError(f"Unexpected template shape for unit {uid}: {tmpl.shape}")
        if locs.ndim != 2 or locs.shape[1] < 2:
            raise ValueError(f"Unexpected locations shape for unit {uid}: {locs.shape}")

        tmpl_ch_by_t = np.asarray(tmpl).T
        locs_xy = np.asarray(locs)[:, :2]

        if bool(replot_summaries_only):
            from .plotting import write_unit_summary_plots_from_disk

            fs_hz_for_summary: Optional[float] = None
            try:
                meta_path = full_meta_json if (use_full and full_meta_json.exists()) else merged_meta_json
                meta = _read_json(meta_path)
                if meta.get("sampling_frequency_hz") is not None:
                    fs_hz_for_summary = float(meta.get("sampling_frequency_hz"))
            except Exception:
                fs_hz_for_summary = None

            out = write_unit_summary_plots_from_disk(
                uid=uid,
                out_unit_dir=out_unit_dir,
                template_ch_by_t=tmpl_ch_by_t,
                locs_xy=locs_xy,
                fs_hz=fs_hz_for_summary,
                force_restart=bool(force_restart),
                logger=logger,
            )
            unit_summary["outputs"].update(out)
            return {
                "unit_summary": unit_summary,
                "unit_polylines": unit_polylines,
                "overview_locations_npy": None,
            }

        if bool(recompute_branches_raw_only):
            fs_hz = None
            try:
                meta_path = full_meta_json if (use_full and full_meta_json.exists()) else merged_meta_json
                meta = _read_json(meta_path)
                fs_hz = (
                    float(meta.get("sampling_frequency_hz"))
                    if meta.get("sampling_frequency_hz") is not None
                    else None
                )
            except Exception:
                fs_hz = None
            if fs_hz is None:
                fs_hz = 10_000.0

            gtr = av.compute_graph_propagation_velocity(tmpl_ch_by_t, locs_xy, float(fs_hz), **params)

            if bool(per_unit_write_branches_raw_json):
                raw_branches_json = out_unit_dir / "branches_raw.json"
                raw_path, raw_warn = _write_branches_raw_json(
                    uid=uid,
                    gtr=gtr,
                    raw_branches_json=raw_branches_json,
                    logger=logger,
                )
                if raw_path is not None:
                    unit_summary["outputs"]["branches_raw_json"] = raw_path
                if raw_warn is not None:
                    logger.warning("Unit %s: %s", uid, raw_warn)

            return {
                "unit_summary": unit_summary,
                "unit_polylines": unit_polylines,
                "overview_locations_npy": None,
            }

        fs_hz = None
        try:
            meta_path = full_meta_json if (use_full and full_meta_json.exists()) else merged_meta_json
            meta = _read_json(meta_path)
            fs_hz = (
                float(meta.get("sampling_frequency_hz"))
                if meta.get("sampling_frequency_hz") is not None
                else None
            )
        except Exception:
            fs_hz = None
        if fs_hz is None:
            fs_hz = 10_000.0

        gtr = av.compute_graph_propagation_velocity(tmpl_ch_by_t, locs_xy, float(fs_hz), **params)

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
                unit_polylines.append({"unit_id": _jsonable(uid), "branch_index": int(bi), "polyline_xy": xy})

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
        raw_branches_json = out_unit_dir / "branches_raw.json"
        heuristics_json = out_unit_dir / "heuristics.json"
        if bool(per_unit_write_branches_json):
            _write_json(branches_json, {"unit_id": _jsonable(uid), "branches": branches_out})
            unit_summary["outputs"]["branches_json"] = str(branches_json)

        if bool(per_unit_write_heuristics_json):
            _write_json(heuristics_json, {"unit_id": _jsonable(uid), "heuristics": heuristics})
            unit_summary["outputs"]["heuristics_json"] = str(heuristics_json)

        if bool(per_unit_write_branches_raw_json):
            raw_path, raw_warn = _write_branches_raw_json(
                uid=uid,
                gtr=gtr,
                raw_branches_json=raw_branches_json,
                logger=logger,
            )
            if raw_path is not None:
                unit_summary["outputs"]["branches_raw_json"] = raw_path
            if raw_warn is not None:
                logger.warning("Unit %s: %s", uid, raw_warn)

        if write_unit_pdfs:
            try:
                plot_outputs = write_unit_reconstruction_pdfs(
                    uid=uid,
                    gtr=gtr,
                    locs_xy=locs_xy,
                    out_unit_dir=out_unit_dir,
                    force_restart=bool(force_restart),
                    write_template_movie_gif=bool(write_template_movie_gif),
                    logger=logger,
                )
                unit_summary["outputs"].update(plot_outputs)
            except Exception:
                pass

    except Exception as e:
        unit_summary["status"] = "error"
        unit_summary["error"] = exception_to_error_dict(e)
        try:
            _write_json(unit_summary_json, unit_summary)
            _write_json(
                unit_checkpoint_json,
                {
                    "unit_id": _jsonable(uid),
                    "status": "failed",
                    "unit_summary_json": str(unit_summary_json),
                    "error": unit_summary["error"],
                },
            )
        except Exception:
            pass
        try:
            if out_unit_dir.exists() and out_unit_dir.is_dir() and (not any(out_unit_dir.iterdir())):
                out_unit_dir.rmdir()
        except Exception:
            pass

    if str(unit_summary.get("status", "")).lower() == "ok":
        try:
            _write_json(unit_summary_json, unit_summary)
            _write_json(
                unit_checkpoint_json,
                {
                    "unit_id": _jsonable(uid),
                    "status": "completed",
                    "unit_summary_json": str(unit_summary_json),
                },
            )
        except Exception:
            pass

    return {
        "unit_summary": unit_summary,
        "unit_polylines": unit_polylines,
        "overview_locations_npy": overview_locations_npy,
    }


def reconstruct_from_templates(*, inputs: ReconstructionInputs, logger_name_prefix: str = "axon_reconstructor") -> ReconstructionOutputs:
    """Run axon reconstruction/velocity estimation using axon_velocity.

        This stage primarily consumes the dense full-channel template artifacts from templates extraction:
            <well>/stg4_templates_outputs/templates/full/unit_<id>/full_template.npy
            <well>/stg4_templates_outputs/templates/full/unit_<id>/full_channel_locations_xy.npy

        It also reads (best-effort) sampling frequency metadata from:
            <well>/stg4_templates_outputs/templates/merged/unit_<id>/merged_contributing_template_meta.json

    And produces:
      <well>/stg5_reconstruction_outputs/
        reconstruction_summary.json
        all_units_morphology.pdf (optional)
        by_unit/unit_<id>/ ... per-unit json + pdfs

    Focused subset (requested):
    - sparse morphology per unit (branches as polylines)
    - heuristics per unit (selected channels + node heuristics)
    - velocity plots per unit/branch
    - all-units morphology overview
    """

    well_out_dir = compute_mea_analysis_output_dir(
        output_root=inputs.mea_output_root,
        data_file=inputs.h5_path,
        well=inputs.stream_id,
    )

    logger = build_stage_logger(
        well_out_dir=well_out_dir,
        data_file=inputs.h5_path,
        stream_id=inputs.stream_id,
        stage_name="reconstruction",
        logger_name_prefix=logger_name_prefix,
        verbose=True,
    )

    templates_variant = str(inputs.templates_variant_name).strip() if inputs.templates_variant_name else ""
    reconstruction_variant = (
        str(inputs.reconstruction_variant_name).strip() if inputs.reconstruction_variant_name else ""
    )
    if not reconstruction_variant:
        reconstruction_variant = templates_variant

    templates_outputs_dirname = "stg4_templates_outputs" + (f"_{templates_variant}" if templates_variant else "")
    reconstruction_outputs_dirname = RECONSTRUCTION_OUTPUTS_DIRNAME + (
        f"_{reconstruction_variant}" if reconstruction_variant else ""
    )
    reconstruction_stage_name = "reconstruction" + (f"_{reconstruction_variant}" if reconstruction_variant else "")

    recon_out_dir = well_out_dir / reconstruction_outputs_dirname
    by_unit_dir = recon_out_dir / "by_unit"
    summary_json = recon_out_dir / "reconstruction_summary.json"
    if bool(inputs.replot_summaries_only) and bool(inputs.recompute_branches_raw_only):
        raise ValueError("replot_summaries_only and recompute_branches_raw_only are mutually exclusive")

    # In modes that skip normal plotting, avoid generating the all-units overview artifact.
    all_units_overview_pdf = (
        None
        if (bool(inputs.replot_summaries_only) or bool(inputs.recompute_branches_raw_only))
        else (recon_out_dir / "all_units_morphology.pdf" if inputs.write_all_units_overview_pdf else None)
    )

    templates_out_dir = well_out_dir / templates_outputs_dirname
    top_n_density_raw = getattr(inputs, "top_n_density_requested", None)
    if top_n_density_raw is None:
        top_n_density_requested: Optional[int] = None
    else:
        try:
            top_n_parsed = int(top_n_density_raw)
            top_n_density_requested = top_n_parsed if top_n_parsed > 0 else None
        except Exception:
            top_n_density_requested = None

    def _load_all_unit_summaries_from_by_unit_dir() -> list[dict[str, Any]]:
        units_out: list[dict[str, Any]] = []
        for p in sorted(by_unit_dir.glob("unit_*")):
            if not p.is_dir():
                continue
            tok = p.name.split("unit_", 1)[-1]
            try:
                uid: Any = int(tok)
            except Exception:
                uid = tok

            unit_summary_path = p / "unit_reconstruction_summary.json"
            if unit_summary_path.exists():
                try:
                    payload = _read_json(unit_summary_path)
                    if isinstance(payload, dict):
                        if payload.get("unit_id") is None:
                            payload["unit_id"] = _jsonable(uid)
                        units_out.append(payload)
                        continue
                except Exception:
                    pass

            fallback: dict[str, Any] = {
                "unit_id": _jsonable(uid),
                "inputs": {},
                "outputs": {
                    "branches_raw_json": (str(p / "branches_raw.json") if (p / "branches_raw.json").exists() else None),
                    "branches_json": (str(p / "branches.json") if (p / "branches.json").exists() else None),
                    "heuristics_json": (str(p / "heuristics.json") if (p / "heuristics.json").exists() else None),
                },
                "status": ("ok" if (p / "branches_raw.json").exists() else "unknown"),
                "error": None,
            }
            units_out.append(fallback)

        return units_out

    ckpt_file = compute_stage_checkpoint_file(
        well_out_dir=well_out_dir,
        h5_path=inputs.h5_path,
        stream_id=inputs.stream_id,
        stage_name=reconstruction_stage_name,
    )
    ckpt = load_checkpoint(
        checkpoint_file=ckpt_file,
        force_restart=bool(inputs.force_restart),
        output_dir=well_out_dir,
        file_path=inputs.h5_path,
        stream_id=inputs.stream_id,
    )

    # Grid-only mode: update top-density grid from existing reconstruction outputs
    # without recomputing per-unit reconstruction.
    if bool(getattr(inputs, "replot_top_density_grid_only", False)):
        if not by_unit_dir.exists():
            raise FileNotFoundError(
                "replot_top_density_grid_only requires existing reconstruction outputs in the active reconstruction "
                f"out_dir={recon_out_dir}. Missing required artifact: by_unit_dir={by_unit_dir}. "
                "Run reconstruction once without --recon-replot-top-density-grid-only (or disable AXON_RECON_RECON_REPLOT_TOP_DENSITY_GRID_ONLY) "
                "to create per-unit outputs before using grid-only mode."
            )

        if summary_json.exists():
            summary_existing_raw = _read_json(summary_json)
            summary_existing = summary_existing_raw if isinstance(summary_existing_raw, dict) else {}
        else:
            summary_existing = {
                "h5_path": str(inputs.h5_path),
                "stream_id": inputs.stream_id,
                "well_out_dir": str(well_out_dir),
                "templates_variant_name": (templates_variant if templates_variant else None),
                "reconstruction_variant_name": (reconstruction_variant if reconstruction_variant else None),
                "templates_out_dir": str(templates_out_dir),
                "reconstruction_out_dir": str(recon_out_dir),
                "units": [],
            }

        existing_units = [u for u in list(summary_existing.get("units", []) or []) if isinstance(u, dict)]
        refreshed_units = _load_all_unit_summaries_from_by_unit_dir()
        summary_existing["units"] = refreshed_units
        logger.info(
            "Reconstruction grid-only: refreshed summary units from by_unit on disk (%d -> %d)",
            len(existing_units),
            len(refreshed_units),
        )

        selected_unit_ids: list[Any] = []
        for u in list((summary_existing or {}).get("units", []) or []):
            if isinstance(u, dict) and (u.get("unit_id") is not None):
                selected_unit_ids.append(u.get("unit_id"))

        if not selected_unit_ids:
            for p in sorted(by_unit_dir.glob("unit_*")):
                if not p.is_dir():
                    continue
                tok = p.name.split("unit_", 1)[-1]
                try:
                    selected_unit_ids.append(int(tok))
                except Exception:
                    selected_unit_ids.append(tok)

        if bool(inputs.write_top_density_grid):
            density_grid = write_top_density_raw_branch_footprint_grid(
                templates_out_dir=templates_out_dir,
                reconstruction_out_dir=recon_out_dir,
                selected_unit_ids=selected_unit_ids,
                top_n=top_n_density_requested,
                output_subdir=str(inputs.grid_output_subdir),
                write_pdf=bool(inputs.grid_write_pdf),
                write_png=bool(inputs.grid_write_png),
                write_ranking_json=bool(inputs.grid_write_ranking_json),
                log_basename=str(inputs.grid_log_basename),
                linear_basename=str(inputs.grid_linear_basename),
                ranking_filename=str(inputs.grid_ranking_filename),
                ncols=int(inputs.grid_ncols),
                dpi=int(inputs.grid_dpi),
                draw_zoom_range_box=bool(inputs.grid_draw_zoom_range_box),
                panel_background_color=str(inputs.grid_panel_background_color),
                branch_color=str(inputs.grid_branch_color),
                branch_outline_color=str(inputs.grid_branch_outline_color),
                node_radius_um=float(inputs.grid_node_radius_um),
                soma_node_radius_um=float(inputs.grid_soma_node_radius_um),
                soma_node_color=str(inputs.grid_soma_node_color),
                sort_by=str(inputs.grid_sort_by),
                zoom_priority=str(inputs.grid_zoom_priority),
                zoom_padding_percent=float(inputs.grid_zoom_padding_percent),
                force_soma_centering=bool(inputs.grid_force_soma_centering),
                soma_xy_show=bool(inputs.grid_soma_xy_show),
                soma_xy_color=str(inputs.grid_soma_xy_color),
                soma_xy_fontsize=float(inputs.grid_soma_xy_fontsize),
                soma_xy_location=str(inputs.grid_soma_xy_location),
                show_unit_id_in_plot=bool(inputs.grid_show_unit_id_in_plot),
                unit_id_fontsize=float(inputs.grid_unit_id_fontsize),
                unit_id_color=str(inputs.grid_unit_id_color),
                show_minimap=bool(inputs.grid_show_minimap),
                minimap_position=str(inputs.grid_minimap_position),
                minimap_size=float(inputs.grid_minimap_size),
                minimap_outline_color=str(inputs.grid_minimap_outline_color),
                minimap_chip_width_mm=float(inputs.grid_minimap_chip_width_mm),
                minimap_chip_height_mm=float(inputs.grid_minimap_chip_height_mm),
                minimap_inner_box_linestyle=str(inputs.grid_minimap_inner_box_linestyle),
                minimap_inner_box_linewidth=float(inputs.grid_minimap_inner_box_linewidth),
                minimap_include_footprint=bool(inputs.grid_minimap_include_footprint),
                minimap_prevent_occlusions=bool(inputs.grid_minimap_prevent_occlusions),
                legend_show=bool(inputs.grid_legend_show),
                legend_location=str(inputs.grid_legend_location),
                legend_fontsize=float(inputs.grid_legend_fontsize),
                legend_fontcolor=str(inputs.grid_legend_fontcolor),
                legend_marker_size=float(inputs.grid_legend_marker_size),
                legend_show_nodes_in_legend=bool(inputs.grid_legend_show_nodes_in_legend),
                legend_show_footprint_in_legend=bool(inputs.grid_legend_show_footprint_in_legend),
                emit_debug_logs=bool(inputs.grid_emit_debug_logs),
                show_scale_debug_text=bool(inputs.show_density_scale_debug_text),
                show_global_debug_text=bool(inputs.show_density_scale_global_debug_text),
                show_local_debug_text=bool(inputs.show_density_scale_local_debug_text),
                logger=logger,
            )
            if isinstance(density_grid, dict) and density_grid:
                summary_existing.update({k: v for k, v in density_grid.items() if v is not None})
                _write_json(summary_json, summary_existing)

        logger.info("Replotted top-density grid only from existing reconstruction outputs at %s", recon_out_dir)
        return ReconstructionOutputs(
            well_out_dir=well_out_dir,
            reconstruction_out_dir=recon_out_dir,
            summary_json=summary_json,
            by_unit_dir=by_unit_dir,
            all_units_overview_pdf=(all_units_overview_pdf if (all_units_overview_pdf and all_units_overview_pdf.exists()) else None),
        )

    # Resume shortcut.
    if (
        (not inputs.force_restart)
        and (not bool(inputs.replot_summaries_only))
        and (not bool(inputs.recompute_branches_raw_only))
        and summary_json.exists()
        and ((all_units_overview_pdf is None) or all_units_overview_pdf.exists())
        and by_unit_dir.exists()
    ):
        try:
            summary_existing_raw = _read_json(summary_json)
            summary_existing = summary_existing_raw if isinstance(summary_existing_raw, dict) else {}
            existing_units = [u for u in list(summary_existing.get("units", []) or []) if isinstance(u, dict)]
            refreshed_units = _load_all_unit_summaries_from_by_unit_dir()
            summary_existing["units"] = refreshed_units
            logger.info(
                "Reconstruction resume: refreshed summary units from by_unit on disk (%d -> %d)",
                len(existing_units),
                len(refreshed_units),
            )
            selected_unit_ids: list[Any] = []
            for u in list((summary_existing or {}).get("units", []) or []):
                if isinstance(u, dict) and (u.get("unit_id") is not None):
                    selected_unit_ids.append(u.get("unit_id"))

            if bool(inputs.write_top_density_grid):
                density_grid = write_top_density_raw_branch_footprint_grid(
                    templates_out_dir=templates_out_dir,
                    reconstruction_out_dir=recon_out_dir,
                    selected_unit_ids=selected_unit_ids,
                    top_n=top_n_density_requested,
                    output_subdir=str(inputs.grid_output_subdir),
                    write_pdf=bool(inputs.grid_write_pdf),
                    write_png=bool(inputs.grid_write_png),
                    write_ranking_json=bool(inputs.grid_write_ranking_json),
                    log_basename=str(inputs.grid_log_basename),
                    linear_basename=str(inputs.grid_linear_basename),
                    ranking_filename=str(inputs.grid_ranking_filename),
                    ncols=int(inputs.grid_ncols),
                    dpi=int(inputs.grid_dpi),
                    draw_zoom_range_box=bool(inputs.grid_draw_zoom_range_box),
                    panel_background_color=str(inputs.grid_panel_background_color),
                    branch_color=str(inputs.grid_branch_color),
                    branch_outline_color=str(inputs.grid_branch_outline_color),
                    node_radius_um=float(inputs.grid_node_radius_um),
                    soma_node_radius_um=float(inputs.grid_soma_node_radius_um),
                    soma_node_color=str(inputs.grid_soma_node_color),
                    sort_by=str(inputs.grid_sort_by),
                    zoom_priority=str(inputs.grid_zoom_priority),
                    zoom_padding_percent=float(inputs.grid_zoom_padding_percent),
                    force_soma_centering=bool(inputs.grid_force_soma_centering),
                    soma_xy_show=bool(inputs.grid_soma_xy_show),
                    soma_xy_color=str(inputs.grid_soma_xy_color),
                    soma_xy_fontsize=float(inputs.grid_soma_xy_fontsize),
                    soma_xy_location=str(inputs.grid_soma_xy_location),
                    show_unit_id_in_plot=bool(inputs.grid_show_unit_id_in_plot),
                    unit_id_fontsize=float(inputs.grid_unit_id_fontsize),
                    unit_id_color=str(inputs.grid_unit_id_color),
                    show_minimap=bool(inputs.grid_show_minimap),
                    minimap_position=str(inputs.grid_minimap_position),
                    minimap_size=float(inputs.grid_minimap_size),
                    minimap_outline_color=str(inputs.grid_minimap_outline_color),
                    minimap_chip_width_mm=float(inputs.grid_minimap_chip_width_mm),
                    minimap_chip_height_mm=float(inputs.grid_minimap_chip_height_mm),
                    minimap_inner_box_linestyle=str(inputs.grid_minimap_inner_box_linestyle),
                    minimap_inner_box_linewidth=float(inputs.grid_minimap_inner_box_linewidth),
                    minimap_include_footprint=bool(inputs.grid_minimap_include_footprint),
                    minimap_prevent_occlusions=bool(inputs.grid_minimap_prevent_occlusions),
                    legend_show=bool(inputs.grid_legend_show),
                    legend_location=str(inputs.grid_legend_location),
                    legend_fontsize=float(inputs.grid_legend_fontsize),
                    legend_fontcolor=str(inputs.grid_legend_fontcolor),
                    legend_marker_size=float(inputs.grid_legend_marker_size),
                    legend_show_nodes_in_legend=bool(inputs.grid_legend_show_nodes_in_legend),
                    legend_show_footprint_in_legend=bool(inputs.grid_legend_show_footprint_in_legend),
                    emit_debug_logs=bool(inputs.grid_emit_debug_logs),
                    show_scale_debug_text=bool(inputs.show_density_scale_debug_text),
                    show_global_debug_text=bool(inputs.show_density_scale_global_debug_text),
                    show_local_debug_text=bool(inputs.show_density_scale_local_debug_text),
                    logger=logger,
                )
                if isinstance(density_grid, dict) and density_grid:
                    summary_existing.update({k: v for k, v in density_grid.items() if v is not None})
                    _write_json(summary_json, summary_existing)
        except Exception as e:
            logger.exception("Resume replot of top-density grid failed")
            raise RuntimeError(
                "Resume replot of top-density grid failed; aborting resume to avoid stale grid artifacts."
            ) from e

        logger.info("Resuming reconstruction: existing outputs found at %s", recon_out_dir)
        return ReconstructionOutputs(
            well_out_dir=well_out_dir,
            reconstruction_out_dir=recon_out_dir,
            summary_json=summary_json,
            by_unit_dir=by_unit_dir,
            all_units_overview_pdf=(all_units_overview_pdf if (all_units_overview_pdf and all_units_overview_pdf.exists()) else None),
        )

    ckpt = save_stage_started(
        checkpoint_file=ckpt_file,
        state=ckpt,
        stage=ProcessingStage.ANALYZER,
        out_dir=recon_out_dir,
        extra_fields={"reconstruction_out_dir": str(recon_out_dir)},
    )
    log_stage_start(logger, stage="reconstruction", out_dir=str(recon_out_dir))

    def _mark_failed_and_raise(exc: Exception | BaseException, *, failed_stage: str = "RECONSTRUCTION") -> None:
        save_stage_failed(
            checkpoint_file=ckpt_file,
            state=ckpt,
            stage=ProcessingStage.ANALYZER,
            failed_stage=failed_stage,
            error=exc,
            extra_fields={"reconstruction_out_dir": str(recon_out_dir)},
        )
        log_stage_failure(
            logger,
            stage="reconstruction",
            checkpoint_file=str(ckpt_file),
            error=exc,
        )
        raise exc

    recon_out_dir.mkdir(parents=True, exist_ok=True)
    by_unit_dir.mkdir(parents=True, exist_ok=True)

    templates_dir = templates_out_dir / "templates"

    merged_units_dir = templates_dir / "merged"
    full_channels_templates_dir = templates_dir / "full"

    # Legacy fallback.
    if not merged_units_dir.exists():
        legacy = templates_out_dir / "merged_units"
        if legacy.exists():
            merged_units_dir = legacy
    if not full_channels_templates_dir.exists():
        legacy = templates_out_dir / "full_channels_templates"
        if legacy.exists():
            full_channels_templates_dir = legacy
    if not merged_units_dir.exists():
        _mark_failed_and_raise(FileNotFoundError(f"Missing merged templates at {merged_units_dir}"), failed_stage="RECON_INPUTS")
    if bool(inputs.use_full_channels_templates) and bool(inputs.require_full_channels_templates):
        if not full_channels_templates_dir.exists():
            _mark_failed_and_raise(
                FileNotFoundError(
                    "Reconstruction is configured to require full-channel templates for axon_velocity, "
                    f"but {full_channels_templates_dir} does not exist. "
                    "Re-run templates with save_full_channels_templates=True, or set "
                    "ReconstructionInputs(require_full_channels_templates=False)."
                ),
                failed_stage="RECON_INPUTS",
            )

    try:
        import numpy as np  # type: ignore[import-not-found]
    except Exception as e:  # pragma: no cover
        _mark_failed_and_raise(RuntimeError("Reconstruction requires numpy"), failed_stage="RECON_DEPENDENCIES")

    try:
        av = _import_axon_velocity(repo_root=inputs.axon_velocity_repo_root)
    except Exception as e:
        _mark_failed_and_raise(e, failed_stage="RECON_DEPENDENCIES")

    # Determine unit list.
    discovered_unit_ids: list[Any] = []
    for p in sorted(merged_units_dir.glob("unit_*")):
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
        "templates_variant_name": (templates_variant if templates_variant else None),
        "reconstruction_variant_name": (reconstruction_variant if reconstruction_variant else None),
        "templates_out_dir": str(templates_out_dir),
        "templates_merged_units_dir": str(merged_units_dir),
        "templates_full_channels_templates_dir": (
            str(full_channels_templates_dir) if full_channels_templates_dir.exists() else None
        ),
        "reconstruction_out_dir": str(recon_out_dir),
        "axon_velocity_params": {k: _jsonable(v) for k, v in params.items()},
        "template_source": {
            "use_full_channels_templates": bool(inputs.use_full_channels_templates),
            "require_full_channels_templates": bool(inputs.require_full_channels_templates),
        },
        "unit_workers": int(max(1, int(inputs.unit_workers))),
        "units": [],
    }

    all_unit_polylines: list[dict[str, Any]] = []
    all_locations: list[Any] = []

    unit_workers = max(1, int(inputs.unit_workers))
    total_units = len(unit_ids)
    logger.info("Reconstructing %d units with unit_workers=%d", total_units, unit_workers)

    def _accumulate_unit_result(result: dict[str, Any]) -> None:
        unit_summary = dict(result.get("unit_summary") or {})
        summary["units"].append(unit_summary)
        unit_polys = result.get("unit_polylines") or []
        if unit_polys:
            all_unit_polylines.extend(unit_polys)

        if bool(inputs.replot_summaries_only) or bool(inputs.recompute_branches_raw_only):
            return

        loc_path_raw = result.get("overview_locations_npy")
        if not loc_path_raw:
            return
        try:
            loc_arr = np.load(Path(str(loc_path_raw)))
            if getattr(loc_arr, "ndim", 0) == 2 and loc_arr.shape[1] >= 2:
                all_locations.append(np.asarray(loc_arr)[:, :2])
        except Exception as e:
            logger.warning("Failed loading overview locations for unit %s: %s", unit_summary.get("unit_id"), e)

    if unit_workers == 1:
        completed_units = 0
        for idx, uid in enumerate(unit_ids, start=1):
            logger.info("[reconstruction] unit start %d/%d: unit_id=%s", idx, total_units, uid)
            result = _run_single_unit_reconstruction(
                uid=uid,
                reconstruction_out_dir=recon_out_dir,
                merged_units_dir=merged_units_dir,
                full_channels_templates_dir=full_channels_templates_dir,
                use_full_channels_templates=bool(inputs.use_full_channels_templates),
                require_full_channels_templates=bool(inputs.require_full_channels_templates),
                axon_velocity_repo_root=inputs.axon_velocity_repo_root,
                params=params,
                force_restart=bool(inputs.force_restart),
                write_unit_pdfs=bool(inputs.write_unit_pdfs),
                write_template_movie_gif=bool(inputs.write_template_movie_gif),
                replot_summaries_only=bool(inputs.replot_summaries_only),
                recompute_branches_raw_only=bool(inputs.recompute_branches_raw_only),
                per_unit_write_branches_raw_json=bool(inputs.per_unit_write_branches_raw_json),
                per_unit_write_branches_json=bool(inputs.per_unit_write_branches_json),
                per_unit_write_heuristics_json=bool(inputs.per_unit_write_heuristics_json),
            )
            _accumulate_unit_result(result)
            completed_units += 1
            unit_status = str((result.get("unit_summary") or {}).get("status", "unknown"))
            resume_skipped = bool((result.get("unit_summary") or {}).get("resume_skipped_completed", False))
            logger.info(
                "[reconstruction] unit done %d/%d: unit_id=%s status=%s resume_skipped=%s",
                completed_units,
                total_units,
                uid,
                unit_status,
                str(resume_skipped).lower(),
            )
    else:
        futures: dict[concurrent.futures.Future[dict[str, Any]], Any] = {}
        with concurrent.futures.ProcessPoolExecutor(max_workers=unit_workers) as pool:
            for uid in unit_ids:
                logger.info("[reconstruction] unit queued: unit_id=%s", uid)
                fut = pool.submit(
                    _run_single_unit_reconstruction,
                    uid=uid,
                    reconstruction_out_dir=recon_out_dir,
                    merged_units_dir=merged_units_dir,
                    full_channels_templates_dir=full_channels_templates_dir,
                    use_full_channels_templates=bool(inputs.use_full_channels_templates),
                    require_full_channels_templates=bool(inputs.require_full_channels_templates),
                    axon_velocity_repo_root=inputs.axon_velocity_repo_root,
                    params=params,
                    force_restart=bool(inputs.force_restart),
                    write_unit_pdfs=bool(inputs.write_unit_pdfs),
                    write_template_movie_gif=bool(inputs.write_template_movie_gif),
                    replot_summaries_only=bool(inputs.replot_summaries_only),
                    recompute_branches_raw_only=bool(inputs.recompute_branches_raw_only),
                    per_unit_write_branches_raw_json=bool(inputs.per_unit_write_branches_raw_json),
                    per_unit_write_branches_json=bool(inputs.per_unit_write_branches_json),
                    per_unit_write_heuristics_json=bool(inputs.per_unit_write_heuristics_json),
                )
                futures[fut] = uid

            completed_units = 0
            for fut in concurrent.futures.as_completed(futures):
                uid = futures[fut]
                try:
                    result = fut.result()
                except Exception as e:
                    result = {
                        "unit_summary": {
                            "unit_id": _jsonable(uid),
                            "inputs": {},
                            "outputs": {},
                            "status": "error",
                            "error": exception_to_error_dict(e),
                        },
                        "unit_polylines": [],
                        "overview_locations_npy": None,
                    }
                _accumulate_unit_result(result)
                completed_units += 1
                unit_status = str((result.get("unit_summary") or {}).get("status", "unknown"))
                resume_skipped = bool((result.get("unit_summary") or {}).get("resume_skipped_completed", False))
                logger.info(
                    "[reconstruction] unit done %d/%d: unit_id=%s status=%s resume_skipped=%s",
                    completed_units,
                    total_units,
                    uid,
                    unit_status,
                    str(resume_skipped).lower(),
                )

    # Keep summary stable in unit_id order.
    id_to_index = {str(_jsonable(uid)): i for i, uid in enumerate(unit_ids)}
    summary["units"].sort(key=lambda x: id_to_index.get(str(x.get("unit_id")), len(id_to_index)))

    # All-units overview morphology plot.
    if (not bool(inputs.replot_summaries_only)) and (all_units_overview_pdf is not None):
        try:
            ok = write_all_units_overview_pdf(
                all_units_overview_pdf=all_units_overview_pdf,
                all_locations=all_locations,
                all_unit_polylines=all_unit_polylines,
                stream_id=inputs.stream_id,
                force_restart=bool(inputs.force_restart),
                logger=logger,
            )
            if ok:
                summary["all_units_overview_pdf"] = str(all_units_overview_pdf)
                all_units_overview_png = all_units_overview_pdf.with_suffix(".png")
                if all_units_overview_png.exists():
                    summary["all_units_overview_png"] = str(all_units_overview_png)
        except Exception as e:
            logger.warning("Failed writing all-units overview pdf: %s", e)

    # Top-density (spikes/channel/area) raw-branch morphology overlays on log footprints.
    if bool(inputs.write_top_density_grid) and (not bool(inputs.recompute_branches_raw_only)):
        try:
            density_grid = write_top_density_raw_branch_footprint_grid(
                templates_out_dir=templates_out_dir,
                reconstruction_out_dir=recon_out_dir,
                selected_unit_ids=list(unit_ids),
                top_n=top_n_density_requested,
                output_subdir=str(inputs.grid_output_subdir),
                write_pdf=bool(inputs.grid_write_pdf),
                write_png=bool(inputs.grid_write_png),
                write_ranking_json=bool(inputs.grid_write_ranking_json),
                log_basename=str(inputs.grid_log_basename),
                linear_basename=str(inputs.grid_linear_basename),
                ranking_filename=str(inputs.grid_ranking_filename),
                ncols=int(inputs.grid_ncols),
                dpi=int(inputs.grid_dpi),
                draw_zoom_range_box=bool(inputs.grid_draw_zoom_range_box),
                panel_background_color=str(inputs.grid_panel_background_color),
                branch_color=str(inputs.grid_branch_color),
                branch_outline_color=str(inputs.grid_branch_outline_color),
                node_radius_um=float(inputs.grid_node_radius_um),
                soma_node_radius_um=float(inputs.grid_soma_node_radius_um),
                soma_node_color=str(inputs.grid_soma_node_color),
                sort_by=str(inputs.grid_sort_by),
                zoom_priority=str(inputs.grid_zoom_priority),
                zoom_padding_percent=float(inputs.grid_zoom_padding_percent),
                force_soma_centering=bool(inputs.grid_force_soma_centering),
                soma_xy_show=bool(inputs.grid_soma_xy_show),
                soma_xy_color=str(inputs.grid_soma_xy_color),
                soma_xy_fontsize=float(inputs.grid_soma_xy_fontsize),
                soma_xy_location=str(inputs.grid_soma_xy_location),
                show_unit_id_in_plot=bool(inputs.grid_show_unit_id_in_plot),
                unit_id_fontsize=float(inputs.grid_unit_id_fontsize),
                unit_id_color=str(inputs.grid_unit_id_color),
                show_minimap=bool(inputs.grid_show_minimap),
                minimap_position=str(inputs.grid_minimap_position),
                minimap_size=float(inputs.grid_minimap_size),
                minimap_outline_color=str(inputs.grid_minimap_outline_color),
                minimap_chip_width_mm=float(inputs.grid_minimap_chip_width_mm),
                minimap_chip_height_mm=float(inputs.grid_minimap_chip_height_mm),
                minimap_inner_box_linestyle=str(inputs.grid_minimap_inner_box_linestyle),
                minimap_inner_box_linewidth=float(inputs.grid_minimap_inner_box_linewidth),
                minimap_include_footprint=bool(inputs.grid_minimap_include_footprint),
                minimap_prevent_occlusions=bool(inputs.grid_minimap_prevent_occlusions),
                legend_show=bool(inputs.grid_legend_show),
                legend_location=str(inputs.grid_legend_location),
                legend_fontsize=float(inputs.grid_legend_fontsize),
                legend_fontcolor=str(inputs.grid_legend_fontcolor),
                legend_marker_size=float(inputs.grid_legend_marker_size),
                legend_show_nodes_in_legend=bool(inputs.grid_legend_show_nodes_in_legend),
                legend_show_footprint_in_legend=bool(inputs.grid_legend_show_footprint_in_legend),
                emit_debug_logs=bool(inputs.grid_emit_debug_logs),
                show_scale_debug_text=bool(inputs.show_density_scale_debug_text),
                show_global_debug_text=bool(inputs.show_density_scale_global_debug_text),
                show_local_debug_text=bool(inputs.show_density_scale_local_debug_text),
                logger=logger,
            )
            if isinstance(density_grid, dict) and density_grid:
                summary.update({k: v for k, v in density_grid.items() if v is not None})
        except Exception as e:
            logger.warning("Failed writing top-density raw-branch footprint grid: %s", e)

    try:
        _write_json(summary_json, summary)

        ckpt = save_stage_completed(
            checkpoint_file=ckpt_file,
            state=ckpt,
            stage=ProcessingStage.ANALYZER_COMPLETE,
            extra_fields={
                "reconstruction_out_dir": str(recon_out_dir),
                "reconstruction_summary_json": str(summary_json),
                "all_units_overview_pdf": str(all_units_overview_pdf) if all_units_overview_pdf else None,
            },
        )
        log_stage_complete(logger, stage="reconstruction", summary_json=str(summary_json), units=len(summary.get("units", [])))

        return ReconstructionOutputs(
            well_out_dir=well_out_dir,
            reconstruction_out_dir=recon_out_dir,
            summary_json=summary_json,
            by_unit_dir=by_unit_dir,
            all_units_overview_pdf=all_units_overview_pdf,
        )
    except Exception as e:
        _mark_failed_and_raise(e)


__all__ = [
    "ReconstructionInputs",
    "ReconstructionOutputs",
    "reconstruct_from_templates",
]
