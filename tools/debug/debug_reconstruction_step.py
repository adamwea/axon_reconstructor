#!/usr/bin/env python3
"""Project-local entrypoint for stepping through reconstruction.

Consumes:
    <well>/templates_outputs/templates/full/unit_<id>/full_template.npy
    <well>/templates_outputs/templates/full/unit_<id>/full_channel_locations_xy.npy
Produces:
  <well>/reconstruction_outputs/
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import debug_env


# Dataset + repo configuration comes from debug.env (or CLI overrides).
AXON_VELOCITY_REPO_ROOT: Path | None = None
H5_PATH: Path | None = None
STREAM_ID: str | None = None
MEA_OUTPUT_ROOT: Path | None = None

DEBUG = False

# If True, ignore existing reconstruction outputs/checkpoint.
FORCE_RESTART = True

# If True, skip tracking and only (re)render per-unit reconstruction summaries.
REPLOT_SUMMARIES_ONLY = False

# If True, rerun AV tracking but only (re)write branches_raw.json (skip plots).
RECOMPUTE_BRANCHES_RAW_ONLY = False

# Keep this low while iterating; set to None to use all units.
UNIT_LIMIT = 100

# Optional allowlist.
UNIT_IDS: list[int] | None = None

# axon_velocity knobs (optional). These are merged onto axon_velocity defaults.
# You can add/remove keys as you iterate.
# Start permissive so we get *some* branches/outputs.
# You can tighten these once the end-to-end plumbing is validated.
AXON_VELOCITY_PARAMS: dict = {}
# Axon Velocity Params:
# NOTE: This dict is intentionally empty so the installed `axon_velocity` defaults apply
# unless you override via `AXON_RECON_AV_*` env vars or `--av-params*` CLI options.

WRITE_UNIT_PDFS = True
WRITE_ALL_UNITS_OVERVIEW_PDF = True
VERBOSE = False
UNIT_WORKERS = 1


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


def _parse_int_or_none(raw: str) -> int | None:
    v = str(raw).strip().lower()
    if v in {"none", "null", "all"}:
        return None
    return int(v)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Debug reconstruction step")
    p.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Path to env file (default: ./debug.env)",
    )
    p.add_argument(
        "--debug",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable debug logging",
    )

    p.add_argument("--axon-velocity-repo-root", type=Path, default=None)
    p.add_argument("--h5-path", type=Path, default=None)
    p.add_argument("--stream-id", type=str, default=None)
    p.add_argument("--mea-output-root", type=Path, default=None)

    p.add_argument(
        "--force-restart",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Ignore cached outputs and re-run reconstruction",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Alias for --force-restart (kept for backwards compatibility)",
    )

    p.add_argument("--unit-limit", type=str, default=None, help="Int or 'none'")
    p.add_argument("--unit-ids", nargs="*", default=None, help="Optional unit ids (space-separated).")

    p.add_argument(
        "--replot-summaries-only",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Only re-render summary_clean/summary_raw plots (skip tracking)",
    )

    p.add_argument(
        "--recompute-branches-raw-only",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Only (re)write branches_raw.json (runs tracking, skips plots)",
    )

    p.add_argument("--r2-threshold", type=float, default=None, help="Override AV (axon_velocity) r2_threshold")

    # AV params overrides
    p.add_argument(
        "--av-params",
        type=str,
        default=None,
        dest="av_params",
        help="JSON string to merge into AV params",
    )
    p.add_argument(
        "--av-params-json",
        type=Path,
        default=None,
        dest="av_params_json",
        help="Path to a JSON file to merge into AV params",
    )

    p.add_argument(
        "--write-unit-pdfs",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Write per-unit reconstruction PDFs",
    )
    p.add_argument(
        "--write-all-units-overview-pdf",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Write all-units overview PDF",
    )
    p.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Verbose reconstruction logging",
    )
    p.add_argument(
        "--unit-workers",
        type=int,
        default=None,
        help="Number of units to reconstruct in parallel within a single well (default: 1)",
    )

    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.env_file is not None:
        env_files = [Path(args.env_file)]
    else:
        env_files = debug_env.default_env_paths(script_path=__file__)
    debug_env.load_env_files_into_os(env_files=env_files, override_existing=False)

    debug_enabled = debug_env.env_bool("AXON_RECON_DEBUG", default=bool(DEBUG)) if args.debug is None else bool(args.debug)
    log_level = logging.DEBUG if debug_enabled else logging.INFO
    logging.basicConfig(level=log_level, format="[%(levelname)s] %(message)s")
    logger = logging.getLogger("projects.debug_reconstruction_step")
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    axon_velocity_repo_root = (
        Path(args.axon_velocity_repo_root)
        if args.axon_velocity_repo_root is not None
        else debug_env.env_required_path("AXON_RECON_AXON_VELOCITY_REPO_ROOT")
    )
    h5_path = Path(args.h5_path) if args.h5_path is not None else debug_env.env_required_path("AXON_RECON_H5_PATH")
    stream_id = str(args.stream_id) if args.stream_id is not None else debug_env.env_required_str("AXON_RECON_STREAM_ID")
    mea_output_root = (
        Path(args.mea_output_root)
        if args.mea_output_root is not None
        else debug_env.env_required_path("AXON_RECON_MEA_OUTPUT_ROOT")
    )

    if args.force_restart is not None:
        force_restart = bool(args.force_restart)
    elif bool(args.force):
        force_restart = True
    else:
        force_restart = debug_env.env_bool("AXON_RECON_FORCE_RESTART", default=bool(FORCE_RESTART))

    unit_limit = _env_int_or_none(
        "AXON_RECON_RECON_UNIT_LIMIT",
        default=_env_int_or_none("AXON_RECON_UNIT_LIMIT", default=(int(UNIT_LIMIT) if UNIT_LIMIT is not None else None)),
    )
    if args.unit_limit is not None:
        unit_limit = _parse_int_or_none(args.unit_limit)

    unit_ids = debug_env.env_int_list("AXON_RECON_UNIT_IDS")
    if args.unit_ids:
        unit_ids = [int(x) for x in args.unit_ids]
    elif UNIT_IDS is not None:
        unit_ids = list(UNIT_IDS)

    # --- AV (axon_velocity) params ---
    # Prefer AXON_RECON_AV_* naming.
    av_params = dict(AXON_VELOCITY_PARAMS)

    # 1) Env JSON override
    av_params = debug_env.merge_json_overrides(
        av_params,
        json_str=debug_env.env_str("AXON_RECON_AV_PARAMS_JSON", default=None),
        json_path=debug_env.env_path("AXON_RECON_AV_PARAMS_JSON_PATH", default=None),
    )

    # 2) Env per-param overrides (AXON_RECON_AV_<PARAM>)
    # Source of truth: axon_velocity.tracking_classes.GraphAxonTracking.default_params
    av_param_keys = [
        "upsample",
        "init_delay",
        "detect_threshold",
        "kurt_threshold",
        "peak_std_threshold",
        "peak_std_distance",
        "remove_isolated",
        "detection_type",
        "min_selected_points",
        "min_path_length",
        "min_path_points",
        "min_points_after_branching",
        "r2_threshold",
        "max_distance_for_edge",
        "max_distance_to_init",
        "mad_threshold",
        "n_neighbors",
        "init_amp_peak_ratio",
        "edge_dist_amp_ratio",
        "distance_exp",
        "max_peak_latency_for_splitting",
        "r2_threshold_for_outliers",
        "min_outlier_tracking_error",
        "theilsen_maxiter",
        "neighbor_radius",
        "split_paths",
    ]
    for k in av_param_keys:
        env_key = f"AXON_RECON_AV_{k.upper()}"
        raw = debug_env.env_str(env_key, default=None)
        if raw is None:
            continue
        av_params[k] = debug_env.parse_typed_value(raw)

    # 3) CLI JSON override (preferred)
    av_params = debug_env.merge_json_overrides(
        av_params,
        json_str=args.av_params,
        json_path=args.av_params_json,
    )

    # 4) CLI convenience override
    if args.r2_threshold is not None:
        av_params["r2_threshold"] = float(args.r2_threshold)

    write_unit_pdfs = (
        debug_env.env_bool("AXON_RECON_RECON_WRITE_UNIT_PDFS", default=bool(WRITE_UNIT_PDFS))
        if args.write_unit_pdfs is None
        else bool(args.write_unit_pdfs)
    )
    write_all_units_overview_pdf = (
        debug_env.env_bool(
            "AXON_RECON_RECON_WRITE_ALL_UNITS_OVERVIEW_PDF",
            default=bool(WRITE_ALL_UNITS_OVERVIEW_PDF),
        )
        if args.write_all_units_overview_pdf is None
        else bool(args.write_all_units_overview_pdf)
    )
    verbose = debug_env.env_bool("AXON_RECON_RECON_VERBOSE", default=bool(VERBOSE)) if args.verbose is None else bool(args.verbose)
    unit_workers = (
        int(debug_env.env_int("AXON_RECON_RECON_UNIT_WORKERS", default=int(UNIT_WORKERS)) or int(UNIT_WORKERS))
        if args.unit_workers is None
        else int(args.unit_workers)
    )
    unit_workers = max(1, unit_workers)

    replot_summaries_only = (
        debug_env.env_bool(
            "AXON_RECON_RECON_REPLOT_SUMMARIES_ONLY",
            default=bool(REPLOT_SUMMARIES_ONLY),
        )
        if args.replot_summaries_only is None
        else bool(args.replot_summaries_only)
    )

    recompute_branches_raw_only = (
        debug_env.env_bool(
            "AXON_RECON_RECON_RECOMPUTE_BRANCHES_RAW_ONLY",
            default=bool(RECOMPUTE_BRANCHES_RAW_ONLY),
        )
        if args.recompute_branches_raw_only is None
        else bool(args.recompute_branches_raw_only)
    )

    # Make axon_velocity importable (even if not installed in this interpreter).
    sys.path.insert(0, str(axon_velocity_repo_root))

    # Quick dependency sanity check (axon_velocity imports sklearn at module import time).
    try:
        import sklearn  # type: ignore[import-not-found]
    except Exception:
        logger.error("Missing dependency: scikit-learn (import sklearn failed). Install via `pip install scikit-learn` or `conda install scikit-learn`.")
        raise

    from axon_reconstructor.pipeline.reconstruction import ReconstructionInputs, reconstruct_from_templates

    inputs = ReconstructionInputs(
        h5_path=h5_path,
        stream_id=stream_id,
        mea_output_root=mea_output_root,
        unit_ids=unit_ids,
        unit_limit=unit_limit,
        axon_velocity_params=av_params,
        write_unit_pdfs=write_unit_pdfs,
        write_all_units_overview_pdf=write_all_units_overview_pdf,
        verbose=verbose,
        unit_workers=unit_workers,
        force_restart=force_restart,
        replot_summaries_only=replot_summaries_only,
        recompute_branches_raw_only=recompute_branches_raw_only,
    )

    logger.info("Debug reconstruction inputs: %s", inputs)
    out = reconstruct_from_templates(inputs=inputs)
    logger.info("Reconstruction written under: %s", out.reconstruction_out_dir)
    logger.info("Summary JSON: %s", out.summary_json)
    if out.all_units_overview_pdf is not None:
        logger.info("All-units overview PDF: %s", out.all_units_overview_pdf)


if __name__ == "__main__":
    main()
