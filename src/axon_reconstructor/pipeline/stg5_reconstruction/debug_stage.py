from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .runner import ReconstructionInputs


def _parse_int_or_none(raw: str) -> int | None:
    v = str(raw).strip().lower()
    if v in {"none", "null", "all"}:
        return None
    return int(v)


def _env_int_or_none(*, env: Any, name: str, default: int | None) -> int | None:
    raw = env.env_str(name, default=None)
    if raw is None:
        return default
    return _parse_int_or_none(raw)


@dataclass(frozen=True)
class ReconstructionDebugConfig:
    inputs: ReconstructionInputs


def build_reconstruction_debug_config(*, args: Any, env: Any) -> ReconstructionDebugConfig:
    axon_velocity_repo_root = (
        Path(args.axon_velocity_repo_root)
        if args.axon_velocity_repo_root is not None
        else env.env_required_path("AXON_RECON_AXON_VELOCITY_REPO_ROOT")
    )
    h5_path = Path(args.h5_path) if args.h5_path is not None else env.env_required_path("AXON_RECON_H5_PATH")
    stream_id = str(args.stream_id) if args.stream_id is not None else env.env_required_str("AXON_RECON_STREAM_ID")
    mea_output_root = (
        Path(args.mea_output_root)
        if args.mea_output_root is not None
        else env.env_required_path("AXON_RECON_MEA_OUTPUT_ROOT")
    )

    if args.force_restart is not None:
        force_restart = bool(args.force_restart)
    elif bool(args.force):
        force_restart = True
    else:
        force_restart = env.env_bool("AXON_RECON_FORCE_RESTART", default=True)

    unit_limit = _env_int_or_none(
        env=env,
        name="AXON_RECON_RECON_UNIT_LIMIT",
        default=_env_int_or_none(env=env, name="AXON_RECON_UNIT_LIMIT", default=100),
    )
    if args.unit_limit is not None:
        unit_limit = _parse_int_or_none(args.unit_limit)

    unit_ids = env.env_int_list("AXON_RECON_UNIT_IDS")
    if args.unit_ids:
        unit_ids = [int(x) for x in args.unit_ids]

    av_params: dict[str, Any] = {}

    av_params = env.merge_json_overrides(
        av_params,
        json_str=env.env_str("AXON_RECON_AV_PARAMS_JSON", default=None),
        json_path=env.env_path("AXON_RECON_AV_PARAMS_JSON_PATH", default=None),
    )

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
    for key in av_param_keys:
        env_key = f"AXON_RECON_AV_{key.upper()}"
        raw = env.env_str(env_key, default=None)
        if raw is None:
            continue
        av_params[key] = env.parse_typed_value(raw)

    av_params = env.merge_json_overrides(
        av_params,
        json_str=args.av_params,
        json_path=args.av_params_json,
    )
    if args.r2_threshold is not None:
        av_params["r2_threshold"] = float(args.r2_threshold)

    write_unit_pdfs = (
        env.env_bool("AXON_RECON_RECON_WRITE_UNIT_PDFS", default=True)
        if args.write_unit_pdfs is None
        else bool(args.write_unit_pdfs)
    )
    write_all_units_overview_pdf = (
        env.env_bool("AXON_RECON_RECON_WRITE_ALL_UNITS_OVERVIEW_PDF", default=True)
        if args.write_all_units_overview_pdf is None
        else bool(args.write_all_units_overview_pdf)
    )
    verbose = env.env_bool("AXON_RECON_RECON_VERBOSE", default=False) if args.verbose is None else bool(args.verbose)
    unit_workers = (
        int(env.env_int("AXON_RECON_RECON_UNIT_WORKERS", default=1) or 1)
        if args.unit_workers is None
        else int(args.unit_workers)
    )
    unit_workers = max(1, unit_workers)

    replot_summaries_only = (
        env.env_bool("AXON_RECON_RECON_REPLOT_SUMMARIES_ONLY", default=False)
        if args.replot_summaries_only is None
        else bool(args.replot_summaries_only)
    )
    recompute_branches_raw_only = (
        env.env_bool("AXON_RECON_RECON_RECOMPUTE_BRANCHES_RAW_ONLY", default=False)
        if args.recompute_branches_raw_only is None
        else bool(args.recompute_branches_raw_only)
    )

    top_n_density_requested = _env_int_or_none(
        env=env,
        name="AXON_RECON_RECON_TOP_N_DENSITY_REQUESTED",
        default=None,
    )
    if getattr(args, "top_n_density_requested", None) is not None:
        top_n_density_requested = _parse_int_or_none(str(getattr(args, "top_n_density_requested")))

    write_top_density_grid = (
        env.env_bool("AXON_RECON_RECON_WRITE_TOP_DENSITY_GRID", default=True)
        if getattr(args, "write_top_density_grid", None) is None
        else bool(getattr(args, "write_top_density_grid"))
    )

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
        top_n_density_requested=top_n_density_requested,
        write_top_density_grid=write_top_density_grid,
        axon_velocity_repo_root=axon_velocity_repo_root,
    )

    return ReconstructionDebugConfig(inputs=inputs)
