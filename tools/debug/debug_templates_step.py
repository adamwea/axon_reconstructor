#!/usr/bin/env python3
"""Project-local entrypoint for stepping through template extraction + merging.

This expects that waveforms have already been extracted for the target well:
  <well>/waveforms_outputs/concat_waveforms/
  <well>/waveforms_outputs/segment_waveforms/segXX_*/

Outputs:
  <well>/templates_outputs/
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import debug_env


# Dataset configuration comes from debug.env (or CLI overrides).
H5_PATH: Path | None = None
STREAM_ID: str | None = None
MEA_OUTPUT_ROOT: Path | None = None

N_JOBS = 8

# Logging
DEBUG = False

# If True, ignore existing template outputs/checkpoint.
FORCE_RESTART = True
REPLOT_FROM_DISK = False

# Which analyzers to include
INCLUDE_CONCAT = True
INCLUDE_SEGMENTS = True

# Keep this low while iterating; set to None to use all units.
UNIT_LIMIT = None

# Plotting
PLOT_TEMPLATES_GRID_PDF = True
PLOT_MULTI_SOURCE_TEMPLATES_PDF = True

# For non-merged sources, plot only top-N channels by PTP.
# Use 0 to plot all channels.
# Note: merged_contributing always plots *all* contributing channels.
TOP_CHANNELS_PER_TEMPLATE = 0

# Optional template (time-axis) upsampling after spikesorting.
# Factor of 1 disables. Method: "sinc" (scipy resample_poly) or "linear".
TEMPLATE_TIME_UPSAMPLE_FACTOR = 1
TEMPLATE_TIME_UPSAMPLE_METHOD = "sinc"

# Define specific Unit ids if desired (overrides unit_limit). Example: UNIT_IDS = [68, 123] or UNIT_IDS = ["unit_68", "unit_123"].
#UNIT_IDS: list[int] | None = [68]
UNIT_IDS: list[int] | None = None

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


def _parse_int_or_none(raw: str) -> int | None:
    v = str(raw).strip().lower()
    if v in {"none", "null", "all"}:
        return None
    return int(v)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Debug templates step / replot from persisted artifacts")

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
        help="Enable/disable debug logging (very verbose, including matplotlib internals).",
    )

    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--replot-from-disk",
        action="store_true",
        help="Regenerate plots from <well>/templates_outputs/templates/{merged,full} only (no analyzers/waveforms)",
    )
    mode.add_argument(
        "--run-templates",
        action="store_true",
        help="Run templates extraction+merging (default if no mode is specified)",
    )

    p.add_argument("--unit-ids", nargs="*", default=None, help="Optional unit ids (space-separated).")
    p.add_argument("--force", action="store_true", help="Overwrite existing outputs")

    p.add_argument("--h5-path", type=Path, default=None)
    p.add_argument("--stream-id", type=str, default=None)
    p.add_argument("--mea-output-root", type=Path, default=None)

    p.add_argument("--n-jobs", type=int, default=None)
    p.add_argument(
        "--force-restart",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Ignore cached outputs and re-run templates",
    )
    p.add_argument("--unit-limit", type=str, default=None, help="Int or 'none'")

    p.add_argument(
        "--include-concat",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Include concat analyzer templates",
    )
    p.add_argument(
        "--include-segments",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Include segment analyzer templates",
    )

    p.add_argument(
        "--plot-templates-grid-pdf",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Write templates grid PDF",
    )
    p.add_argument(
        "--plot-multi-source-templates-pdf",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Write multi-source per-unit templates PDFs",
    )

    p.add_argument(
        "--require-curated-units",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="If true (default), templates runs only on curated units derived from spikesorting metrics. Disable to run on all units.",
    )

    p.add_argument("--top-channels-per-template", type=int, default=None)

    p.add_argument(
        "--template-time-upsample-factor",
        type=int,
        default=None,
        help="Upsample persisted templates along time axis by this integer factor (default: 1 = disabled)",
    )
    p.add_argument(
        "--template-time-upsample-method",
        type=str,
        default=None,
        help="Upsampling method for templates: 'sinc' (preferred) or 'linear'",
    )

    p.add_argument(
        "--rebuild-full-channels-templates",
        action="store_true",
        help="In replot-from-disk mode, rebuild templates_outputs/templates/full from merged templates using Maxwell full-chip electrode ids (fixes sparse full templates for topo plots).",
    )

    # Replot controls
    p.add_argument("--propagation-top-channels", type=int, default=25)
    p.add_argument("--propagation-channels-per-panel", type=int, default=25)
    p.add_argument("--propagation-channel-overlap", type=int, default=5)

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
    logger = logging.getLogger("projects.debug_templates_step")

    # Matplotlib can be extremely noisy at DEBUG.
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    h5_path = Path(args.h5_path) if args.h5_path is not None else debug_env.env_required_path("AXON_RECON_H5_PATH")
    stream_id = str(args.stream_id) if args.stream_id is not None else debug_env.env_required_str("AXON_RECON_STREAM_ID")
    mea_output_root = (
        Path(args.mea_output_root)
        if args.mea_output_root is not None
        else debug_env.env_required_path("AXON_RECON_MEA_OUTPUT_ROOT")
    )
    n_jobs = int(args.n_jobs) if args.n_jobs is not None else int(debug_env.env_int("AXON_RECON_N_JOBS", default=int(N_JOBS)) or int(N_JOBS))

    include_concat = debug_env.env_bool("AXON_RECON_INCLUDE_CONCAT", default=bool(INCLUDE_CONCAT)) if args.include_concat is None else bool(args.include_concat)
    include_segments = debug_env.env_bool("AXON_RECON_INCLUDE_SEGMENTS", default=bool(INCLUDE_SEGMENTS)) if args.include_segments is None else bool(args.include_segments)

    plot_templates_grid_pdf = (
        debug_env.env_bool("AXON_RECON_TEMPLATES_PLOT_GRID_PDF", default=bool(PLOT_TEMPLATES_GRID_PDF))
        if args.plot_templates_grid_pdf is None
        else bool(args.plot_templates_grid_pdf)
    )
    plot_multi_source_templates_pdf = (
        debug_env.env_bool(
            "AXON_RECON_TEMPLATES_PLOT_MULTI_SOURCE_PDF",
            default=bool(PLOT_MULTI_SOURCE_TEMPLATES_PDF),
        )
        if args.plot_multi_source_templates_pdf is None
        else bool(args.plot_multi_source_templates_pdf)
    )

    require_curated_units = (
        debug_env.env_bool("AXON_RECON_TEMPLATES_REQUIRE_CURATED_UNITS", default=True)
        if args.require_curated_units is None
        else bool(args.require_curated_units)
    )

    top_channels_per_template = (
        int(debug_env.env_int("AXON_RECON_TEMPLATES_TOP_CHANNELS_PER_TEMPLATE", default=int(TOP_CHANNELS_PER_TEMPLATE)) or int(TOP_CHANNELS_PER_TEMPLATE))
        if args.top_channels_per_template is None
        else int(args.top_channels_per_template)
    )

    template_time_upsample_factor = (
        int(debug_env.env_int("AXON_RECON_TEMPLATES_TIME_UPSAMPLE_FACTOR", default=int(TEMPLATE_TIME_UPSAMPLE_FACTOR)) or int(TEMPLATE_TIME_UPSAMPLE_FACTOR))
        if args.template_time_upsample_factor is None
        else int(args.template_time_upsample_factor)
    )
    template_time_upsample_method = (
        str(os.environ.get("AXON_RECON_TEMPLATES_TIME_UPSAMPLE_METHOD", TEMPLATE_TIME_UPSAMPLE_METHOD)).strip()
        if args.template_time_upsample_method is None
        else str(args.template_time_upsample_method).strip()
    )

    # Import the pipeline directly so you can step through the actual code.
    from axon_reconstructor.pipeline.templates import TemplateExtractInputs, extract_and_merge_templates, replot_templates_outputs_from_disk

    # Resolve well output directory from (h5_path, output_root, stream_id).
    from axon_reconstructor.pipeline.pipeline_driver import _compute_mea_analysis_output_dir

    well_out_dir = _compute_mea_analysis_output_dir(
        output_root=mea_output_root,
        data_file=h5_path,
        well=stream_id,
    )
    templates_out_dir = well_out_dir / "templates_outputs"

    # Precedence: CLI > env > ALL-CAPS constants.
    if args.force_restart is not None:
        force_restart = bool(args.force_restart)
    elif bool(args.force):
        force_restart = True
    else:
        force_restart = debug_env.env_bool("AXON_RECON_FORCE_RESTART", default=bool(FORCE_RESTART))

    unit_limit = _env_int_or_none(
        "AXON_RECON_TEMPLATES_UNIT_LIMIT",
        default=_env_int_or_none("AXON_RECON_UNIT_LIMIT", default=UNIT_LIMIT),
    )
    if args.unit_limit is not None:
        unit_limit = _parse_int_or_none(args.unit_limit)

    unit_ids = None
    if args.unit_ids or UNIT_IDS:
        raw_unit_ids = list(args.unit_ids) if args.unit_ids else (list(UNIT_IDS) if UNIT_IDS is not None else [])
        logger.info("Using unit ids: %s (from args.unit_ids=%s, UNIT_IDS=%s)", raw_unit_ids, args.unit_ids, UNIT_IDS)
        # Best-effort: use int ids when possible.
        parsed: list[object] = []
        for x in raw_unit_ids:
            try:
                parsed.append(int(x))
            except Exception:
                parsed.append(x)
        unit_ids = parsed

    # Mode selection precedence:
    # 1) CLI explicitly selects a mode
    # 2) Otherwise, use env default (fallback: hardcoded REPLOT_FROM_DISK)
    if bool(args.replot_from_disk):
        do_replot = True
    elif bool(args.run_templates):
        do_replot = False
    else:
        do_replot = debug_env.env_bool("AXON_RECON_TEMPLATES_REPLOT_FROM_DISK", default=bool(REPLOT_FROM_DISK))

    # Mode: replot from disk.
    if do_replot:
        logger.info("Replotting templates outputs from disk: %s", templates_out_dir)
        replot_templates_outputs_from_disk(
            well_out_dir=well_out_dir,
            unit_ids=unit_ids,
            force=bool(args.force),
            rebuild_full_channels_templates=bool(args.rebuild_full_channels_templates),
            propagation_top_channels=int(args.propagation_top_channels),
            propagation_channels_per_panel=int(args.propagation_channels_per_panel),
            propagation_channel_overlap=int(args.propagation_channel_overlap),
        )
        logger.info("Replot complete.")
        return

    # Default: run templates.
    if force_restart and templates_out_dir.exists():
        import shutil

        logger.info("Removing existing templates_outputs: %s", templates_out_dir)
        shutil.rmtree(templates_out_dir)
        logger.info("Removed existing templates_outputs.")

    inputs = TemplateExtractInputs(
        h5_path=h5_path,
        stream_id=stream_id,
        mea_output_root=mea_output_root,
        include_concat=include_concat,
        include_segments=include_segments,
        unit_ids=unit_ids,
        require_curated_units=bool(require_curated_units),
        plot_templates_grid_pdf=plot_templates_grid_pdf,
        plot_multi_source_templates_pdf=plot_multi_source_templates_pdf,
        top_channels_per_template=top_channels_per_template,
        template_time_upsample_factor=int(template_time_upsample_factor),
        template_time_upsample_method=str(template_time_upsample_method),
        propagation_show_electrode_ids=True,
        propagation_trace_gain=5.0,
        n_jobs=n_jobs,
        unit_limit=unit_limit,
        force_restart=force_restart,
    )

    logger.info("Debug templates inputs: %s", inputs)
    out = extract_and_merge_templates(inputs=inputs)
    logger.info("Templates written under: %s", out.templates_out_dir)
    logger.info("Extracted templates: %s", out.extracted_templates_dir)
    logger.info("Summary JSON: %s", out.summary_json)
    if out.templates_grid_pdf is not None:
        logger.info("Templates grid PDF: %s", out.templates_grid_pdf)
    if out.multi_source_templates_dir is not None:
        logger.info("Per-unit templates PDFs: %s", out.multi_source_templates_dir)


if __name__ == "__main__":
    main()
