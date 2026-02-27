#!/usr/bin/env python3
"""Project-local entrypoint for stepping through the new analysis stage.

Goal
----
Generate per-unit summary grids that stitch together:
  - reconstruction plots (png)
  - templates plots (png)
  - waveforms panel (svg; best-effort rasterization)

Consumes (expected)
------------------
Under <well_out_dir>/
  - reconstruction_outputs/by_unit/unit_<id>/*.png
    - templates_outputs/footprints/3D/unit_<id>.png
  - templates_outputs/propagation_plots/unit_<id>.png
  - waveforms_outputs/panels/{curated,uncurated}/unit_<id>.svg

Produces
--------
Under <well_out_dir>/analysis_outputs/by_unit/unit_<id>/
  - unit_summary_grid.png
  - unit_summary_grid.pdf
and a run-level summary:
  - <well_out_dir>/analysis_outputs/analysis_summary.json

Controls
--------
- AXON_RECON_FORCE_RESTART=1   overwrite existing analysis outputs
- AXON_RECON_UNIT_LIMIT=5      limit to first N units ("none" => all)
- AXON_RECON_UNIT_IDS=1,2,26   run only selected units (comma-separated)

BOTM validation (optional)
-------------------------
Configured via debug.env keys under the "--- Analysis ---" section.
CLI flags can override env values.

Examples
--------
  python debug_analysis_step.py
  AXON_RECON_UNIT_IDS=26 python debug_analysis_step.py
  AXON_RECON_UNIT_LIMIT=10 python debug_analysis_step.py
  AXON_RECON_FORCE_RESTART=1 AXON_RECON_UNIT_IDS=26 python debug_analysis_step.py
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

DEBUG = False

PREFER_CURATED_WAVEFORMS_PANELS = True

# Resume/overwrite controls
FORCE_RESTART = False

# Keep this low while iterating; set to None to use all units.
UNIT_LIMIT: int | None = None

# Optional allowlist.
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
    p = argparse.ArgumentParser(description="Debug analysis stage")
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

    p.add_argument("--h5-path", type=Path, default=None)
    p.add_argument("--stream-id", type=str, default=None)
    p.add_argument("--mea-output-root", type=Path, default=None)

    p.add_argument(
        "--force-restart",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Overwrite existing analysis outputs",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Alias for --force-restart (kept for backwards compatibility)",
    )

    p.add_argument("--unit-limit", type=str, default=None, help="Int or 'none'")
    p.add_argument("--unit-ids", nargs="*", default=None, help="Optional unit ids (space-separated).")

    p.add_argument(
        "--prefer-curated-waveforms-panels",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Prefer curated waveforms panels if present",
    )

    # --- BOTM validation (optional; raw-snippet validator) ---
    p.add_argument(
        "--botm-enable",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable BOTM validation metrics (writes under analysis_outputs/botm_validation/)",
    )
    p.add_argument("--botm-n-events", type=int, default=None, help="Number of unit spike events to score")
    p.add_argument("--botm-n-noise-windows", type=int, default=None, help="Number of spike-free noise windows")
    p.add_argument("--botm-seed", type=int, default=None, help="RNG seed for subsampling/events/noise")
    p.add_argument(
        "--botm-prior-signal",
        type=float,
        default=None,
        help="Prior P(signal) at a candidate event time (default: 0.5)",
    )
    p.add_argument(
        "--botm-match-fraction-threshold",
        type=float,
        default=None,
        help="Good-channel cutoff: match fraction must be > this value (default: 0.70)",
    )
    p.add_argument(
        "--botm-sorter",
        type=str,
        default=None,
        help="Sorter name used to load concat sorting output (default: kilosort4)",
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
    logger = logging.getLogger("projects.debug_analysis_step")
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

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
        "AXON_RECON_UNIT_LIMIT",
        default=debug_env.env_int("AXON_RECON_UNIT_LIMIT", default=UNIT_LIMIT),
    )
    if args.unit_limit is not None:
        unit_limit = _parse_int_or_none(args.unit_limit)

    unit_ids = debug_env.env_int_list("AXON_RECON_UNIT_IDS")
    if args.unit_ids:
        unit_ids = [int(x) for x in args.unit_ids]
    elif UNIT_IDS is not None:
        unit_ids = list(UNIT_IDS)

    prefer_curated = (
        debug_env.env_bool(
            "AXON_RECON_ANALYSIS_PREFER_CURATED_WAVEFORMS_PANELS",
            default=bool(PREFER_CURATED_WAVEFORMS_PANELS),
        )
        if args.prefer_curated_waveforms_panels is None
        else bool(args.prefer_curated_waveforms_panels)
    )

    # Precedence: CLI > env > defaults.
    if args.botm_enable is not None:
        compute_botm_validation = bool(args.botm_enable)
    else:
        compute_botm_validation = debug_env.env_bool("AXON_RECON_ANALYSIS_BOTM_ENABLE", default=False)

    botm_n_events = (
        int(args.botm_n_events)
        if args.botm_n_events is not None
        else int(debug_env.env_int("AXON_RECON_ANALYSIS_BOTM_N_SPIKE", default=200) or 200)
    )
    botm_n_noise_windows = (
        int(args.botm_n_noise_windows)
        if args.botm_n_noise_windows is not None
        else int(debug_env.env_int("AXON_RECON_ANALYSIS_BOTM_N_NOISE", default=2000) or 2000)
    )
    botm_seed = (
        int(args.botm_seed)
        if args.botm_seed is not None
        else int(debug_env.env_int("AXON_RECON_ANALYSIS_BOTM_SEED", default=0) or 0)
    )
    botm_prior_signal = (
        float(args.botm_prior_signal)
        if args.botm_prior_signal is not None
        else float(debug_env.env_float("AXON_RECON_ANALYSIS_BOTM_CHANNEL_MATCH_PRIOR_SIGNAL", default=0.5) or 0.5)
    )
    botm_match_fraction_threshold = (
        float(args.botm_match_fraction_threshold)
        if args.botm_match_fraction_threshold is not None
        else float(debug_env.env_float("AXON_RECON_ANALYSIS_BOTM_CHANNEL_MATCH_FRACTION_THRESHOLD", default=0.70) or 0.70)
    )
    botm_sorter = (
        str(args.botm_sorter)
        if args.botm_sorter is not None
        else str(debug_env.env_str("AXON_RECON_ANALYSIS_BOTM_SORTER", default="kilosort4") or "kilosort4")
    )

    # Dependency sanity checks.
    # The analysis montage can render without matplotlib (Pillow fallback), but it does need Pillow.
    try:
        __import__("PIL")
    except Exception:
        logger.error(
            "Missing deps for analysis montage: PIL/pillow. "
            "Install with pip/conda (e.g. `pip install pillow`)."
        )
        raise SystemExit(2)

    try:
        __import__("matplotlib")
    except Exception:
        logger.info("matplotlib not available; analysis will use Pillow montage fallback")

    # Note: the analysis grid currently omits the waveforms panel, so SVG rasterization
    # (cairosvg) is not required.

    from axon_reconstructor.pipeline.analysis import AnalysisInputs, analyze_units

    inputs = AnalysisInputs(
        h5_path=h5_path,
        stream_id=stream_id,
        mea_output_root=mea_output_root,
        unit_ids=unit_ids,
        unit_limit=unit_limit,
        prefer_curated_waveforms_panels=prefer_curated,
        compute_botm_validation=compute_botm_validation,
        botm_n_events=botm_n_events,
        botm_n_noise_windows=botm_n_noise_windows,
        botm_seed=botm_seed,
        botm_prior_signal=float(botm_prior_signal),
        botm_match_fraction_threshold=float(botm_match_fraction_threshold),
        botm_sorter=str(botm_sorter),
        force_restart=force_restart,
    )

    logger.info("Debug analysis inputs: %s", inputs)
    out = analyze_units(inputs=inputs, logger_name_prefix="projects")

    logger.info("Analysis written under: %s", out.analysis_out_dir)
    logger.info("By-unit outputs: %s", out.by_unit_dir)
    logger.info("Summary JSON: %s", out.summary_json)


if __name__ == "__main__":
    main()
