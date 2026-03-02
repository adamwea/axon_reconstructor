from __future__ import annotations

import argparse


STAGE_CHOICES = ("preprocess", "spikesort", "waveforms", "templates", "reconstruct", "analysis")


def add_stage_selector_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("stage", choices=STAGE_CHOICES)


def add_stage_common_required_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--h5-path",
        required=False,
        default=None,
        help="Path to raw .h5 file (or set AXON_RECON_H5_PATH via --env-file/env).",
    )
    parser.add_argument(
        "--stream-id",
        required=False,
        default=None,
        help="Well/stream id (e.g. well003) (or set AXON_RECON_STREAM_ID via --env-file/env).",
    )
    parser.add_argument(
        "--mea-output-root",
        required=False,
        default=None,
        help="MEA output root for per-well stage outputs (or set AXON_RECON_MEA_OUTPUT_ROOT via --env-file/env).",
    )


def add_stage_spikesort_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mea-analysis-repo-root", default=None, help="Required for spikesort stage")
    parser.add_argument("--sorter", default="kilosort4")
    parser.add_argument("--docker-image", default=None)
    parser.add_argument("--chunk-duration", default=None)


def add_stage_execution_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--n-jobs", type=int, default=None, help="Override worker count (env fallback: AXON_RECON_N_JOBS).")
    parser.add_argument(
        "--force-restart",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override restart behavior (env fallback: AXON_RECON_FORCE_RESTART).",
    )
    parser.add_argument(
        "--debug",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable debug logging (env fallback: AXON_RECON_DEBUG).",
    )


def add_stage_debug_controls(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--break-before-run",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Drop into debugger before running preprocess/spikesort stage logic (env: AXON_RECON_BREAK_BEFORE_RUN).",
    )
    parser.add_argument(
        "--debug-max-units",
        type=int,
        default=None,
        help="Limit waveforms stage to first N units (env: AXON_RECON_WF_DEBUG_MAX_UNITS).",
    )
    parser.add_argument(
        "--debug-max-segments",
        type=int,
        default=None,
        help="Limit waveforms stage to first N segments when per-segment extraction is enabled (env: AXON_RECON_WF_DEBUG_MAX_SEGMENTS).",
    )


def add_stage_analysis_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--unit-limit",
        default=None,
        help="Limit analysis to first N units; accepts none/null/all for no limit (env: AXON_RECON_UNIT_LIMIT).",
    )
    parser.add_argument(
        "--unit-ids",
        nargs="*",
        default=None,
        help="Run analysis for explicit unit ids (env: AXON_RECON_UNIT_IDS as comma-separated list).",
    )
    parser.add_argument(
        "--prefer-curated-waveforms-panels",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Prefer curated waveforms panels in analysis outputs (env: AXON_RECON_ANALYSIS_PREFER_CURATED_WAVEFORMS_PANELS).",
    )
    parser.add_argument(
        "--botm-enable",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable BOTM validation metrics (env: AXON_RECON_ANALYSIS_BOTM_ENABLE).",
    )
    parser.add_argument("--botm-n-events", type=int, default=None, help="BOTM event count (env: AXON_RECON_ANALYSIS_BOTM_N_SPIKE).")
    parser.add_argument(
        "--botm-n-noise-windows",
        type=int,
        default=None,
        help="BOTM noise window count (env: AXON_RECON_ANALYSIS_BOTM_N_NOISE).",
    )
    parser.add_argument("--botm-seed", type=int, default=None, help="BOTM RNG seed (env: AXON_RECON_ANALYSIS_BOTM_SEED).")
    parser.add_argument(
        "--botm-prior-signal",
        type=float,
        default=None,
        help="BOTM prior signal probability (env: AXON_RECON_ANALYSIS_BOTM_CHANNEL_MATCH_PRIOR_SIGNAL).",
    )
    parser.add_argument(
        "--botm-match-fraction-threshold",
        type=float,
        default=None,
        help="BOTM match fraction threshold (env: AXON_RECON_ANALYSIS_BOTM_CHANNEL_MATCH_FRACTION_THRESHOLD).",
    )
    parser.add_argument("--botm-sorter", type=str, default=None, help="BOTM sorter id (env: AXON_RECON_ANALYSIS_BOTM_SORTER).")


def add_stage_kwargs_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--stage-kwargs", default=None, help="JSON object of stage-specific keyword args")
    parser.add_argument("--stage-kwargs-file", default=None, help="Path to JSON file with stage-specific keyword args")
