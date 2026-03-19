#!/usr/bin/env python3
"""Direct UnitMatch runner debug harness.

Loads recording/sorting from stage outputs and calls
MEA_Analysis.IPNAnalysis.UnitMatch.runner.run_unitmatch_merge_with_recursion
without invoking the full axon_reconstructor stage pipeline.

Intended for step-through debugging of UnitMatch dry-run behavior.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any


def _load_env_file(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip()
    return env


def _resolve_arg(cli: str | None, env: dict[str, str], key: str, default: str | None = None) -> str | None:
    if cli is not None and str(cli).strip() != "":
        return str(cli)
    if key in env and str(env[key]).strip() != "":
        return str(env[key])
    return default


def _add_repo_import_paths(axon_repo_root: Path, mea_repo_root: Path) -> None:
    axon_src = axon_repo_root / "src"
    if str(axon_src) not in sys.path:
        sys.path.insert(0, str(axon_src))

    mea_parent = mea_repo_root.parent
    if str(mea_parent) not in sys.path:
        sys.path.insert(0, str(mea_parent))



def _compute_well_output_dir(
    *,
    axon_repo_root: Path,
    output_root: Path,
    h5_path: Path,
    stream_id: str,
) -> Path:
    from axon_reconstructor.pipeline.output_paths import compute_mea_analysis_output_dir

    return compute_mea_analysis_output_dir(
        output_root=output_root,
        data_file=h5_path,
        well=stream_id,
    )



def _load_recording(recording_dir: Path, logger: logging.Logger) -> Any:
    import spikeinterface.full as si

    logger.info("Loading recording from %s", recording_dir)
    try:
        return si.load(recording_dir)
    except Exception:
        logger.warning("si.load failed; retrying with si.load_extractor")
        return si.load_extractor(recording_dir)



def _sorting_candidates(spikesort_output_dir: Path) -> list[Path]:
    return [
        spikesort_output_dir / "sorter_output",
        spikesort_output_dir / "sorter_output" / "sorter_output",
        spikesort_output_dir / "sorter_output" / "in_container_sorting",
    ]



def _load_sorting(spikesort_output_dir: Path, logger: logging.Logger) -> tuple[Any, Path]:
    import spikeinterface.full as si

    candidates = _sorting_candidates(spikesort_output_dir)
    last_error: Exception | None = None
    for cand in candidates:
        if not cand.exists():
            continue
        logger.info("Trying sorting load from %s", cand)
        try:
            sorting = si.load_extractor(cand)
            return sorting, cand
        except Exception as exc:
            last_error = exc
            logger.warning("Failed loading sorting from %s: %s", cand, exc)

    msg = "No valid sorting extractor found under spikesort outputs"
    if last_error is not None:
        raise RuntimeError(f"{msg}; last_error={last_error}") from last_error
    raise RuntimeError(msg)



def _import_runner():
    from MEA_Analysis.IPNAnalysis.UnitMatch.runner import UnitMatchConfig, run_unitmatch_merge_with_recursion
    from MEA_Analysis.IPNAnalysis.UnitMatch.reporting import (
        UnitMatchReportConfig,
        generate_unitmatch_static_report_pack,
    )

    return UnitMatchConfig, run_unitmatch_merge_with_recursion, UnitMatchReportConfig, generate_unitmatch_static_report_pack


def _clone_sorting_for_debug(sorting: Any, logger: logging.Logger) -> Any:
    # Keep the loaded sorting object untouched when testing merge application.
    if hasattr(sorting, "clone"):
        try:
            return sorting.clone()
        except Exception as exc:
            logger.warning("sorting.clone() failed, trying other copy methods: %s", exc)
    if hasattr(sorting, "copy"):
        try:
            return sorting.copy()
        except Exception as exc:
            logger.warning("sorting.copy() failed, trying deepcopy: %s", exc)
    try:
        return copy.deepcopy(sorting)
    except Exception as exc:
        logger.warning("deepcopy failed; using original sorting object: %s", exc)
        return sorting



def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Direct UnitMatch runner debug harness")
    p.add_argument("--env-file", type=str, default=str(Path(__file__).with_name("debug.env")))

    p.add_argument("--axon-repo-root", type=str, default=str(Path(__file__).resolve().parents[2]))
    p.add_argument("--mea-analysis-repo-root", type=str, default=None)

    p.add_argument("--well-output-dir", type=str, default=None,
                   help="Direct per-well output dir containing stg1_preprocess_outputs and stg2_spikesorting_outputs")

    p.add_argument("--output-root", type=str, default=None,
                   help="Used to infer well-output-dir when not provided")
    p.add_argument("--h5-path", type=str, default=None,
                   help="Used to infer well-output-dir when not provided")
    p.add_argument("--stream-id", type=str, default=None,
                   help="Used to infer well-output-dir when not provided")

    p.add_argument("--recording-dir", type=str, default=None,
                   help="Override recording dir (default: <well-output-dir>/stg1_preprocess_outputs/preprocessed_recording)")
    p.add_argument("--spikesort-output-dir", type=str, default=None,
                   help="Override spikesort output dir (default: <well-output-dir>/stg2_spikesorting_outputs)")

    p.add_argument("--unitmatch-output-subdir", type=str, default="unitmatch_outputs")
    p.add_argument("--unitmatch-throughput-subdir", type=str, default="unitmatch_throughput")
    p.add_argument("--max-candidate-pairs", type=int, default=20000,
                   help="Ignored in this debug harness (hardcoded to -1/unlimited)")
    p.add_argument("--oversplit-min-probability", type=float, default=0.975)
    p.add_argument("--oversplit-max-suggestions", type=int, default=2000,
                   help="Ignored in this debug harness (hardcoded to -1/unlimited)")
    p.add_argument("--apply-merges", action="store_true",
                   help="Ignored in this debug harness (hardcoded enabled)")
    p.add_argument("--unitmatch-report-subdir", type=str, default="unitmatch_reports",
                   help="Static report output subdir under spikesort output dir")
    p.add_argument("--unitmatch-report-max-heatmap-units", type=int, default=200,
                   help="Maximum units shown in report heatmap")
    p.add_argument("--no-unitmatch-reporting", action="store_true",
                   help="Disable static report generation after UnitMatch run")

    p.add_argument("--no-scored-dry-run", action="store_true")
    p.add_argument("--fail-open", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--pdb-before-run", action="store_true")
    return p



def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    env = _load_env_file(Path(args.env_file))

    axon_repo_root = Path(_resolve_arg(args.axon_repo_root, env, "AXON_RECON_AXON_REPO_ROOT", args.axon_repo_root)).expanduser().resolve()
    mea_repo_root_s = _resolve_arg(args.mea_analysis_repo_root, env, "AXON_RECON_MEA_ANALYSIS_REPO_ROOT", "/home/adamm/dev/pkgs/MEA_Analysis")
    if mea_repo_root_s is None:
        raise SystemExit("MEA_Analysis repo root is required")
    mea_repo_root = Path(mea_repo_root_s).expanduser().resolve()

    _add_repo_import_paths(axon_repo_root, mea_repo_root)

    logger = logging.getLogger("debug_unitmatch_runner_direct")
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

    well_output_dir_s = _resolve_arg(args.well_output_dir, env, "AXON_RECON_WELL_OUTPUT_DIR", None)
    if well_output_dir_s is None:
        output_root_s = _resolve_arg(args.output_root, env, "AXON_RECON_MEA_OUTPUT_ROOT", None)
        h5_path_s = _resolve_arg(args.h5_path, env, "AXON_RECON_H5_PATH", None)
        stream_id = _resolve_arg(args.stream_id, env, "AXON_RECON_STREAM_ID", None)
        if output_root_s is None or h5_path_s is None or stream_id is None:
            raise SystemExit(
                "Need either --well-output-dir OR (--output-root, --h5-path, --stream-id / env equivalents)"
            )
        well_output_dir = _compute_well_output_dir(
            axon_repo_root=axon_repo_root,
            output_root=Path(output_root_s).expanduser().resolve(),
            h5_path=Path(h5_path_s).expanduser().resolve(),
            stream_id=stream_id,
        )
    else:
        well_output_dir = Path(well_output_dir_s).expanduser().resolve()

    recording_dir = Path(
        _resolve_arg(
            args.recording_dir,
            env,
            "AXON_RECON_RECORDING_DIR",
            str(well_output_dir / "stg1_preprocess_outputs" / "preprocessed_recording"),
        )
    ).expanduser().resolve()

    spikesort_output_dir = Path(
        _resolve_arg(
            args.spikesort_output_dir,
            env,
            "AXON_RECON_SPIKESORT_OUTPUT_DIR",
            str(well_output_dir / "stg2_spikesorting_outputs"),
        )
    ).expanduser().resolve()

    logger.info("well_output_dir=%s", well_output_dir)
    logger.info("recording_dir=%s", recording_dir)
    logger.info("spikesort_output_dir=%s", spikesort_output_dir)

    if not recording_dir.exists():
        raise SystemExit(f"Recording dir not found: {recording_dir}")
    if not spikesort_output_dir.exists():
        raise SystemExit(f"Spikesort output dir not found: {spikesort_output_dir}")

    recording = _load_recording(recording_dir, logger)
    sorting, sorting_loaded_from = _load_sorting(spikesort_output_dir, logger)
    logger.info("sorting_loaded_from=%s", sorting_loaded_from)

    (
        UnitMatchConfig,
        run_unitmatch_merge_with_recursion,
        UnitMatchReportConfig,
        generate_unitmatch_static_report_pack,
    ) = _import_runner()

    debug_sorting = _clone_sorting_for_debug(sorting, logger)
    logger.info("Using copied sorting object for debug run when available")

    config = UnitMatchConfig(
        enabled=True,
        dry_run=False,
        scored_dry_run=not bool(args.no_scored_dry_run),
        fail_open=bool(args.fail_open),
        max_candidate_pairs=-1,
        output_subdir_name=str(args.unitmatch_output_subdir),
        throughput_subdir_name=str(args.unitmatch_throughput_subdir),
        oversplit_min_probability=float(args.oversplit_min_probability),
        oversplit_max_suggestions=-1,
        apply_merges=True,
        recursive=True,
        max_iterations=0,
        uncapped_iterations=True,
        keep_all_iterations=True,
    )

    logger.info(
        "Hardcoded debug policy: apply_merges=True dry_run=False recursive=True uncapped_iterations=True max_candidate_pairs=-1 oversplit_max_suggestions=-1"
    )

    logger.info("unitmatch_config=%s", json.dumps(config.__dict__, indent=2))

    if bool(args.pdb_before_run):
        breakpoint()

    merged_sorting, summary = run_unitmatch_merge_with_recursion(
        sorting=debug_sorting,
        recording=recording,
        output_dir=spikesort_output_dir,
        logger=logger,
        config=config,
    )

    if not bool(args.no_unitmatch_reporting):
        try:
            report = generate_unitmatch_static_report_pack(
                output_dir=spikesort_output_dir,
                logger=logger,
                config=UnitMatchReportConfig(
                    throughput_subdir_name=str(args.unitmatch_throughput_subdir),
                    output_subdir_name=str(args.unitmatch_output_subdir),
                    report_subdir_name=str(args.unitmatch_report_subdir),
                    max_heatmap_units=int(args.unitmatch_report_max_heatmap_units),
                ),
            )
            logger.info("unitmatch_report_root=%s", report.get("report_root"))
        except Exception as exc:
            logger.warning("UnitMatch reporting failed-open in debug harness: %s", exc)

    logger.info("merged_sorting_type=%s", type(merged_sorting).__name__)

    print("\n=== UnitMatch summary ===")
    print(json.dumps(summary, indent=2))

    summary_path = Path(spikesort_output_dir) / str(config.output_subdir_name) / "unitmatch_summary.json"
    print(f"\nsummary_path={summary_path}")
    print(f"summary_exists={summary_path.exists()}")
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
