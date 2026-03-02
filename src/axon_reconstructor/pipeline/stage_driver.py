from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
import logging
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .analysis import AnalysisInputs, analyze_units
from .output_paths import compute_mea_analysis_output_dir
from .raw_preprocessing import run_preprocess_stage
from .reconstruction import ReconstructionInputs, reconstruct_from_templates
from .scope_config import ScopeConfig
from .spikesorting import SpikeSortingInputs, run_spikesorting_stage
from .templates import TemplateExtractInputs, extract_and_merge_templates
from .waveforms import WaveformExtractInputs, extract_waveforms


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


@dataclass(frozen=True)
class StageExecutionContext:
    h5_path: Path
    stream_id: str
    mea_output_root: Path
    force_restart: bool
    n_jobs: int = 8
    sorter: str = "kilosort4"
    docker_image: Optional[str] = None
    chunk_duration: Optional[str] = None
    mea_analysis_repo_root: Optional[Path] = None
    verbose: bool = False


@dataclass(frozen=True)
class StageExecutionResult:
    stage: str
    artifacts: dict[str, Any]


def execute_stage(
    *,
    stage: str,
    context: StageExecutionContext,
    stage_kwargs: Optional[dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None,
) -> StageExecutionResult:
    stage = str(stage)
    kwargs = dict(stage_kwargs or {})

    if stage == "preprocess":
        preprocess_kwargs = {
            "h5_path": context.h5_path,
            "stream_id": context.stream_id,
            "mea_output_root": context.mea_output_root,
            "force_restart": bool(context.force_restart),
            "n_jobs": int(context.n_jobs),
            "overwrite_saved_recording": bool(context.force_restart),
        }
        preprocess_kwargs.update(kwargs)
        _, common_el = run_preprocess_stage(**preprocess_kwargs)
        return StageExecutionResult(stage=stage, artifacts={"n_common_electrodes": int(len(common_el))})

    if stage == "spikesort":
        if context.mea_analysis_repo_root is None:
            raise RuntimeError("spikesort stage requires mea_analysis_repo_root")

        spikesort_fields = {
            "h5_path": context.h5_path,
            "stream_id": context.stream_id,
            "mea_output_root": context.mea_output_root,
            "mea_analysis_repo_root": context.mea_analysis_repo_root,
            "sorter": context.sorter,
            "docker_image": context.docker_image,
            "force_restart": bool(context.force_restart),
            "verbose": bool(context.verbose),
            "n_jobs": int(context.n_jobs),
            "chunk_duration": context.chunk_duration,
        }
        spikesort_fields.update(kwargs)
        out = run_spikesorting_stage(
            inputs=SpikeSortingInputs(**spikesort_fields),
            logger=logger or logging.getLogger(f"axon_reconstructor.stage.{context.stream_id}.spikesort"),
        )
        return StageExecutionResult(
            stage=stage,
            artifacts={
                "sorter_output_dir": str(out.sorter_output_dir),
                "output_dir": str(out.output_dir),
            },
        )

    if stage == "waveforms":
        waveform_fields = {
            "h5_path": context.h5_path,
            "stream_id": context.stream_id,
            "mea_output_root": context.mea_output_root,
            "sorter": context.sorter,
            "n_jobs": int(context.n_jobs),
            "force_restart": bool(context.force_restart),
        }
        waveform_fields.update(kwargs)
        out = extract_waveforms(inputs=WaveformExtractInputs(**waveform_fields))
        return StageExecutionResult(stage=stage, artifacts={"waveforms_out_dir": str(out.waveforms_out_dir)})

    if stage == "templates":
        template_fields = {
            "h5_path": context.h5_path,
            "stream_id": context.stream_id,
            "mea_output_root": context.mea_output_root,
            "n_jobs": int(context.n_jobs),
            "force_restart": bool(context.force_restart),
        }
        template_fields.update(kwargs)
        out = extract_and_merge_templates(inputs=TemplateExtractInputs(**template_fields))
        return StageExecutionResult(stage=stage, artifacts={"templates_out_dir": str(out.templates_out_dir)})

    if stage == "reconstruct":
        recon_fields = {
            "h5_path": context.h5_path,
            "stream_id": context.stream_id,
            "mea_output_root": context.mea_output_root,
            "force_restart": bool(context.force_restart),
        }
        recon_fields.update(kwargs)
        out = reconstruct_from_templates(inputs=ReconstructionInputs(**recon_fields))
        return StageExecutionResult(
            stage=stage,
            artifacts={"reconstruction_out_dir": str(out.reconstruction_out_dir)},
        )

    if stage == "analysis":
        analysis_fields = {
            "h5_path": context.h5_path,
            "stream_id": context.stream_id,
            "mea_output_root": context.mea_output_root,
            "force_restart": bool(context.force_restart),
        }
        analysis_fields.update(kwargs)
        out = analyze_units(inputs=AnalysisInputs(**analysis_fields))
        return StageExecutionResult(stage=stage, artifacts={"analysis_out_dir": str(out.analysis_out_dir)})

    raise ValueError(f"Unsupported stage: {stage}")


@dataclass(frozen=True)
class ScopeTarget:
    dataset_id: str
    h5_path: Path
    stream_id: str
    stage_kwargs: dict[str, dict[str, Any]]


def _now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _scope_barrier_checkpoint_file(*, config: ScopeConfig) -> Path:
    return Path(config.mea_output_root) / "scope_run_barrier_checkpoint.json"


def _load_scope_barrier_checkpoint(*, checkpoint_file: Path, force_restart: bool) -> dict[str, Any]:
    if bool(force_restart) or (not checkpoint_file.exists()):
        return {"completed_stages": [], "stage_entries": {}}
    try:
        payload = json.loads(checkpoint_file.read_text(encoding="utf-8"))
        completed = payload.get("completed_stages")
        entries = payload.get("stage_entries")
        return {
            "completed_stages": list(completed) if isinstance(completed, list) else [],
            "stage_entries": dict(entries) if isinstance(entries, dict) else {},
        }
    except Exception:
        return {"completed_stages": [], "stage_entries": {}}


def _save_scope_barrier_checkpoint(*, checkpoint_file: Path, barrier_state: dict[str, Any]) -> None:
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "last_updated_utc": _now_utc(),
        "completed_stages": list(barrier_state.get("completed_stages", [])),
        "stage_entries": dict(barrier_state.get("stage_entries", {})),
    }
    checkpoint_file.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _merge_stage_kwargs(*, stage: str, config: ScopeConfig, target: ScopeTarget) -> dict[str, Any]:
    out: dict[str, Any] = {}
    out.update(config.stage_kwargs.get(stage, {}))
    out.update(target.stage_kwargs.get(stage, {}))
    return out


def _build_targets(config: ScopeConfig) -> list[ScopeTarget]:
    targets: list[ScopeTarget] = []
    for i, dataset in enumerate(config.datasets):
        if not dataset.enabled:
            continue
        dataset_id = dataset.dataset_id or f"dataset_{i:03d}"
        for well in dataset.wells:
            if not well.enabled:
                continue
            stage_kwargs: dict[str, dict[str, Any]] = {}
            for stage_name in set(dataset.stage_kwargs.keys()) | set(well.stage_kwargs.keys()):
                merged = dict(dataset.stage_kwargs.get(stage_name, {}))
                merged.update(well.stage_kwargs.get(stage_name, {}))
                stage_kwargs[stage_name] = merged

            targets.append(
                ScopeTarget(
                    dataset_id=dataset_id,
                    h5_path=dataset.h5_path,
                    stream_id=well.stream_id,
                    stage_kwargs=stage_kwargs,
                )
            )
    return targets


def _run_single_stage_target(*, stage: str, config: ScopeConfig, target: ScopeTarget) -> dict[str, Any]:
    stage_kwargs = _merge_stage_kwargs(stage=stage, config=config, target=target)
    started_at = _now_utc()

    context = StageExecutionContext(
        h5_path=target.h5_path,
        stream_id=target.stream_id,
        mea_output_root=config.mea_output_root,
        force_restart=bool(config.force_restart),
        n_jobs=int(config.n_jobs),
        sorter=config.sorter,
        docker_image=config.docker_image,
        chunk_duration=config.chunk_duration,
        mea_analysis_repo_root=config.mea_analysis_repo_root,
        verbose=True,
    )
    result = execute_stage(
        stage=stage,
        context=context,
        stage_kwargs=stage_kwargs,
        logger=logging.getLogger(f"axon_reconstructor.scope.{target.stream_id}.{stage}"),
    )
    return {
        "status": "ok",
        "stage": stage,
        "dataset_id": target.dataset_id,
        "h5_path": str(target.h5_path),
        "stream_id": target.stream_id,
        "started_at": started_at,
        "finished_at": _now_utc(),
        **dict(result.artifacts),
    }


def _expected_sorter_output_dir(*, config: ScopeConfig, target: ScopeTarget) -> Path:
    well_out_dir = compute_mea_analysis_output_dir(
        output_root=config.mea_output_root,
        data_file=target.h5_path,
        well=target.stream_id,
    )
    return well_out_dir / "spikesorting_outputs" / "sorter_output"


def _run_transition_stage(
    *,
    stage: str,
    config: ScopeConfig,
    targets: list[ScopeTarget],
    summary: dict[str, Any],
) -> dict[str, Any]:
    if stage == "unit_match":
        stage_entries: list[dict[str, Any]] = []
        for target in targets:
            sorter_output_dir = _expected_sorter_output_dir(config=config, target=target)
            if sorter_output_dir.exists():
                stage_entries.append(
                    {
                        "status": "ok",
                        "stage": stage,
                        "dataset_id": target.dataset_id,
                        "h5_path": str(target.h5_path),
                        "stream_id": target.stream_id,
                        "started_at": _now_utc(),
                        "finished_at": _now_utc(),
                        "gate": "spikesort_artifacts_ready",
                        "sorter_output_dir": str(sorter_output_dir),
                    }
                )
            else:
                stage_entries.append(
                    {
                        "status": "error",
                        "stage": stage,
                        "dataset_id": target.dataset_id,
                        "h5_path": str(target.h5_path),
                        "stream_id": target.stream_id,
                        "started_at": None,
                        "finished_at": _now_utc(),
                        "gate": "spikesort_artifacts_ready",
                        "sorter_output_dir": str(sorter_output_dir),
                        "error": "Missing spikesort artifacts required before unit_match",
                    }
                )

        failed = sum(1 for entry in stage_entries if entry.get("status") != "ok")
        return {"stage": stage, "results": stage_entries, "failed": int(failed)}

    if stage == "merge_update":
        unit_match_stage = next((s for s in summary.get("stages", []) if s.get("stage") == "unit_match"), None)
        missing_or_failed_unit_match = unit_match_stage is None or int(unit_match_stage.get("failed", 0) or 0) > 0

        stage_entries: list[dict[str, Any]] = []
        for target in targets:
            if missing_or_failed_unit_match:
                stage_entries.append(
                    {
                        "status": "error",
                        "stage": stage,
                        "dataset_id": target.dataset_id,
                        "h5_path": str(target.h5_path),
                        "stream_id": target.stream_id,
                        "started_at": None,
                        "finished_at": _now_utc(),
                        "gate": "merge_updates_after_unit_match",
                        "error": "unit_match must complete successfully before merge_update",
                    }
                )
            else:
                stage_entries.append(
                    {
                        "status": "ok",
                        "stage": stage,
                        "dataset_id": target.dataset_id,
                        "h5_path": str(target.h5_path),
                        "stream_id": target.stream_id,
                        "started_at": _now_utc(),
                        "finished_at": _now_utc(),
                        "gate": "merge_updates_after_unit_match",
                    }
                )

        failed = sum(1 for entry in stage_entries if entry.get("status") != "ok")
        return {"stage": stage, "results": stage_entries, "failed": int(failed)}

    raise ValueError(f"Unsupported transition stage: {stage}")


def run_scope_stage_barriers(
    *,
    config: ScopeConfig,
    dry_run: bool = False,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    logger = logger or logging.getLogger("axon_reconstructor.scope")
    targets = _build_targets(config)
    stage_order = list(config.effective_stage_order())

    summary: dict[str, Any] = {
        "generated_utc": _now_utc(),
        "dry_run": bool(dry_run),
        "stage_order": stage_order,
        "targets": [
            {
                "dataset_id": t.dataset_id,
                "h5_path": str(t.h5_path),
                "stream_id": t.stream_id,
            }
            for t in targets
        ],
        "stages": [],
    }

    barrier_checkpoint_file = _scope_barrier_checkpoint_file(config=config)
    summary["barrier_checkpoint_file"] = str(barrier_checkpoint_file)
    barrier_state = _load_scope_barrier_checkpoint(
        checkpoint_file=barrier_checkpoint_file,
        force_restart=bool(config.force_restart),
    )

    if dry_run:
        for stage in stage_order:
            stage_entries = []
            for target in targets:
                if stage in {"unit_match", "merge_update"}:
                    gate_name = "spikesort_artifacts_ready" if stage == "unit_match" else "merge_updates_after_unit_match"
                    stage_entries.append(
                        {
                            "status": "planned",
                            "stage": stage,
                            "dataset_id": target.dataset_id,
                            "h5_path": str(target.h5_path),
                            "stream_id": target.stream_id,
                            "gate": gate_name,
                            "stage_kwargs": _merge_stage_kwargs(stage=stage, config=config, target=target),
                        }
                    )
                    continue

                stage_entries.append(
                    {
                        "status": "planned",
                        "stage": stage,
                        "dataset_id": target.dataset_id,
                        "h5_path": str(target.h5_path),
                        "stream_id": target.stream_id,
                        "stage_kwargs": _merge_stage_kwargs(stage=stage, config=config, target=target),
                    }
                )
            summary["stages"].append({"stage": stage, "results": stage_entries, "failed": 0})
        return summary

    for stage in stage_order:
        if (not bool(config.force_restart)) and (stage in set(barrier_state.get("completed_stages", []))):
            resumed_stage_entry = dict(
                barrier_state.get("stage_entries", {}).get(
                    stage,
                    {
                        "stage": stage,
                        "results": [],
                        "failed": 0,
                    },
                )
            )
            resumed_stage_entry["resumed_from_barrier_checkpoint"] = True
            summary["stages"].append(resumed_stage_entry)
            logger.info("Stage barrier resumed from checkpoint: %s", stage)
            continue

        logger.info("Stage barrier starting: %s (targets=%d, parallelism=%d)", stage, len(targets), int(config.per_well_parallelism))

        if stage in {"unit_match", "merge_update"}:
            transition_summary = _run_transition_stage(stage=stage, config=config, targets=targets, summary=summary)
            summary["stages"].append(transition_summary)
            failed = int(transition_summary.get("failed", 0) or 0)
            barrier_state.setdefault("stage_entries", {})[stage] = transition_summary
            if failed == 0:
                completed = list(barrier_state.get("completed_stages", []))
                if stage not in completed:
                    completed.append(stage)
                barrier_state["completed_stages"] = completed
            else:
                barrier_state["completed_stages"] = [s for s in list(barrier_state.get("completed_stages", [])) if s != stage]
            _save_scope_barrier_checkpoint(checkpoint_file=barrier_checkpoint_file, barrier_state=barrier_state)
            if failed > 0 and bool(config.fail_fast):
                logger.error("Stage barrier failed: %s (failed=%d). Stopping because fail_fast=true", stage, int(failed))
                break
            continue

        stage_results: list[dict[str, Any]] = []
        if int(config.per_well_parallelism) <= 1 or len(targets) <= 1:
            for target in targets:
                try:
                    result = _run_single_stage_target(stage=stage, config=config, target=target)
                except Exception as e:
                    result = {
                        "status": "error",
                        "stage": stage,
                        "dataset_id": target.dataset_id,
                        "h5_path": str(target.h5_path),
                        "stream_id": target.stream_id,
                        "started_at": None,
                        "finished_at": _now_utc(),
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                    }
                stage_results.append(result)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=int(config.per_well_parallelism)) as pool:
                future_map: dict[concurrent.futures.Future[dict[str, Any]], ScopeTarget] = {}
                for target in targets:
                    fut = pool.submit(_run_single_stage_target, stage=stage, config=config, target=target)
                    future_map[fut] = target

                for fut in concurrent.futures.as_completed(future_map):
                    target = future_map[fut]
                    try:
                        result = fut.result()
                    except Exception as e:
                        result = {
                            "status": "error",
                            "stage": stage,
                            "dataset_id": target.dataset_id,
                            "h5_path": str(target.h5_path),
                            "stream_id": target.stream_id,
                            "started_at": None,
                            "finished_at": _now_utc(),
                            "error": str(e),
                            "traceback": traceback.format_exc(),
                        }
                    stage_results.append(result)

        stage_results.sort(key=lambda x: (str(x.get("dataset_id")), str(x.get("stream_id"))))
        failed = sum(1 for r in stage_results if r.get("status") != "ok")
        stage_entry = {"stage": stage, "results": stage_results, "failed": int(failed)}
        summary["stages"].append(stage_entry)
        barrier_state.setdefault("stage_entries", {})[stage] = stage_entry
        if failed == 0:
            completed = list(barrier_state.get("completed_stages", []))
            if stage not in completed:
                completed.append(stage)
            barrier_state["completed_stages"] = completed
        else:
            barrier_state["completed_stages"] = [s for s in list(barrier_state.get("completed_stages", [])) if s != stage]
        _save_scope_barrier_checkpoint(checkpoint_file=barrier_checkpoint_file, barrier_state=barrier_state)

        if failed > 0 and bool(config.fail_fast):
            logger.error("Stage barrier failed: %s (failed=%d). Stopping because fail_fast=true", stage, int(failed))
            break

    return summary


def write_scope_run_summary(*, summary: dict[str, Any], out_path: Path) -> Path:
    out_path = Path(out_path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


__all__ = [
    "STAGE_CHOICES",
    "add_stage_selector_arg",
    "add_stage_common_required_args",
    "add_stage_spikesort_args",
    "add_stage_execution_args",
    "add_stage_debug_controls",
    "add_stage_analysis_args",
    "add_stage_kwargs_args",
    "StageExecutionContext",
    "StageExecutionResult",
    "execute_stage",
    "ScopeTarget",
    "run_scope_stage_barriers",
    "write_scope_run_summary",
]
