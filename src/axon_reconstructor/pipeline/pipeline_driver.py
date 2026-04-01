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

from .alias_modules.analysis import AnalysisInputs, analyze_units
from .alias_modules.preprocessing import run_preprocess_stage
from .alias_modules.reconstruction import ReconstructionInputs, reconstruct_from_templates
from .alias_modules.spikesorting import SpikeSortingInputs, run_spikesorting_stage
from .alias_modules.templates import TemplateExtractInputs, extract_and_merge_templates
from .alias_modules.waveforms import WaveformExtractInputs, extract_waveforms
from .output_paths import compute_mea_analysis_output_dir
from .publish import publish_path_to_final, remap_path_string_to_final
from .scope_config import ScopeConfig


STAGE_CHOICES = ("preprocess", "spikesort", "waveforms", "templates", "reconstruct", "analysis")


def add_stage_selector_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("stage", choices=(*STAGE_CHOICES, "all"))


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
    parser.add_argument(
        "--scratch-output-root",
        required=False,
        default=None,
        help=(
            "Optional scratch output root for high-throughput local writes; artifacts are published to --mea-output-root "
            "(or AXON_RECON_SCRATCH_OUTPUT_ROOT)."
        ),
    )


def add_stage_spikesort_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--sorter", default="kilosort4")
    parser.add_argument("--docker-image", default=None)
    parser.add_argument("--chunk-duration", default=None)
    parser.add_argument(
        "--resume-from",
        default=None,
        help="Resume MEA_Analysis from stage token (e.g. merge) (env: AXON_RECON_SPIKESORT_RESUME_FROM).",
    )
    parser.add_argument(
        "--ks-th-universal",
        type=float,
        default=None,
        help="Override Kilosort4 Th_universal (env: AXON_RECON_KS_TH_UNIVERSAL).",
    )
    parser.add_argument(
        "--ks-th-learned",
        type=float,
        default=None,
        help="Override Kilosort4 Th_learned (env: AXON_RECON_KS_TH_LEARNED).",
    )
    parser.add_argument(
        "--ks-th-single-ch",
        type=float,
        default=None,
        help="Override Kilosort4 Th_single_ch (env: AXON_RECON_KS_TH_SINGLE_CH).",
    )
    parser.add_argument(
        "--ks-cluster-downsampling",
        type=int,
        default=None,
        help="Override Kilosort4 cluster_downsampling (env: AXON_RECON_KS_CLUSTER_DOWNSAMPLING).",
    )
    parser.add_argument(
        "--ks-nearest-chans",
        type=int,
        default=None,
        help="Override Kilosort4 nearest_chans (env: AXON_RECON_KS_NEAREST_CHANS).",
    )
    parser.add_argument(
        "--ks-max-channel-distance",
        type=float,
        default=None,
        help="Override Kilosort4 max_channel_distance (env: AXON_RECON_KS_MAX_CHANNEL_DISTANCE).",
    )


def add_stage_waveforms_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--prefer-merged-sorting",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Prefer canonical merged sorting artifact from unitmatch_outputs/final_merged_sorting (env: AXON_RECON_WF_PREFER_MERGED_SORTING).",
    )
    parser.add_argument(
        "--merged-sorting-dir",
        default=None,
        help="Optional explicit merged sorting folder override (env: AXON_RECON_WF_MERGED_SORTING_DIR).",
    )
    parser.add_argument(
        "--max-spikes-per-unit",
        type=str,
        default=None,
        help="Waveforms random_spikes cap per unit; accepts int or uncapped tokens (-1/all/unlimited/none).",
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


def add_stage_execution_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--n-jobs", type=int, default=None, help="Override worker count (env fallback: AXON_RECON_N_JOBS).")
    parser.add_argument(
        "--force-restart",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override restart behavior (env fallback: AXON_RECON_FORCE_RESTART).",
    )
    parser.add_argument(
        "--force-replot",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override replot behavior (env fallback: AXON_RECON_FORCE_REPLOT).",
    )
    parser.add_argument(
        "--debug",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable debug logging (env fallback: AXON_RECON_DEBUG).",
    )


def add_stage_reconstruct_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--unit-id",
        type=int,
        default=None,
        help="Reconstruct/replot only a single unit id (reconstruct stage convenience).",
    )
    parser.add_argument(
        "--force-restart-per-unit",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Force per-unit recomputation for selected unit_ids by bypassing per-unit completed checkpoints; "
            "does not imply full stage restart."
        ),
    )
    parser.add_argument(
        "--force-replot-per-unit",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Force per-unit plot rewrite while preserving reconstruction JSON branch/heuristic artifacts; "
            "also disables top-density grid for this run."
        ),
    )
    parser.add_argument(
        "--recon-templates-variant-name",
        type=str,
        default=None,
        help="Templates variant name for reconstruction input routing (env: AXON_RECON_RECON_TEMPLATES_VARIANT_NAME).",
    )
    parser.add_argument(
        "--recon-variant-name",
        type=str,
        default=None,
        help="Reconstruction variant name for output routing (env: AXON_RECON_RECON_VARIANT_NAME).",
    )
    parser.add_argument(
        "--recon-top-n-density-requested",
        type=str,
        default=None,
        help="Top-N units for density grid; accepts int or none/null/all (env: AXON_RECON_RECON_TOP_N_DENSITY_REQUESTED).",
    )
    parser.add_argument(
        "--recon-write-top-density-grid",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable top-density grid write (env: AXON_RECON_RECON_WRITE_TOP_DENSITY_GRID).",
    )
    parser.add_argument(
        "--recon-show-density-scale-debug-text",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable legacy density debug text (env: AXON_RECON_RECON_SHOW_DENSITY_SCALE_DEBUG_TEXT).",
    )
    parser.add_argument(
        "--recon-show-density-scale-global-debug-text",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable global density debug panel (env: AXON_RECON_RECON_SHOW_DENSITY_SCALE_GLOBAL_DEBUG_TEXT).",
    )
    parser.add_argument(
        "--recon-show-density-scale-local-debug-text",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable per-unit density debug text (env: AXON_RECON_RECON_SHOW_DENSITY_SCALE_LOCAL_DEBUG_TEXT).",
    )
    parser.add_argument(
        "--recon-replot-top-density-grid-only",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Replot top-density grid only from existing reconstruction outputs (env: AXON_RECON_RECON_REPLOT_TOP_DENSITY_GRID_ONLY).",
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
    final_output_root: Path | None = None
    scratch_output_root: Path | None = None
    n_jobs: int = 8
    sorter: str = "kilosort4"
    docker_image: Optional[str] = None
    chunk_duration: Optional[str] = None
    verbose: bool = False


@dataclass(frozen=True)
class StageExecutionResult:
    stage: str
    artifacts: dict[str, Any]


def _publish_roots_from_context(context: StageExecutionContext) -> tuple[Path, Path] | None:
    active_root = Path(context.mea_output_root).expanduser().resolve()
    final_root = Path(context.final_output_root or context.mea_output_root).expanduser().resolve()
    if active_root == final_root:
        return None
    return active_root, final_root


def _publish_artifacts_to_final(*, artifacts: dict[str, Any], context: StageExecutionContext) -> dict[str, Any]:
    roots = _publish_roots_from_context(context)
    if roots is None:
        return dict(artifacts)

    active_root, final_root = roots
    out: dict[str, Any] = {}
    for key, value in artifacts.items():
        if value is None:
            out[str(key)] = None
            continue
        if isinstance(value, (str, Path)):
            raw_path = Path(value).expanduser().resolve()
            mapped = publish_path_to_final(
                path=raw_path,
                active_root=active_root,
                final_root=final_root,
                mode="copy",
            )
            if mapped is not None:
                out[str(key)] = remap_path_string_to_final(
                    raw=value,
                    active_root=active_root,
                    final_root=final_root,
                )
                continue
        out[str(key)] = value
    return out


def _stage_result(*, stage: str, artifacts: dict[str, Any], context: StageExecutionContext) -> StageExecutionResult:
    return StageExecutionResult(stage=stage, artifacts=_publish_artifacts_to_final(artifacts=artifacts, context=context))


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
        return _stage_result(stage=stage, artifacts={"n_common_electrodes": int(len(common_el))}, context=context)

    if stage == "spikesort":
        spikesort_fields = {
            "h5_path": context.h5_path,
            "stream_id": context.stream_id,
            "mea_output_root": context.mea_output_root,
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
        return _stage_result(
            stage=stage,
            context=context,
            artifacts={
                "sorter_output_dir": str(out.sorter_output_dir),
                "output_dir": str(out.output_dir),
                "merged_sorting_dir": (str(out.merged_sorting_dir) if out.merged_sorting_dir is not None else None),
                "merged_sorter_output_dir": (str(out.merged_sorter_output_dir) if out.merged_sorter_output_dir is not None else None),
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
        if waveform_fields.get("merged_sorting_dir") is not None:
            waveform_fields["merged_sorting_dir"] = Path(waveform_fields["merged_sorting_dir"]).expanduser().resolve()
        out = extract_waveforms(inputs=WaveformExtractInputs(**waveform_fields))
        return _stage_result(stage=stage, context=context, artifacts={"waveforms_out_dir": str(out.waveforms_out_dir)})

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
        return _stage_result(stage=stage, context=context, artifacts={"templates_out_dir": str(out.templates_out_dir)})

    if stage == "reconstruct":
        recon_fields = {
            "h5_path": context.h5_path,
            "stream_id": context.stream_id,
            "mea_output_root": context.mea_output_root,
            "n_jobs": int(context.n_jobs),
            "force_restart": bool(context.force_restart),
        }
        recon_fields.update(kwargs)

        out = reconstruct_from_templates(inputs=ReconstructionInputs(**recon_fields))
        return _stage_result(
            stage=stage,
            context=context,
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
        return _stage_result(stage=stage, context=context, artifacts={"analysis_out_dir": str(out.analysis_out_dir)})

    raise ValueError(f"Unsupported stage: {stage}")


@dataclass(frozen=True)
class ScopeTarget:
    dataset_id: str
    h5_path: Path
    stream_id: str
    mea_output_root: Path
    scratch_output_root: Path | None
    stage_kwargs: dict[str, dict[str, Any]]

    @property
    def active_output_root(self) -> Path:
        return Path(self.scratch_output_root or self.mea_output_root).expanduser().resolve()


def _now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _scope_barrier_checkpoint_file(*, config: ScopeConfig) -> Path:
    return Path(config.mea_output_root).expanduser().resolve() / "scope_run_barrier_checkpoint.json"


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
        dataset_output_root = Path(dataset.mea_output_root or config.mea_output_root).expanduser().resolve()
        dataset_scratch_root = dataset.scratch_output_root if dataset.scratch_output_root is not None else config.scratch_output_root
        dataset_scratch_root = (
            Path(dataset_scratch_root).expanduser().resolve() if dataset_scratch_root is not None else None
        )
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
                    mea_output_root=dataset_output_root,
                    scratch_output_root=dataset_scratch_root,
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
        mea_output_root=target.active_output_root,
        force_restart=bool(config.force_restart),
        final_output_root=target.mea_output_root,
        scratch_output_root=target.scratch_output_root,
        n_jobs=int(config.n_jobs),
        sorter=config.sorter,
        docker_image=config.docker_image,
        chunk_duration=config.chunk_duration,
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
        "mea_output_root": str(target.mea_output_root),
        "active_output_root": str(target.active_output_root),
        "scratch_output_root": (str(target.scratch_output_root) if target.scratch_output_root is not None else None),
        "started_at": started_at,
        "finished_at": _now_utc(),
        **dict(result.artifacts),
    }


def _expected_sorter_output_dir(*, target: ScopeTarget) -> Path:
    well_out_dir = compute_mea_analysis_output_dir(
        output_root=target.mea_output_root,
        data_file=target.h5_path,
        well=target.stream_id,
    )
    return well_out_dir / "stg2_spikesorting_outputs" / "sorter_output"


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
            sorter_output_dir = _expected_sorter_output_dir(target=target)
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
                "mea_output_root": str(t.mea_output_root),
                "active_output_root": str(t.active_output_root),
                "scratch_output_root": (str(t.scratch_output_root) if t.scratch_output_root is not None else None),
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
