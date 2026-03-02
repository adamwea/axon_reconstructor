from __future__ import annotations

import concurrent.futures
import datetime as dt
import json
import logging
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .analysis import AnalysisInputs, analyze_units
from .pipeline_driver import AxonReconstructor, _compute_mea_analysis_output_dir
from .reconstruction import ReconstructionInputs, reconstruct_from_templates
from .scope_config import ScopeConfig
from .spikesorting import SpikeSortingInputs, run_spikesorting_stage
from .templates import TemplateExtractInputs, extract_and_merge_templates
from .waveforms import WaveformExtractInputs, extract_waveforms


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

    if stage == "preprocess":
        recon = AxonReconstructor(
            h5_parent_dirs=[target.h5_path],
            mea_analysis_output_root=str(config.mea_output_root),
            force_restart=bool(config.force_restart),
        )
        preprocess_kwargs = {
            "h5_path": target.h5_path,
            "stream_id": target.stream_id,
            "n_jobs": int(config.n_jobs),
            "overwrite_saved_recording": bool(config.force_restart),
        }
        preprocess_kwargs.update(stage_kwargs)
        _, common_el = recon.preprocess_for_spikesorting(**preprocess_kwargs)
        return {
            "status": "ok",
            "stage": stage,
            "dataset_id": target.dataset_id,
            "h5_path": str(target.h5_path),
            "stream_id": target.stream_id,
            "started_at": started_at,
            "finished_at": _now_utc(),
            "n_common_electrodes": int(len(common_el)),
        }

    if stage == "spikesort":
        if config.mea_analysis_repo_root is None:
            raise RuntimeError("spikesort stage requires mea_analysis_repo_root in scope config")

        spikesort_fields = {
            "h5_path": target.h5_path,
            "stream_id": target.stream_id,
            "mea_output_root": config.mea_output_root,
            "mea_analysis_repo_root": config.mea_analysis_repo_root,
            "sorter": config.sorter,
            "docker_image": config.docker_image,
            "force_restart": bool(config.force_restart),
            "verbose": True,
            "n_jobs": int(config.n_jobs),
            "chunk_duration": config.chunk_duration,
        }
        spikesort_fields.update(stage_kwargs)
        inputs = SpikeSortingInputs(**spikesort_fields)
        out = run_spikesorting_stage(
            inputs=inputs,
            logger=logging.getLogger(f"axon_reconstructor.scope.{target.stream_id}.spikesort"),
        )
        return {
            "status": "ok",
            "stage": stage,
            "dataset_id": target.dataset_id,
            "h5_path": str(target.h5_path),
            "stream_id": target.stream_id,
            "started_at": started_at,
            "finished_at": _now_utc(),
            "sorter_output_dir": str(out.sorter_output_dir),
            "output_dir": str(out.output_dir),
        }

    if stage == "waveforms":
        waveform_fields = {
            "h5_path": target.h5_path,
            "stream_id": target.stream_id,
            "mea_output_root": config.mea_output_root,
            "sorter": config.sorter,
            "n_jobs": int(config.n_jobs),
            "force_restart": bool(config.force_restart),
        }
        waveform_fields.update(stage_kwargs)
        inputs = WaveformExtractInputs(**waveform_fields)
        out = extract_waveforms(inputs=inputs)
        return {
            "status": "ok",
            "stage": stage,
            "dataset_id": target.dataset_id,
            "h5_path": str(target.h5_path),
            "stream_id": target.stream_id,
            "started_at": started_at,
            "finished_at": _now_utc(),
            "waveforms_out_dir": str(out.waveforms_out_dir),
        }

    if stage == "templates":
        template_fields = {
            "h5_path": target.h5_path,
            "stream_id": target.stream_id,
            "mea_output_root": config.mea_output_root,
            "n_jobs": int(config.n_jobs),
            "force_restart": bool(config.force_restart),
        }
        template_fields.update(stage_kwargs)
        inputs = TemplateExtractInputs(**template_fields)
        out = extract_and_merge_templates(inputs=inputs)
        return {
            "status": "ok",
            "stage": stage,
            "dataset_id": target.dataset_id,
            "h5_path": str(target.h5_path),
            "stream_id": target.stream_id,
            "started_at": started_at,
            "finished_at": _now_utc(),
            "templates_out_dir": str(out.templates_out_dir),
        }

    if stage == "reconstruct":
        recon_fields = {
            "h5_path": target.h5_path,
            "stream_id": target.stream_id,
            "mea_output_root": config.mea_output_root,
            "force_restart": bool(config.force_restart),
        }
        recon_fields.update(stage_kwargs)
        inputs = ReconstructionInputs(**recon_fields)
        out = reconstruct_from_templates(inputs=inputs)
        return {
            "status": "ok",
            "stage": stage,
            "dataset_id": target.dataset_id,
            "h5_path": str(target.h5_path),
            "stream_id": target.stream_id,
            "started_at": started_at,
            "finished_at": _now_utc(),
            "reconstruction_out_dir": str(out.reconstruction_out_dir),
        }

    if stage == "analysis":
        analysis_fields = {
            "h5_path": target.h5_path,
            "stream_id": target.stream_id,
            "mea_output_root": config.mea_output_root,
            "force_restart": bool(config.force_restart),
        }
        analysis_fields.update(stage_kwargs)
        inputs = AnalysisInputs(**analysis_fields)
        out = analyze_units(inputs=inputs)
        return {
            "status": "ok",
            "stage": stage,
            "dataset_id": target.dataset_id,
            "h5_path": str(target.h5_path),
            "stream_id": target.stream_id,
            "started_at": started_at,
            "finished_at": _now_utc(),
            "analysis_out_dir": str(out.analysis_out_dir),
        }

    raise ValueError(f"Unsupported stage: {stage}")


def _expected_sorter_output_dir(*, config: ScopeConfig, target: ScopeTarget) -> Path:
    well_out_dir = _compute_mea_analysis_output_dir(
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
            for t in targets:
                if stage in {"unit_match", "merge_update"}:
                    gate_name = "spikesort_artifacts_ready" if stage == "unit_match" else "merge_updates_after_unit_match"
                    stage_entries.append(
                        {
                            "status": "planned",
                            "stage": stage,
                            "dataset_id": t.dataset_id,
                            "h5_path": str(t.h5_path),
                            "stream_id": t.stream_id,
                            "gate": gate_name,
                            "stage_kwargs": _merge_stage_kwargs(stage=stage, config=config, target=t),
                        }
                    )
                    continue

                stage_entries.append(
                    {
                        "status": "planned",
                        "stage": stage,
                        "dataset_id": t.dataset_id,
                        "h5_path": str(t.h5_path),
                        "stream_id": t.stream_id,
                        "stage_kwargs": _merge_stage_kwargs(stage=stage, config=config, target=t),
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
