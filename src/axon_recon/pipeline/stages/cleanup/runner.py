from __future__ import annotations

import logging
import traceback
from pathlib import Path

from ...checkpoint import find_first_broken_phase
from ...execution.context import ExecutionTarget
from ...execution.results import MultiTargetStageResult, TargetStageResult

from .config import CleanupStageConfig
from .models.inputs import CleanupInputs
from .phases.wipe_src_scratch import (
	_resolve_summary_json_path as _resolve_wipe_summary_path,
	run_cleanup_wipe_src_scratch_phase,
)


LOGGER = logging.getLogger("axon_recon.cleanup.runner")


def build_cleanup_inputs_for_target(
	*,
	target: ExecutionTarget,
	stage_config: CleanupStageConfig,
) -> CleanupInputs:
	"""Project an `ExecutionTarget` + parsed stage config into per-target `CleanupInputs`.

	Mirrors `build_init_inputs_for_target` (the init stage's slice-5 sibling).
	`copied_to_scratch` is computed the same way: source_h5 differs from h5
	iff `select_execution_targets` materialized the file under the dataset's
	scratch_input_root upstream (typically by the init stage's
	`copy_src_to_scratch`).
	"""

	source_h5_path = target.source_h5_path or target.h5_path
	try:
		copied_to_scratch = Path(source_h5_path).expanduser().resolve() != Path(target.h5_path).expanduser().resolve()
	except Exception:
		copied_to_scratch = Path(source_h5_path) != Path(target.h5_path)
	return CleanupInputs(
		h5_path=target.h5_path,
		stream_id=target.stream_id,
		mea_output_root=target.mea_output_root,
		source_h5_path=source_h5_path,
		copied_to_scratch=bool(copied_to_scratch),
		output_rel_root=stage_config.output_rel_root,
		force_restart=stage_config.force_restart,
		replot=stage_config.replot,
		phase_sequence=stage_config.phase_sequence,
		phases=stage_config.phases,
	)


def _run_cleanup_wipe_src_scratch_target(
	*,
	target: ExecutionTarget,
	stage_config: CleanupStageConfig,
) -> TargetStageResult:
	inputs = build_cleanup_inputs_for_target(target=target, stage_config=stage_config)
	try:
		payload = run_cleanup_wipe_src_scratch_phase(inputs)
		return TargetStageResult(target=target, status="ok", result=payload, error=None)
	except Exception as exc:
		LOGGER.exception(
			"cleanup wipe_src_scratch failed for target dataset=%s stream_id=%s",
			str(getattr(target, "dataset_index", "?")),
			str(getattr(target, "stream_id", "?")),
		)
		return TargetStageResult(
			target=target,
			status="error",
			result=None,
			error=f"{exc!s}\n{traceback.format_exc()}",
		)


# Lookup table: phase_name -> per-target runner. Slice 6 ships a single entry;
# additional cleanup phases (the long-term consolidation of
# spikesort.cleanup_concat_binary / spikesort.cleanup_analyzers /
# reconstruct.clear_templates_cache; see `tech_debt.md` §"Finalize the phase
# roster") will plug in here without touching `run_cleanup_stage`'s control
# flow.
_CLEANUP_TARGET_RUNNERS = {
	"wipe_src_scratch": _run_cleanup_wipe_src_scratch_target,
}


# phase_name -> function computing the phase's summary_json_path from a
# fully-built CleanupInputs. Used by the slice-14 auto-restart check to
# read each phase's checkpoint status without invoking the phase itself.
_CLEANUP_SUMMARY_PATH_RESOLVERS = {
	"wipe_src_scratch": _resolve_wipe_summary_path,
}


def _summary_json_paths_for_target(
	*,
	target: ExecutionTarget,
	stage_config: CleanupStageConfig,
) -> dict[str, Path]:
	"""Compute the summary_json path for each phase in `phase_sequence` on `target`."""
	inputs = build_cleanup_inputs_for_target(target=target, stage_config=stage_config)
	out: dict[str, Path] = {}
	for phase_name in stage_config.phase_sequence:
		resolver = _CLEANUP_SUMMARY_PATH_RESOLVERS.get(str(phase_name))
		if resolver is None:
			continue
		out[str(phase_name)] = resolver(inputs)
	return out


def _yaml_skipped_phases(stage_config: CleanupStageConfig) -> frozenset[str]:
	"""Return the set of phase names whose YAML block has enabled:false.

	The cleanup stage's wipe_src_scratch defaults to enabled:false in both
	debug.runtime.yml files, so this helper is load-bearing for "don't
	rerun a YAML-disabled phase whose summary is missing".
	"""
	skipped: list[str] = []
	for phase_name in stage_config.phase_sequence:
		phase_cfg = getattr(stage_config.phases, str(phase_name), None)
		if phase_cfg is not None and not bool(getattr(phase_cfg, "enabled", True)):
			skipped.append(str(phase_name))
	return frozenset(skipped)


def run_cleanup_stage(
	stage_config: CleanupStageConfig,
	*,
	targets: list[ExecutionTarget] | None = None,
) -> MultiTargetStageResult:
	"""Run the cleanup stage's configured phase sequence for the supplied targets.

	When the stage is disabled OR its `phase_sequence` is empty (the YAML
	default), the runner short-circuits to an empty `MultiTargetStageResult`.
	When phases are configured but no targets are supplied (the no-bundle
	dataclass-only invocation used in unit tests) the runner likewise no-ops:
	the runtime entry point `run_cleanup_from_runtime` is the production path
	that resolves and supplies targets.

	An unsupported phase in `phase_sequence` raises `NotImplementedError` so
	a half-wired YAML surfaces loudly rather than silently passing.
	"""

	if not stage_config.enabled or not stage_config.phase_sequence:
		LOGGER.debug(
			"cleanup stage no-op enabled=%s phase_sequence=%s",
			bool(stage_config.enabled),
			tuple(stage_config.phase_sequence),
		)
		return MultiTargetStageResult(
			stage="cleanup",
			total_targets=0,
			succeeded_targets=0,
			failed_targets=0,
			target_results=[],
		)

	target_list: list[ExecutionTarget] = list(targets or [])
	if not target_list:
		LOGGER.info(
			"cleanup stage configured but no targets supplied; returning empty MultiTargetStageResult",
		)
		return MultiTargetStageResult(
			stage="cleanup",
			total_targets=0,
			succeeded_targets=0,
			failed_targets=0,
			target_results=[],
		)

	for phase_name in stage_config.phase_sequence:
		if phase_name not in _CLEANUP_TARGET_RUNNERS:
			raise NotImplementedError(
				f"cleanup stage has no target runner registered for phase '{phase_name}'"
			)

	all_target_results: list[TargetStageResult] = []
	final_stage_name = "cleanup"

	# Auto-restart-from-first-broken (slice 14): mirrors init/runner.py.
	# When no --force-restart / --replot is set, the runner walks each
	# target's phase summaries and identifies the first phase that is NOT
	# ok. Phases before it are reused; phases from there onwards run.
	# --force-restart bypasses the check; every phase runs. --replot
	# bypasses too, but cleanup has no plot/report phases so it's a noop.
	bypass_auto_restart = bool(stage_config.force_restart) or bool(stage_config.replot)
	yaml_skipped = _yaml_skipped_phases(stage_config)

	for target in target_list:
		restart_from_index: int | None = None
		if not bypass_auto_restart:
			summary_paths = _summary_json_paths_for_target(
				target=target, stage_config=stage_config
			)
			broken = find_first_broken_phase(
				stage_config.phase_sequence,
				summary_paths,
				yaml_skipped_phases=yaml_skipped,
			)
			if broken is None:
				LOGGER.info(
					"cleanup stage auto-restart: all phases ok for target dataset=%s stream_id=%s; skipping",
					str(getattr(target, "dataset_index", "?")),
					str(getattr(target, "stream_id", "?")),
				)
				all_target_results.append(
					TargetStageResult(
						target=target,
						status="ok",
						result={"stage": "cleanup", "status": "skipped", "reason": "all_phases_ok"},
						error=None,
					)
				)
				continue
			restart_from_index, restart_status = broken
			LOGGER.info(
				"cleanup stage auto-restart: first broken phase index=%d status=%s target dataset=%s stream_id=%s",
				int(restart_from_index),
				str(restart_status),
				str(getattr(target, "dataset_index", "?")),
				str(getattr(target, "stream_id", "?")),
			)
		for phase_index, phase_name in enumerate(stage_config.phase_sequence):
			if restart_from_index is not None and phase_index < restart_from_index:
				continue
			runner_fn = _CLEANUP_TARGET_RUNNERS[phase_name]
			result = runner_fn(target=target, stage_config=stage_config)
			all_target_results.append(result)

	succeeded = sum(1 for item in all_target_results if item.status == "ok")
	failed = sum(1 for item in all_target_results if item.status != "ok")
	return MultiTargetStageResult(
		stage=final_stage_name,
		total_targets=len(all_target_results),
		succeeded_targets=succeeded,
		failed_targets=failed,
		target_results=all_target_results,
	)


__all__ = [
	"MultiTargetStageResult",
	"TargetStageResult",
	"build_cleanup_inputs_for_target",
	"run_cleanup_stage",
]
