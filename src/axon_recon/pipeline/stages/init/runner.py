from __future__ import annotations

import logging
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any

from ...checkpoint import find_first_broken_phase, read_checkpoint_status
from ...execution.context import ExecutionTarget
from ...execution.results import MultiTargetStageResult, TargetStageResult

from .config import InitStageConfig
from .models.inputs import InitInputs
from .phases.copy_src_to_scratch import (
	_resolve_summary_json_path as _resolve_copy_src_summary_path,
	run_init_copy_src_to_scratch_phase,
)


LOGGER = logging.getLogger("axon_recon.init.runner")


def build_init_inputs_for_target(
	*,
	target: ExecutionTarget,
	stage_config: InitStageConfig,
) -> InitInputs:
	"""Project an `ExecutionTarget` + parsed stage config into per-target `InitInputs`.

	Mirrors `build_preprocess_inputs_for_target` but for the slimmer init
	dataclass. `copied_to_scratch` is computed the same way: source_h5 differs
	from h5 iff `select_execution_targets` materialized the file under the
	dataset's scratch_input_root.
	"""

	source_h5_path = target.source_h5_path or target.h5_path
	try:
		copied_to_scratch = Path(source_h5_path).expanduser().resolve() != Path(target.h5_path).expanduser().resolve()
	except Exception:
		copied_to_scratch = Path(source_h5_path) != Path(target.h5_path)
	return InitInputs(
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


def _run_init_copy_src_to_scratch_target(
	*,
	target: ExecutionTarget,
	stage_config: InitStageConfig,
) -> TargetStageResult:
	inputs = build_init_inputs_for_target(target=target, stage_config=stage_config)
	try:
		payload = run_init_copy_src_to_scratch_phase(inputs)
		return TargetStageResult(target=target, status="ok", result=payload, error=None)
	except Exception as exc:
		LOGGER.exception(
			"init copy_src_to_scratch failed for target dataset=%s stream_id=%s",
			str(getattr(target, "dataset_index", "?")),
			str(getattr(target, "stream_id", "?")),
		)
		return TargetStageResult(
			target=target,
			status="error",
			result=None,
			error=f"{exc!s}\n{traceback.format_exc()}",
		)


# Lookup table: phase_name -> per-target runner. Slice 5 ships a single entry;
# additional once-per-data-config phases plug in here without touching
# `run_init_stage`'s control flow.
_INIT_TARGET_RUNNERS = {
	"copy_src_to_scratch": _run_init_copy_src_to_scratch_target,
}


# Lookup table: phase_name -> function computing the phase's summary_json_path
# from a fully-built InitInputs. Used by the auto-restart-from-first-broken
# check in `run_init_stage` to read each phase's current checkpoint status
# without invoking the phase itself.
_INIT_SUMMARY_PATH_RESOLVERS = {
	"copy_src_to_scratch": _resolve_copy_src_summary_path,
}


def _summary_json_paths_for_target(
	*,
	target: ExecutionTarget,
	stage_config: InitStageConfig,
) -> dict[str, Path]:
	"""Compute the summary_json path for each phase in `phase_sequence` on `target`."""
	inputs = build_init_inputs_for_target(target=target, stage_config=stage_config)
	out: dict[str, Path] = {}
	for phase_name in stage_config.phase_sequence:
		resolver = _INIT_SUMMARY_PATH_RESOLVERS.get(str(phase_name))
		if resolver is None:
			continue
		out[str(phase_name)] = resolver(inputs)
	return out


def _yaml_skipped_phases(stage_config: InitStageConfig) -> frozenset[str]:
	"""Return the set of phase names whose YAML block has enabled:false.

	Used by `find_first_broken_phase` to distinguish "this phase is
	disabled, its missing summary is intentional" from "this phase is
	supposed to run but never did".
	"""
	skipped: list[str] = []
	for phase_name in stage_config.phase_sequence:
		phase_cfg = getattr(stage_config.phases, str(phase_name), None)
		if phase_cfg is not None and not bool(getattr(phase_cfg, "enabled", True)):
			skipped.append(str(phase_name))
	return frozenset(skipped)


def run_init_stage(
	stage_config: InitStageConfig,
	*,
	targets: list[ExecutionTarget] | None = None,
) -> MultiTargetStageResult:
	"""Run the init stage's configured phase sequence for the supplied targets.

	When the stage is disabled OR its `phase_sequence` is empty (the YAML
	default), the runner short-circuits to an empty `MultiTargetStageResult`.
	When phases are configured but no targets are supplied (the no-bundle
	dataclass-only invocation used in unit tests) the runner likewise no-ops:
	the runtime entry point `run_init_from_runtime` is the production path
	that resolves and supplies targets.

	An unsupported phase in `phase_sequence` raises `NotImplementedError` so
	a half-wired YAML surfaces loudly rather than silently passing.
	"""

	if not stage_config.enabled or not stage_config.phase_sequence:
		LOGGER.debug(
			"init stage no-op enabled=%s phase_sequence=%s",
			bool(stage_config.enabled),
			tuple(stage_config.phase_sequence),
		)
		return MultiTargetStageResult(
			stage="init",
			total_targets=0,
			succeeded_targets=0,
			failed_targets=0,
			target_results=[],
		)

	target_list: list[ExecutionTarget] = list(targets or [])
	if not target_list:
		LOGGER.info(
			"init stage configured but no targets supplied; returning empty MultiTargetStageResult",
		)
		return MultiTargetStageResult(
			stage="init",
			total_targets=0,
			succeeded_targets=0,
			failed_targets=0,
			target_results=[],
		)

	for phase_name in stage_config.phase_sequence:
		if phase_name not in _INIT_TARGET_RUNNERS:
			raise NotImplementedError(
				f"init stage has no target runner registered for phase '{phase_name}'"
			)

	all_target_results: list[TargetStageResult] = []
	final_stage_name = "init"

	# Auto-restart-from-first-broken (slice 14): when neither --force-restart
	# nor --replot is set, walk each target's phase summaries and identify
	# the first phase that is NOT ok. Phases before the broken index reuse
	# their cached outputs (treated as a clean "skipped — already ok" result
	# in the aggregate). Phases from the broken index onwards run as normal
	# — the phase's own write path will overwrite stale markers via the
	# slice 13 `with_checkpoint_marker` machinery.
	#
	# --force-restart bypass: every phase runs (the phase's own
	# delete_outputs_on_force_restart logic handles cleanup).
	# --replot bypass: only plot/report phases run; init has none, so this
	# bypass effectively no-ops the entire stage. The pipeline-level CLI
	# layer already filters which phases run per `is_plot_or_report_phase`
	# (slice 11); the stage runner can therefore proceed without
	# special-casing it here.
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
					"init stage auto-restart: all phases ok for target dataset=%s stream_id=%s; skipping",
					str(getattr(target, "dataset_index", "?")),
					str(getattr(target, "stream_id", "?")),
				)
				all_target_results.append(
					TargetStageResult(
						target=target,
						status="ok",
						result={"stage": "init", "status": "skipped", "reason": "all_phases_ok"},
						error=None,
					)
				)
				continue
			restart_from_index, restart_status = broken
			LOGGER.info(
				"init stage auto-restart: first broken phase index=%d status=%s target dataset=%s stream_id=%s",
				int(restart_from_index),
				str(restart_status),
				str(getattr(target, "dataset_index", "?")),
				str(getattr(target, "stream_id", "?")),
			)
		for phase_index, phase_name in enumerate(stage_config.phase_sequence):
			if restart_from_index is not None and phase_index < restart_from_index:
				# Phase ran successfully on a previous invocation; skip.
				continue
			runner_fn = _INIT_TARGET_RUNNERS[phase_name]
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
	"build_init_inputs_for_target",
	"run_init_stage",
]
