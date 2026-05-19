from __future__ import annotations

import copy
import logging
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

from axon_recon.pipeline.publish import publish_path_to_final, remap_path_string_to_final

from .config import (
	PipelineRuntimeBundle,
	load_pipeline_runtime_bundle,
	parse_resources_config_for_bundle,
	resolve_stage_parallelism,
	select_execution_targets,
)
from .cpu_allocation import (
	ContainerAffinityReadiness,
	TaskAllocationPlan,
	TaskSlot,
	_THREAD_ENV_VARS,
	apply_thread_env_context,
	build_task_allocation_plan,
	capture_sample_worker_environment,
	current_task_slot,
	detect_cpu_topology,
	format_cpu_set,
	phase_budgets_context,
	probe_container_readiness,
	task_allocation_context,
	task_slot_affinity_context,
)
from .execution.context import ExecutionTarget
from .execution.distributor import distribute_targets
from .execution.read_groups import count_target_read_groups
from .execution.logging_context import (
	install_pipeline_log_record_factory,
	pipeline_log_context_for_target,
)
from .mpi_adapter import current_mpi_context, log_mpi_context, partition_targets_by_mpi_rank
from .execution.phase_chain import PhaseDescriptor, run_phase_chain
from .execution.progress import PipelineProgress, ProgressSpec, pipeline_progress_context
from .execution.results import MultiTargetStageResult, TargetStageResult
from .resource_budget import ResourceBudgetManager, stage_resource_budget_context
from .resources import TaskAllocationConfig, get_active_profile, get_active_resource_profile, parse_resources_config
from .shared.maxwell_plugin import install_maxwell_hdf5_plugin_message_filter
from .stages.preprocess.api import (
	run_preprocess,
	run_preprocess_plot_raster_threshold,
	run_preprocess_plot_segment_channel_layouts,
	run_preprocess_plot_segment_traces,
	run_preprocess_preprocess_segments,
	run_preprocess_save_rec_metadata,
)
from .stages.preprocess.config import (
	build_preprocess_inputs_for_target,
	parse_preprocess_stage_config,
)
from .stages.preprocess.models.results import PreprocessResult
from .stages.preprocess.runner import target_all_preprocess_phases_succeeded
from .stages.reconstruct.api import (
	run_reconstruct,
	run_reconstruct_clear_templates_cache,
	run_reconstruct_axon_velocity_gtrs,
	run_reconstruct_plot_branch_propagations,
	run_reconstruct_plot_branch_velocities,
	run_reconstruct_plot_recons,
	run_reconstruct_plot_unit_summary,
	run_reconstruct_report_full_chip_layout,
	run_reconstruct_report_recon_grid,
	run_reconstruct_report_recons,
	run_reconstruct_report_summaries,
	run_reconstruct_templates_analyzers,
	run_reconstruct_templates_build_templates,
	run_reconstruct_templates_compute_template_similarity,
	run_reconstruct_templates_extract_partial_templates,
	run_reconstruct_templates_plot_templates_v2,
	run_reconstruct_templates_report_templates,
	run_reconstruct_templates_resolve_sources,
)
from .stages.reconstruct.config import (
	build_reconstruct_templates_runtime_config,
	build_reconstruction_inputs_for_target,
	parse_reconstruction_stage_config,
)
from .stages.reconstruct.models.results import ReconstructionResult, UnitReconstructionResult
from .stages.reconstruct.runner import (
	_display_reconstruct_stage_phase_name,
	_reconstruct_stage_phase_resource_class,
)
from .stages.reconstruct.templates.config import (
	build_templates_inputs_for_target,
	parse_probe_geometry_from_data_config,
	parse_reconstruct_templates_config,
)
from .stages.spikesort.api import (
	build_spikesort_concat_analyzer,
	build_spikesort_concat_binary,
	cleanup_spikesort_analyzers,
	cleanup_spikesort_concat_binary,
	restore_spikesort_sorter_output,
	run_spikesort,
	run_spikesort_bombcell,
	run_spikesort_bombcell_pass2,
	run_spikesort_merge,
	run_spikesort_plot_concat_channel_layout,
	run_spikesort_plot_concat_traces,
	snapshot_spikesort_sorter_output,
	summarize_spikesort,
)
from .stages.spikesort.config import (
	DEFAULT_SPIKESORT_PHASE_SEQUENCE,
	build_spikesort_inputs_for_target,
	normalize_spikesort_phase_name,
	parse_spikesort_stage_config,
)
from .stages.spikesort.models.results import (
	SpikesortBombcellResult,
	SpikesortMergeResult,
	SpikesortResult,
)
from .stages.spikesort.orchestrators.merge_slay import run_spikesort_merge_slay
from .stages.analysis.api import run_analysis_metrics
from .stages.analysis.config import (
	AnalysisStageConfig,
	parse_analysis_stage_config,
)
from .stages.analysis.models.results import AnalysisResult
from .stages.init.api import run_init_copy_src_to_scratch
from .stages.init.config import (
	InitStageConfig,
	parse_init_stage_config,
)
from .stages.init.runner import build_init_inputs_for_target, run_init_stage
from .stages.cleanup.api import run_cleanup_wipe_src_scratch
from .stages.cleanup.config import (
	CleanupStageConfig,
	parse_cleanup_stage_config,
)
from .stages.cleanup.runner import build_cleanup_inputs_for_target, run_cleanup_stage

LOGGER = logging.getLogger("axon_recon.pipeline.runner")
_WARNED_IGNORED_PHASE_DEBUG_LIMITS: set[tuple[str, str]] = set()


@dataclass(frozen=True)
class _SpikesortRuntimePhase:
	name: str
	phase_label: str
	debug_enabled_attr: str
	debug_limit_datasets_attr: str
	debug_limit_wells_attr: str
	target_runner: Callable[..., Any]
	debug_limit_wells_per_dataset_attr: str | None = None
	resource_class: str | None = None

	def __iter__(self):
		yield self.name
		yield self.target_runner


@dataclass(frozen=True)
class PublishPolicy:
	publish_outputs: bool = True
	wipe_scratch_roots: bool = False

	def publish_mode(self) -> str:
		return "move" if bool(self.wipe_scratch_roots) else "copy"


def _warn_ignored_phase_debug_limits(*, stage_name: str, phase_label: str) -> None:
	token = (str(stage_name), str(phase_label))
	if token in _WARNED_IGNORED_PHASE_DEBUG_LIMITS:
		return
	_WARNED_IGNORED_PHASE_DEBUG_LIMITS.add(token)
	LOGGER.warning(
		"Ignoring deprecated phase-level debug_mode limits for %s (%s). Use stage-wide debug_mode or CLI --limit overrides instead.",
		str(stage_name),
		str(phase_label),
	)


def is_plot_or_report_phase(phase_name: str) -> bool:
	"""Return True iff `phase_name` is a plot or report phase.

	Per `guardrails/force_restart.md` §"Three invocation modes", `--replot`
	selects phases whose name starts with `plot_` or `report_`, OR whose
	name contains a `_plot_` / `_report_` segment introduced by a stage's
	namespacing prefix (e.g. the reconstruct stage's `templates_plot_templates_v2`
	and `templates_report_templates`). Compute phases are skipped.
	The classification is naming-convention based; phases not following
	either convention are treated as compute and skipped.
	"""

	name = str(phase_name or "").strip()
	if not name:
		return False
	if name.startswith("plot_") or name.startswith("report_"):
		return True
	if "_plot_" in name or "_report_" in name:
		return True
	return False


def _filter_phase_sequence_for_replot(
	phase_sequence: tuple[str, ...] | list[str],
	*,
	replot: bool,
	stage_name: str = "",
) -> tuple[str, ...]:
	"""When `replot` is True, return only plot/report phases from `phase_sequence`.

	When `replot` is False, return `phase_sequence` unchanged.
	"""

	if not replot:
		return tuple(str(item) for item in phase_sequence)
	kept: list[str] = []
	skipped: list[str] = []
	for item in phase_sequence:
		name = str(item)
		if is_plot_or_report_phase(name):
			kept.append(name)
		else:
			skipped.append(name)
	if skipped:
		LOGGER.info(
			"replot: stage=%s skipping non-plot/report phases %s (kept plot/report phases: %s)",
			stage_name or "<unspecified>",
			skipped,
			kept,
			extra={
				"event": "replot_skipped",
				"replot_stage": stage_name,
				"skipped_phases": skipped,
				"kept_phases": kept,
			},
		)
	return tuple(kept)


def _apply_replot_phase_filter(stage_config: Any, *, replot: bool, stage_name: str) -> Any:
	"""Return `stage_config` with `phase_sequence` filtered to plot/report phases.

	When `replot` is True, drops non-plot/report phases from the stage's
	configured `phase_sequence`. When `replot` is False, returns the
	`stage_config` unchanged. The stage_config dataclass must expose
	`phase_sequence`; this helper uses `dataclasses.replace` so the
	original (frozen) dataclass is not mutated.
	"""

	if not replot:
		return stage_config
	original = getattr(stage_config, "phase_sequence", None)
	if original is None:
		return stage_config
	filtered = _filter_phase_sequence_for_replot(
		original,
		replot=True,
		stage_name=stage_name,
	)
	try:
		return replace(stage_config, phase_sequence=filtered)
	except TypeError:
		# Not a frozen dataclass that accepts replace; fall back to attribute
		# assignment when possible. Best-effort; if neither works, return the
		# original config and let the caller proceed with the full sequence.
		try:
			object.__setattr__(stage_config, "phase_sequence", filtered)
		except Exception:
			pass
		return stage_config


def _preprocess_copy_phase_enabled(stage_config: Any) -> bool:
	"""Always False after slice 5: `copy_src_to_scratch` moved to the init stage.

	Retained as a thin shim so the few call sites that gated
	`materialize_scratch_inputs` on it (the `run_preprocess_from_runtime` path)
	keep compiling without a wider refactor. New scratch materialization is
	triggered by the init stage's own `_init_copy_phase_enabled` check.
	"""

	del stage_config
	return False


def _preprocess_stage_phase_in_sequence(stage_config: Any, phase_name: str) -> bool:
	sequence = getattr(stage_config, "phase_sequence", None)
	if sequence is None:
		return True
	return str(phase_name) in {str(item) for item in sequence}


def _select_preprocess_execution_targets(
	*,
	bundle: PipelineRuntimeBundle,
	stage_config: Any,
	materialize_scratch_inputs: bool,
	target_datasets: list[int] | None = None,
) -> list[Any]:
	kwargs: dict[str, Any] = {
		"bundle": bundle,
		"materialize_scratch_inputs": bool(materialize_scratch_inputs),
	}
	limit_datasets = getattr(stage_config, "debug_limit_datasets", None)
	limit_wells = getattr(stage_config, "debug_limit_wells", None)
	limit_wells_per_dataset = getattr(stage_config, "debug_limit_wells_per_dataset", None)
	if target_datasets is not None:
		kwargs["target_datasets"] = list(target_datasets)
	if limit_datasets is not None:
		kwargs["limit_datasets"] = int(limit_datasets)
	if limit_wells is not None:
		kwargs["limit_wells"] = int(limit_wells)
	if limit_wells_per_dataset is not None:
		kwargs["limit_wells_per_dataset"] = int(limit_wells_per_dataset)
	return select_execution_targets(**kwargs)


def _preprocess_stage_uses_nested_workers(stage_config: Any) -> bool:
	try:
		phases = stage_config.phases
	except Exception:
		return True
	preprocess_segments_enabled = bool(getattr(getattr(phases, "preprocess_segments", None), "enabled", False))
	return bool(
		preprocess_segments_enabled and _preprocess_stage_phase_in_sequence(stage_config, "preprocess_segments")
	)


def _preprocess_substage_uses_nested_workers(stage_name: str) -> bool:
	return str(stage_name).strip() in {
		"preprocess.preprocess_segments",
	}


def _preprocess_runtime_uses_nested_workers(*, stage_name: str, stage_config: Any) -> bool:
	return (
		_preprocess_stage_uses_nested_workers(stage_config)
		if str(stage_name).strip() == "preprocess"
		else _preprocess_substage_uses_nested_workers(stage_name)
	)


def _preprocess_runtime_n_jobs_source(*, stage_config: Any, uses_nested_workers: bool) -> str:
	if getattr(stage_config, "n_jobs", None) is not None:
		return "configured"
	return "derived" if bool(uses_nested_workers) else "serial"


def _resolve_preprocess_runtime_unit_workers(*, stage_name: str, parallelism: Any, stage_config: Any) -> int:
	uses_nested_workers = _preprocess_runtime_uses_nested_workers(stage_name=stage_name, stage_config=stage_config)
	configured_n_jobs = getattr(stage_config, "n_jobs", None)
	if configured_n_jobs is not None:
		n_jobs = max(1, int(configured_n_jobs))
	else:
		n_jobs = int(parallelism.unit_workers) if bool(uses_nested_workers) else 1
	n_jobs_source = _preprocess_runtime_n_jobs_source(
		stage_config=stage_config,
		uses_nested_workers=bool(uses_nested_workers),
	)
	emit_subphase_dividers_to_stdout = not (bool(uses_nested_workers) and int(parallelism.well_workers) > 1)
	LOGGER.info(
		"Preprocess worker allocation stage=%s well_workers=%d n_jobs=%d n_jobs_source=%s uses_nested_workers=%s emit_subphase_dividers_to_stdout=%s",
		str(stage_name),
		int(parallelism.well_workers),
		int(max(1, n_jobs)),
		str(n_jobs_source),
		bool(uses_nested_workers),
		bool(emit_subphase_dividers_to_stdout),
	)
	return max(1, int(n_jobs))


def _resolve_preprocess_subphase_dividers_to_stdout(*, stage_name: str, parallelism: Any, stage_config: Any) -> bool:
	uses_nested_workers = _preprocess_runtime_uses_nested_workers(stage_name=stage_name, stage_config=stage_config)
	return not (bool(uses_nested_workers) and int(parallelism.well_workers) > 1)


def _resolve_runtime_stage_parallelism(
	*,
	bundle: PipelineRuntimeBundle,
	stage_name: str,
	target_count: int,
	targets: list[Any] | None = None,
	phase_resource_classes: list[str] | tuple[str, ...] | None = None,
	task_allocation_override: dict[str, Any] | None = None,
	active_profile_override: str | None = None,
):
	try:
		parallelism = resolve_stage_parallelism(
			bundle=bundle,
			stage_name=stage_name,
			target_count=int(target_count),
			phase_resource_classes=list(phase_resource_classes or ()),
		)
	except TypeError as exc:
		if "phase_resource_classes" not in str(exc) and "target_count" not in str(exc):
			raise
		try:
			parallelism = resolve_stage_parallelism(
				bundle=bundle,
				stage_name=stage_name,
				target_count=int(target_count),
			)
		except TypeError as inner_exc:
			if "target_count" not in str(inner_exc):
				raise
			parallelism = resolve_stage_parallelism(bundle=bundle, stage_name=stage_name)
	return _attach_task_allocation_plan(
		bundle=bundle,
		parallelism=parallelism,
		target_count=int(target_count),
		task_allocation_override=task_allocation_override,
		active_profile_override=active_profile_override,
	)


def _available_shm_gb(path: str = "/dev/shm") -> float | None:
	try:
		stats = os.statvfs(path)
	except Exception:
		return None
	return float(int(stats.f_bavail) * int(stats.f_frsize)) / float(1024**3)


def _attach_task_allocation_plan(
	*,
	bundle: PipelineRuntimeBundle,
	parallelism: Any,
	target_count: int,
	task_allocation_override: dict[str, Any] | None = None,
	active_profile_override: str | None = None,
) -> Any:
	if not callable(getattr(getattr(bundle, "runtime_config", None), "get", None)):
		return parallelism
	# Per-call override (rare) takes precedence over the bundle-stored override.
	# Both go through parse_resources_config_for_bundle so the resolved profile name
	# is validated against resources.profiles consistently.
	if active_profile_override is not None and active_profile_override != getattr(bundle, "active_profile_override", None):
		effective_bundle = replace(bundle, active_profile_override=str(active_profile_override).strip() or None)
	else:
		effective_bundle = bundle
	try:
		resources_config = parse_resources_config_for_bundle(effective_bundle)
	except ValueError:
		# Validation errors (e.g. --profile points at an undefined profile name)
		# must propagate so the user sees the actionable message instead of
		# silently falling back to no task plan.
		raise
	except Exception:
		return parallelism
	_active_prof = get_active_profile(resources_config)
	task_config = _active_prof.task_allocation if _active_prof is not None else TaskAllocationConfig()
	if task_allocation_override:
		task_config = replace(task_config, **task_allocation_override)
	if not bool(getattr(task_config, "enabled", False)):
		return parallelism
	backend_value = str(getattr(task_config, "backend", "none") or "none")
	plan = build_task_allocation_plan(
		config=task_config,
		topology=detect_cpu_topology(logger=LOGGER),
		target_count=max(0, int(target_count)),
		resource_profile=get_active_resource_profile(resources_config),
		available_shm_gb=_available_shm_gb(),
	)
	if plan is None:
		# MPI backend (or other non-local backends) produce no local task slots, but still
		# propagate thread-env settings so _distribute_runtime_targets can apply them per rank.
		return replace(
			parallelism,
			set_thread_env=bool(task_config.set_thread_env),
			nested_thread_policy=str(task_config.nested_thread_policy or "preserve_existing"),
			use_hyperthreads=bool(task_config.use_hyperthreads),
			task_allocation_backend=backend_value,
		)
	if int(target_count) > 0 and not plan.slots:
		raise ValueError(
			"Task allocation is enabled but produced no available task slots. "
			"Lower resources.task_allocation.cpus_per_task or reserve_cpus, or disable task allocation."
		)
	effective_well_workers = max(1, min(int(getattr(parallelism, "well_workers", 1)), len(plan.slots) or 1))
	return replace(
		parallelism,
		well_workers=int(effective_well_workers),
		task_allocation_plan=plan,
		task_allocation_backend=backend_value,
	)


def _unique_resource_classes(resource_classes: list[str] | tuple[str, ...]) -> tuple[str, ...]:
	seen: set[str] = set()
	ordered: list[str] = []
	for resource_class in resource_classes:
		resolved = str(resource_class).strip()
		if not resolved or resolved in seen:
			continue
		seen.add(resolved)
		ordered.append(resolved)
	return tuple(ordered)


def _enabled_phase_names_from_config(stage_config: Any) -> tuple[str, ...]:
	phase_sequence = tuple(str(item) for item in (getattr(stage_config, "phase_sequence", None) or ()) if str(item).strip())
	if phase_sequence:
		return phase_sequence
	phases = getattr(stage_config, "phases", None)
	if phases is None:
		return ()
	return tuple(
		str(name)
		for name, value in vars(phases).items()
		if bool(getattr(value, "enabled", False))
	)


def _phase_resource_classes_for_names(stage_config: Any, phase_names: list[str] | tuple[str, ...]) -> tuple[str, ...]:
	phases = getattr(stage_config, "phases", None)
	if phases is None:
		return ()
	resource_classes: list[str] = []
	for phase_name in phase_names:
		phase_cfg = getattr(phases, str(phase_name), None)
		if phase_cfg is None or not bool(getattr(phase_cfg, "enabled", True)):
			continue
		resource_class = getattr(phase_cfg, "resource_class", None)
		if resource_class is not None and str(resource_class).strip():
			resource_classes.append(str(resource_class).strip())
	return _unique_resource_classes(tuple(resource_classes))


def _preprocess_runtime_phase_resource_classes(stage_config: Any, stage_name: str) -> tuple[str, ...]:
	canonical_stage_name = str(stage_name).strip()
	if canonical_stage_name != "preprocess" and canonical_stage_name.startswith("preprocess."):
		return _phase_resource_classes_for_names(stage_config, (canonical_stage_name.split(".", 1)[1],))
	return _phase_resource_classes_for_names(stage_config, _enabled_phase_names_from_config(stage_config))


def _spikesort_runtime_phase_resource_classes(phase_plan: list[_SpikesortRuntimePhase]) -> tuple[str, ...]:
	return _unique_resource_classes(
		tuple(
			str(phase.resource_class).strip()
			for phase in phase_plan
			if getattr(phase, "resource_class", None) is not None and str(phase.resource_class).strip()
		)
	)


def _spikesort_phase_resource_classes_from_labels(
	stage_config: Any,
	phase_labels: list[str] | tuple[str, ...],
) -> tuple[str, ...]:
	resource_attr_by_phase_label = {
		"concat_binary": "concat_binary_resource_class",
		"sort": "sort_resource_class",
		"summarize_sort": "summarize_sort_resource_class",
		"snapshot_sorter_output": "snapshot_sorter_output_resource_class",
		"restore_sorter_output": "restore_sorter_output_resource_class",
		"concat_analyzer": "concat_analyzer_resource_class",
		"bombcell_label": "bombcell_label_resource_class",
		"merge_slay": "merge_slay_resource_class",
		"bombcell_label_pass2": "bombcell_label_pass2_resource_class",
		"cleanup_concat_binary": "cleanup_concat_binary_resource_class",
		"cleanup_analyzers": "cleanup_analyzers_resource_class",
	}
	resource_classes: list[str] = []
	for phase_label in phase_labels:
		canonical_label = str(phase_label).strip().replace("SLAy", "slay")
		resource_attr = resource_attr_by_phase_label.get(canonical_label, None)
		if resource_attr is None:
			continue
		resource_class = getattr(stage_config, resource_attr, None)
		if resource_class is not None and str(resource_class).strip():
			resource_classes.append(str(resource_class).strip())
	return _unique_resource_classes(tuple(resource_classes))


def _first_resource_class(resource_classes: list[str] | tuple[str, ...]) -> str | None:
	classes = _unique_resource_classes(tuple(resource_classes))
	return classes[0] if classes else None


def _direct_target_label(target: Any) -> str:
	return f"{getattr(target, 'dataset_index', 'unknown')}:{getattr(target, 'stream_id', 'unknown')}"


def _run_direct_phase_with_resource_tracking(
	*,
	phase_name: str,
	runner: Callable[[], Any],
	resource_class: str | None,
	pipeline_thread_count: int | None,
	target: Any,
) -> Any:
	chain_result = run_phase_chain(
		phases=[
			PhaseDescriptor(
				name=str(phase_name),
				runner=runner,
				resource_class=resource_class,
				pipeline_thread_count=pipeline_thread_count,
			)
		],
		logger=LOGGER,
		target_label=_direct_target_label(target),
		resource_key_context=target,
	)
	if chain_result.result is None:
		raise RuntimeError(f"{phase_name} phase chain produced no result")
	return chain_result.result


def _reconstruct_runtime_phase_resource_classes(
	stage_config: Any,
	reconstruct_templates_config: Any | None,
	stage_name: str = "reconstruct",
) -> tuple[str, ...]:
	canonical_stage_name = str(stage_name).strip()
	if canonical_stage_name != "reconstruct" and canonical_stage_name.startswith("reconstruct."):
		direct_phase_name = canonical_stage_name.split(".", 1)[1]
		template_phase_names = {
			"resolve_sources",
			"templates_resolve_sources",
			"analyzers",
			"templates_analyzers",
			"extract_partial_templates",
			"templates_extract_partial_templates",
			"build_templates",
			"templates_build_templates",
			"compute_template_similarity",
			"templates_compute_template_similarity",
			"plot_templates_v2",
			"templates_plot_templates_v2",
			"report_templates",
			"templates_report_templates",
		}
		if direct_phase_name in template_phase_names:
			if reconstruct_templates_config is None:
				return ()
			return _phase_resource_classes_for_names(
				reconstruct_templates_config,
				(direct_phase_name.removeprefix("templates_"),),
			)
		return _phase_resource_classes_for_names(stage_config, (direct_phase_name,))
	resource_classes: list[str] = list(
		_phase_resource_classes_for_names(stage_config, _enabled_phase_names_from_config(stage_config))
	)
	if reconstruct_templates_config is not None:
		resource_classes.extend(
			_phase_resource_classes_for_names(
				reconstruct_templates_config,
				_enabled_phase_names_from_config(reconstruct_templates_config),
			)
		)
	return _unique_resource_classes(tuple(resource_classes))


def _build_stage_resource_budget_manager(
	*,
	bundle: PipelineRuntimeBundle,
	parallelism: Any,
	phase_resource_classes: list[str] | tuple[str, ...],
	target_count: int,
) -> ResourceBudgetManager | None:
	resolved_phase_resource_classes = _unique_resource_classes(tuple(phase_resource_classes))
	if not resolved_phase_resource_classes:
		return None
	# parse_resources_config_for_bundle applies the bundle's active_profile_override
	# (set by --profile / --task-profile CLI flag) so the gate sees the right
	# capacity numbers. Bypassing this is the root cause of the spikesort_full
	# gate-deadlock on gpu_sort_slots=0.
	resources_config = parse_resources_config_for_bundle(bundle)
	if resources_config.active_profile is None:
		return None
	return ResourceBudgetManager(
		resources=resources_config,
		planned_target_count=max(0, int(target_count)),
		well_workers=max(1, int(getattr(parallelism, "well_workers", 1))),
	)


def _mpi_context_for_partition(
	*,
	mpi_context: Any | None,
	parallelism: Any,
	logger: logging.Logger | None = None,
) -> Any | None:
	"""Return the MPI context only when the active task-allocation backend opts into MPI partitioning.

	Slice 11 of `nersc_shaped_local_affinity_plan.md` gates target partitioning on
	an explicit `resources.task_allocation.backend: mpi` (or `--task-backend mpi`).
	Without that opt-in, an `mpirun` launch that leaves backend at the default
	will not auto-partition. This keeps the local-affinity and none backends
	behaviorally insulated from accidental MPI env exposure.

	When MPI is detected but skipped, a single INFO line is emitted so logs
	preserve enough rank metadata to reconstruct the worker placement decision.
	"""
	if mpi_context is None:
		return None
	backend = str(getattr(parallelism, "task_allocation_backend", "none") or "none").strip().lower()
	if backend in {"mpi", "slurm"}:
		return mpi_context
	resolved_logger = logger or LOGGER
	if int(getattr(mpi_context, "size", 1)) > 1:
		resolved_logger.info(
			"MPI context detected (rank=%d size=%d) but task_allocation_backend=%s; skipping rank partitioning",
			int(getattr(mpi_context, "rank", 0)),
			int(getattr(mpi_context, "size", 1)),
			str(backend),
			extra={
				"event": "mpi_partition_skipped",
				"mpi_rank": int(getattr(mpi_context, "rank", 0)),
				"mpi_size": int(getattr(mpi_context, "size", 1)),
				"task_allocation_backend": str(backend),
			},
		)
	return None


def _distribute_runtime_targets(
	*,
	targets: list[Any],
	parallelism: Any,
	worker_fn: Callable[[Any], Any],
	stage_name: str | None = None,
	progress: PipelineProgress | None = None,
	advance_progress_on_target_complete: bool = False,
	bundle: Any | None = None,
) -> list[TargetStageResult]:
	install_pipeline_log_record_factory()
	install_maxwell_hdf5_plugin_message_filter()

	# Bind the active phase_budgets dict (from the parsed resources config) to a
	# ContextVar so per-phase code can call current_phase_budget("stage", "phase")
	# to look up its budget without threading it through every call signature.
	_phase_budgets: dict[str, Any] | None = None
	if bundle is not None and callable(getattr(getattr(bundle, "runtime_config", None), "get", None)):
		try:
			_resources_for_budgets = parse_resources_config_for_bundle(bundle)
			_phase_budgets = dict(_resources_for_budgets.phase_budgets or {})
		except Exception:
			_phase_budgets = None

	# Detect MPI context and log if active
	mpi_context = current_mpi_context()
	if mpi_context is not None:
		log_mpi_context(logger=LOGGER, context=mpi_context)

	plan = getattr(parallelism, "task_allocation_plan", None)
	task_slots = tuple(getattr(plan, "slots", ()) or ()) if plan is not None else ()
	# In MPI mode the launcher binds each rank to a CPU subset; build a synthetic slot
	# so resolve_inner_worker_count reads the rank's actual CPU count inside workers.
	if not task_slots and mpi_context is not None and int(getattr(mpi_context, "size", 1)) > 1:
		_mpi_affinity_topo = detect_cpu_topology(logger=LOGGER)
		_mpi_use_ht_slot = bool(getattr(parallelism, "use_hyperthreads", False))
		_mpi_slot_cpus = (
			_mpi_affinity_topo.visible_cpus
			if _mpi_use_ht_slot
			else tuple(core.logical_cpus[0] for core in _mpi_affinity_topo.cores if core.logical_cpus)
		)
		task_slots = (TaskSlot(
			slot_id=int(getattr(mpi_context, "rank", 0)),
			logical_cpus=_mpi_slot_cpus,
			core_ids=tuple(core.core_id for core in _mpi_affinity_topo.cores),
			package_ids=_mpi_affinity_topo.package_ids,
		),)
	apply_task_affinity = bool(
		plan is not None
		and str(getattr(plan, "backend", "none")) == "local_affinity"
		and str(getattr(plan, "bind", "none")) != "none"
	)
	_plan_set_thread_env = (
		bool(getattr(plan, "set_thread_env", False)) if plan is not None
		else bool(getattr(parallelism, "set_thread_env", False))
	)
	_plan_thread_policy = (
		str(getattr(plan, "nested_thread_policy", "preserve_existing") or "preserve_existing") if plan is not None
		else str(getattr(parallelism, "nested_thread_policy", "preserve_existing") or "preserve_existing")
	)
	_plan_cpus_per_task = int(getattr(plan, "cpus_per_task", 1)) if plan is not None else 1

	# For MPI backend (plan is None), apply thread env once per rank based on affinity-visible CPUs.
	# Each MPI rank is bound to its own CPU set; detect_cpu_topology() reflects that binding.
	# We apply vars here and skip the per-target apply_thread_env_context to avoid slot=None → count=1.
	_mpi_thread_env_applied = False
	if (
		mpi_context is not None
		and int(getattr(mpi_context, "size", 1)) > 1
		and plan is None
		and bool(_plan_set_thread_env)
		and str(_plan_thread_policy) in ("match_cpus_per_task", "force_1")
	):
		_mpi_topology = detect_cpu_topology(logger=LOGGER)
		_mpi_use_ht = bool(getattr(parallelism, "use_hyperthreads", False))
		if str(_plan_thread_policy) == "match_cpus_per_task":
			_mpi_thread_count = _mpi_topology.logical_cpu_count if _mpi_use_ht else _mpi_topology.physical_core_count
		else:
			_mpi_thread_count = 1
		for _var in _THREAD_ENV_VARS:
			os.environ[_var] = str(_mpi_thread_count)
		LOGGER.info(
			"mpi thread env applied: rank=%d policy=%s count=%d vars=%s",
			int(getattr(mpi_context, "rank", 0)),
			str(_plan_thread_policy),
			int(_mpi_thread_count),
			" ".join(f"{v}={_mpi_thread_count}" for v in _THREAD_ENV_VARS),
			extra={"event": "mpi_thread_env_applied", "mpi_thread_count": int(_mpi_thread_count)},
		)
		_mpi_thread_env_applied = True

	def _worker_thread_env_str(slot: Any | None) -> str:
		"""Compute the would-be thread env string for a slot, for logging only."""
		if not _plan_set_thread_env or _plan_thread_policy == "preserve_existing":
			return "disabled"
		if _plan_thread_policy == "force_1":
			count = 1
		elif _plan_thread_policy == "match_cpus_per_task":
			count = int(slot.cpu_count) if slot is not None else _plan_cpus_per_task
		else:
			return _plan_thread_policy
		return " ".join(f"{var}={count}" for var in _THREAD_ENV_VARS)

	def worker_with_log_context(target: Any) -> Any:
		with pipeline_log_context_for_target(target, stage=stage_name), pipeline_progress_context(progress), phase_budgets_context(_phase_budgets):
			task_slot = current_task_slot()
			alloc_meta: dict[str, Any] | None = None
			if plan is not None:
				alloc_meta = {
					"task_allocation_backend": str(getattr(plan, "backend", "none")),
					"task_slot_id": int(task_slot.slot_id) if task_slot is not None else None,
					"task_cpu_set": format_cpu_set(task_slot.logical_cpus) if task_slot is not None else None,
					"task_cpus_per_task": _plan_cpus_per_task,
					"task_allocation_tasks_per_node": int(getattr(plan, "effective_tasks_per_node", 1)),
					"task_thread_env_policy": str(_plan_thread_policy) if bool(_plan_set_thread_env) else "disabled",
				}
			with task_allocation_context(alloc_meta), task_slot_affinity_context(
				task_slot,
				enabled=bool(apply_task_affinity),
				soft_failure=True,
				logger=LOGGER,
			), apply_thread_env_context(
				task_slot,
				# Skip per-target application when MPI rank already applied thread env at startup.
				enabled=bool(_plan_set_thread_env) and not _mpi_thread_env_applied,
				policy=str(_plan_thread_policy),
				logger=LOGGER,
			):
				started = time.perf_counter()
				if task_slot is not None:
					thread_env_str = _worker_thread_env_str(task_slot)
					LOGGER.info(
						"target task allocation dataset=%s well=%s stage=%s task_slot=%d cpus=%s affinity=%s thread_env=%s",
						getattr(target, "dataset_id", "unknown"),
						getattr(target, "stream_id", "unknown"),
						str(stage_name or "unknown"),
						int(task_slot.slot_id),
						format_cpu_set(task_slot.logical_cpus),
						"enabled" if bool(apply_task_affinity) else "disabled",
						thread_env_str,
						extra={
							"event": "task_allocation_target_assigned",
							"task_slot_id": int(task_slot.slot_id),
							"task_cpu_set": format_cpu_set(task_slot.logical_cpus),
							"task_cpu_count": int(task_slot.cpu_count),
							"task_cpus_per_task": _plan_cpus_per_task,
							"task_affinity_enabled": bool(apply_task_affinity),
							"task_thread_env_policy": str(_plan_thread_policy) if bool(_plan_set_thread_env) else "disabled",
							"task_thread_env_values": thread_env_str,
						},
					)
				LOGGER.info(
					"target started dataset=%s well=%s stage=%s",
					getattr(target, "dataset_id", "unknown"),
					getattr(target, "stream_id", "unknown"),
					str(stage_name or "unknown"),
					extra={"event": "well_started"},
				)
				try:
					result = worker_fn(target)
				except Exception:
					LOGGER.exception(
						"target failed dataset=%s well=%s stage=%s",
						getattr(target, "dataset_id", "unknown"),
						getattr(target, "stream_id", "unknown"),
						str(stage_name or "unknown"),
						extra={"event": "well_failed", "elapsed_s": float(max(0.0, time.perf_counter() - started))},
					)
					raise
				LOGGER.info(
					"target completed dataset=%s well=%s stage=%s",
					getattr(target, "dataset_id", "unknown"),
					getattr(target, "stream_id", "unknown"),
					str(stage_name or "unknown"),
					extra={"event": "well_completed", "elapsed_s": float(max(0.0, time.perf_counter() - started))},
				)
				return result

	def _on_target_complete(_result: TargetStageResult) -> None:
		if progress is not None and bool(advance_progress_on_target_complete):
			progress.update(1)

	progress_context = progress if progress is not None else nullcontext()
	mpi_partition_context = _mpi_context_for_partition(
		mpi_context=mpi_context, parallelism=parallelism, logger=LOGGER
	)
	with progress_context:
		return distribute_targets(
			targets=targets,
			well_workers=int(parallelism.well_workers),
			worker_fn=worker_with_log_context,
			max_simultaneous_well_reads_per_dataset=getattr(
				parallelism,
				"max_simultaneous_well_reads_per_dataset",
				None,
			),
			on_target_complete=_on_target_complete,
			task_slots=task_slots,
			mpi_context=mpi_partition_context,
		)


def _reconstruct_unit_progress(stage_name: str) -> PipelineProgress | None:
	if str(stage_name) not in {"reconstruct", "reconstruct.axon_velocity_gtrs"}:
		return None
	return PipelineProgress(ProgressSpec(label=f"{stage_name} units", total=0, unit="unit"))


def _stage_config_with_runtime_n_jobs(stage_config: Any, *, unit_workers: int) -> Any:
	if getattr(stage_config, "n_jobs", None) is not None:
		return stage_config
	runtime_n_jobs = max(1, int(unit_workers))
	if getattr(stage_config, "__dataclass_fields__", None) is not None:
		try:
			return replace(stage_config, n_jobs=runtime_n_jobs)
		except Exception:
			return stage_config
	try:
		stage_config_copy = copy.copy(stage_config)
		setattr(stage_config_copy, "n_jobs", runtime_n_jobs)
		return stage_config_copy
	except Exception:
		return stage_config


def _runtime_n_jobs_source(stage_config: Any) -> str:
	return "configured" if getattr(stage_config, "n_jobs", None) is not None else "derived"


def _runtime_n_jobs_from_stage_config(stage_config: Any, *, fallback_n_jobs: int) -> int:
	raw_n_jobs = getattr(stage_config, "n_jobs", None)
	if raw_n_jobs is None:
		raw_n_jobs = fallback_n_jobs
	return max(1, int(raw_n_jobs))


def _with_preprocess_runtime_worker_allocation(
	inputs: Any,
	*,
	parallelism: Any,
	n_jobs_source: str,
) -> Any:
	try:
		return replace(
			inputs,
			runtime_well_workers=max(1, int(parallelism.well_workers)),
			runtime_n_jobs_source=str(n_jobs_source),
		)
	except Exception:
		return inputs


def _target_log_label(target: Any) -> str:
	return f"{getattr(target, 'dataset_index', 'unknown')}:{getattr(target, 'stream_id', 'unknown')}"


def _log_runtime_stage_topology(*, stage_name: str, targets: list[Any], parallelism: Any) -> None:
	LOGGER.info("Starting stage: %s", str(stage_name), extra={"event": "stage_started"})

	# Log MPI context if active
	mpi_context = current_mpi_context()
	if mpi_context is not None and int(mpi_context.size) > 1:
		log_mpi_context(logger=LOGGER, context=mpi_context)

	LOGGER.info(
		"Execution topology: stage_global_order=true, well_local_phase_sequence=true",
		extra={
			"event": "stage_topology",
			"stage_global_order": True,
			"well_local_phase_sequence": True,
		},
	)
	LOGGER.info(
		"Selected wells: %d",
		len(targets),
		extra={
			"event": "stage_topology",
			"selected_wells": int(len(targets)),
		},
	)
	LOGGER.info(
		"well_workers=%d",
		int(parallelism.well_workers),
		extra={
			"event": "stage_topology",
			"well_workers": int(parallelism.well_workers),
		},
	)
	plan = getattr(parallelism, "task_allocation_plan", None)
	if plan is not None:
		topology = getattr(plan, "topology", None)
		visible_cpus_str = format_cpu_set(getattr(topology, "visible_cpus", ())) if topology is not None else "unknown"
		physical_cores = int(getattr(topology, "physical_core_count", 0)) if topology is not None else 0
		LOGGER.info(
			"Task allocation backend=%s task_unit=%s visible_cpus=%s physical_cores=%d cpus_per_task=%d tasks_per_node=%d bind=%s use_hyperthreads=%s slots=%d",
			str(plan.backend),
			str(getattr(plan, "task_unit", "well")),
			visible_cpus_str,
			physical_cores,
			int(plan.cpus_per_task),
			int(plan.effective_tasks_per_node),
			str(plan.bind),
			str(plan.use_hyperthreads).lower(),
			int(len(plan.slots)),
			extra={
				"event": "task_allocation_plan",
				"task_allocation_backend": str(plan.backend),
				"task_allocation_task_unit": str(getattr(plan, "task_unit", "well")),
				"task_allocation_visible_cpus": visible_cpus_str,
				"task_allocation_physical_cores": physical_cores,
				"task_allocation_bind": str(plan.bind),
				"task_allocation_use_hyperthreads": bool(plan.use_hyperthreads),
				"task_allocation_cpus_per_task": int(plan.cpus_per_task),
				"task_allocation_tasks_per_node": int(plan.effective_tasks_per_node),
				"task_allocation_slot_count": int(len(plan.slots)),
				"task_allocation_cpu_capacity_tasks": int(plan.cpu_capacity_tasks),
			},
		)
		if str(getattr(plan, "backend", "none")) == "local_affinity":
			readiness = probe_container_readiness()
			readiness_fields = readiness.to_dict()
			LOGGER.info(
				"Container affinity readiness: affinity_api=%s sysfs=%s shm_gb=%s visible_cpus=%d warnings=%d",
				str(readiness.affinity_api_available),
				str(readiness.sysfs_topology_readable),
				f"{readiness.shm_available_gb:.2f}" if readiness.shm_available_gb is not None else "unavailable",
				int(readiness.visible_cpu_count),
				len(readiness.warnings),
				extra={"event": "container_affinity_readiness", **readiness_fields},
			)
			for warning in readiness.warnings:
				LOGGER.warning(
					"Container affinity readiness warning: %s",
					str(warning),
					extra={"event": "container_affinity_readiness_warning"},
				)


def _log_spikesort_phase_worker_allocation(
	*,
	phase_name: str,
	target: Any,
	parallelism: Any,
	n_jobs: int,
	n_jobs_source: str,
) -> None:
	LOGGER.info(
		"Spikesort phase worker allocation stage=spikesort phase=%s target=%s well_workers=%d n_jobs=%d n_jobs_source=%s",
		str(phase_name),
		_target_log_label(target),
		int(parallelism.well_workers),
		max(1, int(n_jobs)),
		str(n_jobs_source),
	)


def _run_spikesort_phase_with_optional_sort_gate(
	*,
	phase_label: str,
	target: Any,
	runner: Callable[[], Any],
	sort_phase_gate: Any | None,
) -> Any:
	_ = (phase_label, target, sort_phase_gate)
	return runner()


def _coerce_bool_or_none(value: Any) -> bool | None:
	if value is None:
		return None
	if isinstance(value, bool):
		return value
	token = str(value).strip().lower()
	if token in {"1", "true", "yes", "on"}:
		return True
	if token in {"0", "false", "no", "off"}:
		return False
	return None


def _with_debug_limit_overrides(
	stage_config: Any,
	*,
	limit_segments_override: int | None = None,
	limit_datasets_override: int | None = None,
	limit_wells_per_dataset_override: int | None = None,
) -> Any:
	replace_kwargs: dict[str, Any] = {}
	field_names = set(getattr(stage_config, "__dataclass_fields__", {}) or {})
	target_limit_replaced = False
	if limit_segments_override is not None:
		segment_limit = int(limit_segments_override)
		if not field_names:
			replace_kwargs["debug_limit_segments_per_well"] = segment_limit
		else:
			segment_limit_replaced = False
			if "debug_limit_segments_per_well" in field_names:
				replace_kwargs["debug_limit_segments_per_well"] = segment_limit
				segment_limit_replaced = True
			if "concat_binary_debug_limit_segments_per_well" in field_names:
				replace_kwargs["concat_binary_debug_limit_segments_per_well"] = segment_limit
				segment_limit_replaced = True
			if not segment_limit_replaced and "limit_segments" in field_names:
				replace_kwargs["limit_segments"] = segment_limit
	if limit_datasets_override is not None and (
		not field_names or "debug_limit_datasets" in field_names
	):
		replace_kwargs["debug_limit_datasets"] = int(limit_datasets_override)
		target_limit_replaced = True
	if limit_wells_per_dataset_override is not None and (
		not field_names or "debug_limit_wells_per_dataset" in field_names
	):
		replace_kwargs["debug_limit_wells_per_dataset"] = int(limit_wells_per_dataset_override)
		target_limit_replaced = True
	if target_limit_replaced and "debug_mode_enabled" in field_names:
		replace_kwargs["debug_mode_enabled"] = True
	if not replace_kwargs:
		return stage_config
	if field_names:
		return replace(stage_config, **replace_kwargs)
	updated = copy.copy(stage_config)
	for field_name, value in replace_kwargs.items():
		setattr(updated, field_name, value)
	return updated


def _debug_target_limit_kwargs(stage_config: Any) -> dict[str, int | None]:
	return {
		"limit_datasets": getattr(stage_config, "debug_limit_datasets", None),
		"limit_wells": getattr(stage_config, "debug_limit_wells", None),
		"limit_wells_per_dataset": getattr(stage_config, "debug_limit_wells_per_dataset", None),
	}


def _select_execution_targets_with_debug_limits(
	*,
	bundle: PipelineRuntimeBundle,
	stage_name: str,
	stage_config: Any,
	target_datasets: list[int] | None = None,
) -> list[Any]:
	limit_kwargs = _debug_target_limit_kwargs(stage_config)
	if target_datasets is not None or any(value is not None for value in limit_kwargs.values()):
		LOGGER.info(
			"%s: applying target debug limits before scratch materialization datasets=%s target_datasets=%s wells=%s wells_per_dataset=%s",
			str(stage_name),
			limit_kwargs.get("limit_datasets", None),
			target_datasets,
			limit_kwargs.get("limit_wells", None),
			limit_kwargs.get("limit_wells_per_dataset", None),
		)
	select_kwargs = {key: value for key, value in limit_kwargs.items() if value is not None}
	if target_datasets is not None:
		select_kwargs["target_datasets"] = list(target_datasets)
	return select_execution_targets(bundle=bundle, **select_kwargs)


def _apply_preprocess_debug_target_limits(
	*,
	stage_name: str,
	targets: list[Any],
	limit_datasets: Any,
	limit_wells: Any,
	limit_wells_per_dataset: Any = None,
) -> list[Any]:
	limited_targets = list(targets)
	if limit_datasets is not None:
		dataset_limit = max(1, int(limit_datasets))
		selected_dataset_indices: list[int] = []
		seen_dataset_indices: set[int] = set()
		for target in limited_targets:
			try:
				dataset_index = int(getattr(target, "dataset_index", -1))
			except Exception:
				continue
			if dataset_index in seen_dataset_indices:
				continue
			seen_dataset_indices.add(dataset_index)
			selected_dataset_indices.append(dataset_index)
			if len(selected_dataset_indices) >= dataset_limit:
				break
		selected_dataset_index_set = set(selected_dataset_indices)
		original_count = len(limited_targets)
		limited_targets = [
			item
			for item in limited_targets
			if int(getattr(item, "dataset_index", -1)) in selected_dataset_index_set
		]
		if len(limited_targets) < original_count:
			LOGGER.info(
				"Applying %s debug dataset limit: %d -> %d target(s) dataset_indices=%s",
				str(stage_name),
				original_count,
				len(limited_targets),
				selected_dataset_indices,
			)

	if limit_wells_per_dataset is not None:
		well_limit_per_dataset = max(1, int(limit_wells_per_dataset))
		original_count = len(limited_targets)
		selected_by_dataset: dict[int, int] = {}
		per_dataset_limited_targets: list[Any] = []
		for target in limited_targets:
			try:
				dataset_index = int(getattr(target, "dataset_index", -1))
			except Exception:
				dataset_index = -1
			current_count = int(selected_by_dataset.get(dataset_index, 0))
			if current_count >= well_limit_per_dataset:
				continue
			per_dataset_limited_targets.append(target)
			selected_by_dataset[dataset_index] = current_count + 1
		limited_targets = per_dataset_limited_targets
		if len(limited_targets) < original_count:
			LOGGER.info(
				"Applying %s debug wells-per-dataset limit: %d -> %d target(s) limit_per_dataset=%d dataset_counts=%s",
				str(stage_name),
				original_count,
				len(limited_targets),
				well_limit_per_dataset,
				dict(sorted(selected_by_dataset.items(), key=lambda kv: kv[0])),
			)

	if limit_wells is not None and len(limited_targets) > int(limit_wells):
		well_limit = max(1, int(limit_wells))
		LOGGER.info(
			"Applying %s debug well limit: %d -> %d target(s)",
			str(stage_name),
			len(limited_targets),
			well_limit,
		)
		limited_targets = list(limited_targets[:well_limit])

	return limited_targets


def _apply_preprocess_stage_debug_limits(
	*,
	stage_name: str,
	stage_config: Any,
	targets: list[Any],
) -> list[Any]:
	return _apply_preprocess_debug_target_limits(
		stage_name=stage_name,
		targets=list(targets),
		limit_datasets=getattr(stage_config, "debug_limit_datasets", None),
		limit_wells=getattr(stage_config, "debug_limit_wells", None),
		limit_wells_per_dataset=getattr(stage_config, "debug_limit_wells_per_dataset", None),
	)


def _apply_preprocess_substage_phase_debug_limits(
	*,
	stage_name: str,
	stage_config: Any,
	targets: list[Any],
) -> list[Any]:
	phase_attr_by_stage_name = {
		"preprocess.save_rec_metadata": "save_rec_metadata",
		"preprocess.plot_raster_threshold": "plot_raster_threshold",
	}
	phase_attr = phase_attr_by_stage_name.get(str(stage_name).strip())
	phase_cfg = None if phase_attr is None else getattr(getattr(stage_config, "phases", None), str(phase_attr), None)
	if phase_cfg is not None and bool(getattr(phase_cfg, "debug_mode_enabled", False)):
		_warn_ignored_phase_debug_limits(stage_name=stage_name, phase_label=str(phase_attr))
	return list(targets)


def _apply_spikesort_sort_debug_limits(
	*,
	stage_name: str,
	stage_config: Any,
	targets: list[Any],
) -> list[Any]:
	if bool(getattr(stage_config, "sort_debug_mode_enabled", False)):
		_warn_ignored_phase_debug_limits(stage_name=stage_name, phase_label="sort")
	return list(targets)


def _apply_spikesort_debug_target_limits(
	*,
	stage_name: str,
	targets: list[Any],
	limit_datasets: Any,
	limit_wells: Any,
	limit_wells_per_dataset: Any = None,
	phase_label: str | None = None,
) -> list[Any]:
	limited_targets = list(targets)
	debug_label = str(stage_name)
	if phase_label is not None:
		debug_label = f"{debug_label} {phase_label}"

	if limit_datasets is not None:
		dataset_limit = max(1, int(limit_datasets))
		selected_dataset_indices: list[int] = []
		seen_dataset_indices: set[int] = set()
		for target in limited_targets:
			try:
				dataset_index = int(getattr(target, "dataset_index", -1))
			except Exception:
				continue
			if dataset_index in seen_dataset_indices:
				continue
			seen_dataset_indices.add(dataset_index)
			selected_dataset_indices.append(dataset_index)
			if len(selected_dataset_indices) >= dataset_limit:
				break
		selected_dataset_index_set = set(selected_dataset_indices)
		original_count = len(limited_targets)
		limited_targets = [
			item
			for item in limited_targets
			if int(getattr(item, "dataset_index", -1)) in selected_dataset_index_set
		]
		if len(limited_targets) < original_count:
			LOGGER.info(
				"Applying %s debug dataset limit: %d -> %d target(s) dataset_indices=%s",
				debug_label,
				original_count,
				len(limited_targets),
				selected_dataset_indices,
			)

	if limit_wells_per_dataset is not None:
		well_limit_per_dataset = max(1, int(limit_wells_per_dataset))
		original_count = len(limited_targets)
		selected_by_dataset: dict[int, int] = {}
		per_dataset_limited_targets: list[Any] = []
		for target in limited_targets:
			try:
				dataset_index = int(getattr(target, "dataset_index", -1))
			except Exception:
				dataset_index = -1
			current_count = int(selected_by_dataset.get(dataset_index, 0))
			if current_count >= well_limit_per_dataset:
				continue
			per_dataset_limited_targets.append(target)
			selected_by_dataset[dataset_index] = current_count + 1
		limited_targets = per_dataset_limited_targets
		if len(limited_targets) < original_count:
			LOGGER.info(
				"Applying %s debug wells-per-dataset limit: %d -> %d target(s) limit_per_dataset=%d dataset_counts=%s",
				debug_label,
				original_count,
				len(limited_targets),
				well_limit_per_dataset,
				dict(sorted(selected_by_dataset.items(), key=lambda kv: kv[0])),
			)

	if limit_wells is not None and len(limited_targets) > int(limit_wells):
		well_limit = max(1, int(limit_wells))
		LOGGER.info(
			"Applying %s debug well limit: %d -> %d target(s)",
			debug_label,
			len(limited_targets),
			well_limit,
		)
		limited_targets = list(limited_targets[:well_limit])

	return limited_targets


def _apply_spikesort_stage_debug_limits(
	*,
	stage_name: str,
	stage_config: Any,
	targets: list[Any],
) -> list[Any]:
	return _apply_spikesort_debug_target_limits(
		stage_name=stage_name,
		targets=list(targets),
		limit_datasets=getattr(stage_config, "debug_limit_datasets", None),
		limit_wells=getattr(stage_config, "debug_limit_wells", None),
		limit_wells_per_dataset=getattr(stage_config, "debug_limit_wells_per_dataset", None),
	)


def _apply_spikesort_phase_debug_limits(
	*,
	stage_name: str,
	stage_config: Any,
	targets: list[Any],
	phase_label: str,
	enabled_attr: str,
	limit_datasets_attr: str,
	limit_wells_attr: str,
	limit_wells_per_dataset_attr: str | None = None,
) -> list[Any]:
	_ = (
		stage_name,
		phase_label,
		limit_datasets_attr,
		limit_wells_attr,
		limit_wells_per_dataset_attr,
	)
	if bool(getattr(stage_config, str(enabled_attr), False)):
		_warn_ignored_phase_debug_limits(stage_name=stage_name, phase_label=str(phase_label))
	return list(targets)


def _read_bool_setting(config: Any, *, path: str) -> bool | None:
	if config is None:
		return None

	getter_bool = getattr(config, "get_bool", None)
	if callable(getter_bool):
		try:
			return getter_bool(path, default=None)
		except TypeError:
			try:
				return getter_bool(path)
			except Exception:
				pass
		except Exception as exc:
			LOGGER.warning("Invalid boolean runtime setting path=%s error=%s", path, exc)

	if isinstance(config, dict):
		node: Any = config
		for part in str(path).split("."):
			if not isinstance(node, dict) or part not in node:
				return None
			node = node[part]
		return _coerce_bool_or_none(node)

	getter = getattr(config, "get", None)
	if callable(getter):
		try:
			value = getter(path, None)
		except TypeError:
			try:
				value = getter(path)
			except Exception:
				return None
		except Exception:
			return None
		return _coerce_bool_or_none(value)

	return None


def _resolve_publish_policy(*, runtime_config: Any, data_config: Any) -> PublishPolicy:
	publish_paths = (
		"pipeline.publish_outputs",
		"paths.publish_outputs",
		"publish_outputs",
	)
	wipe_paths = (
		"pipeline.wipe_scratch_roots",
		"paths.wipe_scratch_roots",
		"wipe_scratch_roots",
	)

	publish_outputs: bool | None = None
	for path in publish_paths:
		publish_outputs = _read_bool_setting(data_config, path=path)
		if publish_outputs is not None:
			break
	for path in publish_paths:
		if publish_outputs is not None:
			break
		publish_outputs = _read_bool_setting(runtime_config, path=path)
		if publish_outputs is not None:
			break
	if publish_outputs is None:
		publish_outputs = True

	wipe_scratch_roots: bool | None = None
	for path in wipe_paths:
		wipe_scratch_roots = _read_bool_setting(data_config, path=path)
		if wipe_scratch_roots is not None:
			break
	for path in wipe_paths:
		if wipe_scratch_roots is not None:
			break
		wipe_scratch_roots = _read_bool_setting(runtime_config, path=path)
		if wipe_scratch_roots is not None:
			break
	if wipe_scratch_roots is None:
		wipe_scratch_roots = False

	if not bool(publish_outputs) and bool(wipe_scratch_roots):
		LOGGER.info(
			"Publish policy requested wipe_scratch_roots=true while publish_outputs=false; forcing wipe_scratch_roots=false"
		)
		wipe_scratch_roots = False

	return PublishPolicy(
		publish_outputs=bool(publish_outputs),
		wipe_scratch_roots=bool(wipe_scratch_roots),
	)


def _log_publish_policy(*, stage_name: str, policy: PublishPolicy) -> None:
	LOGGER.info(
		"Publish policy stage=%s publish_outputs=%s wipe_scratch_roots=%s",
		stage_name,
		bool(policy.publish_outputs),
		bool(policy.wipe_scratch_roots),
	)


def _target_ids(target: Any) -> tuple[str, str]:
	return str(getattr(target, "dataset_id", "unknown")), str(getattr(target, "stream_id", "unknown"))


def _publish_stage_output(
	*,
	stage_name: str,
	target: Any,
	path: Path,
	active_root: Path,
	final_root: Path,
	policy: PublishPolicy,
) -> bool:
	dataset_id, stream_id = _target_ids(target)
	source_path = Path(path).expanduser().resolve()
	if not bool(policy.publish_outputs):
		LOGGER.info(
			"Publish skipped stage=%s dataset_id=%s stream_id=%s reason=publish_outputs_disabled source=%s final_root=%s",
			stage_name,
			dataset_id,
			stream_id,
			source_path,
			final_root,
		)
		LOGGER.info(
			"Scratch wipe skipped stage=%s dataset_id=%s stream_id=%s reason=publish_outputs_disabled",
			stage_name,
			dataset_id,
			stream_id,
		)
		return False

	mode = policy.publish_mode()
	LOGGER.info(
		"Publish start stage=%s dataset_id=%s stream_id=%s mode=%s source=%s final_root=%s",
		stage_name,
		dataset_id,
		stream_id,
		mode,
		source_path,
		final_root,
	)
	publish_path_to_final(path=source_path, active_root=active_root, final_root=final_root, mode=mode)
	LOGGER.info(
		"Publish complete stage=%s dataset_id=%s stream_id=%s mode=%s source=%s final_root=%s",
		stage_name,
		dataset_id,
		stream_id,
		mode,
		source_path,
		final_root,
	)

	if bool(policy.wipe_scratch_roots):
		LOGGER.info(
			"Scratch wipe complete stage=%s dataset_id=%s stream_id=%s source=%s",
			stage_name,
			dataset_id,
			stream_id,
			source_path,
		)
	else:
		LOGGER.info(
			"Scratch wipe skipped stage=%s dataset_id=%s stream_id=%s reason=wipe_scratch_roots_disabled",
			stage_name,
			dataset_id,
			stream_id,
		)

	return True


def _publish_roots_for_target(target) -> tuple[Path, Path] | None:
	active_root = getattr(target, "scratch_output_root", None)
	if active_root is None:
		return None
	final_root = getattr(target, "final_output_root", None) or getattr(target, "mea_output_root", None)
	if final_root is None:
		return None
	active_path = Path(active_root).expanduser().resolve()
	final_path = Path(final_root).expanduser().resolve()
	if active_path == final_path:
		return None
	return active_path, final_path


def _remap_stage_path(path: Path, *, active_root: Path, final_root: Path) -> Path:
	return Path(remap_path_string_to_final(raw=path, active_root=active_root, final_root=final_root)).expanduser().resolve()


def _remap_output_map(outputs: dict[str, str], *, active_root: Path, final_root: Path) -> dict[str, str]:
	out: dict[str, str] = {}
	for key, value in outputs.items():
		out[str(key)] = remap_path_string_to_final(raw=value, active_root=active_root, final_root=final_root)
	return out


def _publish_preprocess_target_result(item: TargetStageResult, *, policy: PublishPolicy | None = None) -> TargetStageResult:
	publish_policy = policy or PublishPolicy()
	if item.status != "ok" or not isinstance(item.result, PreprocessResult):
		return item
	roots = _publish_roots_for_target(item.target)
	if roots is None:
		return item
	active_root, final_root = roots
	result = item.result
	published = _publish_stage_output(
		stage_name="preprocess",
		target=item.target,
		path=result.preprocess_out_dir,
		active_root=active_root,
		final_root=final_root,
		policy=publish_policy,
	)
	if not published:
		return item
	updated = PreprocessResult(
		well_out_dir=_remap_stage_path(result.well_out_dir, active_root=active_root, final_root=final_root),
		preprocess_out_dir=_remap_stage_path(result.preprocess_out_dir, active_root=active_root, final_root=final_root),
		summary_json=_remap_stage_path(result.summary_json, active_root=active_root, final_root=final_root),
		outputs=_remap_output_map(result.outputs, active_root=active_root, final_root=final_root),
	)
	return TargetStageResult(target=item.target, status=item.status, result=updated, error=item.error)


def _publish_spikesort_target_result(item: TargetStageResult, *, policy: PublishPolicy | None = None) -> TargetStageResult:
	publish_policy = policy or PublishPolicy()
	if item.status != "ok" or not isinstance(item.result, SpikesortResult):
		return item
	roots = _publish_roots_for_target(item.target)
	if roots is None:
		return item
	active_root, final_root = roots
	result = item.result
	published = _publish_stage_output(
		stage_name="spikesort",
		target=item.target,
		path=result.spikesort_out_dir,
		active_root=active_root,
		final_root=final_root,
		policy=publish_policy,
	)
	if not published:
		return item
	updated = SpikesortResult(
		well_out_dir=_remap_stage_path(result.well_out_dir, active_root=active_root, final_root=final_root),
		spikesort_out_dir=_remap_stage_path(result.spikesort_out_dir, active_root=active_root, final_root=final_root),
		summary_json=_remap_stage_path(result.summary_json, active_root=active_root, final_root=final_root),
		outputs=_remap_output_map(result.outputs, active_root=active_root, final_root=final_root),
	)
	return TargetStageResult(target=item.target, status=item.status, result=updated, error=item.error)


def _publish_spikesort_merge_target_result(item: TargetStageResult, *, policy: PublishPolicy | None = None) -> TargetStageResult:
	publish_policy = policy or PublishPolicy()
	if item.status != "ok" or not isinstance(item.result, SpikesortMergeResult):
		return item
	roots = _publish_roots_for_target(item.target)
	if roots is None:
		return item
	active_root, final_root = roots
	result = item.result
	published = _publish_stage_output(
		stage_name="spikesort.merge",
		target=item.target,
		path=result.merge_out_dir,
		active_root=active_root,
		final_root=final_root,
		policy=publish_policy,
	)
	if not published:
		return item
	updated = SpikesortMergeResult(
		well_out_dir=_remap_stage_path(result.well_out_dir, active_root=active_root, final_root=final_root),
		merge_out_dir=_remap_stage_path(result.merge_out_dir, active_root=active_root, final_root=final_root),
		summary_json=_remap_stage_path(result.summary_json, active_root=active_root, final_root=final_root),
		outputs=_remap_output_map(result.outputs, active_root=active_root, final_root=final_root),
	)
	return TargetStageResult(target=item.target, status=item.status, result=updated, error=item.error)


def _publish_spikesort_bombcell_target_result(item: TargetStageResult, *, policy: PublishPolicy | None = None) -> TargetStageResult:
	publish_policy = policy or PublishPolicy()
	if item.status != "ok" or not isinstance(item.result, SpikesortBombcellResult):
		return item
	roots = _publish_roots_for_target(item.target)
	if roots is None:
		return item
	active_root, final_root = roots
	result = item.result
	published = _publish_stage_output(
		stage_name="spikesort.bombcell_label",
		target=item.target,
		path=result.bombcell_out_dir,
		active_root=active_root,
		final_root=final_root,
		policy=publish_policy,
	)
	if not published:
		return item
	updated = SpikesortBombcellResult(
		well_out_dir=_remap_stage_path(result.well_out_dir, active_root=active_root, final_root=final_root),
		bombcell_out_dir=_remap_stage_path(result.bombcell_out_dir, active_root=active_root, final_root=final_root),
		summary_json=(
			_remap_stage_path(result.summary_json, active_root=active_root, final_root=final_root)
			if result.summary_json is not None
			else None
		),
		outputs=_remap_output_map(result.outputs, active_root=active_root, final_root=final_root),
	)
	return TargetStageResult(target=item.target, status=item.status, result=updated, error=item.error)


def _publish_spikesort_chain_target_result(
	item: TargetStageResult,
	*,
	policy: PublishPolicy | None = None,
	output_rel_root: str = "spikesort_outputs",
) -> TargetStageResult:
	publish_policy = policy or PublishPolicy()
	if item.status != "ok" or not isinstance(
		item.result,
		(SpikesortResult, SpikesortMergeResult, SpikesortBombcellResult),
	):
		return item
	roots = _publish_roots_for_target(item.target)
	if roots is None:
		return item
	active_root, final_root = roots
	result = item.result
	well_out_dir = Path(getattr(result, "well_out_dir"))
	stage_output_root = well_out_dir / (str(output_rel_root).strip().lstrip("/") or "spikesort_outputs")
	published = _publish_stage_output(
		stage_name="spikesort",
		target=item.target,
		path=stage_output_root,
		active_root=active_root,
		final_root=final_root,
		policy=publish_policy,
	)
	if not published:
		return item
	if isinstance(result, SpikesortResult):
		updated = SpikesortResult(
			well_out_dir=_remap_stage_path(result.well_out_dir, active_root=active_root, final_root=final_root),
			spikesort_out_dir=_remap_stage_path(result.spikesort_out_dir, active_root=active_root, final_root=final_root),
			summary_json=_remap_stage_path(result.summary_json, active_root=active_root, final_root=final_root),
			outputs=_remap_output_map(result.outputs, active_root=active_root, final_root=final_root),
		)
	elif isinstance(result, SpikesortMergeResult):
		updated = SpikesortMergeResult(
			well_out_dir=_remap_stage_path(result.well_out_dir, active_root=active_root, final_root=final_root),
			merge_out_dir=_remap_stage_path(result.merge_out_dir, active_root=active_root, final_root=final_root),
			summary_json=_remap_stage_path(result.summary_json, active_root=active_root, final_root=final_root),
			outputs=_remap_output_map(result.outputs, active_root=active_root, final_root=final_root),
		)
	else:
		updated = SpikesortBombcellResult(
			well_out_dir=_remap_stage_path(result.well_out_dir, active_root=active_root, final_root=final_root),
			bombcell_out_dir=_remap_stage_path(result.bombcell_out_dir, active_root=active_root, final_root=final_root),
			summary_json=(
				_remap_stage_path(result.summary_json, active_root=active_root, final_root=final_root)
				if result.summary_json is not None
				else None
			),
			outputs=_remap_output_map(result.outputs, active_root=active_root, final_root=final_root),
		)
	return TargetStageResult(target=item.target, status=item.status, result=updated, error=item.error)


def _publish_reconstruct_target_result(item: TargetStageResult, *, policy: PublishPolicy | None = None) -> TargetStageResult:
	publish_policy = policy or PublishPolicy()
	if item.status != "ok" or not isinstance(item.result, ReconstructionResult):
		return item
	roots = _publish_roots_for_target(item.target)
	if roots is None:
		return item
	active_root, final_root = roots
	result = item.result
	published = _publish_stage_output(
		stage_name="reconstruct",
		target=item.target,
		path=result.reconstruction_out_dir,
		active_root=active_root,
		final_root=final_root,
		policy=publish_policy,
	)
	if not published:
		return item
	updated_units = [
		UnitReconstructionResult(
			unit_id=unit.unit_id,
			status=unit.status,
			outputs=_remap_output_map(unit.outputs, active_root=active_root, final_root=final_root),
			error=unit.error,
		)
		for unit in result.units
	]
	updated = ReconstructionResult(
		well_out_dir=_remap_stage_path(result.well_out_dir, active_root=active_root, final_root=final_root),
		reconstruction_out_dir=_remap_stage_path(result.reconstruction_out_dir, active_root=active_root, final_root=final_root),
		summary_json=_remap_stage_path(result.summary_json, active_root=active_root, final_root=final_root),
		units=updated_units,
	)
	return TargetStageResult(target=item.target, status=item.status, result=updated, error=item.error)


@dataclass(frozen=True)
class StageAllocationPreview:
	stage: str
	target_count: int
	target_labels: tuple[str, ...]
	phase_resource_classes: tuple[str, ...]
	parallelism: Any
	allocation_backend: str | None = None
	mpi_rank: int | None = None
	mpi_size: int | None = None


def _allocation_target_labels(targets: list[Any], *, limit: int = 8) -> tuple[str, ...]:
	labels: list[str] = []
	for target in list(targets)[: max(0, int(limit))]:
		labels.append(f"{getattr(target, 'dataset_index', 'unknown')}:{getattr(target, 'stream_id', 'unknown')}")
	return tuple(labels)


def _resolve_preview_allocation_backend(
	*,
	bundle: PipelineRuntimeBundle,
	parallelism: Any,
	task_allocation_override: dict[str, Any] | None,
) -> str:
	if task_allocation_override and task_allocation_override.get("backend") is not None:
		return str(task_allocation_override.get("backend") or "none")
	plan = getattr(parallelism, "task_allocation_plan", None)
	if plan is not None:
		return str(getattr(plan, "backend", "local_affinity") or "local_affinity")
	try:
		resources_config = parse_resources_config_for_bundle(bundle)
		_prof = get_active_profile(resources_config)
		task_config = _prof.task_allocation if _prof is not None else TaskAllocationConfig()
		if not bool(getattr(task_config, "enabled", False)):
			return "none"
		return str(getattr(task_config, "backend", "none") or "none")
	except Exception:
		return "none"


def _partition_targets_for_preview(
	*,
	targets: list[Any],
	allocation_backend: str,
) -> tuple[list[Any], int | None, int | None]:
	if str(allocation_backend).strip().lower() != "mpi":
		return list(targets), None, None
	mpi_context = current_mpi_context()
	if mpi_context is None:
		return list(targets), None, None
	partitioned = partition_targets_by_mpi_rank(targets=list(targets), mpi_context=mpi_context)
	return partitioned, int(mpi_context.rank), int(mpi_context.size)


def _build_preprocess_allocation_preview(
	*,
	config_path: str,
	stage_name: str,
	target_datasets_override: list[int] | None,
	limit_segments_override: int | None,
	limit_datasets_override: int | None,
	limit_wells_per_dataset_override: int | None,
	force_restart_override: bool | None,
	replot_override: bool | None,
	task_allocation_override: dict[str, Any] | None = None,
) -> StageAllocationPreview:
	bundle = load_pipeline_runtime_bundle(config_path=config_path)
	stage_config = parse_preprocess_stage_config(
		runtime_config=bundle.runtime_config,
		force_restart_override=force_restart_override,
		replot_override=replot_override,
	)
	stage_config = _with_debug_limit_overrides(
		stage_config,
		limit_segments_override=limit_segments_override,
		limit_datasets_override=limit_datasets_override,
		limit_wells_per_dataset_override=limit_wells_per_dataset_override,
	)
	targets = _select_preprocess_execution_targets(
		bundle=bundle,
		stage_config=stage_config,
		materialize_scratch_inputs=False,
		target_datasets=target_datasets_override,
	)
	targets = _apply_preprocess_stage_debug_limits(
		stage_name="preprocess",
		stage_config=stage_config,
		targets=list(targets),
	)
	if str(stage_name).strip() != "preprocess":
		targets = _apply_preprocess_substage_phase_debug_limits(
			stage_name=stage_name,
			stage_config=stage_config,
			targets=list(targets),
		)
	phase_resource_classes = _preprocess_runtime_phase_resource_classes(stage_config, stage_name)
	parallelism = _resolve_runtime_stage_parallelism(
		bundle=bundle,
		stage_name="preprocess",
		target_count=len(targets),
		targets=targets,
		phase_resource_classes=phase_resource_classes,
		task_allocation_override=task_allocation_override,
	)
	allocation_backend = _resolve_preview_allocation_backend(
		bundle=bundle,
		parallelism=parallelism,
		task_allocation_override=task_allocation_override,
	)
	preview_targets, mpi_rank, mpi_size = _partition_targets_for_preview(
		targets=list(targets),
		allocation_backend=allocation_backend,
	)
	return StageAllocationPreview(
		stage=str(stage_name),
		target_count=len(preview_targets),
		target_labels=_allocation_target_labels(list(preview_targets)),
		phase_resource_classes=phase_resource_classes,
		parallelism=parallelism,
		allocation_backend=allocation_backend,
		mpi_rank=mpi_rank,
		mpi_size=mpi_size,
	)


_SPIKESORT_DIRECT_PHASE_LABELS: dict[str, str] = {
	"spikesort.concat_binary": "concat_binary",
	"spikesort.cleanup_concat_binary": "cleanup_concat_binary",
	"spikesort.cleanup_analyzers": "cleanup_analyzers",
	"spikesort.sort": "sort",
	"spikesort.summarize_sort": "summarize_sort",
	"spikesort.bombcell_label": "bombcell_label",
	"spikesort.bombcell_label_pass2": "bombcell_label_pass2",
	"spikesort.merge": "merge",
	"spikesort.merge_SLAy": "merge_slay",
}


_ANALYSIS_DIRECT_PHASE_LABELS: dict[str, str] = {
	"analysis.compute_metrics": "compute_metrics",
	"analysis.unitmatch": "unitmatch",
}


_ANALYSIS_RESOURCE_ATTR_BY_PHASE_LABEL: dict[str, str] = {
	"compute_metrics": "compute_metrics_resource_class",
	"unitmatch": "unitmatch_resource_class",
}


def _spikesort_allocation_phase_labels(stage_config: Any, stage_name: str) -> tuple[str, ...]:
	stage_name = str(stage_name).strip()
	if stage_name == "spikesort":
		return tuple(str(phase.phase_label) for phase in _enabled_spikesort_runtime_phase_plan(stage_config))
	phase_label = _SPIKESORT_DIRECT_PHASE_LABELS.get(stage_name)
	if phase_label is None:
		return ()
	if phase_label != "merge":
		return (phase_label,)
	labels: list[str] = []
	for token in tuple(getattr(stage_config, "merge_sequence", ()) or ("SLAy",)):
		normalized = str(token).strip().lower().replace("-", "_")
		if normalized in {"slay", "merge_slay"}:
			labels.append("merge_slay")
	return _unique_resource_classes(tuple(labels))


def _build_spikesort_allocation_preview(
	*,
	config_path: str,
	stage_name: str,
	target_datasets_override: list[int] | None,
	limit_segments_override: int | None,
	limit_datasets_override: int | None,
	limit_wells_per_dataset_override: int | None,
	force_restart_override: bool | None,
	replot_override: bool | None,
	task_allocation_override: dict[str, Any] | None = None,
) -> StageAllocationPreview:
	bundle = load_pipeline_runtime_bundle(config_path=config_path)
	stage_config = parse_spikesort_stage_config(
		runtime_config=bundle.runtime_config,
		force_restart_override=force_restart_override,
		replot_override=replot_override,
	)
	stage_config = _with_debug_limit_overrides(
		stage_config,
		limit_segments_override=limit_segments_override,
		limit_datasets_override=limit_datasets_override,
		limit_wells_per_dataset_override=limit_wells_per_dataset_override,
	)
	phase_labels = _spikesort_allocation_phase_labels(stage_config, stage_name)
	targets = _select_execution_targets_with_debug_limits(
		bundle=bundle,
		stage_name=stage_name,
		stage_config=stage_config,
		target_datasets=target_datasets_override,
	)
	targets = _apply_spikesort_stage_debug_limits(
		stage_name="spikesort",
		stage_config=stage_config,
		targets=list(targets),
	)
	phase_resource_classes = _spikesort_phase_resource_classes_from_labels(stage_config, phase_labels)
	parallelism = _resolve_runtime_stage_parallelism(
		bundle=bundle,
		stage_name="spikesort",
		target_count=len(targets),
		targets=targets,
		phase_resource_classes=phase_resource_classes,
		task_allocation_override=task_allocation_override,
	)
	allocation_backend = _resolve_preview_allocation_backend(
		bundle=bundle,
		parallelism=parallelism,
		task_allocation_override=task_allocation_override,
	)
	preview_targets, mpi_rank, mpi_size = _partition_targets_for_preview(
		targets=list(targets),
		allocation_backend=allocation_backend,
	)
	return StageAllocationPreview(
		stage=str(stage_name),
		target_count=len(preview_targets),
		target_labels=_allocation_target_labels(list(preview_targets)),
		phase_resource_classes=phase_resource_classes,
		parallelism=parallelism,
		allocation_backend=allocation_backend,
		mpi_rank=mpi_rank,
		mpi_size=mpi_size,
	)


def _build_reconstruct_allocation_preview(
	*,
	config_path: str,
	stage_name: str,
	target_datasets_override: list[int] | None,
	unit_id_override: int | None,
	unit_ids_override: list[int] | None,
	unit_limit_override: int | None,
	limit_segments_override: int | None,
	limit_datasets_override: int | None,
	limit_wells_per_dataset_override: int | None,
	force_restart_override: bool | None,
	replot_override: bool | None,
	task_allocation_override: dict[str, Any] | None = None,
) -> StageAllocationPreview:
	bundle = load_pipeline_runtime_bundle(config_path=config_path)
	probe_geometry = parse_probe_geometry_from_data_config(data_config=bundle.data_config)
	stage_config = parse_reconstruction_stage_config(
		runtime_config=bundle.runtime_config,
		unit_id_override=unit_id_override,
		unit_ids_override=unit_ids_override,
		unit_limit_override=unit_limit_override,
		limit_segments_override=limit_segments_override,
		force_restart_override=force_restart_override,
		replot_override=replot_override,
	)
	stage_config = _with_debug_limit_overrides(
		stage_config,
		limit_segments_override=limit_segments_override,
		limit_datasets_override=limit_datasets_override,
		limit_wells_per_dataset_override=limit_wells_per_dataset_override,
	)
	if str(stage_name).strip() == "reconstruct":
		stage_config = _apply_replot_phase_filter(
			stage_config,
			replot=bool(replot_override),
			stage_name="reconstruct",
		)
	targets = _select_execution_targets_with_debug_limits(
		bundle=bundle,
		stage_name=stage_name,
		stage_config=stage_config,
		target_datasets=target_datasets_override,
	)
	reconstruct_templates_config = None
	if callable(getattr(bundle.runtime_config, "get", None)):
		templates_runtime_config = build_reconstruct_templates_runtime_config(bundle.runtime_config)
		reconstruct_templates_config = parse_reconstruct_templates_config(
			runtime_config=templates_runtime_config,
			probe_geometry=probe_geometry,
			unit_id_override=unit_id_override,
			unit_ids_override=unit_ids_override,
			unit_limit_override=stage_config.unit_limit,
			limit_segments_override=stage_config.limit_segments,
			force_restart_override=force_restart_override,
			replot_override=replot_override,
		)
		reconstruct_templates_config = _with_debug_limit_overrides(
			reconstruct_templates_config,
			limit_segments_override=limit_segments_override,
			limit_datasets_override=limit_datasets_override,
			limit_wells_per_dataset_override=limit_wells_per_dataset_override,
		)
		if str(stage_name).strip() == "reconstruct":
			reconstruct_templates_config = _apply_replot_phase_filter(
				reconstruct_templates_config,
				replot=bool(replot_override),
				stage_name="reconstruct.templates",
			)
	if bool(getattr(stage_config, "debug_mode_enabled", False)):
		targets = _apply_spikesort_debug_target_limits(
			stage_name=stage_name,
			targets=list(targets),
			limit_datasets=getattr(stage_config, "debug_limit_datasets", None),
			limit_wells=getattr(stage_config, "debug_limit_wells", None),
			limit_wells_per_dataset=getattr(stage_config, "debug_limit_wells_per_dataset", None),
		)
	phase_resource_classes = _reconstruct_runtime_phase_resource_classes(
		stage_config,
		reconstruct_templates_config,
		stage_name=stage_name,
	)
	parallelism = _resolve_runtime_stage_parallelism(
		bundle=bundle,
		stage_name="reconstruct",
		target_count=len(targets),
		targets=targets,
		phase_resource_classes=phase_resource_classes,
		task_allocation_override=task_allocation_override,
	)
	allocation_backend = _resolve_preview_allocation_backend(
		bundle=bundle,
		parallelism=parallelism,
		task_allocation_override=task_allocation_override,
	)
	preview_targets, mpi_rank, mpi_size = _partition_targets_for_preview(
		targets=list(targets),
		allocation_backend=allocation_backend,
	)
	return StageAllocationPreview(
		stage=str(stage_name),
		target_count=len(preview_targets),
		target_labels=_allocation_target_labels(list(preview_targets)),
		phase_resource_classes=phase_resource_classes,
		parallelism=parallelism,
		allocation_backend=allocation_backend,
		mpi_rank=mpi_rank,
		mpi_size=mpi_size,
	)


def build_stage_allocation_previews(
	*,
	config_path: str,
	stages: list[str] | tuple[str, ...],
	target_datasets_override: list[int] | None = None,
	unit_id_override: int | None = None,
	unit_ids_override: list[int] | None = None,
	unit_limit_override: int | None = None,
	limit_segments_override: int | None = None,
	limit_datasets_override: int | None = None,
	limit_wells_per_dataset_override: int | None = None,
	force_restart_override: bool | None = None,
	replot_override: bool | None = None,
	task_allocation_override: dict[str, Any] | None = None,
) -> list[StageAllocationPreview]:
	previews: list[StageAllocationPreview] = []
	for stage_name in tuple(str(item).strip() for item in stages if str(item).strip()):
		if stage_name == "preprocess" or stage_name.startswith("preprocess."):
			previews.append(
				_build_preprocess_allocation_preview(
					config_path=config_path,
					stage_name=stage_name,
					target_datasets_override=target_datasets_override,
					limit_segments_override=limit_segments_override,
					limit_datasets_override=limit_datasets_override,
					limit_wells_per_dataset_override=limit_wells_per_dataset_override,
					force_restart_override=force_restart_override,
					replot_override=replot_override,
					task_allocation_override=task_allocation_override,
				)
			)
		elif stage_name == "spikesort" or stage_name.startswith("spikesort."):
			previews.append(
				_build_spikesort_allocation_preview(
					config_path=config_path,
					stage_name=stage_name,
					target_datasets_override=target_datasets_override,
					limit_segments_override=limit_segments_override,
					limit_datasets_override=limit_datasets_override,
					limit_wells_per_dataset_override=limit_wells_per_dataset_override,
					force_restart_override=force_restart_override,
					replot_override=replot_override,
					task_allocation_override=task_allocation_override,
				)
			)
		elif stage_name == "reconstruct" or stage_name.startswith("reconstruct."):
			previews.append(
				_build_reconstruct_allocation_preview(
					config_path=config_path,
					stage_name=stage_name,
					target_datasets_override=target_datasets_override,
					unit_id_override=unit_id_override,
					unit_ids_override=unit_ids_override,
					unit_limit_override=unit_limit_override,
					limit_segments_override=limit_segments_override,
					limit_datasets_override=limit_datasets_override,
					limit_wells_per_dataset_override=limit_wells_per_dataset_override,
					force_restart_override=force_restart_override,
					replot_override=replot_override,
					task_allocation_override=task_allocation_override,
				)
			)
		else:
			raise ValueError(f"Unsupported stage for allocation preview: {stage_name}")
	return previews


def _format_allocation_plan_summary(
	plan: TaskAllocationPlan | None,
	*,
	allocation_backend: str | None = None,
	mpi_rank: int | None = None,
	mpi_size: int | None = None,
) -> list[str]:
	backend = str(allocation_backend or "none").strip().lower()
	if plan is None:
		if backend == "mpi":
			mpi_line = "  mpi_context: unavailable"
			if mpi_rank is not None and mpi_size is not None:
				mpi_line = f"  mpi_context: rank={int(mpi_rank)} size={int(mpi_size)}"
			lines = [
				"task_allocation: enabled backend=mpi",
				"",
				mpi_line,
				"  local_task_slots: n/a (rank partitioning handled by MPI backend)",
			]
			try:
				topology = detect_cpu_topology(logger=LOGGER)
				lines += [
					"",
					"  cpu_topology:",
					f"    visible_cpus:    {format_cpu_set(topology.visible_cpus)}",
					f"    physical_cores:  {topology.physical_core_count}",
					f"    logical_cpus:    {topology.logical_cpu_count}",
				]
			except Exception:
				pass
			lines += [
				"",
				"  thread_env: current",
			]
			for var in _THREAD_ENV_VARS:
				lines.append(f"    {var}={os.environ.get(var, 'unset')}")
			return lines
		return ["task_allocation: disabled"]
	lines = [
		f"task_allocation: enabled backend={plan.backend} bind={plan.bind}",
		"",
		"  cpu_topology:",
		f"    visible_cpus:    {format_cpu_set(plan.topology.visible_cpus)}",
		f"    physical_cores:  {plan.topology.physical_core_count}",
		f"    logical_cpus:    {plan.topology.logical_cpu_count}",
		"",
		"  task_shape:",
		f"    cpus_per_task:   {plan.cpus_per_task}  (source: {plan.cpus_per_task_source})",
		f"    tasks_per_node:  {plan.effective_tasks_per_node}",
		f"    cpu_capacity:    {plan.cpu_capacity_tasks} tasks",
	]
	clamp_inputs: list[str] = [f"cpu_capacity={plan.cpu_capacity_tasks}"]
	if plan.ram_capacity_tasks is not None:
		clamp_inputs.append(f"ram_capacity={plan.ram_capacity_tasks}")
	if plan.shm_capacity_tasks is not None:
		clamp_inputs.append(f"shm_capacity={plan.shm_capacity_tasks}")
	if plan.target_count is not None:
		clamp_inputs.append(f"target_count={plan.target_count}")
	if plan.keyed_read_cap is not None:
		clamp_inputs.append(f"keyed_h5_read={plan.keyed_read_cap}")
	lines += [
		"",
		f"  slot_clamps:  {', '.join(clamp_inputs)}  ->  effective={plan.effective_tasks_per_node}",
	]
	thread_env_note = f"policy={plan.nested_thread_policy}" if bool(plan.set_thread_env) else "disabled"
	lines += [
		"",
		f"  thread_env: {thread_env_note}",
		"    current:",
	]
	for var in _THREAD_ENV_VARS:
		val = os.environ.get(var, "unset")
		lines.append(f"      {var}={val}")
	if bool(plan.set_thread_env):
		_policy = str(plan.nested_thread_policy).strip()
		if _policy == "match_cpus_per_task":
			_val = str(plan.cpus_per_task)
			lines.append("    worker:  (set per-worker at execution)")
			for var in _THREAD_ENV_VARS:
				lines.append(f"      {var}={_val}")
		elif _policy == "force_1":
			lines.append("    worker:  (set per-worker at execution)")
			for var in _THREAD_ENV_VARS:
				lines.append(f"      {var}=1")
		else:
			lines.append(f"    worker:  no-op  (policy={_policy}, vars preserved as-is)")
	lines += [
		"",
		f"  slots: {len(plan.slots)}",
	]
	for slot in plan.slots:
		lines.append(f"    slot[{slot.slot_id}]: cpus={format_cpu_set(slot.logical_cpus)}")
	return lines


def format_stage_allocation_previews(previews: list[StageAllocationPreview] | tuple[StageAllocationPreview, ...]) -> str:
	lines = ["Allocation preview", "No stage work was run."]
	for preview in previews:
		parallelism = preview.parallelism
		lines.append("")
		lines.append(f"stage: {preview.stage}")
		lines.append(f"selected_targets: {preview.target_count}")
		if preview.target_labels:
			target_suffix = "" if int(preview.target_count) <= len(preview.target_labels) else " ..."
			lines.append(f"selected_target_sample: {', '.join(preview.target_labels)}{target_suffix}")
		lines.append(f"phase_resource_classes: {', '.join(preview.phase_resource_classes) if preview.phase_resource_classes else 'none'}")
		lines.append(
			"parallelism: "
			f"well_workers={int(getattr(parallelism, 'well_workers', 1))} "
			f"unit_workers={int(getattr(parallelism, 'unit_workers', 1))}"
		)
		if preview.allocation_backend is not None and str(preview.allocation_backend).strip().lower() == "mpi":
			if preview.mpi_rank is not None and preview.mpi_size is not None and int(preview.mpi_size) > 1:
				lines.append(f"mpi_partition: rank={int(preview.mpi_rank)} size={int(preview.mpi_size)}")
		plan = getattr(parallelism, "task_allocation_plan", None)
		lines.extend(
			_format_allocation_plan_summary(
				plan,
				allocation_backend=preview.allocation_backend,
				mpi_rank=preview.mpi_rank,
				mpi_size=preview.mpi_size,
			)
		)
		lines.extend(
			_format_sample_worker_environment(
				plan=plan,
				mpi_context=None,
				allocation_backend=preview.allocation_backend,
			)
		)
	return "\n".join(lines)


def _format_sample_worker_environment(
	*,
	plan: TaskAllocationPlan | None,
	mpi_context: Any | None = None,
	allocation_backend: str | None = None,
) -> list[str]:
	"""Format the environment that would be set for sample workers."""
	lines = []
	backend = str(allocation_backend or "none").strip().lower()

	# If no plan, check for MPI backend
	if plan is None:
		if backend == "mpi":
			lines.append("")
			lines.append("sample_worker_environment_preview:")
			lines.append("  (mpi backend, no local task slots)")
			lines.append("  each rank partitions targets independently")
		return lines

	# If plan has slots, spawn sample worker for each
	if plan.slots:
		lines.append("")
		lines.append("sample_worker_environment_preview:")

		for slot in plan.slots:
			env_state = capture_sample_worker_environment(
				slot,
				plan=plan,
			)

			lines.append(f"  slot[{slot.slot_id}]:")

			if "error" in env_state:
				lines.append(f"    error: {env_state['error']}")
				continue

			if "cpu_affinity" in env_state:
				lines.append(f"    cpu_affinity: {env_state['cpu_affinity']}")
			if "cpu_count" in env_state:
				lines.append(f"    cpu_count: {env_state['cpu_count']}")

			if "thread_env" in env_state:
				thread_vars = env_state["thread_env"]
				set_vars = {k: v for k, v in thread_vars.items() if v is not None}
				if set_vars:
					lines.append(f"    thread_env: {' '.join(f'{k}={v}' for k, v in sorted(set_vars.items()))}")
				else:
					lines.append("    thread_env: (preserved from parent)")

	return lines


def print_stage_allocation_preview(
	*,
	config_path: str,
	stages: list[str] | tuple[str, ...],
	target_datasets_override: list[int] | None = None,
	unit_id_override: int | None = None,
	unit_ids_override: list[int] | None = None,
	unit_limit_override: int | None = None,
	limit_segments_override: int | None = None,
	limit_datasets_override: int | None = None,
	limit_wells_per_dataset_override: int | None = None,
	force_restart_override: bool | None = None,
	replot_override: bool | None = None,
	task_allocation_override: dict[str, Any] | None = None,
) -> None:
	previews = build_stage_allocation_previews(
		config_path=config_path,
		stages=stages,
		target_datasets_override=target_datasets_override,
		unit_id_override=unit_id_override,
		unit_ids_override=unit_ids_override,
		unit_limit_override=unit_limit_override,
		limit_segments_override=limit_segments_override,
		limit_datasets_override=limit_datasets_override,
		limit_wells_per_dataset_override=limit_wells_per_dataset_override,
		force_restart_override=force_restart_override,
		replot_override=replot_override,
		task_allocation_override=task_allocation_override,
	)
	print(format_stage_allocation_previews(previews))


def run_preprocess_from_runtime(
	*,
	config_path: str,
	limit_segments_override: int | None = None,
	limit_datasets_override: int | None = None,
	target_datasets_override: list[int] | None = None,
	limit_wells_per_dataset_override: int | None = None,
	force_restart_override: bool | None = None,
	replot_override: bool | None = None,
	task_allocation_override: dict[str, Any] | None = None,
) -> MultiTargetStageResult:
	bundle: PipelineRuntimeBundle = load_pipeline_runtime_bundle(config_path=config_path)
	publish_policy = _resolve_publish_policy(runtime_config=bundle.runtime_config, data_config=bundle.data_config)
	_log_publish_policy(stage_name="preprocess", policy=publish_policy)
	stage_config = parse_preprocess_stage_config(
		runtime_config=bundle.runtime_config,
		force_restart_override=force_restart_override,
		replot_override=replot_override,
	)
	stage_config = _with_debug_limit_overrides(
		stage_config,
		limit_segments_override=limit_segments_override,
		limit_datasets_override=limit_datasets_override,
		limit_wells_per_dataset_override=limit_wells_per_dataset_override,
	)
	stage_config = _apply_replot_phase_filter(
		stage_config,
		replot=bool(replot_override),
		stage_name="preprocess",
	)
	targets = _select_preprocess_execution_targets(
		bundle=bundle,
		stage_config=stage_config,
		materialize_scratch_inputs=_preprocess_copy_phase_enabled(stage_config),
		target_datasets=target_datasets_override,
	)
	targets = _apply_preprocess_stage_debug_limits(
		stage_name="preprocess",
		stage_config=stage_config,
		targets=list(targets),
	)
	phase_resource_classes = _preprocess_runtime_phase_resource_classes(stage_config, "preprocess")
	parallelism = _resolve_runtime_stage_parallelism(
		bundle=bundle,
		stage_name="preprocess",
		target_count=len(targets),
		targets=targets,
		phase_resource_classes=phase_resource_classes,
		task_allocation_override=task_allocation_override,
	)
	_log_runtime_stage_topology(stage_name="preprocess", targets=list(targets), parallelism=parallelism)
	resource_budget_manager = _build_stage_resource_budget_manager(
		bundle=bundle,
		parallelism=parallelism,
		phase_resource_classes=phase_resource_classes,
		target_count=len(targets),
	)
	unit_workers = _resolve_preprocess_runtime_unit_workers(
		stage_name="preprocess",
		parallelism=parallelism,
		stage_config=stage_config,
	)
	n_jobs_source = _preprocess_runtime_n_jobs_source(
		stage_config=stage_config,
		uses_nested_workers=_preprocess_runtime_uses_nested_workers(stage_name="preprocess", stage_config=stage_config),
	)
	emit_subphase_dividers_to_stdout = _resolve_preprocess_subphase_dividers_to_stdout(
		stage_name="preprocess",
		parallelism=parallelism,
		stage_config=stage_config,
	)

	bypass_auto_restart_skip = bool(
		getattr(stage_config, "force_restart", False)
		or getattr(stage_config, "replot", False)
	)

	def _worker(target):
		inputs = build_preprocess_inputs_for_target(
			target=target,
			stage_config=stage_config,
			unit_workers=int(unit_workers),
		)
		inputs = _with_preprocess_runtime_worker_allocation(
			inputs,
			parallelism=parallelism,
			n_jobs_source=str(n_jobs_source),
		)
		inputs = replace(
			inputs,
			logging_subphase_dividers_to_stdout=bool(emit_subphase_dividers_to_stdout),
		)
		# Slice 14c (target-level auto-restart skip): when neither
		# --force-restart nor --replot is set, skip targets whose every
		# phase summary is already ok. The monolithic run_preprocess is
		# itself phase-aware via slice 13's `with_checkpoint_marker`, but
		# bailing here saves the per-target setup cost (path resolution,
		# logger spin-up, resource budget acquisition) for already-done
		# targets — meaningful at the cohort scale where most targets
		# are typically up-to-date on a re-run.
		if not bypass_auto_restart_skip and target_all_preprocess_phases_succeeded(inputs):
			return {
				"stage": "preprocess",
				"status": "skipped",
				"reason": "all_phases_ok",
			}
		return run_preprocess(inputs)

	with stage_resource_budget_context(resource_budget_manager):
		target_results = _distribute_runtime_targets(
			targets=targets,
			parallelism=parallelism,
			worker_fn=_worker,
			stage_name="preprocess",
			progress=PipelineProgress(ProgressSpec(label="preprocess wells", total=len(targets), unit="well")),
			advance_progress_on_target_complete=True,
			bundle=bundle,
		)
	target_results = [_publish_preprocess_target_result(item, policy=publish_policy) for item in target_results]

	succeeded = sum(1 for item in target_results if item.status == "ok")
	failed = sum(1 for item in target_results if item.status != "ok")
	return MultiTargetStageResult(
		stage="preprocess",
		total_targets=len(target_results),
		succeeded_targets=succeeded,
		failed_targets=failed,
		target_results=target_results,
	)


def _run_preprocess_substage_from_runtime(
	*,
	config_path: str,
	stage_name: str,
	runner_fn: Callable[[Any], Any],
	limit_segments_override: int | None = None,
	limit_datasets_override: int | None = None,
	target_datasets_override: list[int] | None = None,
	limit_wells_per_dataset_override: int | None = None,
	force_restart_override: bool | None = None,
	replot_override: bool | None = None,
	task_allocation_override: dict[str, Any] | None = None,
) -> MultiTargetStageResult:
	bundle: PipelineRuntimeBundle = load_pipeline_runtime_bundle(config_path=config_path)
	stage_config = parse_preprocess_stage_config(
		runtime_config=bundle.runtime_config,
		force_restart_override=force_restart_override,
		replot_override=replot_override,
	)
	stage_config = _with_debug_limit_overrides(
		stage_config,
		limit_segments_override=limit_segments_override,
		limit_datasets_override=limit_datasets_override,
		limit_wells_per_dataset_override=limit_wells_per_dataset_override,
	)
	# `copy_src_to_scratch` moved to the init stage in slice 5; preprocess
	# substages never materialize scratch inputs themselves. The init
	# stage's run_init_*_from_runtime entry points trigger materialization
	# when their copy phase is configured.
	targets = _select_preprocess_execution_targets(
		bundle=bundle,
		stage_config=stage_config,
		materialize_scratch_inputs=False,
		target_datasets=target_datasets_override,
	)
	targets = _apply_preprocess_stage_debug_limits(
		stage_name="preprocess",
		stage_config=stage_config,
		targets=list(targets),
	)
	targets = _apply_preprocess_substage_phase_debug_limits(
		stage_name=stage_name,
		stage_config=stage_config,
		targets=list(targets),
	)
	phase_resource_classes = _preprocess_runtime_phase_resource_classes(stage_config, stage_name)
	parallelism = _resolve_runtime_stage_parallelism(
		bundle=bundle,
		stage_name="preprocess",
		target_count=len(targets),
		targets=targets,
		phase_resource_classes=phase_resource_classes,
		task_allocation_override=task_allocation_override,
	)
	resource_budget_manager = _build_stage_resource_budget_manager(
		bundle=bundle,
		parallelism=parallelism,
		phase_resource_classes=phase_resource_classes,
		target_count=len(targets),
	)
	unit_workers = _resolve_preprocess_runtime_unit_workers(
		stage_name=stage_name,
		parallelism=parallelism,
		stage_config=stage_config,
	)
	n_jobs_source = _preprocess_runtime_n_jobs_source(
		stage_config=stage_config,
		uses_nested_workers=_preprocess_runtime_uses_nested_workers(stage_name=stage_name, stage_config=stage_config),
	)
	emit_subphase_dividers_to_stdout = _resolve_preprocess_subphase_dividers_to_stdout(
		stage_name=stage_name,
		parallelism=parallelism,
		stage_config=stage_config,
	)

	def _worker(target):
		inputs = build_preprocess_inputs_for_target(
			target=target,
			stage_config=stage_config,
			unit_workers=int(unit_workers),
		)
		inputs = _with_preprocess_runtime_worker_allocation(
			inputs,
			parallelism=parallelism,
			n_jobs_source=str(n_jobs_source),
		)
		inputs = replace(
			inputs,
			logging_subphase_dividers_to_stdout=bool(emit_subphase_dividers_to_stdout),
		)
		return runner_fn(inputs)

	with stage_resource_budget_context(resource_budget_manager):
		target_results = _distribute_runtime_targets(
			targets=targets,
			parallelism=parallelism,
			worker_fn=_worker,
			stage_name=stage_name,
			progress=PipelineProgress(ProgressSpec(label=f"{stage_name} wells", total=len(targets), unit="well")),
			advance_progress_on_target_complete=True,
			bundle=bundle,
		)
	succeeded = sum(1 for item in target_results if item.status == "ok")
	failed = sum(1 for item in target_results if item.status != "ok")
	return MultiTargetStageResult(
		stage=stage_name,
		total_targets=len(target_results),
		succeeded_targets=succeeded,
		failed_targets=failed,
		target_results=target_results,
	)


def run_preprocess_save_rec_metadata_from_runtime(
	*,
	config_path: str,
	limit_segments_override: int | None = None,
	limit_datasets_override: int | None = None,
	target_datasets_override: list[int] | None = None,
	limit_wells_per_dataset_override: int | None = None,
	force_restart_override: bool | None = None,
	replot_override: bool | None = None,
	task_allocation_override: dict[str, Any] | None = None,
) -> MultiTargetStageResult:
	return _run_preprocess_substage_from_runtime(
		config_path=config_path,
		stage_name="preprocess.save_rec_metadata",
		runner_fn=run_preprocess_save_rec_metadata,
		limit_segments_override=limit_segments_override,
		limit_datasets_override=limit_datasets_override,
		target_datasets_override=target_datasets_override,
		limit_wells_per_dataset_override=limit_wells_per_dataset_override,
		force_restart_override=force_restart_override,
		replot_override=replot_override,
		task_allocation_override=task_allocation_override,
	)


def run_preprocess_plot_segment_channel_layouts_from_runtime(
	*,
	config_path: str,
	limit_segments_override: int | None = None,
	limit_datasets_override: int | None = None,
	target_datasets_override: list[int] | None = None,
	limit_wells_per_dataset_override: int | None = None,
	force_restart_override: bool | None = None,
	replot_override: bool | None = None,
	task_allocation_override: dict[str, Any] | None = None,
) -> MultiTargetStageResult:
	return _run_preprocess_substage_from_runtime(
		config_path=config_path,
		stage_name="preprocess.plot_segment_channel_layouts",
		runner_fn=run_preprocess_plot_segment_channel_layouts,
		limit_segments_override=limit_segments_override,
		limit_datasets_override=limit_datasets_override,
		target_datasets_override=target_datasets_override,
		limit_wells_per_dataset_override=limit_wells_per_dataset_override,
		force_restart_override=force_restart_override,
		replot_override=replot_override,
		task_allocation_override=task_allocation_override,
	)


def run_preprocess_preprocess_segments_from_runtime(
	*,
	config_path: str,
	limit_segments_override: int | None = None,
	limit_datasets_override: int | None = None,
	target_datasets_override: list[int] | None = None,
	limit_wells_per_dataset_override: int | None = None,
	force_restart_override: bool | None = None,
	replot_override: bool | None = None,
	task_allocation_override: dict[str, Any] | None = None,
) -> MultiTargetStageResult:
	return _run_preprocess_substage_from_runtime(
		config_path=config_path,
		stage_name="preprocess.preprocess_segments",
		runner_fn=run_preprocess_preprocess_segments,
		limit_segments_override=limit_segments_override,
		limit_datasets_override=limit_datasets_override,
		target_datasets_override=target_datasets_override,
		limit_wells_per_dataset_override=limit_wells_per_dataset_override,
		force_restart_override=force_restart_override,
		replot_override=replot_override,
		task_allocation_override=task_allocation_override,
	)


def run_preprocess_plot_segment_traces_from_runtime(
	*,
	config_path: str,
	limit_segments_override: int | None = None,
	limit_datasets_override: int | None = None,
	target_datasets_override: list[int] | None = None,
	limit_wells_per_dataset_override: int | None = None,
	force_restart_override: bool | None = None,
	replot_override: bool | None = None,
	task_allocation_override: dict[str, Any] | None = None,
) -> MultiTargetStageResult:
	return _run_preprocess_substage_from_runtime(
		config_path=config_path,
		stage_name="preprocess.plot_segment_traces",
		runner_fn=run_preprocess_plot_segment_traces,
		limit_segments_override=limit_segments_override,
		limit_datasets_override=limit_datasets_override,
		target_datasets_override=target_datasets_override,
		limit_wells_per_dataset_override=limit_wells_per_dataset_override,
		force_restart_override=force_restart_override,
		replot_override=replot_override,
		task_allocation_override=task_allocation_override,
	)


def run_preprocess_plot_raster_threshold_from_runtime(
	*,
	config_path: str,
	limit_segments_override: int | None = None,
	limit_datasets_override: int | None = None,
	target_datasets_override: list[int] | None = None,
	limit_wells_per_dataset_override: int | None = None,
	force_restart_override: bool | None = None,
	replot_override: bool | None = None,
	task_allocation_override: dict[str, Any] | None = None,
) -> MultiTargetStageResult:
	return _run_preprocess_substage_from_runtime(
		config_path=config_path,
		stage_name="preprocess.plot_raster_threshold",
		runner_fn=run_preprocess_plot_raster_threshold,
		limit_segments_override=limit_segments_override,
		limit_datasets_override=limit_datasets_override,
		target_datasets_override=target_datasets_override,
		limit_wells_per_dataset_override=limit_wells_per_dataset_override,
		force_restart_override=force_restart_override,
		replot_override=replot_override,
		task_allocation_override=task_allocation_override,
	)


def run_spikesort_from_runtime(
	*,
	config_path: str,
	limit_segments_override: int | None = None,
	limit_datasets_override: int | None = None,
	target_datasets_override: list[int] | None = None,
	limit_wells_per_dataset_override: int | None = None,
	force_restart_override: bool | None = None,
	replot_override: bool | None = None,
	task_allocation_override: dict[str, Any] | None = None,
) -> MultiTargetStageResult:
	bundle: PipelineRuntimeBundle = load_pipeline_runtime_bundle(config_path=config_path)
	publish_policy = _resolve_publish_policy(runtime_config=bundle.runtime_config, data_config=bundle.data_config)
	_log_publish_policy(stage_name="spikesort", policy=publish_policy)
	stage_config = parse_spikesort_stage_config(
		runtime_config=bundle.runtime_config,
		force_restart_override=force_restart_override,
		replot_override=replot_override,
	)
	stage_config = _with_debug_limit_overrides(
		stage_config,
		limit_segments_override=limit_segments_override,
		limit_datasets_override=limit_datasets_override,
		limit_wells_per_dataset_override=limit_wells_per_dataset_override,
	)
	stage_config = _apply_replot_phase_filter(
		stage_config,
		replot=bool(replot_override),
		stage_name="spikesort",
	)
	phase_plan = _enabled_spikesort_runtime_phase_plan(stage_config)
	if not phase_plan:
		LOGGER.info("spikesort: no enabled phases")
		return MultiTargetStageResult(
			stage="spikesort",
			total_targets=0,
			succeeded_targets=0,
			failed_targets=0,
			target_results=[],
		)
	targets = _select_execution_targets_with_debug_limits(
		bundle=bundle,
		stage_name="spikesort",
		stage_config=stage_config,
		target_datasets=target_datasets_override,
	)
	targets = _apply_spikesort_runtime_phase_plan_debug_limits(
		stage_config=stage_config,
		targets=list(targets),
		phase_plan=phase_plan,
	)
	phase_resource_classes = _spikesort_runtime_phase_resource_classes(phase_plan)
	parallelism = _resolve_runtime_stage_parallelism(
		bundle=bundle,
		stage_name="spikesort",
		target_count=len(targets),
		targets=targets,
		phase_resource_classes=phase_resource_classes,
		task_allocation_override=task_allocation_override,
	)
	n_jobs_source = _runtime_n_jobs_source(stage_config)
	runtime_stage_config = _stage_config_with_runtime_n_jobs(
		stage_config,
		unit_workers=int(parallelism.unit_workers),
	)
	runtime_n_jobs = _runtime_n_jobs_from_stage_config(
		runtime_stage_config,
		fallback_n_jobs=int(parallelism.unit_workers),
	)
	_log_runtime_stage_topology(stage_name="spikesort", targets=list(targets), parallelism=parallelism)
	LOGGER.info(
		"spikesort: starting target-local phase chains targets=%d phases=%s well_workers=%d n_jobs=%d n_jobs_source=%s",
		len(targets),
		[phase.name for phase in phase_plan],
		int(parallelism.well_workers),
		int(runtime_n_jobs),
		str(n_jobs_source),
	)
	resource_budget_manager = _build_stage_resource_budget_manager(
		bundle=bundle,
		parallelism=parallelism,
		phase_resource_classes=phase_resource_classes,
		target_count=len(targets),
	)

	from .stages.spikesort.runner import target_all_spikesort_phases_succeeded

	bypass_auto_restart_skip = bool(
		getattr(stage_config, "force_restart", False)
		or getattr(stage_config, "replot", False)
	)

	def _worker(target):
		# Slice 14c (target-level auto-restart skip) — only fires when
		# every phase in the plan is in the convention-named-relpath map
		# AND its summary on disk shows ok. Phase plans containing
		# sort / summarize_sort / merge_* / etc fall through to normal
		# dispatch (see `_SPIKESORT_PHASE_SUMMARY_RELPATH_ATTRS`).
		if not bypass_auto_restart_skip and target_all_spikesort_phases_succeeded(
			target=target,
			stage_config=runtime_stage_config,
			phase_plan=phase_plan,
		):
			return {
				"stage": "spikesort",
				"status": "skipped",
				"reason": "all_phases_ok",
			}

		def _descriptor_for_phase(phase: _SpikesortRuntimePhase) -> PhaseDescriptor:
			def _run_phase(phase: _SpikesortRuntimePhase = phase):
				_log_spikesort_phase_worker_allocation(
					phase_name=str(phase.phase_label),
					target=target,
					parallelism=parallelism,
					n_jobs=int(runtime_n_jobs),
					n_jobs_source=str(n_jobs_source),
				)

				def _run_target_phase() -> Any:
					return phase.target_runner(
						target=target,
						stage_config=runtime_stage_config,
						unit_workers=int(runtime_n_jobs),
					)

				return _run_spikesort_phase_with_optional_sort_gate(
					phase_label=str(phase.phase_label),
					target=target,
					runner=_run_target_phase,
					sort_phase_gate=None,
				)

			return PhaseDescriptor(
				name=phase.name,
				runner=_run_phase,
				resource_class=phase.resource_class,
				pipeline_thread_count=int(runtime_n_jobs),
			)

		chain_result = run_phase_chain(
			phases=[_descriptor_for_phase(phase) for phase in phase_plan],
			logger=LOGGER,
			target_label=f"{getattr(target, 'dataset_index', 'unknown')}:{getattr(target, 'stream_id', 'unknown')}",
			resource_key_context=target,
		)
		if chain_result.result is None:
			raise RuntimeError("spikesort phase chain produced no result")
		return chain_result.result

	with stage_resource_budget_context(resource_budget_manager):
		target_results = _distribute_runtime_targets(
			targets=targets,
			parallelism=parallelism,
			worker_fn=_worker,
			stage_name="spikesort",
			progress=PipelineProgress(
				ProgressSpec(
					label="spikesort wells",
					total=len(targets),
					unit="well",
					enabled=True,
				)
			),
			advance_progress_on_target_complete=True,
			bundle=bundle,
		)
	target_results = [
		_publish_spikesort_chain_target_result(
			item,
			policy=publish_policy,
			output_rel_root=str(getattr(stage_config, "output_rel_root", "spikesort_outputs") or "spikesort_outputs"),
		)
		for item in target_results
	]
	succeeded = sum(1 for item in target_results if item.status == "ok")
	failed = sum(1 for item in target_results if item.status != "ok")
	return MultiTargetStageResult(
		stage="spikesort",
		total_targets=len(target_results),
		succeeded_targets=succeeded,
		failed_targets=failed,
		target_results=target_results,
	)


def _enabled_spikesort_runtime_phase_plan(
	stage_config: Any,
) -> list[_SpikesortRuntimePhase]:
	available_phases: dict[str, _SpikesortRuntimePhase] = {}
	if bool(getattr(stage_config, "concat_binary_enabled", False)):
		available_phases["concat_binary"] = (
			_SpikesortRuntimePhase(
				name="spikesort.concat_binary",
				phase_label="concat_binary",
				debug_enabled_attr="concat_binary_debug_mode_enabled",
				debug_limit_datasets_attr="concat_binary_debug_limit_datasets",
				debug_limit_wells_attr="concat_binary_debug_limit_wells",
				target_runner=_run_spikesort_concat_binary_target,
				debug_limit_wells_per_dataset_attr="concat_binary_debug_limit_wells_per_dataset",
				resource_class=getattr(stage_config, "concat_binary_resource_class", None),
			)
		)
	if bool(getattr(stage_config, "plot_concat_traces_enabled", False)):
		available_phases["plot_concat_traces"] = (
			_SpikesortRuntimePhase(
				name="spikesort.plot_concat_traces",
				phase_label="plot_concat_traces",
				debug_enabled_attr="plot_concat_traces_debug_mode_enabled",
				debug_limit_datasets_attr="plot_concat_traces_debug_limit_datasets",
				debug_limit_wells_attr="plot_concat_traces_debug_limit_wells",
				target_runner=_run_spikesort_plot_concat_traces_target,
				debug_limit_wells_per_dataset_attr="plot_concat_traces_debug_limit_wells_per_dataset",
				resource_class=getattr(stage_config, "plot_concat_traces_resource_class", None),
			)
		)
	if bool(getattr(stage_config, "plot_concat_channel_layout_enabled", False)):
		available_phases["plot_concat_channel_layout"] = (
			_SpikesortRuntimePhase(
				name="spikesort.plot_concat_channel_layout",
				phase_label="plot_concat_channel_layout",
				debug_enabled_attr="plot_concat_channel_layout_debug_mode_enabled",
				debug_limit_datasets_attr="plot_concat_channel_layout_debug_limit_datasets",
				debug_limit_wells_attr="plot_concat_channel_layout_debug_limit_wells",
				target_runner=_run_spikesort_plot_concat_channel_layout_target,
				debug_limit_wells_per_dataset_attr="plot_concat_channel_layout_debug_limit_wells_per_dataset",
				resource_class=getattr(stage_config, "plot_concat_channel_layout_resource_class", None),
			)
		)
	if bool(getattr(stage_config, "sort_enabled", True)):
		available_phases["sort"] = (
			_SpikesortRuntimePhase(
				name="spikesort.sort",
				phase_label="sort",
				debug_enabled_attr="sort_debug_mode_enabled",
				debug_limit_datasets_attr="sort_debug_limit_datasets",
				debug_limit_wells_attr="sort_debug_limit_wells",
				target_runner=_run_spikesort_sort_target,
				debug_limit_wells_per_dataset_attr="sort_debug_limit_wells_per_dataset",
				resource_class=getattr(stage_config, "sort_resource_class", None),
			)
		)
	if bool(getattr(stage_config, "summarize_sort_enabled", False)):
		available_phases["summarize_sort"] = (
			_SpikesortRuntimePhase(
				name="spikesort.summarize_sort",
				phase_label="summarize_sort",
				debug_enabled_attr="summarize_sort_debug_mode_enabled",
				debug_limit_datasets_attr="summarize_sort_debug_limit_datasets",
				debug_limit_wells_attr="summarize_sort_debug_limit_wells",
				target_runner=_run_spikesort_summarize_sort_target,
				debug_limit_wells_per_dataset_attr="summarize_sort_debug_limit_wells_per_dataset",
				resource_class=getattr(stage_config, "summarize_sort_resource_class", None),
			)
		)
	if bool(getattr(stage_config, "snapshot_sorter_output_enabled", False)):
		available_phases["snapshot_sorter_output"] = (
			_SpikesortRuntimePhase(
				name="spikesort.snapshot_sorter_output",
				phase_label="snapshot_sorter_output",
				debug_enabled_attr="snapshot_sorter_output_debug_mode_enabled",
				debug_limit_datasets_attr="snapshot_sorter_output_debug_limit_datasets",
				debug_limit_wells_attr="snapshot_sorter_output_debug_limit_wells",
				target_runner=_run_spikesort_snapshot_sorter_output_target,
				debug_limit_wells_per_dataset_attr="snapshot_sorter_output_debug_limit_wells_per_dataset",
				resource_class=getattr(stage_config, "snapshot_sorter_output_resource_class", None),
			)
		)
	if bool(getattr(stage_config, "concat_analyzer_enabled", False)):
		available_phases["concat_analyzer"] = (
			_SpikesortRuntimePhase(
				name="spikesort.concat_analyzer",
				phase_label="concat_analyzer",
				debug_enabled_attr="concat_analyzer_debug_mode_enabled",
				debug_limit_datasets_attr="concat_analyzer_debug_limit_datasets",
				debug_limit_wells_attr="concat_analyzer_debug_limit_wells",
				target_runner=_run_spikesort_concat_analyzer_target,
				debug_limit_wells_per_dataset_attr="concat_analyzer_debug_limit_wells_per_dataset",
				resource_class=getattr(stage_config, "concat_analyzer_resource_class", None),
			)
		)
	if bool(getattr(stage_config, "bombcell_label_enabled", False)):
		available_phases["bombcell_label"] = (
			_SpikesortRuntimePhase(
				name="spikesort.bombcell_label",
				phase_label="bombcell_label",
				debug_enabled_attr="bombcell_label_debug_mode_enabled",
				debug_limit_datasets_attr="bombcell_label_debug_limit_datasets",
				debug_limit_wells_attr="bombcell_label_debug_limit_wells",
				target_runner=_run_spikesort_bombcell_label_target,
				debug_limit_wells_per_dataset_attr="bombcell_label_debug_limit_wells_per_dataset",
				resource_class=getattr(stage_config, "bombcell_label_resource_class", None),
			)
		)
	if bool(getattr(stage_config, "merge_slay_enabled", False)):
		available_phases["merge_SLAy"] = (
			_SpikesortRuntimePhase(
				name="spikesort.merge_SLAy",
				phase_label="merge_SLAy",
				debug_enabled_attr="merge_slay_debug_mode_enabled",
				debug_limit_datasets_attr="merge_slay_debug_limit_datasets",
				debug_limit_wells_attr="merge_slay_debug_limit_wells",
				target_runner=_run_spikesort_merge_slay_target,
				debug_limit_wells_per_dataset_attr="merge_slay_debug_limit_wells_per_dataset",
				resource_class=getattr(stage_config, "merge_slay_resource_class", None),
			)
		)
	if bool(getattr(stage_config, "bombcell_label_pass2_enabled", False)):
		available_phases["bombcell_label_pass2"] = (
			_SpikesortRuntimePhase(
				name="spikesort.bombcell_label_pass2",
				phase_label="bombcell_label_pass2",
				debug_enabled_attr="bombcell_label_pass2_debug_mode_enabled",
				debug_limit_datasets_attr="bombcell_label_pass2_debug_limit_datasets",
				debug_limit_wells_attr="bombcell_label_pass2_debug_limit_wells",
				target_runner=_run_spikesort_bombcell_label_pass2_target,
				debug_limit_wells_per_dataset_attr="bombcell_label_pass2_debug_limit_wells_per_dataset",
				resource_class=getattr(stage_config, "bombcell_label_pass2_resource_class", None),
			)
		)
	if bool(getattr(stage_config, "cleanup_concat_binary_enabled", False)):
		available_phases["cleanup_concat_binary"] = (
			_SpikesortRuntimePhase(
				name="spikesort.cleanup_concat_binary",
				phase_label="cleanup_concat_binary",
				debug_enabled_attr="cleanup_concat_binary_debug_mode_enabled",
				debug_limit_datasets_attr="cleanup_concat_binary_debug_limit_datasets",
				debug_limit_wells_attr="cleanup_concat_binary_debug_limit_wells",
				target_runner=_run_spikesort_cleanup_concat_binary_target,
				debug_limit_wells_per_dataset_attr="cleanup_concat_binary_debug_limit_wells_per_dataset",
				resource_class=getattr(stage_config, "cleanup_concat_binary_resource_class", None),
			)
		)
	if bool(getattr(stage_config, "cleanup_analyzers_enabled", False)):
		available_phases["cleanup_analyzers"] = (
			_SpikesortRuntimePhase(
				name="spikesort.cleanup_analyzers",
				phase_label="cleanup_analyzers",
				debug_enabled_attr="cleanup_analyzers_debug_mode_enabled",
				debug_limit_datasets_attr="cleanup_analyzers_debug_limit_datasets",
				debug_limit_wells_attr="cleanup_analyzers_debug_limit_wells",
				target_runner=_run_spikesort_cleanup_analyzers_target,
				debug_limit_wells_per_dataset_attr="cleanup_analyzers_debug_limit_wells_per_dataset",
				resource_class=getattr(stage_config, "cleanup_analyzers_resource_class", None),
			)
		)
	configured_sequence = tuple(getattr(stage_config, "phase_sequence", None) or DEFAULT_SPIKESORT_PHASE_SEQUENCE)
	phase_plan: list[_SpikesortRuntimePhase] = []
	for phase_name in configured_sequence:
		canonical_phase_name = normalize_spikesort_phase_name(phase_name)
		phase = available_phases.get(canonical_phase_name)
		if phase is not None:
			phase_plan.append(phase)
	return phase_plan


def _apply_spikesort_runtime_phase_plan_debug_limits(
	*,
	stage_config: Any,
	targets: list[Any],
	phase_plan: list[_SpikesortRuntimePhase],
) -> list[Any]:
	_ = phase_plan
	return _apply_spikesort_stage_debug_limits(
		stage_name="spikesort",
		stage_config=stage_config,
		targets=list(targets),
	)


def _spikesort_output_rel_root(stage_config: Any) -> str:
	return str(getattr(stage_config, "output_rel_root", "spikesort_outputs") or "spikesort_outputs")


def _run_spikesort_concat_binary_target(*, target: Any, stage_config: Any, unit_workers: int) -> SpikesortResult:
	return build_spikesort_concat_binary(
		h5_path=target.h5_path,
		stream_id=target.stream_id,
		mea_output_root=target.mea_output_root,
		output_rel_root=_spikesort_output_rel_root(stage_config),
		stage_config=stage_config,
		force_restart=bool(getattr(stage_config, "force_restart", False)),
	)


def _run_spikesort_cleanup_concat_binary_target(*, target: Any, stage_config: Any, unit_workers: int) -> SpikesortResult:
	return cleanup_spikesort_concat_binary(
		h5_path=target.h5_path,
		stream_id=target.stream_id,
		mea_output_root=target.mea_output_root,
		output_rel_root=_spikesort_output_rel_root(stage_config),
		stage_config=stage_config,
		force_restart=bool(getattr(stage_config, "force_restart", False)),
	)


def _run_spikesort_cleanup_analyzers_target(*, target: Any, stage_config: Any, unit_workers: int) -> SpikesortResult:
	return cleanup_spikesort_analyzers(
		h5_path=target.h5_path,
		stream_id=target.stream_id,
		mea_output_root=target.mea_output_root,
		output_rel_root=_spikesort_output_rel_root(stage_config),
		stage_config=stage_config,
		force_restart=bool(getattr(stage_config, "force_restart", False)),
	)


def _run_spikesort_plot_concat_traces_target(*, target: Any, stage_config: Any, unit_workers: int) -> SpikesortResult:
	return run_spikesort_plot_concat_traces(
		h5_path=target.h5_path,
		stream_id=target.stream_id,
		mea_output_root=target.mea_output_root,
		output_rel_root=_spikesort_output_rel_root(stage_config),
		stage_config=stage_config,
		force_restart=bool(getattr(stage_config, "force_restart", False)),
	)


def _run_spikesort_plot_concat_channel_layout_target(*, target: Any, stage_config: Any, unit_workers: int) -> SpikesortResult:
	return run_spikesort_plot_concat_channel_layout(
		h5_path=target.h5_path,
		stream_id=target.stream_id,
		mea_output_root=target.mea_output_root,
		output_rel_root=_spikesort_output_rel_root(stage_config),
		stage_config=stage_config,
		force_restart=bool(getattr(stage_config, "force_restart", False)),
	)


def _analysis_output_rel_root(stage_config: Any) -> str:
	return str(getattr(stage_config, "output_rel_root", "analysis_outputs") or "analysis_outputs")


def _run_analysis_compute_metrics_target(*, target: Any, stage_config: Any, unit_workers: int) -> AnalysisResult:
	return run_analysis_metrics(
		dataset_index=int(target.dataset_index),
		dataset_id=str(getattr(target, "dataset_id", "") or "") or None,
		h5_path=target.h5_path,
		stream_id=target.stream_id,
		mea_output_root=target.mea_output_root,
		output_rel_root=_analysis_output_rel_root(stage_config),
		stage_config=stage_config,
		force_restart=bool(getattr(stage_config, "force_restart", False)),
	)


def _run_analysis_unitmatch_target(*, target: Any, stage_config: Any, unit_workers: int) -> Any:
	"""Slice 1 of unitmatch_phase_plan: scaffold runner that wraps the
	per-target orchestrator. Returns a noop result so the analysis stage
	can carry the phase in its default sequence without doing real
	work. Subsequent slices replace the orchestrator's body."""

	from .stages.analysis.orchestrators import run_analysis_unitmatch

	_ = unit_workers
	return run_analysis_unitmatch(
		dataset_index=int(target.dataset_index),
		dataset_id=str(getattr(target, "dataset_id", "") or "") or None,
		h5_path=target.h5_path,
		stream_id=target.stream_id,
		mea_output_root=target.mea_output_root,
		output_rel_root=_analysis_output_rel_root(stage_config),
		stage_config=stage_config,
		force_restart=bool(getattr(stage_config, "force_restart", False)),
	)


def _run_analysis_propagation_video_target(
	*, target: Any, stage_config: Any, unit_workers: int
) -> Any:
	"""Slice 2 of analysis_propagation_video_plan: scaffold runner.

	Wraps the per-target orchestrator. Returns a noop/skipped sentinel
	until slice 4 ships the real per-unit video render logic.
	"""

	from .stages.analysis.orchestrators import run_analysis_propagation_video

	_ = unit_workers
	return run_analysis_propagation_video(
		dataset_index=int(target.dataset_index),
		dataset_id=str(getattr(target, "dataset_id", "") or "") or None,
		h5_path=target.h5_path,
		stream_id=target.stream_id,
		mea_output_root=target.mea_output_root,
		output_rel_root=_analysis_output_rel_root(stage_config),
		stage_config=stage_config,
		force_restart=bool(getattr(stage_config, "force_restart", False)),
	)


def _run_spikesort_sort_target(*, target: Any, stage_config: Any, unit_workers: int) -> SpikesortResult:
	inputs = build_spikesort_inputs_for_target(
		target=target,
		stage_config=stage_config,
		unit_workers=int(unit_workers),
	)
	return run_spikesort(inputs)


def _run_spikesort_summarize_sort_target(*, target: Any, stage_config: Any, unit_workers: int) -> SpikesortResult:
	inputs = build_spikesort_inputs_for_target(
		target=target,
		stage_config=stage_config,
		unit_workers=int(unit_workers),
	)
	return summarize_spikesort(inputs)


def _run_spikesort_snapshot_sorter_output_target(*, target: Any, stage_config: Any, unit_workers: int) -> SpikesortResult:
	return snapshot_spikesort_sorter_output(
		h5_path=target.h5_path,
		stream_id=target.stream_id,
		mea_output_root=target.mea_output_root,
		output_rel_root=_spikesort_output_rel_root(stage_config),
		stage_config=stage_config,
		force_restart=bool(getattr(stage_config, "force_restart", False)),
	)


def _run_spikesort_concat_analyzer_target(*, target: Any, stage_config: Any, unit_workers: int) -> SpikesortResult:
	return build_spikesort_concat_analyzer(
		h5_path=target.h5_path,
		stream_id=target.stream_id,
		mea_output_root=target.mea_output_root,
		output_rel_root=_spikesort_output_rel_root(stage_config),
		stage_config=stage_config,
		force_restart=bool(getattr(stage_config, "force_restart", False)),
	)


def _run_spikesort_bombcell_label_target(*, target: Any, stage_config: Any, unit_workers: int) -> SpikesortBombcellResult:
	return run_spikesort_bombcell(
		h5_path=target.h5_path,
		stream_id=target.stream_id,
		mea_output_root=target.mea_output_root,
		output_rel_root=_spikesort_output_rel_root(stage_config),
		stage_config=stage_config,
		force_restart=bool(getattr(stage_config, "force_restart", False)),
	)


def _run_spikesort_bombcell_label_pass2_target(*, target: Any, stage_config: Any, unit_workers: int) -> SpikesortBombcellResult:
	return run_spikesort_bombcell_pass2(
		h5_path=target.h5_path,
		stream_id=target.stream_id,
		mea_output_root=target.mea_output_root,
		output_rel_root=_spikesort_output_rel_root(stage_config),
		stage_config=stage_config,
		force_restart=bool(getattr(stage_config, "force_restart", False)),
	)


def _run_spikesort_merge_slay_target(*, target: Any, stage_config: Any, unit_workers: int) -> SpikesortMergeResult:
	return run_spikesort_merge_slay(
		h5_path=target.h5_path,
		stream_id=target.stream_id,
		mea_output_root=target.mea_output_root,
		output_rel_root=_spikesort_output_rel_root(stage_config),
		stage_config=stage_config,
		force_restart=bool(getattr(stage_config, "merge_slay_force_restart", False)),
		replot=bool(getattr(stage_config, "merge_slay_replot", False)),
	)


def run_spikesort_sort_from_runtime(
	*,
	config_path: str,
	limit_segments_override: int | None = None,
	limit_datasets_override: int | None = None,
	target_datasets_override: list[int] | None = None,
	limit_wells_per_dataset_override: int | None = None,
	force_restart_override: bool | None = None,
	replot_override: bool | None = None,
	task_allocation_override: dict[str, Any] | None = None,
) -> MultiTargetStageResult:
	return _run_spikesort_sort_from_runtime(
		config_path=config_path,
		stage_name="spikesort.sort",
		limit_segments_override=limit_segments_override,
		limit_datasets_override=limit_datasets_override,
		target_datasets_override=target_datasets_override,
		limit_wells_per_dataset_override=limit_wells_per_dataset_override,
		force_restart_override=force_restart_override,
		replot_override=replot_override,
		task_allocation_override=task_allocation_override,
	)


def run_spikesort_summarize_sort_from_runtime(
	*,
	config_path: str,
	limit_segments_override: int | None = None,
	limit_datasets_override: int | None = None,
	target_datasets_override: list[int] | None = None,
	limit_wells_per_dataset_override: int | None = None,
	force_restart_override: bool | None = None,
	replot_override: bool | None = None,
	task_allocation_override: dict[str, Any] | None = None,
) -> MultiTargetStageResult:
	bundle: PipelineRuntimeBundle = load_pipeline_runtime_bundle(config_path=config_path)
	stage_config = parse_spikesort_stage_config(
		runtime_config=bundle.runtime_config,
		force_restart_override=force_restart_override,
		replot_override=replot_override,
	)
	stage_config = _with_debug_limit_overrides(
		stage_config,
		limit_segments_override=limit_segments_override,
		limit_datasets_override=limit_datasets_override,
		limit_wells_per_dataset_override=limit_wells_per_dataset_override,
	)
	targets = _select_execution_targets_with_debug_limits(
		bundle=bundle,
		stage_name="spikesort.summarize_sort",
		stage_config=stage_config,
		target_datasets=target_datasets_override,
	)
	targets = _apply_spikesort_stage_debug_limits(
		stage_name="spikesort",
		stage_config=stage_config,
		targets=list(targets),
	)
	targets = _apply_spikesort_phase_debug_limits(
		stage_name="spikesort.summarize_sort",
		stage_config=stage_config,
		targets=list(targets),
		phase_label="summarize_sort",
		enabled_attr="summarize_sort_debug_mode_enabled",
		limit_datasets_attr="summarize_sort_debug_limit_datasets",
		limit_wells_attr="summarize_sort_debug_limit_wells",
		limit_wells_per_dataset_attr="summarize_sort_debug_limit_wells_per_dataset",
	)
	phase_resource_classes = _spikesort_phase_resource_classes_from_labels(stage_config, ("summarize_sort",))
	parallelism = _resolve_runtime_stage_parallelism(
		bundle=bundle,
		stage_name="spikesort",
		target_count=len(targets),
		targets=targets,
		phase_resource_classes=phase_resource_classes,
		task_allocation_override=task_allocation_override,
	)
	n_jobs_source = _runtime_n_jobs_source(stage_config)
	resource_budget_manager = _build_stage_resource_budget_manager(
		bundle=bundle,
		parallelism=parallelism,
		phase_resource_classes=phase_resource_classes,
		target_count=len(targets),
	)

	def _worker(target):
		inputs = build_spikesort_inputs_for_target(
			target=target,
			stage_config=stage_config,
			unit_workers=int(parallelism.unit_workers),
		)
		_log_spikesort_phase_worker_allocation(
			phase_name="summarize_sort",
			target=target,
			parallelism=parallelism,
			n_jobs=_runtime_n_jobs_from_stage_config(stage_config, fallback_n_jobs=int(parallelism.unit_workers)),
			n_jobs_source=str(n_jobs_source),
		)
		return _run_direct_phase_with_resource_tracking(
			phase_name="summarize_sort",
			runner=lambda: summarize_spikesort(inputs),
			resource_class=_first_resource_class(phase_resource_classes),
			pipeline_thread_count=_runtime_n_jobs_from_stage_config(
				stage_config,
				fallback_n_jobs=int(parallelism.unit_workers),
			),
			target=target,
		)

	with stage_resource_budget_context(resource_budget_manager):
		target_results = _distribute_runtime_targets(
			targets=targets,
			parallelism=parallelism,
			worker_fn=_worker,
			stage_name="spikesort.summarize_sort",
			progress=PipelineProgress(ProgressSpec(label="spikesort.summarize_sort wells", total=len(targets), unit="well")),
			advance_progress_on_target_complete=True,
			bundle=bundle,
		)
	succeeded = sum(1 for item in target_results if item.status == "ok")
	failed = sum(1 for item in target_results if item.status != "ok")
	return MultiTargetStageResult(
		stage="spikesort.summarize_sort",
		total_targets=len(target_results),
		succeeded_targets=succeeded,
		failed_targets=failed,
		target_results=target_results,
	)


def _run_spikesort_concat_binary_phase_from_runtime(
	*,
	config_path: str,
	stage_name: str,
	runner_fn: Callable[..., SpikesortResult],
	limit_segments_override: int | None = None,
	limit_datasets_override: int | None = None,
	target_datasets_override: list[int] | None = None,
	limit_wells_per_dataset_override: int | None = None,
	force_restart_override: bool | None = None,
	replot_override: bool | None = None,
	task_allocation_override: dict[str, Any] | None = None,
	debug_phase_label: str,
	debug_enabled_attr: str,
	debug_limit_datasets_attr: str,
	debug_limit_wells_attr: str,
	debug_limit_wells_per_dataset_attr: str | None = None,
	publish_after_run: bool = False,
) -> MultiTargetStageResult:
	bundle: PipelineRuntimeBundle = load_pipeline_runtime_bundle(config_path=config_path)
	publish_policy = _resolve_publish_policy(runtime_config=bundle.runtime_config, data_config=bundle.data_config)
	if publish_after_run:
		_log_publish_policy(stage_name=stage_name, policy=publish_policy)
	stage_config = parse_spikesort_stage_config(
		runtime_config=bundle.runtime_config,
		force_restart_override=force_restart_override,
		replot_override=replot_override,
	)
	stage_config = _with_debug_limit_overrides(
		stage_config,
		limit_segments_override=limit_segments_override,
		limit_datasets_override=limit_datasets_override,
		limit_wells_per_dataset_override=limit_wells_per_dataset_override,
	)
	targets = _select_execution_targets_with_debug_limits(
		bundle=bundle,
		stage_name=stage_name,
		stage_config=stage_config,
		target_datasets=target_datasets_override,
	)
	targets = _apply_spikesort_stage_debug_limits(
		stage_name="spikesort",
		stage_config=stage_config,
		targets=list(targets),
	)
	targets = _apply_spikesort_phase_debug_limits(
		stage_name=stage_name,
		stage_config=stage_config,
		targets=list(targets),
		phase_label=debug_phase_label,
		enabled_attr=debug_enabled_attr,
		limit_datasets_attr=debug_limit_datasets_attr,
		limit_wells_attr=debug_limit_wells_attr,
		limit_wells_per_dataset_attr=debug_limit_wells_per_dataset_attr,
	)
	phase_resource_classes = _spikesort_phase_resource_classes_from_labels(stage_config, (str(debug_phase_label),))
	parallelism = _resolve_runtime_stage_parallelism(
		bundle=bundle,
		stage_name="spikesort",
		target_count=len(targets),
		targets=targets,
		phase_resource_classes=phase_resource_classes,
		task_allocation_override=task_allocation_override,
	)
	n_jobs_source = _runtime_n_jobs_source(stage_config)
	runtime_stage_config = _stage_config_with_runtime_n_jobs(
		stage_config,
		unit_workers=int(parallelism.unit_workers),
	)
	runtime_n_jobs = _runtime_n_jobs_from_stage_config(
		runtime_stage_config,
		fallback_n_jobs=int(parallelism.unit_workers),
	)
	resource_budget_manager = _build_stage_resource_budget_manager(
		bundle=bundle,
		parallelism=parallelism,
		phase_resource_classes=phase_resource_classes,
		target_count=len(targets),
	)

	def _worker(target):
		_log_spikesort_phase_worker_allocation(
			phase_name=str(debug_phase_label),
			target=target,
			parallelism=parallelism,
			n_jobs=int(runtime_n_jobs),
			n_jobs_source=str(n_jobs_source),
		)
		return _run_direct_phase_with_resource_tracking(
			phase_name=str(debug_phase_label),
			runner=lambda: runner_fn(
				h5_path=target.h5_path,
				stream_id=target.stream_id,
				mea_output_root=target.mea_output_root,
				output_rel_root=runtime_stage_config.output_rel_root,
				stage_config=runtime_stage_config,
				force_restart=bool(runtime_stage_config.force_restart),
			),
			resource_class=_first_resource_class(phase_resource_classes),
			pipeline_thread_count=int(runtime_n_jobs),
			target=target,
		)

	with stage_resource_budget_context(resource_budget_manager):
		target_results = _distribute_runtime_targets(
			targets=targets,
			parallelism=parallelism,
			worker_fn=_worker,
			stage_name=stage_name,
			progress=PipelineProgress(ProgressSpec(label=f"{stage_name} wells", total=len(targets), unit="well")),
			advance_progress_on_target_complete=True,
			bundle=bundle,
		)
	if publish_after_run:
		target_results = [_publish_spikesort_target_result(item, policy=publish_policy) for item in target_results]

	succeeded = sum(1 for item in target_results if item.status == "ok")
	failed = sum(1 for item in target_results if item.status != "ok")
	return MultiTargetStageResult(
		stage=stage_name,
		total_targets=len(target_results),
		succeeded_targets=succeeded,
		failed_targets=failed,
		target_results=target_results,
	)


def run_spikesort_concat_binary_from_runtime(
	*,
	config_path: str,
	limit_segments_override: int | None = None,
	limit_datasets_override: int | None = None,
	target_datasets_override: list[int] | None = None,
	limit_wells_per_dataset_override: int | None = None,
	force_restart_override: bool | None = None,
	replot_override: bool | None = None,
	task_allocation_override: dict[str, Any] | None = None,
) -> MultiTargetStageResult:
	return _run_spikesort_concat_binary_phase_from_runtime(
		config_path=config_path,
		stage_name="spikesort.concat_binary",
		runner_fn=build_spikesort_concat_binary,
		limit_segments_override=limit_segments_override,
		limit_datasets_override=limit_datasets_override,
		target_datasets_override=target_datasets_override,
		limit_wells_per_dataset_override=limit_wells_per_dataset_override,
		force_restart_override=force_restart_override,
		replot_override=replot_override,
		task_allocation_override=task_allocation_override,
		debug_phase_label="concat_binary",
		debug_enabled_attr="concat_binary_debug_mode_enabled",
		debug_limit_datasets_attr="concat_binary_debug_limit_datasets",
		debug_limit_wells_attr="concat_binary_debug_limit_wells",
		debug_limit_wells_per_dataset_attr="concat_binary_debug_limit_wells_per_dataset",
		publish_after_run=False,
	)


def run_spikesort_cleanup_concat_binary_from_runtime(
	*,
	config_path: str,
	limit_segments_override: int | None = None,
	limit_datasets_override: int | None = None,
	target_datasets_override: list[int] | None = None,
	limit_wells_per_dataset_override: int | None = None,
	force_restart_override: bool | None = None,
	replot_override: bool | None = None,
	task_allocation_override: dict[str, Any] | None = None,
) -> MultiTargetStageResult:
	return _run_spikesort_concat_binary_phase_from_runtime(
		config_path=config_path,
		stage_name="spikesort.cleanup_concat_binary",
		runner_fn=cleanup_spikesort_concat_binary,
		limit_segments_override=limit_segments_override,
		limit_datasets_override=limit_datasets_override,
		target_datasets_override=target_datasets_override,
		limit_wells_per_dataset_override=limit_wells_per_dataset_override,
		force_restart_override=force_restart_override,
		replot_override=replot_override,
		task_allocation_override=task_allocation_override,
		debug_phase_label="cleanup_concat_binary",
		debug_enabled_attr="cleanup_concat_binary_debug_mode_enabled",
		debug_limit_datasets_attr="cleanup_concat_binary_debug_limit_datasets",
		debug_limit_wells_attr="cleanup_concat_binary_debug_limit_wells",
		debug_limit_wells_per_dataset_attr="cleanup_concat_binary_debug_limit_wells_per_dataset",
		publish_after_run=True,
	)


def run_spikesort_plot_concat_traces_from_runtime(
	*,
	config_path: str,
	limit_segments_override: int | None = None,
	limit_datasets_override: int | None = None,
	target_datasets_override: list[int] | None = None,
	limit_wells_per_dataset_override: int | None = None,
	force_restart_override: bool | None = None,
	replot_override: bool | None = None,
	task_allocation_override: dict[str, Any] | None = None,
) -> MultiTargetStageResult:
	return _run_spikesort_concat_binary_phase_from_runtime(
		config_path=config_path,
		stage_name="spikesort.plot_concat_traces",
		runner_fn=run_spikesort_plot_concat_traces,
		limit_segments_override=limit_segments_override,
		limit_datasets_override=limit_datasets_override,
		target_datasets_override=target_datasets_override,
		limit_wells_per_dataset_override=limit_wells_per_dataset_override,
		force_restart_override=force_restart_override,
		replot_override=replot_override,
		task_allocation_override=task_allocation_override,
		debug_phase_label="plot_concat_traces",
		debug_enabled_attr="plot_concat_traces_debug_mode_enabled",
		debug_limit_datasets_attr="plot_concat_traces_debug_limit_datasets",
		debug_limit_wells_attr="plot_concat_traces_debug_limit_wells",
		debug_limit_wells_per_dataset_attr="plot_concat_traces_debug_limit_wells_per_dataset",
		publish_after_run=False,
	)


def run_spikesort_plot_concat_channel_layout_from_runtime(
	*,
	config_path: str,
	limit_segments_override: int | None = None,
	limit_datasets_override: int | None = None,
	target_datasets_override: list[int] | None = None,
	limit_wells_per_dataset_override: int | None = None,
	force_restart_override: bool | None = None,
	replot_override: bool | None = None,
	task_allocation_override: dict[str, Any] | None = None,
) -> MultiTargetStageResult:
	return _run_spikesort_concat_binary_phase_from_runtime(
		config_path=config_path,
		stage_name="spikesort.plot_concat_channel_layout",
		runner_fn=run_spikesort_plot_concat_channel_layout,
		limit_segments_override=limit_segments_override,
		limit_datasets_override=limit_datasets_override,
		target_datasets_override=target_datasets_override,
		limit_wells_per_dataset_override=limit_wells_per_dataset_override,
		force_restart_override=force_restart_override,
		replot_override=replot_override,
		task_allocation_override=task_allocation_override,
		debug_phase_label="plot_concat_channel_layout",
		debug_enabled_attr="plot_concat_channel_layout_debug_mode_enabled",
		debug_limit_datasets_attr="plot_concat_channel_layout_debug_limit_datasets",
		debug_limit_wells_attr="plot_concat_channel_layout_debug_limit_wells",
		debug_limit_wells_per_dataset_attr="plot_concat_channel_layout_debug_limit_wells_per_dataset",
		publish_after_run=False,
	)


def run_spikesort_cleanup_analyzers_from_runtime(
	*,
	config_path: str,
	limit_segments_override: int | None = None,
	limit_datasets_override: int | None = None,
	target_datasets_override: list[int] | None = None,
	limit_wells_per_dataset_override: int | None = None,
	force_restart_override: bool | None = None,
	replot_override: bool | None = None,
	task_allocation_override: dict[str, Any] | None = None,
) -> MultiTargetStageResult:
	return _run_spikesort_concat_binary_phase_from_runtime(
		config_path=config_path,
		stage_name="spikesort.cleanup_analyzers",
		runner_fn=cleanup_spikesort_analyzers,
		limit_segments_override=limit_segments_override,
		limit_datasets_override=limit_datasets_override,
		target_datasets_override=target_datasets_override,
		limit_wells_per_dataset_override=limit_wells_per_dataset_override,
		force_restart_override=force_restart_override,
		replot_override=replot_override,
		task_allocation_override=task_allocation_override,
		debug_phase_label="cleanup_analyzers",
		debug_enabled_attr="cleanup_analyzers_debug_mode_enabled",
		debug_limit_datasets_attr="cleanup_analyzers_debug_limit_datasets",
		debug_limit_wells_attr="cleanup_analyzers_debug_limit_wells",
		debug_limit_wells_per_dataset_attr="cleanup_analyzers_debug_limit_wells_per_dataset",
		publish_after_run=True,
	)


def run_spikesort_snapshot_sorter_output_from_runtime(
	*,
	config_path: str,
	limit_segments_override: int | None = None,
	limit_datasets_override: int | None = None,
	target_datasets_override: list[int] | None = None,
	limit_wells_per_dataset_override: int | None = None,
	force_restart_override: bool | None = None,
	replot_override: bool | None = None,
	task_allocation_override: dict[str, Any] | None = None,
) -> MultiTargetStageResult:
	return _run_spikesort_concat_binary_phase_from_runtime(
		config_path=config_path,
		stage_name="spikesort.snapshot_sorter_output",
		runner_fn=snapshot_spikesort_sorter_output,
		limit_segments_override=limit_segments_override,
		limit_datasets_override=limit_datasets_override,
		target_datasets_override=target_datasets_override,
		limit_wells_per_dataset_override=limit_wells_per_dataset_override,
		force_restart_override=force_restart_override,
		replot_override=replot_override,
		task_allocation_override=task_allocation_override,
		debug_phase_label="snapshot_sorter_output",
		debug_enabled_attr="snapshot_sorter_output_debug_mode_enabled",
		debug_limit_datasets_attr="snapshot_sorter_output_debug_limit_datasets",
		debug_limit_wells_attr="snapshot_sorter_output_debug_limit_wells",
		debug_limit_wells_per_dataset_attr="snapshot_sorter_output_debug_limit_wells_per_dataset",
		publish_after_run=False,
	)


def run_spikesort_concat_analyzer_from_runtime(
	*,
	config_path: str,
	limit_segments_override: int | None = None,
	limit_datasets_override: int | None = None,
	target_datasets_override: list[int] | None = None,
	limit_wells_per_dataset_override: int | None = None,
	force_restart_override: bool | None = None,
	replot_override: bool | None = None,
	task_allocation_override: dict[str, Any] | None = None,
) -> MultiTargetStageResult:
	return _run_spikesort_concat_binary_phase_from_runtime(
		config_path=config_path,
		stage_name="spikesort.concat_analyzer",
		runner_fn=build_spikesort_concat_analyzer,
		limit_segments_override=limit_segments_override,
		limit_datasets_override=limit_datasets_override,
		target_datasets_override=target_datasets_override,
		limit_wells_per_dataset_override=limit_wells_per_dataset_override,
		force_restart_override=force_restart_override,
		replot_override=replot_override,
		task_allocation_override=task_allocation_override,
		debug_phase_label="concat_analyzer",
		debug_enabled_attr="concat_analyzer_debug_mode_enabled",
		debug_limit_datasets_attr="concat_analyzer_debug_limit_datasets",
		debug_limit_wells_attr="concat_analyzer_debug_limit_wells",
		debug_limit_wells_per_dataset_attr="concat_analyzer_debug_limit_wells_per_dataset",
		publish_after_run=False,
	)


def run_spikesort_restore_sorter_output_from_runtime(
	*,
	config_path: str,
	limit_segments_override: int | None = None,
	limit_datasets_override: int | None = None,
	target_datasets_override: list[int] | None = None,
	limit_wells_per_dataset_override: int | None = None,
	force_restart_override: bool | None = None,
	replot_override: bool | None = None,
	task_allocation_override: dict[str, Any] | None = None,
) -> MultiTargetStageResult:
	return _run_spikesort_concat_binary_phase_from_runtime(
		config_path=config_path,
		stage_name="spikesort.restore_sorter_output",
		runner_fn=restore_spikesort_sorter_output,
		limit_segments_override=limit_segments_override,
		limit_datasets_override=limit_datasets_override,
		target_datasets_override=target_datasets_override,
		limit_wells_per_dataset_override=limit_wells_per_dataset_override,
		force_restart_override=force_restart_override,
		replot_override=replot_override,
		task_allocation_override=task_allocation_override,
		debug_phase_label="restore_sorter_output",
		debug_enabled_attr="restore_sorter_output_debug_mode_enabled",
		debug_limit_datasets_attr="restore_sorter_output_debug_limit_datasets",
		debug_limit_wells_attr="restore_sorter_output_debug_limit_wells",
		debug_limit_wells_per_dataset_attr="restore_sorter_output_debug_limit_wells_per_dataset",
		publish_after_run=False,
	)


def _parse_analysis_stage_config_for_runtime(
	*,
	bundle: PipelineRuntimeBundle,
	force_restart_override: bool | None,
	replot_override: bool | None,
) -> AnalysisStageConfig:
	return parse_analysis_stage_config(
		runtime_config=bundle.runtime_config,
		data_config=bundle.data_config,
		force_restart_override=force_restart_override,
		replot_override=replot_override,
	)


def _analysis_phase_resource_classes_from_labels(
	stage_config: Any,
	phase_labels: tuple[str, ...],
) -> tuple[str, ...]:
	resource_classes: list[str] = []
	for phase_label in phase_labels:
		attr = _ANALYSIS_RESOURCE_ATTR_BY_PHASE_LABEL.get(str(phase_label).strip())
		if attr is None:
			continue
		resource_class = getattr(stage_config, attr, None)
		if resource_class is not None and str(resource_class).strip():
			resource_classes.append(str(resource_class).strip())
	return _unique_resource_classes(tuple(resource_classes))


def _run_analysis_phase_from_runtime(
	*,
	config_path: str,
	stage_name: str,
	phase_label: str,
	target_runner: Callable[..., AnalysisResult],
	limit_segments_override: int | None,
	limit_datasets_override: int | None,
	target_datasets_override: list[int] | None,
	limit_wells_per_dataset_override: int | None,
	force_restart_override: bool | None,
	replot_override: bool | None,
	task_allocation_override: dict[str, Any] | None,
) -> MultiTargetStageResult:
	bundle: PipelineRuntimeBundle = load_pipeline_runtime_bundle(config_path=config_path)
	stage_config = _parse_analysis_stage_config_for_runtime(
		bundle=bundle,
		force_restart_override=force_restart_override,
		replot_override=replot_override,
	)
	stage_config = _with_debug_limit_overrides(
		stage_config,
		limit_segments_override=limit_segments_override,
		limit_datasets_override=limit_datasets_override,
		limit_wells_per_dataset_override=limit_wells_per_dataset_override,
	)
	targets = _select_execution_targets_with_debug_limits(
		bundle=bundle,
		stage_name=stage_name,
		stage_config=stage_config,
		target_datasets=target_datasets_override,
	)
	phase_resource_classes = _analysis_phase_resource_classes_from_labels(stage_config, (str(phase_label),))
	parallelism = _resolve_runtime_stage_parallelism(
		bundle=bundle,
		stage_name="analysis",
		target_count=len(targets),
		targets=targets,
		phase_resource_classes=phase_resource_classes,
		task_allocation_override=task_allocation_override,
	)
	resource_budget_manager = _build_stage_resource_budget_manager(
		bundle=bundle,
		parallelism=parallelism,
		phase_resource_classes=phase_resource_classes,
		target_count=len(targets),
	)

	# Slice 14c (target-level auto-restart skip) for analysis: when neither
	# --force-restart nor --replot is set, short-circuit targets whose
	# per-phase summary on disk already shows the phase is complete.
	# Each per-phase orchestrator (compute_metrics, unitmatch) writes a
	# per-target marker whose ``status`` reads "ok" / "skipped" / "noop"
	# when the work is done; reading that summary here saves the per-target
	# resource budget acquisition + log spin-up cost for already-done
	# targets at cohort scale.
	from .stages.analysis.runner import target_analysis_phase_summary_ok

	bypass_auto_restart_skip = bool(
		getattr(stage_config, "force_restart", False)
		or getattr(stage_config, "replot", False)
	)

	def _worker(target):
		if not bypass_auto_restart_skip and target_analysis_phase_summary_ok(
			phase_name=str(phase_label),
			target=target,
			stage_config=stage_config,
		):
			return {
				"stage": str(stage_name),
				"status": "skipped",
				"reason": f"{phase_label}_already_ok",
				"phase": str(phase_label),
			}
		return _run_direct_phase_with_resource_tracking(
			phase_name=str(phase_label),
			runner=lambda: target_runner(
				target=target,
				stage_config=stage_config,
				unit_workers=int(parallelism.unit_workers),
			),
			resource_class=_first_resource_class(phase_resource_classes),
			pipeline_thread_count=int(parallelism.unit_workers),
			target=target,
		)

	with stage_resource_budget_context(resource_budget_manager):
		target_results = _distribute_runtime_targets(
			targets=targets,
			parallelism=parallelism,
			worker_fn=_worker,
			stage_name=stage_name,
			progress=PipelineProgress(ProgressSpec(label=f"{stage_name} wells", total=len(targets), unit="well")),
			advance_progress_on_target_complete=True,
			bundle=bundle,
		)
	succeeded = sum(1 for item in target_results if item.status == "ok")
	failed = sum(1 for item in target_results if item.status != "ok")
	return MultiTargetStageResult(
		stage=stage_name,
		total_targets=len(target_results),
		succeeded_targets=succeeded,
		failed_targets=failed,
		target_results=target_results,
	)


def run_analysis_unitmatch_from_runtime(
	*,
	config_path: str,
	limit_segments_override: int | None = None,
	limit_datasets_override: int | None = None,
	target_datasets_override: list[int] | None = None,
	limit_wells_per_dataset_override: int | None = None,
	force_restart_override: bool | None = None,
	replot_override: bool | None = None,
	task_allocation_override: dict[str, Any] | None = None,
) -> MultiTargetStageResult:
	"""Slice 1 of unitmatch_phase_plan: runtime entry point for the
	analysis.unitmatch phase. Mirrors `run_analysis_compute_metrics_from_runtime`'s
	plumbing so the analysis runner can dispatch it uniformly. The actual
	orchestrator is a scaffold; slices 2-5 fill in the real work."""

	return _run_analysis_phase_from_runtime(
		config_path=config_path,
		stage_name="analysis.unitmatch",
		phase_label="unitmatch",
		target_runner=_run_analysis_unitmatch_target,
		limit_segments_override=limit_segments_override,
		limit_datasets_override=limit_datasets_override,
		target_datasets_override=target_datasets_override,
		limit_wells_per_dataset_override=limit_wells_per_dataset_override,
		force_restart_override=force_restart_override,
		replot_override=replot_override,
		task_allocation_override=task_allocation_override,
	)


def run_analysis_propagation_video_from_runtime(
	*,
	config_path: str,
	limit_segments_override: int | None = None,
	limit_datasets_override: int | None = None,
	target_datasets_override: list[int] | None = None,
	limit_wells_per_dataset_override: int | None = None,
	force_restart_override: bool | None = None,
	replot_override: bool | None = None,
	task_allocation_override: dict[str, Any] | None = None,
) -> MultiTargetStageResult:
	"""Slice 2 of analysis_propagation_video_plan: runtime entry point.

	Mirrors `run_analysis_unitmatch_from_runtime`'s plumbing. The actual
	orchestrator is a scaffold (noop when disabled; "skipped:
	not_implemented_yet" when enabled); slice 4 ports the per-unit
	video render logic identified in the slice-1 archeology audit.
	"""

	return _run_analysis_phase_from_runtime(
		config_path=config_path,
		stage_name="analysis.propagation_video",
		phase_label="propagation_video",
		target_runner=_run_analysis_propagation_video_target,
		limit_segments_override=limit_segments_override,
		limit_datasets_override=limit_datasets_override,
		target_datasets_override=target_datasets_override,
		limit_wells_per_dataset_override=limit_wells_per_dataset_override,
		force_restart_override=force_restart_override,
		replot_override=replot_override,
		task_allocation_override=task_allocation_override,
	)


def run_analysis_compute_metrics_from_runtime(
	*,
	config_path: str,
	limit_segments_override: int | None = None,
	limit_datasets_override: int | None = None,
	target_datasets_override: list[int] | None = None,
	limit_wells_per_dataset_override: int | None = None,
	force_restart_override: bool | None = None,
	replot_override: bool | None = None,
	task_allocation_override: dict[str, Any] | None = None,
) -> MultiTargetStageResult:
	return _run_analysis_phase_from_runtime(
		config_path=config_path,
		stage_name="analysis.compute_metrics",
		phase_label="compute_metrics",
		target_runner=_run_analysis_compute_metrics_target,
		limit_segments_override=limit_segments_override,
		limit_datasets_override=limit_datasets_override,
		target_datasets_override=target_datasets_override,
		limit_wells_per_dataset_override=limit_wells_per_dataset_override,
		force_restart_override=force_restart_override,
		replot_override=replot_override,
		task_allocation_override=task_allocation_override,
	)


def run_analysis_from_runtime(
	*,
	config_path: str,
	limit_segments_override: int | None = None,
	limit_datasets_override: int | None = None,
	target_datasets_override: list[int] | None = None,
	limit_wells_per_dataset_override: int | None = None,
	force_restart_override: bool | None = None,
	replot_override: bool | None = None,
	task_allocation_override: dict[str, Any] | None = None,
) -> MultiTargetStageResult:
	bundle: PipelineRuntimeBundle = load_pipeline_runtime_bundle(config_path=config_path)
	stage_config = _parse_analysis_stage_config_for_runtime(
		bundle=bundle,
		force_restart_override=force_restart_override,
		replot_override=replot_override,
	)
	phase_sequence = tuple(getattr(stage_config, "phase_sequence", ()) or ("compute_metrics",))
	phase_sequence = _filter_phase_sequence_for_replot(
		phase_sequence,
		replot=bool(replot_override),
		stage_name="analysis",
	)
	results: list[TargetStageResult] = []
	final_stage = "analysis"
	total_targets = 0
	succeeded = 0
	failed = 0
	for phase_label in phase_sequence:
		stage_name = f"analysis.{phase_label}"
		if phase_label == "compute_metrics":
			phase_result = run_analysis_compute_metrics_from_runtime(
				config_path=config_path,
				limit_segments_override=limit_segments_override,
				limit_datasets_override=limit_datasets_override,
				target_datasets_override=target_datasets_override,
				limit_wells_per_dataset_override=limit_wells_per_dataset_override,
				force_restart_override=force_restart_override,
				replot_override=replot_override,
				task_allocation_override=task_allocation_override,
			)
		elif phase_label == "unitmatch":
			# Slice 1 of unitmatch_phase_plan: scaffold. The phase is a
			# no-op when disabled in YAML; even when enabled, the slice-1
			# orchestrator returns a scaffold marker until slice 2+ ships.
			phase_result = run_analysis_unitmatch_from_runtime(
				config_path=config_path,
				limit_segments_override=limit_segments_override,
				limit_datasets_override=limit_datasets_override,
				target_datasets_override=target_datasets_override,
				limit_wells_per_dataset_override=limit_wells_per_dataset_override,
				force_restart_override=force_restart_override,
				replot_override=replot_override,
				task_allocation_override=task_allocation_override,
			)
		elif phase_label == "propagation_video":
			# Slice 2 of analysis_propagation_video_plan: scaffold. The
			# phase is a no-op when disabled in YAML; the orchestrator
			# returns a `skipped: not_implemented_yet` marker when
			# enabled until slice 4 ports the per-unit video render
			# logic from the slice-1 archeology audit.
			phase_result = run_analysis_propagation_video_from_runtime(
				config_path=config_path,
				limit_segments_override=limit_segments_override,
				limit_datasets_override=limit_datasets_override,
				target_datasets_override=target_datasets_override,
				limit_wells_per_dataset_override=limit_wells_per_dataset_override,
				force_restart_override=force_restart_override,
				replot_override=replot_override,
				task_allocation_override=task_allocation_override,
			)
		else:
			raise ValueError(f"Unknown analysis phase: {phase_label}")
		results.extend(phase_result.target_results)
		total_targets += phase_result.total_targets
		succeeded += phase_result.succeeded_targets
		failed += phase_result.failed_targets
	return MultiTargetStageResult(
		stage=final_stage,
		total_targets=total_targets,
		succeeded_targets=succeeded,
		failed_targets=failed,
		target_results=results,
	)


def _init_copy_phase_enabled(stage_config: InitStageConfig) -> bool:
	"""True iff `copy_src_to_scratch` is enabled AND listed in `phase_sequence`."""

	if not bool(stage_config.enabled):
		return False
	if "copy_src_to_scratch" not in tuple(stage_config.phase_sequence):
		return False
	return bool(getattr(stage_config.phases.copy_src_to_scratch, "enabled", False))


def _select_init_execution_targets(
	*,
	bundle: PipelineRuntimeBundle,
	stage_config: InitStageConfig,
	target_datasets: list[int] | None = None,
	limit_datasets: int | None = None,
	limit_wells_per_dataset: int | None = None,
) -> list[ExecutionTarget]:
	"""Build execution targets for the init stage.

	`materialize_scratch_inputs` is True iff the configured `copy_src_to_scratch`
	phase is going to run — that's the same contract preprocess uses (and the
	contract `select_execution_targets` ultimately enforces by triggering the
	on-disk copy). All other init-stage invocations operate on whatever h5
	paths the data config already exposes.
	"""

	kwargs: dict[str, Any] = {
		"bundle": bundle,
		"materialize_scratch_inputs": bool(_init_copy_phase_enabled(stage_config)),
	}
	if target_datasets is not None:
		kwargs["target_datasets"] = list(target_datasets)
	if limit_datasets is not None:
		kwargs["limit_datasets"] = int(limit_datasets)
	if limit_wells_per_dataset is not None:
		kwargs["limit_wells_per_dataset"] = int(limit_wells_per_dataset)
	return select_execution_targets(**kwargs)


def run_init_copy_src_to_scratch_from_runtime(
	*,
	config_path: str,
	limit_segments_override: int | None = None,
	limit_datasets_override: int | None = None,
	target_datasets_override: list[int] | None = None,
	limit_wells_per_dataset_override: int | None = None,
	force_restart_override: bool | None = None,
	replot_override: bool | None = None,
	task_allocation_override: dict[str, Any] | None = None,
) -> MultiTargetStageResult:
	"""Run just the init `copy_src_to_scratch` phase end-to-end.

	Modeled on `run_analysis_compute_metrics_from_runtime`: load the runtime
	bundle, parse the init stage config, select execution targets with scratch
	materialization enabled, then iterate per-target and call the orchestrator.
	`limit_segments_override` and `task_allocation_override` are unused (init
	doesn't have segments and is single-CPU per target), kept for signature
	parity with sibling `run_*_from_runtime` entries.
	"""

	del limit_segments_override
	del task_allocation_override

	bundle: PipelineRuntimeBundle = load_pipeline_runtime_bundle(config_path=config_path)
	stage_config = parse_init_stage_config(
		runtime_config=bundle.runtime_config,
		data_config=bundle.data_config,
		force_restart_override=force_restart_override,
		replot_override=replot_override,
	)
	# Force the copy phase enabled when invoked through this explicit handler:
	# the user typed `init.copy_src_to_scratch`, so honor it even if the YAML's
	# stage-level `enabled` flag is false. This matches the preprocess substage
	# handler semantics (which materialize_scratch_inputs via stage_name=='preprocess.copy_src_to_scratch').
	from dataclasses import replace as _replace
	from .stages.init.models.inputs import InitCopySrcToScratchPhaseConfig, InitPhasesConfig

	current_copy_phase = stage_config.phases.copy_src_to_scratch
	stage_config = _replace(
		stage_config,
		enabled=True,
		phase_sequence=("copy_src_to_scratch",),
		phases=InitPhasesConfig(
			copy_src_to_scratch=InitCopySrcToScratchPhaseConfig(
				enabled=True,
				requires_use_scratch_root=bool(current_copy_phase.requires_use_scratch_root),
				summary_json_relpath=str(current_copy_phase.summary_json_relpath),
				resource_class=current_copy_phase.resource_class,
			),
		),
	)
	targets = _select_init_execution_targets(
		bundle=bundle,
		stage_config=stage_config,
		target_datasets=target_datasets_override,
		limit_datasets=limit_datasets_override,
		limit_wells_per_dataset=limit_wells_per_dataset_override,
	)
	return run_init_stage(stage_config, targets=list(targets))


def run_init_from_runtime(
	*,
	config_path: str,
	limit_segments_override: int | None = None,
	limit_datasets_override: int | None = None,
	target_datasets_override: list[int] | None = None,
	limit_wells_per_dataset_override: int | None = None,
	force_restart_override: bool | None = None,
	replot_override: bool | None = None,
	task_allocation_override: dict[str, Any] | None = None,
) -> MultiTargetStageResult:
	"""Run the init stage's full phase_sequence end-to-end.

	Walks `stage_config.phase_sequence` and dispatches each phase to the
	corresponding `run_init_<phase>_from_runtime` helper, accumulating results
	into a single `MultiTargetStageResult`. Mirrors `run_analysis_from_runtime`'s
	shape so the CLI plumbing stays uniform.

	Slice 5 ships a single phase (`copy_src_to_scratch`); the loop scales to
	additional once-per-data-config phases without restructuring.
	"""

	bundle: PipelineRuntimeBundle = load_pipeline_runtime_bundle(config_path=config_path)
	stage_config = parse_init_stage_config(
		runtime_config=bundle.runtime_config,
		data_config=bundle.data_config,
		force_restart_override=force_restart_override,
		replot_override=replot_override,
	)

	if not stage_config.enabled or not stage_config.phase_sequence:
		# No-op early so we don't pointlessly load+select targets when the
		# stage is disabled or has nothing to do.
		return run_init_stage(stage_config)

	stage_phase_sequence = _filter_phase_sequence_for_replot(
		tuple(stage_config.phase_sequence),
		replot=bool(replot_override),
		stage_name="init",
	)
	results: list[TargetStageResult] = []
	final_stage = "init"
	total_targets = 0
	succeeded = 0
	failed = 0
	for phase_label in stage_phase_sequence:
		stage_name = f"init.{phase_label}"
		if phase_label == "copy_src_to_scratch":
			phase_result = run_init_copy_src_to_scratch_from_runtime(
				config_path=config_path,
				limit_segments_override=limit_segments_override,
				limit_datasets_override=limit_datasets_override,
				target_datasets_override=target_datasets_override,
				limit_wells_per_dataset_override=limit_wells_per_dataset_override,
				force_restart_override=force_restart_override,
				replot_override=replot_override,
				task_allocation_override=task_allocation_override,
			)
		else:
			raise ValueError(f"Unknown init phase: {phase_label}")
		results.extend(phase_result.target_results)
		total_targets += phase_result.total_targets
		succeeded += phase_result.succeeded_targets
		failed += phase_result.failed_targets
	return MultiTargetStageResult(
		stage=final_stage,
		total_targets=total_targets,
		succeeded_targets=succeeded,
		failed_targets=failed,
		target_results=results,
	)


def _cleanup_wipe_phase_enabled(stage_config: CleanupStageConfig) -> bool:
	"""True iff `wipe_src_scratch` is enabled AND listed in `phase_sequence`."""

	if not bool(stage_config.enabled):
		return False
	if "wipe_src_scratch" not in tuple(stage_config.phase_sequence):
		return False
	return bool(getattr(stage_config.phases.wipe_src_scratch, "enabled", False))


def _select_cleanup_execution_targets(
	*,
	bundle: PipelineRuntimeBundle,
	stage_config: CleanupStageConfig,
	target_datasets: list[int] | None = None,
	limit_datasets: int | None = None,
	limit_wells_per_dataset: int | None = None,
) -> list[ExecutionTarget]:
	"""Build execution targets for the cleanup stage.

	`materialize_scratch_inputs=False`: cleanup operates on whatever h5 paths
	the data config (+ init stage's prior copy) already exposes. It never
	triggers a fresh copy — it only deletes what's already in scratch.
	"""

	kwargs: dict[str, Any] = {
		"bundle": bundle,
		"materialize_scratch_inputs": False,
	}
	if target_datasets is not None:
		kwargs["target_datasets"] = list(target_datasets)
	if limit_datasets is not None:
		kwargs["limit_datasets"] = int(limit_datasets)
	if limit_wells_per_dataset is not None:
		kwargs["limit_wells_per_dataset"] = int(limit_wells_per_dataset)
	del stage_config  # currently unused; kept for signature parity with init's variant
	return select_execution_targets(**kwargs)


def run_cleanup_wipe_src_scratch_from_runtime(
	*,
	config_path: str,
	limit_segments_override: int | None = None,
	limit_datasets_override: int | None = None,
	target_datasets_override: list[int] | None = None,
	limit_wells_per_dataset_override: int | None = None,
	force_restart_override: bool | None = None,
	replot_override: bool | None = None,
	task_allocation_override: dict[str, Any] | None = None,
) -> MultiTargetStageResult:
	"""Run just the cleanup `wipe_src_scratch` phase end-to-end.

	Mirrors `run_init_copy_src_to_scratch_from_runtime` (slice 5): load the
	runtime bundle, parse the cleanup stage config, force the wipe phase
	enabled (the user typed `cleanup.wipe_src_scratch` explicitly so honor it
	even when the stage-level `enabled` flag is false), then iterate per
	target and call the orchestrator. `limit_segments_override` and
	`task_allocation_override` are unused (cleanup has no segments and is
	single-CPU per target).
	"""

	del limit_segments_override
	del task_allocation_override

	bundle: PipelineRuntimeBundle = load_pipeline_runtime_bundle(config_path=config_path)
	stage_config = parse_cleanup_stage_config(
		runtime_config=bundle.runtime_config,
		data_config=bundle.data_config,
		force_restart_override=force_restart_override,
		replot_override=replot_override,
	)
	from dataclasses import replace as _replace
	from .stages.cleanup.models.inputs import CleanupPhasesConfig, CleanupWipeSrcScratchPhaseConfig

	current_wipe_phase = stage_config.phases.wipe_src_scratch
	stage_config = _replace(
		stage_config,
		enabled=True,
		phase_sequence=("wipe_src_scratch",),
		phases=CleanupPhasesConfig(
			wipe_src_scratch=CleanupWipeSrcScratchPhaseConfig(
				enabled=True,
				dry_run=bool(current_wipe_phase.dry_run),
				requires_use_scratch_root=bool(current_wipe_phase.requires_use_scratch_root),
				summary_json_relpath=str(current_wipe_phase.summary_json_relpath),
				resource_class=current_wipe_phase.resource_class,
			),
		),
	)
	targets = _select_cleanup_execution_targets(
		bundle=bundle,
		stage_config=stage_config,
		target_datasets=target_datasets_override,
		limit_datasets=limit_datasets_override,
		limit_wells_per_dataset=limit_wells_per_dataset_override,
	)
	return run_cleanup_stage(stage_config, targets=list(targets))


def run_cleanup_from_runtime(
	*,
	config_path: str,
	limit_segments_override: int | None = None,
	limit_datasets_override: int | None = None,
	target_datasets_override: list[int] | None = None,
	limit_wells_per_dataset_override: int | None = None,
	force_restart_override: bool | None = None,
	replot_override: bool | None = None,
	task_allocation_override: dict[str, Any] | None = None,
) -> MultiTargetStageResult:
	"""Run the cleanup stage's full phase_sequence end-to-end.

	Walks `stage_config.phase_sequence` and dispatches each phase to the
	corresponding `run_cleanup_<phase>_from_runtime` helper. Mirrors
	`run_init_from_runtime`'s shape so the CLI plumbing stays uniform.

	Slice 6 ships a single phase (`wipe_src_scratch`); the loop scales to the
	additional end-of-run cleanup phases planned in `tech_debt.md`
	§"Finalize the phase roster" without restructuring.
	"""

	bundle: PipelineRuntimeBundle = load_pipeline_runtime_bundle(config_path=config_path)
	stage_config = parse_cleanup_stage_config(
		runtime_config=bundle.runtime_config,
		data_config=bundle.data_config,
		force_restart_override=force_restart_override,
		replot_override=replot_override,
	)

	if not stage_config.enabled or not stage_config.phase_sequence:
		# No-op early so we don't pointlessly load+select targets when the
		# stage is disabled or has nothing to do.
		return run_cleanup_stage(stage_config)

	stage_phase_sequence = _filter_phase_sequence_for_replot(
		tuple(stage_config.phase_sequence),
		replot=bool(replot_override),
		stage_name="cleanup",
	)
	results: list[TargetStageResult] = []
	final_stage = "cleanup"
	total_targets = 0
	succeeded = 0
	failed = 0
	for phase_label in stage_phase_sequence:
		stage_name = f"cleanup.{phase_label}"
		if phase_label == "wipe_src_scratch":
			phase_result = run_cleanup_wipe_src_scratch_from_runtime(
				config_path=config_path,
				limit_segments_override=limit_segments_override,
				limit_datasets_override=limit_datasets_override,
				target_datasets_override=target_datasets_override,
				limit_wells_per_dataset_override=limit_wells_per_dataset_override,
				force_restart_override=force_restart_override,
				replot_override=replot_override,
				task_allocation_override=task_allocation_override,
			)
		else:
			raise ValueError(f"Unknown cleanup phase: {phase_label}")
		results.extend(phase_result.target_results)
		total_targets += phase_result.total_targets
		succeeded += phase_result.succeeded_targets
		failed += phase_result.failed_targets
	return MultiTargetStageResult(
		stage=final_stage,
		total_targets=total_targets,
		succeeded_targets=succeeded,
		failed_targets=failed,
		target_results=results,
	)


def _run_spikesort_sort_from_runtime(
	*,
	config_path: str,
	stage_name: str,
	limit_segments_override: int | None = None,
	limit_datasets_override: int | None = None,
	target_datasets_override: list[int] | None = None,
	limit_wells_per_dataset_override: int | None = None,
	force_restart_override: bool | None = None,
	replot_override: bool | None = None,
	task_allocation_override: dict[str, Any] | None = None,
) -> MultiTargetStageResult:
	bundle: PipelineRuntimeBundle = load_pipeline_runtime_bundle(config_path=config_path)
	publish_policy = _resolve_publish_policy(runtime_config=bundle.runtime_config, data_config=bundle.data_config)
	_log_publish_policy(stage_name=stage_name, policy=publish_policy)
	stage_config = parse_spikesort_stage_config(
		runtime_config=bundle.runtime_config,
		force_restart_override=force_restart_override,
		replot_override=replot_override,
	)
	stage_config = _with_debug_limit_overrides(
		stage_config,
		limit_segments_override=limit_segments_override,
		limit_datasets_override=limit_datasets_override,
		limit_wells_per_dataset_override=limit_wells_per_dataset_override,
	)
	targets = _select_execution_targets_with_debug_limits(
		bundle=bundle,
		stage_name=stage_name,
		stage_config=stage_config,
		target_datasets=target_datasets_override,
	)
	targets = _apply_spikesort_stage_debug_limits(
		stage_name="spikesort",
		stage_config=stage_config,
		targets=list(targets),
	)
	targets = _apply_spikesort_sort_debug_limits(
		stage_name=stage_name,
		stage_config=stage_config,
		targets=list(targets),
	)
	phase_resource_classes = _spikesort_phase_resource_classes_from_labels(stage_config, ("sort",))
	parallelism = _resolve_runtime_stage_parallelism(
		bundle=bundle,
		stage_name="spikesort",
		target_count=len(targets),
		targets=targets,
		phase_resource_classes=phase_resource_classes,
		task_allocation_override=task_allocation_override,
	)
	n_jobs_source = _runtime_n_jobs_source(stage_config)
	resource_budget_manager = _build_stage_resource_budget_manager(
		bundle=bundle,
		parallelism=parallelism,
		phase_resource_classes=phase_resource_classes,
		target_count=len(targets),
	)

	def _worker(target):
		inputs = build_spikesort_inputs_for_target(
			target=target,
			stage_config=stage_config,
			unit_workers=int(parallelism.unit_workers),
		)
		_log_spikesort_phase_worker_allocation(
			phase_name="sort",
			target=target,
			parallelism=parallelism,
			n_jobs=_runtime_n_jobs_from_stage_config(stage_config, fallback_n_jobs=int(parallelism.unit_workers)),
			n_jobs_source=str(n_jobs_source),
		)

		def _run_sort() -> SpikesortResult:
			return _run_direct_phase_with_resource_tracking(
				phase_name="sort",
				runner=lambda: run_spikesort(inputs),
				resource_class=_first_resource_class(phase_resource_classes),
				pipeline_thread_count=int(_runtime_n_jobs_from_stage_config(stage_config, fallback_n_jobs=int(parallelism.unit_workers))),
				target=target,
			)

		return _run_spikesort_phase_with_optional_sort_gate(
			phase_label="sort",
			target=target,
			runner=_run_sort,
			sort_phase_gate=None,
		)

	with stage_resource_budget_context(resource_budget_manager):
		target_results = _distribute_runtime_targets(
			targets=targets,
			parallelism=parallelism,
			worker_fn=_worker,
			stage_name=stage_name,
			progress=PipelineProgress(ProgressSpec(label=f"{stage_name} wells", total=len(targets), unit="well")),
			advance_progress_on_target_complete=True,
			bundle=bundle,
		)
	target_results = [_publish_spikesort_target_result(item, policy=publish_policy) for item in target_results]

	succeeded = sum(1 for item in target_results if item.status == "ok")
	failed = sum(1 for item in target_results if item.status != "ok")
	return MultiTargetStageResult(
		stage=stage_name,
		total_targets=len(target_results),
		succeeded_targets=succeeded,
		failed_targets=failed,
		target_results=target_results,
	)


def run_spikesort_merge_from_runtime(
	*,
	config_path: str,
	limit_segments_override: int | None = None,
	limit_datasets_override: int | None = None,
	target_datasets_override: list[int] | None = None,
	limit_wells_per_dataset_override: int | None = None,
	force_restart_override: bool | None = None,
	replot_override: bool | None = None,
	task_allocation_override: dict[str, Any] | None = None,
	merge_sequence_override: tuple[str, ...] | list[str] | None = None,
	stage_config_transformer: Callable[[Any], Any] | None = None,
	debug_phase_label: str | None = None,
	debug_enabled_attr: str | None = None,
	debug_limit_datasets_attr: str | None = None,
	debug_limit_wells_attr: str | None = None,
	debug_limit_wells_per_dataset_attr: str | None = None,
	stage_name: str = "spikesort.merge",
) -> MultiTargetStageResult:
	bundle: PipelineRuntimeBundle = load_pipeline_runtime_bundle(config_path=config_path)
	publish_policy = _resolve_publish_policy(runtime_config=bundle.runtime_config, data_config=bundle.data_config)
	_log_publish_policy(stage_name=stage_name, policy=publish_policy)
	stage_config = parse_spikesort_stage_config(
		runtime_config=bundle.runtime_config,
		force_restart_override=force_restart_override,
		replot_override=replot_override,
	)
	if merge_sequence_override is not None:
		normalized_override = tuple(str(token).strip() for token in tuple(merge_sequence_override) if str(token).strip())
		if normalized_override:
			stage_config = replace(stage_config, merge_sequence=normalized_override)
	if stage_config_transformer is not None:
		stage_config = stage_config_transformer(stage_config)
	stage_config = _with_debug_limit_overrides(
		stage_config,
		limit_segments_override=limit_segments_override,
		limit_datasets_override=limit_datasets_override,
		limit_wells_per_dataset_override=limit_wells_per_dataset_override,
	)

	inherit_2panel_probe_dimensions = bool(
		getattr(stage_config, "merge_reports_2panel_inherit_probe_dimensions", False)
	)
	inherit_template_heatmap_probe_dimensions = bool(
		getattr(stage_config, "merge_reports_template_heatmaps_inherit_probe_dimensions", False)
	)
	if inherit_2panel_probe_dimensions or inherit_template_heatmap_probe_dimensions:
		probe_geometry = parse_probe_geometry_from_data_config(data_config=bundle.data_config)
		if probe_geometry is not None:
			replace_kwargs: dict[str, float | None] = {}
			if inherit_2panel_probe_dimensions:
				existing_x = getattr(stage_config, "merge_reports_2panel_probe_dim_x_um", None)
				existing_y = getattr(stage_config, "merge_reports_2panel_probe_dim_y_um", None)
				resolved_x = existing_x
				resolved_y = existing_y
				if resolved_x is None:
					resolved_x = getattr(probe_geometry, "active_area_um_x", None)
				if resolved_y is None:
					resolved_y = getattr(probe_geometry, "active_area_um_y", None)
				replace_kwargs["merge_reports_2panel_probe_dim_x_um"] = (
					float(resolved_x) if resolved_x is not None else None
				)
				replace_kwargs["merge_reports_2panel_probe_dim_y_um"] = (
					float(resolved_y) if resolved_y is not None else None
				)
			if inherit_template_heatmap_probe_dimensions:
				existing_x = getattr(stage_config, "merge_reports_template_heatmaps_probe_dim_x_um", None)
				existing_y = getattr(stage_config, "merge_reports_template_heatmaps_probe_dim_y_um", None)
				resolved_x = existing_x
				resolved_y = existing_y
				if resolved_x is None:
					resolved_x = getattr(probe_geometry, "active_area_um_x", None)
				if resolved_y is None:
					resolved_y = getattr(probe_geometry, "active_area_um_y", None)
				replace_kwargs["merge_reports_template_heatmaps_probe_dim_x_um"] = (
					float(resolved_x) if resolved_x is not None else None
				)
				replace_kwargs["merge_reports_template_heatmaps_probe_dim_y_um"] = (
					float(resolved_y) if resolved_y is not None else None
				)
				existing_pitch = getattr(stage_config, "merge_reports_template_heatmaps_probe_pitch_um", None)
				existing_electrode_x = getattr(
					stage_config,
					"merge_reports_template_heatmaps_probe_electrode_size_um_x",
					None,
				)
				existing_electrode_y = getattr(
					stage_config,
					"merge_reports_template_heatmaps_probe_electrode_size_um_y",
					None,
				)
				resolved_pitch = existing_pitch
				resolved_electrode_x = existing_electrode_x
				resolved_electrode_y = existing_electrode_y
				if resolved_pitch is None:
					resolved_pitch = getattr(probe_geometry, "pitch_um", None)
				if resolved_electrode_x is None:
					resolved_electrode_x = getattr(probe_geometry, "electrode_size_um_x", None)
				if resolved_electrode_y is None:
					resolved_electrode_y = getattr(probe_geometry, "electrode_size_um_y", None)
				replace_kwargs["merge_reports_template_heatmaps_probe_pitch_um"] = (
					float(resolved_pitch) if resolved_pitch is not None else None
				)
				replace_kwargs["merge_reports_template_heatmaps_probe_electrode_size_um_x"] = (
					float(resolved_electrode_x) if resolved_electrode_x is not None else None
				)
				replace_kwargs["merge_reports_template_heatmaps_probe_electrode_size_um_y"] = (
					float(resolved_electrode_y) if resolved_electrode_y is not None else None
				)
			if replace_kwargs:
				if getattr(stage_config, "__dataclass_fields__", None) is not None:
					stage_config = replace(stage_config, **replace_kwargs)
				else:
					for field_name, field_value in replace_kwargs.items():
						setattr(stage_config, field_name, field_value)
	targets = _select_execution_targets_with_debug_limits(
		bundle=bundle,
		stage_name=stage_name,
		stage_config=stage_config,
		target_datasets=target_datasets_override,
	)
	targets = _apply_spikesort_stage_debug_limits(
		stage_name="spikesort",
		stage_config=stage_config,
		targets=list(targets),
	)
	if (
		debug_phase_label is not None
		and debug_enabled_attr is not None
		and debug_limit_datasets_attr is not None
		and debug_limit_wells_attr is not None
	):
		targets = _apply_spikesort_phase_debug_limits(
			stage_name=stage_name,
			stage_config=stage_config,
			targets=list(targets),
			phase_label=debug_phase_label,
			enabled_attr=debug_enabled_attr,
			limit_datasets_attr=debug_limit_datasets_attr,
			limit_wells_attr=debug_limit_wells_attr,
			limit_wells_per_dataset_attr=debug_limit_wells_per_dataset_attr,
		)
	phase_resource_classes = _spikesort_phase_resource_classes_from_labels(stage_config, (str(debug_phase_label),))
	parallelism = _resolve_runtime_stage_parallelism(
		bundle=bundle,
		stage_name="spikesort",
		target_count=len(targets),
		targets=targets,
		phase_resource_classes=phase_resource_classes,
		task_allocation_override=task_allocation_override,
	)
	n_jobs_source = _runtime_n_jobs_source(stage_config)
	runtime_stage_config = _stage_config_with_runtime_n_jobs(
		stage_config,
		unit_workers=int(parallelism.unit_workers),
	)
	runtime_n_jobs = _runtime_n_jobs_from_stage_config(
		runtime_stage_config,
		fallback_n_jobs=int(parallelism.unit_workers),
	)
	resource_budget_manager = _build_stage_resource_budget_manager(
		bundle=bundle,
		parallelism=parallelism,
		phase_resource_classes=phase_resource_classes,
		target_count=len(targets),
	)

	def _worker(target):
		_log_spikesort_phase_worker_allocation(
			phase_name=str(stage_name).split(".", 1)[1] if "." in str(stage_name) else str(stage_name),
			target=target,
			parallelism=parallelism,
			n_jobs=int(runtime_n_jobs),
			n_jobs_source=str(n_jobs_source),
		)
		return _run_direct_phase_with_resource_tracking(
			phase_name=str(debug_phase_label or "merge"),
			runner=lambda: run_spikesort_merge(
				h5_path=target.h5_path,
				stream_id=target.stream_id,
				mea_output_root=target.mea_output_root,
				output_rel_root=runtime_stage_config.output_rel_root,
				stage_config=runtime_stage_config,
				force_restart=bool(
					getattr(
						runtime_stage_config,
						"merge_force_restart",
						bool(runtime_stage_config.force_restart),
					)
				),
				replot=bool(
					getattr(
						runtime_stage_config,
						"merge_replot",
						bool(runtime_stage_config.replot),
					)
				)
			),
			resource_class=_first_resource_class(phase_resource_classes),
			pipeline_thread_count=int(runtime_n_jobs),
			target=target,
		)

	with stage_resource_budget_context(resource_budget_manager):
		target_results = _distribute_runtime_targets(
			targets=targets,
			parallelism=parallelism,
			worker_fn=_worker,
			stage_name=stage_name,
			progress=PipelineProgress(ProgressSpec(label=f"{stage_name} wells", total=len(targets), unit="well")),
			advance_progress_on_target_complete=True,
			bundle=bundle,
		)
	target_results = [_publish_spikesort_merge_target_result(item, policy=publish_policy) for item in target_results]

	succeeded = sum(1 for item in target_results if item.status == "ok")
	failed = sum(1 for item in target_results if item.status != "ok")
	return MultiTargetStageResult(
		stage=stage_name,
		total_targets=len(target_results),
		succeeded_targets=succeeded,
		failed_targets=failed,
		target_results=target_results,
	)


def run_spikesort_bombcell_label_from_runtime(
	*,
	config_path: str,
	limit_segments_override: int | None = None,
	limit_datasets_override: int | None = None,
	target_datasets_override: list[int] | None = None,
	limit_wells_per_dataset_override: int | None = None,
	force_restart_override: bool | None = None,
	replot_override: bool | None = None,
	task_allocation_override: dict[str, Any] | None = None,
	stage_name: str = "spikesort.bombcell_label",
) -> MultiTargetStageResult:
	bundle: PipelineRuntimeBundle = load_pipeline_runtime_bundle(config_path=config_path)
	publish_policy = _resolve_publish_policy(runtime_config=bundle.runtime_config, data_config=bundle.data_config)
	_log_publish_policy(stage_name=stage_name, policy=publish_policy)
	stage_config = parse_spikesort_stage_config(
		runtime_config=bundle.runtime_config,
		force_restart_override=force_restart_override,
		replot_override=replot_override,
	)
	stage_config = _with_debug_limit_overrides(
		stage_config,
		limit_segments_override=limit_segments_override,
		limit_datasets_override=limit_datasets_override,
		limit_wells_per_dataset_override=limit_wells_per_dataset_override,
	)
	targets = _select_execution_targets_with_debug_limits(
		bundle=bundle,
		stage_name=stage_name,
		stage_config=stage_config,
		target_datasets=target_datasets_override,
	)
	targets = _apply_spikesort_stage_debug_limits(
		stage_name="spikesort",
		stage_config=stage_config,
		targets=list(targets),
	)
	targets = _apply_spikesort_phase_debug_limits(
		stage_name=stage_name,
		stage_config=stage_config,
		targets=list(targets),
		phase_label="bombcell_label",
		enabled_attr="bombcell_label_debug_mode_enabled",
		limit_datasets_attr="bombcell_label_debug_limit_datasets",
		limit_wells_attr="bombcell_label_debug_limit_wells",
		limit_wells_per_dataset_attr="bombcell_label_debug_limit_wells_per_dataset",
	)
	phase_resource_classes = _spikesort_phase_resource_classes_from_labels(stage_config, ("bombcell_label",))
	parallelism = _resolve_runtime_stage_parallelism(
		bundle=bundle,
		stage_name="spikesort",
		target_count=len(targets),
		targets=targets,
		phase_resource_classes=phase_resource_classes,
		task_allocation_override=task_allocation_override,
	)
	n_jobs_source = _runtime_n_jobs_source(stage_config)
	runtime_stage_config = _stage_config_with_runtime_n_jobs(
		stage_config,
		unit_workers=int(parallelism.unit_workers),
	)
	runtime_n_jobs = _runtime_n_jobs_from_stage_config(
		runtime_stage_config,
		fallback_n_jobs=int(parallelism.unit_workers),
	)
	resource_budget_manager = _build_stage_resource_budget_manager(
		bundle=bundle,
		parallelism=parallelism,
		phase_resource_classes=phase_resource_classes,
		target_count=len(targets),
	)

	def _worker(target):
		_log_spikesort_phase_worker_allocation(
			phase_name="bombcell_label",
			target=target,
			parallelism=parallelism,
			n_jobs=int(runtime_n_jobs),
			n_jobs_source=str(n_jobs_source),
		)
		return _run_direct_phase_with_resource_tracking(
			phase_name="bombcell_label",
			runner=lambda: run_spikesort_bombcell(
				h5_path=target.h5_path,
				stream_id=target.stream_id,
				mea_output_root=target.mea_output_root,
				output_rel_root=runtime_stage_config.output_rel_root,
				stage_config=runtime_stage_config,
				force_restart=bool(runtime_stage_config.force_restart),
			),
			resource_class=_first_resource_class(phase_resource_classes),
			pipeline_thread_count=int(runtime_n_jobs),
			target=target,
		)

	with stage_resource_budget_context(resource_budget_manager):
		target_results = _distribute_runtime_targets(
			targets=targets,
			parallelism=parallelism,
			worker_fn=_worker,
			stage_name=stage_name,
			progress=PipelineProgress(ProgressSpec(label=f"{stage_name} wells", total=len(targets), unit="well")),
			advance_progress_on_target_complete=True,
			bundle=bundle,
		)
	target_results = [
		_publish_spikesort_bombcell_target_result(item, policy=publish_policy)
		for item in target_results
	]

	succeeded = sum(1 for item in target_results if item.status == "ok")
	failed = sum(1 for item in target_results if item.status != "ok")
	return MultiTargetStageResult(
		stage=stage_name,
		total_targets=len(target_results),
		succeeded_targets=succeeded,
		failed_targets=failed,
		target_results=target_results,
	)


def run_spikesort_bombcell_label_pass2_from_runtime(
	*,
	config_path: str,
	limit_segments_override: int | None = None,
	limit_datasets_override: int | None = None,
	target_datasets_override: list[int] | None = None,
	limit_wells_per_dataset_override: int | None = None,
	force_restart_override: bool | None = None,
	replot_override: bool | None = None,
	task_allocation_override: dict[str, Any] | None = None,
	stage_name: str = "spikesort.bombcell_label_pass2",
) -> MultiTargetStageResult:
	bundle: PipelineRuntimeBundle = load_pipeline_runtime_bundle(config_path=config_path)
	publish_policy = _resolve_publish_policy(runtime_config=bundle.runtime_config, data_config=bundle.data_config)
	_log_publish_policy(stage_name=stage_name, policy=publish_policy)
	stage_config = parse_spikesort_stage_config(
		runtime_config=bundle.runtime_config,
		force_restart_override=force_restart_override,
		replot_override=replot_override,
	)
	stage_config = _with_debug_limit_overrides(
		stage_config,
		limit_segments_override=limit_segments_override,
		limit_datasets_override=limit_datasets_override,
		limit_wells_per_dataset_override=limit_wells_per_dataset_override,
	)
	targets = _select_execution_targets_with_debug_limits(
		bundle=bundle,
		stage_name=stage_name,
		stage_config=stage_config,
		target_datasets=target_datasets_override,
	)
	targets = _apply_spikesort_stage_debug_limits(
		stage_name="spikesort",
		stage_config=stage_config,
		targets=list(targets),
	)
	targets = _apply_spikesort_phase_debug_limits(
		stage_name=stage_name,
		stage_config=stage_config,
		targets=list(targets),
		phase_label="bombcell_label_pass2",
		enabled_attr="bombcell_label_pass2_debug_mode_enabled",
		limit_datasets_attr="bombcell_label_pass2_debug_limit_datasets",
		limit_wells_attr="bombcell_label_pass2_debug_limit_wells",
		limit_wells_per_dataset_attr="bombcell_label_pass2_debug_limit_wells_per_dataset",
	)
	phase_resource_classes = _spikesort_phase_resource_classes_from_labels(
		stage_config, ("bombcell_label_pass2",)
	)
	parallelism = _resolve_runtime_stage_parallelism(
		bundle=bundle,
		stage_name="spikesort",
		target_count=len(targets),
		targets=targets,
		phase_resource_classes=phase_resource_classes,
		task_allocation_override=task_allocation_override,
	)
	n_jobs_source = _runtime_n_jobs_source(stage_config)
	runtime_stage_config = _stage_config_with_runtime_n_jobs(
		stage_config,
		unit_workers=int(parallelism.unit_workers),
	)
	runtime_n_jobs = _runtime_n_jobs_from_stage_config(
		runtime_stage_config,
		fallback_n_jobs=int(parallelism.unit_workers),
	)
	resource_budget_manager = _build_stage_resource_budget_manager(
		bundle=bundle,
		parallelism=parallelism,
		phase_resource_classes=phase_resource_classes,
		target_count=len(targets),
	)

	def _worker(target):
		_log_spikesort_phase_worker_allocation(
			phase_name="bombcell_label_pass2",
			target=target,
			parallelism=parallelism,
			n_jobs=int(runtime_n_jobs),
			n_jobs_source=str(n_jobs_source),
		)
		return _run_direct_phase_with_resource_tracking(
			phase_name="bombcell_label_pass2",
			runner=lambda: run_spikesort_bombcell_pass2(
				h5_path=target.h5_path,
				stream_id=target.stream_id,
				mea_output_root=target.mea_output_root,
				output_rel_root=runtime_stage_config.output_rel_root,
				stage_config=runtime_stage_config,
				force_restart=bool(runtime_stage_config.force_restart),
			),
			resource_class=_first_resource_class(phase_resource_classes),
			pipeline_thread_count=int(runtime_n_jobs),
			target=target,
		)

	with stage_resource_budget_context(resource_budget_manager):
		target_results = _distribute_runtime_targets(
			targets=targets,
			parallelism=parallelism,
			worker_fn=_worker,
			stage_name=stage_name,
			progress=PipelineProgress(ProgressSpec(label=f"{stage_name} wells", total=len(targets), unit="well")),
			advance_progress_on_target_complete=True,
			bundle=bundle,
		)
	target_results = [
		_publish_spikesort_bombcell_target_result(item, policy=publish_policy)
		for item in target_results
	]

	succeeded = sum(1 for item in target_results if item.status == "ok")
	failed = sum(1 for item in target_results if item.status != "ok")
	return MultiTargetStageResult(
		stage=stage_name,
		total_targets=len(target_results),
		succeeded_targets=succeeded,
		failed_targets=failed,
		target_results=target_results,
	)


def _raise_reconstruct_unit_failures(*, stage_name: str, result: object) -> object:
	if isinstance(result, ReconstructionResult):
		ok_units = [u for u in result.units if str(getattr(u, "status", "ok")).strip().lower() == "ok"]
		failed_units = [u for u in result.units if str(getattr(u, "status", "ok")).strip().lower() != "ok"]
		if ok_units:
			return result
		if failed_units:
			first = failed_units[0]
			raise RuntimeError(
				f"{stage_name} unit failures: "
				f"succeeded={len(ok_units)}/{len(result.units)} "
				f"failed={len(failed_units)}/{len(result.units)} "
				f"first_unit={getattr(first, 'unit_id', 'unknown')} "
				f"first_error={getattr(first, 'error', None) or getattr(first, 'status', 'error')}"
			)
		raise RuntimeError(f"{stage_name} produced no successful units")

	if isinstance(result, dict):
		if bool(result.get("skipped", False)):
			return result
		has_unit_fields = any(key in result for key in ("units_ok", "units_error", "units"))
		if not has_unit_fields:
			return result
		units_ok = int(result.get("units_ok", 0) or 0)
		units_error = int(result.get("units_error", 0) or 0)
		if units_ok > 0:
			return result
		units = list(result.get("units", []) or [])
		if units_error > 0 and units:
			first = units[0] if isinstance(units[0], dict) else {}
			raise RuntimeError(
				f"{stage_name} unit failures: "
				f"succeeded={units_ok}/{len(units)} "
				f"failed={units_error}/{len(units)} "
				f"first_unit={first.get('unit_id', 'unknown')} "
				f"first_error={first.get('error') or first.get('status', 'error')}"
			)
		raise RuntimeError(f"{stage_name} produced no successful units")

	return result


def _run_reconstruct_substage_from_runtime(
	*,
	config_path: str,
	stage_name: str,
	runner_fn: Callable[[Any], Any],
	unit_id_override: int | None = None,
	unit_ids_override: list[int] | None = None,
	unit_limit_override: int | None = None,
	limit_segments_override: int | None = None,
	limit_datasets_override: int | None = None,
	target_datasets_override: list[int] | None = None,
	limit_wells_per_dataset_override: int | None = None,
	force_restart_override: bool | None = None,
	replot_override: bool | None = None,
	task_allocation_override: dict[str, Any] | None = None,
	publish_outputs: bool = False,
) -> MultiTargetStageResult:
	bundle: PipelineRuntimeBundle = load_pipeline_runtime_bundle(config_path=config_path)
	publish_policy = _resolve_publish_policy(runtime_config=bundle.runtime_config, data_config=bundle.data_config)
	if publish_outputs:
		_log_publish_policy(stage_name=stage_name, policy=publish_policy)
	probe_geometry = parse_probe_geometry_from_data_config(data_config=bundle.data_config)
	stage_config = parse_reconstruction_stage_config(
		runtime_config=bundle.runtime_config,
		unit_id_override=unit_id_override,
		unit_ids_override=unit_ids_override,
		unit_limit_override=unit_limit_override,
		limit_segments_override=limit_segments_override,
		force_restart_override=force_restart_override,
		replot_override=replot_override,
	)
	stage_config = _with_debug_limit_overrides(
		stage_config,
		limit_segments_override=limit_segments_override,
		limit_datasets_override=limit_datasets_override,
		limit_wells_per_dataset_override=limit_wells_per_dataset_override,
	)
	if str(stage_name).strip() == "reconstruct":
		stage_config = _apply_replot_phase_filter(
			stage_config,
			replot=bool(replot_override),
			stage_name="reconstruct",
		)
	targets = _select_execution_targets_with_debug_limits(
		bundle=bundle,
		stage_name=stage_name,
		stage_config=stage_config,
		target_datasets=target_datasets_override,
	)
	reconstruct_templates_config = None
	if callable(getattr(bundle.runtime_config, "get", None)):
		templates_runtime_config = build_reconstruct_templates_runtime_config(bundle.runtime_config)
		reconstruct_templates_config = parse_reconstruct_templates_config(
			runtime_config=templates_runtime_config,
			probe_geometry=probe_geometry,
			unit_id_override=unit_id_override,
			unit_ids_override=unit_ids_override,
			unit_limit_override=stage_config.unit_limit,
			limit_segments_override=stage_config.limit_segments,
			force_restart_override=force_restart_override,
			replot_override=replot_override,
		)
		reconstruct_templates_config = _with_debug_limit_overrides(
			reconstruct_templates_config,
			limit_segments_override=limit_segments_override,
			limit_datasets_override=limit_datasets_override,
			limit_wells_per_dataset_override=limit_wells_per_dataset_override,
		)
		if str(stage_name).strip() == "reconstruct":
			reconstruct_templates_config = _apply_replot_phase_filter(
				reconstruct_templates_config,
				replot=bool(replot_override),
				stage_name="reconstruct.templates",
			)
	if bool(getattr(stage_config, "debug_mode_enabled", False)):
		targets = _apply_spikesort_debug_target_limits(
			stage_name=stage_name,
			targets=list(targets),
			limit_datasets=getattr(stage_config, "debug_limit_datasets", None),
			limit_wells=getattr(stage_config, "debug_limit_wells", None),
			limit_wells_per_dataset=getattr(stage_config, "debug_limit_wells_per_dataset", None),
		)
	phase_resource_classes = _reconstruct_runtime_phase_resource_classes(
		stage_config,
		reconstruct_templates_config,
		stage_name=stage_name,
	)
	parallelism = _resolve_runtime_stage_parallelism(
		bundle=bundle,
		stage_name="reconstruct",
		target_count=len(targets),
		targets=targets,
		phase_resource_classes=phase_resource_classes,
		task_allocation_override=task_allocation_override,
	)
	if str(stage_name).strip() == "reconstruct":
		_log_runtime_stage_topology(stage_name="reconstruct", targets=list(targets), parallelism=parallelism)
	resource_budget_manager = _build_stage_resource_budget_manager(
		bundle=bundle,
		parallelism=parallelism,
		phase_resource_classes=phase_resource_classes,
		target_count=len(targets),
	)

	from .stages.reconstruct.runner import target_all_reconstruct_phases_succeeded

	bypass_auto_restart_skip = bool(
		getattr(stage_config, "force_restart", False)
		or getattr(stage_config, "replot", False)
	)

	def _worker(target):
		inputs = build_reconstruction_inputs_for_target(
			target=target,
			stage_config=stage_config,
			unit_workers=int(parallelism.unit_workers),
			probe_geometry=probe_geometry,
		)
		if reconstruct_templates_config is not None:
			templates_inputs = build_templates_inputs_for_target(
				target=target,
				stage_config=reconstruct_templates_config,
				unit_workers=int(parallelism.unit_workers),
				probe_geometry=probe_geometry,
			)
			inputs = replace(inputs, templates_inputs=templates_inputs)
		if str(stage_name).strip() == "reconstruct":
			# Slice 14c (target-level auto-restart skip) — only fires for
			# the full-stage dispatch path. Individual phase substage entry
			# points (e.g. reconstruct.axon_velocity_gtrs) keep their own
			# per-phase per-target idempotency via slice 13's
			# `with_checkpoint_marker`.
			if not bypass_auto_restart_skip and target_all_reconstruct_phases_succeeded(inputs):
				return {
					"stage": "reconstruct",
					"status": "skipped",
					"reason": "all_phases_ok",
				}
			result = runner_fn(inputs)
		else:
			direct_phase_name = str(stage_name).split(".", 1)[1] if "." in str(stage_name) else str(stage_name)
			resource_class = _reconstruct_stage_phase_resource_class(inputs, direct_phase_name)
			LOGGER.info(
				"reconstruct phase worker allocation: phase=%s resource_class=%s n_jobs=%d",
				_display_reconstruct_stage_phase_name(direct_phase_name),
				str(resource_class or "none"),
				int(parallelism.unit_workers),
			)
			result = _run_direct_phase_with_resource_tracking(
				phase_name=_display_reconstruct_stage_phase_name(direct_phase_name),
				runner=lambda: runner_fn(inputs),
				resource_class=resource_class,
				pipeline_thread_count=int(parallelism.unit_workers),
				target=target,
			)
		return _raise_reconstruct_unit_failures(stage_name=stage_name, result=result)

	with stage_resource_budget_context(resource_budget_manager):
		target_results = _distribute_runtime_targets(
			targets=targets,
			parallelism=parallelism,
			worker_fn=_worker,
			stage_name=stage_name,
			progress=_reconstruct_unit_progress(stage_name),
			bundle=bundle,
		)
	if publish_outputs:
		target_results = [_publish_reconstruct_target_result(item, policy=publish_policy) for item in target_results]

	succeeded = sum(1 for item in target_results if item.status == "ok")
	failed = sum(1 for item in target_results if item.status != "ok")
	return MultiTargetStageResult(
		stage=stage_name,
		total_targets=len(target_results),
		succeeded_targets=succeeded,
		failed_targets=failed,
		target_results=target_results,
	)


def run_reconstruct_from_runtime(
	*,
	config_path: str,
	unit_id_override: int | None = None,
	unit_ids_override: list[int] | None = None,
	unit_limit_override: int | None = None,
	limit_segments_override: int | None = None,
	limit_datasets_override: int | None = None,
	target_datasets_override: list[int] | None = None,
	limit_wells_per_dataset_override: int | None = None,
	force_restart_override: bool | None = None,
	replot_override: bool | None = None,
	task_allocation_override: dict[str, Any] | None = None,
) -> MultiTargetStageResult:
	return _run_reconstruct_substage_from_runtime(
		config_path=config_path,
		stage_name="reconstruct",
		runner_fn=run_reconstruct,
		unit_id_override=unit_id_override,
		unit_ids_override=unit_ids_override,
		unit_limit_override=unit_limit_override,
		limit_segments_override=limit_segments_override,
		limit_datasets_override=limit_datasets_override,
		target_datasets_override=target_datasets_override,
		limit_wells_per_dataset_override=limit_wells_per_dataset_override,
		force_restart_override=force_restart_override,
		replot_override=replot_override,
		task_allocation_override=task_allocation_override,
		publish_outputs=True,
	)


def run_reconstruct_templates_resolve_sources_from_runtime(
	*,
	config_path: str,
	unit_id_override: int | None = None,
	unit_ids_override: list[int] | None = None,
	unit_limit_override: int | None = None,
	limit_segments_override: int | None = None,
	limit_datasets_override: int | None = None,
	target_datasets_override: list[int] | None = None,
	limit_wells_per_dataset_override: int | None = None,
	force_restart_override: bool | None = None,
	replot_override: bool | None = None,
	task_allocation_override: dict[str, Any] | None = None,
) -> MultiTargetStageResult:
	return _run_reconstruct_substage_from_runtime(
		config_path=config_path,
		stage_name="reconstruct.resolve_sources",
		runner_fn=run_reconstruct_templates_resolve_sources,
		unit_id_override=unit_id_override,
		unit_ids_override=unit_ids_override,
		unit_limit_override=unit_limit_override,
		limit_segments_override=limit_segments_override,
		limit_datasets_override=limit_datasets_override,
		target_datasets_override=target_datasets_override,
		limit_wells_per_dataset_override=limit_wells_per_dataset_override,
		force_restart_override=force_restart_override,
		replot_override=replot_override,
		task_allocation_override=task_allocation_override,
	)


def run_reconstruct_templates_analyzers_from_runtime(
	*,
	config_path: str,
	unit_id_override: int | None = None,
	unit_ids_override: list[int] | None = None,
	unit_limit_override: int | None = None,
	limit_segments_override: int | None = None,
	limit_datasets_override: int | None = None,
	target_datasets_override: list[int] | None = None,
	limit_wells_per_dataset_override: int | None = None,
	force_restart_override: bool | None = None,
	replot_override: bool | None = None,
	task_allocation_override: dict[str, Any] | None = None,
) -> MultiTargetStageResult:
	return _run_reconstruct_substage_from_runtime(
		config_path=config_path,
		stage_name="reconstruct.analyzers",
		runner_fn=run_reconstruct_templates_analyzers,
		unit_id_override=unit_id_override,
		unit_ids_override=unit_ids_override,
		unit_limit_override=unit_limit_override,
		limit_segments_override=limit_segments_override,
		limit_datasets_override=limit_datasets_override,
		target_datasets_override=target_datasets_override,
		limit_wells_per_dataset_override=limit_wells_per_dataset_override,
		force_restart_override=force_restart_override,
		replot_override=replot_override,
		task_allocation_override=task_allocation_override,
	)


def run_reconstruct_templates_extract_partial_templates_from_runtime(
	*,
	config_path: str,
	unit_id_override: int | None = None,
	unit_ids_override: list[int] | None = None,
	unit_limit_override: int | None = None,
	limit_segments_override: int | None = None,
	limit_datasets_override: int | None = None,
	target_datasets_override: list[int] | None = None,
	limit_wells_per_dataset_override: int | None = None,
	force_restart_override: bool | None = None,
	replot_override: bool | None = None,
	task_allocation_override: dict[str, Any] | None = None,
) -> MultiTargetStageResult:
	return _run_reconstruct_substage_from_runtime(
		config_path=config_path,
		stage_name="reconstruct.extract_partial_templates",
		runner_fn=run_reconstruct_templates_extract_partial_templates,
		unit_id_override=unit_id_override,
		unit_ids_override=unit_ids_override,
		unit_limit_override=unit_limit_override,
		limit_segments_override=limit_segments_override,
		limit_datasets_override=limit_datasets_override,
		target_datasets_override=target_datasets_override,
		limit_wells_per_dataset_override=limit_wells_per_dataset_override,
		force_restart_override=force_restart_override,
		replot_override=replot_override,
		task_allocation_override=task_allocation_override,
	)


def run_reconstruct_templates_build_templates_from_runtime(
	*,
	config_path: str,
	unit_id_override: int | None = None,
	unit_ids_override: list[int] | None = None,
	unit_limit_override: int | None = None,
	limit_segments_override: int | None = None,
	limit_datasets_override: int | None = None,
	target_datasets_override: list[int] | None = None,
	limit_wells_per_dataset_override: int | None = None,
	force_restart_override: bool | None = None,
	replot_override: bool | None = None,
	task_allocation_override: dict[str, Any] | None = None,
) -> MultiTargetStageResult:
	return _run_reconstruct_substage_from_runtime(
		config_path=config_path,
		stage_name="reconstruct.build_templates",
		runner_fn=run_reconstruct_templates_build_templates,
		unit_id_override=unit_id_override,
		unit_ids_override=unit_ids_override,
		unit_limit_override=unit_limit_override,
		limit_segments_override=limit_segments_override,
		limit_datasets_override=limit_datasets_override,
		target_datasets_override=target_datasets_override,
		limit_wells_per_dataset_override=limit_wells_per_dataset_override,
		force_restart_override=force_restart_override,
		replot_override=replot_override,
		task_allocation_override=task_allocation_override,
	)


def run_reconstruct_templates_compute_template_similarity_from_runtime(
	*,
	config_path: str,
	unit_id_override: int | None = None,
	unit_ids_override: list[int] | None = None,
	unit_limit_override: int | None = None,
	limit_segments_override: int | None = None,
	limit_datasets_override: int | None = None,
	target_datasets_override: list[int] | None = None,
	limit_wells_per_dataset_override: int | None = None,
	force_restart_override: bool | None = None,
	replot_override: bool | None = None,
	task_allocation_override: dict[str, Any] | None = None,
) -> MultiTargetStageResult:
	return _run_reconstruct_substage_from_runtime(
		config_path=config_path,
		stage_name="reconstruct.compute_template_similarity",
		runner_fn=run_reconstruct_templates_compute_template_similarity,
		unit_id_override=unit_id_override,
		unit_ids_override=unit_ids_override,
		unit_limit_override=unit_limit_override,
		limit_segments_override=limit_segments_override,
		limit_datasets_override=limit_datasets_override,
		target_datasets_override=target_datasets_override,
		limit_wells_per_dataset_override=limit_wells_per_dataset_override,
		force_restart_override=force_restart_override,
		replot_override=replot_override,
		task_allocation_override=task_allocation_override,
	)


def run_reconstruct_templates_plot_templates_v2_from_runtime(
	*,
	config_path: str,
	unit_id_override: int | None = None,
	unit_ids_override: list[int] | None = None,
	unit_limit_override: int | None = None,
	limit_segments_override: int | None = None,
	limit_datasets_override: int | None = None,
	target_datasets_override: list[int] | None = None,
	limit_wells_per_dataset_override: int | None = None,
	force_restart_override: bool | None = None,
	replot_override: bool | None = None,
	task_allocation_override: dict[str, Any] | None = None,
) -> MultiTargetStageResult:
	return _run_reconstruct_substage_from_runtime(
		config_path=config_path,
		stage_name="reconstruct.plot_templates_v2",
		runner_fn=run_reconstruct_templates_plot_templates_v2,
		unit_id_override=unit_id_override,
		unit_ids_override=unit_ids_override,
		unit_limit_override=unit_limit_override,
		limit_segments_override=limit_segments_override,
		limit_datasets_override=limit_datasets_override,
		target_datasets_override=target_datasets_override,
		limit_wells_per_dataset_override=limit_wells_per_dataset_override,
		force_restart_override=force_restart_override,
		replot_override=replot_override,
		task_allocation_override=task_allocation_override,
	)


def run_reconstruct_templates_report_templates_from_runtime(
	*,
	config_path: str,
	unit_id_override: int | None = None,
	unit_ids_override: list[int] | None = None,
	unit_limit_override: int | None = None,
	limit_segments_override: int | None = None,
	limit_datasets_override: int | None = None,
	target_datasets_override: list[int] | None = None,
	limit_wells_per_dataset_override: int | None = None,
	force_restart_override: bool | None = None,
	replot_override: bool | None = None,
	task_allocation_override: dict[str, Any] | None = None,
) -> MultiTargetStageResult:
	return _run_reconstruct_substage_from_runtime(
		config_path=config_path,
		stage_name="reconstruct.report_templates",
		runner_fn=run_reconstruct_templates_report_templates,
		unit_id_override=unit_id_override,
		unit_ids_override=unit_ids_override,
		unit_limit_override=unit_limit_override,
		limit_segments_override=limit_segments_override,
		limit_datasets_override=limit_datasets_override,
		target_datasets_override=target_datasets_override,
		limit_wells_per_dataset_override=limit_wells_per_dataset_override,
		force_restart_override=force_restart_override,
		replot_override=replot_override,
		task_allocation_override=task_allocation_override,
	)


def run_reconstruct_axon_velocity_gtrs_from_runtime(
	*,
	config_path: str,
	unit_id_override: int | None = None,
	unit_ids_override: list[int] | None = None,
	unit_limit_override: int | None = None,
	limit_segments_override: int | None = None,
	limit_datasets_override: int | None = None,
	target_datasets_override: list[int] | None = None,
	limit_wells_per_dataset_override: int | None = None,
	force_restart_override: bool | None = None,
	replot_override: bool | None = None,
	task_allocation_override: dict[str, Any] | None = None,
) -> MultiTargetStageResult:
	return _run_reconstruct_substage_from_runtime(
		config_path=config_path,
		stage_name="reconstruct.axon_velocity_gtrs",
		runner_fn=run_reconstruct_axon_velocity_gtrs,
		unit_id_override=unit_id_override,
		unit_ids_override=unit_ids_override,
		unit_limit_override=unit_limit_override,
		limit_segments_override=limit_segments_override,
		limit_datasets_override=limit_datasets_override,
		target_datasets_override=target_datasets_override,
		limit_wells_per_dataset_override=limit_wells_per_dataset_override,
		force_restart_override=force_restart_override,
		replot_override=replot_override,
		task_allocation_override=task_allocation_override,
	)


def run_reconstruct_plot_recons_from_runtime(
	*,
	config_path: str,
	unit_id_override: int | None = None,
	unit_ids_override: list[int] | None = None,
	unit_limit_override: int | None = None,
	limit_segments_override: int | None = None,
	limit_datasets_override: int | None = None,
	target_datasets_override: list[int] | None = None,
	limit_wells_per_dataset_override: int | None = None,
	force_restart_override: bool | None = None,
	replot_override: bool | None = None,
	task_allocation_override: dict[str, Any] | None = None,
) -> MultiTargetStageResult:
	return _run_reconstruct_substage_from_runtime(
		config_path=config_path,
		stage_name="reconstruct.plot_recons",
		runner_fn=run_reconstruct_plot_recons,
		unit_id_override=unit_id_override,
		unit_ids_override=unit_ids_override,
		unit_limit_override=unit_limit_override,
		limit_segments_override=limit_segments_override,
		limit_datasets_override=limit_datasets_override,
		target_datasets_override=target_datasets_override,
		limit_wells_per_dataset_override=limit_wells_per_dataset_override,
		force_restart_override=force_restart_override,
		replot_override=replot_override,
		task_allocation_override=task_allocation_override,
	)


def run_reconstruct_plot_branch_propagations_from_runtime(
	*,
	config_path: str,
	unit_id_override: int | None = None,
	unit_ids_override: list[int] | None = None,
	unit_limit_override: int | None = None,
	limit_segments_override: int | None = None,
	limit_datasets_override: int | None = None,
	target_datasets_override: list[int] | None = None,
	limit_wells_per_dataset_override: int | None = None,
	force_restart_override: bool | None = None,
	replot_override: bool | None = None,
	task_allocation_override: dict[str, Any] | None = None,
) -> MultiTargetStageResult:
	return _run_reconstruct_substage_from_runtime(
		config_path=config_path,
		stage_name="reconstruct.plot_branch_propagations",
		runner_fn=run_reconstruct_plot_branch_propagations,
		unit_id_override=unit_id_override,
		unit_ids_override=unit_ids_override,
		unit_limit_override=unit_limit_override,
		limit_segments_override=limit_segments_override,
		limit_datasets_override=limit_datasets_override,
		target_datasets_override=target_datasets_override,
		limit_wells_per_dataset_override=limit_wells_per_dataset_override,
		force_restart_override=force_restart_override,
		replot_override=replot_override,
		task_allocation_override=task_allocation_override,
	)


def run_reconstruct_plot_branch_velocities_from_runtime(
	*,
	config_path: str,
	unit_id_override: int | None = None,
	unit_ids_override: list[int] | None = None,
	unit_limit_override: int | None = None,
	limit_segments_override: int | None = None,
	limit_datasets_override: int | None = None,
	target_datasets_override: list[int] | None = None,
	limit_wells_per_dataset_override: int | None = None,
	force_restart_override: bool | None = None,
	replot_override: bool | None = None,
	task_allocation_override: dict[str, Any] | None = None,
) -> MultiTargetStageResult:
	return _run_reconstruct_substage_from_runtime(
		config_path=config_path,
		stage_name="reconstruct.plot_branch_velocities",
		runner_fn=run_reconstruct_plot_branch_velocities,
		unit_id_override=unit_id_override,
		unit_ids_override=unit_ids_override,
		unit_limit_override=unit_limit_override,
		limit_segments_override=limit_segments_override,
		limit_datasets_override=limit_datasets_override,
		target_datasets_override=target_datasets_override,
		limit_wells_per_dataset_override=limit_wells_per_dataset_override,
		force_restart_override=force_restart_override,
		replot_override=replot_override,
		task_allocation_override=task_allocation_override,
	)


def run_reconstruct_plot_unit_summary_from_runtime(
	*,
	config_path: str,
	unit_id_override: int | None = None,
	unit_ids_override: list[int] | None = None,
	unit_limit_override: int | None = None,
	limit_segments_override: int | None = None,
	limit_datasets_override: int | None = None,
	target_datasets_override: list[int] | None = None,
	limit_wells_per_dataset_override: int | None = None,
	force_restart_override: bool | None = None,
	replot_override: bool | None = None,
	task_allocation_override: dict[str, Any] | None = None,
) -> MultiTargetStageResult:
	return _run_reconstruct_substage_from_runtime(
		config_path=config_path,
		stage_name="reconstruct.plot_unit_summary",
		runner_fn=run_reconstruct_plot_unit_summary,
		unit_id_override=unit_id_override,
		unit_ids_override=unit_ids_override,
		unit_limit_override=unit_limit_override,
		limit_segments_override=limit_segments_override,
		limit_datasets_override=limit_datasets_override,
		target_datasets_override=target_datasets_override,
		limit_wells_per_dataset_override=limit_wells_per_dataset_override,
		force_restart_override=force_restart_override,
		replot_override=replot_override,
		task_allocation_override=task_allocation_override,
	)


def run_reconstruct_report_recons_from_runtime(
	*,
	config_path: str,
	unit_id_override: int | None = None,
	unit_ids_override: list[int] | None = None,
	unit_limit_override: int | None = None,
	limit_segments_override: int | None = None,
	limit_datasets_override: int | None = None,
	target_datasets_override: list[int] | None = None,
	limit_wells_per_dataset_override: int | None = None,
	force_restart_override: bool | None = None,
	replot_override: bool | None = None,
	task_allocation_override: dict[str, Any] | None = None,
) -> MultiTargetStageResult:
	return _run_reconstruct_substage_from_runtime(
		config_path=config_path,
		stage_name="reconstruct.report_recons",
		runner_fn=run_reconstruct_report_recons,
		unit_id_override=unit_id_override,
		unit_ids_override=unit_ids_override,
		unit_limit_override=unit_limit_override,
		limit_segments_override=limit_segments_override,
		limit_datasets_override=limit_datasets_override,
		target_datasets_override=target_datasets_override,
		limit_wells_per_dataset_override=limit_wells_per_dataset_override,
		force_restart_override=force_restart_override,
		replot_override=replot_override,
		task_allocation_override=task_allocation_override,
	)


def run_reconstruct_report_recon_grid_from_runtime(
	*,
	config_path: str,
	unit_id_override: int | None = None,
	unit_ids_override: list[int] | None = None,
	unit_limit_override: int | None = None,
	limit_segments_override: int | None = None,
	limit_datasets_override: int | None = None,
	target_datasets_override: list[int] | None = None,
	limit_wells_per_dataset_override: int | None = None,
	force_restart_override: bool | None = None,
	replot_override: bool | None = None,
	task_allocation_override: dict[str, Any] | None = None,
) -> MultiTargetStageResult:
	return _run_reconstruct_substage_from_runtime(
		config_path=config_path,
		stage_name="reconstruct.report_recon_grid",
		runner_fn=run_reconstruct_report_recon_grid,
		unit_id_override=unit_id_override,
		unit_ids_override=unit_ids_override,
		unit_limit_override=unit_limit_override,
		limit_segments_override=limit_segments_override,
		limit_datasets_override=limit_datasets_override,
		target_datasets_override=target_datasets_override,
		limit_wells_per_dataset_override=limit_wells_per_dataset_override,
		force_restart_override=force_restart_override,
		replot_override=replot_override,
		task_allocation_override=task_allocation_override,
	)


def run_reconstruct_report_full_chip_layout_from_runtime(
	*,
	config_path: str,
	unit_id_override: int | None = None,
	unit_ids_override: list[int] | None = None,
	unit_limit_override: int | None = None,
	limit_segments_override: int | None = None,
	limit_datasets_override: int | None = None,
	target_datasets_override: list[int] | None = None,
	limit_wells_per_dataset_override: int | None = None,
	force_restart_override: bool | None = None,
	replot_override: bool | None = None,
	task_allocation_override: dict[str, Any] | None = None,
) -> MultiTargetStageResult:
	return _run_reconstruct_substage_from_runtime(
		config_path=config_path,
		stage_name="reconstruct.report_full_chip_layout",
		runner_fn=run_reconstruct_report_full_chip_layout,
		unit_id_override=unit_id_override,
		unit_ids_override=unit_ids_override,
		unit_limit_override=unit_limit_override,
		limit_segments_override=limit_segments_override,
		limit_datasets_override=limit_datasets_override,
		target_datasets_override=target_datasets_override,
		limit_wells_per_dataset_override=limit_wells_per_dataset_override,
		force_restart_override=force_restart_override,
		replot_override=replot_override,
		task_allocation_override=task_allocation_override,
	)


def run_reconstruct_report_summaries_from_runtime(
	*,
	config_path: str,
	unit_id_override: int | None = None,
	unit_ids_override: list[int] | None = None,
	unit_limit_override: int | None = None,
	limit_segments_override: int | None = None,
	limit_datasets_override: int | None = None,
	target_datasets_override: list[int] | None = None,
	limit_wells_per_dataset_override: int | None = None,
	force_restart_override: bool | None = None,
	replot_override: bool | None = None,
	task_allocation_override: dict[str, Any] | None = None,
) -> MultiTargetStageResult:
	return _run_reconstruct_substage_from_runtime(
		config_path=config_path,
		stage_name="reconstruct.report_summaries",
		runner_fn=run_reconstruct_report_summaries,
		unit_id_override=unit_id_override,
		unit_ids_override=unit_ids_override,
		unit_limit_override=unit_limit_override,
		limit_segments_override=limit_segments_override,
		limit_datasets_override=limit_datasets_override,
		target_datasets_override=target_datasets_override,
		limit_wells_per_dataset_override=limit_wells_per_dataset_override,
		force_restart_override=force_restart_override,
		replot_override=replot_override,
		task_allocation_override=task_allocation_override,
	)


def run_reconstruct_clear_templates_cache_from_runtime(
	*,
	config_path: str,
	unit_id_override: int | None = None,
	unit_ids_override: list[int] | None = None,
	unit_limit_override: int | None = None,
	limit_segments_override: int | None = None,
	limit_datasets_override: int | None = None,
	target_datasets_override: list[int] | None = None,
	limit_wells_per_dataset_override: int | None = None,
	force_restart_override: bool | None = None,
	replot_override: bool | None = None,
) -> MultiTargetStageResult:
	return _run_reconstruct_substage_from_runtime(
		config_path=config_path,
		stage_name="reconstruct.clear_templates_cache",
		runner_fn=run_reconstruct_clear_templates_cache,
		unit_id_override=unit_id_override,
		unit_ids_override=unit_ids_override,
		unit_limit_override=unit_limit_override,
		limit_segments_override=limit_segments_override,
		limit_datasets_override=limit_datasets_override,
		target_datasets_override=target_datasets_override,
		limit_wells_per_dataset_override=limit_wells_per_dataset_override,
		force_restart_override=force_restart_override,
		replot_override=replot_override,
	)
