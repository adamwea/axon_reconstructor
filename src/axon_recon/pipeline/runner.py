from __future__ import annotations

from contextlib import nullcontext
import copy
from dataclasses import dataclass, replace
import logging
from pathlib import Path
import threading
import time
from typing import Any, Callable

from axon_recon.pipeline.publish import publish_path_to_final, remap_path_string_to_final

from .config import (
	PipelineRuntimeBundle,
	constrain_stage_parallelism_to_read_groups,
	load_pipeline_runtime_bundle,
	resolve_stage_parallelism,
	select_execution_targets,
)
from .execution.distributor import distribute_targets
from .execution.logging_context import install_pipeline_log_record_factory, pipeline_log_context_for_target
from .execution.phase_chain import PhaseDescriptor, run_phase_chain
from .execution.progress import PipelineProgress, ProgressSpec, pipeline_progress_context
from .execution.results import MultiTargetStageResult, TargetStageResult
from .shared.maxwell_plugin import install_maxwell_hdf5_plugin_message_filter
from .stages.preprocess.api import (
	run_preprocess_concat_segments,
	run_preprocess_copy_src_to_scratch,
	run_preprocess,
	run_preprocess_plot_concat_channel_layout,
	run_preprocess_prepare_raw_binaries,
	run_preprocess_plot_concat_traces,
	run_preprocess_plot_raster_threshold,
	run_preprocess_plot_segment_channel_layouts,
	run_preprocess_plot_segment_traces,
	run_preprocess_preprocess_segments,
	run_preprocess_save_rec_metadata,
	run_preprocess_wipe_src_scratch,
)
from .stages.preprocess.config import build_preprocess_inputs_for_target, parse_preprocess_stage_config
from .stages.preprocess.models.results import PreprocessResult
from .stages.reconstruct.api import (
	run_reconstruct,
	run_reconstruct_clear_templates_cache,
	run_reconstruct_generate_gtrs,
	run_reconstruct_plot_branch_propagations,
	run_reconstruct_plot_branch_velocities,
	run_reconstruct_plot_unit_summary,
	run_reconstruct_plot_recons,
	run_reconstruct_report_full_chip_layout,
	run_reconstruct_report_recons,
	run_reconstruct_report_summaries,
	run_reconstruct_templates_analyzers,
	run_reconstruct_templates_build_templates,
	run_reconstruct_templates_compute_template_similarity,
	run_reconstruct_templates_extract_template_segments,
	run_reconstruct_templates_plot_templates,
	run_reconstruct_templates_report_templates,
	run_reconstruct_templates_reports,
	run_reconstruct_templates_resolve_sources,
)
from .stages.reconstruct.config import (
	build_reconstruct_templates_runtime_config,
	build_reconstruction_inputs_for_target,
	parse_reconstruction_stage_config,
)
from .stages.reconstruct.models.results import ReconstructionResult, UnitReconstructionResult
from .stages.spikesort.api import (
	bootstrap_spikesort_concat_binary,
	cleanup_spikesort_concat_binary,
	run_spikesort,
	run_spikesort_bombcell,
	run_spikesort_merge,
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
from .stages.spikesort.orchestrators.merge_si_auto import run_spikesort_merge_si_auto
from .stages.spikesort.orchestrators.merge_slay import run_spikesort_merge_slay
from .stages.spikesort.orchestrators.merge_unitmatch import run_spikesort_merge_unitmatch
from .stages.reconstruct.templates.config import (
	build_templates_inputs_for_target,
	parse_probe_geometry_from_data_config,
	parse_templates_stage_config,
)


LOGGER = logging.getLogger("axon_recon.pipeline.runner")


@dataclass(frozen=True)
class _SpikesortRuntimePhase:
	name: str
	phase_label: str
	debug_enabled_attr: str
	debug_limit_datasets_attr: str
	debug_limit_wells_attr: str
	target_runner: Callable[..., Any]
	debug_limit_wells_per_dataset_attr: str | None = None

	def __iter__(self):
		yield self.name
		yield self.target_runner


@dataclass(frozen=True)
class PublishPolicy:
	publish_outputs: bool = True
	wipe_scratch_roots: bool = False

	def publish_mode(self) -> str:
		return "move" if bool(self.wipe_scratch_roots) else "copy"


def _preprocess_copy_phase_enabled(stage_config: Any) -> bool:
	try:
		return bool(stage_config.phases.copy_src_to_scratch.enabled) and _preprocess_stage_phase_in_sequence(
			stage_config,
			"copy_src_to_scratch",
		)
	except Exception:
		return False


def _preprocess_stage_phase_in_sequence(stage_config: Any, phase_name: str) -> bool:
	sequence = getattr(stage_config, "phase_sequence", None)
	if sequence is None:
		return True
	return str(phase_name) in {str(item) for item in sequence}


def _preprocess_stage_uses_nested_workers(stage_config: Any) -> bool:
	try:
		phases = stage_config.phases
	except Exception:
		return True
	preprocess_segments_enabled = bool(getattr(getattr(phases, "preprocess_segments", None), "enabled", False))
	concat_segments_enabled = bool(getattr(getattr(phases, "concat_segments", None), "enabled", False))
	prepare_raw_binaries_enabled = bool(getattr(getattr(phases, "prepare_raw_binaries", None), "enabled", False))
	return bool(
		(prepare_raw_binaries_enabled and _preprocess_stage_phase_in_sequence(stage_config, "prepare_raw_binaries"))
		or (preprocess_segments_enabled and _preprocess_stage_phase_in_sequence(stage_config, "preprocess_segments"))
		or (concat_segments_enabled and _preprocess_stage_phase_in_sequence(stage_config, "concat_segments"))
	)


def _preprocess_substage_uses_nested_workers(stage_name: str) -> bool:
	return str(stage_name).strip() in {
		"preprocess.prepare_raw_binaries",
		"preprocess.preprocess_segments",
		"preprocess.concat_segments",
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
		"Preprocess worker allocation stage=%s stage_workers=%d well_workers=%d n_jobs=%d n_jobs_source=%s uses_nested_workers=%s emit_subphase_dividers_to_stdout=%s",
		str(stage_name),
		int(parallelism.max_stage_workers),
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
):
	try:
		parallelism = resolve_stage_parallelism(
			bundle=bundle,
			stage_name=stage_name,
			target_count=int(target_count),
		)
	except TypeError as exc:
		if "target_count" not in str(exc):
			raise
		parallelism = resolve_stage_parallelism(bundle=bundle, stage_name=stage_name)
	if targets is None:
		return parallelism
	return constrain_stage_parallelism_to_read_groups(
		parallelism=parallelism,
		targets=list(targets),
	)


def _distribute_runtime_targets(
	*,
	targets: list[Any],
	parallelism: Any,
	worker_fn: Callable[[Any], Any],
	stage_name: str | None = None,
	progress: PipelineProgress | None = None,
	advance_progress_on_target_complete: bool = False,
) -> list[TargetStageResult]:
	install_pipeline_log_record_factory()
	install_maxwell_hdf5_plugin_message_filter()

	def worker_with_log_context(target: Any) -> Any:
		with pipeline_log_context_for_target(target, stage=stage_name), pipeline_progress_context(progress):
			started = time.perf_counter()
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
		)


def _reconstruct_unit_progress(stage_name: str) -> PipelineProgress | None:
	if str(stage_name) not in {"reconstruct", "reconstruct.generate_gtrs"}:
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
			runtime_stage_workers=max(1, int(parallelism.max_stage_workers)),
			runtime_well_workers=max(1, int(parallelism.well_workers)),
			runtime_n_jobs_source=str(n_jobs_source),
		)
	except Exception:
		return inputs


def _target_log_label(target: Any) -> str:
	return f"{getattr(target, 'dataset_index', 'unknown')}:{getattr(target, 'stream_id', 'unknown')}"


def _log_spikesort_phase_worker_allocation(
	*,
	phase_name: str,
	target: Any,
	parallelism: Any,
	n_jobs: int,
	n_jobs_source: str,
) -> None:
	LOGGER.info(
		"Spikesort phase worker allocation stage=spikesort phase=%s target=%s stage_workers=%d well_workers=%d n_jobs=%d n_jobs_source=%s",
		str(phase_name),
		_target_log_label(target),
		int(parallelism.max_stage_workers),
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
	if str(phase_label) != "sort" or sort_phase_gate is None:
		return runner()

	target_label = _target_log_label(target)
	LOGGER.info("spikesort.sort: waiting for single-well sort gate target=%s", target_label)
	sort_phase_gate.acquire()
	LOGGER.info("spikesort.sort: acquired single-well sort gate target=%s", target_label)
	try:
		return runner()
	finally:
		sort_phase_gate.release()
		LOGGER.info("spikesort.sort: released single-well sort gate target=%s", target_label)


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
		"preprocess.concat_segments": "concat_segments",
		"preprocess.plot_raster_threshold": "plot_raster_threshold",
	}
	phase_attr = phase_attr_by_stage_name.get(str(stage_name).strip())
	if phase_attr is None:
		return list(targets)

	phase_cfg = getattr(getattr(stage_config, "phases", None), str(phase_attr), None)
	if phase_cfg is None or not bool(getattr(phase_cfg, "debug_mode_enabled", False)):
		return list(targets)

	limit_datasets = getattr(phase_cfg, "debug_limit_datasets", None)
	limit_wells = getattr(phase_cfg, "debug_limit_wells", None)
	limit_wells_per_dataset = getattr(phase_cfg, "debug_limit_wells_per_dataset", None)
	return _apply_preprocess_debug_target_limits(
		stage_name=stage_name,
		targets=list(targets),
		limit_datasets=limit_datasets,
		limit_wells=limit_wells,
		limit_wells_per_dataset=limit_wells_per_dataset,
	)


def _apply_spikesort_sort_debug_limits(
	*,
	stage_name: str,
	stage_config: Any,
	targets: list[Any],
) -> list[Any]:
	return _apply_spikesort_phase_debug_limits(
		stage_name=stage_name,
		stage_config=stage_config,
		targets=targets,
		phase_label="sort",
		enabled_attr="sort_debug_mode_enabled",
		limit_datasets_attr="sort_debug_limit_datasets",
		limit_wells_attr="sort_debug_limit_wells",
		limit_wells_per_dataset_attr="sort_debug_limit_wells_per_dataset",
	)


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
	limited_targets = list(targets)
	if bool(getattr(stage_config, str(enabled_attr), False)):
		limit_datasets = getattr(stage_config, str(limit_datasets_attr), None)
		limit_wells = getattr(stage_config, str(limit_wells_attr), None)
		limit_wells_per_dataset = (
			getattr(stage_config, str(limit_wells_per_dataset_attr), None)
			if limit_wells_per_dataset_attr is not None
			else None
		)
		return _apply_spikesort_debug_target_limits(
			stage_name=stage_name,
			targets=limited_targets,
			limit_datasets=limit_datasets,
			limit_wells=limit_wells,
			limit_wells_per_dataset=limit_wells_per_dataset,
			phase_label=str(phase_label),
		)

	return limited_targets


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


def run_preprocess_from_runtime(
	*,
	config_path: str,
	force_restart_override: bool | None = None,
	force_replot_override: bool | None = None,
) -> MultiTargetStageResult:
	bundle: PipelineRuntimeBundle = load_pipeline_runtime_bundle(config_path=config_path)
	publish_policy = _resolve_publish_policy(runtime_config=bundle.runtime_config, data_config=bundle.data_config)
	_log_publish_policy(stage_name="preprocess", policy=publish_policy)
	stage_config = parse_preprocess_stage_config(
		runtime_config=bundle.runtime_config,
		force_restart_override=force_restart_override,
		force_replot_override=force_replot_override,
	)
	targets = select_execution_targets(
		bundle=bundle,
		materialize_scratch_inputs=_preprocess_copy_phase_enabled(stage_config),
	)
	targets = _apply_preprocess_stage_debug_limits(
		stage_name="preprocess",
		stage_config=stage_config,
		targets=list(targets),
	)
	parallelism = _resolve_runtime_stage_parallelism(
		bundle=bundle,
		stage_name="preprocess",
		target_count=len(targets),
		targets=targets,
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
		return run_preprocess(inputs)

	target_results = _distribute_runtime_targets(
		targets=targets,
		parallelism=parallelism,
		worker_fn=_worker,
		stage_name="preprocess",
		progress=PipelineProgress(ProgressSpec(label="preprocess wells", total=len(targets), unit="well")),
		advance_progress_on_target_complete=True,
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
	force_restart_override: bool | None = None,
	force_replot_override: bool | None = None,
) -> MultiTargetStageResult:
	bundle: PipelineRuntimeBundle = load_pipeline_runtime_bundle(config_path=config_path)
	stage_config = parse_preprocess_stage_config(
		runtime_config=bundle.runtime_config,
		force_restart_override=force_restart_override,
		force_replot_override=force_replot_override,
	)
	targets = select_execution_targets(
		bundle=bundle,
		materialize_scratch_inputs=(str(stage_name).strip() == "preprocess.copy_src_to_scratch"),
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
	parallelism = _resolve_runtime_stage_parallelism(
		bundle=bundle,
		stage_name="preprocess",
		target_count=len(targets),
		targets=targets,
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

	target_results = _distribute_runtime_targets(
		targets=targets,
		parallelism=parallelism,
		worker_fn=_worker,
		stage_name=stage_name,
		progress=PipelineProgress(ProgressSpec(label=f"{stage_name} wells", total=len(targets), unit="well")),
		advance_progress_on_target_complete=True,
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


def run_preprocess_copy_src_to_scratch_from_runtime(
	*,
	config_path: str,
	force_restart_override: bool | None = None,
	force_replot_override: bool | None = None,
) -> MultiTargetStageResult:
	return _run_preprocess_substage_from_runtime(
		config_path=config_path,
		stage_name="preprocess.copy_src_to_scratch",
		runner_fn=run_preprocess_copy_src_to_scratch,
		force_restart_override=force_restart_override,
		force_replot_override=force_replot_override,
	)


def run_preprocess_save_rec_metadata_from_runtime(
	*,
	config_path: str,
	force_restart_override: bool | None = None,
	force_replot_override: bool | None = None,
) -> MultiTargetStageResult:
	return _run_preprocess_substage_from_runtime(
		config_path=config_path,
		stage_name="preprocess.save_rec_metadata",
		runner_fn=run_preprocess_save_rec_metadata,
		force_restart_override=force_restart_override,
		force_replot_override=force_replot_override,
	)


def run_preprocess_prepare_raw_binaries_from_runtime(
	*,
	config_path: str,
	force_restart_override: bool | None = None,
	force_replot_override: bool | None = None,
) -> MultiTargetStageResult:
	return _run_preprocess_substage_from_runtime(
		config_path=config_path,
		stage_name="preprocess.prepare_raw_binaries",
		runner_fn=run_preprocess_prepare_raw_binaries,
		force_restart_override=force_restart_override,
		force_replot_override=force_replot_override,
	)


def run_preprocess_plot_segment_channel_layouts_from_runtime(
	*,
	config_path: str,
	force_restart_override: bool | None = None,
	force_replot_override: bool | None = None,
) -> MultiTargetStageResult:
	return _run_preprocess_substage_from_runtime(
		config_path=config_path,
		stage_name="preprocess.plot_segment_channel_layouts",
		runner_fn=run_preprocess_plot_segment_channel_layouts,
		force_restart_override=force_restart_override,
		force_replot_override=force_replot_override,
	)


def run_preprocess_wipe_src_scratch_from_runtime(
	*,
	config_path: str,
	force_restart_override: bool | None = None,
	force_replot_override: bool | None = None,
) -> MultiTargetStageResult:
	return _run_preprocess_substage_from_runtime(
		config_path=config_path,
		stage_name="preprocess.wipe_src_scratch",
		runner_fn=run_preprocess_wipe_src_scratch,
		force_restart_override=force_restart_override,
		force_replot_override=force_replot_override,
	)


def run_preprocess_preprocess_segments_from_runtime(
	*,
	config_path: str,
	force_restart_override: bool | None = None,
	force_replot_override: bool | None = None,
) -> MultiTargetStageResult:
	return _run_preprocess_substage_from_runtime(
		config_path=config_path,
		stage_name="preprocess.preprocess_segments",
		runner_fn=run_preprocess_preprocess_segments,
		force_restart_override=force_restart_override,
		force_replot_override=force_replot_override,
	)


def run_preprocess_plot_segment_traces_from_runtime(
	*,
	config_path: str,
	force_restart_override: bool | None = None,
	force_replot_override: bool | None = None,
) -> MultiTargetStageResult:
	return _run_preprocess_substage_from_runtime(
		config_path=config_path,
		stage_name="preprocess.plot_segment_traces",
		runner_fn=run_preprocess_plot_segment_traces,
		force_restart_override=force_restart_override,
		force_replot_override=force_replot_override,
	)


def run_preprocess_concat_segments_from_runtime(
	*,
	config_path: str,
	force_restart_override: bool | None = None,
	force_replot_override: bool | None = None,
) -> MultiTargetStageResult:
	return _run_preprocess_substage_from_runtime(
		config_path=config_path,
		stage_name="preprocess.concat_segments",
		runner_fn=run_preprocess_concat_segments,
		force_restart_override=force_restart_override,
		force_replot_override=force_replot_override,
	)


def run_preprocess_plot_concat_traces_from_runtime(
	*,
	config_path: str,
	force_restart_override: bool | None = None,
	force_replot_override: bool | None = None,
) -> MultiTargetStageResult:
	return _run_preprocess_substage_from_runtime(
		config_path=config_path,
		stage_name="preprocess.plot_concat_traces",
		runner_fn=run_preprocess_plot_concat_traces,
		force_restart_override=force_restart_override,
		force_replot_override=force_replot_override,
	)


def run_preprocess_plot_concat_channel_layout_from_runtime(
	*,
	config_path: str,
	force_restart_override: bool | None = None,
	force_replot_override: bool | None = None,
) -> MultiTargetStageResult:
	return _run_preprocess_substage_from_runtime(
		config_path=config_path,
		stage_name="preprocess.plot_concat_channel_layout",
		runner_fn=run_preprocess_plot_concat_channel_layout,
		force_restart_override=force_restart_override,
		force_replot_override=force_replot_override,
	)


def run_preprocess_plot_raster_threshold_from_runtime(
	*,
	config_path: str,
	force_restart_override: bool | None = None,
	force_replot_override: bool | None = None,
) -> MultiTargetStageResult:
	return _run_preprocess_substage_from_runtime(
		config_path=config_path,
		stage_name="preprocess.plot_raster_threshold",
		runner_fn=run_preprocess_plot_raster_threshold,
		force_restart_override=force_restart_override,
		force_replot_override=force_replot_override,
	)


def run_spikesort_from_runtime(
	*,
	config_path: str,
	force_restart_override: bool | None = None,
	force_replot_override: bool | None = None,
) -> MultiTargetStageResult:
	bundle: PipelineRuntimeBundle = load_pipeline_runtime_bundle(config_path=config_path)
	publish_policy = _resolve_publish_policy(runtime_config=bundle.runtime_config, data_config=bundle.data_config)
	_log_publish_policy(stage_name="spikesort", policy=publish_policy)
	stage_config = parse_spikesort_stage_config(
		runtime_config=bundle.runtime_config,
		force_restart_override=force_restart_override,
		force_replot_override=force_replot_override,
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
	targets = select_execution_targets(bundle=bundle)
	targets = _apply_spikesort_runtime_phase_plan_debug_limits(
		stage_config=stage_config,
		targets=list(targets),
		phase_plan=phase_plan,
	)
	parallelism = _resolve_runtime_stage_parallelism(
		bundle=bundle,
		stage_name="spikesort",
		target_count=len(targets),
		targets=targets,
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
	LOGGER.info(
		"spikesort: starting target-local phase chains targets=%d phases=%s stage_workers=%d well_workers=%d n_jobs=%d n_jobs_source=%s",
		len(targets),
		[phase.name for phase in phase_plan],
		int(parallelism.max_stage_workers),
		int(parallelism.well_workers),
		int(runtime_n_jobs),
		str(n_jobs_source),
	)
	sort_phase_gate = threading.Lock() if bool(getattr(stage_config, "force_single_well_sort", False)) else None
	if sort_phase_gate is not None and any(str(phase.phase_label) == "sort" for phase in phase_plan):
		LOGGER.info(
			"spikesort: force_single_well_sort enabled; sort phase will run one well at a time targets=%d well_workers=%d",
			len(targets),
			int(parallelism.well_workers),
		)

	def _worker(target):
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
					sort_phase_gate=sort_phase_gate,
				)

			return PhaseDescriptor(
				name=phase.name,
				runner=_run_phase,
			)

		chain_result = run_phase_chain(
			phases=[_descriptor_for_phase(phase) for phase in phase_plan],
			logger=LOGGER,
			target_label=f"{getattr(target, 'dataset_index', 'unknown')}:{getattr(target, 'stream_id', 'unknown')}",
		)
		if chain_result.result is None:
			raise RuntimeError("spikesort phase chain produced no result")
		return chain_result.result

	target_results = _distribute_runtime_targets(
		targets=targets,
		parallelism=parallelism,
		worker_fn=_worker,
		stage_name="spikesort",
		progress=PipelineProgress(ProgressSpec(label="spikesort wells", total=len(targets), unit="well")),
		advance_progress_on_target_complete=True,
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
	if bool(getattr(stage_config, "bootstrap_concat_binary_enabled", False)):
		available_phases["bootstrap_concat_binary"] = (
			_SpikesortRuntimePhase(
				name="spikesort.bootstrap_concat_binary",
				phase_label="bootstrap_concat_binary",
				debug_enabled_attr="bootstrap_concat_binary_debug_mode_enabled",
				debug_limit_datasets_attr="bootstrap_concat_binary_debug_limit_datasets",
				debug_limit_wells_attr="bootstrap_concat_binary_debug_limit_wells",
				target_runner=_run_spikesort_bootstrap_concat_binary_target,
				debug_limit_wells_per_dataset_attr="bootstrap_concat_binary_debug_limit_wells_per_dataset",
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
			)
		)
	if bool(getattr(stage_config, "merge_si_auto_enabled", False)):
		available_phases["merge_si_auto"] = (
			_SpikesortRuntimePhase(
				name="spikesort.merge_si_auto",
				phase_label="merge_si_auto",
				debug_enabled_attr="merge_si_auto_debug_mode_enabled",
				debug_limit_datasets_attr="merge_si_auto_debug_limit_datasets",
				debug_limit_wells_attr="merge_si_auto_debug_limit_wells",
				target_runner=_run_spikesort_merge_si_auto_target,
				debug_limit_wells_per_dataset_attr="merge_si_auto_debug_limit_wells_per_dataset",
			)
		)
	if bool(getattr(stage_config, "merge_unitmatch_enabled", False)):
		available_phases["merge_unitmatch"] = (
			_SpikesortRuntimePhase(
				name="spikesort.merge_unitmatch",
				phase_label="merge_unitmatch",
				debug_enabled_attr="merge_unitmatch_debug_mode_enabled",
				debug_limit_datasets_attr="merge_unitmatch_debug_limit_datasets",
				debug_limit_wells_attr="merge_unitmatch_debug_limit_wells",
				target_runner=_run_spikesort_merge_unitmatch_target,
				debug_limit_wells_per_dataset_attr="merge_unitmatch_debug_limit_wells_per_dataset",
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
	limited_targets = _apply_spikesort_stage_debug_limits(
		stage_name="spikesort",
		stage_config=stage_config,
		targets=list(targets),
	)
	for phase in phase_plan:
		limited_targets = _apply_spikesort_phase_debug_limits(
			stage_name=phase.name,
			stage_config=stage_config,
			targets=limited_targets,
			phase_label=phase.phase_label,
			enabled_attr=phase.debug_enabled_attr,
			limit_datasets_attr=phase.debug_limit_datasets_attr,
			limit_wells_attr=phase.debug_limit_wells_attr,
			limit_wells_per_dataset_attr=phase.debug_limit_wells_per_dataset_attr,
		)
	return limited_targets


def _spikesort_output_rel_root(stage_config: Any) -> str:
	return str(getattr(stage_config, "output_rel_root", "spikesort_outputs") or "spikesort_outputs")


def _run_spikesort_bootstrap_concat_binary_target(*, target: Any, stage_config: Any, unit_workers: int) -> SpikesortResult:
	return bootstrap_spikesort_concat_binary(
		h5_path=target.h5_path,
		stream_id=target.stream_id,
		mea_output_root=target.mea_output_root,
		output_rel_root=_spikesort_output_rel_root(stage_config),
		stage_config=stage_config,
		force_restart=bool(getattr(stage_config, "force_restart", False) or getattr(stage_config, "force_replot", False)),
	)


def _run_spikesort_cleanup_concat_binary_target(*, target: Any, stage_config: Any, unit_workers: int) -> SpikesortResult:
	return cleanup_spikesort_concat_binary(
		h5_path=target.h5_path,
		stream_id=target.stream_id,
		mea_output_root=target.mea_output_root,
		output_rel_root=_spikesort_output_rel_root(stage_config),
		stage_config=stage_config,
		force_restart=bool(getattr(stage_config, "force_restart", False) or getattr(stage_config, "force_replot", False)),
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


def _run_spikesort_bombcell_label_target(*, target: Any, stage_config: Any, unit_workers: int) -> SpikesortBombcellResult:
	return run_spikesort_bombcell(
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
		force_replot=bool(getattr(stage_config, "merge_slay_force_replot", False)),
	)


def _run_spikesort_merge_si_auto_target(*, target: Any, stage_config: Any, unit_workers: int) -> SpikesortMergeResult:
	return run_spikesort_merge_si_auto(
		h5_path=target.h5_path,
		stream_id=target.stream_id,
		mea_output_root=target.mea_output_root,
		output_rel_root=_spikesort_output_rel_root(stage_config),
		stage_config=stage_config,
		force_restart=bool(getattr(stage_config, "merge_si_auto_force_restart", False)),
		force_replot=bool(getattr(stage_config, "merge_si_auto_force_replot", False)),
	)


def _run_spikesort_merge_unitmatch_target(*, target: Any, stage_config: Any, unit_workers: int) -> SpikesortMergeResult:
	return run_spikesort_merge_unitmatch(
		h5_path=target.h5_path,
		stream_id=target.stream_id,
		mea_output_root=target.mea_output_root,
		output_rel_root=_spikesort_output_rel_root(stage_config),
		stage_config=stage_config,
		force_restart=bool(getattr(stage_config, "merge_unitmatch_force_restart", False)),
		force_replot=bool(getattr(stage_config, "merge_unitmatch_force_replot", False)),
	)


def run_spikesort_sort_from_runtime(
	*,
	config_path: str,
	force_restart_override: bool | None = None,
	force_replot_override: bool | None = None,
) -> MultiTargetStageResult:
	return _run_spikesort_sort_from_runtime(
		config_path=config_path,
		stage_name="spikesort.sort",
		force_restart_override=force_restart_override,
		force_replot_override=force_replot_override,
	)


def run_spikesort_summarize_sort_from_runtime(
	*,
	config_path: str,
	force_restart_override: bool | None = None,
	force_replot_override: bool | None = None,
) -> MultiTargetStageResult:
	bundle: PipelineRuntimeBundle = load_pipeline_runtime_bundle(config_path=config_path)
	stage_config = parse_spikesort_stage_config(
		runtime_config=bundle.runtime_config,
		force_restart_override=force_restart_override,
		force_replot_override=force_replot_override,
	)
	targets = select_execution_targets(bundle=bundle)
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
	parallelism = _resolve_runtime_stage_parallelism(
		bundle=bundle,
		stage_name="spikesort",
		target_count=len(targets),
		targets=targets,
	)
	n_jobs_source = _runtime_n_jobs_source(stage_config)

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
		return summarize_spikesort(inputs)

	target_results = _distribute_runtime_targets(
		targets=targets,
		parallelism=parallelism,
		worker_fn=_worker,
		stage_name="spikesort.summarize_sort",
		progress=PipelineProgress(ProgressSpec(label="spikesort.summarize_sort wells", total=len(targets), unit="well")),
		advance_progress_on_target_complete=True,
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
	force_restart_override: bool | None = None,
	force_replot_override: bool | None = None,
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
		force_replot_override=force_replot_override,
	)
	targets = select_execution_targets(bundle=bundle)
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
	parallelism = _resolve_runtime_stage_parallelism(
		bundle=bundle,
		stage_name="spikesort",
		target_count=len(targets),
		targets=targets,
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

	def _worker(target):
		_log_spikesort_phase_worker_allocation(
			phase_name=str(debug_phase_label),
			target=target,
			parallelism=parallelism,
			n_jobs=int(runtime_n_jobs),
			n_jobs_source=str(n_jobs_source),
		)
		return runner_fn(
			h5_path=target.h5_path,
			stream_id=target.stream_id,
			mea_output_root=target.mea_output_root,
			output_rel_root=runtime_stage_config.output_rel_root,
			stage_config=runtime_stage_config,
			force_restart=bool(runtime_stage_config.force_restart or runtime_stage_config.force_replot),
		)

	target_results = _distribute_runtime_targets(
		targets=targets,
		parallelism=parallelism,
		worker_fn=_worker,
		stage_name=stage_name,
		progress=PipelineProgress(ProgressSpec(label=f"{stage_name} wells", total=len(targets), unit="well")),
		advance_progress_on_target_complete=True,
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


def run_spikesort_bootstrap_concat_binary_from_runtime(
	*,
	config_path: str,
	force_restart_override: bool | None = None,
	force_replot_override: bool | None = None,
) -> MultiTargetStageResult:
	return _run_spikesort_concat_binary_phase_from_runtime(
		config_path=config_path,
		stage_name="spikesort.bootstrap_concat_binary",
		runner_fn=bootstrap_spikesort_concat_binary,
		force_restart_override=force_restart_override,
		force_replot_override=force_replot_override,
		debug_phase_label="bootstrap_concat_binary",
		debug_enabled_attr="bootstrap_concat_binary_debug_mode_enabled",
		debug_limit_datasets_attr="bootstrap_concat_binary_debug_limit_datasets",
		debug_limit_wells_attr="bootstrap_concat_binary_debug_limit_wells",
		debug_limit_wells_per_dataset_attr="bootstrap_concat_binary_debug_limit_wells_per_dataset",
		publish_after_run=False,
	)


def run_spikesort_cleanup_concat_binary_from_runtime(
	*,
	config_path: str,
	force_restart_override: bool | None = None,
	force_replot_override: bool | None = None,
) -> MultiTargetStageResult:
	return _run_spikesort_concat_binary_phase_from_runtime(
		config_path=config_path,
		stage_name="spikesort.cleanup_concat_binary",
		runner_fn=cleanup_spikesort_concat_binary,
		force_restart_override=force_restart_override,
		force_replot_override=force_replot_override,
		debug_phase_label="cleanup_concat_binary",
		debug_enabled_attr="cleanup_concat_binary_debug_mode_enabled",
		debug_limit_datasets_attr="cleanup_concat_binary_debug_limit_datasets",
		debug_limit_wells_attr="cleanup_concat_binary_debug_limit_wells",
		debug_limit_wells_per_dataset_attr="cleanup_concat_binary_debug_limit_wells_per_dataset",
		publish_after_run=True,
	)


def _run_spikesort_sort_from_runtime(
	*,
	config_path: str,
	stage_name: str,
	force_restart_override: bool | None = None,
	force_replot_override: bool | None = None,
) -> MultiTargetStageResult:
	bundle: PipelineRuntimeBundle = load_pipeline_runtime_bundle(config_path=config_path)
	publish_policy = _resolve_publish_policy(runtime_config=bundle.runtime_config, data_config=bundle.data_config)
	_log_publish_policy(stage_name=stage_name, policy=publish_policy)
	stage_config = parse_spikesort_stage_config(
		runtime_config=bundle.runtime_config,
		force_restart_override=force_restart_override,
		force_replot_override=force_replot_override,
	)
	targets = select_execution_targets(bundle=bundle)
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
	parallelism = _resolve_runtime_stage_parallelism(
		bundle=bundle,
		stage_name="spikesort",
		target_count=len(targets),
		targets=targets,
	)
	n_jobs_source = _runtime_n_jobs_source(stage_config)
	sort_phase_gate = threading.Lock() if bool(getattr(stage_config, "force_single_well_sort", False)) else None
	if sort_phase_gate is not None:
		LOGGER.info(
			"%s: force_single_well_sort enabled; sort phase will run one well at a time targets=%d well_workers=%d",
			stage_name,
			len(targets),
			int(parallelism.well_workers),
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
			return run_spikesort(inputs)

		return _run_spikesort_phase_with_optional_sort_gate(
			phase_label="sort",
			target=target,
			runner=_run_sort,
			sort_phase_gate=sort_phase_gate,
		)

	target_results = _distribute_runtime_targets(
		targets=targets,
		parallelism=parallelism,
		worker_fn=_worker,
		stage_name=stage_name,
		progress=PipelineProgress(ProgressSpec(label=f"{stage_name} wells", total=len(targets), unit="well")),
		advance_progress_on_target_complete=True,
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
	force_restart_override: bool | None = None,
	force_replot_override: bool | None = None,
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
		force_replot_override=force_replot_override,
	)
	if merge_sequence_override is not None:
		normalized_override = tuple(str(token).strip() for token in tuple(merge_sequence_override) if str(token).strip())
		if normalized_override:
			stage_config = replace(stage_config, merge_sequence=normalized_override)
	if stage_config_transformer is not None:
		stage_config = stage_config_transformer(stage_config)

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
	targets = select_execution_targets(bundle=bundle)
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
	parallelism = _resolve_runtime_stage_parallelism(
		bundle=bundle,
		stage_name="spikesort",
		target_count=len(targets),
		targets=targets,
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

	def _worker(target):
		_log_spikesort_phase_worker_allocation(
			phase_name=str(stage_name).split(".", 1)[1] if "." in str(stage_name) else str(stage_name),
			target=target,
			parallelism=parallelism,
			n_jobs=int(runtime_n_jobs),
			n_jobs_source=str(n_jobs_source),
		)
		return run_spikesort_merge(
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
			force_replot=bool(
				getattr(
					runtime_stage_config,
					"merge_force_replot",
					bool(runtime_stage_config.force_replot),
				)
			),
		)

	target_results = _distribute_runtime_targets(
		targets=targets,
		parallelism=parallelism,
		worker_fn=_worker,
		stage_name=stage_name,
		progress=PipelineProgress(ProgressSpec(label=f"{stage_name} wells", total=len(targets), unit="well")),
		advance_progress_on_target_complete=True,
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
	force_restart_override: bool | None = None,
	force_replot_override: bool | None = None,
	stage_name: str = "spikesort.bombcell_label",
) -> MultiTargetStageResult:
	bundle: PipelineRuntimeBundle = load_pipeline_runtime_bundle(config_path=config_path)
	publish_policy = _resolve_publish_policy(runtime_config=bundle.runtime_config, data_config=bundle.data_config)
	_log_publish_policy(stage_name=stage_name, policy=publish_policy)
	stage_config = parse_spikesort_stage_config(
		runtime_config=bundle.runtime_config,
		force_restart_override=force_restart_override,
		force_replot_override=force_replot_override,
	)
	targets = select_execution_targets(bundle=bundle)
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
	parallelism = _resolve_runtime_stage_parallelism(
		bundle=bundle,
		stage_name="spikesort",
		target_count=len(targets),
		targets=targets,
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

	def _worker(target):
		_log_spikesort_phase_worker_allocation(
			phase_name="bombcell_label",
			target=target,
			parallelism=parallelism,
			n_jobs=int(runtime_n_jobs),
			n_jobs_source=str(n_jobs_source),
		)
		return run_spikesort_bombcell(
			h5_path=target.h5_path,
			stream_id=target.stream_id,
			mea_output_root=target.mea_output_root,
			output_rel_root=runtime_stage_config.output_rel_root,
			stage_config=runtime_stage_config,
			force_restart=bool(runtime_stage_config.force_restart),
		)

	target_results = _distribute_runtime_targets(
		targets=targets,
		parallelism=parallelism,
		worker_fn=_worker,
		stage_name=stage_name,
		progress=PipelineProgress(ProgressSpec(label=f"{stage_name} wells", total=len(targets), unit="well")),
		advance_progress_on_target_complete=True,
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
	force_restart_override: bool | None = None,
	force_replot_override: bool | None = None,
	publish_outputs: bool = False,
) -> MultiTargetStageResult:
	bundle: PipelineRuntimeBundle = load_pipeline_runtime_bundle(config_path=config_path)
	publish_policy = _resolve_publish_policy(runtime_config=bundle.runtime_config, data_config=bundle.data_config)
	if publish_outputs:
		_log_publish_policy(stage_name=stage_name, policy=publish_policy)
	targets = select_execution_targets(bundle=bundle)
	probe_geometry = parse_probe_geometry_from_data_config(data_config=bundle.data_config)
	stage_config = parse_reconstruction_stage_config(
		runtime_config=bundle.runtime_config,
		unit_id_override=unit_id_override,
		unit_ids_override=unit_ids_override,
		unit_limit_override=unit_limit_override,
		limit_segments_override=limit_segments_override,
		force_restart_override=force_restart_override,
		force_replot_override=force_replot_override,
	)
	templates_stage_config = None
	if callable(getattr(bundle.runtime_config, "get", None)):
		templates_runtime_config = build_reconstruct_templates_runtime_config(bundle.runtime_config)
		templates_stage_config = parse_templates_stage_config(
			runtime_config=templates_runtime_config,
			probe_geometry=probe_geometry,
			unit_id_override=unit_id_override,
			unit_ids_override=unit_ids_override,
			unit_limit_override=stage_config.unit_limit,
			limit_segments_override=stage_config.limit_segments,
			force_restart_override=force_restart_override,
			force_replot_override=force_replot_override,
		)
	if bool(getattr(stage_config, "debug_mode_enabled", False)):
		targets = _apply_spikesort_debug_target_limits(
			stage_name=stage_name,
			targets=list(targets),
			limit_datasets=getattr(stage_config, "debug_limit_datasets", None),
			limit_wells=getattr(stage_config, "debug_limit_wells", None),
			limit_wells_per_dataset=getattr(stage_config, "debug_limit_wells_per_dataset", None),
		)
	parallelism = _resolve_runtime_stage_parallelism(
		bundle=bundle,
		stage_name="reconstruct",
		target_count=len(targets),
		targets=targets,
	)

	def _worker(target):
		inputs = build_reconstruction_inputs_for_target(
			target=target,
			stage_config=stage_config,
			unit_workers=int(parallelism.unit_workers),
			probe_geometry=probe_geometry,
		)
		if templates_stage_config is not None:
			templates_inputs = build_templates_inputs_for_target(
				target=target,
				stage_config=templates_stage_config,
				unit_workers=int(parallelism.unit_workers),
				probe_geometry=probe_geometry,
			)
			inputs = replace(inputs, templates_inputs=templates_inputs)
		result = runner_fn(inputs)
		return _raise_reconstruct_unit_failures(stage_name=stage_name, result=result)

	target_results = _distribute_runtime_targets(
		targets=targets,
		parallelism=parallelism,
		worker_fn=_worker,
		stage_name=stage_name,
		progress=_reconstruct_unit_progress(stage_name),
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
	force_restart_override: bool | None = None,
	force_replot_override: bool | None = None,
) -> MultiTargetStageResult:
	return _run_reconstruct_substage_from_runtime(
		config_path=config_path,
		stage_name="reconstruct",
		runner_fn=run_reconstruct,
		unit_id_override=unit_id_override,
		unit_ids_override=unit_ids_override,
		unit_limit_override=unit_limit_override,
		limit_segments_override=limit_segments_override,
		force_restart_override=force_restart_override,
		force_replot_override=force_replot_override,
		publish_outputs=True,
	)


def run_reconstruct_templates_resolve_sources_from_runtime(
	*,
	config_path: str,
	unit_id_override: int | None = None,
	unit_ids_override: list[int] | None = None,
	unit_limit_override: int | None = None,
	limit_segments_override: int | None = None,
	force_restart_override: bool | None = None,
	force_replot_override: bool | None = None,
) -> MultiTargetStageResult:
	return _run_reconstruct_substage_from_runtime(
		config_path=config_path,
		stage_name="reconstruct.resolve_sources",
		runner_fn=run_reconstruct_templates_resolve_sources,
		unit_id_override=unit_id_override,
		unit_ids_override=unit_ids_override,
		unit_limit_override=unit_limit_override,
		limit_segments_override=limit_segments_override,
		force_restart_override=force_restart_override,
		force_replot_override=force_replot_override,
	)

def run_reconstruct_templates_analyzers_from_runtime(
	*,
	config_path: str,
	unit_id_override: int | None = None,
	unit_ids_override: list[int] | None = None,
	unit_limit_override: int | None = None,
	limit_segments_override: int | None = None,
	force_restart_override: bool | None = None,
	force_replot_override: bool | None = None,
) -> MultiTargetStageResult:
	return _run_reconstruct_substage_from_runtime(
		config_path=config_path,
		stage_name="reconstruct.analyzers",
		runner_fn=run_reconstruct_templates_analyzers,
		unit_id_override=unit_id_override,
		unit_ids_override=unit_ids_override,
		unit_limit_override=unit_limit_override,
		limit_segments_override=limit_segments_override,
		force_restart_override=force_restart_override,
		force_replot_override=force_replot_override,
	)


def run_reconstruct_templates_extract_template_segments_from_runtime(
	*,
	config_path: str,
	unit_id_override: int | None = None,
	unit_ids_override: list[int] | None = None,
	unit_limit_override: int | None = None,
	limit_segments_override: int | None = None,
	force_restart_override: bool | None = None,
	force_replot_override: bool | None = None,
) -> MultiTargetStageResult:
	return _run_reconstruct_substage_from_runtime(
		config_path=config_path,
		stage_name="reconstruct.extract_template_segments",
		runner_fn=run_reconstruct_templates_extract_template_segments,
		unit_id_override=unit_id_override,
		unit_ids_override=unit_ids_override,
		unit_limit_override=unit_limit_override,
		limit_segments_override=limit_segments_override,
		force_restart_override=force_restart_override,
		force_replot_override=force_replot_override,
	)


def run_reconstruct_templates_build_templates_from_runtime(
	*,
	config_path: str,
	unit_id_override: int | None = None,
	unit_ids_override: list[int] | None = None,
	unit_limit_override: int | None = None,
	limit_segments_override: int | None = None,
	force_restart_override: bool | None = None,
	force_replot_override: bool | None = None,
) -> MultiTargetStageResult:
	return _run_reconstruct_substage_from_runtime(
		config_path=config_path,
		stage_name="reconstruct.build_templates",
		runner_fn=run_reconstruct_templates_build_templates,
		unit_id_override=unit_id_override,
		unit_ids_override=unit_ids_override,
		unit_limit_override=unit_limit_override,
		limit_segments_override=limit_segments_override,
		force_restart_override=force_restart_override,
		force_replot_override=force_replot_override,
	)


def run_reconstruct_templates_compute_template_similarity_from_runtime(
	*,
	config_path: str,
	unit_id_override: int | None = None,
	unit_ids_override: list[int] | None = None,
	unit_limit_override: int | None = None,
	limit_segments_override: int | None = None,
	force_restart_override: bool | None = None,
	force_replot_override: bool | None = None,
) -> MultiTargetStageResult:
	return _run_reconstruct_substage_from_runtime(
		config_path=config_path,
		stage_name="reconstruct.compute_template_similarity",
		runner_fn=run_reconstruct_templates_compute_template_similarity,
		unit_id_override=unit_id_override,
		unit_ids_override=unit_ids_override,
		unit_limit_override=unit_limit_override,
		limit_segments_override=limit_segments_override,
		force_restart_override=force_restart_override,
		force_replot_override=force_replot_override,
	)


def run_reconstruct_templates_plot_templates_from_runtime(
	*,
	config_path: str,
	unit_id_override: int | None = None,
	unit_ids_override: list[int] | None = None,
	unit_limit_override: int | None = None,
	limit_segments_override: int | None = None,
	force_restart_override: bool | None = None,
	force_replot_override: bool | None = None,
) -> MultiTargetStageResult:
	return _run_reconstruct_substage_from_runtime(
		config_path=config_path,
		stage_name="reconstruct.plot_templates",
		runner_fn=run_reconstruct_templates_plot_templates,
		unit_id_override=unit_id_override,
		unit_ids_override=unit_ids_override,
		unit_limit_override=unit_limit_override,
		limit_segments_override=limit_segments_override,
		force_restart_override=force_restart_override,
		force_replot_override=force_replot_override,
	)


def run_reconstruct_templates_report_templates_from_runtime(
	*,
	config_path: str,
	unit_id_override: int | None = None,
	unit_ids_override: list[int] | None = None,
	unit_limit_override: int | None = None,
	limit_segments_override: int | None = None,
	force_restart_override: bool | None = None,
	force_replot_override: bool | None = None,
) -> MultiTargetStageResult:
	return _run_reconstruct_substage_from_runtime(
		config_path=config_path,
		stage_name="reconstruct.report_templates",
		runner_fn=run_reconstruct_templates_report_templates,
		unit_id_override=unit_id_override,
		unit_ids_override=unit_ids_override,
		unit_limit_override=unit_limit_override,
		limit_segments_override=limit_segments_override,
		force_restart_override=force_restart_override,
		force_replot_override=force_replot_override,
	)


def run_reconstruct_templates_reports_from_runtime(
	*,
	config_path: str,
	unit_id_override: int | None = None,
	unit_ids_override: list[int] | None = None,
	unit_limit_override: int | None = None,
	limit_segments_override: int | None = None,
	force_restart_override: bool | None = None,
	force_replot_override: bool | None = None,
) -> MultiTargetStageResult:
	return _run_reconstruct_substage_from_runtime(
		config_path=config_path,
		stage_name="reconstruct.reports",
		runner_fn=run_reconstruct_templates_reports,
		unit_id_override=unit_id_override,
		unit_ids_override=unit_ids_override,
		unit_limit_override=unit_limit_override,
		limit_segments_override=limit_segments_override,
		force_restart_override=force_restart_override,
		force_replot_override=force_replot_override,
	)


def run_reconstruct_generate_gtrs_from_runtime(
	*,
	config_path: str,
	unit_id_override: int | None = None,
	unit_ids_override: list[int] | None = None,
	unit_limit_override: int | None = None,
	limit_segments_override: int | None = None,
	force_restart_override: bool | None = None,
	force_replot_override: bool | None = None,
) -> MultiTargetStageResult:
	return _run_reconstruct_substage_from_runtime(
		config_path=config_path,
		stage_name="reconstruct.generate_gtrs",
		runner_fn=run_reconstruct_generate_gtrs,
		unit_id_override=unit_id_override,
		unit_ids_override=unit_ids_override,
		unit_limit_override=unit_limit_override,
		limit_segments_override=limit_segments_override,
		force_restart_override=force_restart_override,
		force_replot_override=force_replot_override,
	)


def run_reconstruct_plot_recons_from_runtime(
	*,
	config_path: str,
	unit_id_override: int | None = None,
	unit_ids_override: list[int] | None = None,
	unit_limit_override: int | None = None,
	limit_segments_override: int | None = None,
	force_restart_override: bool | None = None,
	force_replot_override: bool | None = None,
) -> MultiTargetStageResult:
	return _run_reconstruct_substage_from_runtime(
		config_path=config_path,
		stage_name="reconstruct.plot_recons",
		runner_fn=run_reconstruct_plot_recons,
		unit_id_override=unit_id_override,
		unit_ids_override=unit_ids_override,
		unit_limit_override=unit_limit_override,
		limit_segments_override=limit_segments_override,
		force_restart_override=force_restart_override,
		force_replot_override=force_replot_override,
	)


def run_reconstruct_plot_branch_propagations_from_runtime(
	*,
	config_path: str,
	unit_id_override: int | None = None,
	unit_ids_override: list[int] | None = None,
	unit_limit_override: int | None = None,
	limit_segments_override: int | None = None,
	force_restart_override: bool | None = None,
	force_replot_override: bool | None = None,
) -> MultiTargetStageResult:
	return _run_reconstruct_substage_from_runtime(
		config_path=config_path,
		stage_name="reconstruct.plot_branch_propagations",
		runner_fn=run_reconstruct_plot_branch_propagations,
		unit_id_override=unit_id_override,
		unit_ids_override=unit_ids_override,
		unit_limit_override=unit_limit_override,
		limit_segments_override=limit_segments_override,
		force_restart_override=force_restart_override,
		force_replot_override=force_replot_override,
	)


def run_reconstruct_plot_branch_velocities_from_runtime(
	*,
	config_path: str,
	unit_id_override: int | None = None,
	unit_ids_override: list[int] | None = None,
	unit_limit_override: int | None = None,
	limit_segments_override: int | None = None,
	force_restart_override: bool | None = None,
	force_replot_override: bool | None = None,
) -> MultiTargetStageResult:
	return _run_reconstruct_substage_from_runtime(
		config_path=config_path,
		stage_name="reconstruct.plot_branch_velocities",
		runner_fn=run_reconstruct_plot_branch_velocities,
		unit_id_override=unit_id_override,
		unit_ids_override=unit_ids_override,
		unit_limit_override=unit_limit_override,
		limit_segments_override=limit_segments_override,
		force_restart_override=force_restart_override,
		force_replot_override=force_replot_override,
	)


def run_reconstruct_plot_unit_summary_from_runtime(
	*,
	config_path: str,
	unit_id_override: int | None = None,
	unit_ids_override: list[int] | None = None,
	unit_limit_override: int | None = None,
	limit_segments_override: int | None = None,
	force_restart_override: bool | None = None,
	force_replot_override: bool | None = None,
) -> MultiTargetStageResult:
	return _run_reconstruct_substage_from_runtime(
		config_path=config_path,
		stage_name="reconstruct.plot_unit_summary",
		runner_fn=run_reconstruct_plot_unit_summary,
		unit_id_override=unit_id_override,
		unit_ids_override=unit_ids_override,
		unit_limit_override=unit_limit_override,
		limit_segments_override=limit_segments_override,
		force_restart_override=force_restart_override,
		force_replot_override=force_replot_override,
	)


def run_reconstruct_report_recons_from_runtime(
	*,
	config_path: str,
	unit_id_override: int | None = None,
	unit_ids_override: list[int] | None = None,
	unit_limit_override: int | None = None,
	limit_segments_override: int | None = None,
	force_restart_override: bool | None = None,
	force_replot_override: bool | None = None,
) -> MultiTargetStageResult:
	return _run_reconstruct_substage_from_runtime(
		config_path=config_path,
		stage_name="reconstruct.report_recons",
		runner_fn=run_reconstruct_report_recons,
		unit_id_override=unit_id_override,
		unit_ids_override=unit_ids_override,
		unit_limit_override=unit_limit_override,
		limit_segments_override=limit_segments_override,
		force_restart_override=force_restart_override,
		force_replot_override=force_replot_override,
	)


def run_reconstruct_report_full_chip_layout_from_runtime(
	*,
	config_path: str,
	unit_id_override: int | None = None,
	unit_ids_override: list[int] | None = None,
	unit_limit_override: int | None = None,
	limit_segments_override: int | None = None,
	force_restart_override: bool | None = None,
	force_replot_override: bool | None = None,
) -> MultiTargetStageResult:
	return _run_reconstruct_substage_from_runtime(
		config_path=config_path,
		stage_name="reconstruct.report_full_chip_layout",
		runner_fn=run_reconstruct_report_full_chip_layout,
		unit_id_override=unit_id_override,
		unit_ids_override=unit_ids_override,
		unit_limit_override=unit_limit_override,
		limit_segments_override=limit_segments_override,
		force_restart_override=force_restart_override,
		force_replot_override=force_replot_override,
	)


def run_reconstruct_report_summaries_from_runtime(
	*,
	config_path: str,
	unit_id_override: int | None = None,
	unit_ids_override: list[int] | None = None,
	unit_limit_override: int | None = None,
	limit_segments_override: int | None = None,
	force_restart_override: bool | None = None,
	force_replot_override: bool | None = None,
) -> MultiTargetStageResult:
	return _run_reconstruct_substage_from_runtime(
		config_path=config_path,
		stage_name="reconstruct.report_summaries",
		runner_fn=run_reconstruct_report_summaries,
		unit_id_override=unit_id_override,
		unit_ids_override=unit_ids_override,
		unit_limit_override=unit_limit_override,
		limit_segments_override=limit_segments_override,
		force_restart_override=force_restart_override,
		force_replot_override=force_replot_override,
	)


def run_reconstruct_clear_templates_cache_from_runtime(
	*,
	config_path: str,
	unit_id_override: int | None = None,
	unit_ids_override: list[int] | None = None,
	unit_limit_override: int | None = None,
	limit_segments_override: int | None = None,
	force_restart_override: bool | None = None,
	force_replot_override: bool | None = None,
) -> MultiTargetStageResult:
	return _run_reconstruct_substage_from_runtime(
		config_path=config_path,
		stage_name="reconstruct.clear_templates_cache",
		runner_fn=run_reconstruct_clear_templates_cache,
		unit_id_override=unit_id_override,
		unit_ids_override=unit_ids_override,
		unit_limit_override=unit_limit_override,
		limit_segments_override=limit_segments_override,
		force_restart_override=force_restart_override,
		force_replot_override=force_replot_override,
	)
