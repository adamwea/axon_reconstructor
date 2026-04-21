from __future__ import annotations

from dataclasses import dataclass, replace
import logging
from pathlib import Path
from typing import Any, Callable

from axon_reconstructor.pipeline.publish import publish_path_to_final, remap_path_string_to_final

from .config import (
	PipelineRuntimeBundle,
	load_pipeline_runtime_bundle,
	resolve_stage_parallelism,
	select_execution_targets,
)
from .execution.distributor import distribute_targets
from .execution.results import MultiTargetStageResult, TargetStageResult
from .stages.analysis.api import run_analysis
from .stages.analysis.config import build_analysis_inputs_for_target, parse_analysis_stage_config
from .stages.analysis.cross_well import generate_cross_well_artifacts
from .stages.analysis.models.results import AnalysisResult
from .stages.preprocess.api import (
	run_preprocess_concat_segments,
	run_preprocess_concatenate_recordings,
	run_preprocess_concatenate_preprocessed_recordings,
	run_preprocess_copy_src_to_scratch,
	run_preprocess,
	run_preprocess_plot_concat_traces,
	run_preprocess_plot_segment_traces,
	run_preprocess_preprocess_segments,
	run_preprocess_save_rec_metadata,
	run_preprocess_save_common_electrodes,
	run_preprocess_wipe_src_scratch,
)
from .stages.preprocess.config import build_preprocess_inputs_for_target, parse_preprocess_stage_config
from .stages.preprocess.models.results import PreprocessResult
from .stages.reconstruct.api import (
	run_reconstruct,
	run_reconstruct_generate_gtrs,
	run_reconstruct_plot_branch_propagations,
	run_reconstruct_plot_branch_velocities,
	run_reconstruct_plot_unit_summary,
	run_reconstruct_plot_recons,
	run_reconstruct_report_full_chip_layout,
	run_reconstruct_report_recons,
	run_reconstruct_report_summaries,
)
from .stages.reconstruct.config import build_reconstruction_inputs_for_target, parse_reconstruction_stage_config
from .stages.reconstruct.models.results import ReconstructionResult, UnitReconstructionResult
from .stages.spikesort.api import run_spikesort, run_spikesort_merge
from .stages.spikesort.config import build_spikesort_inputs_for_target, parse_spikesort_stage_config
from .stages.spikesort.models.results import SpikesortMergeResult, SpikesortResult
from .stages.templates.api import (
	run_templates,
	run_templates_analyzers,
	run_templates_build_templates,
	run_templates_compute_template_similarity,
	run_templates_extract_template_segments,
	run_templates_per_unit_processing,
	run_templates_plot_templates,
	run_templates_report_templates,
	run_templates_reports,
	run_templates_resolve_sources,
)
from .stages.templates.config import (
	build_templates_inputs_for_target,
	parse_probe_geometry_from_data_config,
	parse_templates_stage_config,
)
from .stages.templates.models.results import TemplatesResult, UnitTemplatesResult


LOGGER = logging.getLogger("axon_recon.pipeline.runner")


@dataclass(frozen=True)
class PublishPolicy:
	publish_outputs: bool = True
	wipe_scratch_roots: bool = False

	def publish_mode(self) -> str:
		return "move" if bool(self.wipe_scratch_roots) else "copy"


def _preprocess_copy_phase_enabled(stage_config: Any) -> bool:
	try:
		return bool(stage_config.phases.copy_src_to_scratch.enabled)
	except Exception:
		return False


def _preprocess_stage_uses_nested_workers(stage_config: Any) -> bool:
	try:
		phases = stage_config.phases
	except Exception:
		return True
	preprocess_segments_enabled = bool(getattr(getattr(phases, "preprocess_segments", None), "enabled", False))
	concat_segments_enabled = bool(getattr(getattr(phases, "concat_segments", None), "enabled", False))
	concatenate_recordings_enabled = bool(getattr(getattr(phases, "concatenate_recordings", None), "enabled", False))
	legacy_concatenate_enabled = bool(
		getattr(getattr(phases, "concatenate_preprocessed_recordings", None), "enabled", False)
	)
	return bool(
		preprocess_segments_enabled
		or concat_segments_enabled
		or concatenate_recordings_enabled
		or legacy_concatenate_enabled
	)


def _preprocess_substage_uses_nested_workers(stage_name: str) -> bool:
	return str(stage_name).strip() in {
		"preprocess.preprocess_segments",
		"preprocess.build_preprocessed_recording",
		"preprocess.save_segment_recordings",
		"preprocess.concat_segments",
		"preprocess.concatenate_recordings",
		"preprocess.concatenate_preprocessed_recordings",
		"preprocess.save_concatenated_recording",
	}


def _preprocess_runtime_uses_nested_workers(*, stage_name: str, stage_config: Any) -> bool:
	return (
		_preprocess_stage_uses_nested_workers(stage_config)
		if str(stage_name).strip() == "preprocess"
		else _preprocess_substage_uses_nested_workers(stage_name)
	)


def _resolve_preprocess_runtime_unit_workers(*, stage_name: str, parallelism: Any, stage_config: Any) -> int:
	uses_nested_workers = _preprocess_runtime_uses_nested_workers(stage_name=stage_name, stage_config=stage_config)
	unit_workers = int(parallelism.unit_workers) if bool(uses_nested_workers) else 1
	emit_subphase_dividers_to_stdout = not (bool(uses_nested_workers) and int(parallelism.well_workers) > 1)
	LOGGER.info(
		"Preprocess worker allocation stage=%s well_workers=%d unit_workers=%d uses_nested_workers=%s emit_subphase_dividers_to_stdout=%s",
		str(stage_name),
		int(parallelism.well_workers),
		int(max(1, unit_workers)),
		bool(uses_nested_workers),
		bool(emit_subphase_dividers_to_stdout),
	)
	return max(1, int(unit_workers))


def _resolve_preprocess_subphase_dividers_to_stdout(*, stage_name: str, parallelism: Any, stage_config: Any) -> bool:
	uses_nested_workers = _preprocess_runtime_uses_nested_workers(stage_name=stage_name, stage_config=stage_config)
	return not (bool(uses_nested_workers) and int(parallelism.well_workers) > 1)


def _resolve_runtime_stage_parallelism(
	*,
	bundle: PipelineRuntimeBundle,
	stage_name: str,
	target_count: int,
):
	try:
		return resolve_stage_parallelism(
			bundle=bundle,
			stage_name=stage_name,
			target_count=int(target_count),
		)
	except TypeError as exc:
		if "target_count" not in str(exc):
			raise
		return resolve_stage_parallelism(bundle=bundle, stage_name=stage_name)


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


def _publish_templates_target_result(item: TargetStageResult, *, policy: PublishPolicy | None = None) -> TargetStageResult:
	publish_policy = policy or PublishPolicy()
	if item.status != "ok" or not isinstance(item.result, TemplatesResult):
		return item
	roots = _publish_roots_for_target(item.target)
	if roots is None:
		return item
	active_root, final_root = roots
	result = item.result
	published = _publish_stage_output(
		stage_name="templates",
		target=item.target,
		path=result.templates_out_dir,
		active_root=active_root,
		final_root=final_root,
		policy=publish_policy,
	)
	if not published:
		return item
	updated_units = [
		UnitTemplatesResult(
			unit_id=unit.unit_id,
			status=unit.status,
			outputs=_remap_output_map(unit.outputs, active_root=active_root, final_root=final_root),
			error=unit.error,
		)
		for unit in result.units
	]
	updated = TemplatesResult(
		well_out_dir=_remap_stage_path(result.well_out_dir, active_root=active_root, final_root=final_root),
		templates_out_dir=_remap_stage_path(result.templates_out_dir, active_root=active_root, final_root=final_root),
		summary_json=_remap_stage_path(result.summary_json, active_root=active_root, final_root=final_root),
		units=updated_units,
		report_outputs=_remap_output_map(result.report_outputs, active_root=active_root, final_root=final_root),
	)
	return TargetStageResult(target=item.target, status=item.status, result=updated, error=item.error)


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


def _publish_analysis_target_result(item: TargetStageResult, *, policy: PublishPolicy | None = None) -> TargetStageResult:
	publish_policy = policy or PublishPolicy()
	if item.status != "ok" or not isinstance(item.result, AnalysisResult):
		return item
	roots = _publish_roots_for_target(item.target)
	if roots is None:
		return item
	active_root, final_root = roots
	result = item.result
	published = _publish_stage_output(
		stage_name="analysis",
		target=item.target,
		path=result.analysis_out_dir,
		active_root=active_root,
		final_root=final_root,
		policy=publish_policy,
	)
	if not published:
		return item
	updated = AnalysisResult(
		well_out_dir=_remap_stage_path(result.well_out_dir, active_root=active_root, final_root=final_root),
		analysis_out_dir=_remap_stage_path(result.analysis_out_dir, active_root=active_root, final_root=final_root),
		summary_json=_remap_stage_path(result.summary_json, active_root=active_root, final_root=final_root),
		outputs=_remap_output_map(result.outputs, active_root=active_root, final_root=final_root),
		deferred_warnings=list(result.deferred_warnings),
	)
	return TargetStageResult(target=item.target, status=item.status, result=updated, error=item.error)


def _publish_cross_well_outputs(
	*,
	cross_outputs: dict[str, str],
	target_results: list[TargetStageResult],
	policy: PublishPolicy | None = None,
) -> dict[str, str]:
	publish_policy = policy or PublishPolicy()
	if not cross_outputs:
		return cross_outputs
	if not target_results:
		return cross_outputs
	roots = _publish_roots_for_target(target_results[0].target)
	if roots is None:
		return cross_outputs
	active_root, final_root = roots
	summary_json_raw = cross_outputs.get("cross_well.summary_json")
	if summary_json_raw:
		summary_path = Path(str(summary_json_raw)).expanduser().resolve()
		if summary_path.exists():
			published = _publish_stage_output(
				stage_name="analysis.cross_well",
				target=target_results[0].target,
				path=summary_path.parent,
				active_root=active_root,
				final_root=final_root,
				policy=publish_policy,
			)
			if not published:
				return cross_outputs
		else:
			LOGGER.info(
				"Publish skipped stage=analysis.cross_well reason=summary_path_missing summary_path=%s",
				summary_path,
			)
			return cross_outputs
	else:
		LOGGER.info("Publish skipped stage=analysis.cross_well reason=summary_path_unset")
		return cross_outputs
	return _remap_output_map(cross_outputs, active_root=active_root, final_root=final_root)


def _attach_cross_well_outputs_to_analysis_results(
	*,
	target_results: list[TargetStageResult],
	cross_outputs: dict[str, str],
	cross_warnings: list[str],
) -> list[TargetStageResult]:
	if not cross_outputs and not cross_warnings:
		return target_results

	merged_results: list[TargetStageResult] = []
	for item in target_results:
		if item.status != "ok" or not isinstance(item.result, AnalysisResult):
			merged_results.append(item)
			continue

		result = item.result
		outputs = dict(result.outputs)
		outputs.update(cross_outputs)

		warnings = list(result.deferred_warnings)
		for warning in cross_warnings:
			if warning not in warnings:
				warnings.append(warning)

		merged_results.append(
			TargetStageResult(
				target=item.target,
				status=item.status,
				result=AnalysisResult(
					well_out_dir=result.well_out_dir,
					analysis_out_dir=result.analysis_out_dir,
					summary_json=result.summary_json,
					outputs=outputs,
					deferred_warnings=warnings,
				),
				error=item.error,
			)
		)

	return merged_results


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
	if stage_config.debug_limit_wells is not None:
		limit_wells = max(1, int(stage_config.debug_limit_wells))
		if len(targets) > limit_wells:
			LOGGER.info(
				"Applying preprocess debug well limit: %d -> %d target(s)",
				len(targets),
				limit_wells,
			)
			targets = list(targets[:limit_wells])
	parallelism = _resolve_runtime_stage_parallelism(
		bundle=bundle,
		stage_name="preprocess",
		target_count=len(targets),
	)
	unit_workers = _resolve_preprocess_runtime_unit_workers(
		stage_name="preprocess",
		parallelism=parallelism,
		stage_config=stage_config,
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
		inputs = replace(
			inputs,
			logging_subphase_dividers_to_stdout=bool(emit_subphase_dividers_to_stdout),
		)
		return run_preprocess(inputs)

	target_results = distribute_targets(
		targets=targets,
		well_workers=int(parallelism.well_workers),
		worker_fn=_worker,
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
	if stage_config.debug_limit_wells is not None:
		limit_wells = max(1, int(stage_config.debug_limit_wells))
		if len(targets) > limit_wells:
			LOGGER.info(
				"Applying preprocess debug well limit: %d -> %d target(s)",
				len(targets),
				limit_wells,
			)
			targets = list(targets[:limit_wells])
	parallelism = _resolve_runtime_stage_parallelism(
		bundle=bundle,
		stage_name="preprocess",
		target_count=len(targets),
	)
	unit_workers = _resolve_preprocess_runtime_unit_workers(
		stage_name=stage_name,
		parallelism=parallelism,
		stage_config=stage_config,
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
		inputs = replace(
			inputs,
			logging_subphase_dividers_to_stdout=bool(emit_subphase_dividers_to_stdout),
		)
		return runner_fn(inputs)

	target_results = distribute_targets(
		targets=targets,
		well_workers=int(parallelism.well_workers),
		worker_fn=_worker,
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


def run_preprocess_concatenate_recordings_from_runtime(
	*,
	config_path: str,
	force_restart_override: bool | None = None,
	force_replot_override: bool | None = None,
) -> MultiTargetStageResult:
	return run_preprocess_concat_segments_from_runtime(
		config_path=config_path,
		force_restart_override=force_restart_override,
		force_replot_override=force_replot_override,
	)


def run_preprocess_concatenate_preprocessed_recordings_from_runtime(
	*,
	config_path: str,
	force_restart_override: bool | None = None,
	force_replot_override: bool | None = None,
) -> MultiTargetStageResult:
	return run_preprocess_concatenate_recordings_from_runtime(
		config_path=config_path,
		force_restart_override=force_restart_override,
		force_replot_override=force_replot_override,
	)


def run_preprocess_save_common_electrodes_from_runtime(
	*,
	config_path: str,
	force_restart_override: bool | None = None,
	force_replot_override: bool | None = None,
) -> MultiTargetStageResult:
	return _run_preprocess_substage_from_runtime(
		config_path=config_path,
		stage_name="preprocess.save_common_electrodes",
		runner_fn=run_preprocess_save_common_electrodes,
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


def run_preprocess_build_preprocessed_recording_from_runtime(
	*,
	config_path: str,
	force_restart_override: bool | None = None,
	force_replot_override: bool | None = None,
) -> MultiTargetStageResult:
	return run_preprocess_preprocess_segments_from_runtime(
		config_path=config_path,
		force_restart_override=force_restart_override,
		force_replot_override=force_replot_override,
	)


def run_preprocess_save_concatenated_recording_from_runtime(
	*,
	config_path: str,
	force_restart_override: bool | None = None,
	force_replot_override: bool | None = None,
) -> MultiTargetStageResult:
	return run_preprocess_concatenate_recordings_from_runtime(
		config_path=config_path,
		force_restart_override=force_restart_override,
		force_replot_override=force_replot_override,
	)


def run_preprocess_save_segment_recordings_from_runtime(
	*,
	config_path: str,
	force_restart_override: bool | None = None,
	force_replot_override: bool | None = None,
) -> MultiTargetStageResult:
	return run_preprocess_preprocess_segments_from_runtime(
		config_path=config_path,
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
	targets = select_execution_targets(bundle=bundle)
	if stage_config.debug_limit_wells is not None:
		limit_wells = max(1, int(stage_config.debug_limit_wells))
		if len(targets) > limit_wells:
			LOGGER.info(
				"Applying spikesort debug well limit: %d -> %d target(s)",
				len(targets),
				limit_wells,
			)
			targets = list(targets[:limit_wells])
	parallelism = _resolve_runtime_stage_parallelism(
		bundle=bundle,
		stage_name="spikesort",
		target_count=len(targets),
	)

	def _worker(target):
		inputs = build_spikesort_inputs_for_target(
			target=target,
			stage_config=stage_config,
			unit_workers=int(parallelism.unit_workers),
		)
		return run_spikesort(inputs)

	target_results = distribute_targets(
		targets=targets,
		well_workers=int(parallelism.well_workers),
		worker_fn=_worker,
	)
	target_results = [_publish_spikesort_target_result(item, policy=publish_policy) for item in target_results]

	succeeded = sum(1 for item in target_results if item.status == "ok")
	failed = sum(1 for item in target_results if item.status != "ok")
	return MultiTargetStageResult(
		stage="spikesort",
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
	if stage_config.debug_limit_wells is not None:
		limit_wells = max(1, int(stage_config.debug_limit_wells))
		if len(targets) > limit_wells:
			LOGGER.info(
				"Applying spikesort.merge debug well limit: %d -> %d target(s)",
				len(targets),
				limit_wells,
			)
			targets = list(targets[:limit_wells])
	parallelism = _resolve_runtime_stage_parallelism(
		bundle=bundle,
		stage_name="spikesort",
		target_count=len(targets),
	)

	def _worker(target):
		return run_spikesort_merge(
			h5_path=target.h5_path,
			stream_id=target.stream_id,
			mea_output_root=target.mea_output_root,
			output_rel_root=stage_config.output_rel_root,
			stage_config=stage_config,
			force_restart=bool(
				getattr(
					stage_config,
					"merge_force_restart",
					bool(stage_config.force_restart),
				)
			),
			force_replot=bool(
				getattr(
					stage_config,
					"merge_force_replot",
					bool(stage_config.force_replot),
				)
			),
		)

	target_results = distribute_targets(
		targets=targets,
		well_workers=int(parallelism.well_workers),
		worker_fn=_worker,
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
	force_restart_override: bool | None = None,
	force_replot_override: bool | None = None,
	publish_outputs: bool = False,
) -> MultiTargetStageResult:
	bundle: PipelineRuntimeBundle = load_pipeline_runtime_bundle(config_path=config_path)
	publish_policy = _resolve_publish_policy(runtime_config=bundle.runtime_config, data_config=bundle.data_config)
	if publish_outputs:
		_log_publish_policy(stage_name=stage_name, policy=publish_policy)
	targets = select_execution_targets(bundle=bundle)
	parallelism = _resolve_runtime_stage_parallelism(
		bundle=bundle,
		stage_name="reconstruct",
		target_count=len(targets),
	)
	probe_geometry = parse_probe_geometry_from_data_config(data_config=bundle.data_config)
	stage_config = parse_reconstruction_stage_config(
		runtime_config=bundle.runtime_config,
		unit_id_override=unit_id_override,
		unit_ids_override=unit_ids_override,
		force_restart_override=force_restart_override,
		force_replot_override=force_replot_override,
	)

	def _worker(target):
		inputs = build_reconstruction_inputs_for_target(
			target=target,
			stage_config=stage_config,
			unit_workers=int(parallelism.unit_workers),
			probe_geometry=probe_geometry,
		)
		result = runner_fn(inputs)
		return _raise_reconstruct_unit_failures(stage_name=stage_name, result=result)

	target_results = distribute_targets(
		targets=targets,
		well_workers=int(parallelism.well_workers),
		worker_fn=_worker,
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
	force_restart_override: bool | None = None,
	force_replot_override: bool | None = None,
) -> MultiTargetStageResult:
	return _run_reconstruct_substage_from_runtime(
		config_path=config_path,
		stage_name="reconstruct",
		runner_fn=run_reconstruct,
		unit_id_override=unit_id_override,
		unit_ids_override=unit_ids_override,
		force_restart_override=force_restart_override,
		force_replot_override=force_replot_override,
		publish_outputs=True,
	)


def run_reconstruct_generate_gtrs_from_runtime(
	*,
	config_path: str,
	unit_id_override: int | None = None,
	unit_ids_override: list[int] | None = None,
	force_restart_override: bool | None = None,
	force_replot_override: bool | None = None,
) -> MultiTargetStageResult:
	return _run_reconstruct_substage_from_runtime(
		config_path=config_path,
		stage_name="reconstruct.generate_gtrs",
		runner_fn=run_reconstruct_generate_gtrs,
		unit_id_override=unit_id_override,
		unit_ids_override=unit_ids_override,
		force_restart_override=force_restart_override,
		force_replot_override=force_replot_override,
	)


def run_reconstruct_plot_recons_from_runtime(
	*,
	config_path: str,
	unit_id_override: int | None = None,
	unit_ids_override: list[int] | None = None,
	force_restart_override: bool | None = None,
	force_replot_override: bool | None = None,
) -> MultiTargetStageResult:
	return _run_reconstruct_substage_from_runtime(
		config_path=config_path,
		stage_name="reconstruct.plot_recons",
		runner_fn=run_reconstruct_plot_recons,
		unit_id_override=unit_id_override,
		unit_ids_override=unit_ids_override,
		force_restart_override=force_restart_override,
		force_replot_override=force_replot_override,
	)


def run_reconstruct_plot_branch_propagations_from_runtime(
	*,
	config_path: str,
	unit_id_override: int | None = None,
	unit_ids_override: list[int] | None = None,
	force_restart_override: bool | None = None,
	force_replot_override: bool | None = None,
) -> MultiTargetStageResult:
	return _run_reconstruct_substage_from_runtime(
		config_path=config_path,
		stage_name="reconstruct.plot_branch_propagations",
		runner_fn=run_reconstruct_plot_branch_propagations,
		unit_id_override=unit_id_override,
		unit_ids_override=unit_ids_override,
		force_restart_override=force_restart_override,
		force_replot_override=force_replot_override,
	)


def run_reconstruct_plot_branch_velocities_from_runtime(
	*,
	config_path: str,
	unit_id_override: int | None = None,
	unit_ids_override: list[int] | None = None,
	force_restart_override: bool | None = None,
	force_replot_override: bool | None = None,
) -> MultiTargetStageResult:
	return _run_reconstruct_substage_from_runtime(
		config_path=config_path,
		stage_name="reconstruct.plot_branch_velocities",
		runner_fn=run_reconstruct_plot_branch_velocities,
		unit_id_override=unit_id_override,
		unit_ids_override=unit_ids_override,
		force_restart_override=force_restart_override,
		force_replot_override=force_replot_override,
	)


def run_reconstruct_plot_unit_summary_from_runtime(
	*,
	config_path: str,
	unit_id_override: int | None = None,
	unit_ids_override: list[int] | None = None,
	force_restart_override: bool | None = None,
	force_replot_override: bool | None = None,
) -> MultiTargetStageResult:
	return _run_reconstruct_substage_from_runtime(
		config_path=config_path,
		stage_name="reconstruct.plot_unit_summary",
		runner_fn=run_reconstruct_plot_unit_summary,
		unit_id_override=unit_id_override,
		unit_ids_override=unit_ids_override,
		force_restart_override=force_restart_override,
		force_replot_override=force_replot_override,
	)


def run_reconstruct_report_recons_from_runtime(
	*,
	config_path: str,
	unit_id_override: int | None = None,
	unit_ids_override: list[int] | None = None,
	force_restart_override: bool | None = None,
	force_replot_override: bool | None = None,
) -> MultiTargetStageResult:
	return _run_reconstruct_substage_from_runtime(
		config_path=config_path,
		stage_name="reconstruct.report_recons",
		runner_fn=run_reconstruct_report_recons,
		unit_id_override=unit_id_override,
		unit_ids_override=unit_ids_override,
		force_restart_override=force_restart_override,
		force_replot_override=force_replot_override,
	)


def run_reconstruct_report_full_chip_layout_from_runtime(
	*,
	config_path: str,
	unit_id_override: int | None = None,
	unit_ids_override: list[int] | None = None,
	force_restart_override: bool | None = None,
	force_replot_override: bool | None = None,
) -> MultiTargetStageResult:
	return _run_reconstruct_substage_from_runtime(
		config_path=config_path,
		stage_name="reconstruct.report_full_chip_layout",
		runner_fn=run_reconstruct_report_full_chip_layout,
		unit_id_override=unit_id_override,
		unit_ids_override=unit_ids_override,
		force_restart_override=force_restart_override,
		force_replot_override=force_replot_override,
	)


def run_reconstruct_report_summaries_from_runtime(
	*,
	config_path: str,
	unit_id_override: int | None = None,
	unit_ids_override: list[int] | None = None,
	force_restart_override: bool | None = None,
	force_replot_override: bool | None = None,
) -> MultiTargetStageResult:
	return _run_reconstruct_substage_from_runtime(
		config_path=config_path,
		stage_name="reconstruct.report_summaries",
		runner_fn=run_reconstruct_report_summaries,
		unit_id_override=unit_id_override,
		unit_ids_override=unit_ids_override,
		force_restart_override=force_restart_override,
		force_replot_override=force_replot_override,
	)


def run_analysis_from_runtime(
	*,
	config_path: str,
	unit_id_override: int | None = None,
	force_restart_override: bool | None = None,
	force_replot_override: bool | None = None,
) -> MultiTargetStageResult:
	bundle: PipelineRuntimeBundle = load_pipeline_runtime_bundle(config_path=config_path)
	publish_policy = _resolve_publish_policy(runtime_config=bundle.runtime_config, data_config=bundle.data_config)
	_log_publish_policy(stage_name="analysis", policy=publish_policy)
	targets = select_execution_targets(bundle=bundle)
	parallelism = _resolve_runtime_stage_parallelism(
		bundle=bundle,
		stage_name="analysis",
		target_count=len(targets),
	)
	try:
		probe_pitch_um = float(bundle.data_config.get("Probe.pitch_um", None))
	except Exception:
		probe_pitch_um = None
	stage_config = parse_analysis_stage_config(
		runtime_config=bundle.runtime_config,
		unit_id_override=unit_id_override,
		force_restart_override=force_restart_override,
		force_replot_override=force_replot_override,
	)
	metrics_cfg = getattr(stage_config, "metrics", {})
	metrics_cfg = dict(metrics_cfg) if isinstance(metrics_cfg, dict) else {}
	output_rel_root = str(getattr(stage_config, "output_rel_root", "analysis_outputs"))
	force_restart = bool(getattr(stage_config, "force_restart", False))
	force_replot = bool(getattr(stage_config, "force_replot", False))

	def _worker(target):
		inputs = build_analysis_inputs_for_target(
			target=target,
			stage_config=stage_config,
			unit_workers=int(parallelism.unit_workers),
			probe_pitch_um=probe_pitch_um,
		)
		return run_analysis(inputs)

	target_results = distribute_targets(
		targets=targets,
		well_workers=int(parallelism.well_workers),
		worker_fn=_worker,
	)

	cross_outputs, cross_warnings = generate_cross_well_artifacts(
		target_results=target_results,
		metrics_cfg=metrics_cfg,
		output_rel_root=output_rel_root,
		force_restart=force_restart,
		force_replot=force_replot,
	)
	target_results = [_publish_analysis_target_result(item, policy=publish_policy) for item in target_results]
	cross_outputs = _publish_cross_well_outputs(
		cross_outputs=cross_outputs,
		target_results=target_results,
		policy=publish_policy,
	)
	target_results = _attach_cross_well_outputs_to_analysis_results(
		target_results=target_results,
		cross_outputs=cross_outputs,
		cross_warnings=cross_warnings,
	)

	succeeded = sum(1 for item in target_results if item.status == "ok")
	failed = sum(1 for item in target_results if item.status != "ok")
	return MultiTargetStageResult(
		stage="analysis",
		total_targets=len(target_results),
		succeeded_targets=succeeded,
		failed_targets=failed,
		target_results=target_results,
	)


def run_templates_from_runtime(
	*,
	config_path: str,
	unit_id_override: int | None = None,
	unit_ids_override: list[int] | None = None,
	force_restart_override: bool | None = None,
	force_replot_override: bool | None = None,
) -> MultiTargetStageResult:
	bundle: PipelineRuntimeBundle = load_pipeline_runtime_bundle(config_path=config_path)
	publish_policy = _resolve_publish_policy(runtime_config=bundle.runtime_config, data_config=bundle.data_config)
	_log_publish_policy(stage_name="templates", policy=publish_policy)
	targets = select_execution_targets(bundle=bundle)
	parallelism = _resolve_runtime_stage_parallelism(
		bundle=bundle,
		stage_name="templates",
		target_count=len(targets),
	)
	probe_geometry = parse_probe_geometry_from_data_config(data_config=bundle.data_config)
	stage_config = parse_templates_stage_config(
		runtime_config=bundle.runtime_config,
		probe_geometry=probe_geometry,
		unit_id_override=unit_id_override,
		unit_ids_override=unit_ids_override,
		force_restart_override=force_restart_override,
		force_replot_override=force_replot_override,
	)

	def _worker(target):
		inputs = build_templates_inputs_for_target(
			target=target,
			stage_config=stage_config,
			unit_workers=int(parallelism.unit_workers),
			probe_geometry=probe_geometry,
		)
		return run_templates(inputs)

	target_results = distribute_targets(
		targets=targets,
		well_workers=int(parallelism.well_workers),
		worker_fn=_worker,
	)
	target_results = [_publish_templates_target_result(item, policy=publish_policy) for item in target_results]

	succeeded = sum(1 for item in target_results if item.status == "ok")
	failed = sum(1 for item in target_results if item.status != "ok")
	return MultiTargetStageResult(
		stage="templates",
		total_targets=len(target_results),
		succeeded_targets=succeeded,
		failed_targets=failed,
		target_results=target_results,
	)


def _run_templates_substage_from_runtime(
	*,
	config_path: str,
	stage_name: str,
	runner_fn: Callable[[Any], Any],
	unit_id_override: int | None = None,
	unit_ids_override: list[int] | None = None,
	force_restart_override: bool | None = None,
	force_replot_override: bool | None = None,
	publish_outputs: bool = True,
) -> MultiTargetStageResult:
	bundle: PipelineRuntimeBundle = load_pipeline_runtime_bundle(config_path=config_path)
	publish_policy = _resolve_publish_policy(runtime_config=bundle.runtime_config, data_config=bundle.data_config)
	if publish_outputs:
		_log_publish_policy(stage_name=stage_name, policy=publish_policy)
	targets = select_execution_targets(bundle=bundle)
	parallelism = _resolve_runtime_stage_parallelism(
		bundle=bundle,
		stage_name="templates",
		target_count=len(targets),
	)
	probe_geometry = parse_probe_geometry_from_data_config(data_config=bundle.data_config)
	stage_config = parse_templates_stage_config(
		runtime_config=bundle.runtime_config,
		probe_geometry=probe_geometry,
		unit_id_override=unit_id_override,
		unit_ids_override=unit_ids_override,
		force_restart_override=force_restart_override,
		force_replot_override=force_replot_override,
	)

	def _worker(target):
		inputs = build_templates_inputs_for_target(
			target=target,
			stage_config=stage_config,
			unit_workers=int(parallelism.unit_workers),
			probe_geometry=probe_geometry,
		)
		return runner_fn(inputs)

	target_results = distribute_targets(
		targets=targets,
		well_workers=int(parallelism.well_workers),
		worker_fn=_worker,
	)
	if publish_outputs:
		target_results = [_publish_templates_target_result(item, policy=publish_policy) for item in target_results]

	succeeded = sum(1 for item in target_results if item.status == "ok")
	failed = sum(1 for item in target_results if item.status != "ok")
	return MultiTargetStageResult(
		stage=stage_name,
		total_targets=len(target_results),
		succeeded_targets=succeeded,
		failed_targets=failed,
		target_results=target_results,
	)


def run_templates_resolve_sources_from_runtime(
	*,
	config_path: str,
	unit_id_override: int | None = None,
	unit_ids_override: list[int] | None = None,
	force_restart_override: bool | None = None,
	force_replot_override: bool | None = None,
) -> MultiTargetStageResult:
	return _run_templates_substage_from_runtime(
		config_path=config_path,
		stage_name="templates.resolve_sources",
		runner_fn=run_templates_resolve_sources,
		unit_id_override=unit_id_override,
		unit_ids_override=unit_ids_override,
		force_restart_override=force_restart_override,
		force_replot_override=force_replot_override,
		publish_outputs=False,
	)


def run_templates_analyzers_from_runtime(
	*,
	config_path: str,
	unit_id_override: int | None = None,
	unit_ids_override: list[int] | None = None,
	force_restart_override: bool | None = None,
	force_replot_override: bool | None = None,
) -> MultiTargetStageResult:
	return _run_templates_substage_from_runtime(
		config_path=config_path,
		stage_name="templates.analyzers",
		runner_fn=run_templates_analyzers,
		unit_id_override=unit_id_override,
		unit_ids_override=unit_ids_override,
		force_restart_override=force_restart_override,
		force_replot_override=force_replot_override,
	)


def run_templates_analyzers_concat_from_runtime(
	*,
	config_path: str,
	unit_id_override: int | None = None,
	unit_ids_override: list[int] | None = None,
	force_restart_override: bool | None = None,
	force_replot_override: bool | None = None,
) -> MultiTargetStageResult:
	return _run_templates_substage_from_runtime(
		config_path=config_path,
		stage_name="templates.analyzers.concat",
		runner_fn=lambda inputs: run_templates_analyzers(inputs, source_scope="concat"),
		unit_id_override=unit_id_override,
		unit_ids_override=unit_ids_override,
		force_restart_override=force_restart_override,
		force_replot_override=force_replot_override,
	)


def run_templates_analyzers_segments_from_runtime(
	*,
	config_path: str,
	unit_id_override: int | None = None,
	unit_ids_override: list[int] | None = None,
	force_restart_override: bool | None = None,
	force_replot_override: bool | None = None,
) -> MultiTargetStageResult:
	return _run_templates_substage_from_runtime(
		config_path=config_path,
		stage_name="templates.analyzers.segments",
		runner_fn=lambda inputs: run_templates_analyzers(inputs, source_scope="segments"),
		unit_id_override=unit_id_override,
		unit_ids_override=unit_ids_override,
		force_restart_override=force_restart_override,
		force_replot_override=force_replot_override,
	)


def run_templates_extract_template_segments_from_runtime(
	*,
	config_path: str,
	unit_id_override: int | None = None,
	unit_ids_override: list[int] | None = None,
	force_restart_override: bool | None = None,
	force_replot_override: bool | None = None,
) -> MultiTargetStageResult:
	return _run_templates_substage_from_runtime(
		config_path=config_path,
		stage_name="templates.per_unit_processing.extract_template_segments",
		runner_fn=run_templates_extract_template_segments,
		unit_id_override=unit_id_override,
		unit_ids_override=unit_ids_override,
		force_restart_override=force_restart_override,
		force_replot_override=force_replot_override,
	)


def run_templates_build_templates_from_runtime(
	*,
	config_path: str,
	unit_id_override: int | None = None,
	unit_ids_override: list[int] | None = None,
	force_restart_override: bool | None = None,
	force_replot_override: bool | None = None,
) -> MultiTargetStageResult:
	return _run_templates_substage_from_runtime(
		config_path=config_path,
		stage_name="templates.build_templates",
		runner_fn=run_templates_build_templates,
		unit_id_override=unit_id_override,
		unit_ids_override=unit_ids_override,
		force_restart_override=force_restart_override,
		force_replot_override=force_replot_override,
	)


def run_templates_compute_template_similarity_from_runtime(
	*,
	config_path: str,
	unit_id_override: int | None = None,
	unit_ids_override: list[int] | None = None,
	force_restart_override: bool | None = None,
	force_replot_override: bool | None = None,
) -> MultiTargetStageResult:
	return _run_templates_substage_from_runtime(
		config_path=config_path,
		stage_name="templates.compute_template_similarity",
		runner_fn=run_templates_compute_template_similarity,
		unit_id_override=unit_id_override,
		unit_ids_override=unit_ids_override,
		force_restart_override=force_restart_override,
		force_replot_override=force_replot_override,
	)


def run_templates_plot_templates_from_runtime(
	*,
	config_path: str,
	unit_id_override: int | None = None,
	unit_ids_override: list[int] | None = None,
	force_restart_override: bool | None = None,
	force_replot_override: bool | None = None,
) -> MultiTargetStageResult:
	return _run_templates_substage_from_runtime(
		config_path=config_path,
		stage_name="templates.plot_templates",
		runner_fn=run_templates_plot_templates,
		unit_id_override=unit_id_override,
		unit_ids_override=unit_ids_override,
		force_restart_override=force_restart_override,
		force_replot_override=force_replot_override,
	)


def run_templates_report_templates_from_runtime(
	*,
	config_path: str,
	unit_id_override: int | None = None,
	unit_ids_override: list[int] | None = None,
	force_restart_override: bool | None = None,
	force_replot_override: bool | None = None,
) -> MultiTargetStageResult:
	return _run_templates_substage_from_runtime(
		config_path=config_path,
		stage_name="templates.report_templates",
		runner_fn=run_templates_report_templates,
		unit_id_override=unit_id_override,
		unit_ids_override=unit_ids_override,
		force_restart_override=force_restart_override,
		force_replot_override=force_replot_override,
	)


def run_templates_per_unit_processing_from_runtime(
	*,
	config_path: str,
	unit_id_override: int | None = None,
	unit_ids_override: list[int] | None = None,
	force_restart_override: bool | None = None,
	force_replot_override: bool | None = None,
) -> MultiTargetStageResult:
	return _run_templates_substage_from_runtime(
		config_path=config_path,
		stage_name="templates.per_unit_processing",
		runner_fn=run_templates_per_unit_processing,
		unit_id_override=unit_id_override,
		unit_ids_override=unit_ids_override,
		force_restart_override=force_restart_override,
		force_replot_override=force_replot_override,
	)


def run_templates_reports_from_runtime(
	*,
	config_path: str,
	unit_id_override: int | None = None,
	unit_ids_override: list[int] | None = None,
	force_restart_override: bool | None = None,
	force_replot_override: bool | None = None,
) -> MultiTargetStageResult:
	return _run_templates_substage_from_runtime(
		config_path=config_path,
		stage_name="templates.reports",
		runner_fn=run_templates_reports,
		unit_id_override=unit_id_override,
		unit_ids_override=unit_ids_override,
		force_restart_override=force_restart_override,
		force_replot_override=force_replot_override,
	)


def run_templates_reports_locations_from_runtime(
	*,
	config_path: str,
	unit_id_override: int | None = None,
	unit_ids_override: list[int] | None = None,
	force_restart_override: bool | None = None,
	force_replot_override: bool | None = None,
) -> MultiTargetStageResult:
	return _run_templates_substage_from_runtime(
		config_path=config_path,
		stage_name="templates.reports.locations",
		runner_fn=lambda inputs: run_templates_reports(inputs, report_scope="locations"),
		unit_id_override=unit_id_override,
		unit_ids_override=unit_ids_override,
		force_restart_override=force_restart_override,
		force_replot_override=force_replot_override,
	)


def run_templates_reports_footprints_from_runtime(
	*,
	config_path: str,
	unit_id_override: int | None = None,
	unit_ids_override: list[int] | None = None,
	force_restart_override: bool | None = None,
	force_replot_override: bool | None = None,
) -> MultiTargetStageResult:
	return _run_templates_substage_from_runtime(
		config_path=config_path,
		stage_name="templates.reports.footprints",
		runner_fn=lambda inputs: run_templates_reports(inputs, report_scope="footprints"),
		unit_id_override=unit_id_override,
		unit_ids_override=unit_ids_override,
		force_restart_override=force_restart_override,
		force_replot_override=force_replot_override,
	)


def run_templates_reports_overlays_from_runtime(
	*,
	config_path: str,
	unit_id_override: int | None = None,
	unit_ids_override: list[int] | None = None,
	force_restart_override: bool | None = None,
	force_replot_override: bool | None = None,
) -> MultiTargetStageResult:
	return _run_templates_substage_from_runtime(
		config_path=config_path,
		stage_name="templates.reports.overlays",
		runner_fn=lambda inputs: run_templates_reports(inputs, report_scope="overlays"),
		unit_id_override=unit_id_override,
		unit_ids_override=unit_ids_override,
		force_restart_override=force_restart_override,
		force_replot_override=force_replot_override,
	)
