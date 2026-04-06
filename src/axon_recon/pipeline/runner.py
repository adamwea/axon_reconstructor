from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any

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
from .stages.preprocess.api import run_preprocess
from .stages.preprocess.config import build_preprocess_inputs_for_target, parse_preprocess_stage_config
from .stages.preprocess.models.results import PreprocessResult
from .stages.reconstruct.api import run_reconstruct
from .stages.reconstruct.config import build_reconstruction_inputs_for_target, parse_reconstruction_stage_config
from .stages.reconstruct.models.results import ReconstructionResult, UnitReconstructionResult
from .stages.spikesort.api import run_spikesort, run_spikesort_merge
from .stages.spikesort.config import build_spikesort_inputs_for_target, parse_spikesort_stage_config
from .stages.spikesort.models.results import SpikesortMergeResult, SpikesortResult
from .stages.templates.api import run_templates, run_templates_resolve_sources
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
	targets = select_execution_targets(bundle=bundle)
	if stage_config.debug_limit_wells is not None:
		limit_wells = max(1, int(stage_config.debug_limit_wells))
		if len(targets) > limit_wells:
			LOGGER.info(
				"Applying preprocess debug well limit: %d -> %d target(s)",
				len(targets),
				limit_wells,
			)
			targets = list(targets[:limit_wells])
	parallelism = resolve_stage_parallelism(bundle=bundle, stage_name="preprocess")

	def _worker(target):
		inputs = build_preprocess_inputs_for_target(
			target=target,
			stage_config=stage_config,
			unit_workers=int(parallelism.unit_workers),
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
	parallelism = resolve_stage_parallelism(bundle=bundle, stage_name="spikesort")

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
) -> MultiTargetStageResult:
	bundle: PipelineRuntimeBundle = load_pipeline_runtime_bundle(config_path=config_path)
	publish_policy = _resolve_publish_policy(runtime_config=bundle.runtime_config, data_config=bundle.data_config)
	_log_publish_policy(stage_name="spikesort.merge", policy=publish_policy)
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
				"Applying spikesort.merge debug well limit: %d -> %d target(s)",
				len(targets),
				limit_wells,
			)
			targets = list(targets[:limit_wells])
	parallelism = resolve_stage_parallelism(bundle=bundle, stage_name="spikesort")

	def _worker(target):
		return run_spikesort_merge(
			h5_path=target.h5_path,
			stream_id=target.stream_id,
			mea_output_root=target.mea_output_root,
			output_rel_root=stage_config.output_rel_root,
			stage_config=stage_config,
			force_restart=bool(stage_config.force_restart or stage_config.force_replot),
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
		stage="spikesort.merge",
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
	bundle: PipelineRuntimeBundle = load_pipeline_runtime_bundle(config_path=config_path)
	publish_policy = _resolve_publish_policy(runtime_config=bundle.runtime_config, data_config=bundle.data_config)
	_log_publish_policy(stage_name="reconstruct", policy=publish_policy)
	targets = select_execution_targets(bundle=bundle)
	parallelism = resolve_stage_parallelism(bundle=bundle, stage_name="reconstruct")
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
		result = run_reconstruct(inputs)
		ok_units = [u for u in result.units if str(getattr(u, "status", "ok")).strip().lower() == "ok"]
		failed_units = [u for u in result.units if str(getattr(u, "status", "ok")).strip().lower() != "ok"]
		if ok_units:
			return result
		if failed_units:
			first = failed_units[0]
			raise RuntimeError(
				"reconstruct unit failures: "
				f"succeeded={len(ok_units)}/{len(result.units)} "
				f"failed={len(failed_units)}/{len(result.units)} "
				f"first_unit={getattr(first, 'unit_id', 'unknown')} "
				f"first_error={getattr(first, 'error', None) or getattr(first, 'status', 'error')}"
			)
		raise RuntimeError("reconstruct produced no successful units")

	target_results = distribute_targets(
		targets=targets,
		well_workers=int(parallelism.well_workers),
		worker_fn=_worker,
	)
	target_results = [_publish_reconstruct_target_result(item, policy=publish_policy) for item in target_results]

	succeeded = sum(1 for item in target_results if item.status == "ok")
	failed = sum(1 for item in target_results if item.status != "ok")
	return MultiTargetStageResult(
		stage="reconstruct",
		total_targets=len(target_results),
		succeeded_targets=succeeded,
		failed_targets=failed,
		target_results=target_results,
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
	parallelism = resolve_stage_parallelism(bundle=bundle, stage_name="analysis")
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
	parallelism = resolve_stage_parallelism(bundle=bundle, stage_name="templates")
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


def run_templates_resolve_sources_from_runtime(
	*,
	config_path: str,
	unit_id_override: int | None = None,
	unit_ids_override: list[int] | None = None,
	force_restart_override: bool | None = None,
	force_replot_override: bool | None = None,
) -> MultiTargetStageResult:
	bundle: PipelineRuntimeBundle = load_pipeline_runtime_bundle(config_path=config_path)
	targets = select_execution_targets(bundle=bundle)
	parallelism = resolve_stage_parallelism(bundle=bundle, stage_name="templates")
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
		return run_templates_resolve_sources(inputs)

	target_results = distribute_targets(
		targets=targets,
		well_workers=int(parallelism.well_workers),
		worker_fn=_worker,
	)

	succeeded = sum(1 for item in target_results if item.status == "ok")
	failed = sum(1 for item in target_results if item.status != "ok")
	return MultiTargetStageResult(
		stage="templates.resolve_sources",
		total_targets=len(target_results),
		succeeded_targets=succeeded,
		failed_targets=failed,
		target_results=target_results,
	)
