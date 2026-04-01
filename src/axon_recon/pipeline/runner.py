from __future__ import annotations

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
from .stages.reconstruct.api import run_reconstruct
from .stages.reconstruct.config import build_reconstruction_inputs_for_target, parse_reconstruction_stage_config
from .stages.templates.api import run_templates
from .stages.templates.config import (
	build_templates_inputs_for_target,
	parse_probe_geometry_from_data_config,
	parse_templates_stage_config,
)


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


def run_reconstruct_from_runtime(
	*,
	config_path: str,
	unit_id_override: int | None = None,
	unit_ids_override: list[int] | None = None,
	force_restart_override: bool | None = None,
	force_replot_override: bool | None = None,
) -> MultiTargetStageResult:
	bundle: PipelineRuntimeBundle = load_pipeline_runtime_bundle(config_path=config_path)
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

	succeeded = sum(1 for item in target_results if item.status == "ok")
	failed = sum(1 for item in target_results if item.status != "ok")
	return MultiTargetStageResult(
		stage="templates",
		total_targets=len(target_results),
		succeeded_targets=succeeded,
		failed_targets=failed,
		target_results=target_results,
	)
