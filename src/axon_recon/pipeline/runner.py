from __future__ import annotations

from .config import (
	PipelineRuntimeBundle,
	load_pipeline_runtime_bundle,
	resolve_stage_parallelism,
	select_execution_targets,
)
from .execution.distributor import distribute_targets
from .execution.results import MultiTargetStageResult
from .stages.reconstruct.api import run_reconstruct
from .stages.reconstruct.config import build_reconstruction_inputs_for_target, parse_reconstruction_stage_config
from .stages.templates.api import run_templates
from .stages.templates.config import build_templates_inputs_for_target, parse_templates_stage_config


def run_reconstruct_from_runtime(
	*,
	config_path: str,
	unit_id_override: int | None = None,
	force_restart_override: bool | None = None,
	force_replot_override: bool | None = None,
) -> MultiTargetStageResult:
	bundle: PipelineRuntimeBundle = load_pipeline_runtime_bundle(config_path=config_path)
	targets = select_execution_targets(bundle=bundle)
	parallelism = resolve_stage_parallelism(bundle=bundle, stage_name="reconstruct")
	stage_config = parse_reconstruction_stage_config(
		runtime_config=bundle.runtime_config,
		unit_id_override=unit_id_override,
		force_restart_override=force_restart_override,
		force_replot_override=force_replot_override,
	)

	def _worker(target):
		inputs = build_reconstruction_inputs_for_target(
			target=target,
			stage_config=stage_config,
			unit_workers=int(parallelism.unit_workers),
		)
		return run_reconstruct(inputs)

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


def run_templates_from_runtime(
	*,
	config_path: str,
	unit_id_override: int | None = None,
	force_restart_override: bool | None = None,
	force_replot_override: bool | None = None,
) -> MultiTargetStageResult:
	bundle: PipelineRuntimeBundle = load_pipeline_runtime_bundle(config_path=config_path)
	targets = select_execution_targets(bundle=bundle)
	parallelism = resolve_stage_parallelism(bundle=bundle, stage_name="templates")
	stage_config = parse_templates_stage_config(
		runtime_config=bundle.runtime_config,
		unit_id_override=unit_id_override,
		force_restart_override=force_restart_override,
		force_replot_override=force_replot_override,
	)

	def _worker(target):
		inputs = build_templates_inputs_for_target(
			target=target,
			stage_config=stage_config,
			unit_workers=int(parallelism.unit_workers),
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

