from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from axon_reconstructor.runtime_config import RuntimeConfig
from axon_recon.pipeline.shared.plotting import build_stage_plot_block
from axon_recon.pipeline.shared.plotting import SharedHeatmapConfig

from ...execution.context import ExecutionTarget
from .models.inputs import PerUnitOutputsConfig, ReconstructionInputs


def _as_bool(value: Any, default: bool) -> bool:
	if value is None:
		return bool(default)
	if isinstance(value, bool):
		return value
	token = str(value).strip().lower()
	if token in {"1", "true", "yes", "on"}:
		return True
	if token in {"0", "false", "no", "off"}:
		return False
	return bool(default)


def _resolve_data_config_path(runtime_config_path: Path, data_ref: str | None) -> Path:
	if not data_ref:
		raise ValueError("Runtime config must define data: <path-to-data-config>")
	p = Path(str(data_ref)).expanduser()
	if not p.is_absolute():
		p = (runtime_config_path.parent / p).resolve()
	return p


def _deep_merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
	merged = dict(base)
	for key, value in override.items():
		existing = merged.get(key)
		if isinstance(existing, dict) and isinstance(value, dict):
			merged[key] = _deep_merge_dict(existing, value)
		else:
			merged[key] = value
	return merged


def _normalize_png_relpath(raw: Any, default: str) -> str:
	text = str(raw).strip() if raw is not None else str(default)
	if not text:
		text = str(default)
	if "." not in Path(text).name:
		return f"{text}.png"
	return text


def _get_reconstruct_amplitude_map_block(runtime_config: RuntimeConfig) -> dict[str, Any]:
	stage_block = build_stage_plot_block(
		runtime_config=runtime_config,
		stage_paths=(
			"stages.reconstruct.outputs.amplitude_map",
		),
		global_paths=(
			"default",
			"reconstruct",
			"reconstruct.amplitude_map",
			"footprint",
			"footprint_plots.default",
			"footprint_plots.amplitude_map",
		),
	)
	per_unit_block = runtime_config.get("stages.reconstruct.outputs.per_unit_outputs.amplitude_map", {})
	if isinstance(per_unit_block, dict) and per_unit_block:
		return _deep_merge_dict(stage_block, dict(per_unit_block))
	return stage_block


@dataclass(frozen=True)
class ReconstructionStageConfig:
	output_rel_root: str
	write_summary_png: bool
	summary_png_relpath: str
	summary_grid_ncols: int
	write_report_md: bool
	report_md_relpath: str
	per_unit_outputs: PerUnitOutputsConfig
	unit_ids: list[int] | None
	unit_limit: int | None
	use_full_channels_templates: bool
	require_full_channels_templates: bool
	force_restart: bool
	force_replot: bool
	axon_velocity_params: dict[str, Any]


def parse_reconstruction_stage_config(
	*,
	runtime_config: RuntimeConfig,
	unit_id_override: int | None = None,
	force_restart_override: bool | None = None,
	force_replot_override: bool | None = None,
) -> ReconstructionStageConfig:
	stage_cfg = runtime_config.get("stages.reconstruct", {})
	stage_cfg = stage_cfg if isinstance(stage_cfg, dict) else {}
	execution_cfg = stage_cfg.get("execution", {}) if isinstance(stage_cfg.get("execution", {}), dict) else {}
	outputs_cfg = stage_cfg.get("outputs", {}) if isinstance(stage_cfg.get("outputs", {}), dict) else {}
	per_unit_cfg = outputs_cfg.get("per_unit_outputs", {}) if isinstance(outputs_cfg.get("per_unit_outputs", {}), dict) else {}
	av_cfg = stage_cfg.get("av", {}) if isinstance(stage_cfg.get("av", {}), dict) else {}
	amplitude_map_cfg = _get_reconstruct_amplitude_map_block(runtime_config)

	force_restart = _as_bool(execution_cfg.get("force_restart", False), False)
	force_replot = _as_bool(execution_cfg.get("force_replot", False), False)
	if force_restart_override is not None:
		force_restart = bool(force_restart_override)
	if force_replot_override is not None:
		force_replot = bool(force_replot_override)

	unit_limit_raw = stage_cfg.get("unit_limit", None)
	unit_limit: int | None
	if unit_limit_raw is None:
		unit_limit = None
	else:
		try:
			parsed = int(unit_limit_raw)
			unit_limit = parsed if parsed > 0 else None
		except Exception:
			unit_limit = None

	unit_ids = [int(unit_id_override)] if unit_id_override is not None else None

	write_summary_png = _as_bool(outputs_cfg.get("write_summary", False), False)
	summary_png_relpath = _normalize_png_relpath(outputs_cfg.get("summary_relpath", "summary.png"), "summary.png")
	write_report_md = _as_bool(outputs_cfg.get("write_report_md", False), False)
	report_md_relpath = str(outputs_cfg.get("report_md_relpath", "report.md"))
	try:
		summary_grid_ncols = max(1, int(outputs_cfg.get("summary_grid_ncols", 5)))
	except Exception:
		summary_grid_ncols = 5

	if "write_amplitude_map_png" in per_unit_cfg:
		write_amplitude_map_png = _as_bool(per_unit_cfg.get("write_amplitude_map_png", False), False)
	else:
		write_amplitude_map_png = _as_bool(amplitude_map_cfg.get("write_png", False), False)

	if "amplitude_map_png_relpath" in per_unit_cfg:
		amplitude_map_png_relpath = str(per_unit_cfg.get("amplitude_map_png_relpath", "amplitude_map.png"))
	else:
		amplitude_map_png_relpath = _normalize_png_relpath(
			amplitude_map_cfg.get("relpath", "amplitude_map"),
			"amplitude_map.png",
		)

	per_unit = PerUnitOutputsConfig(
		unit_reldir=str(per_unit_cfg.get("unit_reldir", "units/{unit_id:04d}/")),
		write_branches_raw_json=_as_bool(per_unit_cfg.get("write_branches_raw_json", True), True),
		branches_raw_relpath=str(per_unit_cfg.get("branches_raw_relpath", "branches_raw.json")),
		write_branches_json=_as_bool(per_unit_cfg.get("write_branches_json", True), True),
		branches_relpath=str(per_unit_cfg.get("branches_relpath", "branches.json")),
		write_heuristics_json=_as_bool(per_unit_cfg.get("write_heuristics_json", True), True),
		heuristics_relpath=str(per_unit_cfg.get("heuristics_relpath", "heuristics.json")),
		write_gtr_pkl=_as_bool(per_unit_cfg.get("write_gtr_pkl", True), True),
		gtr_pkl_relpath=str(per_unit_cfg.get("gtr_pkl_relpath", "gtr.pkl")),
		write_gtr_json=_as_bool(per_unit_cfg.get("write_gtr_json", False), False),
		gtr_json_relpath=str(per_unit_cfg.get("gtr_json_relpath", "gtr.json")),
		write_amplitude_map_png=write_amplitude_map_png,
		amplitude_map_png_relpath=amplitude_map_png_relpath,
		amplitude_map_heatmap=SharedHeatmapConfig.from_block(amplitude_map_cfg),
	)

	return ReconstructionStageConfig(
		output_rel_root=str(outputs_cfg.get("output_rel_root", "recon_outputs")),
		write_summary_png=write_summary_png,
		summary_png_relpath=summary_png_relpath,
		summary_grid_ncols=summary_grid_ncols,
		write_report_md=write_report_md,
		report_md_relpath=report_md_relpath,
		per_unit_outputs=per_unit,
		unit_ids=unit_ids,
		unit_limit=unit_limit,
		use_full_channels_templates=True,
		require_full_channels_templates=True,
		force_restart=force_restart,
		force_replot=force_replot,
		axon_velocity_params=dict(av_cfg),
	)


def build_reconstruction_inputs_for_target(
	*,
	target: ExecutionTarget,
	stage_config: ReconstructionStageConfig,
	unit_workers: int,
) -> ReconstructionInputs:
	return ReconstructionInputs(
		h5_path=target.h5_path,
		stream_id=target.stream_id,
		mea_output_root=target.mea_output_root,
		output_rel_root=stage_config.output_rel_root,
		write_summary_png=stage_config.write_summary_png,
		summary_png_relpath=stage_config.summary_png_relpath,
		summary_grid_ncols=stage_config.summary_grid_ncols,
		write_report_md=stage_config.write_report_md,
		report_md_relpath=stage_config.report_md_relpath,
		per_unit_outputs=stage_config.per_unit_outputs,
		unit_ids=stage_config.unit_ids,
		unit_limit=stage_config.unit_limit,
		use_full_channels_templates=stage_config.use_full_channels_templates,
		require_full_channels_templates=stage_config.require_full_channels_templates,
		force_restart=stage_config.force_restart,
		force_replot=stage_config.force_replot,
		n_jobs=max(1, int(unit_workers)),
		axon_velocity_params=dict(stage_config.axon_velocity_params),
	)


def load_reconstruction_inputs_from_runtime(
	*,
	config_path: str,
	unit_id_override: int | None = None,
	force_restart_override: bool | None = None,
	force_replot_override: bool | None = None,
) -> ReconstructionInputs:
	runtime_config_path = Path(config_path).expanduser().resolve()
	runtime_cfg = RuntimeConfig.load(runtime_config_path)
	data_cfg_path = _resolve_data_config_path(runtime_config_path, runtime_cfg.get("data", None))
	data_cfg = RuntimeConfig.load(data_cfg_path)

	datasets = data_cfg.get("datasets", [])
	if not isinstance(datasets, list) or not datasets:
		raise ValueError("Data config must define a non-empty datasets list")
	selected = next((ds for ds in datasets if isinstance(ds, dict) and _as_bool(ds.get("include_in_runtime", False), False)), None)
	if selected is None:
		selected = next((ds for ds in datasets if isinstance(ds, dict)), None)
	if selected is None:
		raise ValueError("No valid dataset object found in data config")

	h5_raw = selected.get("raw_data_h5_path")
	if not h5_raw:
		raise ValueError("Selected dataset missing raw_data_h5_path")
	h5_path = Path(str(h5_raw)).expanduser().resolve()
	output_root = Path(str(data_cfg.get("output_root", ""))).expanduser().resolve()
	if str(output_root).strip() == "":
		raise ValueError("Data config missing output_root")

	wells = selected.get("wells", [])
	stream_id = "well000"
	if isinstance(wells, list) and wells and isinstance(wells[0], dict) and wells[0].get("well_id"):
		stream_id = str(wells[0].get("well_id"))

	stage_cfg = parse_reconstruction_stage_config(
		runtime_config=runtime_cfg,
		unit_id_override=unit_id_override,
		force_restart_override=force_restart_override,
		force_replot_override=force_replot_override,
	)

	return ReconstructionInputs(
		h5_path=h5_path,
		stream_id=stream_id,
		mea_output_root=output_root,
		output_rel_root=stage_cfg.output_rel_root,
		write_summary_png=stage_cfg.write_summary_png,
		summary_png_relpath=stage_cfg.summary_png_relpath,
		summary_grid_ncols=stage_cfg.summary_grid_ncols,
		write_report_md=stage_cfg.write_report_md,
		report_md_relpath=stage_cfg.report_md_relpath,
		per_unit_outputs=stage_cfg.per_unit_outputs,
		unit_ids=stage_cfg.unit_ids,
		unit_limit=stage_cfg.unit_limit,
		use_full_channels_templates=stage_cfg.use_full_channels_templates,
		require_full_channels_templates=stage_cfg.require_full_channels_templates,
		force_restart=stage_cfg.force_restart,
		force_replot=stage_cfg.force_replot,
		n_jobs=1,
		axon_velocity_params=stage_cfg.axon_velocity_params,
	)

