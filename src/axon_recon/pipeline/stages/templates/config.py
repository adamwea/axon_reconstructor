from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from axon_reconstructor.runtime_config import RuntimeConfig

from ...execution.context import ExecutionTarget
from .models.inputs import PerUnitTemplatesOutputsConfig, TemplatePlotConfig, TemplatesInputs


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


def _as_float(value: Any, default: float) -> float:
	if value is None:
		return float(default)
	try:
		return float(value)
	except Exception:
		return float(default)


def _normalize_channel_scope(raw: Any) -> str:
	v = str(raw or "").strip().lower().replace("-", "_").replace(" ", "_")
	if v in {"contributing", "contributing_channel", "contributing_channels", "branches"}:
		return "contributing_channels"
	if v in {"recorded", "recorded_channel", "recorded_channels"}:
		return "recorded_channels"
	if v in {"all", "all_channel", "all_channels"}:
		return "all_channels"
	return "contributing_channels"


def _get_template_block(runtime_config: RuntimeConfig) -> dict[str, Any]:
	stage_block = runtime_config.get("stages.templates.outputs.per_unit_outputs.template", {})
	if isinstance(stage_block, dict) and stage_block:
		return dict(stage_block)

	legacy_block = runtime_config.get("stages.reconstruct.outputs.per_unit_outputs.template", {})
	if isinstance(legacy_block, dict):
		return dict(legacy_block)

	return {}


def _get_unit_reldir(runtime_config: RuntimeConfig) -> str:
	raw = runtime_config.get("stages.templates.outputs.per_unit_outputs.unit_reldir", None)
	if raw is not None and str(raw).strip() != "":
		return str(raw)

	legacy = runtime_config.get("stages.reconstruct.outputs.per_unit_outputs.unit_reldir", None)
	if legacy is not None and str(legacy).strip() != "":
		return str(legacy)

	return "units/{unit_id:04d}/"


@dataclass(frozen=True)
class TemplatesStageConfig:
	output_rel_root: str
	per_unit_outputs: PerUnitTemplatesOutputsConfig
	unit_ids: list[int] | None
	unit_limit: int | None
	force_restart: bool
	force_replot: bool


def parse_templates_stage_config(
	*,
	runtime_config: RuntimeConfig,
	unit_id_override: int | None = None,
	force_restart_override: bool | None = None,
	force_replot_override: bool | None = None,
) -> TemplatesStageConfig:
	stage_cfg = runtime_config.get("stages.templates", {})
	stage_cfg = stage_cfg if isinstance(stage_cfg, dict) else {}
	execution_cfg = stage_cfg.get("execution", {}) if isinstance(stage_cfg.get("execution", {}), dict) else {}
	outputs_cfg = stage_cfg.get("outputs", {}) if isinstance(stage_cfg.get("outputs", {}), dict) else {}

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

	tpl_cfg = _get_template_block(runtime_config)
	tpl = TemplatePlotConfig(
		write_png=_as_bool(tpl_cfg.get("write_png", True), True),
		write_svg=_as_bool(tpl_cfg.get("write_svg", False), False),
		relpath=str(tpl_cfg.get("relpath", "template")),
		channel_scope=_normalize_channel_scope(tpl_cfg.get("channel_scope", "contributing_channels")),
		background=str(tpl_cfg.get("background", "black")),
		signal_color=str(tpl_cfg.get("signal_color", "white")),
		force_center_soma=_as_bool(tpl_cfg.get("force_center_soma", False), False),
		force_square_aspect=_as_bool(tpl_cfg.get("force_square_aspect", True), True),
		show_scale_bar=_as_bool(tpl_cfg.get("show_scale_bar", True), True),
		scale_bar_color=str(tpl_cfg.get("scale_bar_color", "white")),
		scale_bar_text_offset_frac=_as_float(tpl_cfg.get("scale_bar_text_offset_frac", 0.02), 0.02),
		scale_bar_y_offset_frac=_as_float(tpl_cfg.get("scale_bar_y_offset_frac", 0.06), 0.06),
		scale_bar_fontsize=_as_float(tpl_cfg.get("scale_bar_fontsize", 6.0), 6.0),
		scale_bar_linewidth=_as_float(tpl_cfg.get("scale_bar_linewidth", 1.8), 1.8),
		scale_bar_length_um=(
			None
			if tpl_cfg.get("scale_bar_length_um", None) is None
			else _as_float(tpl_cfg.get("scale_bar_length_um", None), 0.0)
		),
	)

	per_unit = PerUnitTemplatesOutputsConfig(
		unit_reldir=_get_unit_reldir(runtime_config),
		template=tpl,
	)

	return TemplatesStageConfig(
		output_rel_root=str(outputs_cfg.get("output_rel_root", "templates_outputs")),
		per_unit_outputs=per_unit,
		unit_ids=unit_ids,
		unit_limit=unit_limit,
		force_restart=force_restart,
		force_replot=force_replot,
	)


def build_templates_inputs_for_target(
	*,
	target: ExecutionTarget,
	stage_config: TemplatesStageConfig,
	unit_workers: int,
) -> TemplatesInputs:
	return TemplatesInputs(
		h5_path=target.h5_path,
		stream_id=target.stream_id,
		mea_output_root=target.mea_output_root,
		output_rel_root=stage_config.output_rel_root,
		per_unit_outputs=stage_config.per_unit_outputs,
		unit_ids=stage_config.unit_ids,
		unit_limit=stage_config.unit_limit,
		force_restart=stage_config.force_restart,
		force_replot=stage_config.force_replot,
		n_jobs=max(1, int(unit_workers)),
	)


def load_templates_inputs_from_runtime(
	*,
	config_path: str,
	unit_id_override: int | None = None,
	force_restart_override: bool | None = None,
	force_replot_override: bool | None = None,
) -> TemplatesInputs:
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

	stage_cfg = parse_templates_stage_config(
		runtime_config=runtime_cfg,
		unit_id_override=unit_id_override,
		force_restart_override=force_restart_override,
		force_replot_override=force_replot_override,
	)

	return TemplatesInputs(
		h5_path=h5_path,
		stream_id=stream_id,
		mea_output_root=output_root,
		output_rel_root=stage_cfg.output_rel_root,
		per_unit_outputs=stage_cfg.per_unit_outputs,
		unit_ids=stage_cfg.unit_ids,
		unit_limit=stage_cfg.unit_limit,
		force_restart=stage_cfg.force_restart,
		force_replot=stage_cfg.force_replot,
		n_jobs=1,
	)
