from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from axon_reconstructor.runtime_config import RuntimeConfig

from ...execution.context import ExecutionTarget
from .models.inputs import AnalysisInputs


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


def _as_unit_ids(raw: Any) -> list[int] | None:
	if raw is None:
		return None
	if isinstance(raw, (str, bytes)):
		try:
			return [int(raw)]
		except Exception:
			return None
	values = raw if isinstance(raw, (list, tuple)) else [raw]
	out: list[int] = []
	for value in values:
		try:
			out.append(int(value))
		except Exception:
			continue
	return out or None


def _as_unit_limit(raw: Any) -> int | None:
	if raw is None:
		return None
	try:
		limit = int(raw)
	except Exception:
		return None
	return limit if limit > 0 else None


def _as_float_or_none(raw: Any) -> float | None:
	if raw is None:
		return None
	try:
		return float(raw)
	except Exception:
		return None


def _collect_deferred_warnings(metrics_cfg: dict[str, Any]) -> list[str]:
	warnings: list[str] = []
	cross_well = metrics_cfg.get("cross_well", {}) if isinstance(metrics_cfg.get("cross_well", {}), dict) else {}
	stats_cfg = cross_well.get("statistical_testing", {}) if isinstance(cross_well.get("statistical_testing", {}), dict) else {}
	if not _as_bool(stats_cfg.get("enable", False), False):
		return warnings

	unsupported: list[str] = []
	if _as_bool(stats_cfg.get("paired_wells", False), False):
		unsupported.append("paired_wells")

	normality_cfg = stats_cfg.get("normality_check", {}) if isinstance(stats_cfg.get("normality_check", {}), dict) else {}
	if _as_bool(normality_cfg.get("enable", False), False):
		unsupported.append("normality_check")

	equal_var_cfg = stats_cfg.get("equal_variance_check", {}) if isinstance(stats_cfg.get("equal_variance_check", {}), dict) else {}
	if _as_bool(equal_var_cfg.get("enable", False), False):
		unsupported.append("equal_variance_check")

	two_group = stats_cfg.get("two_group_test", {}) if isinstance(stats_cfg.get("two_group_test", {}), dict) else {}
	selection = str(two_group.get("selection", "auto") or "auto").strip().lower()
	if selection not in {"auto", "mannwhitneyu"}:
		unsupported.append("two_group_test.selection")

	perm_cfg = (
		two_group.get("permutation_fallback", {})
		if isinstance(two_group.get("permutation_fallback", {}), dict)
		else {}
	)
	if _as_bool(perm_cfg.get("enable", False), False):
		unsupported.append("two_group_test.permutation_fallback")

	if unsupported:
		unsupported_list = ", ".join(sorted(set(unsupported)))
		warnings.append(
			"Cross-well statistical_testing options are partially unsupported in v2: "
			f"{unsupported_list}. "
			"Current implementation supports pairwise Mann-Whitney U with optional p-value correction and effect sizes."
		)

	return warnings


@dataclass(frozen=True)
class AnalysisStageConfig:
	output_rel_root: str
	metrics: dict[str, Any]
	unit_ids: list[int] | None
	unit_limit: int | None
	force_restart: bool
	force_replot: bool
	deferred_warnings: list[str]


def parse_analysis_stage_config(
	*,
	runtime_config: RuntimeConfig,
	unit_id_override: int | None = None,
	force_restart_override: bool | None = None,
	force_replot_override: bool | None = None,
) -> AnalysisStageConfig:
	stage_cfg = runtime_config.get("stages.analysis", {})
	stage_cfg = stage_cfg if isinstance(stage_cfg, dict) else {}
	execution_cfg = stage_cfg.get("execution", {}) if isinstance(stage_cfg.get("execution", {}), dict) else {}
	outputs_cfg = stage_cfg.get("outputs", {}) if isinstance(stage_cfg.get("outputs", {}), dict) else {}
	metrics_cfg = outputs_cfg.get("metrics", {}) if isinstance(outputs_cfg.get("metrics", {}), dict) else {}

	force_restart = _as_bool(execution_cfg.get("force_restart", False), False)
	force_replot = _as_bool(execution_cfg.get("force_replot", False), False)
	if force_restart_override is not None:
		force_restart = bool(force_restart_override)
	if force_replot_override is not None:
		force_replot = bool(force_replot_override)

	unit_ids = _as_unit_ids(stage_cfg.get("unit_ids", None))
	if unit_id_override is not None:
		unit_ids = [int(unit_id_override)]

	return AnalysisStageConfig(
		output_rel_root=str(outputs_cfg.get("output_rel_root", "analysis_outputs")),
		metrics=dict(metrics_cfg),
		unit_ids=unit_ids,
		unit_limit=_as_unit_limit(stage_cfg.get("unit_limit", None)),
		force_restart=force_restart,
		force_replot=force_replot,
		deferred_warnings=_collect_deferred_warnings(metrics_cfg),
	)


def build_analysis_inputs_for_target(
	*,
	target: ExecutionTarget,
	stage_config: AnalysisStageConfig,
	unit_workers: int,
	probe_pitch_um: float | None = None,
) -> AnalysisInputs:
	return AnalysisInputs(
		h5_path=target.h5_path,
		stream_id=target.stream_id,
		mea_output_root=target.mea_output_root,
		final_output_root=(target.final_output_root or target.mea_output_root),
		probe_pitch_um=probe_pitch_um,
		output_rel_root=stage_config.output_rel_root,
		unit_ids=stage_config.unit_ids,
		unit_limit=stage_config.unit_limit,
		force_restart=stage_config.force_restart,
		force_replot=stage_config.force_replot,
		n_jobs=max(1, int(unit_workers)),
		metrics=dict(stage_config.metrics),
		deferred_warnings=list(stage_config.deferred_warnings),
	)


def load_analysis_inputs_from_runtime(
	*,
	config_path: str,
	unit_id_override: int | None = None,
	force_restart_override: bool | None = None,
	force_replot_override: bool | None = None,
) -> AnalysisInputs:
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
	probe_pitch_um = _as_float_or_none(data_cfg.get("Probe.pitch_um", None))

	wells = selected.get("wells", [])
	stream_id = "well000"
	if isinstance(wells, list) and wells and isinstance(wells[0], dict) and wells[0].get("well_id"):
		stream_id = str(wells[0].get("well_id"))

	stage_cfg = parse_analysis_stage_config(
		runtime_config=runtime_cfg,
		unit_id_override=unit_id_override,
		force_restart_override=force_restart_override,
		force_replot_override=force_replot_override,
	)

	return AnalysisInputs(
		h5_path=h5_path,
		stream_id=stream_id,
		mea_output_root=output_root,
		final_output_root=output_root,
		probe_pitch_um=probe_pitch_um,
		output_rel_root=stage_cfg.output_rel_root,
		unit_ids=stage_cfg.unit_ids,
		unit_limit=stage_cfg.unit_limit,
		force_restart=stage_cfg.force_restart,
		force_replot=stage_cfg.force_replot,
		n_jobs=1,
		metrics=dict(stage_cfg.metrics),
		deferred_warnings=list(stage_cfg.deferred_warnings),
	)
