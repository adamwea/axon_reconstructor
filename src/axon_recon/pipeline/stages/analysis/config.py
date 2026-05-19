from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Any

from axon_recon.runtime_config import RuntimeConfig

from ...resources import parse_resources_config, validate_phase_resource_class


LOGGER = logging.getLogger("axon_recon.analysis.config")

_DEFAULT_OUTPUT_REL_ROOT = "analysis_outputs"

DEFAULT_ANALYSIS_PHASE_SEQUENCE: tuple[str, ...] = ("compute_metrics", "unitmatch")

_ANALYSIS_PHASE_ALIASES: dict[str, str] = {
	"compute_metrics": "compute_metrics",
	"metrics": "compute_metrics",
	"compute": "compute_metrics",
	"analysis": "compute_metrics",
	"unitmatch": "unitmatch",
	"unit_match": "unitmatch",
	"match": "unitmatch",
}


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


def _as_optional_positive_int(value: Any) -> int | None:
	if value is None:
		return None
	try:
		parsed = int(value)
	except Exception:
		return None
	return parsed if parsed > 0 else None


def _as_section(value: Any) -> dict[str, Any]:
	return dict(value) if isinstance(value, dict) else {}


def _as_list_of_strings(value: Any) -> list[str]:
	if value is None:
		return []
	if isinstance(value, str):
		items = [value]
	elif isinstance(value, (list, tuple, set)):
		items = list(value)
	else:
		items = [value]
	return [str(item).strip() for item in items if str(item).strip()]


def _normalize_output_rel_root(raw: Any) -> str:
	text = str(raw or _DEFAULT_OUTPUT_REL_ROOT).strip()
	if not text:
		return _DEFAULT_OUTPUT_REL_ROOT
	text = text.lstrip("/")
	return text or _DEFAULT_OUTPUT_REL_ROOT


def normalize_analysis_phase_name(value: Any) -> str:
	text = str(value or "").strip()
	if not text:
		raise ValueError("analysis phase_sequence contains an empty phase name")
	if text.startswith("analysis."):
		text = text.split(".", 1)[1]
	token = text.strip().replace("-", "_").replace(" ", "_").lower()
	canonical = _ANALYSIS_PHASE_ALIASES.get(token)
	if canonical is None:
		raise ValueError(f"Unknown analysis phase_sequence entry: {value!r}")
	return canonical


def _normalize_analysis_phase_sequence(value: Any) -> tuple[str, ...]:
	items = _as_list_of_strings(value)
	if not items:
		return DEFAULT_ANALYSIS_PHASE_SEQUENCE
	return tuple(normalize_analysis_phase_name(item) for item in items)


@dataclass(frozen=True)
class AnalysisStageConfig:
	output_rel_root: str
	phase_sequence: tuple[str, ...]
	compute_metrics_enabled: bool
	compute_metrics_resource_class: str | None
	debug_mode_enabled: bool
	debug_limit_datasets: int | None
	debug_limit_wells: int | None
	debug_limit_wells_per_dataset: int | None
	compute_metrics_debug_mode_enabled: bool
	compute_metrics_debug_limit_datasets: int | None
	compute_metrics_debug_limit_wells: int | None
	compute_metrics_debug_limit_wells_per_dataset: int | None
	recon_output_rel_root: str
	manifest_relpath: str
	tables_relpath: str
	pipeline_version: str
	# Slice 1 of unitmatch_phase_plan: scaffold-only fields. Phase defaults
	# to disabled; subsequent slices add per-phase YAML knobs (match_threshold,
	# good_units_only, backend, …) as separate AnalysisStageConfig fields.
	unitmatch_enabled: bool = False
	unitmatch_resource_class: str | None = None
	unitmatch_rel_output_root: str = "unitmatch"
	well_metadata_lookup: dict[tuple[int, str], dict[str, Any]] = field(default_factory=dict)
	force_restart: bool = False
	replot: bool = False


def parse_analysis_stage_config(
	*,
	runtime_config: RuntimeConfig,
	data_config: RuntimeConfig | None = None,
	force_restart_override: bool | None = None,
	replot_override: bool | None = None,
) -> AnalysisStageConfig:
	stage_cfg = runtime_config.get("stages.analysis", {})
	stage_cfg = stage_cfg if isinstance(stage_cfg, dict) else {}
	phases_cfg = _as_section(stage_cfg.get("phases", {}))
	compute_metrics_phase_cfg = _as_section(phases_cfg.get("compute_metrics", {}))
	unitmatch_phase_cfg = _as_section(phases_cfg.get("unitmatch", {}))

	resources_config = parse_resources_config(runtime_config=runtime_config, logger=LOGGER)
	compute_metrics_resource_class = validate_phase_resource_class(
		resource_class=compute_metrics_phase_cfg.get("resource_class", None),
		resources=resources_config,
		phase_name="analysis.compute_metrics",
	)
	unitmatch_resource_class = validate_phase_resource_class(
		resource_class=unitmatch_phase_cfg.get("resource_class", None),
		resources=resources_config,
		phase_name="analysis.unitmatch",
	)

	debug_mode_cfg = _as_section(stage_cfg.get("debug_mode", {}))
	debug_mode_enabled = _as_bool(debug_mode_cfg.get("enabled", None), False)

	compute_metrics_debug_cfg = _as_section(compute_metrics_phase_cfg.get("debug_mode", {}))
	compute_metrics_debug_enabled = _as_bool(compute_metrics_debug_cfg.get("enabled", None), False)

	output_rel_root = _normalize_output_rel_root(stage_cfg.get("output_rel_root", None))
	phase_sequence = _normalize_analysis_phase_sequence(stage_cfg.get("phase_sequence", None))

	manifest_relpath = str(stage_cfg.get("manifest_relpath", "manifest.json"))
	tables_relpath = str(stage_cfg.get("tables_relpath", "tables"))
	recon_output_rel_root = str(stage_cfg.get("recon_output_rel_root", "recon_outputs"))

	pipeline_version = str(stage_cfg.get("pipeline_version", _resolve_pipeline_version()))

	compute_metrics_enabled = _as_bool(compute_metrics_phase_cfg.get("enabled", True), True)
	unitmatch_enabled = _as_bool(unitmatch_phase_cfg.get("enabled", False), False)
	unitmatch_rel_output_root = str(unitmatch_phase_cfg.get("rel_output_root", "unitmatch") or "unitmatch")

	well_metadata_lookup: dict[tuple[int, str], dict[str, Any]] = {}
	if data_config is not None:
		well_metadata_lookup = build_well_metadata_lookup(data_config)

	return AnalysisStageConfig(
		output_rel_root=output_rel_root,
		phase_sequence=phase_sequence,
		compute_metrics_enabled=compute_metrics_enabled,
		compute_metrics_resource_class=compute_metrics_resource_class,
		debug_mode_enabled=debug_mode_enabled,
		debug_limit_datasets=_as_optional_positive_int(debug_mode_cfg.get("limit_datasets", None)),
		debug_limit_wells=_as_optional_positive_int(debug_mode_cfg.get("limit_wells", None)),
		debug_limit_wells_per_dataset=_as_optional_positive_int(
			debug_mode_cfg.get("limit_wells_per_dataset", None)
		),
		compute_metrics_debug_mode_enabled=compute_metrics_debug_enabled,
		compute_metrics_debug_limit_datasets=_as_optional_positive_int(
			compute_metrics_debug_cfg.get("limit_datasets", None)
		),
		compute_metrics_debug_limit_wells=_as_optional_positive_int(
			compute_metrics_debug_cfg.get("limit_wells", None)
		),
		compute_metrics_debug_limit_wells_per_dataset=_as_optional_positive_int(
			compute_metrics_debug_cfg.get("limit_wells_per_dataset", None)
		),
		recon_output_rel_root=recon_output_rel_root,
		manifest_relpath=manifest_relpath,
		tables_relpath=tables_relpath,
		pipeline_version=pipeline_version,
		unitmatch_enabled=unitmatch_enabled,
		unitmatch_resource_class=unitmatch_resource_class,
		unitmatch_rel_output_root=unitmatch_rel_output_root,
		well_metadata_lookup=well_metadata_lookup,
		force_restart=bool(force_restart_override or False),
		replot=bool(replot_override or False),
	)


def _resolve_pipeline_version() -> str:
	try:
		from importlib.metadata import version as _pkg_version

		return str(_pkg_version("axon_recon"))
	except Exception:
		return "unknown"


def build_well_metadata_lookup(data_config: RuntimeConfig) -> dict[tuple[int, str], dict[str, Any]]:
	"""Index per-well metadata from a data config by (dataset_index, well_id)."""
	datasets = data_config.get("datasets", [])
	if not isinstance(datasets, list):
		return {}
	out: dict[tuple[int, str], dict[str, Any]] = {}
	for dataset_index, dataset in enumerate(datasets):
		if not isinstance(dataset, dict):
			continue
		raw_h5 = dataset.get("raw_data_h5_path", None)
		identity = _parse_identity_from_h5_path(raw_h5)
		div = dataset.get("DIV", None)
		dataset_id = dataset.get("dataset_id", None)
		wells = dataset.get("wells", [])
		if not isinstance(wells, list):
			continue
		for well in wells:
			if not isinstance(well, dict):
				continue
			well_id = well.get("well_id", None)
			if well_id is None:
				continue
			attributes = well.get("attributes", {})
			attributes = dict(attributes) if isinstance(attributes, dict) else {}
			entry: dict[str, Any] = {
				"DIV": div,
				"dataset_id": dataset_id,
				"well_attributes": attributes,
			}
			entry.update(identity)
			out[(dataset_index, str(well_id))] = entry
	return out


def _parse_identity_from_h5_path(raw_h5: Any) -> dict[str, Any]:
	"""Parse <project>/<YYMMDD>/<chip>/<scan_type>/<run>/data.raw.h5 → identity fields."""
	if raw_h5 is None:
		return {
			"project": None,
			"recording_date": None,
			"chip_id": None,
			"scan_type": None,
			"run_id": None,
		}
	path = Path(str(raw_h5))
	# expected tail: <project>/<YYMMDD>/<chip>/<scan_type>/<run>/data.raw.h5
	parts = path.parts
	identity: dict[str, Any] = {
		"project": None,
		"recording_date": None,
		"chip_id": None,
		"scan_type": None,
		"run_id": None,
	}
	if len(parts) < 6:
		return identity
	run_id = parts[-2]
	scan_type = parts[-3]
	chip_id = parts[-4]
	date_token = parts[-5]
	project = parts[-6]
	identity["project"] = project
	identity["chip_id"] = chip_id
	identity["scan_type"] = scan_type
	identity["run_id"] = run_id
	identity["recording_date"] = _normalize_yyMMdd(date_token)
	return identity


def _normalize_yyMMdd(token: str) -> str | None:
	text = str(token or "").strip()
	if len(text) != 6 or not text.isdigit():
		return text or None
	year = 2000 + int(text[0:2])
	month = int(text[2:4])
	day = int(text[4:6])
	if not (1 <= month <= 12 and 1 <= day <= 31):
		return text
	return f"{year:04d}-{month:02d}-{day:02d}"
