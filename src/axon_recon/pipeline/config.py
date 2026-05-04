from __future__ import annotations

from dataclasses import dataclass, replace
import logging
from pathlib import Path
from typing import Any

from axon_recon.pipeline.scratch_layout import resolve_optional_path, resolve_scratch_layout
from axon_recon.runtime_config import RuntimeConfig

from .execution.context import ExecutionTarget, StageParallelism
from .execution.read_groups import count_target_read_groups
from .resources import (
	SOURCE_H5_PATH_KEYED_RESOURCE,
	get_active_resource_profile,
	get_keyed_resource_limit_max_concurrent,
	get_max_phase_resource_demands,
	get_resource_default,
	parse_resources_config,
)
from .stages.preprocess.core.copy_src_to_scratch import resolve_copy_src_to_scratch_input_path


LOGGER = logging.getLogger("axon_recon.pipeline.config")
_WARNED_LEGACY_STAGE_PARALLELISM_KEYS: set[tuple[str, tuple[str, ...]]] = set()


def _warn_legacy_scratch_input_keys(*, scope: str) -> None:
	LOGGER.warning(
		"Legacy scratch input keys detected for %s; scratch_input_root/use_scratch_input_root are deprecated. "
		"Prefer scratch_root and the canonical scratch_root/axon_recon_scratch/{inputs,outputs} layout.",
		str(scope),
	)


@dataclass(frozen=True)
class PipelineRuntimeBundle:
	runtime_config_path: Path
	data_config_path: Path
	runtime_config: RuntimeConfig
	data_config: RuntimeConfig


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


def _as_int(value: Any, default: int) -> int:
	if value is None:
		return int(default)
	try:
		return int(value)
	except Exception:
		return int(default)


def _as_optional_positive_int(value: Any) -> int | None:
	if value is None:
		return None
	try:
		parsed = int(value)
	except Exception:
		return None
	if parsed <= 0:
		return None
	return parsed


def _warn_legacy_stage_parallelism_keys(*, runtime_cfg: RuntimeConfig, stage_name: str) -> None:
	has_fn = getattr(runtime_cfg, "has", None)
	legacy_keys: list[str] = []
	for key in ("max_stage_workers", "well_workers", "divide_stage_workers_by_wells"):
		path = f"stages.{stage_name}.resources.{key}"
		try:
			has_value = bool(has_fn(path)) if callable(has_fn) else False
		except Exception:
			has_value = False
		if has_value:
			legacy_keys.append(key)
	if not legacy_keys:
		return
	token = (str(stage_name), tuple(sorted(legacy_keys)))
	if token in _WARNED_LEGACY_STAGE_PARALLELISM_KEYS:
		return
	_WARNED_LEGACY_STAGE_PARALLELISM_KEYS.add(token)
	LOGGER.warning(
		"Ignoring legacy stage resource keys for %s: %s. Stage parallelism now derives from resources.active_profile and enabled phase resource_class values.",
		str(stage_name),
		", ".join(sorted(legacy_keys)),
	)


def _as_path_list(value: Any, *, base_dir: Path | None = None) -> list[Path]:
	if value is None:
		return []
	if isinstance(value, (list, tuple, set)):
		items = list(value)
	else:
		items = [value]

	paths: list[Path] = []
	seen: set[Path] = set()
	for item in items:
		if item is None:
			continue
		token = str(item).strip()
		if token == "":
			continue
		path = _normalize_config_input_path(token, base_dir=base_dir) if base_dir is not None else Path(token).expanduser()
		if path in seen:
			continue
		seen.add(path)
		paths.append(path)
	return paths


def _resolve_data_config_path(runtime_config_path: Path, data_ref: str | None) -> Path:
	if not data_ref:
		raise ValueError("Runtime config must define data: <path-to-data-config>")
	p = Path(str(data_ref)).expanduser()
	if not p.is_absolute():
		p = (runtime_config_path.parent / p).resolve()
	return p


def _normalize_config_input_path(raw_path: Any, *, base_dir: Path) -> Path:
	path = Path(str(raw_path)).expanduser()
	if not path.is_absolute():
		path = base_dir / path
	return path


def load_pipeline_runtime_bundle(*, config_path: str) -> PipelineRuntimeBundle:
	runtime_config_path = Path(config_path).expanduser().resolve()
	runtime_cfg = RuntimeConfig.load(runtime_config_path)

	data_cfg_path = _resolve_data_config_path(runtime_config_path, runtime_cfg.get("data", None))
	data_cfg = RuntimeConfig.load(data_cfg_path)

	return PipelineRuntimeBundle(
		runtime_config_path=runtime_config_path,
		data_config_path=data_cfg_path,
		runtime_config=runtime_cfg,
		data_config=data_cfg,
	)


def _dataset_id_for_item(item: dict[str, Any], *, index: int) -> str:
	raw = item.get("dataset_id", None)
	if raw is not None and str(raw).strip() != "":
		return str(raw)
	h5 = item.get("raw_data_h5_path", None)
	if h5 is not None and str(h5).strip() != "":
		return f"dataset_{index:03d}:{Path(str(h5)).name}"
	return f"dataset_{index:03d}"


def select_execution_targets(
	*,
	bundle: PipelineRuntimeBundle,
	materialize_scratch_inputs: bool = False,
	limit_datasets: int | None = None,
	limit_wells: int | None = None,
	limit_wells_per_dataset: int | None = None,
) -> list[ExecutionTarget]:
	datasets = bundle.data_config.get("datasets", [])
	if not isinstance(datasets, list) or not datasets:
		raise ValueError("Data config must define a non-empty datasets list")

	enabled: list[tuple[int, dict[str, Any]]] = []
	for idx, item in enumerate(datasets):
		if not isinstance(item, dict):
			continue
		if _as_bool(item.get("include_in_runtime", False), False):
			enabled.append((idx, item))

	if not enabled:
		raise ValueError(
			"No datasets enabled for runtime execution. Set datasets[*].include_in_runtime=true "
			"for each recording you want to include."
		)
	if limit_datasets is not None:
		dataset_limit = max(1, int(limit_datasets))
		if len(enabled) > dataset_limit:
			LOGGER.info(
				"Applying execution target dataset limit before scratch materialization: %d -> %d dataset(s)",
				len(enabled),
				dataset_limit,
			)
			enabled = enabled[:dataset_limit]
	global_well_limit = max(1, int(limit_wells)) if limit_wells is not None else None
	well_limit_per_dataset = max(1, int(limit_wells_per_dataset)) if limit_wells_per_dataset is not None else None

	output_root_raw = bundle.data_config.get("output_root", None)
	if not output_root_raw:
		raise ValueError("Data config missing output_root")
	output_root = _normalize_config_input_path(output_root_raw, base_dir=bundle.data_config_path.parent)
	default_lookup_roots = _as_path_list(
		bundle.data_config.get("output_root_2", None),
		base_dir=bundle.data_config_path.parent,
	)
	scratch_root_raw = bundle.data_config.get("scratch_root", None)
	use_scratch_root = _as_bool(bundle.data_config.get("use_scratch_root", True), True)
	default_scratch_layout = resolve_scratch_layout(scratch_root_raw) if bool(use_scratch_root) else None
	default_scratch_output_root = None if default_scratch_layout is None else default_scratch_layout.outputs_root
	default_scratch_input_root = None if default_scratch_layout is None else default_scratch_layout.inputs_root

	scratch_input_root_raw = bundle.data_config.get("scratch_input_root", None)
	use_scratch_input_root_raw = bundle.data_config.get("use_scratch_input_root", None)
	use_scratch_input_root = _as_bool(use_scratch_input_root_raw, False)
	if scratch_input_root_raw is not None or use_scratch_input_root_raw is not None:
		_warn_legacy_scratch_input_keys(scope="data config")
		if bool(use_scratch_input_root):
			default_scratch_input_root = resolve_optional_path(scratch_input_root_raw) or default_scratch_input_root
		else:
			default_scratch_input_root = None

	targets: list[ExecutionTarget] = []
	for idx, item in enabled:
		if global_well_limit is not None and len(targets) >= int(global_well_limit):
			break
		h5_raw = item.get("raw_data_h5_path", None)
		if not h5_raw:
			continue
		h5_path = _normalize_config_input_path(h5_raw, base_dir=bundle.data_config_path.parent)
		dataset_id = _dataset_id_for_item(item, index=idx)

		dataset_scratch_root_raw = item.get("scratch_root", None)
		dataset_use_scratch_root = _as_bool(item.get("use_scratch_root", use_scratch_root), use_scratch_root)
		if bool(dataset_use_scratch_root):
			dataset_scratch_layout = (
				resolve_scratch_layout(dataset_scratch_root_raw)
				if dataset_scratch_root_raw is not None and str(dataset_scratch_root_raw).strip() != ""
				else default_scratch_layout
			)
		else:
			dataset_scratch_layout = None
		dataset_scratch_output_root = None if dataset_scratch_layout is None else dataset_scratch_layout.outputs_root
		dataset_scratch_input_root = None if dataset_scratch_layout is None else dataset_scratch_layout.inputs_root

		dataset_scratch_input_root_raw = item.get("scratch_input_root", None)
		dataset_use_scratch_input_root_raw = item.get("use_scratch_input_root", None)
		if dataset_scratch_input_root_raw is not None or dataset_use_scratch_input_root_raw is not None:
			_warn_legacy_scratch_input_keys(scope=f"dataset {dataset_id}")
			dataset_use_scratch_input_root = _as_bool(dataset_use_scratch_input_root_raw, use_scratch_input_root)
			if bool(dataset_use_scratch_input_root):
				dataset_scratch_input_root = (
					resolve_optional_path(dataset_scratch_input_root_raw) or default_scratch_input_root
				)
			else:
				dataset_scratch_input_root = None

		active_root = dataset_scratch_output_root if dataset_scratch_output_root is not None else output_root
		artifact_lookup_roots: list[Path] = []
		for candidate_root in _as_path_list(
			item.get("output_root_2", None),
			base_dir=bundle.data_config_path.parent,
		) + default_lookup_roots:
			if candidate_root == active_root:
				continue
			if candidate_root in artifact_lookup_roots:
				continue
			artifact_lookup_roots.append(candidate_root)

		wells = item.get("wells", [])
		if not isinstance(wells, list) or not wells:
			wells = [{"well_id": "well000"}]

		selected_stream_ids: list[str] = []
		for well_item in wells:
			if well_limit_per_dataset is not None and len(selected_stream_ids) >= int(well_limit_per_dataset):
				break
			if global_well_limit is not None and (len(targets) + len(selected_stream_ids)) >= int(global_well_limit):
				break
			stream_id = "well000"
			well_enabled = False
			if isinstance(well_item, dict):
				well_enabled = _as_bool(well_item.get("include_in_runtime", False), False)
				if well_item.get("well_id"):
					stream_id = str(well_item.get("well_id"))
			if not well_enabled:
				continue
			selected_stream_ids.append(str(stream_id))

		if not selected_stream_ids:
			continue

		target_h5_path = resolve_copy_src_to_scratch_input_path(
			source_h5_path=h5_path,
			scratch_input_root=dataset_scratch_input_root,
			dataset_id=str(dataset_id),
			materialize_scratch_inputs=bool(materialize_scratch_inputs),
		)

		for stream_id in selected_stream_ids:
			targets.append(
				ExecutionTarget(
					dataset_index=int(idx),
					dataset_id=str(dataset_id),
					h5_path=target_h5_path,
					source_h5_path=h5_path,
					stream_id=str(stream_id),
					mea_output_root=active_root,
					final_output_root=output_root,
					scratch_output_root=dataset_scratch_output_root,
					artifact_lookup_roots=tuple(artifact_lookup_roots),
				)
			)

	if not targets:
		raise ValueError(
			"No execution targets were produced. Ensure both datasets[*].include_in_runtime=true "
			"and wells[*].include_in_runtime=true for each well to run."
		)

	targets.sort(key=lambda t: (t.dataset_index, t.stream_id))
	return targets


def resolve_stage_parallelism(
	*,
	bundle: PipelineRuntimeBundle,
	stage_name: str,
	target_count: int | None = None,
	phase_resource_classes: list[str] | tuple[str, ...] | None = None,
) -> StageParallelism:
	runtime_cfg = bundle.runtime_config
	resources_config = parse_resources_config(runtime_config=runtime_cfg, logger=LOGGER)
	active_profile = get_active_resource_profile(resources_config)
	legacy_max_workers = _as_int(
		runtime_cfg.get(
			"resources.max_workers",
			get_resource_default(runtime_config=runtime_cfg, key="max_workers", default=8, logger=LOGGER),
		),
		8,
	)
	max_workers = max(1, int(active_profile.cpu_cores)) if active_profile is not None and active_profile.cpu_cores else max(1, int(legacy_max_workers))
	stage_workers = int(max_workers)
	resolved_phase_resource_classes = tuple(
		str(resource_class)
		for resource_class in (phase_resource_classes or ())
		if str(resource_class).strip()
	)
	resource_driven_parallelism = bool(active_profile is not None and resolved_phase_resource_classes)
	if resource_driven_parallelism:
		_warn_legacy_stage_parallelism_keys(runtime_cfg=runtime_cfg, stage_name=stage_name)
		max_phase_demands = get_max_phase_resource_demands(
			resources=resources_config,
			resource_classes=list(resolved_phase_resource_classes),
			dimensions=("cpu_cores", "ram_gb"),
		)
		candidate_well_worker_limits: list[int] = []
		cpu_demand = max(0, int(max_phase_demands.get("cpu_cores", 0)))
		ram_demand = max(0, int(max_phase_demands.get("ram_gb", 0)))
		if target_count is not None:
			resolved_target_count = max(0, int(target_count))
			if resolved_target_count > 0:
				candidate_well_worker_limits.append(int(resolved_target_count))
		if cpu_demand > 0:
			candidate_well_worker_limits.append(max(1, int(stage_workers // cpu_demand)))
		if ram_demand > 0 and active_profile is not None and active_profile.ram_gb is not None:
			ram_budget = max(0, int(active_profile.ram_gb))
			candidate_well_worker_limits.append(max(1, int(ram_budget // ram_demand)))
		well_workers = min(candidate_well_worker_limits) if candidate_well_worker_limits else 1
		well_workers = max(1, int(well_workers))
		divide_stage_workers_by_wells = False
		derived_unit_workers = max(1, int(cpu_demand or max(1, stage_workers // well_workers)))
		derived_unit_workers_source = "resource_class.cpu_cores" if cpu_demand > 0 else "derived"
	else:
		stage_workers = _as_int(runtime_cfg.get(f"stages.{stage_name}.resources.max_stage_workers", max_workers), max_workers)
		stage_workers = max(1, int(stage_workers))

		well_workers = _as_int(runtime_cfg.get(f"stages.{stage_name}.resources.well_workers", 1), 1)
		well_workers = max(1, int(well_workers))
		if target_count is not None:
			resolved_target_count = max(0, int(target_count))
			if resolved_target_count > 0:
				well_workers = min(well_workers, resolved_target_count)

		divide_stage_workers_by_wells = _as_bool(
			runtime_cfg.get(f"stages.{stage_name}.resources.divide_stage_workers_by_wells", True),
			True,
		)
		if bool(divide_stage_workers_by_wells) and int(well_workers) > 1:
			derived_unit_workers = max(1, int(stage_workers // well_workers))
		else:
			derived_unit_workers = max(1, int(stage_workers))
		derived_unit_workers_source = "derived"
	read_cap_raw = None
	for read_cap_key in (
		f"stages.{stage_name}.resources.max_simultaneous_well_reads_per_h5_file",
		f"stages.{stage_name}.resources.max_simultaneous_well_reads_per_dataset",
	):
		if runtime_cfg.has(read_cap_key):
			read_cap_raw = runtime_cfg.get(read_cap_key, None)
			break
	if read_cap_raw is None:
		read_cap_raw = get_keyed_resource_limit_max_concurrent(
			resources_config,
			SOURCE_H5_PATH_KEYED_RESOURCE,
		)
	max_simultaneous_well_reads_per_dataset = _as_optional_positive_int(read_cap_raw)

	unit_workers_key = f"stages.{stage_name}.resources.unit_workers"
	unit_workers_raw = runtime_cfg.get(unit_workers_key, None) if runtime_cfg.has(unit_workers_key) else None
	if unit_workers_raw is None:
		unit_workers = int(derived_unit_workers)
		unit_workers_source = str(derived_unit_workers_source)
	else:
		unit_workers = max(1, _as_int(unit_workers_raw, int(derived_unit_workers)))
		unit_workers_source = "resources.unit_workers"

	return StageParallelism(
		max_workers=max_workers,
		max_stage_workers=stage_workers,
		well_workers=well_workers,
		unit_workers=unit_workers,
		unit_workers_source=unit_workers_source,
		max_simultaneous_well_reads_per_dataset=max_simultaneous_well_reads_per_dataset,
		divide_stage_workers_by_wells=bool(divide_stage_workers_by_wells),
	)


def constrain_stage_parallelism_to_read_groups(
	*,
	parallelism: StageParallelism,
	targets: list[Any],
) -> StageParallelism:
	read_cap = getattr(parallelism, "max_simultaneous_well_reads_per_dataset", None)
	if read_cap is None:
		return parallelism
	if not targets:
		return parallelism

	read_group_count = count_target_read_groups(list(targets))
	if read_group_count <= 0:
		return parallelism

	max_effective_well_workers = min(len(targets), int(read_group_count) * int(read_cap))
	effective_well_workers = min(int(parallelism.well_workers), max(1, int(max_effective_well_workers)))
	if effective_well_workers == int(parallelism.well_workers):
		return parallelism

	if str(getattr(parallelism, "unit_workers_source", "derived")) != "derived":
		unit_workers = max(1, int(parallelism.unit_workers))
	elif bool(getattr(parallelism, "divide_stage_workers_by_wells", True)) and int(effective_well_workers) > 1:
		unit_workers = max(1, int(parallelism.max_stage_workers) // int(effective_well_workers))
	else:
		unit_workers = max(1, int(parallelism.max_stage_workers))

	return replace(
		parallelism,
		well_workers=int(effective_well_workers),
		unit_workers=int(unit_workers),
	)
