from __future__ import annotations

from dataclasses import dataclass, replace
import logging
from pathlib import Path
from typing import Any

from axon_recon.pipeline.scratch_layout import resolve_optional_path, resolve_scratch_layout
from axon_recon.runtime_config import RuntimeConfig

from .execution.context import ExecutionTarget, StageParallelism
from .resources import (
	SOURCE_H5_PATH_KEYED_RESOURCE,
	get_active_resource_profile,
	get_keyed_resource_limit_max_concurrent,
	get_max_phase_resource_demands,
	parse_resources_config,
)
from .stages.preprocess.core.copy_src_to_scratch import resolve_copy_src_to_scratch_input_path


LOGGER = logging.getLogger("axon_recon.pipeline.config")


# Process-wide --target-wells override. Set once at CLI entry; honored by
# select_execution_targets when no explicit `target_wells=` is passed in.
# This avoids threading a `target_wells_override` parameter through ~80 runner
# helpers the same way `target_datasets_override` already is — the well filter
# can apply at the leaf (`select_execution_targets`) without touching every
# intermediate.
_TARGET_WELLS_OVERRIDE: list[str] | None = None


def set_target_wells_override(wells: list[str] | None) -> None:
	"""Set the process-wide well-id allowlist for execution-target selection.

	Pass an empty list or `None` to clear. Normally called once by the CLI
	main() based on `--target-wells`. Reset to `None` at process exit by
	whoever sets it.
	"""
	global _TARGET_WELLS_OVERRIDE
	if wells is None:
		_TARGET_WELLS_OVERRIDE = None
		return
	_TARGET_WELLS_OVERRIDE = [str(w).strip() for w in wells if str(w).strip()]
	if not _TARGET_WELLS_OVERRIDE:
		_TARGET_WELLS_OVERRIDE = None


def get_target_wells_override() -> list[str] | None:
	"""Return a copy of the current process-wide well-id allowlist, or None."""
	return list(_TARGET_WELLS_OVERRIDE) if _TARGET_WELLS_OVERRIDE else None


# Process-wide --targets override. When set, this {dataset_idx: {well_id, ...}}
# map takes precedence over _TARGET_WELLS_OVERRIDE and the target_datasets arg:
# only the listed (dataset, well) pairs run.
_TARGET_PAIRS_OVERRIDE: dict[int, set[str]] | None = None


def set_target_pairs_override(pairs: dict[int, list[str]] | None) -> None:
	"""Set the process-wide (dataset_idx, well_id) pair allowlist for execution-target selection.

	Pass `None` to clear. Normally called once by the CLI main() based on `--targets`.
	Takes precedence over --target-datasets and --target-wells when both are present.
	"""
	global _TARGET_PAIRS_OVERRIDE
	if pairs is None:
		_TARGET_PAIRS_OVERRIDE = None
		return
	normalized: dict[int, set[str]] = {}
	for dataset_idx, wells in pairs.items():
		ds = int(dataset_idx)
		well_set = {str(w).strip() for w in wells if str(w).strip()}
		if well_set:
			normalized[ds] = well_set
	_TARGET_PAIRS_OVERRIDE = normalized or None


def get_target_pairs_override() -> dict[int, set[str]] | None:
	"""Return a copy of the current process-wide pair allowlist, or None."""
	if _TARGET_PAIRS_OVERRIDE is None:
		return None
	return {ds: set(wells) for ds, wells in _TARGET_PAIRS_OVERRIDE.items()}


# Same pattern as _TARGET_WELLS_OVERRIDE: a process-wide toggle set by the CLI
# (`--no-plot`) that downstream phases can consult to decide whether to skip
# their plot/report work. `None` means "no override — honor the YAML setting".
_NO_PLOT_OVERRIDE: bool | None = None


def set_no_plot_override(enabled: bool | None) -> None:
	"""Set the process-wide --no-plot override. Pass None to clear."""
	global _NO_PLOT_OVERRIDE
	_NO_PLOT_OVERRIDE = bool(enabled) if enabled is not None else None


def get_no_plot_override() -> bool | None:
	"""Return the current --no-plot override state, or None if unset."""
	return _NO_PLOT_OVERRIDE


def resolve_plots_enabled(yaml_plots_enabled: bool | None, *, default: bool = True) -> bool:
	"""Combine the YAML `plots_enabled` setting with the CLI `--no-plot` override.

	Rules:
	  - `--no-plot` (override == True) always wins → returns False.
	  - YAML value, when explicitly set, wins over the default.
	  - Falls back to `default` when both are unset.
	"""
	if _NO_PLOT_OVERRIDE is True:
		return False
	if yaml_plots_enabled is None:
		return bool(default)
	return bool(yaml_plots_enabled)


# Process-wide --profile / --task-profile override. Same pattern as
# _TARGET_WELLS_OVERRIDE: set once at CLI entry, honored by load_pipeline_runtime_bundle
# (stashed onto the bundle) and parse_resources_config_for_bundle. This avoids
# threading active_profile_override through ~30 run_*_from_runtime entry points.
_ACTIVE_PROFILE_OVERRIDE: str | None = None


def set_active_profile_override(profile_name: str | None) -> None:
	"""Set the process-wide resources.active_profile override.

	When set, this name replaces resources.active_profile during parse_resources_config_for_bundle
	for every stage that operates on the bundle. Pass None / empty string to clear.
	Normally called once by the CLI based on `--profile` / `--task-profile`.
	"""
	global _ACTIVE_PROFILE_OVERRIDE
	if profile_name is None:
		_ACTIVE_PROFILE_OVERRIDE = None
		return
	token = str(profile_name).strip()
	_ACTIVE_PROFILE_OVERRIDE = token or None


def get_active_profile_override() -> str | None:
	"""Return the current process-wide active-profile override, or None."""
	return _ACTIVE_PROFILE_OVERRIDE


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
	# resources.active_profile override captured from --profile / --task-profile CLI flag.
	# Frozen onto the bundle at load time so every downstream parse_resources_config call
	# can re-apply the same override consistently (resource gate, phase budgets,
	# task allocation plan).
	active_profile_override: str | None = None


def parse_resources_config_for_bundle(bundle: "PipelineRuntimeBundle"):
	"""Parse the runtime resources config and apply the bundle's active_profile_override.

	Every site in runner.py that needs a fresh ResourcesConfig should go through this
	helper instead of calling parse_resources_config directly. That ensures the CLI
	--profile / --task-profile flag is honored uniformly — including by
	_build_stage_resource_budget_manager which historically ignored it (root cause
	of the spikesort_full gate-deadlock on gpu_sort_slots=0).
	"""
	resources_config = parse_resources_config(runtime_config=bundle.runtime_config, logger=LOGGER)
	override = bundle.active_profile_override
	if override is None:
		return resources_config
	profile_name = str(override).strip()
	if not profile_name:
		return resources_config
	if profile_name not in resources_config.profiles:
		raise ValueError(
			f"--profile / --task-profile references an undefined profile: {profile_name!r}. "
			f"Known profiles: {sorted(resources_config.profiles.keys())}"
		)
	return replace(resources_config, active_profile=profile_name)


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


def load_pipeline_runtime_bundle(
	*,
	config_path: str,
	active_profile_override: str | None = None,
) -> PipelineRuntimeBundle:
	runtime_config_path = Path(config_path).expanduser().resolve()
	runtime_cfg = RuntimeConfig.load(runtime_config_path)

	data_cfg_path = _resolve_data_config_path(runtime_config_path, runtime_cfg.get("data", None))
	data_cfg = RuntimeConfig.load(data_cfg_path)

	# Fall back to the process-wide CLI-set override when the caller does not pass one
	# explicitly. The CLI sets _ACTIVE_PROFILE_OVERRIDE once per stage handler, so
	# entry points like run_<stage>_from_runtime don't need to thread the parameter
	# through their signatures.
	if active_profile_override is None:
		active_profile_override = get_active_profile_override()
	override_token = None if active_profile_override is None else str(active_profile_override).strip() or None

	return PipelineRuntimeBundle(
		runtime_config_path=runtime_config_path,
		data_config_path=data_cfg_path,
		runtime_config=runtime_cfg,
		data_config=data_cfg,
		active_profile_override=override_token,
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
	target_datasets: list[int] | None = None,
	limit_wells: int | None = None,
	limit_wells_per_dataset: int | None = None,
	target_wells: list[str] | None = None,
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
	# --targets pair override takes precedence over --target-datasets / --target-wells.
	# Synthesize target_datasets from the pair-override keys; the per-dataset well
	# filter is applied below via `pair_override`.
	pair_override = get_target_pairs_override()
	if pair_override is not None:
		target_datasets = sorted(pair_override.keys())

	if target_datasets is not None:
		requested_dataset_indices: list[int] = []
		seen_requested_dataset_indices: set[int] = set()
		for raw_index in list(target_datasets):
			dataset_index = int(raw_index)
			if dataset_index < 0 or dataset_index in seen_requested_dataset_indices:
				continue
			seen_requested_dataset_indices.add(dataset_index)
			requested_dataset_indices.append(dataset_index)
		selected_dataset_index_set = set(requested_dataset_indices)
		original_count = len(enabled)
		enabled = [item for item in enabled if int(item[0]) in selected_dataset_index_set]
		matched_dataset_indices = [int(index) for index, _item in enabled]
		if not enabled:
			raise ValueError(
				"No enabled datasets matched target_datasets="
				f"{requested_dataset_indices}. Use 0-based dataset indices from debug.data.yml."
			)
		if len(enabled) < original_count:
			LOGGER.info(
				"Applying execution target dataset selection before scratch materialization: %d -> %d dataset(s) requested_dataset_indices=%s matched_dataset_indices=%s",
				original_count,
				len(enabled),
				requested_dataset_indices,
				matched_dataset_indices,
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

	# pair_override (from --targets) wins over --target-wells; only the listed
	# (dataset, well) pairs run. When unset, fall back to the uniform target_wells_set.
	effective_target_wells = target_wells if target_wells is not None else get_target_wells_override()
	target_wells_set: set[str] | None = None
	if pair_override is None and effective_target_wells is not None:
		normalized = [str(w).strip() for w in effective_target_wells if str(w).strip()]
		if normalized:
			target_wells_set = set(normalized)

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

		pair_wells_for_dataset: set[str] | None = (
			pair_override.get(int(idx)) if pair_override is not None else None
		)

		selected_stream_ids: list[str] = []
		excluded_by_target_wells: list[str] = []
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
			if pair_wells_for_dataset is not None:
				if str(stream_id) not in pair_wells_for_dataset:
					excluded_by_target_wells.append(str(stream_id))
					continue
			elif target_wells_set is not None and str(stream_id) not in target_wells_set:
				excluded_by_target_wells.append(str(stream_id))
				continue
			selected_stream_ids.append(str(stream_id))

		active_well_filter = pair_wells_for_dataset if pair_wells_for_dataset is not None else target_wells_set
		if active_well_filter is not None and (selected_stream_ids or excluded_by_target_wells):
			LOGGER.info(
				"Applying target_wells filter to dataset %s: kept %d of %d enabled well(s) requested=%s kept=%s",
				str(dataset_id),
				len(selected_stream_ids),
				len(selected_stream_ids) + len(excluded_by_target_wells),
				sorted(active_well_filter),
				selected_stream_ids,
			)

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
		if pair_override is not None:
			raise ValueError(
				f"No execution targets were produced after applying --targets pair filter="
				f"{ {ds: sorted(wells) for ds, wells in pair_override.items()} }. Verify that each "
				f"<dataset>:<well> pair maps to an enabled dataset/well in your data config."
			)
		if target_wells_set is not None:
			raise ValueError(
				f"No execution targets were produced after applying target_wells="
				f"{sorted(target_wells_set)}. Verify the well IDs match the wells[*].well_id "
				f"strings in your data config (well IDs are case-sensitive)."
			)
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
	if active_profile is None:
		raise ValueError(
			f"resolve_stage_parallelism: no active resource profile found. "
			"Set resources.active_profile in the runtime YAML."
		)
	stage_workers = max(1, int(active_profile.cpu_cores)) if active_profile.cpu_cores else 1
	resolved_phase_resource_classes = tuple(
		str(resource_class)
		for resource_class in (phase_resource_classes or ())
		if str(resource_class).strip()
	)
	if not resolved_phase_resource_classes:
		raise ValueError(
			f"resolve_stage_parallelism for stage={stage_name!r}: no phase_resource_classes provided. "
			"Caller must supply phase_resource_classes from the enabled phase budgets."
		)
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
	if ram_demand > 0 and active_profile.ram_gb is not None:
		ram_budget = max(0, int(active_profile.ram_gb))
		candidate_well_worker_limits.append(max(1, int(ram_budget // ram_demand)))
	well_workers = min(candidate_well_worker_limits) if candidate_well_worker_limits else 1
	well_workers = max(1, int(well_workers))
	derived_unit_workers = max(1, int(cpu_demand or max(1, stage_workers // well_workers)))

	read_cap_raw = get_keyed_resource_limit_max_concurrent(
		resources_config,
		SOURCE_H5_PATH_KEYED_RESOURCE,
	)
	max_simultaneous_well_reads_per_dataset = _as_optional_positive_int(read_cap_raw)

	return StageParallelism(
		well_workers=well_workers,
		unit_workers=int(derived_unit_workers),
		max_simultaneous_well_reads_per_dataset=max_simultaneous_well_reads_per_dataset,
	)
