from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import shutil
from typing import Any

from axon_reconstructor.runtime_config import RuntimeConfig

from .execution.context import ExecutionTarget, StageParallelism


LOGGER = logging.getLogger("axon_recon.pipeline.config")


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


def _as_path_list(value: Any) -> list[Path]:
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
		path = Path(token).expanduser().resolve()
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


def _relative_input_tree_path(source_path: Path) -> Path:
	"""Return a stable relative path for scratch input materialization.

	Prefer preserving tree under a `raw_data/` anchor when present.
	"""

	parts = list(source_path.parts)
	lower_parts = [str(p).lower() for p in parts]
	anchor_indexes = [idx for idx, token in enumerate(lower_parts) if token == "raw_data"]
	if anchor_indexes:
		anchor_idx = anchor_indexes[-1]
		rel_parts = parts[anchor_idx + 1 :]
		if rel_parts:
			return Path(*rel_parts)

	if source_path.is_absolute():
		rel_parts = source_path.parts[1:]
		if rel_parts:
			return Path(*rel_parts)

	return Path(source_path.name)


def _copy_file_if_needed(*, src: Path, dst: Path) -> bool:
	dst.parent.mkdir(parents=True, exist_ok=True)
	if dst.exists():
		try:
			src_stat = src.stat()
			dst_stat = dst.stat()
			if int(src_stat.st_size) == int(dst_stat.st_size) and int(src_stat.st_mtime_ns) == int(dst_stat.st_mtime_ns):
				return False
		except Exception:
			pass
	shutil.copy2(src, dst)
	return True


def _render_copy_progress_bar(*, completed: int, total: int, width: int = 24) -> str:
	total_safe = max(1, int(total))
	completed_safe = min(max(0, int(completed)), total_safe)
	bar_width = max(8, int(width))
	filled = int(round((float(completed_safe) / float(total_safe)) * float(bar_width)))
	filled = min(max(0, int(filled)), bar_width)
	return f"[{'#' * filled}{'-' * (bar_width - filled)}]"


def _materialize_dataset_input_in_scratch(*, source_h5_path: Path, scratch_input_root: Path, dataset_id: str) -> Path:
	source_h5_path = source_h5_path.expanduser().resolve()
	scratch_input_root = scratch_input_root.expanduser().resolve()
	if not source_h5_path.exists():
		raise FileNotFoundError(f"Dataset input H5 not found: {source_h5_path}")

	LOGGER.info(
		"Scratch input materialization start dataset_id=%s source_h5=%s scratch_input_root=%s",
		dataset_id,
		source_h5_path,
		scratch_input_root,
	)

	rel_h5 = _relative_input_tree_path(source_h5_path)
	target_h5 = (scratch_input_root / rel_h5).resolve()
	cfg_paths = sorted(source_h5_path.parent.glob("*.cfg"))
	copy_plan: list[tuple[str, Path, Path]] = [("h5", source_h5_path, target_h5)]
	copy_plan.extend(("cfg", cfg_path.resolve(), (target_h5.parent / cfg_path.name)) for cfg_path in cfg_paths)
	total_files = int(len(copy_plan))

	LOGGER.info(
		"Scratch input copy plan dataset_id=%s total_files=%d",
		dataset_id,
		total_files,
	)

	copied_files = 0
	skipped_files = 0
	progress_log_step_pct = 10.0
	next_progress_pct = progress_log_step_pct
	for file_idx, (file_kind, src_path, dst_path) in enumerate(copy_plan, start=1):
		was_copied = _copy_file_if_needed(src=src_path, dst=dst_path)
		if was_copied:
			copied_files += 1
		else:
			skipped_files += 1

		if LOGGER.isEnabledFor(logging.DEBUG):
			LOGGER.debug(
				"Scratch input file action dataset_id=%s action=%s kind=%s src=%s dst=%s",
				dataset_id,
				("copied" if was_copied else "skipped"),
				str(file_kind),
				src_path,
				dst_path,
			)

		progress_pct = (100.0 * float(file_idx)) / float(max(1, total_files))
		should_log_progress = (
			int(copied_files) > 0
			and (
				(float(progress_pct) + 1e-9) >= float(next_progress_pct)
				or int(file_idx) == int(total_files)
			)
		)
		if should_log_progress:
			progress_bar = _render_copy_progress_bar(completed=int(file_idx), total=int(total_files))
			LOGGER.info(
				"Scratch input copy progress dataset_id=%s %s %d/%d (%.1f%%) copied=%d skipped=%d",
				dataset_id,
				progress_bar,
				int(file_idx),
				int(total_files),
				float(progress_pct),
				int(copied_files),
				int(skipped_files),
			)
			while float(next_progress_pct) <= (float(progress_pct) + 1e-9):
				next_progress_pct += float(progress_log_step_pct)

	if int(copied_files) == 0:
		LOGGER.info(
			"Scratch input already materialized in scratch_inputs; skipping copy dataset_id=%s target_h5=%s total_files=%d",
			dataset_id,
			target_h5,
			int(total_files),
		)

	LOGGER.info(
		"Scratch input materialization complete dataset_id=%s target_h5=%s cfg_files=%d copied=%d skipped=%d",
		dataset_id,
		target_h5,
		int(len(cfg_paths)),
		int(copied_files),
		int(skipped_files),
	)

	return target_h5


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


def select_execution_targets(*, bundle: PipelineRuntimeBundle) -> list[ExecutionTarget]:
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

	output_root_raw = bundle.data_config.get("output_root", None)
	if not output_root_raw:
		raise ValueError("Data config missing output_root")
	output_root = Path(str(output_root_raw)).expanduser().resolve()
	default_lookup_roots = _as_path_list(bundle.data_config.get("output_root_2", None))
	scratch_root_raw = bundle.data_config.get("scratch_root", None)
	use_scratch_root = _as_bool(bundle.data_config.get("use_scratch_root", True), True)
	default_scratch_root = (
		Path(str(scratch_root_raw)).expanduser().resolve()
		if use_scratch_root and scratch_root_raw is not None and str(scratch_root_raw).strip() != ""
		else None
	)

	scratch_input_root_raw = bundle.data_config.get("scratch_input_root", None)
	use_scratch_input_root = _as_bool(bundle.data_config.get("use_scratch_input_root", False), False)
	default_scratch_input_root = (
		Path(str(scratch_input_root_raw)).expanduser().resolve()
		if use_scratch_input_root and scratch_input_root_raw is not None and str(scratch_input_root_raw).strip() != ""
		else None
	)

	targets: list[ExecutionTarget] = []
	for idx, item in enabled:
		h5_raw = item.get("raw_data_h5_path", None)
		if not h5_raw:
			continue
		h5_path = Path(str(h5_raw)).expanduser().resolve()
		dataset_id = _dataset_id_for_item(item, index=idx)

		dataset_scratch_root_raw = item.get("scratch_root", None)
		dataset_use_scratch_root = _as_bool(item.get("use_scratch_root", use_scratch_root), use_scratch_root)
		dataset_scratch_root = (
			Path(str(dataset_scratch_root_raw)).expanduser().resolve()
			if dataset_use_scratch_root and dataset_scratch_root_raw is not None and str(dataset_scratch_root_raw).strip() != ""
			else default_scratch_root
		)

		dataset_scratch_input_root_raw = item.get("scratch_input_root", None)
		dataset_use_scratch_input_root = _as_bool(item.get("use_scratch_input_root", use_scratch_input_root), use_scratch_input_root)
		if dataset_use_scratch_input_root:
			dataset_scratch_input_root = (
				Path(str(dataset_scratch_input_root_raw)).expanduser().resolve()
				if dataset_scratch_input_root_raw is not None and str(dataset_scratch_input_root_raw).strip() != ""
				else default_scratch_input_root
			)
		else:
			dataset_scratch_input_root = None

		target_h5_path = (
			_materialize_dataset_input_in_scratch(
				source_h5_path=h5_path,
				scratch_input_root=dataset_scratch_input_root,
				dataset_id=str(dataset_id),
			)
			if dataset_scratch_input_root is not None
			else h5_path
		)
		active_root = dataset_scratch_root if dataset_scratch_root is not None else output_root
		artifact_lookup_roots: list[Path] = []
		for candidate_root in _as_path_list(item.get("output_root_2", None)) + default_lookup_roots:
			if candidate_root == active_root:
				continue
			if candidate_root in artifact_lookup_roots:
				continue
			artifact_lookup_roots.append(candidate_root)

		wells = item.get("wells", [])
		if not isinstance(wells, list) or not wells:
			wells = [{"well_id": "well000"}]

		for well_item in wells:
			stream_id = "well000"
			well_enabled = False
			if isinstance(well_item, dict):
				well_enabled = _as_bool(well_item.get("include_in_runtime", False), False)
				if well_item.get("well_id"):
					stream_id = str(well_item.get("well_id"))
			if not well_enabled:
				continue

			targets.append(
				ExecutionTarget(
					dataset_index=int(idx),
					dataset_id=str(dataset_id),
					h5_path=target_h5_path,
					stream_id=str(stream_id),
					mea_output_root=active_root,
					final_output_root=output_root,
					scratch_output_root=dataset_scratch_root,
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


def resolve_stage_parallelism(*, bundle: PipelineRuntimeBundle, stage_name: str) -> StageParallelism:
	runtime_cfg = bundle.runtime_config
	max_workers = _as_int(runtime_cfg.get("resources.max_workers", 8), 8)
	max_workers = max(1, int(max_workers))

	stage_workers = _as_int(runtime_cfg.get(f"stages.{stage_name}.resources.max_stage_workers", max_workers), max_workers)
	stage_workers = max(1, int(stage_workers))

	well_workers = _as_int(runtime_cfg.get(f"stages.{stage_name}.resources.well_workers", 1), 1)
	well_workers = max(1, int(well_workers))

	# Preserve legacy semantics: derive per-target unit workers from stage workers split across well workers.
	unit_workers = max(1, int(stage_workers // well_workers))

	return StageParallelism(
		max_workers=max_workers,
		max_stage_workers=stage_workers,
		well_workers=well_workers,
		unit_workers=unit_workers,
	)
