from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from axon_reconstructor.runtime_config import RuntimeConfig

from .execution.context import ExecutionTarget, StageParallelism


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


def _resolve_data_config_path(runtime_config_path: Path, data_ref: str | None) -> Path:
	if not data_ref:
		raise ValueError("Runtime config must define data: <path-to-data-config>")
	p = Path(str(data_ref)).expanduser()
	if not p.is_absolute():
		p = (runtime_config_path.parent / p).resolve()
	return p


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
		first = next(((idx, item) for idx, item in enumerate(datasets) if isinstance(item, dict)), None)
		if first is None:
			raise ValueError("No valid dataset entries in data config")
		enabled = [first]

	output_root_raw = bundle.data_config.get("output_root", None)
	if not output_root_raw:
		raise ValueError("Data config missing output_root")
	output_root = Path(str(output_root_raw)).expanduser().resolve()
	scratch_root_raw = bundle.data_config.get("scratch_root", None)
	default_scratch_root = (
		Path(str(scratch_root_raw)).expanduser().resolve()
		if scratch_root_raw is not None and str(scratch_root_raw).strip() != ""
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
		dataset_scratch_root = (
			Path(str(dataset_scratch_root_raw)).expanduser().resolve()
			if dataset_scratch_root_raw is not None and str(dataset_scratch_root_raw).strip() != ""
			else default_scratch_root
		)
		active_root = dataset_scratch_root if dataset_scratch_root is not None else output_root

		wells = item.get("wells", [])
		if not isinstance(wells, list) or not wells:
			wells = [{"well_id": "well000"}]

		for well_item in wells:
			stream_id = "well000"
			if isinstance(well_item, dict) and well_item.get("well_id"):
				stream_id = str(well_item.get("well_id"))

			targets.append(
				ExecutionTarget(
					dataset_index=int(idx),
					dataset_id=str(dataset_id),
					h5_path=h5_path,
					stream_id=str(stream_id),
					mea_output_root=active_root,
					final_output_root=output_root,
					scratch_output_root=dataset_scratch_root,
				)
			)

	if not targets:
		raise ValueError("No execution targets were produced from selected datasets")

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
