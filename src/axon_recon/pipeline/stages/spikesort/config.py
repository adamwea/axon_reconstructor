from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from axon_reconstructor.runtime_config import RuntimeConfig
from axon_reconstructor.pipeline.stg2_spikesorting.runner import SPIKESORTING_OUTPUTS_DIRNAME

from ...execution.context import ExecutionTarget
from .models.inputs import SpikesortInputs


_DEFAULT_OUTPUT_REL_ROOT = SPIKESORTING_OUTPUTS_DIRNAME


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


def _as_optional_int(value: Any) -> int | None:
	if value is None:
		return None
	try:
		return int(value)
	except Exception:
		return None


def _as_optional_float(value: Any) -> float | None:
	if value is None:
		return None
	try:
		return float(value)
	except Exception:
		return None


def _as_optional_str(value: Any) -> str | None:
	if value is None:
		return None
	text = str(value).strip()
	return text if text else None


def _as_optional_dict(value: Any) -> dict[str, Any] | None:
	if value is None:
		return None
	if isinstance(value, dict):
		return dict(value)
	return None


def _normalize_output_rel_root(raw: Any) -> str:
	text = str(raw or _DEFAULT_OUTPUT_REL_ROOT).strip()
	if not text:
		return _DEFAULT_OUTPUT_REL_ROOT
	text = text.lstrip("/")
	if text == "spikesort_outputs":
		return SPIKESORTING_OUTPUTS_DIRNAME
	return text or _DEFAULT_OUTPUT_REL_ROOT


def _resolve_data_config_path(runtime_config_path: Path, data_ref: str | None) -> Path:
	if not data_ref:
		raise ValueError("Runtime config must define data: <path-to-data-config>")
	p = Path(str(data_ref)).expanduser()
	if not p.is_absolute():
		p = (runtime_config_path.parent / p).resolve()
	return p


@dataclass(frozen=True)
class SpikesortStageConfig:
	output_rel_root: str
	sorter: str
	docker_image: str | None
	recording_num: str
	verbose: bool

	ks_batch_duration_s: float | None
	ks_batch_size: int | None
	ks_th_universal: float | None
	ks_th_learned: float | None
	ks_th_single_ch: float | None
	ks_cluster_downsampling: int | None
	ks_nearest_chans: int | None
	ks_max_channel_distance: float | None

	n_jobs: int | None
	chunk_duration: str | None
	cuda_visible_devices: str | None

	run_analyzer: bool
	run_reports: bool
	no_curation: bool
	export_to_phy: bool
	force_rerun_analyzer: bool
	um_kwargs: dict[str, Any] | None
	am_kwargs: dict[str, Any] | None
	option_kwargs: dict[str, Any] | None

	force_restart: bool
	force_replot: bool
	resume_from: str | None


def parse_spikesort_stage_config(
	*,
	runtime_config: RuntimeConfig,
	force_restart_override: bool | None = None,
	force_replot_override: bool | None = None,
) -> SpikesortStageConfig:
	stage_cfg = runtime_config.get("stages.spikesort", {})
	stage_cfg = stage_cfg if isinstance(stage_cfg, dict) else {}
	execution_cfg = stage_cfg.get("execution", {}) if isinstance(stage_cfg.get("execution", {}), dict) else {}
	outputs_cfg = stage_cfg.get("outputs", {}) if isinstance(stage_cfg.get("outputs", {}), dict) else {}

	force_restart = _as_bool(execution_cfg.get("force_restart", False), False)
	force_replot = _as_bool(execution_cfg.get("force_replot", False), False)
	if force_restart_override is not None:
		force_restart = bool(force_restart_override)
	if force_replot_override is not None:
		force_replot = bool(force_replot_override)

	return SpikesortStageConfig(
		output_rel_root=_normalize_output_rel_root(outputs_cfg.get("output_rel_root", _DEFAULT_OUTPUT_REL_ROOT)),
		sorter=str(execution_cfg.get("sorter", "kilosort4") or "kilosort4"),
		docker_image=_as_optional_str(execution_cfg.get("docker_image", None)),
		recording_num=str(execution_cfg.get("recording_num", "rec0000") or "rec0000"),
		verbose=_as_bool(execution_cfg.get("verbose", False), False),
		ks_batch_duration_s=_as_optional_float(execution_cfg.get("ks_batch_duration_s", None)),
		ks_batch_size=_as_optional_int(execution_cfg.get("ks_batch_size", None)),
		ks_th_universal=_as_optional_float(execution_cfg.get("ks_th_universal", None)),
		ks_th_learned=_as_optional_float(execution_cfg.get("ks_th_learned", None)),
		ks_th_single_ch=_as_optional_float(execution_cfg.get("ks_th_single_ch", None)),
		ks_cluster_downsampling=_as_optional_int(execution_cfg.get("ks_cluster_downsampling", None)),
		ks_nearest_chans=_as_optional_int(execution_cfg.get("ks_nearest_chans", None)),
		ks_max_channel_distance=_as_optional_float(execution_cfg.get("ks_max_channel_distance", None)),
		n_jobs=_as_optional_int(execution_cfg.get("n_jobs", None)),
		chunk_duration=_as_optional_str(execution_cfg.get("chunk_duration", None)),
		cuda_visible_devices=_as_optional_str(execution_cfg.get("cuda_visible_devices", None)),
		run_analyzer=_as_bool(execution_cfg.get("run_analyzer", True), True),
		run_reports=_as_bool(execution_cfg.get("run_reports", True), True),
		no_curation=_as_bool(execution_cfg.get("no_curation", False), False),
		export_to_phy=_as_bool(execution_cfg.get("export_to_phy", False), False),
		force_rerun_analyzer=_as_bool(execution_cfg.get("force_rerun_analyzer", False), False),
		um_kwargs=_as_optional_dict(execution_cfg.get("um_kwargs", None)),
		am_kwargs=_as_optional_dict(execution_cfg.get("am_kwargs", None)),
		option_kwargs=_as_optional_dict(execution_cfg.get("option_kwargs", None)),
		force_restart=force_restart,
		force_replot=force_replot,
		resume_from=_as_optional_str(execution_cfg.get("resume_from", None)),
	)


def build_spikesort_inputs_for_target(
	*,
	target: ExecutionTarget,
	stage_config: SpikesortStageConfig,
	unit_workers: int,
) -> SpikesortInputs:
	n_jobs = stage_config.n_jobs
	if n_jobs is None:
		n_jobs = max(1, int(unit_workers))
	return SpikesortInputs(
		h5_path=target.h5_path,
		stream_id=target.stream_id,
		mea_output_root=target.mea_output_root,
		final_output_root=(target.final_output_root or target.mea_output_root),
		output_rel_root=stage_config.output_rel_root,
		sorter=stage_config.sorter,
		docker_image=stage_config.docker_image,
		recording_num=stage_config.recording_num,
		verbose=stage_config.verbose,
		ks_batch_duration_s=stage_config.ks_batch_duration_s,
		ks_batch_size=stage_config.ks_batch_size,
		ks_th_universal=stage_config.ks_th_universal,
		ks_th_learned=stage_config.ks_th_learned,
		ks_th_single_ch=stage_config.ks_th_single_ch,
		ks_cluster_downsampling=stage_config.ks_cluster_downsampling,
		ks_nearest_chans=stage_config.ks_nearest_chans,
		ks_max_channel_distance=stage_config.ks_max_channel_distance,
		n_jobs=n_jobs,
		chunk_duration=stage_config.chunk_duration,
		cuda_visible_devices=stage_config.cuda_visible_devices,
		run_analyzer=stage_config.run_analyzer,
		run_reports=stage_config.run_reports,
		no_curation=stage_config.no_curation,
		export_to_phy=stage_config.export_to_phy,
		force_rerun_analyzer=stage_config.force_rerun_analyzer,
		um_kwargs=stage_config.um_kwargs,
		am_kwargs=stage_config.am_kwargs,
		option_kwargs=stage_config.option_kwargs,
		force_restart=stage_config.force_restart,
		force_replot=stage_config.force_replot,
		resume_from=stage_config.resume_from,
	)


def load_spikesort_inputs_from_runtime(
	*,
	config_path: str,
	force_restart_override: bool | None = None,
	force_replot_override: bool | None = None,
) -> SpikesortInputs:
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

	stage_cfg = parse_spikesort_stage_config(
		runtime_config=runtime_cfg,
		force_restart_override=force_restart_override,
		force_replot_override=force_replot_override,
	)

	n_jobs = stage_cfg.n_jobs if stage_cfg.n_jobs is not None else 1
	return SpikesortInputs(
		h5_path=h5_path,
		stream_id=stream_id,
		mea_output_root=output_root,
		final_output_root=output_root,
		output_rel_root=stage_cfg.output_rel_root,
		sorter=stage_cfg.sorter,
		docker_image=stage_cfg.docker_image,
		recording_num=stage_cfg.recording_num,
		verbose=stage_cfg.verbose,
		ks_batch_duration_s=stage_cfg.ks_batch_duration_s,
		ks_batch_size=stage_cfg.ks_batch_size,
		ks_th_universal=stage_cfg.ks_th_universal,
		ks_th_learned=stage_cfg.ks_th_learned,
		ks_th_single_ch=stage_cfg.ks_th_single_ch,
		ks_cluster_downsampling=stage_cfg.ks_cluster_downsampling,
		ks_nearest_chans=stage_cfg.ks_nearest_chans,
		ks_max_channel_distance=stage_cfg.ks_max_channel_distance,
		n_jobs=n_jobs,
		chunk_duration=stage_cfg.chunk_duration,
		cuda_visible_devices=stage_cfg.cuda_visible_devices,
		run_analyzer=stage_cfg.run_analyzer,
		run_reports=stage_cfg.run_reports,
		no_curation=stage_cfg.no_curation,
		export_to_phy=stage_cfg.export_to_phy,
		force_rerun_analyzer=stage_cfg.force_rerun_analyzer,
		um_kwargs=stage_cfg.um_kwargs,
		am_kwargs=stage_cfg.am_kwargs,
		option_kwargs=stage_cfg.option_kwargs,
		force_restart=stage_cfg.force_restart,
		force_replot=stage_cfg.force_replot,
		resume_from=stage_cfg.resume_from,
	)
