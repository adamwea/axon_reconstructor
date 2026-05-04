from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
import math
from pathlib import Path
from typing import Any, Iterable

from axon_recon.runtime_config import RuntimeConfig

from .logging.setup import current_pipeline_logging_config
from .resources import (
	PhaseResourceClassConfig,
	ResourcesConfig,
	get_active_resource_profile,
	get_phase_resource_class_config,
	parse_resources_config,
)


LOGGER = logging.getLogger("axon_recon.pipeline.phase_tuning")


@dataclass(frozen=True)
class PhaseTuningConfig:
	output_relpath: str = "resource_tuning"
	ram_safety_factor: float = 1.5
	cpu_safety_factor: float = 1.25
	min_observations_for_underuse: int = 5
	require_limits_unless_confirmed: bool = True
	write_recommendations: bool = True
	update_runtime_yml: bool = False


def _as_mapping(value: Any) -> dict[str, Any]:
	return dict(value) if isinstance(value, dict) else {}


def _as_bool(value: Any, default: bool) -> bool:
	if value is None:
		return bool(default)
	if isinstance(value, bool):
		return bool(value)
	token = str(value).strip().lower()
	if token in {"1", "true", "yes", "on"}:
		return True
	if token in {"0", "false", "no", "off"}:
		return False
	return bool(default)


def _as_float(value: Any, default: float) -> float:
	try:
		return float(value)
	except Exception:
		return float(default)


def _as_int(value: Any, default: int) -> int:
	try:
		return int(value)
	except Exception:
		return int(default)


def parse_phase_tuning_config(runtime_config: RuntimeConfig) -> PhaseTuningConfig:
	block = _as_mapping(runtime_config.get("resources.tuning", None))
	resources_block = _as_mapping(runtime_config.get("resources", None))
	if not block:
		block = _as_mapping(resources_block.get("tuning", None))
	return PhaseTuningConfig(
		output_relpath=str(block.get("output_relpath", "resource_tuning") or "resource_tuning"),
		ram_safety_factor=max(1.0, _as_float(block.get("ram_safety_factor", 1.5), 1.5)),
		cpu_safety_factor=max(1.0, _as_float(block.get("cpu_safety_factor", 1.25), 1.25)),
		min_observations_for_underuse=max(
			1,
			_as_int(block.get("min_observations_for_underuse", 5), 5),
		),
		require_limits_unless_confirmed=_as_bool(block.get("require_limits_unless_confirmed", True), True),
		write_recommendations=_as_bool(block.get("write_recommendations", True), True),
		update_runtime_yml=_as_bool(block.get("update_runtime_yml", False), False),
	)


def _jsonl_records(path: Path) -> Iterable[dict[str, Any]]:
	if not path.exists():
		return
	for line in path.read_text(encoding="utf-8").splitlines():
		text = line.strip()
		if not text:
			continue
		try:
			payload = json.loads(text)
		except Exception:
			continue
		if isinstance(payload, dict):
			yield payload


def _metric(value: Any) -> float | None:
	if value is None:
		return None
	try:
		return float(value)
	except Exception:
		return None


def _int_metric(value: Any) -> int | None:
	if value is None:
		return None
	try:
		return int(value)
	except Exception:
		return None


def collect_phase_resource_observations_from_jsonl(
	path: str | Path,
	*,
	run_id: str | None = None,
	selected_stages: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
	jsonl_path = Path(path).expanduser()
	selected = {str(stage) for stage in (selected_stages or ()) if str(stage).strip()}
	observations: list[dict[str, Any]] = []
	for record in _jsonl_records(jsonl_path):
		if record.get("event") != "phase_resource_usage":
			continue
		if run_id is not None and str(record.get("run_id", "")) != str(run_id):
			continue
		stage = str(record.get("stage", "") or "")
		if selected and stage not in selected and not any(stage.startswith(f"{item}.") for item in selected):
			continue
		usage = record.get("resource_usage", None)
		if not isinstance(usage, dict):
			continue
		wall_time_s = _metric(usage.get("wall_time_s", None))
		disk_read_gb = _metric(usage.get("disk_read_gb", None))
		disk_write_gb = _metric(usage.get("disk_write_gb", None))
		read_gb_per_s = None
		write_gb_per_s = None
		if wall_time_s is not None and wall_time_s > 0:
			if disk_read_gb is not None:
				read_gb_per_s = max(0.0, float(disk_read_gb) / float(wall_time_s))
			if disk_write_gb is not None:
				write_gb_per_s = max(0.0, float(disk_write_gb) / float(wall_time_s))
		observations.append(
			{
				"run_id": record.get("run_id", None),
				"timestamp": record.get("timestamp", None),
				"stage": stage or None,
				"phase": record.get("phase", None),
				"resource_class": record.get("resource_class", None),
				"dataset": record.get("dataset_id", None),
				"recording": record.get("recording_id", None),
				"well": record.get("well_id", None),
				"source_h5_path": record.get("source_h5_path", None),
				"status": record.get("status", None),
				"exception_type": record.get("exception_type", None),
				"wall_time_s": wall_time_s,
				"process_peak_rss_gb": _metric(usage.get("process_peak_rss_gb", None)),
				"child_peak_rss_gb": _metric(usage.get("child_peak_rss_gb", None)),
				"total_peak_rss_gb": _metric(usage.get("total_peak_rss_gb", None)),
				"cpu_time_user_s": _metric(usage.get("cpu_time_user_s", None)),
				"cpu_time_system_s": _metric(usage.get("cpu_time_system_s", None)),
				"max_threads": _int_metric(usage.get("max_threads", None)),
				"observed_process_max_threads": _int_metric(usage.get("observed_process_max_threads", None)),
				"child_process_count_max": _int_metric(usage.get("child_process_count_max", None)),
				"disk_read_gb": disk_read_gb,
				"disk_write_gb": disk_write_gb,
				"disk_read_gb_per_s": read_gb_per_s,
				"disk_write_gb_per_s": write_gb_per_s,
				"gpu_peak_memory_gb": _metric(usage.get("gpu_peak_memory_gb", None)),
				"gpu_utilization_max_pct": _metric(usage.get("gpu_utilization_max_pct", None)),
			}
		)
	return observations


def _nonnull_float_values(items: Iterable[Any]) -> list[float]:
	values: list[float] = []
	for item in items:
		value = _metric(item)
		if value is not None:
			values.append(float(value))
	return values


def _nonnull_int_values(items: Iterable[Any]) -> list[int]:
	values: list[int] = []
	for item in items:
		value = _int_metric(item)
		if value is not None:
			values.append(int(value))
	return values


def _phase_class(resources: ResourcesConfig, resource_class: str | None) -> PhaseResourceClassConfig | None:
	return get_phase_resource_class_config(resources, resource_class)


def _cpu_parallelism_estimate(observation: dict[str, Any]) -> float | None:
	wall_time_s = _metric(observation.get("wall_time_s", None))
	if wall_time_s is None or wall_time_s <= 0:
		return None
	user_s = _metric(observation.get("cpu_time_user_s", None)) or 0.0
	system_s = _metric(observation.get("cpu_time_system_s", None)) or 0.0
	return max(0.0, float(user_s + system_s) / float(wall_time_s))


def _group_key_for_observation(observation: dict[str, Any]) -> tuple[str, str, str]:
	return (
		str(observation.get("stage", "unknown") or "unknown"),
		str(observation.get("phase", "unknown") or "unknown"),
		str(observation.get("resource_class", "") or ""),
	)


def _timestamp_to_epoch_s(value: Any) -> float | None:
	if value is None:
		return None
	text = str(value).strip()
	if not text:
		return None
	if text.endswith("Z"):
		text = f"{text[:-1]}+00:00"
	try:
		parsed = datetime.fromisoformat(text)
	except Exception:
		return None
	if parsed.tzinfo is None:
		parsed = parsed.replace(tzinfo=timezone.utc)
	return float(parsed.timestamp())


def _max_overlapping_slot_demand(
	*,
	observations: list[dict[str, Any]],
	recommendations_by_group: dict[tuple[str, str, str], dict[str, Any]],
	demand_key: str,
) -> tuple[int | None, int]:
	events: list[tuple[float, int]] = []
	interval_count = 0
	for observation in observations:
		wall_time_s = _metric(observation.get("wall_time_s", None))
		end_s = _timestamp_to_epoch_s(observation.get("timestamp", None))
		if wall_time_s is None or wall_time_s <= 0 or end_s is None:
			continue
		recommendation = recommendations_by_group.get(_group_key_for_observation(observation), {})
		demand = _int_metric(recommendation.get(demand_key, None))
		if demand is None or demand <= 0:
			continue
		start_s = max(0.0, float(end_s) - float(wall_time_s))
		events.append((start_s, int(demand)))
		events.append((float(end_s), -int(demand)))
		interval_count += 1
	if not events:
		return None, interval_count
	events.sort(key=lambda item: (float(item[0]), 0 if int(item[1]) < 0 else 1))
	active_demand = 0
	max_demand = 0
	for _event_time, delta in events:
		active_demand = max(0, int(active_demand) + int(delta))
		max_demand = max(int(max_demand), int(active_demand))
	return int(max_demand), interval_count


def _recommend_for_group(
	*,
	resources: ResourcesConfig,
	tuning_config: PhaseTuningConfig,
	group_key: tuple[str, str, str],
	observations: list[dict[str, Any]],
) -> dict[str, Any]:
	stage, phase, resource_class = group_key
	phase_class = _phase_class(resources, resource_class)
	notes: list[str] = []
	warnings: list[str] = []
	observation_count = len(observations)
	if phase_class is None:
		warnings.append("resource_class is missing or undefined; recommendations are limited")

	current_ram_gb = None if phase_class is None else float(phase_class.ram_gb)
	current_cpu_cores = None if phase_class is None else int(phase_class.cpu_cores)
	current_h5_read_slots = None if phase_class is None else int(phase_class.h5_read_slots)
	current_disk_heavy_slots = None if phase_class is None else int(phase_class.disk_heavy_slots)
	current_gpu_sort_slots = None if phase_class is None else int(phase_class.gpu_sort_slots)
	current_plot_slots = None if phase_class is None else int(phase_class.plot_slots)
	current_analyzer_slots = None if phase_class is None else int(phase_class.analyzer_slots)

	peak_values = _nonnull_float_values(item.get("total_peak_rss_gb", None) for item in observations)
	max_peak_ram = max(peak_values) if peak_values else None
	recommended_ram_gb = current_ram_gb
	if max_peak_ram is not None:
		ram_candidate = int(max(1, math.ceil(float(max_peak_ram) * float(tuning_config.ram_safety_factor))))
		if current_ram_gb is None or current_ram_gb <= 0:
			recommended_ram_gb = ram_candidate
		elif ram_candidate > current_ram_gb:
			recommended_ram_gb = float(ram_candidate)
			warnings.append("observed peak RAM is close to or above the configured class estimate")
		elif ram_candidate < current_ram_gb:
			if observation_count >= int(tuning_config.min_observations_for_underuse):
				recommended_ram_gb = float(max(1, ram_candidate))
				notes.append("class may be overestimated; verify with representative non-debug runs before lowering")
			else:
				notes.append("RAM looked underused, but observation count is too small for a lowering recommendation")
		else:
			notes.append("current RAM estimate matches the observed peak plus safety factor")
	else:
		notes.append("RAM metrics were unavailable")

	planned_threads = _nonnull_int_values(item.get("max_threads", None) for item in observations)
	observed_threads = _nonnull_int_values(item.get("observed_process_max_threads", None) for item in observations)
	cpu_parallelism = _nonnull_float_values(_cpu_parallelism_estimate(item) for item in observations)
	max_planned_threads = max(planned_threads) if planned_threads else None
	max_observed_threads = max(observed_threads) if observed_threads else None
	max_cpu_parallelism = max(cpu_parallelism) if cpu_parallelism else None
	cpu_basis = max(
		[value for value in (max_planned_threads, max_cpu_parallelism) if value is not None],
		default=None,
	)
	recommended_cpu_cores = current_cpu_cores
	if cpu_basis is not None:
		cpu_candidate = int(max(1, math.ceil(float(cpu_basis) * float(tuning_config.cpu_safety_factor))))
		if current_cpu_cores is None or current_cpu_cores <= 0:
			recommended_cpu_cores = cpu_candidate
		elif float(cpu_basis) > float(current_cpu_cores) * 1.10 and cpu_candidate > current_cpu_cores:
			recommended_cpu_cores = cpu_candidate
			warnings.append("observed planned threads or CPU time exceed the configured CPU estimate")
		else:
			notes.append("current CPU estimate covers observed pipeline thread demand")
	else:
		notes.append("CPU parallelism metrics were unavailable")
	if max_observed_threads is not None and current_cpu_cores is not None and max_observed_threads > max(1, current_cpu_cores) * 2:
		notes.append("native/library thread count exceeded configured CPU cores; treat as diagnostic, not direct demand")

	max_read_rate = max(_nonnull_float_values(item.get("disk_read_gb_per_s", None) for item in observations), default=None)
	max_write_rate = max(_nonnull_float_values(item.get("disk_write_gb_per_s", None) for item in observations), default=None)
	max_read_gb = max(_nonnull_float_values(item.get("disk_read_gb", None) for item in observations), default=None)
	max_write_gb = max(_nonnull_float_values(item.get("disk_write_gb", None) for item in observations), default=None)
	recommended_h5_read_slots = current_h5_read_slots
	recommended_disk_heavy_slots = current_disk_heavy_slots
	if max_read_rate is not None and max_read_rate > 0.05 and current_h5_read_slots == 0:
		recommended_h5_read_slots = 1
		notes.append("phase performed measurable reads; consider h5_read_slots=1 when concurrent reads share a source file")
	if (
		((max_read_rate is not None and max_read_rate > 0.25) or (max_write_rate is not None and max_write_rate > 0.1))
		and current_disk_heavy_slots == 0
	):
		recommended_disk_heavy_slots = 1
		notes.append("phase showed measurable disk throughput; consider disk_heavy_slots=1 before increasing phase concurrency")
	elif max_read_rate is not None or max_write_rate is not None:
		notes.append("disk read/write rates are observed phase throughput, not a destructive disk benchmark")

	return {
		"stage": stage,
		"phase": phase,
		"resource_class": resource_class or None,
		"observations": observation_count,
		"max_total_peak_rss_gb": max_peak_ram,
		"max_pipeline_threads": max_planned_threads,
		"max_observed_process_threads": max_observed_threads,
		"max_cpu_parallelism_estimate": max_cpu_parallelism,
		"max_disk_read_gb": max_read_gb,
		"max_disk_write_gb": max_write_gb,
		"max_disk_read_gb_per_s": max_read_rate,
		"max_disk_write_gb_per_s": max_write_rate,
		"current_class_ram_gb": current_ram_gb,
		"recommended_class_ram_gb": recommended_ram_gb,
		"current_class_cpu_cores": current_cpu_cores,
		"recommended_class_cpu_cores": recommended_cpu_cores,
		"current_h5_read_slots": current_h5_read_slots,
		"recommended_h5_read_slots": recommended_h5_read_slots,
		"current_disk_heavy_slots": current_disk_heavy_slots,
		"recommended_disk_heavy_slots": recommended_disk_heavy_slots,
		"current_gpu_sort_slots": current_gpu_sort_slots,
		"recommended_gpu_sort_slots": current_gpu_sort_slots,
		"current_plot_slots": current_plot_slots,
		"recommended_plot_slots": current_plot_slots,
		"current_analyzer_slots": current_analyzer_slots,
		"recommended_analyzer_slots": current_analyzer_slots,
		"notes": notes,
		"warnings": warnings,
	}


def _qualified_stage_phase(stage: Any, phase: Any) -> str:
	stage_text = str(stage or "unknown").strip() or "unknown"
	phase_text = str(phase or "unknown").strip() or "unknown"
	if stage_text == phase_text or stage_text.endswith(f".{phase_text}"):
		return stage_text
	return f"{stage_text}.{phase_text}"


def _profile_io_slot_recommendation(
	*,
	resources: ResourcesConfig,
	tuning_config: PhaseTuningConfig,
	observations: list[dict[str, Any]],
	recommendations: list[dict[str, Any]],
) -> dict[str, Any]:
	profile = get_active_resource_profile(resources)
	profile_name = resources.active_profile
	if profile is None:
		return {
			"profile": profile_name,
			"notes": ["active resource profile is undefined; profile slot recommendations are unavailable"],
			"warnings": [],
		}
	recommendations_by_group = {
		(
			str(recommendation.get("stage", "unknown") or "unknown"),
			str(recommendation.get("phase", "unknown") or "unknown"),
			str(recommendation.get("resource_class", "") or ""),
		): recommendation
		for recommendation in recommendations
	}
	notes: list[str] = []
	warnings: list[str] = []
	current_h5_read_slots = max(0, int(profile.h5_read_slots))
	current_disk_heavy_slots = max(0, int(profile.disk_heavy_slots))
	current_h5_demand, h5_interval_count = _max_overlapping_slot_demand(
		observations=observations,
		recommendations_by_group=recommendations_by_group,
		demand_key="current_h5_read_slots",
	)
	recommended_h5_demand, recommended_h5_interval_count = _max_overlapping_slot_demand(
		observations=observations,
		recommendations_by_group=recommendations_by_group,
		demand_key="recommended_h5_read_slots",
	)
	current_disk_demand, disk_interval_count = _max_overlapping_slot_demand(
		observations=observations,
		recommendations_by_group=recommendations_by_group,
		demand_key="current_disk_heavy_slots",
	)
	recommended_disk_demand, recommended_disk_interval_count = _max_overlapping_slot_demand(
		observations=observations,
		recommendations_by_group=recommendations_by_group,
		demand_key="recommended_disk_heavy_slots",
	)

	def _recommend_profile_slots(
		*,
		dimension: str,
		current_slots: int,
		observed_recommended_demand: int | None,
		interval_count: int,
	) -> int:
		if observed_recommended_demand is None or observed_recommended_demand <= 0:
			notes.append(f"{dimension}: selected observations did not show slot-consuming demand")
			return int(current_slots)
		if int(observed_recommended_demand) > int(current_slots):
			warnings.append(
				f"{dimension}: observed concurrent phase demand would exceed the active profile after class recommendations"
			)
			return int(observed_recommended_demand)
		if int(observed_recommended_demand) < int(current_slots):
			if int(interval_count) >= int(tuning_config.min_observations_for_underuse):
				notes.append(
					f"{dimension}: active profile may be overprovisioned for this representative tuning scope"
				)
				return int(observed_recommended_demand)
			notes.append(
				f"{dimension}: observed demand was lower than the active profile, but observation count is too small for a lowering recommendation"
			)
			return int(current_slots)
		notes.append(f"{dimension}: active profile matches observed concurrent phase demand for this tuning scope")
		return int(current_slots)

	recommended_profile_h5_read_slots = _recommend_profile_slots(
		dimension="h5_read_slots",
		current_slots=current_h5_read_slots,
		observed_recommended_demand=recommended_h5_demand,
		interval_count=recommended_h5_interval_count,
	)
	recommended_profile_disk_heavy_slots = _recommend_profile_slots(
		dimension="disk_heavy_slots",
		current_slots=current_disk_heavy_slots,
		observed_recommended_demand=recommended_disk_demand,
		interval_count=recommended_disk_interval_count,
	)
	if observations:
		notes.append("profile slot recommendations use observed phase overlap in the selected tuning scope")
	return {
		"profile": profile_name,
		"current_h5_read_slots": current_h5_read_slots,
		"recommended_h5_read_slots": recommended_profile_h5_read_slots,
		"current_disk_heavy_slots": current_disk_heavy_slots,
		"recommended_disk_heavy_slots": recommended_profile_disk_heavy_slots,
		"max_current_h5_read_slot_demand": current_h5_demand,
		"max_recommended_h5_read_slot_demand": recommended_h5_demand,
		"h5_read_interval_observations": h5_interval_count,
		"recommended_h5_read_interval_observations": recommended_h5_interval_count,
		"max_current_disk_heavy_slot_demand": current_disk_demand,
		"max_recommended_disk_heavy_slot_demand": recommended_disk_demand,
		"disk_heavy_interval_observations": disk_interval_count,
		"recommended_disk_heavy_interval_observations": recommended_disk_interval_count,
		"notes": notes,
		"warnings": warnings,
	}


def build_phase_tuning_summary(
	*,
	resources: ResourcesConfig,
	tuning_config: PhaseTuningConfig,
	observations: list[dict[str, Any]],
	selected_stages: Iterable[str],
	run_id: str | None,
) -> dict[str, Any]:
	groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
	for observation in observations:
		key = (
			str(observation.get("stage", "unknown") or "unknown"),
			str(observation.get("phase", "unknown") or "unknown"),
			str(observation.get("resource_class", "") or ""),
		)
		groups.setdefault(key, []).append(observation)
	recommendations = [
		_recommend_for_group(
			resources=resources,
			tuning_config=tuning_config,
			group_key=key,
			observations=group_observations,
		)
		for key, group_observations in sorted(groups.items())
	]
	profile = get_active_resource_profile(resources)
	active_profile_recommendation = _profile_io_slot_recommendation(
		resources=resources,
		tuning_config=tuning_config,
		observations=observations,
		recommendations=recommendations,
	)
	return {
		"run_id": run_id,
		"selected_stages": list(selected_stages),
		"active_profile": resources.active_profile,
		"active_profile_capacity": None if profile is None else profile.__dict__,
		"active_profile_recommendation": active_profile_recommendation,
		"observation_count": len(observations),
		"recommendation_count": len(recommendations),
		"update_runtime_yml": False,
		"advisory_only": True,
		"recommendations": recommendations,
	}


def format_phase_tuning_report(summary: dict[str, Any]) -> str:
	lines: list[str] = [
		"# Resource Tuning Report",
		"",
		f"Run ID: {summary.get('run_id') or 'unknown'}",
		f"Active profile: {summary.get('active_profile') or 'none'}",
		f"Observations: {summary.get('observation_count', 0)}",
		"Advisory only: runtime YAML was not modified.",
		"",
		"## Active Profile IO Slots",
	]
	profile_recommendation = summary.get("active_profile_recommendation", {})
	if isinstance(profile_recommendation, dict):
		lines.extend(
			[
				f"- profile: {profile_recommendation.get('profile') or 'none'}",
				f"- h5_read_slots: {profile_recommendation.get('current_h5_read_slots')} -> {profile_recommendation.get('recommended_h5_read_slots')}",
				f"- disk_heavy_slots: {profile_recommendation.get('current_disk_heavy_slots')} -> {profile_recommendation.get('recommended_disk_heavy_slots')}",
				f"- max_current_h5_read_slot_demand: {profile_recommendation.get('max_current_h5_read_slot_demand')}",
				f"- max_recommended_h5_read_slot_demand: {profile_recommendation.get('max_recommended_h5_read_slot_demand')}",
				f"- h5_read_interval_observations: {profile_recommendation.get('recommended_h5_read_interval_observations')}",
				f"- max_current_disk_heavy_slot_demand: {profile_recommendation.get('max_current_disk_heavy_slot_demand')}",
				f"- max_recommended_disk_heavy_slot_demand: {profile_recommendation.get('max_recommended_disk_heavy_slot_demand')}",
				f"- disk_heavy_interval_observations: {profile_recommendation.get('recommended_disk_heavy_interval_observations')}",
			]
		)
		for warning in profile_recommendation.get("warnings", []) or []:
			lines.append(f"- warning: {warning}")
		for note in profile_recommendation.get("notes", []) or []:
			lines.append(f"- note: {note}")
	lines.extend(
		[
			"",
			"## Recommendations",
		]
	)
	for recommendation in summary.get("recommendations", []) or []:
		qualified_name = _qualified_stage_phase(recommendation.get("stage"), recommendation.get("phase"))
		lines.extend(
			[
				"",
				f"### {qualified_name}",
				f"- resource_class: {recommendation.get('resource_class') or 'none'}",
				f"- observations: {recommendation.get('observations', 0)}",
				f"- ram_gb: {recommendation.get('current_class_ram_gb')} -> {recommendation.get('recommended_class_ram_gb')}",
				f"- cpu_cores: {recommendation.get('current_class_cpu_cores')} -> {recommendation.get('recommended_class_cpu_cores')}",
				f"- h5_read_slots: {recommendation.get('current_h5_read_slots')} -> {recommendation.get('recommended_h5_read_slots')}",
				f"- disk_heavy_slots: {recommendation.get('current_disk_heavy_slots')} -> {recommendation.get('recommended_disk_heavy_slots')}",
				f"- max_total_peak_rss_gb: {recommendation.get('max_total_peak_rss_gb')}",
				f"- max_disk_read_gb_per_s: {recommendation.get('max_disk_read_gb_per_s')}",
				f"- max_disk_write_gb_per_s: {recommendation.get('max_disk_write_gb_per_s')}",
			]
		)
		for warning in recommendation.get("warnings", []) or []:
			lines.append(f"- warning: {warning}")
		for note in recommendation.get("notes", []) or []:
			lines.append(f"- note: {note}")
	lines.append("")
	return "\n".join(lines)


def write_phase_tuning_artifacts(
	*,
	run_root: Path,
	tuning_config: PhaseTuningConfig,
	observations: list[dict[str, Any]],
	summary: dict[str, Any],
) -> dict[str, Path]:
	output_dir = (run_root / tuning_config.output_relpath).resolve()
	output_dir.mkdir(parents=True, exist_ok=True)
	observations_path = output_dir / "resource_usage_observations.jsonl"
	summary_path = output_dir / "resource_tuning_summary.json"
	report_path = output_dir / "resource_tuning_report.md"
	observations_path.write_text(
		"".join(json.dumps(item, sort_keys=True, default=str) + "\n" for item in observations),
		encoding="utf-8",
	)
	summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
	report_path.write_text(format_phase_tuning_report(summary), encoding="utf-8")
	return {
		"observations": observations_path,
		"summary": summary_path,
		"report": report_path,
	}


def emit_phase_tuning_recommendations(
	*,
	config_path: str | Path,
	selected_stages: Iterable[str],
) -> dict[str, Any]:
	logging_config = current_pipeline_logging_config()
	if logging_config is None:
		raise RuntimeError("phase tuning requires pipeline logging to be configured")
	for handler in logging.getLogger().handlers:
		try:
			handler.flush()
		except Exception:
			continue
	runtime_config = RuntimeConfig.load(Path(config_path).expanduser().resolve())
	tuning_config = parse_phase_tuning_config(runtime_config)
	resources = parse_resources_config(runtime_config=runtime_config, logger=LOGGER)
	structured_path = logging_config.structured.path
	if structured_path is None:
		observations: list[dict[str, Any]] = []
	else:
		observations = collect_phase_resource_observations_from_jsonl(
			structured_path,
			run_id=logging_config.run_id,
			selected_stages=selected_stages,
		)
	summary = build_phase_tuning_summary(
		resources=resources,
		tuning_config=tuning_config,
		observations=observations,
		selected_stages=selected_stages,
		run_id=logging_config.run_id,
	)
	paths = write_phase_tuning_artifacts(
		run_root=logging_config.run_root,
		tuning_config=tuning_config,
		observations=observations,
		summary=summary,
	)
	for recommendation in summary.get("recommendations", []) or []:
		qualified_name = _qualified_stage_phase(recommendation.get("stage"), recommendation.get("phase"))
		LOGGER.info(
			"Resource tuning recommendation: %s resource_class=%s observations=%s ram_gb=%s->%s cpu_cores=%s->%s h5_read_slots=%s->%s disk_heavy_slots=%s->%s",
			qualified_name,
			recommendation.get("resource_class") or "none",
			recommendation.get("observations", 0),
			recommendation.get("current_class_ram_gb"),
			recommendation.get("recommended_class_ram_gb"),
			recommendation.get("current_class_cpu_cores"),
			recommendation.get("recommended_class_cpu_cores"),
			recommendation.get("current_h5_read_slots"),
			recommendation.get("recommended_h5_read_slots"),
			recommendation.get("current_disk_heavy_slots"),
			recommendation.get("recommended_disk_heavy_slots"),
			extra={"event": "phase_tuning_recommendation"},
		)
	profile_recommendation = summary.get("active_profile_recommendation", {})
	if isinstance(profile_recommendation, dict):
		LOGGER.info(
			"Resource profile tuning recommendation: profile=%s h5_read_slots=%s->%s disk_heavy_slots=%s->%s",
			profile_recommendation.get("profile") or "none",
			profile_recommendation.get("current_h5_read_slots"),
			profile_recommendation.get("recommended_h5_read_slots"),
			profile_recommendation.get("current_disk_heavy_slots"),
			profile_recommendation.get("recommended_disk_heavy_slots"),
			extra={"event": "phase_tuning_profile_recommendation"},
		)
	LOGGER.info(
		"Finished resource tuning run observations_written=%d summary_path=%s recommendations_path=%s",
		len(observations),
		str(paths["summary"]),
		str(paths["report"]),
		extra={"event": "phase_tuning_completed"},
	)
	return {"summary": summary, "paths": {key: str(value) for key, value in paths.items()}}