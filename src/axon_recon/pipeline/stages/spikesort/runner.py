from __future__ import annotations

import csv
import gc
import importlib
import inspect
import itertools
import json
import logging
import math
import shutil
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Callable

from axon_recon.pipeline.output_paths import compute_mea_analysis_output_dir
from axon_recon.pipeline.stages.spikesort.legacy_runner import (
	SpikeSortingInputs as LegacySpikeSortingInputs,
	run_spikesorting_stage as run_legacy_spikesorting_stage,
)

from .models.inputs import SpikesortInputs
from .models.results import SpikesortBombcellResult, SpikesortMergeResult, SpikesortResult


LOGGER = logging.getLogger("axon_recon.spikesort")


def _log_phase_step_start(
	message: str,
	*,
	stream_id: Any | None = None,
	well_out_dir: Path | None = None,
	**context: Any,
) -> None:
	context_parts: list[str] = []
	if stream_id is not None:
		context_parts.append(f"stream={stream_id}")
	elif well_out_dir is not None:
		try:
			well_label = Path(well_out_dir).name
		except Exception:
			well_label = str(well_out_dir)
		context_parts.append(f"well={well_label or well_out_dir}")
	for key, value in context.items():
		if value is None:
			continue
		if isinstance(value, Path):
			rendered = str(value)
		else:
			rendered = str(value)
		rendered = rendered.strip()
		if not rendered:
			continue
		context_parts.append(f"{key}={rendered}")
	if context_parts:
		LOGGER.info("%s [%s]", str(message), ", ".join(context_parts))
	else:
		LOGGER.info("%s", str(message))


def _format_memory_mib(raw_kib: float | int | None) -> str | None:
	if raw_kib is None:
		return None
	try:
		value = float(raw_kib) / 1024.0
	except Exception:
		return None
	if not math.isfinite(value) or value < 0.0:
		return None
	return f"{value:.1f}"


def _process_memory_context() -> dict[str, str]:
	if not LOGGER.isEnabledFor(logging.INFO):
		return {}

	context: dict[str, str] = {}
	status_path = Path("/proc/self/status")
	if status_path.exists():
		try:
			for line in status_path.read_text(encoding="utf-8", errors="ignore").splitlines():
				if line.startswith("VmRSS:"):
					parts = line.split()
					if len(parts) >= 2:
						rendered = _format_memory_mib(parts[1])
						if rendered is not None:
							context["rss_mib"] = rendered
				elif line.startswith("VmHWM:"):
					parts = line.split()
					if len(parts) >= 2:
						rendered = _format_memory_mib(parts[1])
						if rendered is not None:
							context["hwm_mib"] = rendered
				elif line.startswith("VmSize:"):
					parts = line.split()
					if len(parts) >= 2:
						rendered = _format_memory_mib(parts[1])
						if rendered is not None:
							context["vmsize_mib"] = rendered
		except Exception:
			pass

	if "hwm_mib" not in context:
		try:
			import resource

			raw_maxrss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
			if sys.platform == "darwin":
				raw_maxrss = raw_maxrss / 1024.0
			rendered = _format_memory_mib(raw_maxrss)
			if rendered is not None:
				context["hwm_mib"] = rendered
		except Exception:
			pass

	return context


def _log_memory_usage(
	message: str,
	*,
	stream_id: Any | None = None,
	well_out_dir: Path | None = None,
	**context: Any,
) -> None:
	merged_context = dict(_process_memory_context())
	merged_context.update(context)
	_log_phase_step_start(
		message,
		stream_id=stream_id,
		well_out_dir=well_out_dir,
		**merged_context,
	)


def _prepare_matplotlib_for_headless_rendering() -> None:
	# Merge reports can run from worker threads; forcing a non-GUI backend avoids
	# GUI backend initialization warnings and occasional shutdown crashes.
	if "matplotlib.pyplot" in sys.modules:
		return
	try:
		import matplotlib  # type: ignore[import-not-found]
		use_backend = getattr(matplotlib, "use", None)
		if callable(use_backend):
			try:
				use_backend("Agg", force=True)
			except TypeError:
				use_backend("Agg")
	except Exception:
		return


def _write_json(path: Path, payload: dict) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _copy_path_to_destination(*, src: Path, dst: Path) -> None:
	src = Path(src).resolve()
	dst = Path(dst).resolve()
	if dst.exists():
		if dst.is_dir():
			shutil.rmtree(dst, ignore_errors=True)
		else:
			dst.unlink(missing_ok=True)
	if src.is_dir():
		shutil.copytree(src, dst)
	else:
		dst.parent.mkdir(parents=True, exist_ok=True)
		shutil.copy2(src, dst)


def _cache_sorting_outputs_before_merge(*, stage_output_root_dir: Path, cache_root_dir: Path) -> dict[str, Any]:
	cache_root_dir = cache_root_dir.resolve()
	if cache_root_dir.exists():
		shutil.rmtree(cache_root_dir, ignore_errors=True)
	cache_root_dir.mkdir(parents=True, exist_ok=True)

	copied_paths: list[str] = []
	missing_sources: list[str] = []
	for rel_name in ("sorter_output", "analyzer_output"):
		src = (stage_output_root_dir / rel_name).resolve()
		dst = (cache_root_dir / rel_name).resolve()
		if not src.exists():
			missing_sources.append(str(src))
			continue
		if src.is_dir():
			shutil.copytree(src, dst)
		else:
			dst.parent.mkdir(parents=True, exist_ok=True)
			shutil.copy2(src, dst)
		copied_paths.append(str(dst))

	summary_json = cache_root_dir / "pre_merge_cache_summary.json"
	_write_json(
		summary_json,
		{
			"status": "ok",
			"stage_output_root_dir": str(stage_output_root_dir),
			"cache_root_dir": str(cache_root_dir),
			"copied_paths": copied_paths,
			"missing_sources": missing_sources,
		},
	)
	return {
		"cache_root_dir": cache_root_dir,
		"summary_json": summary_json,
		"copied_paths": copied_paths,
		"missing_sources": missing_sources,
	}


def _restore_sorting_outputs_from_pre_merge_cache(*, stage_output_root_dir: Path, cache_root_dir: Path) -> dict[str, Any]:
	cache_root_dir = cache_root_dir.resolve()
	restored_paths: list[str] = []
	missing_cache_sources: list[str] = []
	for rel_name in ("sorter_output", "analyzer_output"):
		src = (cache_root_dir / rel_name).resolve()
		dst = (stage_output_root_dir / rel_name).resolve()
		if not src.exists():
			missing_cache_sources.append(str(src))
			continue
		if dst.exists():
			if dst.is_dir():
				shutil.rmtree(dst, ignore_errors=True)
			else:
				dst.unlink(missing_ok=True)
		if src.is_dir():
			shutil.copytree(src, dst)
		else:
			dst.parent.mkdir(parents=True, exist_ok=True)
			shutil.copy2(src, dst)
		restored_paths.append(str(dst))

	return {
		"cache_root_dir": cache_root_dir,
		"restored_paths": restored_paths,
		"missing_cache_sources": missing_cache_sources,
	}


def _cache_canonical_sorter_output_for_merge(*, stage_output_root_dir: Path, cache_root_dir: Path) -> dict[str, Any]:
	cache_root_dir = cache_root_dir.resolve()
	if cache_root_dir.exists():
		shutil.rmtree(cache_root_dir, ignore_errors=True)
	cache_root_dir.mkdir(parents=True, exist_ok=True)

	src = (stage_output_root_dir / "sorter_output").resolve()
	dst = (cache_root_dir / "sorter_output").resolve()
	copied_paths: list[str] = []
	missing_sources: list[str] = []
	if src.exists():
		_copy_path_to_destination(src=src, dst=dst)
		copied_paths.append(str(dst))
	else:
		missing_sources.append(str(src))

	summary_json = cache_root_dir / "working_cache_summary.json"
	_write_json(
		summary_json,
		{
			"status": "ok",
			"stage_output_root_dir": str(stage_output_root_dir),
			"canonical_sorter_output_dir": str(src),
			"working_cache_root_dir": str(cache_root_dir),
			"working_sorter_output_dir": str(dst),
			"copied_paths": copied_paths,
			"missing_sources": missing_sources,
		},
	)
	return {
		"cache_root_dir": cache_root_dir,
		"summary_json": summary_json,
		"copied_paths": copied_paths,
		"missing_sources": missing_sources,
	}


def _publish_working_sorter_output_to_canonical(*, stage_output_root_dir: Path, cache_root_dir: Path) -> dict[str, Any]:
	cache_root_dir = cache_root_dir.resolve()
	src = (cache_root_dir / "sorter_output").resolve()
	dst = (stage_output_root_dir / "sorter_output").resolve()
	restored_paths: list[str] = []
	missing_cache_sources: list[str] = []
	if src.exists():
		_copy_path_to_destination(src=src, dst=dst)
		restored_paths.append(str(dst))
	else:
		missing_cache_sources.append(str(src))
	return {
		"cache_root_dir": cache_root_dir,
		"restored_paths": restored_paths,
		"missing_cache_sources": missing_cache_sources,
	}


def _cleanup_spikesort_outputs_for_force_restart(*, stage_output_root_dir: Path, um_kwargs: dict[str, Any] | None) -> list[str]:
	cleanup_names: set[str] = {
		"sorter_output",
		"analyzer_output",
	}

	if isinstance(um_kwargs, dict):
		output_subdir_name = str(um_kwargs.get("output_subdir_name", "")).strip().lstrip("/")
		throughput_subdir_name = str(um_kwargs.get("throughput_subdir_name", "")).strip().lstrip("/")
		if output_subdir_name:
			cleanup_names.add(output_subdir_name)
		if throughput_subdir_name:
			cleanup_names.add(throughput_subdir_name)

	removed_paths: list[str] = []
	for rel_name in sorted(cleanup_names):
		target = (stage_output_root_dir / rel_name).resolve()
		if not target.exists():
			continue
		shutil.rmtree(target, ignore_errors=True)
		removed_paths.append(str(target))

	return removed_paths


def _resolve_under_well(*, well_out_dir: Path, relpath: str) -> Path:
	candidate = Path(str(relpath).strip()).expanduser()
	if candidate.is_absolute():
		return candidate.resolve()
	return (well_out_dir / str(relpath).lstrip("/")).resolve()


def _resolve_under_spikesort_output_root(*, well_out_dir: Path, output_rel_root: str, relpath: str) -> Path:
	candidate = Path(str(relpath).strip()).expanduser()
	if candidate.is_absolute():
		return candidate.resolve()

	stage_root_rel = str(output_rel_root).strip().lstrip("/") or "spikesort_outputs"
	resolved_stage_root = (well_out_dir / stage_root_rel).resolve()
	rel = str(relpath).strip().lstrip("/")

	# Backward-compat: allow relpaths that still include output_rel_root prefix.
	if rel == stage_root_rel:
		rel = ""
	elif rel.startswith(f"{stage_root_rel}/"):
		rel = rel[len(stage_root_rel) + 1 :]

	if not rel:
		return resolved_stage_root
	return (resolved_stage_root / rel).resolve()


def _as_optional_relpath(raw: Any) -> str | None:
	if raw is None:
		return None
	text = str(raw).strip().lstrip("/")
	if not text or text == ".":
		return None
	return text


def _compose_output_rel_root(*, stage_output_rel_root: str, child_rel_root: str | None) -> str:
	stage_rel = str(stage_output_rel_root).strip().lstrip("/") or "spikesort_outputs"
	child_rel = _as_optional_relpath(child_rel_root)
	if child_rel is None:
		return stage_rel
	if child_rel == stage_rel or child_rel.startswith(f"{stage_rel}/"):
		return child_rel
	return f"{stage_rel}/{child_rel}"


def _normalize_merge_analyzer_density_mode(raw: Any) -> str:
	if isinstance(raw, bool):
		return ("dense" if raw else "auto")
	token = str(raw or "auto").strip().lower()
	if token in {"dense", "full"}:
		return "dense"
	return "auto"


def _merge_analyzer_compute_sparsity_requested(stage_config: Any) -> bool:
	raw = getattr(stage_config, "merge_analyzer_compute_sparsity", None)
	if raw is None:
		return bool(
			_normalize_merge_analyzer_density_mode(
				getattr(stage_config, "merge_analyzer_density_mode", "auto")
			)
			!= "dense"
		)
	if isinstance(raw, bool):
		return bool(raw)
	token = str(raw).strip().lower()
	if token in {"1", "true", "yes", "on"}:
		return True
	if token in {"0", "false", "no", "off"}:
		return False
	return True


def _normalize_merge_template_random_spikes_method(raw: Any) -> str:
	if isinstance(raw, bool):
		return ("all" if raw else "default")
	token = str(raw or "default").strip().lower()
	if token in {"percentage", "percent", "fraction", "proportion"}:
		return "percentage"
	if token in {"all", "full", "every"}:
		return "all"
	return "default"


def _normalize_merge_template_random_spikes_percentage(raw: Any) -> float | None:
	if raw is None:
		return None
	if isinstance(raw, bool):
		return None
	try:
		value = float(raw)
	except Exception:
		return None
	if (not math.isfinite(value)) or value <= 0.0:
		return None
	if value > 1.0:
		if value > 100.0:
			return None
		value = value / 100.0
	return float(value)


def _normalize_merge_analyzer_sparsity_method(raw: Any) -> str:
	token = str(raw or "radius").strip().lower()
	if token in {"best", "best_channel", "best_channels", "num_channels"}:
		return "best_channels"
	if token in {"threshold", "snr"}:
		return "threshold"
	if token in {"by_property", "property", "group"}:
		return "by_property"
	return "radius"


def _normalize_merge_analyzer_peak_sign(raw: Any) -> str:
	token = str(raw or "neg").strip().lower()
	if token in {"pos", "positive"}:
		return "pos"
	if token in {"both", "all"}:
		return "both"
	return "neg"


def _normalize_merge_template_heatmap_magnitude_mode(raw: Any) -> str:
	token = str(raw or "ptp").strip().lower()
	if token in {"peak_to_peak", "ptp"}:
		return "ptp"
	if token in {"abs_peak", "absolute_peak", "extremum"}:
		return "abs_peak"
	if token in {"peak", "positive_peak", "max"}:
		return "peak"
	if token in {"trough", "negative_peak", "neg_peak", "min"}:
		return "trough"
	return "ptp"


def _merge_analyzer_compute_job_kwargs(stage_config: Any) -> dict[str, Any]:
	n_jobs_raw = getattr(stage_config, "merge_analyzer_n_jobs", None)
	if n_jobs_raw is None:
		n_jobs_raw = getattr(stage_config, "n_jobs", None)
	chunk_duration_raw = getattr(stage_config, "merge_analyzer_chunk_duration", None)
	if chunk_duration_raw is None:
		chunk_duration_raw = getattr(stage_config, "chunk_duration", None)

	job_kwargs: dict[str, Any] = {}
	try:
		if n_jobs_raw is not None and int(n_jobs_raw) > 0:
			job_kwargs["n_jobs"] = int(n_jobs_raw)
	except Exception:
		pass
	if chunk_duration_raw is not None:
		chunk_duration = str(chunk_duration_raw).strip()
		if chunk_duration:
			job_kwargs["chunk_duration"] = chunk_duration
	return job_kwargs


def _merge_analyzer_sparsity_settings(stage_config: Any) -> dict[str, Any]:
	method = _normalize_merge_analyzer_sparsity_method(
		getattr(stage_config, "merge_analyzer_sparsity_method", "radius")
	)
	peak_sign = _normalize_merge_analyzer_peak_sign(
		getattr(stage_config, "merge_analyzer_sparsity_peak_sign", "neg")
	)
	try:
		radius_um = float(getattr(stage_config, "merge_analyzer_sparsity_radius_um", 100.0) or 100.0)
	except Exception:
		radius_um = 100.0
	if (not math.isfinite(radius_um)) or radius_um <= 0.0:
		radius_um = 100.0

	try:
		num_channels = int(getattr(stage_config, "merge_analyzer_sparsity_num_channels", 5) or 5)
	except Exception:
		num_channels = 5
	if num_channels <= 0:
		num_channels = 5

	try:
		threshold = float(getattr(stage_config, "merge_analyzer_sparsity_threshold", 5.0) or 5.0)
	except Exception:
		threshold = 5.0
	if (not math.isfinite(threshold)) or threshold <= 0.0:
		threshold = 5.0

	try:
		num_spikes_for_sparsity = int(
			getattr(stage_config, "merge_analyzer_sparsity_num_spikes_for_sparsity", 100) or 100
		)
	except Exception:
		num_spikes_for_sparsity = 100
	if num_spikes_for_sparsity <= 0:
		num_spikes_for_sparsity = 100

	by_property_raw = getattr(stage_config, "merge_analyzer_sparsity_by_property", None)
	by_property = None
	if by_property_raw is not None:
		candidate = str(by_property_raw).strip()
		if candidate:
			by_property = candidate

	return {
		"sparsity_method": str(method),
		"sparsity_radius_um": float(radius_um),
		"sparsity_num_channels": int(num_channels),
		"sparsity_threshold": float(threshold),
		"sparsity_peak_sign": str(peak_sign),
		"sparsity_num_spikes_for_sparsity": int(num_spikes_for_sparsity),
		"sparsity_by_property": by_property,
	}


def _merge_analyzer_random_spikes_settings(stage_config: Any) -> dict[str, Any]:
	random_spikes_method = _normalize_merge_template_random_spikes_method(
		getattr(stage_config, "merge_template_random_spikes_method", "default")
	)
	if (
		random_spikes_method == "default"
		and _normalize_merge_template_random_spikes_percentage(
			getattr(stage_config, "merge_template_random_spikes_percentage", None)
		)
		is not None
	):
		random_spikes_method = "percentage"
	try:
		max_spikes_per_unit_raw = getattr(
			stage_config,
			"merge_template_random_spikes_max_spikes_per_unit",
			None,
		)
		max_spikes_per_unit = (
			int(max_spikes_per_unit_raw)
			if max_spikes_per_unit_raw is not None
			else None
		)
	except Exception:
		max_spikes_per_unit = None
	if max_spikes_per_unit is not None and max_spikes_per_unit <= 0:
		max_spikes_per_unit = None
	if max_spikes_per_unit is None and random_spikes_method != "percentage":
		max_spikes_per_unit = 500

	margin_size_raw = getattr(stage_config, "merge_template_random_spikes_margin_size", None)
	margin_size = None
	try:
		if margin_size_raw is not None:
			margin_size = int(margin_size_raw)
			if margin_size < 0:
				margin_size = None
	except Exception:
		margin_size = None

	seed_raw = getattr(stage_config, "merge_template_random_spikes_seed", None)
	seed = None
	try:
		if seed_raw is not None:
			seed = int(seed_raw)
	except Exception:
		seed = None

	min_spikes_per_unit_raw = getattr(
		stage_config,
		"merge_template_random_spikes_min_spikes_per_unit",
		None,
	)
	min_spikes_per_unit = None
	try:
		if min_spikes_per_unit_raw is not None:
			min_spikes_per_unit = int(min_spikes_per_unit_raw)
			if min_spikes_per_unit <= 0:
				min_spikes_per_unit = None
	except Exception:
		min_spikes_per_unit = None

	log_before_after_spike_counts = bool(
		getattr(
			stage_config,
			"merge_template_random_spikes_log_before_after_spike_counts",
			False,
		)
	)

	percentage = _normalize_merge_template_random_spikes_percentage(
		getattr(stage_config, "merge_template_random_spikes_percentage", None)
	)

	return {
		"template_random_spikes_max_spikes_per_unit": (
			int(max_spikes_per_unit) if max_spikes_per_unit is not None else None
		),
		"template_random_spikes_min_spikes_per_unit": min_spikes_per_unit,
		"template_random_spikes_log_before_after_spike_counts": bool(log_before_after_spike_counts),
		"template_random_spikes_margin_size": margin_size,
		"template_random_spikes_seed": seed,
		"template_random_spikes_percentage": percentage,
	}


def _merge_analyzer_waveform_settings(stage_config: Any) -> dict[str, Any]:
	try:
		ms_before = float(getattr(stage_config, "merge_analyzer_waveforms_ms_before", 1.0) or 1.0)
	except Exception:
		ms_before = 1.0
	if (not math.isfinite(ms_before)) or ms_before < 0.0:
		ms_before = 1.0

	try:
		ms_after = float(getattr(stage_config, "merge_analyzer_waveforms_ms_after", 2.0) or 2.0)
	except Exception:
		ms_after = 2.0
	if (not math.isfinite(ms_after)) or ms_after < 0.0:
		ms_after = 2.0

	dtype_raw = getattr(stage_config, "merge_analyzer_waveforms_dtype", None)
	dtype = None
	if dtype_raw is not None:
		candidate = str(dtype_raw).strip()
		if candidate:
			dtype = candidate

	return {
		"waveforms_ms_before": float(ms_before),
		"waveforms_ms_after": float(ms_after),
		"waveforms_dtype": dtype,
	}


def _merge_analyzer_sparsity_create_kwargs(stage_config: Any) -> dict[str, Any]:
	create_kwargs = _merge_analyzer_compute_job_kwargs(stage_config)
	if _merge_dense_analyzer_requested(stage_config):
		return create_kwargs

	settings = _merge_analyzer_sparsity_settings(stage_config)
	method = str(settings.get("sparsity_method", "radius"))
	create_kwargs["method"] = ("snr" if method == "threshold" else method)
	create_kwargs["peak_sign"] = str(settings.get("sparsity_peak_sign", "neg"))
	create_kwargs["num_spikes_for_sparsity"] = int(
		settings.get("sparsity_num_spikes_for_sparsity", 100)
	)
	if method == "radius":
		create_kwargs["radius_um"] = float(settings.get("sparsity_radius_um", 100.0))
	elif method == "best_channels":
		create_kwargs["num_channels"] = int(settings.get("sparsity_num_channels", 5))
	elif method == "threshold":
		create_kwargs["threshold"] = float(settings.get("sparsity_threshold", 5.0))
	elif method == "by_property":
		by_property = settings.get("sparsity_by_property", None)
		if by_property is not None:
			create_kwargs["by_property"] = str(by_property)
	return create_kwargs


def _merge_analyzer_extension_kwargs(stage_config: Any, extension_name: str) -> dict[str, Any]:
	extension_kwargs = dict(_merge_analyzer_compute_job_kwargs(stage_config))
	random_spikes_settings = _merge_analyzer_random_spikes_settings(stage_config)
	waveform_settings = _merge_analyzer_waveform_settings(stage_config)

	if extension_name == "random_spikes":
		random_spikes_method = _merge_template_random_spikes_method(stage_config)
		log_before_after_spike_counts = bool(
			random_spikes_settings.get("template_random_spikes_log_before_after_spike_counts", False)
		)
		if random_spikes_method == "all":
			extension_kwargs["method"] = "all"
		elif random_spikes_method == "percentage":
			extension_kwargs["method"] = "percentage"
			percentage = random_spikes_settings.get("template_random_spikes_percentage", None)
			if percentage is not None:
				extension_kwargs["percentage"] = float(percentage)
			min_spikes_per_unit = random_spikes_settings.get(
				"template_random_spikes_min_spikes_per_unit",
				None,
			)
			if min_spikes_per_unit is not None:
				extension_kwargs["min_spikes_per_unit"] = int(min_spikes_per_unit)
		max_spikes_per_unit = random_spikes_settings.get(
			"template_random_spikes_max_spikes_per_unit",
			None,
		)
		if max_spikes_per_unit is not None:
			extension_kwargs["max_spikes_per_unit"] = int(max_spikes_per_unit)
		if log_before_after_spike_counts:
			extension_kwargs["log_before_after_spike_counts"] = True
		margin_size = random_spikes_settings.get("template_random_spikes_margin_size", None)
		if margin_size is not None:
			extension_kwargs["margin_size"] = int(margin_size)
		seed = random_spikes_settings.get("template_random_spikes_seed", None)
		if seed is not None:
			extension_kwargs["seed"] = int(seed)
	elif extension_name == "waveforms":
		extension_kwargs["ms_before"] = float(waveform_settings.get("waveforms_ms_before", 1.0))
		extension_kwargs["ms_after"] = float(waveform_settings.get("waveforms_ms_after", 2.0))
		dtype = waveform_settings.get("waveforms_dtype", None)
		if dtype is not None:
			extension_kwargs["dtype"] = str(dtype)
	elif extension_name == "templates":
		extension_kwargs["ms_before"] = float(waveform_settings.get("waveforms_ms_before", 1.0))
		extension_kwargs["ms_after"] = float(waveform_settings.get("waveforms_ms_after", 2.0))

	return extension_kwargs


def _merge_template_heatmap_magnitude_mode(stage_config: Any) -> str:
	return _normalize_merge_template_heatmap_magnitude_mode(
		getattr(stage_config, "merge_reports_template_heatmaps_magnitude_mode", "ptp")
	)


def _requested_merge_analyzer_policy(stage_config: Any) -> dict[str, Any]:
	compute_sparsity = _merge_analyzer_compute_sparsity_requested(stage_config)
	regenerate_on_replot = bool(
		getattr(stage_config, "merge_analyzer_regenerate_on_replot", True)
	)
	random_spikes_method = _normalize_merge_template_random_spikes_method(
		getattr(stage_config, "merge_template_random_spikes_method", "default")
	)
	sparsity_settings = _merge_analyzer_sparsity_settings(stage_config)
	random_spikes_settings = _merge_analyzer_random_spikes_settings(stage_config)
	waveform_settings = _merge_analyzer_waveform_settings(stage_config)
	job_kwargs = _merge_analyzer_compute_job_kwargs(stage_config)
	policy = {
		"requested_compute_sparsity": bool(compute_sparsity),
		"regenerate_on_replot": bool(regenerate_on_replot),
		"template_random_spikes_method": str(random_spikes_method),
		"template_random_spikes_percentage": random_spikes_settings.get(
			"template_random_spikes_percentage",
			None,
		),
	}
	policy.update(sparsity_settings)
	policy.update(random_spikes_settings)
	policy.update(waveform_settings)
	policy["compute_n_jobs"] = job_kwargs.get("n_jobs", None)
	policy["compute_chunk_duration"] = job_kwargs.get("chunk_duration", None)
	return policy


def _merge_dense_analyzer_requested(stage_config: Any) -> bool:
	return not bool(_requested_merge_analyzer_policy(stage_config).get("requested_compute_sparsity", True))


def _merge_template_random_spikes_method(stage_config: Any) -> str:
	return str(_requested_merge_analyzer_policy(stage_config).get("template_random_spikes_method", "default"))


def _analyzer_has_sparsity(analyzer: Any) -> bool:
	try:
		return bool(getattr(analyzer, "sparsity", None) is not None)
	except Exception:
		return False


def _attach_merge_analyzer_policy_info(analyzer: Any, info: dict[str, Any]) -> None:
	if analyzer is None:
		return
	try:
		setattr(analyzer, "_axon_recon_merge_analyzer_policy_info", dict(info))
	except Exception:
		return


def _get_merge_analyzer_policy_info(analyzer: Any) -> dict[str, Any]:
	try:
		info = getattr(analyzer, "_axon_recon_merge_analyzer_policy_info", None)
	except Exception:
		info = None
	return (dict(info) if isinstance(info, dict) else {})


def _copy_stage_config_to_namespace(stage_config: Any) -> SimpleNamespace:
	if stage_config is None:
		return SimpleNamespace()
	try:
		return SimpleNamespace(**dict(vars(stage_config)))
	except Exception:
		return SimpleNamespace()


def _bombcell_analyzer_stage_config(stage_config: Any) -> SimpleNamespace:
	proxy = _copy_stage_config_to_namespace(stage_config)
	compute_sparsity = bool(getattr(stage_config, "bombcell_label_analyzer_compute_sparsity", True))
	setattr(
		proxy,
		"merge_analyzer_regenerate_on_replot",
		bool(getattr(stage_config, "bombcell_label_analyzer_regenerate_on_replot", False)),
	)
	setattr(
		proxy,
		"merge_analyzer_check_if_regen_is_needed",
		bool(getattr(stage_config, "bombcell_label_analyzer_check_if_regen_is_needed", True)),
	)
	setattr(proxy, "merge_analyzer_compute_sparsity", compute_sparsity)
	setattr(proxy, "merge_analyzer_density_mode", ("auto" if compute_sparsity else "dense"))
	setattr(
		proxy,
		"merge_template_random_spikes_method",
		str(getattr(stage_config, "bombcell_label_template_random_spikes_method", "default") or "default"),
	)
	setattr(
		proxy,
		"merge_template_random_spikes_percentage",
		getattr(stage_config, "bombcell_label_template_random_spikes_percentage", None),
	)
	setattr(
		proxy,
		"merge_template_random_spikes_max_spikes_per_unit",
		getattr(stage_config, "bombcell_label_template_random_spikes_max_spikes_per_unit", None),
	)
	setattr(
		proxy,
		"merge_template_random_spikes_min_spikes_per_unit",
		getattr(stage_config, "bombcell_label_template_random_spikes_min_spikes_per_unit", None),
	)
	setattr(
		proxy,
		"merge_template_random_spikes_log_before_after_spike_counts",
		bool(
			getattr(
				stage_config,
				"bombcell_label_template_random_spikes_log_before_after_spike_counts",
				False,
			)
		),
	)
	setattr(
		proxy,
		"merge_template_random_spikes_margin_size",
		getattr(stage_config, "bombcell_label_template_random_spikes_margin_size", None),
	)
	setattr(
		proxy,
		"merge_template_random_spikes_seed",
		getattr(stage_config, "bombcell_label_template_random_spikes_seed", None),
	)
	setattr(proxy, "merge_analyzer_n_jobs", getattr(stage_config, "bombcell_label_analyzer_n_jobs", None))
	setattr(
		proxy,
		"merge_analyzer_chunk_duration",
		getattr(stage_config, "bombcell_label_analyzer_chunk_duration", None),
	)
	setattr(
		proxy,
		"merge_analyzer_sparsity_method",
		str(getattr(stage_config, "bombcell_label_analyzer_sparsity_method", "radius") or "radius"),
	)
	setattr(
		proxy,
		"merge_analyzer_sparsity_radius_um",
		getattr(stage_config, "bombcell_label_analyzer_sparsity_radius_um", None),
	)
	setattr(
		proxy,
		"merge_analyzer_sparsity_num_channels",
		getattr(stage_config, "bombcell_label_analyzer_sparsity_num_channels", None),
	)
	setattr(
		proxy,
		"merge_analyzer_sparsity_threshold",
		getattr(stage_config, "bombcell_label_analyzer_sparsity_threshold", None),
	)
	setattr(
		proxy,
		"merge_analyzer_sparsity_peak_sign",
		str(getattr(stage_config, "bombcell_label_analyzer_sparsity_peak_sign", "neg") or "neg"),
	)
	setattr(
		proxy,
		"merge_analyzer_sparsity_num_spikes_for_sparsity",
		getattr(stage_config, "bombcell_label_analyzer_sparsity_num_spikes_for_sparsity", None),
	)
	setattr(
		proxy,
		"merge_analyzer_sparsity_by_property",
		getattr(stage_config, "bombcell_label_analyzer_sparsity_by_property", None),
	)
	setattr(
		proxy,
		"merge_analyzer_waveforms_ms_before",
		getattr(stage_config, "bombcell_label_analyzer_waveforms_ms_before", None),
	)
	setattr(
		proxy,
		"merge_analyzer_waveforms_ms_after",
		getattr(stage_config, "bombcell_label_analyzer_waveforms_ms_after", None),
	)
	setattr(
		proxy,
		"merge_analyzer_waveforms_dtype",
		getattr(stage_config, "bombcell_label_analyzer_waveforms_dtype", None),
	)
	return proxy


def _install_spikeinterface_random_spikes_percentage_compatibility() -> None:
	try:
		from spikeinterface.core.analyzer_extension_core import ComputeRandomSpikes  # type: ignore[import-not-found]
		from spikeinterface.core.sorting_tools import random_spikes_selection as si_random_spikes_selection  # type: ignore[import-not-found]
		from spikeinterface.core.sorting_tools import spike_vector_to_indices  # type: ignore[import-not-found]
	except Exception:
		return

	if bool(getattr(ComputeRandomSpikes, "_axon_recon_percentage_compat_installed", False)):
		return

	original_set_params = getattr(ComputeRandomSpikes, "_set_params", None)
	original_run = getattr(ComputeRandomSpikes, "_run", None)
	if not callable(original_set_params) or not callable(original_run):
		return

	try:
		supported_param_names = {
			str(name)
			for name in inspect.signature(original_set_params).parameters
			if str(name) != "self"
		}
	except Exception:
		supported_param_names = {"method", "max_spikes_per_unit", "margin_size", "seed"}

	try:
		low_level_supported_param_names = {
			str(name)
			for name in inspect.signature(si_random_spikes_selection).parameters
		}
	except Exception:
		low_level_supported_param_names = {
			"sorting",
			"num_samples",
			"method",
			"max_spikes_per_unit",
			"margin_size",
			"seed",
		}

	low_level_supports_percentage = bool("percentage" in low_level_supported_param_names)

	def _build_random_spikes_unit_candidates(
		*,
		sorting: Any,
		num_samples: Any,
		margin_size: int | None,
	) -> list[dict[str, Any]]:
		import numpy as np

		spikes = sorting.to_spike_vector(concatenated=False)
		cum_sizes = np.cumsum([0] + [segment.size for segment in spikes])
		spike_indices = spike_vector_to_indices(spikes, sorting.unit_ids, absolute_index=False)
		unit_candidates: list[dict[str, Any]] = []
		for unit_id in sorting.unit_ids:
			eligible_indices_by_segment: list[Any] = []
			total_spikes = 0
			for segment_index in range(sorting.get_num_segments()):
				inds_in_seg = spike_indices[segment_index][unit_id]
				total_spikes += int(getattr(inds_in_seg, "size", 0))
				if margin_size is not None:
					if num_samples is None:
						raise ValueError("num_samples must be provided when margin_size is used")
					local_spikes = spikes[segment_index][inds_in_seg]
					mask = (local_spikes["sample_index"] >= margin_size) & (
						local_spikes["sample_index"] < (num_samples[segment_index] - margin_size)
					)
					inds_in_seg = inds_in_seg[mask]
				eligible_indices_by_segment.append(inds_in_seg + cum_sizes[segment_index])
			eligible_indices_by_segment = [
				indices for indices in eligible_indices_by_segment if getattr(indices, "size", 0) > 0
			]
			if eligible_indices_by_segment:
				eligible_indices = np.concatenate(eligible_indices_by_segment)
			else:
				eligible_indices = np.array([], dtype=np.int64)
			unit_candidates.append(
				{
					"unit_id": unit_id,
					"total_spikes": int(total_spikes),
					"eligible_indices": eligible_indices,
					"eligible_spikes": int(getattr(eligible_indices, "size", 0)),
				}
			)
		return unit_candidates

	def _log_random_spikes_selection_counts(
		*,
		sorting: Any,
		num_samples: Any,
		margin_size: int | None,
		selected_indices: Any,
		method: str,
		max_spikes_per_unit: int | None,
		min_spikes_per_unit: int | None,
		percentage: float | None,
	) -> None:
		import numpy as np

		unit_candidates = _build_random_spikes_unit_candidates(
			sorting=sorting,
			num_samples=num_samples,
			margin_size=margin_size,
		)
		selected_array = np.asarray(
			([] if selected_indices is None else selected_indices),
			dtype=np.int64,
		).reshape(-1)
		selected_index_set = set(selected_array.tolist())
		LOGGER.info(
			"Merge analyzer random_spikes counts method=%s units=%d total_spikes=%d eligible_spikes=%d selected_spikes=%d",
			str(method),
			len(unit_candidates),
			sum(int(candidate["total_spikes"]) for candidate in unit_candidates),
			sum(int(candidate["eligible_spikes"]) for candidate in unit_candidates),
			int(selected_array.size),
		)
		for candidate in unit_candidates:
			eligible_indices = np.asarray(candidate["eligible_indices"], dtype=np.int64).reshape(-1)
			selected_count = sum(1 for index in eligible_indices if int(index) in selected_index_set)
			if int(candidate["eligible_spikes"]) <= 0:
				selection_reason = "no_eligible_spikes"
			elif str(method) == "all":
				selection_reason = "all"
			elif str(method) == "percentage":
				_, selection_reason = _resolve_percentage_random_spikes_target(
					total_spikes=int(candidate["total_spikes"]),
					eligible_spikes=int(candidate["eligible_spikes"]),
					percentage=percentage,
					min_spikes_per_unit=min_spikes_per_unit,
					max_spikes_per_unit=max_spikes_per_unit,
				)
			else:
				selection_reason = str(method)
			LOGGER.info(
				"Merge analyzer random_spikes unit=%s total_spikes=%d eligible_spikes=%d selected_spikes=%d selection=%s",
				str(candidate["unit_id"]),
				int(candidate["total_spikes"]),
				int(candidate["eligible_spikes"]),
				int(selected_count),
				selection_reason,
			)

	def _resolve_percentage_random_spikes_target(
		*,
		total_spikes: int,
		eligible_spikes: int,
		percentage: float | None,
		min_spikes_per_unit: int | None,
		max_spikes_per_unit: int | None,
	) -> tuple[int, str]:
		eligible_spikes_int = max(int(eligible_spikes), 0)
		if eligible_spikes_int <= 0:
			return 0, "no_eligible_spikes"
		if percentage is None or not (0.0 < float(percentage) <= 1.0):
			raise ValueError("percentage must be in the interval (0, 1]")
		if min_spikes_per_unit is not None and int(total_spikes) <= int(min_spikes_per_unit):
			return eligible_spikes_int, "all_by_min_spikes_threshold"

		target_count = int(eligible_spikes_int * float(percentage))
		selection_reason = "sampled_percentage"
		if min_spikes_per_unit is not None and target_count < int(min_spikes_per_unit):
			if eligible_spikes_int < int(min_spikes_per_unit):
				target_count = eligible_spikes_int
				selection_reason = "min_spikes_floor_capped_by_eligible"
			else:
				target_count = int(min_spikes_per_unit)
				selection_reason = "min_spikes_floor"
		if max_spikes_per_unit is not None and target_count > int(max_spikes_per_unit):
			target_count = int(max_spikes_per_unit)
			if selection_reason == "sampled_percentage":
				selection_reason = "sampled_percentage_capped_by_max"
			elif selection_reason.startswith("min_spikes_floor"):
				selection_reason = "min_spikes_floor_capped_by_max"
		target_count = min(target_count, eligible_spikes_int)
		return max(int(target_count), 0), selection_reason

	def _random_spikes_selection_percentage_compat(
		*,
		sorting: Any,
		num_samples: Any,
		max_spikes_per_unit: int | None,
		min_spikes_per_unit: int | None,
		margin_size: int | None,
		seed: int | None,
		percentage: float | None,
	) -> Any:
		import numpy as np

		if percentage is None or not (0.0 < float(percentage) <= 1.0):
			raise ValueError("percentage must be in the interval (0, 1]")

		rng = np.random.default_rng(seed=seed)
		unit_candidates = _build_random_spikes_unit_candidates(
			sorting=sorting,
			num_samples=num_samples,
			margin_size=margin_size,
		)

		random_spikes_indices: list[Any] = []
		for candidate in unit_candidates:
			all_unit_indices_concat = candidate["eligible_indices"]
			if int(getattr(all_unit_indices_concat, "size", 0)) <= 0:
				continue
			target_count, _ = _resolve_percentage_random_spikes_target(
				total_spikes=int(candidate["total_spikes"]),
				eligible_spikes=int(candidate["eligible_spikes"]),
				percentage=percentage,
				min_spikes_per_unit=min_spikes_per_unit,
				max_spikes_per_unit=max_spikes_per_unit,
			)
			if target_count >= int(getattr(all_unit_indices_concat, "size", 0)):
				selected_unit_indices = all_unit_indices_concat
			else:
				if target_count <= 0:
					continue
				selected_unit_indices = rng.choice(
					all_unit_indices_concat,
					size=target_count,
					replace=False,
					shuffle=False,
				)
			random_spikes_indices.append(selected_unit_indices)

		if not random_spikes_indices:
			return np.array([], dtype=np.int64)
		return np.sort(np.concatenate(random_spikes_indices))

	def _patched_set_params(
		self: Any,
		method: str = "uniform",
		max_spikes_per_unit: int | None = None,
		min_spikes_per_unit: int | None = None,
		log_before_after_spike_counts: bool | None = None,
		margin_size: int | None = None,
		seed: int | None = None,
		percentage: float | None = None,
		maximum_rate: float | None = None,
	) -> dict[str, Any]:
		original_kwargs: dict[str, Any] = {
			"method": method,
			"margin_size": margin_size,
			"seed": seed,
		}
		if max_spikes_per_unit is not None:
			original_kwargs["max_spikes_per_unit"] = max_spikes_per_unit
		if "percentage" in supported_param_names:
			original_kwargs["percentage"] = percentage
		if "maximum_rate" in supported_param_names:
			original_kwargs["maximum_rate"] = maximum_rate

		params = original_set_params(self, **original_kwargs)
		if not isinstance(params, dict):
			params = {
				"method": method,
				"margin_size": margin_size,
				"seed": seed,
			}
		if max_spikes_per_unit is not None:
			params["max_spikes_per_unit"] = int(max_spikes_per_unit)
		elif method == "percentage":
			params["max_spikes_per_unit"] = None
		if min_spikes_per_unit is not None:
			params["min_spikes_per_unit"] = int(min_spikes_per_unit)
		if log_before_after_spike_counts is not None:
			params["log_before_after_spike_counts"] = bool(log_before_after_spike_counts)
		if percentage is not None:
			params["percentage"] = float(percentage)
		if maximum_rate is not None:
			params["maximum_rate"] = float(maximum_rate)
		return params

	def _patched_run(self: Any, verbose: bool = False) -> None:
		params = dict(getattr(self, "params", {}) or {})
		method = str(params.get("method", "uniform") or "uniform").strip().lower()
		log_before_after_spike_counts = bool(params.get("log_before_after_spike_counts", False))
		max_spikes_per_unit = params.get("max_spikes_per_unit", None)
		try:
			if max_spikes_per_unit is not None:
				max_spikes_per_unit = int(max_spikes_per_unit)
				if max_spikes_per_unit <= 0:
					max_spikes_per_unit = None
		except Exception:
			max_spikes_per_unit = None
		min_spikes_per_unit = params.get("min_spikes_per_unit", None)
		try:
			if min_spikes_per_unit is not None:
				min_spikes_per_unit = int(min_spikes_per_unit)
				if min_spikes_per_unit <= 0:
					min_spikes_per_unit = None
		except Exception:
			min_spikes_per_unit = None
		if method == "percentage" and (
			max_spikes_per_unit is None
			or min_spikes_per_unit is not None
			or not low_level_supports_percentage
		):
			sorting_analyzer = getattr(self, "sorting_analyzer", None)
			if sorting_analyzer is None:
				raise RuntimeError("sorting_analyzer unavailable for percentage random_spikes compatibility")
			sorting = getattr(sorting_analyzer, "sorting", None)
			rec_attributes = getattr(sorting_analyzer, "rec_attributes", {}) or {}
			num_samples = rec_attributes.get("num_samples", None)
			self.data["random_spikes_indices"] = _random_spikes_selection_percentage_compat(
				sorting=sorting,
				num_samples=num_samples,
				max_spikes_per_unit=max_spikes_per_unit,
				min_spikes_per_unit=min_spikes_per_unit,
				margin_size=params.get("margin_size", None),
				seed=params.get("seed", None),
				percentage=params.get("percentage", None),
			)
			if log_before_after_spike_counts:
				_log_random_spikes_selection_counts(
					sorting=sorting,
					num_samples=num_samples,
					margin_size=params.get("margin_size", None),
					selected_indices=self.data.get("random_spikes_indices", None),
					method=method,
					max_spikes_per_unit=max_spikes_per_unit,
					min_spikes_per_unit=min_spikes_per_unit,
					percentage=params.get("percentage", None),
				)
			return
		original_run(self, verbose=verbose)
		if log_before_after_spike_counts:
			sorting_analyzer = getattr(self, "sorting_analyzer", None)
			if sorting_analyzer is None:
				LOGGER.warning(
					"Merge analyzer random_spikes count logging requested but sorting_analyzer is unavailable"
				)
				return
			sorting = getattr(sorting_analyzer, "sorting", None)
			rec_attributes = getattr(sorting_analyzer, "rec_attributes", {}) or {}
			num_samples = rec_attributes.get("num_samples", None)
			_log_random_spikes_selection_counts(
				sorting=sorting,
				num_samples=num_samples,
				margin_size=params.get("margin_size", None),
				selected_indices=self.data.get("random_spikes_indices", None),
				method=method,
				max_spikes_per_unit=max_spikes_per_unit,
				min_spikes_per_unit=min_spikes_per_unit,
				percentage=params.get("percentage", None),
			)

	setattr(ComputeRandomSpikes, "_set_params", _patched_set_params)
	setattr(ComputeRandomSpikes, "_run", _patched_run)
	setattr(ComputeRandomSpikes, "_axon_recon_percentage_compat_installed", True)
	LOGGER.info("Installed SpikeInterface random_spikes percentage compatibility shim")


def _merge_analyzer_policy_info_json_path(analyzer_dir: Path) -> Path:
	return Path(analyzer_dir).resolve() / "axon_recon_merge_analyzer_policy.json"


def _write_merge_analyzer_policy_info_to_dir(*, analyzer_dir: Path, policy_info: dict[str, Any]) -> None:
	policy_path = _merge_analyzer_policy_info_json_path(analyzer_dir)
	_write_json(policy_path, dict(policy_info))


def _read_merge_analyzer_policy_info_from_dir(analyzer_dir: Path) -> dict[str, Any] | None:
	policy_path = _merge_analyzer_policy_info_json_path(analyzer_dir)
	return _read_json_dict(policy_path)


def _merge_analyzer_policy_matches_requested(*, recorded_policy: Any, requested_policy: dict[str, Any]) -> bool:
	if not isinstance(recorded_policy, dict):
		return False
	compatibility_defaults: dict[str, Any] = {
		"template_random_spikes_log_before_after_spike_counts": False,
	}
	for key, value in requested_policy.items():
		recorded_value = recorded_policy.get(key, compatibility_defaults.get(key, None))
		if recorded_value != value:
			return False
	return True


def _path_latest_mtime(path: Path) -> float | None:
	resolved = Path(path).resolve()
	if not resolved.exists():
		return None
	try:
		latest_mtime: float | None = float(resolved.stat().st_mtime)
	except Exception:
		latest_mtime = None
	if not resolved.is_dir():
		return latest_mtime
	for child in resolved.rglob("*"):
		try:
			child_mtime = float(child.stat().st_mtime)
		except Exception:
			continue
		if latest_mtime is None or child_mtime > latest_mtime:
			latest_mtime = child_mtime
	return latest_mtime


def _prepare_replot_workspace_analyzer(
	*,
	si_module: Any,
	well_out_dir: Path,
	sorter_output_dir: Path,
	stage_config: Any,
	analyzer_dir: Path,
	regenerate_on_replot: bool,
	check_if_regen_is_needed: bool,
	fallback_policy: dict[str, Any] | None = None,
) -> tuple[Any, Path, dict[str, Any], bool, str | None]:
	analyzer_dir = Path(analyzer_dir).resolve()
	sorter_output_dir = Path(sorter_output_dir).resolve()
	requested_policy = _requested_merge_analyzer_policy(stage_config)
	load_sorting_analyzer = getattr(si_module, "load_sorting_analyzer", None)
	analyzer_obj: Any | None = None
	regenerated = False
	regen_reason: str | None = None
	should_regenerate = False

	if not analyzer_dir.exists():
		should_regenerate = True
		regen_reason = "analyzer_output_missing"
	elif regenerate_on_replot:
		if not check_if_regen_is_needed:
			should_regenerate = True
			regen_reason = "forced_replot_regeneration"
		else:
			recorded_policy = _read_merge_analyzer_policy_info_from_dir(analyzer_dir)
			if recorded_policy is None and isinstance(fallback_policy, dict):
				recorded_policy = dict(fallback_policy)
			if not _merge_analyzer_policy_matches_requested(
				recorded_policy=recorded_policy,
				requested_policy=requested_policy,
			):
				should_regenerate = True
				regen_reason = "policy_changed"
			else:
				analyzer_mtime = _path_latest_mtime(analyzer_dir)
				sorter_mtime = _path_latest_mtime(sorter_output_dir)
				if analyzer_mtime is None:
					should_regenerate = True
					regen_reason = "analyzer_output_missing"
				elif sorter_mtime is not None and sorter_mtime > (analyzer_mtime + 1e-6):
					should_regenerate = True
					regen_reason = "sorter_output_newer_than_analyzer"
				else:
					if not callable(load_sorting_analyzer):
						raise RuntimeError("load_sorting_analyzer unavailable for replot analyzer reuse")
					try:
						analyzer_obj = _load_spikesort_analyzer_with_recording(
							si_module=si_module,
							analyzer_dir=analyzer_dir,
							well_out_dir=well_out_dir,
							stage_config=stage_config,
						)
					except Exception:
						should_regenerate = True
						regen_reason = "analyzer_load_failed"
					else:
						loaded_analyzer_had_sparsity = _analyzer_has_sparsity(analyzer_obj)
						if (not bool(requested_policy.get("requested_compute_sparsity", True))) and loaded_analyzer_had_sparsity:
							analyzer_obj = None
							should_regenerate = True
							regen_reason = "loaded_sparse_analyzer"

	if should_regenerate:
		analyzer_obj, analyzer_dir = _recompute_sorting_analyzer_to_dir(
			si_module=si_module,
			well_out_dir=well_out_dir,
			sorter_output_dir=sorter_output_dir,
			stage_config=stage_config,
			analyzer_dir=analyzer_dir,
		)
		regenerated = True
	else:
		if analyzer_obj is None:
			if not callable(load_sorting_analyzer):
				raise RuntimeError("load_sorting_analyzer unavailable for replot analyzer reuse")
			analyzer_obj = _load_spikesort_analyzer_with_recording(
				si_module=si_module,
				analyzer_dir=analyzer_dir,
				well_out_dir=well_out_dir,
				stage_config=stage_config,
			)

	loaded_analyzer_had_sparsity = _analyzer_has_sparsity(analyzer_obj)
	policy_info = _describe_merge_analyzer_policy_info(
		analyzer=analyzer_obj,
		stage_config=stage_config,
		loaded_analyzer_had_sparsity=loaded_analyzer_had_sparsity,
		rebuild_reason=(regen_reason if regenerated else None),
		reused_cached_analyzer=(False if regenerated else True),
	)
	_attach_merge_analyzer_policy_info(analyzer_obj, policy_info)
	_write_merge_analyzer_policy_info_to_dir(analyzer_dir=analyzer_dir, policy_info=policy_info)
	return analyzer_obj, analyzer_dir, policy_info, regenerated, regen_reason


def _load_or_recompute_bombcell_sorting_analyzer(
	*,
	si_module: Any,
	well_out_dir: Path,
	bombcell_out_dir: Path,
	sorter_output_dir: Path,
	stage_config: Any,
) -> tuple[Any, Path, bool, str | None]:
	bombcell_stage_config = _bombcell_analyzer_stage_config(stage_config)
	analyzer_obj, analyzer_dir, _policy_info, regenerated, regen_reason = _prepare_replot_workspace_analyzer(
		si_module=si_module,
		well_out_dir=well_out_dir,
		sorter_output_dir=sorter_output_dir,
		stage_config=bombcell_stage_config,
		analyzer_dir=(Path(bombcell_out_dir).resolve() / "analyzer_output"),
		regenerate_on_replot=bool(
			getattr(stage_config, "bombcell_label_analyzer_regenerate_on_replot", False)
		),
		check_if_regen_is_needed=bool(
			getattr(stage_config, "bombcell_label_analyzer_check_if_regen_is_needed", True)
		),
	)
	return analyzer_obj, analyzer_dir, regenerated, regen_reason


def _prepare_bombcell_sorter_output_workspace(
	*,
	stage_output_root_dir: Path,
	bombcell_out_dir: Path,
	sorter_output_dir: Path,
	stage_config: Any,
) -> dict[str, Any]:
	canonical_sorter_output_dir = Path(sorter_output_dir).resolve()
	canonical_analyzer_dir = (Path(stage_output_root_dir).resolve() / "analyzer_output").resolve()
	cache_sorter_output_before_analyzer_gen = bool(
		getattr(stage_config, "bombcell_label_cache_sorter_output_before_analyzer_gen", False)
	)
	publish_cached_sorter_output_on_success = bool(
		getattr(stage_config, "bombcell_label_publish_cached_sorter_output_on_success", False)
	)
	publish_cached_analyzer_on_success = bool(
		getattr(stage_config, "bombcell_label_publish_cached_analyzer_on_success", False)
	)
	cleanup_analyzer_on_success = bool(
		getattr(stage_config, "bombcell_label_cleanup_analyzer_on_success", False)
	)
	cleanup_cached_sorter_output_on_success = bool(
		getattr(stage_config, "bombcell_label_cleanup_cached_sorter_output_on_success", False)
	)
	cached_sorter_output_dir: Path | None = None
	effective_sorter_output_dir = canonical_sorter_output_dir
	if cache_sorter_output_before_analyzer_gen:
		cached_sorter_output_dir = (Path(bombcell_out_dir).resolve() / "cache" / "sorter_output").resolve()
		_copy_path_to_destination(src=canonical_sorter_output_dir, dst=cached_sorter_output_dir)
		effective_sorter_output_dir = cached_sorter_output_dir
	return {
		"cache_sorter_output_before_analyzer_gen": bool(cache_sorter_output_before_analyzer_gen),
		"publish_cached_sorter_output_on_success": bool(publish_cached_sorter_output_on_success),
		"publish_cached_analyzer_on_success": bool(publish_cached_analyzer_on_success),
		"cleanup_analyzer_on_success": bool(cleanup_analyzer_on_success),
		"cleanup_cached_sorter_output_on_success": bool(cleanup_cached_sorter_output_on_success),
		"canonical_sorter_output_dir": canonical_sorter_output_dir,
		"effective_sorter_output_dir": effective_sorter_output_dir,
		"cached_sorter_output_dir": cached_sorter_output_dir,
		"canonical_analyzer_dir": canonical_analyzer_dir,
	}


def _publish_bombcell_cached_workspace_outputs(
	*,
	workspace_info: dict[str, Any],
	analyzer_dir: Path,
) -> dict[str, str]:
	if not bool(workspace_info.get("cache_sorter_output_before_analyzer_gen", False)):
		return {}
	published: dict[str, str] = {}
	effective_sorter_output_dir = workspace_info.get("effective_sorter_output_dir", None)
	canonical_sorter_output_dir = workspace_info.get("canonical_sorter_output_dir", None)
	if bool(workspace_info.get("publish_cached_sorter_output_on_success", False)):
		if isinstance(effective_sorter_output_dir, Path) and isinstance(canonical_sorter_output_dir, Path):
			_copy_path_to_destination(src=effective_sorter_output_dir, dst=canonical_sorter_output_dir)
			published["sorter_output_dir"] = str(canonical_sorter_output_dir)
	canonical_analyzer_dir = workspace_info.get("canonical_analyzer_dir", None)
	if bool(workspace_info.get("publish_cached_analyzer_on_success", False)):
		if isinstance(canonical_analyzer_dir, Path):
			_copy_path_to_destination(src=Path(analyzer_dir).resolve(), dst=canonical_analyzer_dir)
			published["analyzer_dir"] = str(canonical_analyzer_dir)
	return published


def _cleanup_bombcell_success_outputs(
	*,
	workspace_info: dict[str, Any],
	analyzer_dir: Path,
) -> dict[str, str]:
	removed: dict[str, str] = {}
	if bool(workspace_info.get("cleanup_analyzer_on_success", False)):
		resolved_analyzer_dir = Path(analyzer_dir).resolve()
		if resolved_analyzer_dir.exists():
			if resolved_analyzer_dir.is_dir():
				shutil.rmtree(resolved_analyzer_dir, ignore_errors=True)
			else:
				resolved_analyzer_dir.unlink(missing_ok=True)
			removed["analyzer_dir"] = str(resolved_analyzer_dir)
	if bool(workspace_info.get("cleanup_cached_sorter_output_on_success", False)):
		cached_sorter_output_dir = workspace_info.get("cached_sorter_output_dir", None)
		if isinstance(cached_sorter_output_dir, Path):
			resolved_cached_sorter_output_dir = cached_sorter_output_dir.resolve()
			if resolved_cached_sorter_output_dir.exists():
				if resolved_cached_sorter_output_dir.is_dir():
					shutil.rmtree(resolved_cached_sorter_output_dir, ignore_errors=True)
				else:
					resolved_cached_sorter_output_dir.unlink(missing_ok=True)
				removed["cached_sorter_output_dir"] = str(resolved_cached_sorter_output_dir)
	return removed


def _release_loaded_analyzer_extensions(
	*,
	analyzer: Any,
	extension_names: list[str] | tuple[str, ...] | set[str] | None = None,
) -> list[str]:
	loaded_names_getter = getattr(analyzer, "get_loaded_extension_names", None)
	extensions = getattr(analyzer, "extensions", None)
	if not callable(loaded_names_getter) or not isinstance(extensions, dict):
		return []

	requested_names: set[str] | None = None
	if extension_names is not None:
		requested_names = {
			str(name).strip()
			for name in list(extension_names)
			if str(name).strip()
		}

	released: list[str] = []
	for extension_name in list(loaded_names_getter()):
		name = str(extension_name).strip()
		if not name:
			continue
		if requested_names is not None and name not in requested_names:
			continue
		extension_obj = extensions.pop(name, None)
		if extension_obj is None:
			continue
		try:
			data = getattr(extension_obj, "data", None)
			if isinstance(data, dict):
				data.clear()
		except Exception:
			pass
		for attr_name in ("_some_spikes",):
			try:
				delattr(extension_obj, attr_name)
			except Exception:
				pass
		released.append(name)

	if released:
		gc.collect()
	return released


def _describe_merge_analyzer_policy_info(
	*,
	analyzer: Any | None,
	stage_config: Any,
	loaded_analyzer_had_sparsity: bool | None = None,
	rebuild_reason: str | None = None,
	reused_cached_analyzer: bool | None = None,
) -> dict[str, Any]:
	info = _requested_merge_analyzer_policy(stage_config)
	final_has_sparsity = (_analyzer_has_sparsity(analyzer) if analyzer is not None else None)
	info.update(
		{
			"loaded_analyzer_had_sparsity": loaded_analyzer_had_sparsity,
			"rebuild_reason": rebuild_reason,
			"reused_cached_analyzer": reused_cached_analyzer,
			"final_analyzer_has_sparsity": final_has_sparsity,
			"sparsity_validation_error": (
				"compute_sparsity_disabled_but_analyzer_has_sparsity"
				if (not bool(info.get("requested_compute_sparsity", True))) and bool(final_has_sparsity)
				else None
			),
		}
	)
	return info


def _analyzer_has_extension(analyzer: Any, extension_name: str) -> bool:
	has_extension = getattr(analyzer, "has_extension", None)
	if not callable(has_extension):
		return False
	try:
		return bool(has_extension(extension_name))
	except Exception:
		return False


def _compute_analyzer_extension(*, analyzer: Any, extension_name: str, kwargs: dict[str, Any] | None = None) -> bool:
	compute_extension = getattr(analyzer, "compute", None)
	if not callable(compute_extension):
		return False

	extension_kwargs = dict(kwargs or {})
	if extension_name == "random_spikes" and (
		extension_kwargs.get("method") == "percentage"
		or ("min_spikes_per_unit" in extension_kwargs)
		or bool(extension_kwargs.get("log_before_after_spike_counts", False))
	):
		_install_spikeinterface_random_spikes_percentage_compatibility()
	attempts: list[tuple[Any, dict[str, Any]]] = []
	if extension_name == "random_spikes" and extension_kwargs.get("method") == "all":
		random_spikes_payload = {"method": "all"}
		for key in (
			"max_spikes_per_unit",
			"margin_size",
			"seed",
			"percentage",
			"min_spikes_per_unit",
			"log_before_after_spike_counts",
		):
			if key in extension_kwargs:
				random_spikes_payload[key] = extension_kwargs[key]
		random_spikes_job_kwargs = {
			key: value for key, value in extension_kwargs.items() if key not in random_spikes_payload
		}
		attempts.append(({"random_spikes": random_spikes_payload}, random_spikes_job_kwargs))
	attempts.extend(
		[
			(extension_name, extension_kwargs),
			([extension_name], extension_kwargs),
		]
	)

	for candidate, candidate_kwargs in attempts:
		try:
			compute_extension(candidate, **candidate_kwargs)
			return True
		except Exception:
			continue
	return False


def _ensure_merge_analyzer_extensions(
	*,
	analyzer: Any,
	stage_config: Any,
	include_unit_locations: bool,
	include_templates: bool = True,
) -> list[str]:
	required_extensions: list[str] = []
	if include_templates:
		required_extensions.extend(["random_spikes", "waveforms", "templates"])
	if include_unit_locations:
		required_extensions.append("unit_locations")

	computed_extensions: list[str] = []
	for extension_name in required_extensions:
		if _analyzer_has_extension(analyzer, extension_name):
			continue
		extension_kwargs = _merge_analyzer_extension_kwargs(stage_config, extension_name)
		if _compute_analyzer_extension(
			analyzer=analyzer,
			extension_name=extension_name,
			kwargs=extension_kwargs,
		):
			computed_extensions.append(str(extension_name))
	return computed_extensions


def _resolve_slay_model_cache_path(*, well_out_dir: Path, output_rel_root: str, stage_config: Any) -> Path | None:
	slay_model_cache_relpath = _as_optional_relpath(getattr(stage_config, "slay_model_cache_relpath", None))
	if slay_model_cache_relpath is None:
		return None
	merge_output_rel_root = _compose_output_rel_root(
		stage_output_rel_root=output_rel_root,
		child_rel_root=getattr(stage_config, "merge_rel_output_root", None),
	)
	return _resolve_under_spikesort_output_root(
		well_out_dir=well_out_dir,
		output_rel_root=merge_output_rel_root,
		relpath=slay_model_cache_relpath,
	)


def _path_overlaps_preserved_targets(*, candidate: Path, preserved_targets: list[Path]) -> bool:
	for preserved in preserved_targets:
		if candidate == preserved:
			return True
		if preserved in candidate.parents:
			return True
		if candidate in preserved.parents:
			return True
	return False


def _remove_path_preserving_targets(*, target: Path, preserved_targets: list[Path], removed_paths: list[str]) -> None:
	if not target.exists():
		return

	target_resolved = target.resolve()
	if not _path_overlaps_preserved_targets(candidate=target_resolved, preserved_targets=preserved_targets):
		if target.is_dir():
			shutil.rmtree(target, ignore_errors=True)
		else:
			target.unlink(missing_ok=True)
		removed_paths.append(str(target_resolved))
		return

	if not target.is_dir():
		return

	for child in list(target.iterdir()):
		_remove_path_preserving_targets(
			target=child,
			preserved_targets=preserved_targets,
			removed_paths=removed_paths,
		)


def _install_numpy_cupy_fallback_module() -> None:
	import numpy as np

	shim = ModuleType("cupy")
	shim.array = np.array  # type: ignore[attr-defined]
	shim.asarray = np.asarray  # type: ignore[attr-defined]
	shim.asnumpy = np.asarray  # type: ignore[attr-defined]
	shim.mean = np.mean  # type: ignore[attr-defined]
	shim.zeros = np.zeros  # type: ignore[attr-defined]
	shim.float32 = np.float32  # type: ignore[attr-defined]
	shim.ndarray = np.ndarray  # type: ignore[attr-defined]
	sys.modules["cupy"] = shim


def _install_marshmallow_field_fail_compatibility() -> None:
	try:
		from marshmallow import ValidationError
		from marshmallow.fields import Field
	except Exception:
		return

	if callable(getattr(Field, "fail", None)):
		return

	def _compat_fail(self: Any, key: str, **kwargs: Any) -> None:
		make_error = getattr(self, "make_error", None)
		if callable(make_error):
			raise make_error(key, **kwargs)

		message: str | None = None
		try:
			error_messages = getattr(self, "error_messages", None)
			if isinstance(error_messages, dict):
				message = error_messages.get(key)
		except Exception:
			message = None

		raise ValidationError(message or str(key))

	setattr(Field, "fail", _compat_fail)


def _candidate_slay_recording_metadata_paths(*, ks_dir: Path) -> list[Path]:
	paths: list[Path] = []
	for candidate in (
		(ks_dir / "spikeinterface_recording.json").resolve(),
		(ks_dir.parent / "spikeinterface_recording.json").resolve(),
	):
		if candidate not in paths:
			paths.append(candidate)
	return paths


def _load_spikeinterface_recording_dir_from_metadata(*, metadata_path: Path) -> Path | None:
	if not metadata_path.exists():
		return None

	try:
		payload = json.loads(metadata_path.read_text(encoding="utf-8"))
	except Exception:
		return None

	if not isinstance(payload, dict):
		return None
	kwargs = payload.get("kwargs", None)
	if not isinstance(kwargs, dict):
		return None

	folder_path = str(kwargs.get("folder_path", "")).strip()
	if not folder_path:
		return None

	recording_dir = Path(folder_path).expanduser().resolve()
	if recording_dir.exists():
		return recording_dir
	return None


def _rewrite_spikeinterface_recording_dir_metadata(*, metadata_path: Path, recording_dir: Path) -> None:
	if not metadata_path.exists():
		return

	try:
		payload = json.loads(metadata_path.read_text(encoding="utf-8"))
	except Exception:
		return

	if not isinstance(payload, dict):
		return
	kwargs = payload.get("kwargs", None)
	if not isinstance(kwargs, dict):
		return

	desired_folder_path = str(recording_dir.resolve())
	current_folder_path = str(kwargs.get("folder_path", "")).strip()
	if current_folder_path == desired_folder_path:
		return

	kwargs["folder_path"] = desired_folder_path
	metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
	LOGGER.info(
		"SLAy repaired spikeinterface recording metadata metadata_path=%s recording_dir=%s",
		metadata_path,
		recording_dir,
	)


def _rewrite_kilosort_params_dat_path(*, ks_dir: Path, data_filepath: Path) -> None:
	params_path = (ks_dir / "params.py").resolve()
	if not params_path.exists():
		return

	try:
		original_text = params_path.read_text(encoding="utf-8")
	except Exception:
		return

	updated_lines: list[str] = []
	replaced = False
	desired_line = f"dat_path = {str(data_filepath.resolve())!r}"
	for line in original_text.splitlines():
		if (not replaced) and ("=" in line):
			key, _value = line.split("=", 1)
			if str(key).strip() == "dat_path":
				updated_lines.append(desired_line)
				replaced = True
				continue
		updated_lines.append(line)

	if not replaced:
		return

	updated_text = "\n".join(updated_lines)
	if original_text.endswith("\n"):
		updated_text += "\n"
	if updated_text == original_text:
		return

	params_path.write_text(updated_text, encoding="utf-8")
	LOGGER.info(
		"SLAy repaired Kilosort params.py dat_path params_path=%s data_filepath=%s",
		params_path,
		data_filepath,
	)


def _resolve_slay_recording_dir(*, ks_dir: Path, preferred_recording_dir: Path | None = None) -> Path | None:
	metadata_paths = _candidate_slay_recording_metadata_paths(ks_dir=ks_dir)
	if preferred_recording_dir is not None:
		preferred = Path(preferred_recording_dir).expanduser().resolve()
		if preferred.exists():
			for metadata_path in metadata_paths:
				_rewrite_spikeinterface_recording_dir_metadata(
					metadata_path=metadata_path,
					recording_dir=preferred,
				)
			return preferred

	for metadata_path in metadata_paths:
		recording_dir = _load_spikeinterface_recording_dir_from_metadata(metadata_path=metadata_path)
		if recording_dir is not None:
			return recording_dir

	for parent in (ks_dir, *ks_dir.parents):
		candidate = (parent / "preprocess_outputs" / "preprocessed_recording").resolve()
		if not candidate.exists():
			continue
		for metadata_path in metadata_paths:
			_rewrite_spikeinterface_recording_dir_metadata(
				metadata_path=metadata_path,
				recording_dir=candidate,
			)
		return candidate

	return None


def _resolve_slay_data_filepath(*, ks_dir: Path, dat_path: Any, preferred_recording_dir: Path | None = None) -> str | None:
	if isinstance(dat_path, (list, tuple)):
		dat_path = (dat_path[0] if len(dat_path) > 0 else None)
	if dat_path is None:
		return None

	dat_path_text = str(dat_path).strip()
	if not dat_path_text:
		return None

	dat_path_candidate = Path(dat_path_text).expanduser()
	direct_candidate = (
		dat_path_candidate.resolve()
		if dat_path_candidate.is_absolute()
		else (ks_dir / dat_path_text).resolve()
	)
	if direct_candidate.exists():
		return str(direct_candidate)

	recording_dir = _resolve_slay_recording_dir(
		ks_dir=ks_dir,
		preferred_recording_dir=preferred_recording_dir,
	)
	if recording_dir is not None:
		filename = dat_path_candidate.name
		if filename:
			repaired_candidate = (recording_dir / filename).resolve()
			if repaired_candidate.exists():
				_rewrite_kilosort_params_dat_path(ks_dir=ks_dir, data_filepath=repaired_candidate)
				LOGGER.warning(
					"SLAy repaired stale Kilosort data_filepath ks_dir=%s stale_path=%s repaired_path=%s",
					ks_dir,
					dat_path_text,
					repaired_candidate,
				)
				return str(repaired_candidate)

		raw_candidates = sorted(recording_dir.glob("traces_cached_seg*.raw"))
		if len(raw_candidates) == 1:
			_rewrite_kilosort_params_dat_path(ks_dir=ks_dir, data_filepath=raw_candidates[0])
			LOGGER.warning(
				"SLAy inferred Kilosort data_filepath from preprocessed recording directory ks_dir=%s stale_path=%s repaired_path=%s",
				ks_dir,
				dat_path_text,
				raw_candidates[0],
			)
			return str(raw_candidates[0].resolve())

	return str(direct_candidate)


def _resolve_existing_slay_preferred_recording_dir(*, well_out_dir: Path, stage_config: Any) -> Path | None:
	recording_dir = _resolve_under_well(
		well_out_dir=well_out_dir,
		relpath=str(
			getattr(stage_config, "preprocess_concat_recording_relpath", None)
			or "preprocess_outputs/preprocessed_recording"
		),
	)
	if recording_dir.exists():
		return recording_dir.resolve()
	return None


def _preflight_slay_binary_input(*, sorter_output_dir: Path, well_out_dir: Path, stage_config: Any) -> dict[str, Any]:
	ks_dir = _normalize_slay_kilosort_dir(sorter_output_dir=sorter_output_dir)
	params_path = (ks_dir / "params.py").resolve()
	if not params_path.exists():
		raise FileNotFoundError(
			"SLAy preflight requires a Kilosort params.py file; "
			f"ks_dir={ks_dir} params_path={params_path}"
		)

	ksparams: dict[str, Any] = {}
	with params_path.open("r", encoding="utf-8") as f:
		for line in f:
			if "=" not in line:
				continue
			key, value = line.split("=", 1)
			ksparams[str(key).strip()] = eval(str(value).strip())

	dat_path = ksparams.get("dat_path", None)
	preferred_recording_dir = _resolve_existing_slay_preferred_recording_dir(
		well_out_dir=well_out_dir,
		stage_config=stage_config,
	)
	resolved_data_filepath = _resolve_slay_data_filepath(
		ks_dir=ks_dir,
		dat_path=dat_path,
		preferred_recording_dir=preferred_recording_dir,
	)
	resolved_recording_dir = _resolve_slay_recording_dir(
		ks_dir=ks_dir,
		preferred_recording_dir=preferred_recording_dir,
	)

	if resolved_data_filepath is None:
		raise FileNotFoundError(
			"SLAy requires a materialized binary recording file before merge execution, "
			"but params.py does not define dat_path and no fallback binary could be inferred; "
			f"ks_dir={ks_dir} params_path={params_path} preferred_recording_dir={preferred_recording_dir} "
			f"resolved_recording_dir={resolved_recording_dir}"
		)

	resolved_data_path = Path(str(resolved_data_filepath)).expanduser().resolve()
	if (not resolved_data_path.exists()) or (not resolved_data_path.is_file()):
		fallback_raw_candidates: list[str] = []
		if resolved_recording_dir is not None and resolved_recording_dir.exists():
			fallback_raw_candidates = [
				str(path.resolve())
				for path in sorted(resolved_recording_dir.glob("traces_cached_seg*.raw"))
			]
		raise FileNotFoundError(
			"SLAy requires a materialized binary recording file before merge execution; "
			f"ks_dir={ks_dir} params_path={params_path} dat_path={dat_path!r} "
			f"resolved_data_filepath={resolved_data_path} preferred_recording_dir={preferred_recording_dir} "
			f"resolved_recording_dir={resolved_recording_dir} fallback_raw_candidates={fallback_raw_candidates}"
		)

	return {
		"ks_dir": str(ks_dir),
		"params_path": str(params_path),
		"dat_path": dat_path,
		"data_filepath": str(resolved_data_path),
		"preferred_recording_dir": (
			str(preferred_recording_dir)
			if preferred_recording_dir is not None
			else None
		),
		"resolved_recording_dir": (
			str(resolved_recording_dir)
			if resolved_recording_dir is not None
			else None
		),
	}


def _import_slay_run_function(*, allow_numpy_fallback: bool) -> Callable[[dict[str, Any]], None]:
	def _patch_parse_kilosort_params(module: Any) -> None:
		original = getattr(module, "parse_kilosort_params", None)
		if not callable(original):
			return

		def _patched(args: dict[str, Any]) -> dict[str, Any]:
			ks_folder = str(args.get("KS_folder", "")).strip()
			if not ks_folder:
				return original(args)

			preferred_recording_dir_raw = args.pop("__axon_recon_preprocess_recording_dir", None)
			preferred_recording_dir: Path | None = None
			if preferred_recording_dir_raw is not None and str(preferred_recording_dir_raw).strip():
				preferred_recording_dir = Path(str(preferred_recording_dir_raw)).expanduser().resolve()

			ks_dir = Path(ks_folder).expanduser().resolve()
			ksparam_path = ks_dir / "params.py"

			ksparams: dict[str, Any] = {}
			with ksparam_path.open("r", encoding="utf-8") as f:
				for line in f:
					if "=" not in line:
						continue
					key, value = line.split("=", 1)
					ksparams[str(key).strip()] = eval(str(value).strip())

			dat_path = ksparams.pop("dat_path", None)
			data_filepath = _resolve_slay_data_filepath(
				ks_dir=ks_dir,
				dat_path=dat_path,
				preferred_recording_dir=preferred_recording_dir,
			)
			if data_filepath is not None:
				ksparams["data_filepath"] = data_filepath
			if "n_channels_dat" in ksparams:
				ksparams["n_chan"] = ksparams.pop("n_channels_dat")
			args.update(ksparams)
			return args

		setattr(module, "parse_kilosort_params", _patched)

	_install_marshmallow_field_fail_compatibility()

	try:
		module = importlib.import_module("slay.run")
		_patch_parse_kilosort_params(module)
		run_slay = getattr(module, "run_slay", None)
		if callable(run_slay):
			return run_slay
		raise RuntimeError("SLAy import succeeded but slay.run.run_slay is not callable")
	except ModuleNotFoundError as exc:
		if str(getattr(exc, "name", "")) != "cupy" or not bool(allow_numpy_fallback):
			raise
		_install_numpy_cupy_fallback_module()
		module = importlib.import_module("slay.run")
		_patch_parse_kilosort_params(module)
		run_slay = getattr(module, "run_slay", None)
		if callable(run_slay):
			return run_slay
		raise RuntimeError("SLAy import succeeded with numpy fallback but run_slay is not callable")


def _normalize_cluster_id(raw: Any) -> str:
	try:
		return str(int(raw))
	except Exception:
		return str(raw)


def _load_metrics_lookup(metrics_tsv_path: Path) -> dict[tuple[str, str], dict[str, str]]:
	if not metrics_tsv_path.exists():
		return {}
	lookup: dict[tuple[str, str], dict[str, str]] = {}
	with metrics_tsv_path.open("r", encoding="utf-8", newline="") as f:
		reader = csv.DictReader(f, delimiter="\t")
		for row in reader:
			a = _normalize_cluster_id(row.get("Cluster 1", ""))
			b = _normalize_cluster_id(row.get("Cluster 2", ""))
			if not a or not b:
				continue
			key = tuple(sorted((a, b)))
			lookup[key] = dict(row)
	return lookup


def _build_candidate_pair_rows(
	*,
	merge_groups: dict[str, list[Any]],
	metrics_lookup: dict[tuple[str, str], dict[str, str]],
) -> list[dict[str, str]]:
	rows: list[dict[str, str]] = []
	for group_id, members_raw in merge_groups.items():
		if not isinstance(members_raw, list):
			continue
		members = [_normalize_cluster_id(m) for m in members_raw]
		if len(members) < 2:
			continue
		members_sorted = sorted(members, key=lambda token: int(token) if token.isdigit() else token)
		group_members = "|".join(members_sorted)
		for idx_a in range(len(members_sorted)):
			for idx_b in range(idx_a + 1, len(members_sorted)):
				a = members_sorted[idx_a]
				b = members_sorted[idx_b]
				metrics = metrics_lookup.get(tuple(sorted((a, b))), {})
				rows.append(
					{
						"group_id": str(group_id),
						"group_size": str(len(members_sorted)),
						"group_members": group_members,
						"cluster_a": a,
						"cluster_b": b,
						"final_metric": str(metrics.get("Final Metric", "")),
						"similarity": str(metrics.get("Similarity", "")),
						"xcorr_significance": str(metrics.get("Cross-correlation Significance", "")),
						"refractory_penalty": str(metrics.get("Refractory Period Penalty", "")),
					}
				)
	return rows


def _write_candidate_pairs_tsv(path: Path, rows: list[dict[str, str]]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	fieldnames = [
		"group_id",
		"group_size",
		"group_members",
		"cluster_a",
		"cluster_b",
		"final_metric",
		"similarity",
		"xcorr_significance",
		"refractory_penalty",
	]
	with path.open("w", encoding="utf-8", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
		writer.writeheader()
		for row in rows:
			writer.writerow(row)


def _normalize_merge_method_token(raw: Any) -> str:
	text = str(raw or "").strip().lower()
	if text in {"slay", "s_l_a_y"}:
		return "slay"
	if text in {"auto_merge", "automerge", "auto-merge"}:
		return "auto_merge"
	if text in {"unitmatch", "unit_match", "unit-match"}:
		return "unitmatch"
	return text


def _resolve_sorter_output_dir(*, well_out_dir: Path, output_rel_root: str, stage_config: Any) -> Path:
	configured_relpath = getattr(stage_config, "slay_sorter_output_relpath", None)
	default_relpath = "sorter_output"
	sorter_relpath = (str(configured_relpath) if configured_relpath else default_relpath)

	merge_output_rel_root = _compose_output_rel_root(
		stage_output_rel_root=output_rel_root,
		child_rel_root=getattr(stage_config, "merge_rel_output_root", None),
	)
	stage_output_rel_root = str(output_rel_root).strip().lstrip("/") or "spikesort_outputs"

	search_roots: list[str]
	if configured_relpath is None:
		search_roots = [stage_output_rel_root]
		if merge_output_rel_root != stage_output_rel_root:
			search_roots.append(merge_output_rel_root)
	else:
		search_roots = [merge_output_rel_root]
		if merge_output_rel_root != stage_output_rel_root:
			search_roots.append(stage_output_rel_root)

	def _normalize_candidate(path: Path) -> Path:
		has_wrapper_markers = bool(
			(path / "spikeinterface_params.json").exists()
			or (path / "spikeinterface_log.json").exists()
			or (path / "in_container_sorting" / "si_folder.json").exists()
		)
		if (not has_wrapper_markers) and (not (path / "params.py").exists()) and (
			(path / "sorter_output" / "params.py").exists()
		):
			return (path / "sorter_output").resolve()
		return path

	first_candidate: Path | None = None
	for search_root in search_roots:
		candidate = _resolve_under_spikesort_output_root(
			well_out_dir=well_out_dir,
			output_rel_root=search_root,
			relpath=sorter_relpath,
		)
		candidate = _normalize_candidate(candidate)
		if first_candidate is None:
			first_candidate = candidate
		if candidate.exists():
			return candidate

	if first_candidate is not None:
		return first_candidate
	return _normalize_candidate(
		_resolve_under_spikesort_output_root(
			well_out_dir=well_out_dir,
			output_rel_root=merge_output_rel_root,
			relpath=sorter_relpath,
		)
	)


def _normalize_slay_kilosort_dir(*, sorter_output_dir: Path) -> Path:
	resolved = Path(sorter_output_dir).resolve()
	candidates: list[Path] = []
	for candidate in (
		resolved,
		(resolved / "sorter_output").resolve(),
		(resolved / "in_container_sorting").resolve(),
		(resolved.parent.resolve() if resolved.name == "sorter_output" else None),
	):
		if candidate is None or candidate in candidates:
			continue
		candidates.append(candidate)

	for candidate in candidates:
		if (candidate / "params.py").exists():
			return candidate

	return resolved


def _extract_bombcell_label_mapping(labels_obj: Any) -> dict[str, str]:
	out: dict[str, str] = {}
	if labels_obj is None:
		return out

	def _extract_label_value(raw_row: Any) -> str | None:
		row_dict: dict[str, Any] | None = None
		if isinstance(raw_row, dict):
			row_dict = raw_row
		else:
			to_dict = getattr(raw_row, "to_dict", None)
			if callable(to_dict):
				try:
					candidate = to_dict()
				except Exception:
					candidate = None
				if isinstance(candidate, dict):
					row_dict = candidate

		label_raw: Any = None
		if isinstance(row_dict, dict):
			label_raw = row_dict.get("bombcell_label", None)
			if label_raw is None:
				label_raw = row_dict.get("label", None)
			if label_raw is None and len(row_dict) == 1:
				label_raw = next(iter(row_dict.values()))
		else:
			for column_name in ("bombcell_label", "label"):
				try:
					label_raw = raw_row[column_name]
				except Exception:
					label_raw = None
				if label_raw is not None:
					break

		label = str(label_raw).strip() if label_raw is not None else None
		return label or None

	iterrows = getattr(labels_obj, "iterrows", None)
	if callable(iterrows):
		for raw_idx, raw_row in iterrows():
			unit_id = _normalize_cluster_id(raw_idx)
			if not unit_id:
				continue
			label = _extract_label_value(raw_row)
			if label:
				out[unit_id] = label
		return out

	if isinstance(labels_obj, dict):
		for raw_unit_id, raw_label in labels_obj.items():
			unit_id = _normalize_cluster_id(raw_unit_id)
			if not unit_id:
				continue
			label = str(raw_label).strip() if raw_label is not None else ""
			if label:
				out[unit_id] = label
		return out

	return out


def _read_kilosort_cluster_labels_tsv(*, path: Path, default_label_column: str) -> tuple[dict[str, str], str]:
	if not path.exists():
		return {}, str(default_label_column)

	try:
		lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
	except Exception:
		return {}, str(default_label_column)

	if not lines:
		return {}, str(default_label_column)

	header_tokens = lines[0].split()
	if len(header_tokens) < 2:
		return {}, str(default_label_column)

	cluster_col_idx = 0
	label_col_idx = 1
	for idx, token in enumerate(header_tokens):
		normalized = str(token).strip().lower()
		if normalized in {"cluster_id", "clusterid", "id"}:
			cluster_col_idx = idx
			break

	for idx, token in enumerate(header_tokens):
		if idx == cluster_col_idx:
			continue
		label_col_idx = idx
		break

	label_column = str(header_tokens[label_col_idx]).strip() or str(default_label_column)
	out: dict[str, str] = {}
	for raw_line in lines[1:]:
		parts = raw_line.split()
		if len(parts) <= max(cluster_col_idx, label_col_idx):
			continue
		unit_id = _normalize_cluster_id(parts[cluster_col_idx])
		if not unit_id:
			continue
		label = str(parts[label_col_idx]).strip()
		if not label:
			continue
		out[unit_id] = label

	return out, label_column


def _spike_count_for_unit(*, sorting: Any, unit_id: Any) -> int:
	get_num_segments = getattr(sorting, "get_num_segments", None)
	try:
		num_segments = int(get_num_segments()) if callable(get_num_segments) else 1
	except Exception:
		num_segments = 1
	if num_segments <= 0:
		num_segments = 1

	get_unit_spike_train = getattr(sorting, "get_unit_spike_train", None)
	if not callable(get_unit_spike_train):
		return 0

	unit_id_candidates: list[Any] = [unit_id]
	try:
		normalized_int = int(unit_id)
	except Exception:
		normalized_int = None
	if normalized_int is not None and all(candidate != normalized_int for candidate in unit_id_candidates):
		unit_id_candidates.append(normalized_int)

	total = 0
	for segment_index in range(int(num_segments)):
		spike_train = None
		for candidate_unit_id in unit_id_candidates:
			try:
				spike_train = get_unit_spike_train(unit_id=candidate_unit_id, segment_index=int(segment_index))
				break
			except TypeError:
				if segment_index > 0:
					continue
				try:
					spike_train = get_unit_spike_train(candidate_unit_id)
					break
				except Exception:
					continue
			except Exception:
				continue
		if spike_train is None:
			continue
		try:
			total += int(len(spike_train))
		except Exception:
			try:
				total += int(getattr(spike_train, "size", 0))
			except Exception:
				continue
	return int(total)


def _resolve_existing_sorter_output_dir(*, stage_output_root_dir: Path) -> Path:
	candidates = [
		(stage_output_root_dir / "sorter_output" / "sorter_output").resolve(),
		(stage_output_root_dir / "sorter_output" / "in_container_sorting").resolve(),
		(stage_output_root_dir / "sorter_output").resolve(),
		stage_output_root_dir.resolve(),
	]
	for candidate in candidates:
		if not candidate.exists():
			continue
		if _is_kilosort_folder(candidate):
			return candidate
		if (candidate / "cluster_KSLabel.tsv").exists() or (candidate / "cluster_group.tsv").exists():
			return candidate
	for candidate in candidates:
		if candidate.exists():
			return candidate
	raise FileNotFoundError(f"Sorter output directory not found under {stage_output_root_dir}")


def _build_sort_summary_payload(
	*,
	stream_id: str,
	stage_output_root_dir: Path,
	sorter_output_dir: Path,
	sorter_name: str,
) -> dict[str, Any]:
	si_module = _import_spikeinterface_full_module()
	sorting = _load_sorting_from_sorter_output_dir(
		si_module=si_module,
		sorter_output_dir=sorter_output_dir,
		sorter_name=sorter_name,
	)
	unit_ids = list(_unit_ids_from_obj(sorting))
	ks_labels, ks_label_column = _read_kilosort_cluster_labels_tsv(
		path=(sorter_output_dir / "cluster_KSLabel.tsv").resolve(),
		default_label_column="KSLabel",
	)
	group_labels, group_label_column = _read_kilosort_cluster_labels_tsv(
		path=(sorter_output_dir / "cluster_group.tsv").resolve(),
		default_label_column="group",
	)

	units: list[dict[str, Any]] = []
	label_counts: dict[str, int] = {}
	spike_counts: list[int] = []
	for unit_id in sorted(unit_ids, key=_unit_sort_key):
		spike_count = _spike_count_for_unit(sorting=sorting, unit_id=unit_id)
		spike_counts.append(int(spike_count))
		label = str(ks_labels.get(unit_id) or group_labels.get(unit_id) or "unlabeled")
		label_counts[label] = int(label_counts.get(label, 0)) + 1
		units.append(
			{
				"unit_id": str(unit_id),
				"kilosort_label": label,
				"spike_count": int(spike_count),
			}
		)

	payload: dict[str, Any] = {
		"status": "ok",
		"stream_id": str(stream_id),
		"stage_output_root_dir": str(stage_output_root_dir),
		"sorter_output_dir": str(sorter_output_dir),
		"sorter": str(sorter_name),
		"label_sources": {
			"cluster_kslabel_tsv": str((sorter_output_dir / "cluster_KSLabel.tsv").resolve()),
			"cluster_group_tsv": str((sorter_output_dir / "cluster_group.tsv").resolve()),
			"cluster_kslabel_column": str(ks_label_column),
			"cluster_group_column": str(group_label_column),
		},
		"unit_count": int(len(units)),
		"counts_by_label": dict(sorted(label_counts.items())),
		"spike_count_stats": {
			"min": int(min(spike_counts)) if spike_counts else 0,
			"max": int(max(spike_counts)) if spike_counts else 0,
		},
		"units": units,
	}
	return payload


def _emit_sort_summary_logs(*, payload: dict[str, Any]) -> None:
	stream_id = str(payload.get("stream_id", ""))
	unit_count = int(payload.get("unit_count", 0) or 0)
	counts_by_label = dict(payload.get("counts_by_label", {}) or {})
	spike_count_stats = dict(payload.get("spike_count_stats", {}) or {})
	LOGGER.info(
		"Sort summary [stream=%s] units=%d labels=%s spike_count_min=%s spike_count_max=%s",
		stream_id,
		unit_count,
		counts_by_label,
		spike_count_stats.get("min", 0),
		spike_count_stats.get("max", 0),
	)
	for unit_payload in list(payload.get("units", []) or []):
		if not isinstance(unit_payload, dict):
			continue
		LOGGER.info(
			"Sort unit [stream=%s] unit_id=%s kilosort_label=%s spike_count=%s",
			stream_id,
			unit_payload.get("unit_id", ""),
			unit_payload.get("kilosort_label", "unlabeled"),
			unit_payload.get("spike_count", 0),
		)


def _write_sort_summary_artifacts(*, stage_output_root_dir: Path, payload: dict[str, Any]) -> dict[str, str]:
	artifacts: dict[str, str] = {}
	summary_json = (stage_output_root_dir / "summarize_sort_summary.json").resolve()
	units_tsv = (stage_output_root_dir / "summarize_sort_units.tsv").resolve()
	_write_json(summary_json, payload)
	with units_tsv.open("w", encoding="utf-8", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=["unit_id", "kilosort_label", "spike_count"], delimiter="\t")
		writer.writeheader()
		for row in list(payload.get("units", []) or []):
			if not isinstance(row, dict):
				continue
			writer.writerow(
				{
					"unit_id": str(row.get("unit_id", "")),
					"kilosort_label": str(row.get("kilosort_label", "unlabeled")),
					"spike_count": int(row.get("spike_count", 0) or 0),
				}
			)
	artifacts["summarize_sort.summary_json"] = str(summary_json)
	artifacts["summarize_sort.units_tsv"] = str(units_tsv)
	return artifacts


def _run_summarize_sort_phase(
	*,
	stream_id: str,
	stage_output_root_dir: Path,
	sorter_name: str,
	emit_logs: bool,
	generate_artifacts: bool,
) -> dict[str, Any]:
	sorter_output_dir = _resolve_existing_sorter_output_dir(stage_output_root_dir=stage_output_root_dir)
	payload = _build_sort_summary_payload(
		stream_id=stream_id,
		stage_output_root_dir=stage_output_root_dir,
		sorter_output_dir=sorter_output_dir,
		sorter_name=sorter_name,
	)
	payload["emit_logs"] = bool(emit_logs)
	payload["generate_artifacts"] = bool(generate_artifacts)
	artifacts: dict[str, str] = {}
	if bool(emit_logs):
		_emit_sort_summary_logs(payload=payload)
	if bool(generate_artifacts):
		artifacts = _write_sort_summary_artifacts(stage_output_root_dir=stage_output_root_dir, payload=payload)
	payload["outputs"] = dict(artifacts)
	return payload


def _write_kilosort_cluster_labels_tsv(*, path: Path, label_column: str, labels_by_unit: dict[str, str]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8", newline="") as f:
		f.write(f"cluster_id\t{str(label_column).strip() or 'label'}\n")
		for unit_id in sorted(labels_by_unit.keys(), key=_unit_sort_key):
			label = str(labels_by_unit.get(unit_id, "")).strip()
			if not label:
				continue
			f.write(f"{unit_id}\t{label}\n")


def _apply_bombcell_labels_to_kilosort_outputs(
	*,
	ks_dir: Path,
	bombcell_labels_by_unit: dict[str, str],
	write_cluster_group: bool,
) -> dict[str, Any]:
	kslabel_path = (ks_dir / "cluster_KSLabel.tsv").resolve()
	group_path = (ks_dir / "cluster_group.tsv").resolve()

	existing_ks_labels, ks_label_col = _read_kilosort_cluster_labels_tsv(
		path=kslabel_path,
		default_label_column="KSLabel",
	)
	existing_group_labels, group_label_col = _read_kilosort_cluster_labels_tsv(
		path=group_path,
		default_label_column="group",
	)

	all_unit_ids: set[str] = set(_load_kilosort_unit_ids_from_spike_clusters(folder=ks_dir))
	all_unit_ids.update(existing_ks_labels.keys())
	all_unit_ids.update(existing_group_labels.keys())
	all_unit_ids.update(bombcell_labels_by_unit.keys())

	merged_labels_by_unit: dict[str, str] = {}
	for unit_id in all_unit_ids:
		label = bombcell_labels_by_unit.get(unit_id, None)
		if label is None:
			label = existing_ks_labels.get(unit_id, None)
		if label is None:
			label = existing_group_labels.get(unit_id, None)
		if label is None:
			label = "unsorted"
		label_text = str(label).strip()
		if label_text:
			merged_labels_by_unit[unit_id] = label_text

	_write_kilosort_cluster_labels_tsv(
		path=kslabel_path,
		label_column=ks_label_col,
		labels_by_unit=merged_labels_by_unit,
	)

	if write_cluster_group:
		_write_kilosort_cluster_labels_tsv(
			path=group_path,
			label_column=group_label_col,
			labels_by_unit=merged_labels_by_unit,
		)

	return {
		"ks_dir": str(ks_dir),
		"cluster_kslabel_tsv": str(kslabel_path),
		"cluster_group_tsv": (str(group_path) if write_cluster_group else None),
		"n_units_written": int(len(merged_labels_by_unit)),
	}


_BOMBCELL_TEMPLATE_METRIC_NAMES: tuple[str, ...] = (
	"number_of_peaks",
	"peak_to_trough_duration",
	"waveform_baseline_flatness",
	"waveform_ratios",
	"waveform_widths",
	"exp_decay",
)


_BOMBCELL_TEMPLATE_METRIC_COLUMNS: tuple[str, ...] = (
	"num_positive_peaks",
	"num_negative_peaks",
	"peak_to_trough_duration",
	"waveform_baseline_flatness",
	"peak_after_to_trough_ratio",
	"exp_decay",
	"peak_before_to_trough_ratio",
	"peak_before_width",
	"trough_width",
	"peak_before_to_peak_after_ratio",
	"main_peak_to_trough_ratio",
)


_BOMBCELL_QUALITY_METRIC_NAMES: tuple[str, ...] = (
	"amplitude_median",
	"snr",
	"amplitude_cutoff",
	"num_spikes",
	"rp_violation",
	"presence_ratio",
	"drift",
)


_BOMBCELL_QUALITY_METRIC_COLUMNS: tuple[str, ...] = (
	"amplitude_median",
	"snr",
	"amplitude_cutoff",
	"num_spikes",
	"rp_contamination",
	"presence_ratio",
	"drift_ptp",
)


def _analyzer_extension_columns(*, analyzer: Any, extension_name: str) -> set[str]:
	if not _analyzer_has_extension(analyzer, extension_name):
		return set()

	get_extension = getattr(analyzer, "get_extension", None)
	if not callable(get_extension):
		return set()

	try:
		extension = get_extension(extension_name)
	except Exception:
		return set()
	if extension is None:
		return set()

	get_data = getattr(extension, "get_data", None)
	if not callable(get_data):
		return set()

	try:
		data = get_data()
	except Exception:
		return set()

	columns = getattr(data, "columns", None)
	if columns is None:
		return set()

	try:
		return {str(column) for column in list(columns)}
	except Exception:
		return set()


def _analyzer_extension_has_columns(*, analyzer: Any, extension_name: str, required_columns: tuple[str, ...]) -> bool:
	available_columns = _analyzer_extension_columns(analyzer=analyzer, extension_name=extension_name)
	if not available_columns:
		return False
	return all(str(column) in available_columns for column in required_columns)


def _ensure_bombcell_metric_extensions(*, analyzer: Any, stage_config: Any) -> list[str]:
	bombcell_stage_config = _bombcell_analyzer_stage_config(stage_config)
	computed_extensions: list[str] = []
	computed_extensions.extend(
		_ensure_merge_analyzer_extensions(
			analyzer=analyzer,
			stage_config=bombcell_stage_config,
			include_unit_locations=False,
		)
	)
	job_kwargs = _merge_analyzer_compute_job_kwargs(bombcell_stage_config)
	for extension_name in ("noise_levels", "spike_amplitudes", "spike_locations"):
		if _analyzer_has_extension(analyzer, extension_name):
			continue
		if _compute_analyzer_extension(
			analyzer=analyzer,
			extension_name=extension_name,
			kwargs=job_kwargs,
		):
			computed_extensions.append(str(extension_name))

	template_metric_kwargs = dict(job_kwargs)
	template_metric_kwargs["metric_names"] = list(_BOMBCELL_TEMPLATE_METRIC_NAMES)
	template_metric_kwargs["include_multi_channel_metrics"] = True
	if not _analyzer_extension_has_columns(
		analyzer=analyzer,
		extension_name="template_metrics",
		required_columns=_BOMBCELL_TEMPLATE_METRIC_COLUMNS,
	):
		if _compute_analyzer_extension(
			analyzer=analyzer,
			extension_name="template_metrics",
			kwargs=template_metric_kwargs,
		):
			computed_extensions.append("template_metrics")

	quality_metric_kwargs = dict(job_kwargs)
	quality_metric_kwargs["metric_names"] = list(_BOMBCELL_QUALITY_METRIC_NAMES)
	quality_metric_kwargs["skip_pc_metrics"] = True
	if not _analyzer_extension_has_columns(
		analyzer=analyzer,
		extension_name="quality_metrics",
		required_columns=_BOMBCELL_QUALITY_METRIC_COLUMNS,
	):
		if _compute_analyzer_extension(
			analyzer=analyzer,
			extension_name="quality_metrics",
			kwargs=quality_metric_kwargs,
		):
			computed_extensions.append("quality_metrics")

	return computed_extensions


def _run_bombcell_label_phase(
	*,
	well_out_dir: Path,
	stage_output_root_dir: Path,
	output_rel_root: str,
	stage_config: Any,
	force_restart: bool,
	sorter_output_dir: Path | None = None,
) -> dict[str, Any]:
	merge_output_rel_root = _compose_output_rel_root(
		stage_output_rel_root=output_rel_root,
		child_rel_root=getattr(stage_config, "merge_rel_output_root", None),
	)
	bombcell_out_dir = _resolve_under_spikesort_output_root(
		well_out_dir=well_out_dir,
		output_rel_root=merge_output_rel_root,
		relpath=str(getattr(stage_config, "bombcell_label_relpath", "bombcell_label_outputs")),
	)
	reports_enabled = bool(getattr(stage_config, "bombcell_label_reports_enabled", True))
	reports_summary_json_enabled = bool(
		getattr(stage_config, "bombcell_label_reports_summary_json_enabled", True)
	)
	reports_summary_json_relpath = str(
		getattr(stage_config, "bombcell_label_reports_summary_json_relpath", "bombcell_label_summary.json")
		or "bombcell_label_summary.json"
	)
	reports_summary_json_relpath = reports_summary_json_relpath.strip().lstrip("/") or "bombcell_label_summary.json"
	summary_json: Path | None = None
	if reports_enabled and reports_summary_json_enabled:
		summary_json = (bombcell_out_dir / reports_summary_json_relpath).resolve()

	delete_on_force_restart = bool(
		getattr(stage_config, "bombcell_label_delete_outputs_on_force_restart", True)
	)
	removed_on_force_restart: list[str] = []
	if bool(force_restart) and delete_on_force_restart and bombcell_out_dir.exists():
		removed_on_force_restart.append(str(bombcell_out_dir))
		shutil.rmtree(bombcell_out_dir, ignore_errors=True)
	bombcell_out_dir.mkdir(parents=True, exist_ok=True)

	enabled = bool(getattr(stage_config, "bombcell_label_enabled", False))
	if not enabled:
		payload = {
			"status": "skipped",
			"reason": "bombcell_label_disabled",
			"well_out_dir": str(well_out_dir),
			"stage_output_root_dir": str(stage_output_root_dir),
			"bombcell_out_dir": str(bombcell_out_dir),
			"force_restart": bool(force_restart),
			"delete_outputs_on_force_restart": bool(delete_on_force_restart),
			"removed_on_force_restart": list(removed_on_force_restart),
		}
		if summary_json is not None:
			_write_json(summary_json, payload)
		outputs: dict[str, str] = {}
		if summary_json is not None:
			outputs["bombcell_label.summary_json"] = str(summary_json)
		return {
			"name": "bombcell_label",
			"status": "skipped",
			"reason": "bombcell_label_disabled",
			"out_dir": str(bombcell_out_dir),
			"summary_json": (str(summary_json) if summary_json is not None else None),
			"outputs": outputs,
			"removed_on_force_restart": list(removed_on_force_restart),
		}

	_log_phase_step_start(
		"Bombcell label phase start",
		well_out_dir=well_out_dir,
		force_restart=bool(force_restart),
		out_dir=bombcell_out_dir,
	)

	resolved_sorter_output_dir = (
		Path(sorter_output_dir).resolve()
		if sorter_output_dir is not None
		else _resolve_sorter_output_dir(
			well_out_dir=well_out_dir,
			output_rel_root=output_rel_root,
			stage_config=stage_config,
		)
	)
	bombcell_workspace_info = _prepare_bombcell_sorter_output_workspace(
		stage_output_root_dir=stage_output_root_dir,
		bombcell_out_dir=bombcell_out_dir,
		sorter_output_dir=resolved_sorter_output_dir,
		stage_config=stage_config,
	)
	effective_sorter_output_dir = Path(
		bombcell_workspace_info.get("effective_sorter_output_dir", resolved_sorter_output_dir)
	).resolve()
	ks_dir = _normalize_slay_kilosort_dir(sorter_output_dir=effective_sorter_output_dir)
	canonical_ks_dir = _normalize_slay_kilosort_dir(sorter_output_dir=resolved_sorter_output_dir)

	labels_payload_json = bombcell_out_dir / "bombcell_labels.json"
	labels_payload_tsv = bombcell_out_dir / "bombcell_labels.tsv"

	payload: dict[str, Any] = {
		"status": "ok",
		"reason": None,
		"well_out_dir": str(well_out_dir),
		"stage_output_root_dir": str(stage_output_root_dir),
		"bombcell_out_dir": str(bombcell_out_dir),
		"force_restart": bool(force_restart),
		"delete_outputs_on_force_restart": bool(delete_on_force_restart),
		"removed_on_force_restart": list(removed_on_force_restart),
		"sorter_output_dir": str(effective_sorter_output_dir),
		"canonical_sorter_output_dir": str(resolved_sorter_output_dir),
		"ks_dir": str(ks_dir),
		"canonical_ks_dir": str(canonical_ks_dir),
		"cache_sorter_output_before_analyzer_gen": bool(
			bombcell_workspace_info.get("cache_sorter_output_before_analyzer_gen", False)
		),
		"publish_cached_sorter_output_on_success": bool(
			bombcell_workspace_info.get("publish_cached_sorter_output_on_success", False)
		),
		"publish_cached_analyzer_on_success": bool(
			bombcell_workspace_info.get("publish_cached_analyzer_on_success", False)
		),
		"cleanup_analyzer_on_success": bool(
			bombcell_workspace_info.get("cleanup_analyzer_on_success", False)
		),
		"cleanup_cached_sorter_output_on_success": bool(
			bombcell_workspace_info.get("cleanup_cached_sorter_output_on_success", False)
		),
		"label_non_somatic": bool(getattr(stage_config, "bombcell_label_label_non_somatic", True)),
		"split_non_somatic_good_mua": bool(
			getattr(stage_config, "bombcell_label_split_non_somatic_good_mua", True)
		),
		"apply_to_sorter_output": bool(getattr(stage_config, "bombcell_label_apply_to_sorter_output", True)),
		"write_cluster_group": bool(getattr(stage_config, "bombcell_label_write_cluster_group", True)),
	}
	cached_sorter_output_dir = bombcell_workspace_info.get("cached_sorter_output_dir", None)
	if isinstance(cached_sorter_output_dir, Path):
		payload["cached_sorter_output_dir"] = str(cached_sorter_output_dir)

	try:
		_log_phase_step_start(
			"Bombcell label analyzer metrics step start",
			well_out_dir=well_out_dir,
			ks_dir=ks_dir,
			canonical_ks_dir=(canonical_ks_dir if ks_dir != canonical_ks_dir else None),
		)
		si_module = _import_spikeinterface_full_module()
		analyzer, loaded_analyzer_dir, analyzer_rebuilt, analyzer_regen_reason = _load_or_recompute_bombcell_sorting_analyzer(
			si_module=si_module,
			well_out_dir=well_out_dir,
			bombcell_out_dir=bombcell_out_dir,
			sorter_output_dir=effective_sorter_output_dir,
			stage_config=stage_config,
		)
		payload["analyzer_dir"] = str(loaded_analyzer_dir)
		payload["analyzer_rebuilt"] = bool(analyzer_rebuilt)
		if analyzer_regen_reason is not None:
			payload["analyzer_regen_reason"] = str(analyzer_regen_reason)

		computed_extensions = _ensure_bombcell_metric_extensions(
			analyzer=analyzer,
			stage_config=stage_config,
		)
		if computed_extensions:
			payload["computed_extensions"] = list(computed_extensions)

		curation_module = importlib.import_module("spikeinterface.curation")
		bombcell_label_units = getattr(curation_module, "bombcell_label_units", None)
		if not callable(bombcell_label_units):
			raise RuntimeError("spikeinterface.curation.bombcell_label_units_unavailable")

		thresholds_arg: Any = None
		thresholds_dict = getattr(stage_config, "bombcell_label_thresholds", None)
		thresholds_path_raw = getattr(stage_config, "bombcell_label_thresholds_path", None)
		if isinstance(thresholds_dict, dict):
			thresholds_arg = dict(thresholds_dict)
		elif thresholds_path_raw is not None:
			thresholds_path = Path(str(thresholds_path_raw)).expanduser()
			if thresholds_path.is_absolute():
				thresholds_arg = str(thresholds_path)
			else:
				candidate = (bombcell_out_dir / thresholds_path).resolve()
				thresholds_arg = str(candidate if candidate.exists() else thresholds_path)

		labels_obj = bombcell_label_units(
			sorting_analyzer=analyzer,
			thresholds=thresholds_arg,
			label_non_somatic=bool(getattr(stage_config, "bombcell_label_label_non_somatic", True)),
			split_non_somatic_good_mua=bool(
				getattr(stage_config, "bombcell_label_split_non_somatic_good_mua", True)
			),
			external_metrics=None,
		)
		labels_by_unit = _extract_bombcell_label_mapping(labels_obj)
		if not labels_by_unit:
			raise RuntimeError("bombcell_labels_empty")

		counts_by_label: dict[str, int] = {}
		for label in labels_by_unit.values():
			counts_by_label[label] = int(counts_by_label.get(label, 0)) + 1

		labels_payload: dict[str, Any] = {
			"n_units_labeled": int(len(labels_by_unit)),
			"counts_by_label": dict(sorted(counts_by_label.items(), key=lambda kv: kv[0])),
			"labels_by_unit": dict(sorted(labels_by_unit.items(), key=lambda kv: _unit_sort_key(kv[0]))),
		}

		_write_json(labels_payload_json, labels_payload)
		_write_kilosort_cluster_labels_tsv(
			path=labels_payload_tsv,
			label_column="label",
			labels_by_unit=labels_by_unit,
		)

		payload["n_units_labeled"] = int(len(labels_by_unit))
		payload["counts_by_label"] = dict(labels_payload["counts_by_label"])
		payload["labels_json"] = str(labels_payload_json)
		payload["labels_tsv"] = str(labels_payload_tsv)

		sorter_label_update: dict[str, Any] | None = None
		if bool(getattr(stage_config, "bombcell_label_apply_to_sorter_output", True)):
			_log_phase_step_start(
				"Bombcell label sorter writeback step start",
				well_out_dir=well_out_dir,
				ks_dir=ks_dir,
				canonical_ks_dir=(canonical_ks_dir if ks_dir != canonical_ks_dir else None),
			)
			sorter_label_update = _apply_bombcell_labels_to_kilosort_outputs(
				ks_dir=ks_dir,
				bombcell_labels_by_unit=labels_by_unit,
				write_cluster_group=bool(getattr(stage_config, "bombcell_label_write_cluster_group", True)),
			)
			payload["sorter_label_update"] = dict(sorter_label_update)

		published_workspace_outputs = _publish_bombcell_cached_workspace_outputs(
			workspace_info=bombcell_workspace_info,
			analyzer_dir=Path(loaded_analyzer_dir),
		)
		if published_workspace_outputs:
			payload["published_workspace_outputs"] = dict(published_workspace_outputs)

		released_extensions = _release_loaded_analyzer_extensions(analyzer=analyzer)
		if released_extensions:
			payload["released_analyzer_extensions"] = list(released_extensions)
		analyzer = None
		gc.collect()

		removed_on_success = _cleanup_bombcell_success_outputs(
			workspace_info=bombcell_workspace_info,
			analyzer_dir=Path(loaded_analyzer_dir),
		)
		if removed_on_success:
			payload["removed_on_success"] = dict(removed_on_success)

		if summary_json is not None:
			_write_json(summary_json, payload)

		outputs: dict[str, str] = {
			"bombcell_label.labels_json": str(labels_payload_json),
			"bombcell_label.labels_tsv": str(labels_payload_tsv),
		}
		if summary_json is not None:
			outputs["bombcell_label.summary_json"] = str(summary_json)
		if isinstance(sorter_label_update, dict):
			cluster_kslabel_tsv = sorter_label_update.get("cluster_kslabel_tsv", None)
			cluster_group_tsv = sorter_label_update.get("cluster_group_tsv", None)
			if cluster_kslabel_tsv:
				outputs["bombcell_label.cluster_kslabel_tsv"] = str(cluster_kslabel_tsv)
			if cluster_group_tsv:
				outputs["bombcell_label.cluster_group_tsv"] = str(cluster_group_tsv)

		return {
			"name": "bombcell_label",
			"status": "ok",
			"reason": None,
			"out_dir": str(bombcell_out_dir),
			"analyzer_dir": str(loaded_analyzer_dir),
			"summary_json": (str(summary_json) if summary_json is not None else None),
			"outputs": outputs,
			"sorter_output_dir": str(effective_sorter_output_dir),
			"canonical_sorter_output_dir": str(resolved_sorter_output_dir),
			"ks_dir": str(ks_dir),
			"canonical_ks_dir": str(canonical_ks_dir),
			"n_units_labeled": int(len(labels_by_unit)),
			"counts_by_label": dict(payload.get("counts_by_label", {})),
			"removed_on_force_restart": list(removed_on_force_restart),
			"removed_on_success": dict(payload.get("removed_on_success", {})),
		}
	except Exception as exc:
		payload["status"] = "error"
		payload["reason"] = "bombcell_label_failed"
		payload["error"] = f"bombcell_label_failed:{type(exc).__name__}:{exc}"
		if summary_json is not None:
			_write_json(summary_json, payload)
		outputs: dict[str, str] = {}
		if summary_json is not None:
			outputs["bombcell_label.summary_json"] = str(summary_json)
		return {
			"name": "bombcell_label",
			"status": "error",
			"reason": "bombcell_label_failed",
			"error": str(payload["error"]),
			"out_dir": str(bombcell_out_dir),
			"analyzer_dir": payload.get("analyzer_dir", None),
			"summary_json": (str(summary_json) if summary_json is not None else None),
			"outputs": outputs,
			"sorter_output_dir": str(effective_sorter_output_dir),
			"canonical_sorter_output_dir": str(resolved_sorter_output_dir),
			"ks_dir": str(ks_dir),
			"canonical_ks_dir": str(canonical_ks_dir),
			"removed_on_force_restart": list(removed_on_force_restart),
		}


def run_spikesort_bombcell_label_stage(
	*,
	h5_path: Path,
	stream_id: str,
	mea_output_root: Path,
	output_rel_root: str,
	stage_config: Any,
	force_restart: bool,
) -> SpikesortBombcellResult:
	well_out_dir = compute_mea_analysis_output_dir(
		output_root=mea_output_root,
		data_file=h5_path,
		well=stream_id,
	)
	stage_output_root_dir = _resolve_under_well(
		well_out_dir=well_out_dir,
		relpath=str(output_rel_root).strip() or "spikesort_outputs",
	)
	stage_output_root_dir.mkdir(parents=True, exist_ok=True)
	report = _run_bombcell_label_phase(
		well_out_dir=well_out_dir,
		stage_output_root_dir=stage_output_root_dir,
		output_rel_root=output_rel_root,
		stage_config=stage_config,
		force_restart=force_restart,
	)
	bombcell_out_dir = Path(str(report.get("out_dir") or stage_output_root_dir)).resolve()
	summary_json_raw = report.get("summary_json", None)
	summary_json = (
		Path(str(summary_json_raw)).resolve()
		if summary_json_raw is not None
		else None
	)
	return SpikesortBombcellResult(
		well_out_dir=well_out_dir,
		bombcell_out_dir=bombcell_out_dir,
		summary_json=summary_json,
		outputs=dict(report.get("outputs", {}) or {}),
	)


def _assert_method_uses_working_cache_sorter_output(
	*,
	method_name: str,
	sorter_output_dir: Path | None,
	working_cache_root_dir: Path,
	knob_name: str,
) -> None:
	if sorter_output_dir is None:
		raise RuntimeError(
			f"{method_name} expected sorter_output_dir in working cache '{working_cache_root_dir.resolve()}', "
			"but no sorter_output_dir was resolved. "
			f"Disable this assertion with stages.spikesort.phases.merge_units.working_cache.{knob_name}=false."
		)
	resolved_sorter_dir = Path(sorter_output_dir).resolve()
	resolved_working_cache_root = working_cache_root_dir.resolve()
	try:
		resolved_sorter_dir.relative_to(resolved_working_cache_root)
	except ValueError as exc:
		raise RuntimeError(
			f"{method_name} expected working cache sorter output under '{resolved_working_cache_root}', "
			f"got '{resolved_sorter_dir}'. "
			f"Disable this assertion with stages.spikesort.phases.merge_units.working_cache.{knob_name}=false."
		) from exc


def _import_spikeinterface_full_module() -> Any:
	try:
		import spikeinterface.full as si  # type: ignore[import-not-found]

		return si
	except Exception:
		import spikeinterface as si  # type: ignore[import-not-found]

		return si


def _load_preprocessed_recording_from_dir(*, si_module: Any, recording_dir: Path) -> Any:
	if not recording_dir.exists():
		raise FileNotFoundError(f"Preprocessed recording not found: {recording_dir}")
	try:
		return si_module.load(recording_dir)
	except Exception:
		return si_module.load_extractor(recording_dir)


def _resolve_bootstrap_concat_binary_paths(
	*,
	well_out_dir: Path,
	stage_output_root_dir: Path,
	stage_config: Any,
) -> dict[str, Path]:
	cache_root_dir = _resolve_under_spikesort_output_root(
		well_out_dir=well_out_dir,
		output_rel_root=str(getattr(stage_config, "output_rel_root", "spikesort_outputs") or "spikesort_outputs"),
		relpath=str(getattr(stage_config, "bootstrap_concat_binary_cache_relpath", "cache/bootstrap_concat_binary")),
	)
	recording_dir = _resolve_under_spikesort_output_root(
		well_out_dir=well_out_dir,
		output_rel_root=str(getattr(stage_config, "output_rel_root", "spikesort_outputs") or "spikesort_outputs"),
		relpath=str(getattr(stage_config, "bootstrap_concat_binary_recording_relpath", "cache/bootstrap_concat_binary/recording")),
	)
	concat_manifest_path = _resolve_under_spikesort_output_root(
		well_out_dir=well_out_dir,
		output_rel_root=str(getattr(stage_config, "output_rel_root", "spikesort_outputs") or "spikesort_outputs"),
		relpath=str(getattr(stage_config, "bootstrap_concat_binary_manifest_relpath", "cache/bootstrap_concat_binary/concat_segments_manifest.json")),
	)
	summary_json = _resolve_under_spikesort_output_root(
		well_out_dir=well_out_dir,
		output_rel_root=str(getattr(stage_config, "output_rel_root", "spikesort_outputs") or "spikesort_outputs"),
		relpath=str(getattr(stage_config, "bootstrap_concat_binary_summary_json_relpath", "cache/bootstrap_concat_binary/bootstrap_concat_binary_summary.json")),
	)
	segment_manifest_path = _resolve_under_well(
		well_out_dir=well_out_dir,
		relpath=str(
			getattr(
				stage_config,
				"bootstrap_concat_binary_source_segment_manifest_relpath",
				"preprocess_outputs/preprocessed_segments/manifest.json",
			)
		),
	)
	return {
		"cache_root_dir": cache_root_dir,
		"recording_dir": recording_dir,
		"concat_manifest_path": concat_manifest_path,
		"summary_json": summary_json,
		"segment_manifest_path": segment_manifest_path,
		"stage_output_root_dir": stage_output_root_dir,
	}


def run_spikesort_bootstrap_concat_binary_stage(
	*,
	h5_path: Path,
	stream_id: str,
	mea_output_root: Path,
	output_rel_root: str,
	stage_config: Any,
	force_restart: bool,
) -> SpikesortResult:
	well_out_dir = compute_mea_analysis_output_dir(
		output_root=mea_output_root,
		data_file=h5_path,
		well=stream_id,
	)
	stage_output_root_dir = _resolve_under_well(
		well_out_dir=well_out_dir,
		relpath=(str(output_rel_root).strip() or "spikesort_outputs"),
	)
	stage_output_root_dir.mkdir(parents=True, exist_ok=True)
	paths = _resolve_bootstrap_concat_binary_paths(
		well_out_dir=well_out_dir,
		stage_output_root_dir=stage_output_root_dir,
		stage_config=stage_config,
	)
	summary_json = paths["summary_json"]

	if not bool(getattr(stage_config, "bootstrap_concat_binary_enabled", False)):
		payload = {
			"status": "skipped",
			"reason": "bootstrap_concat_binary_disabled",
			"well_out_dir": str(well_out_dir),
			"stage_output_root_dir": str(stage_output_root_dir),
			"outputs": {"summary_json": str(summary_json)},
		}
		_write_json(summary_json, payload)
		return SpikesortResult(
			well_out_dir=well_out_dir,
			spikesort_out_dir=stage_output_root_dir,
			summary_json=summary_json,
			outputs={"summary_json": str(summary_json)},
		)

	from axon_recon.pipeline.stages.preprocess.core.artifacts import load_common_electrodes
	from axon_recon.pipeline.stages.preprocess.core.concat_segments import run_concat_segments_core
	from axon_recon.pipeline.stages.preprocess.core.save_concatenated_recording import (
		run_save_concatenated_recording_core,
	)

	overwrite_saved_recording = bool(getattr(stage_config, "bootstrap_concat_binary_overwrite_existing", False))
	if bool(force_restart) and bool(getattr(stage_config, "bootstrap_concat_binary_overwrite_on_force_restart", True)):
		overwrite_saved_recording = True
	n_jobs = getattr(stage_config, "bootstrap_concat_binary_n_jobs", None)
	if n_jobs is None:
		n_jobs = getattr(stage_config, "n_jobs", None)
	if n_jobs is None:
		n_jobs = 1
	chunk_duration = (
		getattr(stage_config, "bootstrap_concat_binary_chunk_duration", None)
		or getattr(stage_config, "chunk_duration", None)
		or "1s"
	)

	_log_phase_step_start(
		"Spikesort bootstrap concat binary step start",
		stream_id=str(stream_id),
		segment_manifest_path=paths["segment_manifest_path"],
		recording_dir=paths["recording_dir"],
		force_restart=bool(force_restart),
		overwrite_saved_recording=bool(overwrite_saved_recording),
	)
	common_electrodes_path = _resolve_under_well(
		well_out_dir=well_out_dir,
		relpath=str(
			getattr(
				stage_config,
				"bootstrap_concat_binary_source_common_electrodes_relpath",
				"preprocess_outputs/common_electrodes.npy",
			)
		),
	)
	common_electrodes_for_concat: list[int] = []
	try:
		common_electrodes_for_concat = load_common_electrodes(common_electrodes_path)
	except FileNotFoundError:
		LOGGER.info(
			"Spikesort bootstrap_concat_binary: common-electrode artifact not found at %s; concat will use each segment's full channel set",
			str(common_electrodes_path),
		)
	payload = run_concat_segments_core(
		stream_id=str(stream_id),
		segment_manifest_path=paths["segment_manifest_path"],
		recording_dir=paths["recording_dir"],
		concat_manifest_path=paths["concat_manifest_path"],
		overwrite_saved_recording=bool(overwrite_saved_recording),
		output_mode="binary",
		n_jobs=max(1, int(n_jobs)),
		chunk_duration=str(chunk_duration),
		progress_bar=bool(getattr(stage_config, "bootstrap_concat_binary_progress_bar", True)),
		logger=LOGGER,
		run_save_concatenated_recording_core=run_save_concatenated_recording_core,
		common_electrodes=common_electrodes_for_concat,
	)
	binary_candidates = [
		path
		for pattern in ("traces_cached_seg*.raw", "*.raw", "recording.dat")
		for path in paths["recording_dir"].glob(pattern)
		if path.is_file()
	]
	if not binary_candidates:
		raise RuntimeError(
			"bootstrap_concat_binary did not produce a materialized binary recording: "
			f"recording_dir={paths['recording_dir']} reused_existing={payload.get('reused_existing', False)}. "
			"Rerun with --force-restart or set overwrite_existing=true."
		)
	outputs = {
		"summary_json": str(summary_json),
		"bootstrap_concat_binary.recording_dir": str(paths["recording_dir"]),
		"bootstrap_concat_binary.concat_manifest_path": str(paths["concat_manifest_path"]),
		"bootstrap_concat_binary.cache_root_dir": str(paths["cache_root_dir"]),
	}
	if "recording_json_path" in payload:
		outputs["bootstrap_concat_binary.recording_json_path"] = str(payload["recording_json_path"])
	_write_json(
		summary_json,
		{
			"status": "ok",
			"well_out_dir": str(well_out_dir),
			"stage_output_root_dir": str(stage_output_root_dir),
			"cache_root_dir": str(paths["cache_root_dir"]),
			"recording_dir": str(paths["recording_dir"]),
			"segment_manifest_path": str(paths["segment_manifest_path"]),
			"concat_manifest_path": str(paths["concat_manifest_path"]),
			"force_restart": bool(force_restart),
			"overwrite_saved_recording": bool(overwrite_saved_recording),
			"output_mode": str(payload.get("output_mode", "binary")),
			"materialized_recording": bool(payload.get("materialized_recording", True)),
			"saved": bool(payload.get("saved", False)),
			"reused_existing": bool(payload.get("reused_existing", False)),
			"outputs": outputs,
		},
	)
	return SpikesortResult(
		well_out_dir=well_out_dir,
		spikesort_out_dir=stage_output_root_dir,
		summary_json=summary_json,
		outputs=outputs,
	)


def run_spikesort_cleanup_concat_binary_stage(
	*,
	h5_path: Path,
	stream_id: str,
	mea_output_root: Path,
	output_rel_root: str,
	stage_config: Any,
	force_restart: bool,
) -> SpikesortResult:
	well_out_dir = compute_mea_analysis_output_dir(
		output_root=mea_output_root,
		data_file=h5_path,
		well=stream_id,
	)
	stage_output_root_dir = _resolve_under_well(
		well_out_dir=well_out_dir,
		relpath=(str(output_rel_root).strip() or "spikesort_outputs"),
	)
	stage_output_root_dir.mkdir(parents=True, exist_ok=True)
	summary_json = _resolve_under_spikesort_output_root(
		well_out_dir=well_out_dir,
		output_rel_root=output_rel_root,
		relpath=str(getattr(stage_config, "cleanup_concat_binary_summary_json_relpath", "cache/bootstrap_concat_binary_cleanup_summary.json")),
	)
	target_dir = _resolve_under_spikesort_output_root(
		well_out_dir=well_out_dir,
		output_rel_root=output_rel_root,
		relpath=str(getattr(stage_config, "cleanup_concat_binary_relpath", "cache/bootstrap_concat_binary")),
	)
	removed_paths: list[str] = []
	if bool(getattr(stage_config, "cleanup_concat_binary_enabled", False)) and target_dir.exists():
		if target_dir.is_dir():
			shutil.rmtree(target_dir, ignore_errors=True)
		else:
			target_dir.unlink(missing_ok=True)
		removed_paths.append(str(target_dir))
	status = "ok" if bool(getattr(stage_config, "cleanup_concat_binary_enabled", False)) else "skipped"
	reason = None if status == "ok" else "cleanup_concat_binary_disabled"
	outputs = {"summary_json": str(summary_json)}
	_write_json(
		summary_json,
		{
			"status": status,
			"reason": reason,
			"well_out_dir": str(well_out_dir),
			"stage_output_root_dir": str(stage_output_root_dir),
			"target_dir": str(target_dir),
			"removed_paths": removed_paths,
			"force_restart": bool(force_restart),
			"outputs": outputs,
		},
	)
	return SpikesortResult(
		well_out_dir=well_out_dir,
		spikesort_out_dir=stage_output_root_dir,
		summary_json=summary_json,
		outputs=outputs,
	)


def _spikesort_preprocessed_recording_relpath(stage_config: Any) -> str:
	return str(
		getattr(stage_config, "preprocess_concat_recording_relpath", None)
		or "preprocess_outputs/preprocessed_recording"
	)


def _try_load_spikesort_preprocessed_recording(*, si_module: Any, well_out_dir: Path, stage_config: Any) -> Any | None:
	recording_relpath = _spikesort_preprocessed_recording_relpath(stage_config)
	recording_dir = _resolve_under_well(well_out_dir=well_out_dir, relpath=recording_relpath)
	try:
		return _load_preprocessed_recording_from_dir(si_module=si_module, recording_dir=recording_dir)
	except Exception:
		return None


def _attach_temporary_recording_to_spikesort_analyzer_if_missing(*, analyzer: Any, recording: Any | None) -> Any:
	if analyzer is None or recording is None:
		return analyzer
	try:
		if callable(getattr(analyzer, "has_recording", None)) and bool(analyzer.has_recording()):
			return analyzer
	except Exception:
		pass
	try:
		if callable(getattr(analyzer, "has_temporary_recording", None)) and bool(analyzer.has_temporary_recording()):
			return analyzer
	except Exception:
		pass
	setter = getattr(analyzer, "set_temporary_recording", None)
	if not callable(setter):
		return analyzer
	try:
		setter(recording)
	except Exception:
		LOGGER.debug("Failed to attach temporary recording to spikesort analyzer", exc_info=True)
	return analyzer


def _load_spikesort_analyzer_with_recording(
	*,
	si_module: Any,
	analyzer_dir: Path,
	well_out_dir: Path,
	stage_config: Any,
) -> Any:
	load_sorting_analyzer = getattr(si_module, "load_sorting_analyzer", None)
	if not callable(load_sorting_analyzer):
		raise RuntimeError("load_sorting_analyzer unavailable")
	analyzer = load_sorting_analyzer(analyzer_dir)
	recording = _try_load_spikesort_preprocessed_recording(
		si_module=si_module,
		well_out_dir=well_out_dir,
		stage_config=stage_config,
	)
	return _attach_temporary_recording_to_spikesort_analyzer_if_missing(
		analyzer=analyzer,
		recording=recording,
	)


def _is_kilosort_folder(folder: Path) -> bool:
	return bool((folder / "spike_times.npy").exists() and (folder / "spike_clusters.npy").exists())


def _load_kilosort_unit_ids_from_spike_clusters(*, folder: Path) -> list[str]:
	try:
		import numpy as np
	except Exception:
		return []

	spike_clusters_path = (folder / "spike_clusters.npy").resolve()
	if not spike_clusters_path.exists():
		return []

	try:
		labels = np.load(spike_clusters_path, mmap_mode="r")
	except Exception:
		return []

	try:
		labels = labels.reshape(-1)
	except Exception:
		return []

	try:
		unique_labels = np.unique(labels)
	except Exception:
		return []

	out: list[str] = []
	for raw in unique_labels.tolist() if hasattr(unique_labels, "tolist") else list(unique_labels):
		token = _normalize_cluster_id(raw)
		if token and token not in out:
			out.append(token)
	return sorted(out, key=_unit_sort_key)


def _build_numpy_sorting_from_kilosort_raw(
	*,
	si_module: Any,
	folder: Path,
	sampling_frequency: float,
) -> Any:
	try:
		import numpy as np
	except Exception as exc:
		raise RuntimeError(f"numpy_import_failed:{type(exc).__name__}:{exc}") from exc

	numpy_sorting_cls = getattr(si_module, "NumpySorting", None)
	from_times_labels = getattr(numpy_sorting_cls, "from_times_labels", None)
	if not callable(from_times_labels):
		raise RuntimeError("spikeinterface.NumpySorting.from_times_labels_unavailable")

	spike_times_path = (folder / "spike_times.npy").resolve()
	spike_clusters_path = (folder / "spike_clusters.npy").resolve()
	if (not spike_times_path.exists()) or (not spike_clusters_path.exists()):
		raise RuntimeError("kilosort_spike_arrays_missing")

	times = np.load(spike_times_path).reshape(-1)
	labels = np.load(spike_clusters_path).reshape(-1)
	if int(times.shape[0]) != int(labels.shape[0]):
		raise RuntimeError(
			"kilosort_spike_arrays_length_mismatch:"
			f"times={int(times.shape[0])},labels={int(labels.shape[0])}"
		)

	unit_ids = sorted(
		[
			int(token)
			for token in _load_kilosort_unit_ids_from_spike_clusters(folder=folder)
			if str(token).isdigit()
		],
	)

	return from_times_labels(
		times_list=[times],
		labels_list=[labels],
		sampling_frequency=float(sampling_frequency),
		unit_ids=unit_ids,
	)


def _try_load_kilosort_sorting_with_full_unit_ids(
	*,
	si_module: Any,
	folder: Path,
) -> Any | None:
	if not _is_kilosort_folder(folder):
		return None

	read_kilosort = getattr(si_module, "read_kilosort", None)
	if not callable(read_kilosort):
		return None

	sorting_obj: Any | None = None
	for kwargs in (
		{"keep_good_only": False, "remove_empty_units": False},
		{"keep_good_only": False},
		{},
	):
		try:
			sorting_obj = read_kilosort(folder, **kwargs)
			break
		except TypeError:
			continue
		except Exception:
			continue

	if sorting_obj is None:
		return None

	raw_unit_ids = _load_kilosort_unit_ids_from_spike_clusters(folder=folder)
	loaded_unit_ids = _unit_ids_from_obj(sorting_obj)
	if not raw_unit_ids:
		return sorting_obj

	missing_ids = sorted(list(set(raw_unit_ids) - set(loaded_unit_ids)), key=_unit_sort_key)
	if not missing_ids:
		return sorting_obj

	get_sampling_frequency = getattr(sorting_obj, "get_sampling_frequency", None)
	if not callable(get_sampling_frequency):
		return sorting_obj

	try:
		sampling_frequency = float(get_sampling_frequency())
	except Exception:
		return sorting_obj

	try:
		return _build_numpy_sorting_from_kilosort_raw(
			si_module=si_module,
			folder=folder,
			sampling_frequency=sampling_frequency,
		)
	except Exception:
		return sorting_obj


def _load_sorting_from_sorter_output_dir(*, si_module: Any, sorter_output_dir: Path, sorter_name: str) -> Any:
	candidates: list[Path] = []
	for candidate in (
		sorter_output_dir,
		(sorter_output_dir / "sorter_output").resolve(),
		(sorter_output_dir / "in_container_sorting").resolve(),
		(sorter_output_dir.parent if sorter_output_dir.name == "sorter_output" else sorter_output_dir).resolve(),
	):
		if candidate in candidates:
			continue
		if not candidate.exists():
			continue
		candidates.append(candidate)

	sorter_name_token = str(sorter_name or "").lower()
	if "kilosort" in sorter_name_token:
		for candidate in candidates:
			kilosort_sorting = _try_load_kilosort_sorting_with_full_unit_ids(
				si_module=si_module,
				folder=candidate,
			)
			if kilosort_sorting is not None:
				return kilosort_sorting

	if hasattr(si_module, "read_sorter_folder"):
		for candidate in candidates:
			try:
				return si_module.read_sorter_folder(candidate)
			except TypeError:
				pass
			except Exception:
				pass
			try:
				return si_module.read_sorter_folder(candidate, sorter_name=str(sorter_name))
			except TypeError:
				try:
					return si_module.read_sorter_folder(candidate, str(sorter_name))
				except Exception:
					pass
			except Exception:
				pass

	for candidate in candidates:
		try:
			return si_module.load_extractor(candidate)
		except Exception:
			pass

	raise RuntimeError(
		"Could not load sorting from sorter output directory: "
		f"{sorter_output_dir}; tried={','.join(str(p) for p in candidates)}"
	)


def _recompute_spikesort_analyzer(
	*,
	si_module: Any,
	well_out_dir: Path,
	stage_output_root_dir: Path,
	sorter_output_dir: Path,
	stage_config: Any,
) -> tuple[Any, Path]:
	return _recompute_sorting_analyzer_to_dir(
		si_module=si_module,
		well_out_dir=well_out_dir,
		sorter_output_dir=sorter_output_dir,
		stage_config=stage_config,
		analyzer_dir=(stage_output_root_dir / "analyzer_output").resolve(),
	)


def _recompute_sorting_analyzer_to_dir(
	*,
	si_module: Any,
	well_out_dir: Path,
	sorter_output_dir: Path,
	stage_config: Any,
	analyzer_dir: Path,
) -> tuple[Any, Path]:
	recording_relpath = _spikesort_preprocessed_recording_relpath(stage_config)
	recording_dir = _resolve_under_well(well_out_dir=well_out_dir, relpath=recording_relpath)
	recording = _load_preprocessed_recording_from_dir(si_module=si_module, recording_dir=recording_dir)
	sorting = _load_sorting_from_sorter_output_dir(
		si_module=si_module,
		sorter_output_dir=sorter_output_dir,
		sorter_name=str(getattr(stage_config, "sorter", "kilosort4") or "kilosort4"),
	)

	analyzer_dir = Path(analyzer_dir).resolve()
	if analyzer_dir.exists():
		shutil.rmtree(analyzer_dir, ignore_errors=True)

	create_sorting_analyzer = getattr(si_module, "create_sorting_analyzer", None)
	if not callable(create_sorting_analyzer):
		raise RuntimeError("spikeinterface.create_sorting_analyzer is required for analyzer recomputation")

	create_kwargs = {
		"sorting": sorting,
		"recording": recording,
		"format": "binary_folder",
		"folder": analyzer_dir,
	}
	create_kwargs.update(_merge_analyzer_sparsity_create_kwargs(stage_config))
	if _merge_dense_analyzer_requested(stage_config):
		create_kwargs["sparse"] = False

	try:
		analyzer = create_sorting_analyzer(**create_kwargs)
	except TypeError:
		if "sparse" not in create_kwargs:
			raise
		create_kwargs.pop("sparse", None)
		analyzer = create_sorting_analyzer(**create_kwargs)
	analyzer = _attach_temporary_recording_to_spikesort_analyzer_if_missing(
		analyzer=analyzer,
		recording=recording,
	)

	policy_info = _describe_merge_analyzer_policy_info(
		analyzer=analyzer,
		stage_config=stage_config,
		reused_cached_analyzer=False,
	)
	_attach_merge_analyzer_policy_info(analyzer, policy_info)
	_write_merge_analyzer_policy_info_to_dir(analyzer_dir=analyzer_dir, policy_info=policy_info)
	return analyzer, analyzer_dir


def _load_or_recompute_spikesort_analyzer(
	*,
	si_module: Any,
	well_out_dir: Path,
	stage_output_root_dir: Path,
	sorter_output_dir: Path,
	stage_config: Any,
) -> tuple[Any, Path, bool]:
	analyzer_dir = (stage_output_root_dir / "analyzer_output").resolve()
	load_sorting_analyzer = getattr(si_module, "load_sorting_analyzer", None)
	loaded_analyzer_had_sparsity: bool | None = None
	rebuild_reason: str | None = None
	if analyzer_dir.exists() and callable(load_sorting_analyzer):
		try:
			analyzer = _load_spikesort_analyzer_with_recording(
				si_module=si_module,
				analyzer_dir=analyzer_dir,
				well_out_dir=well_out_dir,
				stage_config=stage_config,
			)
			loaded_analyzer_had_sparsity = _analyzer_has_sparsity(analyzer)
			if _merge_dense_analyzer_requested(stage_config) and loaded_analyzer_had_sparsity:
				rebuild_reason = "loaded_sparse_analyzer"
			else:
				policy_info = _describe_merge_analyzer_policy_info(
					analyzer=analyzer,
					stage_config=stage_config,
					loaded_analyzer_had_sparsity=loaded_analyzer_had_sparsity,
					reused_cached_analyzer=True,
				)
				_attach_merge_analyzer_policy_info(analyzer, policy_info)
				return analyzer, analyzer_dir, False
		except Exception:
			pass

	analyzer, rebuilt_dir = _recompute_spikesort_analyzer(
		si_module=si_module,
		well_out_dir=well_out_dir,
		stage_output_root_dir=stage_output_root_dir,
		sorter_output_dir=sorter_output_dir,
		stage_config=stage_config,
	)
	policy_info = _describe_merge_analyzer_policy_info(
		analyzer=analyzer,
		stage_config=stage_config,
		loaded_analyzer_had_sparsity=loaded_analyzer_had_sparsity,
		rebuild_reason=rebuild_reason,
		reused_cached_analyzer=False,
	)
	_attach_merge_analyzer_policy_info(analyzer, policy_info)
	return analyzer, rebuilt_dir, True


def _normalize_merge_groups(raw_groups: Any) -> list[list[str]]:
	if not isinstance(raw_groups, (list, tuple, set)):
		return []
	out: list[list[str]] = []
	for raw_group in raw_groups:
		if not isinstance(raw_group, (list, tuple, set)):
			continue
		members: list[str] = []
		for raw_member in raw_group:
			token = _normalize_cluster_id(raw_member)
			if token and token not in members:
				members.append(token)
		if len(members) >= 2:
			out.append(members)
	return out


def _unit_count(obj: Any) -> int:
	for source in _iter_unit_sources(obj):
		get_num_units = getattr(source, "get_num_units", None)
		if callable(get_num_units):
			try:
				count = int(get_num_units())
				if count >= 0:
					return int(count)
			except Exception:
				pass

		get_unit_ids = getattr(source, "get_unit_ids", None)
		if callable(get_unit_ids):
			try:
				return int(len(_coerce_items(get_unit_ids())))
			except Exception:
				pass

		unit_ids = getattr(source, "unit_ids", None)
		if unit_ids is not None:
			try:
				return int(len(unit_ids))
			except Exception:
				pass

	return 0


def _compute_auto_merge_groups(*, sorting_analyzer: Any, template_diff_thresh: float) -> list[list[str]]:
	curation_module = importlib.import_module("spikeinterface.curation")
	compute_groups = getattr(curation_module, "compute_merge_unit_groups", None)
	if callable(compute_groups):
		raw_groups = compute_groups(
			sorting_analyzer,
			preset="similarity_correlograms",
			resolve_graph=True,
			steps_params={
				"template_similarity": {
					"template_diff_thresh": float(template_diff_thresh),
				}
			},
			compute_needed_extensions=True,
			force_copy=False,
		)
		return _normalize_merge_groups(raw_groups)

	legacy_compute = getattr(curation_module, "get_potential_auto_merge", None)
	if callable(legacy_compute):
		raw_groups = legacy_compute(
			sorting_analyzer,
			preset="similarity_correlograms",
			resolve_graph=True,
			template_diff_thresh=float(template_diff_thresh),
		)
		return _normalize_merge_groups(raw_groups)

	raise RuntimeError("spikeinterface.curation auto-merge APIs are unavailable")


def _build_auto_merge_pair_rows(
	*,
	merge_groups: list[list[str]],
	iteration_index: int,
	template_diff_thresh: float,
) -> list[dict[str, str]]:
	rows: list[dict[str, str]] = []
	for group_idx, members in enumerate(merge_groups, start=1):
		members_sorted = sorted(members, key=lambda token: int(token) if token.isdigit() else token)
		group_id = f"group_{group_idx:03d}"
		group_members = "|".join(members_sorted)
		for cluster_a, cluster_b in itertools.combinations(members_sorted, 2):
			rows.append(
				{
					"iteration": str(int(iteration_index)),
					"template_diff_thresh": f"{float(template_diff_thresh):.12g}",
					"group_id": str(group_id),
					"group_size": str(len(members_sorted)),
					"group_members": str(group_members),
					"cluster_a": str(cluster_a),
					"cluster_b": str(cluster_b),
				}
			)
	return rows


def _write_auto_merge_candidate_pairs_tsv(path: Path, rows: list[dict[str, str]]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	fieldnames = [
		"iteration",
		"template_diff_thresh",
		"group_id",
		"group_size",
		"group_members",
		"cluster_a",
		"cluster_b",
	]
	with path.open("w", encoding="utf-8", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
		writer.writeheader()
		for row in rows:
			writer.writerow(row)


def _unit_sort_key(token: str) -> tuple[int, Any]:
	try:
		return (0, int(str(token)))
	except Exception:
		return (1, str(token))


def _coerce_items(value: Any) -> list[Any]:
	if value is None:
		return []
	try:
		return list(value)
	except Exception:
		return []


def _iter_unit_sources(obj: Any):
	seen: set[int] = set()
	for source in (obj, getattr(obj, "sorting", None)):
		if source is None:
			continue
		source_id = id(source)
		if source_id in seen:
			continue
		seen.add(source_id)
		yield source

	get_sorting = getattr(obj, "get_sorting", None)
	if callable(get_sorting):
		try:
			source = get_sorting()
		except Exception:
			source = None
		if source is not None:
			source_id = id(source)
			if source_id not in seen:
				seen.add(source_id)
				yield source


def _unit_ids_from_obj(obj: Any) -> list[str]:
	for source in _iter_unit_sources(obj):
		items: list[Any] = []
		get_unit_ids = getattr(source, "get_unit_ids", None)
		if callable(get_unit_ids):
			try:
				items = _coerce_items(get_unit_ids())
			except Exception:
				items = []
		if not items:
			unit_ids = getattr(source, "unit_ids", None)
			try:
				items = _coerce_items(unit_ids)
			except Exception:
				items = []

		if not items:
			continue

		out: list[str] = []
		for raw in items:
			token = _normalize_cluster_id(raw)
			if token and token not in out:
				out.append(token)
		if out:
			return out

	return []


def _snapshot_unit_count(payload: dict[str, Any], fallback_ids: list[str]) -> int:
	raw_count = payload.get("unit_count", None)
	try:
		count = int(raw_count)
		if count >= 0:
			return int(count)
	except Exception:
		pass
	return int(len(fallback_ids))


def _extract_unit_locations_from_analyzer(
	*,
	analyzer: Any,
	stage_config: Any | None = None,
) -> tuple[dict[str, dict[str, float]], str | None]:
	has_extension = getattr(analyzer, "has_extension", None)
	get_extension = getattr(analyzer, "get_extension", None)
	compute_extension = getattr(analyzer, "compute", None)
	if not callable(has_extension) or not callable(get_extension):
		return {}, "unit_locations_extension_api_unavailable"

	def _try_compute(extension_input: Any) -> tuple[bool, str | None]:
		if not callable(compute_extension):
			return False, "compute_api_unavailable"

		extension_name: str | None = None
		if isinstance(extension_input, str):
			extension_name = str(extension_input)
		elif isinstance(extension_input, (list, tuple)) and len(extension_input) == 1:
			extension_name = str(extension_input[0])

		extension_kwargs = (
			_merge_analyzer_extension_kwargs(stage_config, extension_name)
			if (stage_config is not None and extension_name is not None)
			else (_merge_analyzer_compute_job_kwargs(stage_config) if stage_config is not None else {})
		)

		try:
			if extension_name is None:
				compute_extension(extension_input)
				return True, None
			if _compute_analyzer_extension(
				analyzer=analyzer,
				extension_name=extension_name,
				kwargs=extension_kwargs,
			):
				return True, None
			return False, "compute_attempt_failed"
		except Exception as exc:
			return False, f"{type(exc).__name__}:{exc}"

	try:
		if not bool(has_extension("unit_locations")):
			if not callable(compute_extension):
				return {}, None

			compute_errors: list[str] = []

			computed, err = _try_compute("unit_locations")
			if not computed:
				if err is not None:
					compute_errors.append(str(err))
				computed_list, err_list = _try_compute(["unit_locations"])
				computed = bool(computed_list)
				if (not computed) and (err_list is not None):
					compute_errors.append(str(err_list))

			if (not computed) or (not bool(has_extension("unit_locations"))):
				# SortingAnalyzer often requires this extension chain before unit_locations is available.
				for extension_name in ("random_spikes", "waveforms", "templates", "unit_locations"):
					if bool(has_extension(extension_name)):
						continue
					ok, dep_err = _try_compute(extension_name)
					if not ok:
						ok_list, dep_err_list = _try_compute([extension_name])
						ok = bool(ok_list)
						if (not ok) and (dep_err_list is not None):
							compute_errors.append(f"{extension_name}:{dep_err_list}")
					elif dep_err is not None:
						compute_errors.append(f"{extension_name}:{dep_err}")

			if not bool(has_extension("unit_locations")):
				if compute_errors:
					return {}, "unit_locations_compute_failed:" + " | ".join(compute_errors)
				return {}, None
	except Exception as exc:
		return {}, f"unit_locations_check_failed:{type(exc).__name__}:{exc}"

	try:
		unit_locations = get_extension("unit_locations").get_data()
	except Exception as exc:
		return {}, f"unit_locations_load_failed:{type(exc).__name__}:{exc}"

	if hasattr(unit_locations, "to_numpy"):
		try:
			unit_locations = unit_locations.to_numpy()
		except Exception:
			pass

	rows_raw: Any
	if hasattr(unit_locations, "tolist"):
		try:
			rows_raw = unit_locations.tolist()
		except Exception:
			rows_raw = None
	else:
		rows_raw = None
	if rows_raw is None:
		try:
			rows_raw = list(unit_locations)
		except Exception as exc:
			return {}, f"unit_locations_iter_failed:{type(exc).__name__}:{exc}"
	if not isinstance(rows_raw, list):
		return {}, "unit_locations_unexpected_type"

	unit_ids = _unit_ids_from_obj(analyzer)
	if not unit_ids:
		return {}, "analyzer_unit_ids_unavailable"

	limit = min(len(unit_ids), len(rows_raw))
	out: dict[str, dict[str, float]] = {}
	for idx in range(limit):
		row = rows_raw[idx]
		if isinstance(row, dict):
			x_raw = row.get("x_um", row.get("x", None))
			y_raw = row.get("y_um", row.get("y", None))
		else:
			try:
				x_raw = row[0]
				y_raw = row[1]
			except Exception:
				continue
		try:
			x = float(x_raw)
			y = float(y_raw)
		except Exception:
			continue
		if not (math.isfinite(x) and math.isfinite(y)):
			continue
		uid = str(unit_ids[idx])
		out[uid] = {"x_um": x, "y_um": y}
		try:
			out[str(int(uid))] = {"x_um": x, "y_um": y}
		except Exception:
			pass
	return out, None


def _refresh_snapshot_analyzer_payload_from_live_analyzer(
	*,
	snapshot: dict[str, Any],
	analyzer: Any,
	stage_config: Any,
	include_unit_locations: bool,
	analyzer_source_dir: Path | None = None,
) -> dict[str, Any]:
	snapshot_payload = (dict(snapshot) if isinstance(snapshot, dict) else {})
	analyzer_payload_raw = snapshot_payload.get("analyzer", {})
	analyzer_payload = (
		dict(analyzer_payload_raw)
		if isinstance(analyzer_payload_raw, dict)
		else {}
	)

	if analyzer_source_dir is not None:
		analyzer_payload["source_dir"] = str(Path(analyzer_source_dir).resolve())
	analyzer_payload["available"] = True
	analyzer_payload["load_error"] = None
	analyzer_payload["unit_count"] = int(_unit_count(analyzer))
	analyzer_payload["unit_ids"] = _unit_ids_from_obj(analyzer)
	analyzer_payload["has_sparsity"] = _analyzer_has_sparsity(analyzer)

	if include_unit_locations:
		locations_by_unit, locations_error = _extract_unit_locations_from_analyzer(
			analyzer=analyzer,
			stage_config=stage_config,
		)
		analyzer_payload["unit_locations_by_unit"] = locations_by_unit
		analyzer_payload["unit_locations_error"] = locations_error

	snapshot_payload["analyzer"] = analyzer_payload
	return snapshot_payload


def _capture_merge_state_snapshot(
	*,
	well_out_dir: Path,
	stage_output_root_dir: Path,
	output_rel_root: str,
	stage_config: Any,
	sorter_output_dir: Path | None = None,
	analyzer_source_dir: Path | None = None,
	analyzer_obj: Any | None = None,
	capture_label: str,
	include_unit_locations: bool,
	allow_analyzer_recompute: bool,
	return_analyzer_obj: bool = False,
) -> dict[str, Any] | tuple[dict[str, Any], Any | None]:
	if sorter_output_dir is None:
		sorter_output_dir = _resolve_sorter_output_dir(
			well_out_dir=well_out_dir,
			output_rel_root=output_rel_root,
			stage_config=stage_config,
		)
	else:
		sorter_output_dir = Path(sorter_output_dir).resolve()
	analyzer_dir = (
		Path(analyzer_source_dir).resolve()
		if analyzer_source_dir is not None
		else (stage_output_root_dir / "analyzer_output").resolve()
	)
	_log_phase_step_start(
		"Merge snapshot capture start",
		well_out_dir=well_out_dir,
		capture=capture_label,
		include_unit_locations=bool(include_unit_locations),
		allow_analyzer_recompute=bool(allow_analyzer_recompute),
		sorter_output_dir=sorter_output_dir,
		analyzer_dir=analyzer_dir,
	)
	requested_policy = _requested_merge_analyzer_policy(stage_config)

	snapshot: dict[str, Any] = {
		"label": str(capture_label),
		"well_out_dir": str(well_out_dir),
		"stage_output_root_dir": str(stage_output_root_dir),
		"sorter": {
			"source_dir": str(sorter_output_dir),
			"available": False,
			"load_error": None,
			"unit_count": None,
			"unit_ids": [],
		},
		"analyzer": {
			"source_dir": str(analyzer_dir),
			"available": False,
			"load_error": None,
			"unit_count": None,
			"unit_ids": [],
			"unit_locations_by_unit": {},
			"unit_locations_error": None,
			"rebuilt": False,
			"requested_compute_sparsity": bool(requested_policy.get("requested_compute_sparsity", True)),
			"requested_template_random_spikes_method": str(
				requested_policy.get("template_random_spikes_method", "default")
			),
			"requested_template_random_spikes_max_spikes_per_unit": requested_policy.get(
				"template_random_spikes_max_spikes_per_unit",
				None,
			),
			"requested_template_random_spikes_percentage": requested_policy.get(
				"template_random_spikes_percentage",
				None,
			),
			"requested_template_random_spikes_log_before_after_spike_counts": requested_policy.get(
				"template_random_spikes_log_before_after_spike_counts",
				False,
			),
			"requested_template_random_spikes_min_spikes_per_unit": requested_policy.get(
				"template_random_spikes_min_spikes_per_unit",
				None,
			),
			"requested_template_random_spikes_margin_size": requested_policy.get(
				"template_random_spikes_margin_size",
				None,
			),
			"requested_template_random_spikes_seed": requested_policy.get(
				"template_random_spikes_seed",
				None,
			),
			"requested_sparsity_method": str(requested_policy.get("sparsity_method", "radius")),
			"requested_sparsity_radius_um": requested_policy.get("sparsity_radius_um", None),
			"requested_sparsity_num_channels": requested_policy.get("sparsity_num_channels", None),
			"requested_sparsity_threshold": requested_policy.get("sparsity_threshold", None),
			"requested_sparsity_peak_sign": requested_policy.get("sparsity_peak_sign", None),
			"requested_sparsity_num_spikes_for_sparsity": requested_policy.get(
				"sparsity_num_spikes_for_sparsity",
				None,
			),
			"requested_sparsity_by_property": requested_policy.get("sparsity_by_property", None),
			"requested_waveforms_ms_before": requested_policy.get("waveforms_ms_before", None),
			"requested_waveforms_ms_after": requested_policy.get("waveforms_ms_after", None),
			"requested_waveforms_dtype": requested_policy.get("waveforms_dtype", None),
			"requested_compute_n_jobs": requested_policy.get("compute_n_jobs", None),
			"requested_compute_chunk_duration": requested_policy.get("compute_chunk_duration", None),
			"loaded_analyzer_had_sparsity": None,
			"has_sparsity": None,
			"sparsity_validation_error": None,
			"rebuild_reason": None,
			"reused_cached_analyzer": None,
			"extensions_computed": [],
		},
	}

	si_module: Any | None = None
	spikeinterface_error: str | None = None
	try:
		si_module = _import_spikeinterface_full_module()
	except Exception as exc:
		spikeinterface_error = f"spikeinterface_import_failed:{type(exc).__name__}:{exc}"
		if analyzer_obj is None:
			snapshot["sorter"]["load_error"] = spikeinterface_error
			snapshot["analyzer"]["load_error"] = spikeinterface_error
			if return_analyzer_obj:
				return snapshot, None
			return snapshot

	sorter_load_error: str | None = spikeinterface_error
	if si_module is not None:
		try:
			sorter = _load_sorting_from_sorter_output_dir(
				si_module=si_module,
				sorter_output_dir=sorter_output_dir,
				sorter_name=str(getattr(stage_config, "sorter", "kilosort4") or "kilosort4"),
			)
			sorter_unit_ids = _unit_ids_from_obj(sorter)
			snapshot["sorter"].update(
				{
					"available": True,
					"unit_count": int(_unit_count(sorter)),
					"unit_ids": sorter_unit_ids,
				}
			)
			sorter_load_error = None
		except Exception as exc:
			sorter_load_error = f"sorter_snapshot_failed:{type(exc).__name__}:{exc}"

	loaded_analyzer_obj: Any | None = analyzer_obj
	if loaded_analyzer_obj is not None:
		snapshot["analyzer"]["source_dir"] = str(analyzer_dir)
		policy_info = _get_merge_analyzer_policy_info(loaded_analyzer_obj)
		if not policy_info:
			policy_info = _describe_merge_analyzer_policy_info(
				analyzer=loaded_analyzer_obj,
				stage_config=stage_config,
				reused_cached_analyzer=True,
			)
			_attach_merge_analyzer_policy_info(loaded_analyzer_obj, policy_info)
	elif allow_analyzer_recompute and (analyzer_source_dir is None):
		if si_module is None:
			snapshot["analyzer"]["load_error"] = str(spikeinterface_error)
		else:
			try:
				loaded_analyzer_obj, loaded_analyzer_dir, analyzer_rebuilt = _load_or_recompute_spikesort_analyzer(
					si_module=si_module,
					well_out_dir=well_out_dir,
					stage_output_root_dir=stage_output_root_dir,
					sorter_output_dir=sorter_output_dir,
					stage_config=stage_config,
				)
				snapshot["analyzer"]["source_dir"] = str(loaded_analyzer_dir)
				snapshot["analyzer"]["rebuilt"] = bool(analyzer_rebuilt)
			except Exception as exc:
				snapshot["analyzer"]["load_error"] = f"analyzer_snapshot_failed:{type(exc).__name__}:{exc}"
	else:
		load_sorting_analyzer = (getattr(si_module, "load_sorting_analyzer", None) if si_module is not None else None)
		loaded_analyzer_had_sparsity: bool | None = None
		rebuild_reason: str | None = None
		if analyzer_dir.exists() and callable(load_sorting_analyzer):
			try:
				loaded_analyzer_obj = _load_spikesort_analyzer_with_recording(
					si_module=si_module,
					analyzer_dir=analyzer_dir,
					well_out_dir=well_out_dir,
					stage_config=stage_config,
				)
				loaded_analyzer_had_sparsity = _analyzer_has_sparsity(loaded_analyzer_obj)
				if (not bool(requested_policy.get("requested_compute_sparsity", True))) and loaded_analyzer_had_sparsity:
					rebuild_reason = "loaded_sparse_analyzer"
					if allow_analyzer_recompute:
						loaded_analyzer_obj = None
				else:
					policy_info = _describe_merge_analyzer_policy_info(
						analyzer=loaded_analyzer_obj,
						stage_config=stage_config,
						loaded_analyzer_had_sparsity=loaded_analyzer_had_sparsity,
						reused_cached_analyzer=True,
					)
					_attach_merge_analyzer_policy_info(loaded_analyzer_obj, policy_info)
			except Exception as exc:
				snapshot["analyzer"]["load_error"] = f"analyzer_snapshot_failed:{type(exc).__name__}:{exc}"
		elif allow_analyzer_recompute and (analyzer_source_dir is not None):
			if si_module is None:
				snapshot["analyzer"]["load_error"] = str(spikeinterface_error)
			else:
				try:
					loaded_analyzer_obj, rebuilt_dir = _recompute_sorting_analyzer_to_dir(
						si_module=si_module,
						well_out_dir=well_out_dir,
						sorter_output_dir=sorter_output_dir,
						stage_config=stage_config,
						analyzer_dir=analyzer_dir,
					)
					snapshot["analyzer"]["source_dir"] = str(rebuilt_dir)
					snapshot["analyzer"]["rebuilt"] = True
					policy_info = _describe_merge_analyzer_policy_info(
						analyzer=loaded_analyzer_obj,
						stage_config=stage_config,
						loaded_analyzer_had_sparsity=loaded_analyzer_had_sparsity,
						rebuild_reason=rebuild_reason,
						reused_cached_analyzer=False,
					)
					_attach_merge_analyzer_policy_info(loaded_analyzer_obj, policy_info)
				except Exception as exc:
					snapshot["analyzer"]["load_error"] = f"analyzer_snapshot_failed:{type(exc).__name__}:{exc}"
		else:
			snapshot["analyzer"]["load_error"] = (
				str(spikeinterface_error)
				if spikeinterface_error is not None
				else "analyzer_output_missing"
			)

	if loaded_analyzer_obj is not None:
		policy_info = _get_merge_analyzer_policy_info(loaded_analyzer_obj)
		analyzer_unit_ids = _unit_ids_from_obj(loaded_analyzer_obj)
		snapshot["analyzer"].update(
			{
				"available": True,
				"unit_count": int(_unit_count(loaded_analyzer_obj)),
				"unit_ids": analyzer_unit_ids,
				"loaded_analyzer_had_sparsity": policy_info.get("loaded_analyzer_had_sparsity", None),
				"has_sparsity": policy_info.get(
					"final_analyzer_has_sparsity",
					_analyzer_has_sparsity(loaded_analyzer_obj),
				),
				"sparsity_validation_error": policy_info.get("sparsity_validation_error", None),
				"rebuild_reason": policy_info.get("rebuild_reason", None),
				"reused_cached_analyzer": policy_info.get("reused_cached_analyzer", None),
			}
		)
		if include_unit_locations:
			computed_extensions = _ensure_merge_analyzer_extensions(
				analyzer=loaded_analyzer_obj,
				stage_config=stage_config,
				include_unit_locations=True,
				include_templates=bool(return_analyzer_obj),
			)
			snapshot["analyzer"]["extensions_computed"] = list(computed_extensions)
			locations_by_unit, locations_error = _extract_unit_locations_from_analyzer(
				analyzer=loaded_analyzer_obj,
				stage_config=stage_config,
			)
			snapshot["analyzer"]["unit_locations_by_unit"] = locations_by_unit
			snapshot["analyzer"]["unit_locations_error"] = locations_error

	if not bool(snapshot["sorter"].get("available", False)):
		analyzer_sorting = (
			getattr(loaded_analyzer_obj, "sorting", None)
			if loaded_analyzer_obj is not None
			else None
		)
		if analyzer_sorting is not None:
			snapshot["sorter"].update(
				{
					"available": True,
					"load_error": None,
					"unit_count": int(_unit_count(analyzer_sorting)),
					"unit_ids": _unit_ids_from_obj(analyzer_sorting),
				}
			)
		elif sorter_load_error is not None:
			snapshot["sorter"]["load_error"] = str(sorter_load_error)

	if return_analyzer_obj:
		return snapshot, loaded_analyzer_obj
	return snapshot


def _call_capture_merge_state_snapshot_compat(**snapshot_kwargs: Any) -> Any:
	current_kwargs = dict(snapshot_kwargs)
	while True:
		try:
			return _capture_merge_state_snapshot(**current_kwargs)
		except TypeError as exc:
			err_text = str(exc)
			if "unexpected keyword argument" not in err_text:
				raise
			removed_key: str | None = None
			for key in ("analyzer_obj", "return_analyzer_obj", "analyzer_source_dir", "sorter_output_dir"):
				if key in current_kwargs and key in err_text:
					removed_key = key
					break
			if removed_key is None:
				raise
			current_kwargs.pop(removed_key, None)
			if removed_key == "analyzer_source_dir":
				current_kwargs["allow_analyzer_recompute"] = True


def _read_json_dict(path: Path) -> dict[str, Any] | None:
	if not path.exists():
		return None
	try:
		payload = json.loads(path.read_text(encoding="utf-8"))
	except Exception:
		return None
	if isinstance(payload, dict):
		return payload
	return None


def _extract_group_lists(raw_groups: Any) -> list[list[str]]:
	if isinstance(raw_groups, dict):
		return _normalize_merge_groups(list(raw_groups.values()))
	return _normalize_merge_groups(raw_groups)


def _extract_slay_applied_merge_operations(*, report: dict[str, Any]) -> list[dict[str, Any]]:
	outputs = report.get("outputs", {})
	if not isinstance(outputs, dict):
		return []
	groups_path_raw = outputs.get("slay.recommended_merge_groups_json", None)
	if groups_path_raw is None:
		return []
	groups_payload = _read_json_dict(Path(str(groups_path_raw)))
	if not isinstance(groups_payload, dict):
		return []
	merge_groups_raw = groups_payload.get("merge_groups", {})
	out: list[dict[str, Any]] = []
	if isinstance(merge_groups_raw, dict):
		for raw_group_id, raw_members in merge_groups_raw.items():
			groups = _normalize_merge_groups([raw_members])
			if not groups:
				continue
			members = list(groups[0])
			post_unit_id_hint: str | None = None
			if raw_group_id is not None and str(raw_group_id).strip():
				post_unit_id_hint = _normalize_cluster_id(raw_group_id)
			out.append(
				{
					"method": "slay",
					"iteration": None,
					"template_diff_thresh": None,
					"group_id": str(raw_group_id),
					"pre_unit_ids": members,
					"post_unit_id_hint": post_unit_id_hint,
				}
			)
		return out

	groups = _extract_group_lists(merge_groups_raw)
	for group_idx, members in enumerate(groups, start=1):
		out.append(
			{
				"method": "slay",
				"iteration": None,
				"template_diff_thresh": None,
				"group_id": f"slay_group_{group_idx:03d}",
				"pre_unit_ids": list(members),
			}
		)
	return out


def _extract_auto_merge_applied_merge_operations(*, report: dict[str, Any]) -> list[dict[str, Any]]:
	summary_path_raw = report.get("summary_json", None)
	if summary_path_raw is None:
		outputs = report.get("outputs", {})
		if isinstance(outputs, dict):
			summary_path_raw = outputs.get("auto_merge.summary_json", None)
	if summary_path_raw is None:
		return []
	summary_payload = _read_json_dict(Path(str(summary_path_raw)))
	if not isinstance(summary_payload, dict):
		return []
	iterations_raw = summary_payload.get("iterations", [])
	if not isinstance(iterations_raw, list):
		return []

	out: list[dict[str, Any]] = []
	for iteration_payload in iterations_raw:
		if not isinstance(iteration_payload, dict):
			continue
		if not bool(iteration_payload.get("applied", False)):
			continue
		iteration_idx = int(iteration_payload.get("iteration", 0) or 0)
		threshold = iteration_payload.get("template_diff_thresh", None)

		groups: list[list[str]] = []
		applied_groups_path = iteration_payload.get("applied_groups_json", None)
		if applied_groups_path is not None:
			applied_payload = _read_json_dict(Path(str(applied_groups_path)))
			if isinstance(applied_payload, dict):
				groups = _extract_group_lists(applied_payload.get("applied_groups", []))
		if not groups:
			groups_json_path = iteration_payload.get("groups_json", None)
			if groups_json_path is not None:
				groups_payload = _read_json_dict(Path(str(groups_json_path)))
				if isinstance(groups_payload, dict):
					groups = _extract_group_lists(groups_payload.get("merge_groups", []))

		for group_idx, members in enumerate(groups, start=1):
			out.append(
				{
					"method": "auto_merge",
					"iteration": int(iteration_idx),
					"template_diff_thresh": threshold,
					"group_id": f"auto_merge_iter_{int(iteration_idx):03d}_group_{group_idx:03d}",
					"pre_unit_ids": list(members),
				}
			)
	return out


def _extract_applied_merge_operations(*, method_reports: list[dict[str, Any]], stage_config: Any) -> list[dict[str, Any]]:
	operations: list[dict[str, Any]] = []
	for report in method_reports:
		if not isinstance(report, dict):
			continue
		if str(report.get("status", "")).strip().lower() != "ok":
			continue
		method = _normalize_merge_method_token(report.get("name", ""))
		if method == "slay":
			if bool(getattr(stage_config, "slay_auto_accept_merges", False)) and bool(report.get("applied_merges", False)):
				operations.extend(_extract_slay_applied_merge_operations(report=report))
			continue
		if method == "auto_merge":
			if bool(getattr(stage_config, "auto_merge_auto_accept_merges", False)):
				operations.extend(_extract_auto_merge_applied_merge_operations(report=report))
			continue
	return operations


def _compute_snapshot_unit_delta(*, before_payload: dict[str, Any], after_payload: dict[str, Any]) -> dict[str, Any]:
	before_available = bool(before_payload.get("available", False))
	after_available = bool(after_payload.get("available", False))
	before_ids_raw = before_payload.get("unit_ids", []) if isinstance(before_payload, dict) else []
	after_ids_raw = after_payload.get("unit_ids", []) if isinstance(after_payload, dict) else []
	before_ids = [str(tok) for tok in list(before_ids_raw or []) if str(tok).strip()]
	after_ids = [str(tok) for tok in list(after_ids_raw or []) if str(tok).strip()]
	before_count = _snapshot_unit_count(before_payload, before_ids)
	after_count = _snapshot_unit_count(after_payload, after_ids)
	before_set = set(before_ids)
	after_set = set(after_ids)

	compared = bool(before_available and after_available)
	added_ids = sorted(list(after_set - before_set), key=_unit_sort_key)
	removed_ids = sorted(list(before_set - after_set), key=_unit_sort_key)
	changed = bool(compared and (added_ids or removed_ids or (before_count != after_count)))

	return {
		"compared": bool(compared),
		"changed": bool(changed),
		"before_unit_count": int(before_count),
		"after_unit_count": int(after_count),
		"before_unit_ids": before_ids,
		"after_unit_ids": after_ids,
		"added_unit_ids": added_ids,
		"removed_unit_ids": removed_ids,
		"before_available": bool(before_available),
		"after_available": bool(after_available),
	}


def _normalize_unit_id_list(raw_ids: Any) -> list[str]:
	out: list[str] = []
	for raw_uid in list(raw_ids or []):
		uid = _normalize_cluster_id(raw_uid)
		if uid and uid not in out:
			out.append(uid)
	return out


def _extract_xy_tuple(location_raw: Any) -> tuple[float, float] | None:
	if isinstance(location_raw, dict):
		x_raw = location_raw.get("x_um", location_raw.get("x", None))
		y_raw = location_raw.get("y_um", location_raw.get("y", None))
	else:
		try:
			x_raw = location_raw[0]
			y_raw = location_raw[1]
		except Exception:
			return None

	try:
		x = float(x_raw)
		y = float(y_raw)
	except Exception:
		return None

	if not (math.isfinite(x) and math.isfinite(y)):
		return None
	return (float(x), float(y))


def _infer_post_unit_id_from_locations(
	*,
	pre_unit_ids: list[str],
	pre_locations: dict[str, Any],
	post_locations: dict[str, Any],
	used_post_unit_ids: set[str] | None = None,
	max_distance_um: float = 40.0,
) -> tuple[str | None, float | None]:
	if not pre_unit_ids:
		return None, None

	pre_points: list[tuple[float, float]] = []
	for uid in pre_unit_ids:
		xy = _extract_xy_tuple(pre_locations.get(str(uid), None))
		if xy is not None:
			pre_points.append(xy)
	if not pre_points:
		return None, None

	cx = float(sum(p[0] for p in pre_points) / float(len(pre_points)))
	cy = float(sum(p[1] for p in pre_points) / float(len(pre_points)))
	spread_um = max((math.hypot(p[0] - cx, p[1] - cy) for p in pre_points), default=0.0)
	allowed_distance_um = max(float(max_distance_um), 3.0 * float(spread_um))

	used = (used_post_unit_ids if isinstance(used_post_unit_ids, set) else set())
	best_uid: str | None = None
	best_distance: float | None = None
	for uid in sorted(list(post_locations.keys()), key=_unit_sort_key):
		if uid in used:
			continue
		xy = _extract_xy_tuple(post_locations.get(uid, None))
		if xy is None:
			continue
		distance = float(math.hypot(xy[0] - cx, xy[1] - cy))
		if best_distance is None or distance < best_distance:
			best_uid = str(uid)
			best_distance = float(distance)

	if best_uid is None or best_distance is None:
		return None, None
	if float(best_distance) > float(allowed_distance_um):
		return None, float(best_distance)
	return str(best_uid), float(best_distance)


def _build_applied_unit_mappings(
	*,
	applied_operations: list[dict[str, Any]],
	pre_analyzer_payload: dict[str, Any],
	post_analyzer_payload: dict[str, Any],
) -> list[dict[str, Any]]:
	pre_ids = set(str(tok) for tok in list(pre_analyzer_payload.get("unit_ids", []) or []))
	post_ids = set(str(tok) for tok in list(post_analyzer_payload.get("unit_ids", []) or []))
	pre_locations_raw = pre_analyzer_payload.get("unit_locations_by_unit", {})
	post_locations_raw = post_analyzer_payload.get("unit_locations_by_unit", {})
	pre_locations = pre_locations_raw if isinstance(pre_locations_raw, dict) else {}
	post_locations = post_locations_raw if isinstance(post_locations_raw, dict) else {}

	new_post_ids = sorted(list(post_ids - pre_ids), key=_unit_sort_key)
	out: list[dict[str, Any]] = []
	used_post_unit_ids: set[str] = set()
	for op in applied_operations:
		members = _normalize_unit_id_list(op.get("pre_unit_ids", []))
		surviving = [tok for tok in members if tok in post_ids]
		resolved_post_unit_id: str | None = None
		resolution = "unresolved"

		post_unit_hint_raw = op.get("post_unit_id", op.get("post_unit_id_hint", None))
		post_unit_hint: str | None = None
		if post_unit_hint_raw is not None and str(post_unit_hint_raw).strip():
			post_unit_hint = _normalize_cluster_id(post_unit_hint_raw)

		if post_unit_hint is not None and post_unit_hint not in used_post_unit_ids:
			resolved_post_unit_id = str(post_unit_hint)
			if post_unit_hint in post_ids:
				resolution = "post_unit_hint"
			else:
				resolution = "post_unit_hint_missing_in_post_snapshot"
		elif len(surviving) == 1:
			resolved_post_unit_id = str(surviving[0])
			resolution = "surviving_pre_unit"
		elif len(surviving) > 1:
			resolution = "ambiguous_multiple_survivors"
		elif len(new_post_ids) == 1 and len(applied_operations) == 1:
			resolved_post_unit_id = str(new_post_ids[0])
			resolution = "single_new_unit"
		elif len(new_post_ids) > 1:
			resolution = "ambiguous_new_units"
		else:
			resolution = "no_post_unit_match"

		if resolved_post_unit_id is not None:
			used_post_unit_ids.add(str(resolved_post_unit_id))

		pre_member_locations: dict[str, Any] = {}
		for member in members:
			pre_member_locations[str(member)] = pre_locations.get(str(member), None)

		out.append(
			{
				"method": str(op.get("method", "unknown")),
				"group_id": str(op.get("group_id", "")),
				"iteration": op.get("iteration", None),
				"template_diff_thresh": op.get("template_diff_thresh", None),
				"pre_unit_ids": members,
				"post_unit_id": resolved_post_unit_id,
				"post_unit_id_in_post_snapshot": bool(
					resolved_post_unit_id is not None and str(resolved_post_unit_id) in post_ids
				),
				"resolution": str(resolution),
				"pre_unit_locations": pre_member_locations,
				"post_unit_location": (
					post_locations.get(str(resolved_post_unit_id), None) if resolved_post_unit_id is not None else None
				),
			}
		)
	return out


def _extract_applied_unit_mappings_for_report(*, merge_metadata_payload: dict[str, Any]) -> list[dict[str, Any]]:
	mappings_raw = merge_metadata_payload.get("applied_unit_mappings", [])
	mappings: list[dict[str, Any]] = []
	if isinstance(mappings_raw, list):
		for item in mappings_raw:
			if not isinstance(item, dict):
				continue
			pre_unit_ids = _normalize_unit_id_list(item.get("pre_unit_ids", []))
			if not pre_unit_ids:
				continue
			mapping = dict(item)
			mapping["pre_unit_ids"] = pre_unit_ids
			post_unit_id_raw = mapping.get("post_unit_id", None)
			if post_unit_id_raw is not None and str(post_unit_id_raw).strip():
				mapping["post_unit_id"] = _normalize_cluster_id(post_unit_id_raw)
			else:
				mapping["post_unit_id"] = None
			mappings.append(mapping)
		return mappings


def _build_merge_unit_diff_report_payload(
	*,
	requested_sequence_raw: list[Any],
	method_reports: list[dict[str, Any]],
	merge_metadata_payload: dict[str, Any] | None,
	before_snapshot: dict[str, Any] | None,
	after_snapshot: dict[str, Any] | None,
	applied_operations: list[dict[str, Any]] | None,
) -> dict[str, Any]:
	before = (dict(before_snapshot) if isinstance(before_snapshot, dict) else {})
	after = (dict(after_snapshot) if isinstance(after_snapshot, dict) else {})
	operations_raw = (applied_operations if isinstance(applied_operations, list) else [])
	operations = [dict(item) for item in operations_raw if isinstance(item, dict)]

	payload: dict[str, Any] = {
		"schema_version": 1,
		"source": "merge_stage_snapshots",
		"requested_sequence": [str(token) for token in requested_sequence_raw],
		"before": before,
		"after": after,
		"applied_merge_operations": operations,
		"applied_merge_group_count": int(len(operations)),
		"applied_unit_mappings": [],
		"change_validation": {},
		"delta": {},
		"methods": [dict(report) for report in method_reports if isinstance(report, dict)],
	}

	if isinstance(merge_metadata_payload, dict):
		payload["source"] = "merge_metadata_summary"
		before_raw = merge_metadata_payload.get("before", None)
		after_raw = merge_metadata_payload.get("after", None)
		if isinstance(before_raw, dict):
			payload["before"] = dict(before_raw)
		if isinstance(after_raw, dict):
			payload["after"] = dict(after_raw)

		operations_raw = merge_metadata_payload.get("applied_merge_operations", None)
		if isinstance(operations_raw, list):
			payload["applied_merge_operations"] = [
				dict(item)
				for item in operations_raw
				if isinstance(item, dict)
			]
		payload["applied_merge_group_count"] = int(
			merge_metadata_payload.get(
				"applied_merge_group_count",
				len(list(payload.get("applied_merge_operations", []) or [])),
			)
			or 0
		)
		payload["applied_unit_mappings"] = _extract_applied_unit_mappings_for_report(
			merge_metadata_payload=merge_metadata_payload
		)
		change_validation_raw = merge_metadata_payload.get("change_validation", {})
		delta_raw = merge_metadata_payload.get("delta", {})
		payload["change_validation"] = (
			dict(change_validation_raw)
			if isinstance(change_validation_raw, dict)
			else {}
		)
		payload["delta"] = (dict(delta_raw) if isinstance(delta_raw, dict) else {})

	return payload


def _build_unit_diff_map_payload(*, unit_diff_payload: dict[str, Any]) -> dict[str, Any]:
	before_raw = unit_diff_payload.get("before", {})
	after_raw = unit_diff_payload.get("after", {})
	before = (dict(before_raw) if isinstance(before_raw, dict) else {})
	after = (dict(after_raw) if isinstance(after_raw, dict) else {})
	before_analyzer = dict(before.get("analyzer", {}))
	after_analyzer = dict(after.get("analyzer", {}))
	before_unit_ids = _normalize_unit_id_list(before_analyzer.get("unit_ids", []))
	after_unit_ids = _normalize_unit_id_list(after_analyzer.get("unit_ids", []))
	before_set = set(before_unit_ids)
	after_set = set(after_unit_ids)
	new_post_unit_ids = sorted(list(after_set - before_set), key=_unit_sort_key)
	missing_pre_unit_ids = sorted(list(before_set - after_set), key=_unit_sort_key)

	operations_raw = unit_diff_payload.get("applied_merge_operations", [])
	operations = [dict(op) for op in list(operations_raw or []) if isinstance(op, dict)]
	mappings_raw = unit_diff_payload.get("applied_unit_mappings", [])
	mappings = [dict(item) for item in list(mappings_raw or []) if isinstance(item, dict)]

	def _op_match(mapping: dict[str, Any], op: dict[str, Any]) -> bool:
		if str(mapping.get("method", "")) != str(op.get("method", "")):
			return False
		mapping_group = str(mapping.get("group_id", ""))
		op_group = str(op.get("group_id", ""))
		if mapping_group and op_group and mapping_group != op_group:
			return False
		mapping_iter = mapping.get("iteration", None)
		op_iter = op.get("iteration", None)
		if mapping_iter is not None and op_iter is not None and mapping_iter != op_iter:
			return False
		mapping_pre = set(_normalize_unit_id_list(mapping.get("pre_unit_ids", [])))
		op_pre = set(_normalize_unit_id_list(op.get("pre_unit_ids", [])))
		if mapping_pre and op_pre and mapping_pre != op_pre:
			return False
		return True

	used_mapping_idx: set[int] = set()
	operation_edges: list[dict[str, Any]] = []
	lineage: dict[str, set[str]] = {uid: {uid} for uid in before_unit_ids}

	for op_index, op in enumerate(operations):
		op_method = str(op.get("method", "unknown"))
		op_group_id = str(op.get("group_id", f"group_{int(op_index + 1):03d}"))
		op_pre_ids = _normalize_unit_id_list(op.get("pre_unit_ids", []))
		hint_post_uid = _normalize_cluster_id(op.get("post_unit_id", op.get("post_unit_id_hint", None)))

		selected_mapping: dict[str, Any] | None = None
		selected_mapping_idx: int | None = None
		selected_score: int | None = None
		op_pre_set = set(op_pre_ids)
		op_group_raw = str(op.get("group_id", "")).strip()
		op_iter_raw = op.get("iteration", None)
		for map_idx, mapping in enumerate(mappings):
			if map_idx in used_mapping_idx:
				continue
			if _op_match(mapping, op):
				score = 0
				mapping_pre_set = set(_normalize_unit_id_list(mapping.get("pre_unit_ids", [])))
				if mapping_pre_set and op_pre_set and mapping_pre_set == op_pre_set:
					score += 4
				mapping_group_raw = str(mapping.get("group_id", "")).strip()
				if mapping_group_raw and op_group_raw and mapping_group_raw == op_group_raw:
					score += 2
				mapping_iter_raw = mapping.get("iteration", None)
				if (
					mapping_iter_raw is not None
					and op_iter_raw is not None
					and mapping_iter_raw == op_iter_raw
				):
					score += 1
				if _normalize_cluster_id(mapping.get("post_unit_id", None)) is not None:
					score += 1

				if (
					selected_score is None
					or score > selected_score
					or (score == selected_score and (selected_mapping_idx is None or map_idx < selected_mapping_idx))
				):
					selected_mapping = mapping
					selected_mapping_idx = int(map_idx)
					selected_score = int(score)

		if selected_mapping_idx is not None:
			used_mapping_idx.add(int(selected_mapping_idx))

		resolved_post_uid = _normalize_cluster_id(
			(
				selected_mapping.get("post_unit_id", None)
				if isinstance(selected_mapping, dict)
				else None
			)
			or hint_post_uid
		)
		resolved_post_in_after = bool(resolved_post_uid is not None and resolved_post_uid in after_set)
		resolution_method = str(
			(
				selected_mapping.get("resolution", None)
				if isinstance(selected_mapping, dict)
				else None
			)
			or ("hint" if hint_post_uid is not None else "unresolved")
		)

		primary_pre_ids: set[str] = set()
		for uid in op_pre_ids:
			if uid in lineage:
				primary_pre_ids.update(set(lineage.get(uid, set())))
			elif uid in before_set:
				primary_pre_ids.add(uid)

		if resolved_post_uid is not None and primary_pre_ids:
			existing = set(lineage.get(resolved_post_uid, set()))
			lineage[resolved_post_uid] = existing.union(primary_pre_ids)

		edge = {
			"operation_index": int(op_index),
			"method": op_method,
			"group_id": op_group_id,
			"iteration": op.get("iteration", None),
			"template_diff_thresh": op.get("template_diff_thresh", None),
			"pre_unit_ids": op_pre_ids,
			"primary_pre_unit_ids": sorted(list(primary_pre_ids), key=_unit_sort_key),
			"post_unit_id_hint": hint_post_uid,
			"resolved_post_unit_id": resolved_post_uid,
			"resolved_post_unit_id_in_post_snapshot": bool(resolved_post_in_after),
			"resolution_method": resolution_method,
		}
		operation_edges.append(edge)

	return {
		"schema_version": 1,
		"requested_sequence": list(unit_diff_payload.get("requested_sequence", [])),
		"before_unit_ids": before_unit_ids,
		"after_unit_ids": after_unit_ids,
		"new_post_unit_ids": new_post_unit_ids,
		"missing_pre_unit_ids": missing_pre_unit_ids,
		"operation_edges": operation_edges,
		"summary": {
			"n_operations": int(len(operation_edges)),
			"n_new_post_units": int(len(new_post_unit_ids)),
			"n_missing_pre_units": int(len(missing_pre_unit_ids)),
			"n_unresolved_operations": int(
				sum(1 for edge in operation_edges if edge.get("resolved_post_unit_id", None) is None)
			),
		},
	}


def _build_unit_diff_map_flat_payload(
	*,
	unit_diff_map_payload: dict[str, Any],
	unit_diff_payload: dict[str, Any],
) -> dict[str, Any]:
	after_raw = unit_diff_payload.get("after", {})
	after = (dict(after_raw) if isinstance(after_raw, dict) else {})
	after_analyzer = dict(after.get("analyzer", {}))
	after_locations_raw = after_analyzer.get("unit_locations_by_unit", {})
	after_locations = (after_locations_raw if isinstance(after_locations_raw, dict) else {})
	after_unit_ids = set(_normalize_unit_id_list(after_analyzer.get("unit_ids", [])))

	edges_raw = unit_diff_map_payload.get("operation_edges", [])
	edges = [dict(edge) for edge in list(edges_raw or []) if isinstance(edge, dict)]

	source_to_target: dict[str, str] = {}
	participants_primary: set[str] = set()
	for edge in edges:
		primary_ids = _normalize_unit_id_list(edge.get("primary_pre_unit_ids", []))
		participants_primary.update(primary_ids)
		resolved_post_uid = _normalize_cluster_id(edge.get("resolved_post_unit_id", None))
		if resolved_post_uid is None:
			continue
		for src_uid in _normalize_unit_id_list(edge.get("pre_unit_ids", [])):
			source_to_target[src_uid] = resolved_post_uid

	def _resolve_terminal_post(uid: str) -> str | None:
		cursor = str(uid)
		seen: set[str] = set()
		while cursor in source_to_target and cursor not in seen:
			seen.add(cursor)
			cursor = str(source_to_target[cursor])
		if cursor in after_unit_ids:
			return str(cursor)
		return None

	primary_to_final_rows: list[dict[str, Any]] = []
	grouped: dict[str, set[str]] = {}
	unresolved_primary_pre_unit_ids: list[str] = []
	for primary_uid in sorted(list(participants_primary), key=_unit_sort_key):
		final_uid = _resolve_terminal_post(primary_uid)
		primary_to_final_rows.append(
			{
				"primary_pre_unit_id": str(primary_uid),
				"final_post_unit_id": final_uid,
			}
		)
		if final_uid is None:
			unresolved_primary_pre_unit_ids.append(str(primary_uid))
			continue
		grouped.setdefault(str(final_uid), set()).add(str(primary_uid))

	flat_groups: list[dict[str, Any]] = []
	for final_uid in sorted(list(grouped.keys()), key=_unit_sort_key):
		location = after_locations.get(final_uid, None)
		flat_groups.append(
			{
				"final_post_unit_id": str(final_uid),
				"primary_pre_unit_ids": sorted(list(grouped.get(final_uid, set())), key=_unit_sort_key),
				"final_post_unit_location": location,
			}
		)

	return {
		"schema_version": 1,
		"flat_scope": "merged_participants_only",
		"groups": flat_groups,
		"primary_unit_to_final_post": primary_to_final_rows,
		"unresolved_primary_pre_unit_ids": unresolved_primary_pre_unit_ids,
		"summary": {
			"n_flat_groups": int(len(flat_groups)),
			"n_primary_units_participating": int(len(participants_primary)),
			"n_unresolved_primary_units": int(len(unresolved_primary_pre_unit_ids)),
		},
	}


def _build_post_merge_unit_locations_payload(*, post_snapshot: dict[str, Any]) -> dict[str, Any]:
	after_raw = post_snapshot.get("analyzer", {})
	after_analyzer = (dict(after_raw) if isinstance(after_raw, dict) else {})
	locations_raw = after_analyzer.get("unit_locations_by_unit", {})
	locations = (locations_raw if isinstance(locations_raw, dict) else {})
	unit_ids = _normalize_unit_id_list(after_analyzer.get("unit_ids", []))

	normalized_locations: dict[str, Any] = {}
	for uid in unit_ids:
		if uid in locations and isinstance(locations.get(uid, None), dict):
			normalized_locations[uid] = dict(locations[uid])
	for uid_raw, loc in locations.items():
		uid = _normalize_cluster_id(uid_raw)
		if uid is None or uid in normalized_locations or not isinstance(loc, dict):
			continue
		normalized_locations[uid] = dict(loc)

	return {
		"schema_version": 1,
		"unit_ids": unit_ids,
		"unit_locations_by_unit": normalized_locations,
		"summary": {
			"n_unit_ids": int(len(unit_ids)),
			"n_locations": int(len(normalized_locations)),
		},
	}


def _build_plot_mappings_from_unit_diff_flat_payload(*, flat_payload: dict[str, Any]) -> list[dict[str, Any]]:
	groups_raw = flat_payload.get("groups", [])
	groups = [dict(group) for group in list(groups_raw or []) if isinstance(group, dict)]
	out: list[dict[str, Any]] = []
	for group in groups:
		post_uid = _normalize_cluster_id(group.get("final_post_unit_id", None))
		if post_uid is None:
			continue
		pre_ids = _normalize_unit_id_list(group.get("primary_pre_unit_ids", []))
		if not pre_ids:
			continue
		out.append(
			{
				"method": "flattened",
				"group_id": str(group.get("group_id", f"flat_{post_uid}")),
				"pre_unit_ids": pre_ids,
				"post_unit_id": post_uid,
			}
		)
	return out


def _extract_plot_inputs_from_unit_diff_report(
	*,
	unit_diff_payload: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
	before_raw = unit_diff_payload.get("before", None)
	after_raw = unit_diff_payload.get("after", None)
	before_snapshot = (dict(before_raw) if isinstance(before_raw, dict) else {})
	after_snapshot = (dict(after_raw) if isinstance(after_raw, dict) else {})
	flat_payload_raw = unit_diff_payload.get("unit_diff_map_flat", None)
	if isinstance(flat_payload_raw, dict):
		applied_unit_mappings = _build_plot_mappings_from_unit_diff_flat_payload(
			flat_payload=flat_payload_raw,
		)
	else:
		applied_unit_mappings = _extract_applied_unit_mappings_for_report(
			merge_metadata_payload=unit_diff_payload,
		)
	return before_snapshot, after_snapshot, applied_unit_mappings


def _build_merge_metadata_summary(
	*,
	requested_sequence_raw: list[Any],
	stage_config: Any,
	pre_snapshot: dict[str, Any],
	post_snapshot: dict[str, Any],
	applied_operations: list[dict[str, Any]],
) -> dict[str, Any]:
	requested_methods = [_normalize_merge_method_token(token) for token in requested_sequence_raw]
	slay_auto_accept_enabled = bool(
		bool(getattr(stage_config, "slay_enabled", False))
		and bool(getattr(stage_config, "slay_auto_accept_merges", False))
		and ("slay" in requested_methods)
	)
	auto_merge_auto_accept_enabled = bool(
		bool(getattr(stage_config, "auto_merge_enabled", False))
		and bool(getattr(stage_config, "auto_merge_auto_accept_merges", False))
		and ("auto_merge" in requested_methods)
	)
	auto_accept_enabled_any = bool(slay_auto_accept_enabled or auto_merge_auto_accept_enabled)

	sorter_delta_raw = _compute_snapshot_unit_delta(
		before_payload=dict(pre_snapshot.get("sorter", {})),
		after_payload=dict(post_snapshot.get("sorter", {})),
	)
	analyzer_delta_raw = _compute_snapshot_unit_delta(
		before_payload=dict(pre_snapshot.get("analyzer", {})),
		after_payload=dict(post_snapshot.get("analyzer", {})),
	)

	applied_mappings = _build_applied_unit_mappings(
		applied_operations=applied_operations,
		pre_analyzer_payload=dict(pre_snapshot.get("analyzer", {})),
		post_analyzer_payload=dict(post_snapshot.get("analyzer", {})),
	)

	merge_removed_unit_ids: list[str] = []
	merge_target_unit_ids: list[str] = []
	merge_unresolved_group_ids: list[str] = []
	for mapping in applied_mappings:
		if not isinstance(mapping, dict):
			continue
		for uid in _normalize_unit_id_list(mapping.get("pre_unit_ids", [])):
			if uid not in merge_removed_unit_ids:
				merge_removed_unit_ids.append(uid)
		post_uid_raw = mapping.get("post_unit_id", None)
		if post_uid_raw is None or not str(post_uid_raw).strip():
			group_id_raw = mapping.get("group_id", None)
			if group_id_raw is not None and str(group_id_raw).strip():
				group_id = str(group_id_raw)
				if group_id not in merge_unresolved_group_ids:
					merge_unresolved_group_ids.append(group_id)
			continue
		post_uid = _normalize_cluster_id(post_uid_raw)
		if post_uid and post_uid not in merge_target_unit_ids:
			merge_target_unit_ids.append(post_uid)

	def _attach_mapping_tracking(delta_payload_raw: dict[str, Any]) -> dict[str, Any]:
		delta_payload = dict(delta_payload_raw)
		added_set_delta = _normalize_unit_id_list(delta_payload.get("added_unit_ids", []))
		removed_set_delta = _normalize_unit_id_list(delta_payload.get("removed_unit_ids", []))
		after_ids = _normalize_unit_id_list(delta_payload.get("after_unit_ids", []))
		target_ids_missing_from_post = sorted(
			list(set(merge_target_unit_ids) - set(after_ids)),
			key=_unit_sort_key,
		)

		delta_payload["added_unit_ids_set_delta"] = list(added_set_delta)
		delta_payload["added_unit_ids_source"] = "set_delta"
		delta_payload["merge_removed_unit_ids_from_operations"] = list(merge_removed_unit_ids)
		delta_payload["merge_target_unit_ids_from_mappings"] = list(merge_target_unit_ids)
		delta_payload["merge_unresolved_group_ids"] = list(merge_unresolved_group_ids)
		delta_payload["merge_tracking_validation"] = {
			"removed_ids_cover_operations": bool(set(merge_removed_unit_ids).issubset(set(removed_set_delta))),
			"added_ids_cover_mapping_targets": bool(set(merge_target_unit_ids).issubset(set(added_set_delta))),
			"target_ids_missing_from_post_snapshot": list(target_ids_missing_from_post),
			"targets_present_in_post_snapshot": bool(not target_ids_missing_from_post),
			"n_removed_ids_set_delta": int(len(removed_set_delta)),
			"n_removed_ids_from_operations": int(len(merge_removed_unit_ids)),
			"n_added_ids_effective": int(len(added_set_delta)),
			"n_added_ids_set_delta": int(len(added_set_delta)),
			"n_mapping_targets": int(len(merge_target_unit_ids)),
			"n_unresolved_mappings": int(len(merge_unresolved_group_ids)),
		}
		return delta_payload

	sorter_delta = _attach_mapping_tracking(sorter_delta_raw)
	analyzer_delta = _attach_mapping_tracking(analyzer_delta_raw)
	observed_change_any = bool(sorter_delta.get("changed", False) or analyzer_delta.get("changed", False))

	applied_group_count = int(len(applied_operations))
	expected_change_if_applied = bool(auto_accept_enabled_any and applied_group_count > 0)
	sorter_tracking_raw = sorter_delta.get("merge_tracking_validation", {})
	analyzer_tracking_raw = analyzer_delta.get("merge_tracking_validation", {})
	sorter_tracking = (sorter_tracking_raw if isinstance(sorter_tracking_raw, dict) else {})
	analyzer_tracking = (analyzer_tracking_raw if isinstance(analyzer_tracking_raw, dict) else {})
	missing_targets_sorter = list(sorter_tracking.get("target_ids_missing_from_post_snapshot", []) or [])
	missing_targets_analyzer = list(analyzer_tracking.get("target_ids_missing_from_post_snapshot", []) or [])
	missing_mapping_targets = bool(expected_change_if_applied and not merge_target_unit_ids)

	validation_passes = bool((not expected_change_if_applied) or observed_change_any)
	if expected_change_if_applied and not observed_change_any:
		validation_reason = "auto_accept_enabled_and_merges_applied_but_no_before_after_unit_change_detected"
	elif auto_accept_enabled_any and not applied_group_count:
		validation_reason = "auto_accept_enabled_but_no_applied_merge_groups_detected"
	else:
		validation_reason = "ok"

	return {
		"requested_sequence": [str(token) for token in requested_sequence_raw],
		"auto_accept": {
			"slay_enabled": bool(slay_auto_accept_enabled),
			"auto_merge_enabled": bool(auto_merge_auto_accept_enabled),
			"any_enabled": bool(auto_accept_enabled_any),
		},
		"before": pre_snapshot,
		"after": post_snapshot,
		"delta": {
			"sorter": sorter_delta,
			"analyzer": analyzer_delta,
			"any_changed": bool(observed_change_any),
		},
		"applied_merge_operations": applied_operations,
		"applied_merge_group_count": int(applied_group_count),
		"applied_unit_mappings": applied_mappings,
		"change_validation": {
			"expected_change_if_auto_accept_enabled": bool(auto_accept_enabled_any),
			"expected_change_if_merges_applied": bool(expected_change_if_applied),
			"observed_change": bool(observed_change_any),
			"merge_target_unit_ids": list(merge_target_unit_ids),
			"missing_merge_target_unit_ids": bool(missing_mapping_targets),
			"merge_target_ids_missing_from_sorter_post": list(missing_targets_sorter),
			"merge_target_ids_missing_from_analyzer_post": list(missing_targets_analyzer),
			"passes": bool(validation_passes),
			"reason": str(validation_reason),
		},
	}


def _build_snapshot_metadata_summary(*, snapshot_label: str, snapshot: dict[str, Any]) -> dict[str, Any]:
	sorter_payload_raw = snapshot.get("sorter", {})
	analyzer_payload_raw = snapshot.get("analyzer", {})
	sorter_payload = (dict(sorter_payload_raw) if isinstance(sorter_payload_raw, dict) else {})
	analyzer_payload = (dict(analyzer_payload_raw) if isinstance(analyzer_payload_raw, dict) else {})

	sorter_ids = _normalize_unit_id_list(sorter_payload.get("unit_ids", []))
	analyzer_ids = _normalize_unit_id_list(analyzer_payload.get("unit_ids", []))

	return {
		"snapshot_label": str(snapshot_label),
		"sorter": sorter_payload,
		"analyzer": analyzer_payload,
		"summary": {
			"sorter": {
				"available": bool(sorter_payload.get("available", False)),
				"unit_count": int(_snapshot_unit_count(sorter_payload, sorter_ids)),
				"unit_ids": sorter_ids,
			},
			"analyzer": {
				"available": bool(analyzer_payload.get("available", False)),
				"unit_count": int(_snapshot_unit_count(analyzer_payload, analyzer_ids)),
				"unit_ids": analyzer_ids,
				"has_sparsity": analyzer_payload.get("has_sparsity", None),
				"requested_compute_sparsity": bool(
					analyzer_payload.get("requested_compute_sparsity", True)
				),
				"requested_sparsity_method": analyzer_payload.get(
					"requested_sparsity_method",
					"radius",
				),
				"requested_template_random_spikes_method": analyzer_payload.get(
					"requested_template_random_spikes_method",
					"default",
				),
				"requested_template_random_spikes_percentage": analyzer_payload.get(
					"requested_template_random_spikes_percentage",
					None,
				),
				"requested_template_random_spikes_log_before_after_spike_counts": analyzer_payload.get(
					"requested_template_random_spikes_log_before_after_spike_counts",
					False,
				),
				"requested_template_random_spikes_min_spikes_per_unit": analyzer_payload.get(
					"requested_template_random_spikes_min_spikes_per_unit",
					None,
				),
				"sparsity_validation_error": analyzer_payload.get("sparsity_validation_error", None),
			},
		},
	}


def _extract_unit_locations_for_plot(snapshot_payload: dict[str, Any]) -> dict[str, tuple[float, float]]:
	analyzer_payload_raw = snapshot_payload.get("analyzer", {})
	analyzer_payload = (analyzer_payload_raw if isinstance(analyzer_payload_raw, dict) else {})
	locations_raw = analyzer_payload.get("unit_locations_by_unit", {})
	if not isinstance(locations_raw, dict):
		return {}

	points: dict[str, tuple[float, float]] = {}
	for raw_uid, loc_raw in locations_raw.items():
		if not isinstance(loc_raw, dict):
			continue
		x_raw = loc_raw.get("x_um", loc_raw.get("x", None))
		y_raw = loc_raw.get("y_um", loc_raw.get("y", None))
		try:
			x = float(x_raw)
			y = float(y_raw)
		except Exception:
			continue
		if not (math.isfinite(x) and math.isfinite(y)):
			continue
		uid = _normalize_cluster_id(raw_uid)
		if uid not in points:
			points[uid] = (x, y)
	return points


def _resolve_report_image_path(*, out_dir: Path, relpath: str, format_name: str) -> Path:
	path = (out_dir / str(relpath).strip().lstrip("/")).resolve()
	suffix = (".svg" if str(format_name).strip().lower() == "svg" else ".png")
	if path.suffix.lower() != suffix:
		path = path.with_suffix(suffix)
	return path


def _safe_file_token(raw: Any) -> str:
	token = str(raw or "").strip()
	if not token:
		return "unknown"
	out_chars: list[str] = []
	for ch in token:
		if ch.isalnum() or ch in ("-", "_"):
			out_chars.append(ch)
		else:
			out_chars.append("_")
	out = "".join(out_chars).strip("_")
	return (out or "unknown")


def _load_sorting_analyzer_from_snapshot(
	*,
	si_module: Any,
	snapshot: dict[str, Any],
	well_out_dir: Path | None = None,
	stage_config: Any | None = None,
) -> tuple[Any | None, str | None]:
	load_sorting_analyzer = getattr(si_module, "load_sorting_analyzer", None)
	if not callable(load_sorting_analyzer):
		return None, "load_sorting_analyzer_api_unavailable"

	analyzer_raw = snapshot.get("analyzer", {})
	analyzer_payload = (analyzer_raw if isinstance(analyzer_raw, dict) else {})
	source_dir_raw = analyzer_payload.get("source_dir", None)
	if source_dir_raw is None:
		return None, "analyzer_source_dir_missing"

	source_dir = Path(str(source_dir_raw)).resolve()
	if not source_dir.exists():
		return None, f"analyzer_source_dir_missing:{source_dir}"

	try:
		resolved_well_out_dir = well_out_dir
		if resolved_well_out_dir is None:
			for parent in source_dir.parents:
				if parent.name == "spikesort_outputs":
					resolved_well_out_dir = parent.parent
					break
		if resolved_well_out_dir is not None and stage_config is not None:
			return (
				_load_spikesort_analyzer_with_recording(
					si_module=si_module,
					analyzer_dir=source_dir,
					well_out_dir=resolved_well_out_dir,
					stage_config=stage_config,
				),
				None,
			)
		return load_sorting_analyzer(source_dir), None
	except Exception as exc:
		return None, f"load_sorting_analyzer_failed:{type(exc).__name__}:{exc}"


def _extract_template_and_locations_for_unit(
	*,
	analyzer: Any,
	unit_id: str,
	stage_config: Any | None = None,
) -> tuple[Any | None, Any | None, str | None]:
	import numpy as np  # type: ignore[import-not-found]

	has_extension = getattr(analyzer, "has_extension", None)
	compute_extension = getattr(analyzer, "compute", None)
	get_extension = getattr(analyzer, "get_extension", None)
	if not callable(has_extension) or not callable(get_extension):
		return None, None, "analyzer_extension_api_unavailable"

	try:
		has_templates = bool(has_extension("templates"))
	except Exception:
		has_templates = False

	if (not has_templates) and stage_config is not None:
		_ensure_merge_analyzer_extensions(
			analyzer=analyzer,
			stage_config=stage_config,
			include_unit_locations=False,
		)
		try:
			has_templates = bool(has_extension("templates"))
		except Exception:
			has_templates = False

	if (not has_templates) and callable(compute_extension):
		if stage_config is not None:
			if _compute_analyzer_extension(
				analyzer=analyzer,
				extension_name="templates",
				kwargs=_merge_analyzer_extension_kwargs(stage_config, "templates"),
			):
				try:
					has_templates = bool(has_extension("templates"))
				except Exception:
					has_templates = False
		else:
			for candidate in ("templates", ["templates"]):
				try:
					compute_extension(candidate)
					if bool(has_extension("templates")):
						has_templates = True
						break
				except Exception:
					continue

	if not has_templates:
		return None, None, "templates_extension_missing"

	try:
		templates_ext = get_extension("templates")
	except Exception as exc:
		return None, None, f"templates_extension_load_failed:{type(exc).__name__}:{exc}"

	template_arr: Any | None = None
	get_unit_template = getattr(templates_ext, "get_unit_template", None)
	unit_ids = _unit_ids_from_obj(analyzer)
	canonical_unit_id: Any | None = None
	for uid in unit_ids:
		if uid == unit_id or str(uid) == str(unit_id):
			canonical_unit_id = uid
			break
	if canonical_unit_id is None:
		return None, None, "unit_id_not_found_in_templates"

	if not callable(get_unit_template):
		return None, None, "templates_get_unit_template_api_unavailable"

	def _unit_id_forms(raw_uid: Any) -> list[Any]:
		forms: list[Any] = []

		def _add(value: Any) -> None:
			for existing in forms:
				if existing == value and type(existing) is type(value):
					return
			forms.append(value)

		_add(raw_uid)
		uid_text = str(raw_uid)
		try:
			_add(int(uid_text))
		except Exception:
			pass
		return forms

	template_load_errors: list[str] = []
	for candidate_unit_id in _unit_id_forms(canonical_unit_id):
		try:
			template_arr = np.asarray(get_unit_template(unit_id=candidate_unit_id), dtype=float)
			if template_arr is not None:
				break
		except Exception as exc:
			template_load_errors.append(
				f"{repr(candidate_unit_id)}:{type(exc).__name__}:{exc}"
			)

	if template_arr is None:
		if template_load_errors:
			return None, None, "unit_template_load_failed:" + " | ".join(template_load_errors)
		return None, None, "unit_template_load_failed:unknown_error"

	if template_arr is None:
		return None, None, "unit_template_unavailable"

	if template_arr.ndim != 2:
		return None, None, "unit_template_not_2d"

	recording = getattr(analyzer, "recording", None)
	if recording is None:
		get_recording = getattr(analyzer, "get_recording", None)
		if callable(get_recording):
			try:
				recording = get_recording()
			except Exception:
				recording = None
	if recording is None:
		return None, None, "analyzer_recording_unavailable"

	get_channel_locations = getattr(recording, "get_channel_locations", None)
	if not callable(get_channel_locations):
		return None, None, "recording_channel_locations_api_unavailable"

	try:
		locations = np.asarray(get_channel_locations(), dtype=float)
	except Exception as exc:
		return None, None, f"channel_locations_load_failed:{type(exc).__name__}:{exc}"

	if locations.ndim != 2 or locations.shape[1] < 2:
		return None, None, "channel_locations_not_2d"

	if template_arr.shape[0] != locations.shape[0]:
		sparsity = getattr(analyzer, "sparsity", None)
		indices: Any | None = None
		lookup_unit_id = canonical_unit_id if canonical_unit_id is not None else unit_id
		lookup_forms = _unit_id_forms(lookup_unit_id)
		if sparsity is not None:
			mapping = getattr(sparsity, "unit_id_to_channel_indices", None)
			if callable(mapping):
				for candidate_lookup in lookup_forms:
					try:
						indices = mapping(candidate_lookup)
						if indices is not None:
							break
					except Exception:
						continue
			elif isinstance(mapping, dict):
				for candidate_lookup in lookup_forms:
					if candidate_lookup in mapping:
						indices = mapping.get(candidate_lookup)
						break
					candidate_lookup_text = str(candidate_lookup)
					if candidate_lookup_text in mapping:
						indices = mapping.get(candidate_lookup_text)
						break
			if indices is None:
				for name in ("get_channel_indices", "get_channel_indices_for_unit"):
					fn = getattr(sparsity, name, None)
					if callable(fn):
						for candidate_lookup in lookup_forms:
							try:
								indices = fn(candidate_lookup)
								if indices is not None:
									break
							except Exception:
								continue
						if indices is not None:
							break

		if indices is not None:
			try:
				idx = np.asarray(indices, dtype=int)
				if idx.ndim == 1 and template_arr.shape[0] == idx.shape[0]:
					locations = locations[idx, :]
			except Exception:
				pass

	if template_arr.shape[0] != locations.shape[0] and template_arr.shape[1] == locations.shape[0]:
		template_arr = np.asarray(template_arr.T, dtype=float)

	if template_arr.shape[0] != locations.shape[0]:
		return None, None, "template_channel_count_mismatch"

	return np.asarray(template_arr, dtype=float), np.asarray(locations[:, :2], dtype=float), None


def _compute_template_channel_magnitude(*, template_ch_by_t: Any, magnitude_mode: str) -> Any:
	import numpy as np  # type: ignore[import-not-found]

	tmpl = np.asarray(template_ch_by_t, dtype=float)
	if tmpl.ndim != 2:
		return np.asarray([], dtype=float)

	mode = _normalize_merge_template_heatmap_magnitude_mode(magnitude_mode)
	if mode == "abs_peak":
		return np.nanmax(np.abs(tmpl), axis=1)
	if mode == "peak":
		return np.nanmax(tmpl, axis=1)
	if mode == "trough":
		return np.abs(np.nanmin(tmpl, axis=1))
	return np.ptp(tmpl, axis=1)


def _prepare_template_heatmap_scatter_payload(
	*,
	template_ch_by_t: Any,
	locations_xy: Any,
	color_vmin: float | None = None,
	color_vmax: float | None = None,
	color_scale_mode: str = "linear",
	log_epsilon: float = 1e-3,
	magnitude_mode: str = "ptp",
) -> tuple[dict[str, Any] | None, str | None]:
	from matplotlib import colors as mcolors  # type: ignore[import-not-found]
	import numpy as np  # type: ignore[import-not-found]

	tmpl = np.asarray(template_ch_by_t, dtype=float)
	locs = np.asarray(locations_xy, dtype=float)
	if tmpl.ndim != 2:
		return None, "template_not_2d"
	if locs.ndim != 2 or locs.shape[1] < 2:
		return None, "locations_not_2d"
	if tmpl.shape[0] != locs.shape[0]:
		return None, "template_location_shape_mismatch"

	amp = np.asarray(
		_compute_template_channel_magnitude(
			template_ch_by_t=tmpl,
			magnitude_mode=magnitude_mode,
		),
		dtype=float,
	)
	if amp.size == 0:
		return None, "template_empty"

	vmin: float | None = None
	vmax: float | None = None
	norm: Any | None = None
	try:
		if color_vmin is not None and color_vmax is not None:
			vmin_candidate = float(color_vmin)
			vmax_candidate = float(color_vmax)
			if math.isfinite(vmin_candidate) and math.isfinite(vmax_candidate) and (vmax_candidate > vmin_candidate):
				vmin = vmin_candidate
				vmax = vmax_candidate
	except Exception:
		vmin = None
		vmax = None

	mode = str(color_scale_mode or "linear").strip().lower()
	if mode in {"log10", "logarithmic"}:
		mode = "log"
	if mode not in {"linear", "log"}:
		mode = "linear"

	if mode == "log":
		try:
			eps = float(log_epsilon)
		except Exception:
			eps = 1e-3
		if (not math.isfinite(eps)) or eps <= 0.0:
			eps = 1e-3

		positive_amp = amp[np.isfinite(amp) & (amp > 0.0)]
		if positive_amp.size <= 0:
			return None, "template_amp_nonpositive_for_log_scale"

		positive_min = float(np.nanmin(positive_amp))
		positive_max = float(np.nanmax(positive_amp))
		if not (math.isfinite(positive_min) and math.isfinite(positive_max) and positive_max > 0.0):
			return None, "template_amp_invalid_for_log_scale"

		if vmin is None or vmax is None:
			vmin = max(eps, positive_min)
			vmax = max(vmin * (1.0 + 1e-6), positive_max)
		else:
			vmin = max(eps, vmin)
			vmax = max(vmin * (1.0 + 1e-6), vmax)

		norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)

	return {
		"amp": amp,
		"locs": np.asarray(locs[:, :2], dtype=float),
		"norm": norm,
		"vmin": vmin,
		"vmax": vmax,
		"color_scale_mode": mode,
	}, None


def _plot_template_amplitude_heatmap_on_axis(
	*,
	ax: Any,
	template_ch_by_t: Any,
	locations_xy: Any,
	title: str,
	cmap: str,
	marker_size: float,
	color_vmin: float | None = None,
	color_vmax: float | None = None,
	color_scale_mode: str = "linear",
	log_epsilon: float = 1e-3,
	magnitude_mode: str = "ptp",
	x_limits: tuple[float, float] | None = None,
	y_limits: tuple[float, float] | None = None,
	show_axis_labels: bool = True,
	title_fontsize: float | None = None,
	tick_labelsize: float | None = None,
) -> tuple[Any | None, str | None]:
	payload, err = _prepare_template_heatmap_scatter_payload(
		template_ch_by_t=template_ch_by_t,
		locations_xy=locations_xy,
		color_vmin=color_vmin,
		color_vmax=color_vmax,
		color_scale_mode=color_scale_mode,
		log_epsilon=log_epsilon,
		magnitude_mode=magnitude_mode,
	)
	if payload is None:
		return None, err

	scatter_kwargs = {
		"c": payload["amp"],
		"cmap": str(cmap),
		"s": float(max(0.1, float(marker_size))),
		"alpha": 0.95,
	}
	if payload["norm"] is not None:
		scatter_kwargs["norm"] = payload["norm"]
	else:
		scatter_kwargs["vmin"] = payload["vmin"]
		scatter_kwargs["vmax"] = payload["vmax"]

	sc = ax.scatter(
		payload["locs"][:, 0],
		payload["locs"][:, 1],
		**scatter_kwargs,
	)
	if title_fontsize is None:
		ax.set_title(str(title))
	else:
		ax.set_title(str(title), fontsize=float(title_fontsize), pad=6.0)
	if show_axis_labels:
		ax.set_xlabel("x_um")
		ax.set_ylabel("y_um")
	else:
		ax.set_xlabel("")
		ax.set_ylabel("")
	if x_limits is not None:
		ax.set_xlim(x_limits)
	if y_limits is not None:
		ax.set_ylim(y_limits)
	ax.invert_yaxis()
	ax.set_aspect("equal", adjustable="box")
	ax.grid(True, alpha=0.2)
	if tick_labelsize is not None and callable(getattr(ax, "tick_params", None)):
		try:
			ax.tick_params(axis="both", labelsize=float(tick_labelsize))
		except Exception:
			pass
	return sc, None


def _compute_template_heatmap_scale_limits(
	*,
	templates: list[Any],
	color_scale_mode: str,
	log_epsilon: float,
	magnitude_mode: str,
) -> tuple[float | None, float | None]:
	import numpy as np  # type: ignore[import-not-found]

	mins: list[float] = []
	maxs: list[float] = []
	mode = str(color_scale_mode or "linear").strip().lower()
	if mode in {"log10", "logarithmic"}:
		mode = "log"
	if mode not in {"linear", "log"}:
		mode = "linear"

	try:
		eps = float(log_epsilon)
	except Exception:
		eps = 1e-3
	if (not math.isfinite(eps)) or eps <= 0.0:
		eps = 1e-3

	for template_arr in list(templates or []):
		try:
			amp = np.asarray(
				_compute_template_channel_magnitude(
					template_ch_by_t=np.asarray(template_arr, dtype=float),
					magnitude_mode=magnitude_mode,
				),
				dtype=float,
			)
		except Exception:
			continue
		if amp.size <= 0:
			continue
		if mode == "log":
			amp = amp[np.isfinite(amp) & (amp > 0.0)]
			if amp.size <= 0:
				continue
		amp_min = float(np.nanmin(amp))
		amp_max = float(np.nanmax(amp))
		if math.isfinite(amp_min) and math.isfinite(amp_max):
			mins.append(amp_min)
			maxs.append(amp_max)

	if not mins or not maxs:
		return None, None

	vmin = float(min(mins))
	vmax = float(max(maxs))
	if mode == "log":
		vmin = max(eps, vmin)
		vmax = max(vmin * (1.0 + 1e-6), vmax)
	elif not (math.isfinite(vmin) and math.isfinite(vmax) and (vmax > vmin)):
		return None, None
	return vmin, vmax


def _write_template_amplitude_heatmap_asset(
	*,
	template_ch_by_t: Any,
	locations_xy: Any,
	out_path: Path,
	title: str,
	cmap: str,
	marker_size: float,
	show_colorbar: bool,
	relative_color_bar_height: float = 1.0,
	color_vmin: float | None = None,
	color_vmax: float | None = None,
	color_scale_mode: str = "linear",
	log_epsilon: float = 1e-3,
	magnitude_mode: str = "ptp",
	x_limits: tuple[float, float] | None = None,
	y_limits: tuple[float, float] | None = None,
) -> tuple[bool, str | None]:
	_prepare_matplotlib_for_headless_rendering()
	import matplotlib.pyplot as plt  # type: ignore[import-not-found]

	fig, ax = plt.subplots(1, 1, figsize=(4.2, 4.0), constrained_layout=True)
	try:
		sc, err = _plot_template_amplitude_heatmap_on_axis(
			ax=ax,
			template_ch_by_t=template_ch_by_t,
			locations_xy=locations_xy,
			title=title,
			cmap=cmap,
			marker_size=marker_size,
			color_vmin=color_vmin,
			color_vmax=color_vmax,
			color_scale_mode=color_scale_mode,
			log_epsilon=log_epsilon,
			magnitude_mode=magnitude_mode,
			x_limits=x_limits,
			y_limits=y_limits,
		)
		if sc is None:
			return False, (err or "template_amp_render_failed")
		if bool(show_colorbar):
			fig.colorbar(
				sc,
				ax=ax,
				shrink=_normalize_template_heatmap_relative_color_bar_height(
					relative_color_bar_height
				),
				pad=0.02,
			)

		out_path.parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(out_path, dpi=320)
		return True, None
	except Exception as exc:
		return False, f"template_amp_render_failed:{type(exc).__name__}:{exc}"
	finally:
		plt.close(fig)


def _probe_plot_limits_from_dimensions(
	*,
	probe_dim_x_um: Any = None,
	probe_dim_y_um: Any = None,
) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
	try:
		probe_dim_x = (float(probe_dim_x_um) if probe_dim_x_um is not None else None)
	except Exception:
		probe_dim_x = None
	try:
		probe_dim_y = (float(probe_dim_y_um) if probe_dim_y_um is not None else None)
	except Exception:
		probe_dim_y = None
	if probe_dim_x is not None and (not math.isfinite(probe_dim_x) or probe_dim_x <= 0.0):
		probe_dim_x = None
	if probe_dim_y is not None and (not math.isfinite(probe_dim_y) or probe_dim_y <= 0.0):
		probe_dim_y = None
	if probe_dim_x is None or probe_dim_y is None:
		return None, None
	return (0.0, float(probe_dim_x)), (0.0, float(probe_dim_y))


def _probe_relative_marker_size_points2(
	*,
	probe_dim_x_um: Any = None,
	probe_dim_y_um: Any = None,
	probe_pitch_um: Any = None,
	electrode_size_um_x: Any = None,
	electrode_size_um_y: Any = None,
	figure_width_in: float = 4.2,
	figure_height_in: float = 4.0,
) -> tuple[float | None, str | None]:
	try:
		x_range_um = float(probe_dim_x_um) if probe_dim_x_um is not None else None
	except Exception:
		x_range_um = None
	try:
		y_range_um = float(probe_dim_y_um) if probe_dim_y_um is not None else None
	except Exception:
		y_range_um = None
	if x_range_um is None or y_range_um is None:
		return None, None
	if (not math.isfinite(x_range_um)) or x_range_um <= 0.0:
		return None, None
	if (not math.isfinite(y_range_um)) or y_range_um <= 0.0:
		return None, None

	try:
		pitch_um = float(probe_pitch_um) if probe_pitch_um is not None else None
	except Exception:
		pitch_um = None
	if pitch_um is not None and ((not math.isfinite(pitch_um)) or pitch_um <= 0.0):
		pitch_um = None

	try:
		electrode_x_um = float(electrode_size_um_x) if electrode_size_um_x is not None else None
	except Exception:
		electrode_x_um = None
	if electrode_x_um is not None and ((not math.isfinite(electrode_x_um)) or electrode_x_um <= 0.0):
		electrode_x_um = None

	try:
		electrode_y_um = float(electrode_size_um_y) if electrode_size_um_y is not None else None
	except Exception:
		electrode_y_um = None
	if electrode_y_um is not None and ((not math.isfinite(electrode_y_um)) or electrode_y_um <= 0.0):
		electrode_y_um = None

	span_x_um: float | None = None
	span_y_um: float | None = None
	source: str | None = None
	if electrode_x_um is not None or electrode_y_um is not None:
		span_x_um = electrode_x_um if electrode_x_um is not None else (pitch_um if pitch_um is not None else electrode_y_um)
		span_y_um = electrode_y_um if electrode_y_um is not None else (pitch_um if pitch_um is not None else electrode_x_um)
		source = "probe_geometry_electrode_size"
	elif pitch_um is not None:
		span_x_um = pitch_um
		span_y_um = pitch_um
		source = "probe_geometry_pitch"
	if span_x_um is None or span_y_um is None or source is None:
		return None, None

	try:
		width_in = float(figure_width_in)
		height_in = float(figure_height_in)
	except Exception:
		return None, None
	if (not math.isfinite(width_in)) or width_in <= 0.0:
		return None, None
	if (not math.isfinite(height_in)) or height_in <= 0.0:
		return None, None

	points_per_um = 72.0 * min(width_in / x_range_um, height_in / y_range_um)
	if (not math.isfinite(points_per_um)) or points_per_um <= 0.0:
		return None, None

	width_points = points_per_um * float(span_x_um)
	height_points = points_per_um * float(span_y_um)
	area_points2 = 0.85 * (math.pi * 0.25) * width_points * height_points
	if (not math.isfinite(area_points2)) or area_points2 <= 0.0:
		return None, None
	return max(0.1, float(area_points2)), source


def _normalize_template_heatmap_relative_color_bar_height(raw: Any, *, default: float = 1.0) -> float:
	try:
		value = float(raw if raw is not None else default)
	except Exception:
		value = float(default)
	if (not math.isfinite(value)) or value <= 0.0:
		value = float(default)
	return max(0.05, min(1.0, float(value)))


def _stack_rendered_images_vertically(*, image_paths: list[Path]) -> tuple[Any | None, str | None]:
	_prepare_matplotlib_for_headless_rendering()
	import matplotlib.pyplot as plt  # type: ignore[import-not-found]
	import numpy as np  # type: ignore[import-not-found]

	if not image_paths:
		return None, "no_images"

	images: list[Any] = []
	for path in image_paths:
		if not path.exists():
			continue
		try:
			img = np.asarray(plt.imread(path))
		except Exception:
			continue
		if img.ndim == 2:
			img = np.repeat(img[:, :, None], 3, axis=2)
		elif img.ndim == 3 and img.shape[2] == 1:
			img = np.repeat(img, 3, axis=2)
		elif img.ndim != 3:
			continue
		images.append(np.asarray(img, dtype=float))

	if not images:
		return None, "images_unreadable"

	max_width = max(int(img.shape[1]) for img in images)
	max_channels = max(int(img.shape[2]) for img in images)
	if max_channels <= 0:
		max_channels = 3

	separator_h = 10
	separator = np.ones((separator_h, max_width, max_channels), dtype=float)

	padded: list[Any] = []
	for img in images:
		arr = np.asarray(img, dtype=float)
		if arr.shape[2] < max_channels:
			pad_c = np.ones((arr.shape[0], arr.shape[1], max_channels - arr.shape[2]), dtype=float)
			arr = np.concatenate([arr, pad_c], axis=2)
		if arr.shape[1] < max_width:
			pad_w = np.ones((arr.shape[0], max_width - arr.shape[1], arr.shape[2]), dtype=float)
			arr = np.concatenate([arr, pad_w], axis=1)
		padded.append(arr)

	stacked = padded[0]
	for arr in padded[1:]:
		stacked = np.concatenate([stacked, separator, arr], axis=0)
	return stacked, None


def _stack_rendered_images_horizontally(*, image_paths: list[Path]) -> tuple[Any | None, str | None]:
	_prepare_matplotlib_for_headless_rendering()
	import matplotlib.pyplot as plt  # type: ignore[import-not-found]
	import numpy as np  # type: ignore[import-not-found]

	if not image_paths:
		return None, "no_images"

	images: list[Any] = []
	for path in image_paths:
		if not path.exists():
			continue
		try:
			img = np.asarray(plt.imread(path))
		except Exception:
			continue
		if img.ndim == 2:
			img = np.repeat(img[:, :, None], 3, axis=2)
		elif img.ndim == 3 and img.shape[2] == 1:
			img = np.repeat(img, 3, axis=2)
		elif img.ndim != 3:
			continue
		images.append(np.asarray(img, dtype=float))

	if not images:
		return None, "images_unreadable"

	max_height = max(int(img.shape[0]) for img in images)
	max_channels = max(int(img.shape[2]) for img in images)
	if max_channels <= 0:
		max_channels = 3

	separator_w = 10
	separator = np.ones((max_height, separator_w, max_channels), dtype=float)

	padded: list[Any] = []
	for img in images:
		arr = np.asarray(img, dtype=float)
		if arr.shape[2] < max_channels:
			pad_c = np.ones((arr.shape[0], arr.shape[1], max_channels - arr.shape[2]), dtype=float)
			arr = np.concatenate([arr, pad_c], axis=2)
		if arr.shape[0] < max_height:
			pad_h = np.ones((max_height - arr.shape[0], arr.shape[1], arr.shape[2]), dtype=float)
			arr = np.concatenate([arr, pad_h], axis=0)
		padded.append(arr)

	stacked = padded[0]
	for arr in padded[1:]:
		stacked = np.concatenate([stacked, separator, arr], axis=1)
	return stacked, None


def _write_merge_template_heatmap_reports(
	*,
	merge_out_dir: Path,
	before_snapshot: dict[str, Any],
	after_snapshot: dict[str, Any],
	applied_unit_mappings: list[dict[str, Any]] | None,
	stage_config: Any,
	before_analyzer: Any | None = None,
	after_analyzer: Any | None = None,
) -> dict[str, Any]:
	_prepare_matplotlib_for_headless_rendering()
	import matplotlib.pyplot as plt  # type: ignore[import-not-found]
	import numpy as np  # type: ignore[import-not-found]

	if not isinstance(applied_unit_mappings, list) or not applied_unit_mappings:
		return {
			"status": "skipped",
			"reason": "template_heatmaps_no_applied_unit_mappings",
			"outputs": {},
		}

	_log_phase_step_start(
		"Merge template heatmap report step start",
		merge_out_dir=merge_out_dir,
		applied_mappings=int(len(applied_unit_mappings)),
	)

	report_relpath = str(
		getattr(stage_config, "merge_reports_template_heatmaps_relpath", "template_heatmaps_per_merge")
		or "template_heatmaps_per_merge"
	).strip().lstrip("/") or "template_heatmaps_per_merge"
	assets_reldir = str(
		getattr(stage_config, "merge_reports_template_heatmaps_assets_reldir", "assets") or "assets"
	).strip().lstrip("/") or "assets"
	report_root_dir = (merge_out_dir / report_relpath).resolve()
	assets_root_dir = (report_root_dir / assets_reldir).resolve()

	write_png = bool(getattr(stage_config, "merge_reports_template_heatmaps_write_png", True))
	write_svg = bool(getattr(stage_config, "merge_reports_template_heatmaps_write_svg", False))
	write_assets_png = bool(getattr(stage_config, "merge_reports_template_heatmaps_write_assets_png", True))
	write_assets_svg = bool(getattr(stage_config, "merge_reports_template_heatmaps_write_assets_svg", False))

	# Keep PNG assets enabled whenever panel output is requested so individual unit panels remain available.
	write_assets_png = bool(write_assets_png or write_png or write_svg)

	panel_width_in = float(getattr(stage_config, "merge_reports_template_heatmaps_panel_width_in", 11.0) or 11.0)
	panel_height_in = float(getattr(stage_config, "merge_reports_template_heatmaps_panel_height_in", 6.0) or 6.0)
	try:
		requested_marker_size = float(
			getattr(stage_config, "merge_reports_template_heatmaps_marker_size", 10.0) or 10.0
		)
	except Exception:
		requested_marker_size = 10.0
	if (not math.isfinite(requested_marker_size)) or requested_marker_size <= 0.0:
		requested_marker_size = 10.0
	cmap = str(getattr(stage_config, "merge_reports_template_heatmaps_cmap", "viridis") or "viridis")
	show_colorbar = bool(getattr(stage_config, "merge_reports_template_heatmaps_show_colorbar", True))
	relative_color_bar_height = _normalize_template_heatmap_relative_color_bar_height(
		getattr(stage_config, "merge_reports_template_heatmaps_relative_color_bar_height", 1.0)
	)
	color_scale_mode_raw = str(
		getattr(stage_config, "merge_reports_template_heatmaps_color_scale", "linear") or "linear"
	)
	color_scale_mode = color_scale_mode_raw.strip().lower()
	if color_scale_mode in {"log10", "logarithmic"}:
		color_scale_mode = "log"
	if color_scale_mode not in {"linear", "log"}:
		color_scale_mode = "linear"
	try:
		log_epsilon = float(getattr(stage_config, "merge_reports_template_heatmaps_log_epsilon", 1e-3) or 1e-3)
	except Exception:
		log_epsilon = 1e-3
	if (not math.isfinite(log_epsilon)) or log_epsilon <= 0.0:
		log_epsilon = 1e-3
	magnitude_mode = _merge_template_heatmap_magnitude_mode(stage_config)
	max_merges_raw = getattr(stage_config, "merge_reports_template_heatmaps_max_merges", None)
	debug_json_relpath = str(
		getattr(
			stage_config,
			"merge_reports_template_heatmaps_debug_json_relpath",
			"template_heatmaps_per_merge_report.json",
		)
		or "template_heatmaps_per_merge_report.json"
	).strip().lstrip("/") or "template_heatmaps_per_merge_report.json"
	inherit_probe_dimensions = bool(
		getattr(stage_config, "merge_reports_template_heatmaps_inherit_probe_dimensions", False)
	)
	probe_x_limits, probe_y_limits = _probe_plot_limits_from_dimensions(
		probe_dim_x_um=getattr(stage_config, "merge_reports_template_heatmaps_probe_dim_x_um", None),
		probe_dim_y_um=getattr(stage_config, "merge_reports_template_heatmaps_probe_dim_y_um", None),
	)
	marker_size = float(requested_marker_size)
	marker_size_source = "configured"
	if inherit_probe_dimensions:
		probe_marker_size, probe_marker_size_source = _probe_relative_marker_size_points2(
			probe_dim_x_um=getattr(stage_config, "merge_reports_template_heatmaps_probe_dim_x_um", None),
			probe_dim_y_um=getattr(stage_config, "merge_reports_template_heatmaps_probe_dim_y_um", None),
			probe_pitch_um=getattr(stage_config, "merge_reports_template_heatmaps_probe_pitch_um", None),
			electrode_size_um_x=getattr(
				stage_config,
				"merge_reports_template_heatmaps_probe_electrode_size_um_x",
				None,
			),
			electrode_size_um_y=getattr(
				stage_config,
				"merge_reports_template_heatmaps_probe_electrode_size_um_y",
				None,
			),
		)
		if probe_marker_size is not None:
			marker_size = float(probe_marker_size)
			marker_size_source = str(probe_marker_size_source or "probe_geometry")

	max_merges: int | None
	try:
		max_merges = int(max_merges_raw) if max_merges_raw is not None else None
		if max_merges is not None and max_merges <= 0:
			max_merges = None
	except Exception:
		max_merges = None

	loaded_before_analyzer = False
	loaded_after_analyzer = False
	before_error: str | None = None
	after_error: str | None = None
	si_module: Any | None = None
	if before_analyzer is None or after_analyzer is None:
		try:
			si_module = _import_spikeinterface_full_module()
		except Exception as exc:
			return {
				"status": "error",
				"error": f"template_heatmaps_spikeinterface_import_failed:{type(exc).__name__}:{exc}",
				"outputs": {},
			}

	if before_analyzer is None:
		before_analyzer, before_error = _load_sorting_analyzer_from_snapshot(
			si_module=si_module,
			snapshot=before_snapshot,
			stage_config=stage_config,
		)
		loaded_before_analyzer = bool(before_analyzer is not None)
	if after_analyzer is None:
		after_analyzer, after_error = _load_sorting_analyzer_from_snapshot(
			si_module=si_module,
			snapshot=after_snapshot,
			stage_config=stage_config,
		)
		loaded_after_analyzer = bool(after_analyzer is not None)
	if before_analyzer is None or after_analyzer is None:
		return {
			"status": "error",
			"error": "template_heatmaps_analyzer_load_failed",
			"before_analyzer_error": before_error,
			"after_analyzer_error": after_error,
			"outputs": {},
		}

	_ensure_merge_analyzer_extensions(
		analyzer=before_analyzer,
		stage_config=stage_config,
		include_unit_locations=False,
	)
	_ensure_merge_analyzer_extensions(
		analyzer=after_analyzer,
		stage_config=stage_config,
		include_unit_locations=False,
	)

	# Strict mode by design: use only snapshot_before as pre-merge template source.
	before_candidates: list[tuple[str, Any]] = [("snapshot_before", before_analyzer)]

	required_pre_unit_ids: set[str] = set()
	for mapping_raw in list(applied_unit_mappings or []):
		mapping = (mapping_raw if isinstance(mapping_raw, dict) else {})
		required_pre_unit_ids.update(_normalize_unit_id_list(mapping.get("pre_unit_ids", [])))

	covered_pre_unit_ids: set[str] = set()
	for _label, candidate_analyzer in before_candidates:
		covered_pre_unit_ids.update(_normalize_unit_id_list(_unit_ids_from_obj(candidate_analyzer)))

	missing_pre_unit_ids = sorted(
		[uid for uid in required_pre_unit_ids if uid not in covered_pre_unit_ids],
		key=_unit_sort_key,
	)
	# No retries from alternate sources; missing pre unit ids remain missing and are reported.

	merge_rows: list[dict[str, Any]] = []
	outputs: dict[str, str] = {}
	processed = 0
	for mapping_idx, mapping_raw in enumerate(applied_unit_mappings, start=1):
		if max_merges is not None and processed >= max_merges:
			break
		mapping = (mapping_raw if isinstance(mapping_raw, dict) else {})
		pre_unit_ids = _normalize_unit_id_list(mapping.get("pre_unit_ids", []))
		post_unit_id = _normalize_cluster_id(mapping.get("post_unit_id", None))
		if not pre_unit_ids:
			continue

		processed += 1
		group_id_raw = str(mapping.get("group_id", f"group_{int(mapping_idx):03d}"))
		group_token = _safe_file_token(group_id_raw)
		row: dict[str, Any] = {
			"mapping_index": int(mapping_idx),
			"group_id": group_id_raw,
			"method": str(mapping.get("method", "unknown")),
			"pre_unit_ids": list(pre_unit_ids),
			"post_unit_id": post_unit_id,
			"pre_assets": [],
			"post_asset": None,
			"panel_outputs": {},
		}

		panel_base = f"merge_{int(mapping_idx):03d}__{group_token}"
		pre_png_assets: list[Path] = []
		pre_render_items: list[tuple[dict[str, Any], Any, Any, str]] = []
		panel_render_items: list[dict[str, Any]] = []

		for pre_uid in pre_unit_ids:
			template: Any | None = None
			locations_xy: Any | None = None
			template_error: str | None = None
			template_source: str | None = None
			candidate_errors: list[str] = []
			for candidate_label, candidate_analyzer in before_candidates:
				template, locations_xy, template_error = _extract_template_and_locations_for_unit(
					analyzer=candidate_analyzer,
					unit_id=str(pre_uid),
					stage_config=stage_config,
				)
				if template is not None and locations_xy is not None:
					template_source = str(candidate_label)
					break
				candidate_errors.append(f"{candidate_label}:{template_error}")
			asset_entry: dict[str, Any] = {
				"unit_id": str(pre_uid),
				"status": "error",
				"error": (
					" | ".join(candidate_errors)
					if candidate_errors
					else template_error
				),
			}
			if template is not None and locations_xy is not None:
				asset_entry["template_source"] = str(template_source or "snapshot_before")
				pre_render_items.append((asset_entry, template, locations_xy, str(pre_uid)))

			row["pre_assets"].append(asset_entry)

		pre_color_vmin: float | None = None
		pre_color_vmax: float | None = None
		if pre_render_items:
			pre_min_candidates: list[float] = []
			pre_max_candidates: list[float] = []
			for _asset_entry, template_arr, _locs, _uid in pre_render_items:
				try:
					amp = np.asarray(
						_compute_template_channel_magnitude(
							template_ch_by_t=np.asarray(template_arr, dtype=float),
							magnitude_mode=magnitude_mode,
						),
						dtype=float,
					)
					if amp.size <= 0:
						continue
					if color_scale_mode == "log":
						amp = amp[np.isfinite(amp) & (amp > 0.0)]
						if amp.size <= 0:
							continue
					amp_min = float(np.nanmin(amp))
					amp_max = float(np.nanmax(amp))
					if math.isfinite(amp_min) and math.isfinite(amp_max):
						pre_min_candidates.append(amp_min)
						pre_max_candidates.append(amp_max)
				except Exception:
					continue
			if pre_min_candidates and pre_max_candidates:
				vmin_candidate = float(min(pre_min_candidates))
				vmax_candidate = float(max(pre_max_candidates))
				if math.isfinite(vmin_candidate) and math.isfinite(vmax_candidate) and (vmax_candidate > vmin_candidate):
					pre_color_vmin = vmin_candidate
					pre_color_vmax = vmax_candidate

		row["pre_color_scale"] = {
			"mode": "dynamic_per_merge_group",
			"scale": str(color_scale_mode),
			"magnitude_mode": str(magnitude_mode),
			"log_epsilon": float(log_epsilon),
			"vmin": pre_color_vmin,
			"vmax": pre_color_vmax,
		}

		for asset_entry, template_arr, locations_xy, pre_uid in pre_render_items:
			panel_render_items.append(
				{
					"title": f"Pre unit {pre_uid}",
					"template": template_arr,
					"locations_xy": locations_xy,
				}
			)
			asset_rel_base = f"{panel_base}__pre_{_safe_file_token(pre_uid)}"
			asset_png_path = (assets_root_dir / f"{asset_rel_base}.png").resolve()
			asset_svg_path = (assets_root_dir / f"{asset_rel_base}.svg").resolve()

			if write_assets_png:
				ok, err = _write_template_amplitude_heatmap_asset(
					template_ch_by_t=template_arr,
					locations_xy=locations_xy,
					out_path=asset_png_path,
					title=f"Pre unit {pre_uid}",
					cmap=cmap,
					marker_size=marker_size,
					show_colorbar=show_colorbar,
					relative_color_bar_height=relative_color_bar_height,
					color_vmin=pre_color_vmin,
					color_vmax=pre_color_vmax,
					color_scale_mode=color_scale_mode,
					log_epsilon=log_epsilon,
					magnitude_mode=magnitude_mode,
					x_limits=probe_x_limits,
					y_limits=probe_y_limits,
				)
				if ok:
					asset_entry["png"] = str(asset_png_path)
					pre_png_assets.append(asset_png_path)
				else:
					asset_entry["error"] = err

			if write_assets_svg:
				ok, err = _write_template_amplitude_heatmap_asset(
					template_ch_by_t=template_arr,
					locations_xy=locations_xy,
					out_path=asset_svg_path,
					title=f"Pre unit {pre_uid}",
					cmap=cmap,
					marker_size=marker_size,
					show_colorbar=show_colorbar,
					relative_color_bar_height=relative_color_bar_height,
					color_vmin=pre_color_vmin,
					color_vmax=pre_color_vmax,
					color_scale_mode=color_scale_mode,
					log_epsilon=log_epsilon,
					magnitude_mode=magnitude_mode,
					x_limits=probe_x_limits,
					y_limits=probe_y_limits,
				)
				if ok:
					asset_entry["svg"] = str(asset_svg_path)
				else:
					asset_entry["error_svg"] = err

			if "png" in asset_entry or "svg" in asset_entry:
				asset_entry["status"] = "ok"

		post_png_asset: Path | None = None
		post_template_arr: Any | None = None
		post_locations_xy: Any | None = None
		if post_unit_id is not None:
			template, locations_xy, template_error = _extract_template_and_locations_for_unit(
				analyzer=after_analyzer,
				unit_id=str(post_unit_id),
				stage_config=stage_config,
			)
			post_asset: dict[str, Any] = {
				"unit_id": str(post_unit_id),
				"status": "error",
				"error": template_error,
			}
			if template is not None and locations_xy is not None:
				post_template_arr = template
				post_locations_xy = locations_xy
				asset_rel_base = f"{panel_base}__post_{_safe_file_token(post_unit_id)}"
				asset_png_path = (assets_root_dir / f"{asset_rel_base}.png").resolve()
				asset_svg_path = (assets_root_dir / f"{asset_rel_base}.svg").resolve()

				if write_assets_png:
					ok, err = _write_template_amplitude_heatmap_asset(
						template_ch_by_t=template,
						locations_xy=locations_xy,
						out_path=asset_png_path,
						title=f"Post unit {post_unit_id}",
						cmap=cmap,
						marker_size=marker_size,
						show_colorbar=show_colorbar,
						relative_color_bar_height=relative_color_bar_height,
						color_vmin=None,
						color_vmax=None,
						color_scale_mode=color_scale_mode,
						log_epsilon=log_epsilon,
						magnitude_mode=magnitude_mode,
						x_limits=probe_x_limits,
						y_limits=probe_y_limits,
					)
					if ok:
						post_asset["png"] = str(asset_png_path)
						post_png_asset = asset_png_path
					else:
						post_asset["error"] = err

				if write_assets_svg:
					ok, err = _write_template_amplitude_heatmap_asset(
						template_ch_by_t=template,
						locations_xy=locations_xy,
						out_path=asset_svg_path,
						title=f"Post unit {post_unit_id}",
						cmap=cmap,
						marker_size=marker_size,
						show_colorbar=show_colorbar,
						relative_color_bar_height=relative_color_bar_height,
						color_vmin=None,
						color_vmax=None,
						color_scale_mode=color_scale_mode,
						log_epsilon=log_epsilon,
						magnitude_mode=magnitude_mode,
						x_limits=probe_x_limits,
						y_limits=probe_y_limits,
					)
					if ok:
						post_asset["svg"] = str(asset_svg_path)
					else:
						post_asset["error_svg"] = err

				if "png" in post_asset or "svg" in post_asset:
					post_asset["status"] = "ok"
			row["post_asset"] = post_asset
			if post_template_arr is not None and post_locations_xy is not None:
				panel_render_items.append(
					{
						"title": f"Post unit {post_unit_id}",
						"template": post_template_arr,
						"locations_xy": post_locations_xy,
					}
				)

		if write_png or write_svg:
			panel_templates = [item["template"] for item in panel_render_items if isinstance(item, dict) and item.get("template", None) is not None]
			panel_color_vmin, panel_color_vmax = _compute_template_heatmap_scale_limits(
				templates=list(panel_templates),
				color_scale_mode=color_scale_mode,
				log_epsilon=log_epsilon,
				magnitude_mode=magnitude_mode,
			)
			row["panel_color_scale"] = {
				"mode": "dynamic_per_merge_strip",
				"scale": str(color_scale_mode),
				"magnitude_mode": str(magnitude_mode),
				"log_epsilon": float(log_epsilon),
				"vmin": panel_color_vmin,
				"vmax": panel_color_vmax,
			}

			panel_slot_count = max(1, int(len(panel_render_items)))
			panel_width_effective_in = float(panel_width_in) * (float(panel_slot_count) / 2.0)
			panel_dpi = 360

			fig, axes = plt.subplots(
				1,
				panel_slot_count,
				figsize=(float(panel_width_effective_in), float(panel_height_in)),
				constrained_layout=True,
			)
			layout_engine = (
				fig.get_layout_engine()
				if callable(getattr(fig, "get_layout_engine", None))
				else None
			)
			if layout_engine is not None and callable(getattr(layout_engine, "set", None)):
				try:
					layout_engine.set(w_pad=0.02, h_pad=0.02, wspace=0.02, hspace=0.02)
				except Exception:
					pass
			elif hasattr(fig, "set_constrained_layout_pads"):
				try:
					fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.02, hspace=0.02)
				except Exception:
					pass
			axes_list = (
				list(axes.flat)
				if hasattr(axes, "flat")
				else ([axes] if not isinstance(axes, list) else axes)
			)
			shared_scatter: Any | None = None
			for idx, ax in enumerate(axes_list):
				if idx >= len(panel_render_items):
					ax.axis("off")
					continue
				item = panel_render_items[idx]
				sc, err = _plot_template_amplitude_heatmap_on_axis(
					ax=ax,
					template_ch_by_t=item["template"],
					locations_xy=item["locations_xy"],
					title=str(item["title"]),
					cmap=cmap,
					marker_size=marker_size,
					color_vmin=panel_color_vmin,
					color_vmax=panel_color_vmax,
					color_scale_mode=color_scale_mode,
					log_epsilon=log_epsilon,
					magnitude_mode=magnitude_mode,
					x_limits=probe_x_limits,
					y_limits=probe_y_limits,
					show_axis_labels=False,
					title_fontsize=10.5,
					tick_labelsize=8.5,
				)
				if sc is not None:
					shared_scatter = sc
				else:
					ax.text(0.5, 0.5, str(err or "render_failed"), ha="center", va="center", transform=ax.transAxes)
					ax.axis("off")
				if idx > 0 and callable(getattr(ax, "tick_params", None)):
					try:
						ax.tick_params(labelleft=False)
					except Exception:
						pass

			if callable(getattr(fig, "supxlabel", None)):
				try:
					fig.supxlabel("x_um", fontsize=10)
				except Exception:
					pass
			if callable(getattr(fig, "supylabel", None)):
				try:
					fig.supylabel("y_um", fontsize=10)
				except Exception:
					pass
			if bool(show_colorbar) and shared_scatter is not None:
				fig.colorbar(
					shared_scatter,
					ax=axes_list,
					shrink=_normalize_template_heatmap_relative_color_bar_height(
						relative_color_bar_height
					),
					pad=0.01,
				)

			panel_rel_base = str((Path(report_relpath) / panel_base).as_posix())
			for fmt, enabled in (("png", write_png), ("svg", write_svg)):
				if not enabled:
					continue
				panel_out = _resolve_report_image_path(
					out_dir=merge_out_dir,
					relpath=f"{panel_rel_base}.{fmt}",
					format_name=fmt,
				)
				panel_out.parent.mkdir(parents=True, exist_ok=True)
				fig.savefig(panel_out, dpi=panel_dpi)
				row["panel_outputs"][fmt] = str(panel_out)
				outputs[f"merge.report.template_heatmap_per_merge_{panel_base}_{fmt}"] = str(panel_out)
			plt.close(fig)

		merge_rows.append(row)

	debug_json_path = (merge_out_dir / debug_json_relpath).resolve()
	debug_payload = {
		"status": "ok",
		"n_mappings_requested": int(len(applied_unit_mappings)),
		"n_mappings_processed": int(len(merge_rows)),
		"asset_dpi": 320,
		"panel_dpi": 360,
		"panel_layout": "single_row",
		"panel_shared_color_scale": bool(True),
		"pre_template_source_mode": "snapshot_before_only",
		"report_relpath": str(report_relpath),
		"assets_reldir": str(assets_reldir),
		"color_scale": str(color_scale_mode),
		"magnitude_mode": str(magnitude_mode),
		"log_epsilon": float(log_epsilon),
		"marker_size_requested": float(requested_marker_size),
		"marker_size_effective": float(marker_size),
		"marker_size_source": str(marker_size_source),
		"relative_color_bar_height": float(relative_color_bar_height),
		"inherit_probe_dimensions": bool(inherit_probe_dimensions),
		"probe_dim_x_um": (probe_x_limits[1] if probe_x_limits is not None else None),
		"probe_dim_y_um": (probe_y_limits[1] if probe_y_limits is not None else None),
		"probe_pitch_um": getattr(stage_config, "merge_reports_template_heatmaps_probe_pitch_um", None),
		"probe_electrode_size_um_x": getattr(
			stage_config,
			"merge_reports_template_heatmaps_probe_electrode_size_um_x",
			None,
		),
		"probe_electrode_size_um_y": getattr(
			stage_config,
			"merge_reports_template_heatmaps_probe_electrode_size_um_y",
			None,
		),
		"missing_pre_unit_ids": list(missing_pre_unit_ids),
		"rows": merge_rows,
	}
	_write_json(debug_json_path, debug_payload)
	outputs["merge.report.template_heatmaps_per_merge_debug_json"] = str(debug_json_path)
	if loaded_before_analyzer:
		_release_loaded_analyzer_extensions(analyzer=before_analyzer)
	if loaded_after_analyzer:
		_release_loaded_analyzer_extensions(analyzer=after_analyzer)

	return {
		"status": "ok",
		"n_mappings_requested": int(len(applied_unit_mappings)),
		"n_mappings_processed": int(len(merge_rows)),
		"debug_json": str(debug_json_path),
		"outputs": outputs,
	}


def _write_merge_unit_location_reports(
	*,
	merge_out_dir: Path,
	before_snapshot: dict[str, Any],
	after_snapshot: dict[str, Any],
	applied_unit_mappings: list[dict[str, Any]] | None = None,
	stage_config: Any,
) -> dict[str, Any]:
	try:
		_prepare_matplotlib_for_headless_rendering()
		import matplotlib.pyplot as plt  # type: ignore[import-not-found]
		try:
			from matplotlib.lines import Line2D  # type: ignore[import-not-found]
		except Exception:
			Line2D = None
		try:
			from matplotlib import colors as mcolors  # type: ignore[import-not-found]
		except Exception:
			mcolors = None
	except Exception as exc:
		return {
			"status": "error",
			"error": f"matplotlib_import_failed:{type(exc).__name__}:{exc}",
			"outputs": {},
		}

	_log_phase_step_start(
		"Merge unit location report step start",
		merge_out_dir=merge_out_dir,
		applied_mappings=(
			int(len(applied_unit_mappings))
			if isinstance(applied_unit_mappings, list)
			else 0
		),
	)

	before_points = _extract_unit_locations_for_plot(before_snapshot)
	after_points = _extract_unit_locations_for_plot(after_snapshot)
	point_size = float(getattr(stage_config, "merge_reports_2panel_point_size", 9.0) or 9.0)
	if not math.isfinite(point_size) or point_size <= 0.0:
		point_size = 9.0

	before_default_color = str(getattr(stage_config, "merge_reports_2panel_before_point_color", "#7a7a7a") or "#7a7a7a")
	after_default_color = str(getattr(stage_config, "merge_reports_2panel_after_point_color", "#7a7a7a") or "#7a7a7a")
	label_pre_and_post_units = bool(
		getattr(stage_config, "merge_reports_2panel_label_pre_and_post_units", False)
	)
	zoom_to_affected_units = bool(getattr(stage_config, "merge_reports_2panel_zoom_to_affected_units", False))
	highlight_enabled = bool(getattr(stage_config, "merge_reports_2panel_highlight_merges_enabled", False))
	highlight_linked = bool(getattr(stage_config, "merge_reports_2panel_highlight_merges_linked", True))
	plot_highlight_after_other_units = bool(
		getattr(stage_config, "merge_reports_2panel_highlight_plot_after_other_units", False)
	)
	label_affected_units = bool(
		getattr(stage_config, "merge_reports_2panel_highlight_label_affected_units", False)
	)
	highlight_show_legend = bool(
		getattr(stage_config, "merge_reports_2panel_highlight_show_legend", False)
	)
	highlight_legend_position = str(
		getattr(stage_config, "merge_reports_2panel_highlight_legend_position", "center left")
		or "center left"
	).strip() or "center left"
	try:
		highlight_legend_x = float(
			getattr(stage_config, "merge_reports_2panel_highlight_legend_x", -0.2)
		)
	except Exception:
		highlight_legend_x = -0.2
	if not math.isfinite(highlight_legend_x):
		highlight_legend_x = -0.2
	try:
		highlight_legend_y = float(
			getattr(stage_config, "merge_reports_2panel_highlight_legend_y", 0.5)
		)
	except Exception:
		highlight_legend_y = 0.5
	if not math.isfinite(highlight_legend_y):
		highlight_legend_y = 0.5
	highlight_sort_pre_legend_by_groups = bool(
		getattr(stage_config, "merge_reports_2panel_highlight_sort_pre_legend_by_groups", False)
	)
	highlight_debug_json_enabled = bool(
		getattr(stage_config, "merge_reports_2panel_highlight_debug_json_enabled", True)
	)
	highlight_debug_json_relpath = str(
		getattr(
			stage_config,
			"merge_reports_2panel_highlight_debug_json_relpath",
			"unit_locations_highlight_linkage.json",
		)
		or "unit_locations_highlight_linkage.json"
	).strip().lstrip("/") or "unit_locations_highlight_linkage.json"
	highlight_before_color = str(getattr(stage_config, "merge_reports_2panel_highlight_before_color", "#ff7f0e") or "#ff7f0e")
	highlight_after_color = str(getattr(stage_config, "merge_reports_2panel_highlight_after_color", "#2ca02c") or "#2ca02c")
	highlight_palette = str(getattr(stage_config, "merge_reports_2panel_highlight_palette", "tab20") or "tab20")

	def _to_rgba_tuple(raw_color: Any) -> tuple[float, float, float, float] | None:
		if raw_color is None:
			return None
		if callable(getattr(mcolors, "to_rgba", None)):
			try:
				rgba = mcolors.to_rgba(raw_color)
				return (float(rgba[0]), float(rgba[1]), float(rgba[2]), float(rgba[3]))
			except Exception:
				return None
		if isinstance(raw_color, (list, tuple)) and len(raw_color) >= 3:
			try:
				r = float(raw_color[0])
				g = float(raw_color[1])
				b = float(raw_color[2])
				a = float(raw_color[3]) if len(raw_color) > 3 else 1.0
			except Exception:
				return None
			if not (math.isfinite(r) and math.isfinite(g) and math.isfinite(b) and math.isfinite(a)):
				return None
			return (r, g, b, a)
		return None

	def _is_grayish(raw_color: Any) -> bool:
		rgba = _to_rgba_tuple(raw_color)
		if rgba is None:
			return False
		r, g, b, _ = rgba
		return bool(max(abs(r - g), abs(g - b), abs(r - b)) < 0.06)

	def _sanitize_highlight_color(raw_color: Any, fallback_color: Any) -> Any:
		if _is_grayish(raw_color):
			return fallback_color
		return raw_color

	def _color_for_debug(raw_color: Any) -> str:
		if callable(getattr(mcolors, "to_hex", None)):
			try:
				return str(mcolors.to_hex(raw_color, keep_alpha=True))
			except Exception:
				pass
		rgba = _to_rgba_tuple(raw_color)
		if rgba is not None:
			return f"rgba({rgba[0]:.6f},{rgba[1]:.6f},{rgba[2]:.6f},{rgba[3]:.6f})"
		return str(raw_color)

	highlight_before_color = _sanitize_highlight_color(highlight_before_color, "#ff7f0e")
	highlight_after_color = _sanitize_highlight_color(highlight_after_color, "#2ca02c")

	before_unit_ids = set(str(uid) for uid in before_points.keys())
	after_added_points = {
		str(uid): xy
		for uid, xy in after_points.items()
		if str(uid) not in before_unit_ids
	}
	eligible_after_points = (after_added_points if after_added_points else dict(after_points))

	before_highlight_map: dict[str, Any] = {}
	after_highlight_map: dict[str, Any] = {}
	highlight_linkage_rows: list[dict[str, Any]] = []
	after_highlighted_inferred_units = 0
	if highlight_enabled and isinstance(applied_unit_mappings, list) and applied_unit_mappings:
		used_after_unit_ids: set[str] = set()
		linked_fallback_palette = [
			"#e41a1c",
			"#377eb8",
			"#4daf4a",
			"#ff7f00",
			"#984ea3",
			"#a65628",
			"#f781bf",
			"#d95f02",
			"#1b9e77",
		]

		def _mapping_pre_locations(mapping: dict[str, Any], pre_unit_ids: list[str]) -> dict[str, Any]:
			locations: dict[str, Any] = {}
			pre_locations_raw = mapping.get("pre_unit_locations", None)
			if isinstance(pre_locations_raw, dict):
				for uid in pre_unit_ids:
					if uid in pre_locations_raw:
						locations[uid] = pre_locations_raw.get(uid)
			for uid in pre_unit_ids:
				if uid in locations:
					continue
				xy = before_points.get(uid, None)
				if xy is not None:
					locations[uid] = {"x_um": float(xy[0]), "y_um": float(xy[1])}
			return locations

		if highlight_linked:
			try:
				cmap = plt.get_cmap(highlight_palette)
			except Exception:
				cmap = plt.get_cmap("tab20")

			n_mappings = max(1, int(len(applied_unit_mappings)))
			linked_rows: list[tuple[dict[str, Any], list[str], Any, dict[str, Any]]] = []
			for idx, mapping_raw in enumerate(applied_unit_mappings):
				mapping = (mapping_raw if isinstance(mapping_raw, dict) else {})
				pre_ids = _normalize_unit_id_list(mapping.get("pre_unit_ids", []))
				if not pre_ids:
					continue
				raw_color = (cmap((float(idx) / float(max(1, n_mappings - 1)))) if n_mappings > 1 else cmap(0.0))
				color = _sanitize_highlight_color(
					raw_color,
					linked_fallback_palette[int(idx) % int(len(linked_fallback_palette))],
				)

				post_uid_raw = mapping.get("post_unit_id", None)
				post_uid: str | None = None
				if post_uid_raw is not None and str(post_uid_raw).strip():
					post_uid = _normalize_cluster_id(post_uid_raw)
				if post_uid is not None and post_uid in after_highlight_map:
					color = after_highlight_map[post_uid]
				for uid in pre_ids:
					before_highlight_map[uid] = color

				linkage_row: dict[str, Any] = {
					"mapping_index": int(idx),
					"group_id": str(mapping.get("group_id", f"group_{int(idx + 1):03d}")),
					"method": str(mapping.get("method", "unknown")),
					"pre_unit_ids": list(pre_ids),
					"requested_post_unit_id": post_uid,
					"resolved_post_unit_id": None,
					"resolution": "unresolved",
					"linked_highlight": True,
					"color": _color_for_debug(color),
				}

				if post_uid is not None and post_uid in after_points:
					after_highlight_map[post_uid] = color
					used_after_unit_ids.add(post_uid)
					linkage_row["resolved_post_unit_id"] = str(post_uid)
					linkage_row["resolution"] = "requested_post_unit_id"
					highlight_linkage_rows.append(linkage_row)
				else:
					linked_rows.append((mapping, pre_ids, color, linkage_row))

			for mapping, pre_ids, color, linkage_row in linked_rows:
				pre_locations = _mapping_pre_locations(mapping, pre_ids)
				inferred_post_uid, _inferred_distance = _infer_post_unit_id_from_locations(
					pre_unit_ids=pre_ids,
					pre_locations=pre_locations,
					post_locations=eligible_after_points,
					used_post_unit_ids=used_after_unit_ids,
					max_distance_um=40.0,
				)
				if inferred_post_uid is not None:
					after_highlight_map[inferred_post_uid] = color
					used_after_unit_ids.add(inferred_post_uid)
					after_highlighted_inferred_units += 1
					linkage_row["resolved_post_unit_id"] = str(inferred_post_uid)
					linkage_row["resolution"] = "inferred_post_location"
				else:
					linkage_row["resolution"] = "no_post_unit_match"
				highlight_linkage_rows.append(linkage_row)
		else:
			for idx, mapping_raw in enumerate(applied_unit_mappings):
				mapping = (mapping_raw if isinstance(mapping_raw, dict) else {})
				pre_ids = _normalize_unit_id_list(mapping.get("pre_unit_ids", []))
				if not pre_ids:
					continue
				for uid in pre_ids:
					before_highlight_map[uid] = highlight_before_color

				post_uid_raw = mapping.get("post_unit_id", None)
				post_uid: str | None = None
				if post_uid_raw is not None and str(post_uid_raw).strip():
					post_uid = _normalize_cluster_id(post_uid_raw)

				linkage_row = {
					"mapping_index": int(idx),
					"group_id": str(mapping.get("group_id", f"group_{int(idx + 1):03d}")),
					"method": str(mapping.get("method", "unknown")),
					"pre_unit_ids": list(pre_ids),
					"requested_post_unit_id": post_uid,
					"resolved_post_unit_id": None,
					"resolution": "unresolved",
					"linked_highlight": False,
					"color": _color_for_debug(highlight_after_color),
				}

				if post_uid is not None and post_uid in after_points:
					after_highlight_map[post_uid] = highlight_after_color
					used_after_unit_ids.add(post_uid)
					linkage_row["resolved_post_unit_id"] = str(post_uid)
					linkage_row["resolution"] = "requested_post_unit_id"
					highlight_linkage_rows.append(linkage_row)
					continue

				pre_locations = _mapping_pre_locations(mapping, pre_ids)
				inferred_post_uid, _inferred_distance = _infer_post_unit_id_from_locations(
					pre_unit_ids=pre_ids,
					pre_locations=pre_locations,
					post_locations=eligible_after_points,
					used_post_unit_ids=used_after_unit_ids,
					max_distance_um=40.0,
				)
				if inferred_post_uid is not None:
					after_highlight_map[inferred_post_uid] = highlight_after_color
					used_after_unit_ids.add(inferred_post_uid)
					after_highlighted_inferred_units += 1
					linkage_row["resolved_post_unit_id"] = str(inferred_post_uid)
					linkage_row["resolution"] = "inferred_post_location"
				else:
					linkage_row["resolution"] = "no_post_unit_match"
				highlight_linkage_rows.append(linkage_row)

	pre_legend_ordered_highlight_uids: list[str] | None = None
	if (
		highlight_enabled
		and bool(highlight_sort_pre_legend_by_groups)
		and bool(before_highlight_map)
		and isinstance(applied_unit_mappings, list)
		and applied_unit_mappings
	):
		ordered_uids: list[str] = []
		seen: set[str] = set()
		for mapping_raw in applied_unit_mappings:
			mapping = (mapping_raw if isinstance(mapping_raw, dict) else {})
			for uid in _normalize_unit_id_list(mapping.get("pre_unit_ids", [])):
				if uid in before_highlight_map and uid not in seen:
					ordered_uids.append(uid)
					seen.add(uid)
		remainder = sorted(
			[uid for uid in before_highlight_map.keys() if uid not in seen],
			key=_unit_sort_key,
		)
		pre_legend_ordered_highlight_uids = ordered_uids + remainder

	probe_dim_x_um = getattr(stage_config, "merge_reports_2panel_probe_dim_x_um", None)
	probe_dim_y_um = getattr(stage_config, "merge_reports_2panel_probe_dim_y_um", None)
	try:
		probe_dim_x = (float(probe_dim_x_um) if probe_dim_x_um is not None else None)
	except Exception:
		probe_dim_x = None
	try:
		probe_dim_y = (float(probe_dim_y_um) if probe_dim_y_um is not None else None)
	except Exception:
		probe_dim_y = None
	if probe_dim_x is not None and (not math.isfinite(probe_dim_x) or probe_dim_x <= 0.0):
		probe_dim_x = None
	if probe_dim_y is not None and (not math.isfinite(probe_dim_y) or probe_dim_y <= 0.0):
		probe_dim_y = None

	affected_points = [
		before_points[uid]
		for uid in before_highlight_map.keys()
		if uid in before_points
	] + [
		after_points[uid]
		for uid in after_highlight_map.keys()
		if uid in after_points
	]
	zoom_to_affected_applied = bool(zoom_to_affected_units and affected_points)

	if zoom_to_affected_applied:
		affected_x = [xy[0] for xy in affected_points]
		affected_y = [xy[1] for xy in affected_points]
		x_min = min(affected_x)
		x_max = max(affected_x)
		y_min = min(affected_y)
		y_max = max(affected_y)
		if x_max <= x_min:
			x_max = x_min + 1.0
		if y_max <= y_min:
			y_max = y_min + 1.0
		margin_x = max(1.0, 0.05 * float(x_max - x_min))
		margin_y = max(1.0, 0.05 * float(y_max - y_min))
		x_limits = (x_min - margin_x, x_max + margin_x)
		y_limits = (y_min - margin_y, y_max + margin_y)
	elif probe_dim_x is not None and probe_dim_y is not None:
		x_limits = (0.0, float(probe_dim_x))
		y_limits = (0.0, float(probe_dim_y))
	else:
		all_x = [xy[0] for xy in list(before_points.values()) + list(after_points.values())]
		all_y = [xy[1] for xy in list(before_points.values()) + list(after_points.values())]
		x_min = (min(all_x) if all_x else 0.0)
		x_max = (max(all_x) if all_x else 1.0)
		y_min = (min(all_y) if all_y else 0.0)
		y_max = (max(all_y) if all_y else 1.0)
		if x_max <= x_min:
			x_max = x_min + 1.0
		if y_max <= y_min:
			y_max = y_min + 1.0

		margin_x = max(1.0, 0.05 * float(x_max - x_min))
		margin_y = max(1.0, 0.05 * float(y_max - y_min))
		x_limits = (x_min - margin_x, x_max + margin_x)
		y_limits = (y_min - margin_y, y_max + margin_y)

	def _plot_points(
		ax: Any,
		points: dict[str, tuple[float, float]],
		title: str,
		default_color: Any,
		highlight_map: dict[str, Any],
		label_units: bool,
		label_highlight_only: bool,
		plot_highlight_after_other: bool,
		show_highlight_legend: bool,
		legend_loc: str,
		legend_anchor: tuple[float, float] | None,
		legend_ordered_highlight_uids: list[str] | None = None,
	) -> None:
		ax.set_title(str(title))
		ax.set_xlabel("x_um")
		ax.set_ylabel("y_um")
		ax.set_xlim(x_limits)
		ax.set_ylim(y_limits)
		ax.invert_yaxis()
		ax.set_aspect("equal", adjustable="box")
		ax.grid(True, alpha=0.25)
		if points:
			ordered_uids = sorted(list(points.keys()), key=_unit_sort_key)
			if plot_highlight_after_other and highlight_map:
				normal_uids = [uid for uid in ordered_uids if uid not in highlight_map]
				highlight_uids = [uid for uid in ordered_uids if uid in highlight_map]
				if normal_uids:
					ax.scatter(
						[points[uid][0] for uid in normal_uids],
						[points[uid][1] for uid in normal_uids],
						s=float(point_size),
						alpha=0.85,
						c=[default_color for _ in normal_uids],
					)
				if highlight_uids:
					ax.scatter(
						[points[uid][0] for uid in highlight_uids],
						[points[uid][1] for uid in highlight_uids],
						s=float(point_size),
						alpha=0.85,
						c=[highlight_map.get(uid, default_color) for uid in highlight_uids],
					)
			else:
				xs = [points[uid][0] for uid in ordered_uids]
				ys = [points[uid][1] for uid in ordered_uids]
				colors = [highlight_map.get(uid, default_color) for uid in ordered_uids]
				ax.scatter(xs, ys, s=float(point_size), alpha=0.85, c=colors)

			if label_units:
				labeled_uids = ordered_uids
			elif label_highlight_only:
				labeled_uids = [uid for uid in ordered_uids if uid in highlight_map]
			else:
				labeled_uids = []
			for uid in labeled_uids:
				x, y = points[uid]
				ax.text(x, y, str(uid), ha="left", va="bottom")

			if show_highlight_legend and highlight_map:
				if isinstance(legend_ordered_highlight_uids, list) and legend_ordered_highlight_uids:
					highlight_uids_for_legend = [
						uid
						for uid in legend_ordered_highlight_uids
						if uid in highlight_map and uid in points
					]
				else:
					highlight_uids_for_legend = sorted(
						[uid for uid in ordered_uids if uid in highlight_map],
						key=_unit_sort_key,
					)
				if highlight_uids_for_legend and callable(getattr(ax, "legend", None)):
					if Line2D is not None:
						handles: list[Any] = []
						for uid in highlight_uids_for_legend:
							color = highlight_map.get(uid, default_color)
							handles.append(
								Line2D(
									[],
									[],
									marker="o",
									linestyle="None",
									markersize=5,
									markerfacecolor=color,
									markeredgecolor=color,
									label=str(uid),
								)
							)
						try:
								legend_kwargs: dict[str, Any] = {
									"handles": handles,
									"title": "Highlighted units",
									"loc": str(legend_loc),
								}
								if legend_anchor is not None:
									legend_kwargs["bbox_to_anchor"] = legend_anchor
								ax.legend(**legend_kwargs)
						except Exception:
							pass
					else:
						try:
								legend_kwargs = {
									"title": "Highlighted units",
									"loc": str(legend_loc),
								}
								if legend_anchor is not None:
									legend_kwargs["bbox_to_anchor"] = legend_anchor
								ax.legend(highlight_uids_for_legend, **legend_kwargs)
						except Exception:
							pass
		else:
			ax.text(0.5, 0.5, "No unit locations", ha="center", va="center", transform=ax.transAxes)

	highlight_legend_anchor: tuple[float, float] | None = (float(highlight_legend_x), float(highlight_legend_y))

	outputs: dict[str, str] = {}
	if highlight_enabled and highlight_debug_json_enabled:
		highlight_debug_json_path = (merge_out_dir / str(highlight_debug_json_relpath)).resolve()
		_write_json(
			highlight_debug_json_path,
			{
				"status": "ok",
				"highlight_enabled": bool(highlight_enabled),
				"highlight_linked": bool(highlight_linked),
				"n_mappings": int(len(applied_unit_mappings or [])),
				"before_highlighted_unit_ids": sorted(list(before_highlight_map.keys()), key=_unit_sort_key),
				"after_highlighted_unit_ids": sorted(list(after_highlight_map.keys()), key=_unit_sort_key),
				"after_highlighted_inferred_units_count": int(after_highlighted_inferred_units),
				"linkage_rows": list(highlight_linkage_rows),
			},
		)
		outputs["merge.report.unit_locations_highlight_linkage_json"] = str(highlight_debug_json_path)

	before_write_png = bool(getattr(stage_config, "merge_reports_2panel_before_write_png", True))
	before_write_svg = bool(getattr(stage_config, "merge_reports_2panel_before_write_svg", False))
	before_relpath = str(getattr(stage_config, "merge_reports_2panel_before_relpath", "unit_locations_before_merge.png"))
	if before_write_png or before_write_svg:
		fig, ax = plt.subplots(1, 1, figsize=(6.5, 5.5))
		_plot_points(
			ax,
			before_points,
			f"Before Merge (n={len(before_points)})",
			before_default_color,
			before_highlight_map,
			label_pre_and_post_units,
			label_affected_units,
			plot_highlight_after_other_units,
			highlight_show_legend,
			highlight_legend_position,
			highlight_legend_anchor,
			pre_legend_ordered_highlight_uids,
		)
		for fmt, enabled in (("png", before_write_png), ("svg", before_write_svg)):
			if not enabled:
				continue
			out_path = _resolve_report_image_path(out_dir=merge_out_dir, relpath=before_relpath, format_name=fmt)
			out_path.parent.mkdir(parents=True, exist_ok=True)
			fig.savefig(out_path, dpi=240)
			outputs[f"merge.report.unit_locations_before_{fmt}"] = str(out_path)
		plt.close(fig)

	after_write_png = bool(getattr(stage_config, "merge_reports_2panel_after_write_png", True))
	after_write_svg = bool(getattr(stage_config, "merge_reports_2panel_after_write_svg", False))
	after_relpath = str(getattr(stage_config, "merge_reports_2panel_after_relpath", "unit_locations_after_merge.png"))
	if after_write_png or after_write_svg:
		fig, ax = plt.subplots(1, 1, figsize=(6.5, 5.5))
		_plot_points(
			ax,
			after_points,
			f"After Merge (n={len(after_points)})",
			after_default_color,
			after_highlight_map,
			label_pre_and_post_units,
			label_affected_units,
			plot_highlight_after_other_units,
			highlight_show_legend,
			highlight_legend_position,
			highlight_legend_anchor,
			None,
		)
		for fmt, enabled in (("png", after_write_png), ("svg", after_write_svg)):
			if not enabled:
				continue
			out_path = _resolve_report_image_path(out_dir=merge_out_dir, relpath=after_relpath, format_name=fmt)
			out_path.parent.mkdir(parents=True, exist_ok=True)
			fig.savefig(out_path, dpi=240)
			outputs[f"merge.report.unit_locations_after_{fmt}"] = str(out_path)
		plt.close(fig)

	panel_write_png = bool(getattr(stage_config, "merge_reports_2panel_write_png", True))
	panel_write_svg = bool(getattr(stage_config, "merge_reports_2panel_write_svg", False))
	panel_relpath = str(getattr(stage_config, "merge_reports_2panel_relpath", "unit_locations_before_after_merge.png"))
	if panel_write_png or panel_write_svg:
		fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.5), constrained_layout=True)
		_plot_points(
			axes[0],
			before_points,
			f"Before Merge (n={len(before_points)})",
			before_default_color,
			before_highlight_map,
			label_pre_and_post_units,
			label_affected_units,
			plot_highlight_after_other_units,
			highlight_show_legend,
			highlight_legend_position,
			highlight_legend_anchor,
			pre_legend_ordered_highlight_uids,
		)
		_plot_points(
			axes[1],
			after_points,
			f"After Merge (n={len(after_points)})",
			after_default_color,
			after_highlight_map,
			label_pre_and_post_units,
			label_affected_units,
			plot_highlight_after_other_units,
			highlight_show_legend,
			highlight_legend_position,
			highlight_legend_anchor,
			None,
		)
		for fmt, enabled in (("png", panel_write_png), ("svg", panel_write_svg)):
			if not enabled:
				continue
			out_path = _resolve_report_image_path(out_dir=merge_out_dir, relpath=panel_relpath, format_name=fmt)
			out_path.parent.mkdir(parents=True, exist_ok=True)
			fig.savefig(out_path, dpi=260)
			outputs[f"merge.report.unit_locations_before_after_{fmt}"] = str(out_path)
		plt.close(fig)

	return {
		"status": "ok",
		"before_unit_locations_count": int(len(before_points)),
		"after_unit_locations_count": int(len(after_points)),
		"eligible_after_unit_locations_count": int(len(eligible_after_points)),
		"before_highlighted_units_count": int(len(before_highlight_map)),
		"after_highlighted_units_count": int(len(after_highlight_map)),
		"after_highlighted_inferred_units_count": int(after_highlighted_inferred_units),
		"highlight_linkage_row_count": int(len(highlight_linkage_rows)),
		"highlight_legend_position": str(highlight_legend_position),
		"highlight_legend_anchor": [float(highlight_legend_x), float(highlight_legend_y)],
		"highlight_pre_legend_sorted_by_groups": bool(highlight_sort_pre_legend_by_groups),
		"zoom_to_affected_units": bool(zoom_to_affected_units),
		"zoom_to_affected_units_applied": bool(zoom_to_affected_applied),
		"outputs": outputs,
	}


def _log_merge_summary_details(
	*,
	stream_id: str,
	status: str,
	method_reports: list[dict[str, Any]],
	summary_json: Path,
	merge_metadata_enabled: bool,
	merge_metadata_json: Path | None,
	merge_metadata_payload: dict[str, Any] | None,
	merge_metadata_error: str | None,
) -> None:
	method_tokens: list[str] = []
	for report in method_reports:
		if not isinstance(report, dict):
			continue
		name = str(report.get("name", "unknown"))
		report_status = str(report.get("status", "unknown"))
		method_tokens.append(f"{name}:{report_status}")

	LOGGER.info(
		"Merge summary [stream=%s] status=%s methods=%s summary_json=%s",
		str(stream_id),
		str(status),
		(",".join(method_tokens) if method_tokens else "none"),
		str(summary_json),
	)

	if merge_metadata_error is not None:
		LOGGER.warning(
			"Merge metadata summary [stream=%s] failed: %s",
			str(stream_id),
			str(merge_metadata_error),
		)
		return

	if not bool(merge_metadata_enabled):
		LOGGER.info("Merge metadata summary [stream=%s] disabled", str(stream_id))
		return

	if not isinstance(merge_metadata_payload, dict):
		LOGGER.info("Merge metadata summary [stream=%s] unavailable", str(stream_id))
		return

	change_validation_raw = merge_metadata_payload.get("change_validation", {})
	change_validation = (change_validation_raw if isinstance(change_validation_raw, dict) else {})
	delta_raw = merge_metadata_payload.get("delta", {})
	delta = (delta_raw if isinstance(delta_raw, dict) else {})
	analyzer_delta_raw = delta.get("analyzer", {})
	sorter_delta_raw = delta.get("sorter", {})
	analyzer_delta = (analyzer_delta_raw if isinstance(analyzer_delta_raw, dict) else {})
	sorter_delta = (sorter_delta_raw if isinstance(sorter_delta_raw, dict) else {})

	LOGGER.info(
		"Merge metadata [stream=%s] applied_groups=%d any_changed=%s passes=%s reason=%s analyzer_count=%s->%s sorter_count=%s->%s metadata_json=%s",
		str(stream_id),
		int(merge_metadata_payload.get("applied_merge_group_count", 0) or 0),
		bool(delta.get("any_changed", False)),
		bool(change_validation.get("passes", False)),
		str(change_validation.get("reason", "")),
		str(analyzer_delta.get("before_unit_count", "n/a")),
		str(analyzer_delta.get("after_unit_count", "n/a")),
		str(sorter_delta.get("before_unit_count", "n/a")),
		str(sorter_delta.get("after_unit_count", "n/a")),
		(str(merge_metadata_json) if merge_metadata_json is not None else "n/a"),
	)


def _run_slay_merge_method(
	*,
	well_out_dir: Path,
	stage_output_root_dir: Path,
	output_rel_root: str,
	stage_config: Any,
	force_restart: bool,
	sorter_output_dir: Path | None = None,
) -> dict[str, Any]:
	merge_output_rel_root = _compose_output_rel_root(
		stage_output_rel_root=output_rel_root,
		child_rel_root=getattr(stage_config, "merge_rel_output_root", None),
	)
	merge_out_dir = _resolve_under_spikesort_output_root(
		well_out_dir=well_out_dir,
		output_rel_root=merge_output_rel_root,
		relpath=str(getattr(stage_config, "slay_relpath", "SLAy_outputs")),
	)
	summary_json = merge_out_dir / "slay_method_summary.json"

	slay_delete_outputs_on_force_restart = bool(getattr(stage_config, "slay_delete_outputs_on_force_restart", True))
	removed_on_force_restart: list[str] = []
	if bool(force_restart) and bool(slay_delete_outputs_on_force_restart) and merge_out_dir.exists():
		removed_on_force_restart.append(str(merge_out_dir))
		shutil.rmtree(merge_out_dir, ignore_errors=True)
	merge_out_dir.mkdir(parents=True, exist_ok=True)

	slay_enabled = bool(getattr(stage_config, "slay_enabled", False))
	if not slay_enabled:
		payload = {
			"status": "skipped",
			"reason": "slay_disabled",
			"well_out_dir": str(well_out_dir),
			"stage_output_root_dir": str(stage_output_root_dir),
			"merge_out_dir": str(merge_out_dir),
			"force_restart": bool(force_restart),
			"slay_delete_outputs_on_force_restart": bool(slay_delete_outputs_on_force_restart),
			"removed_on_force_restart": list(removed_on_force_restart),
		}
		_write_json(summary_json, payload)
		return {
			"name": "slay",
			"status": "skipped",
			"reason": "slay_disabled",
			"out_dir": str(merge_out_dir),
			"summary_json": str(summary_json),
			"outputs": {
				"slay.summary_json": str(summary_json),
			},
			"applied_merges": False,
			"removed_on_force_restart": list(removed_on_force_restart),
		}

	ks_dir = (
		Path(sorter_output_dir).resolve()
		if sorter_output_dir is not None
		else _resolve_sorter_output_dir(
			well_out_dir=well_out_dir,
			output_rel_root=output_rel_root,
			stage_config=stage_config,
		)
	)
	requested_ks_dir = ks_dir
	ks_dir = _normalize_slay_kilosort_dir(sorter_output_dir=ks_dir)
	if not (ks_dir / "params.py").exists():
		raise FileNotFoundError(
			"SLAy requires a Kilosort folder containing params.py. "
			f"Resolved path: {ks_dir}. Requested path: {requested_ks_dir}. "
			"Configure stages.spikesort.phases.merge_units.SLAy.sorter_output_relpath if needed."
		)

	run_slay = _import_slay_run_function(
		allow_numpy_fallback=bool(getattr(stage_config, "slay_allow_numpy_fallback", True)),
	)

	run_output_json = merge_out_dir / str(getattr(stage_config, "slay_output_json_relpath", "run-output.json"))
	run_output_json.parent.mkdir(parents=True, exist_ok=True)
	preprocess_recording_dir = _resolve_existing_slay_preferred_recording_dir(
		well_out_dir=well_out_dir,
		stage_config=stage_config,
	)
	run_args: dict[str, Any] = {
		"KS_folder": str(ks_dir),
		"auto_accept_merges": bool(getattr(stage_config, "slay_auto_accept_merges", False)),
		"plot_merges": bool(getattr(stage_config, "slay_plot_merges", False)),
		"output_json": str(run_output_json),
	}
	if preprocess_recording_dir is not None:
		_resolve_slay_recording_dir(
			ks_dir=ks_dir,
			preferred_recording_dir=preprocess_recording_dir,
		)
		run_args["__axon_recon_preprocess_recording_dir"] = str(preprocess_recording_dir)
	extra_params = getattr(stage_config, "slay_params", None)
	if isinstance(extra_params, dict):
		run_args.update(dict(extra_params))

	model_cache_path = _resolve_slay_model_cache_path(
		well_out_dir=well_out_dir,
		output_rel_root=output_rel_root,
		stage_config=stage_config,
	)
	slay_model_cache_use_cached_model = bool(
		getattr(stage_config, "slay_model_cache_use_cached_model", True)
	)
	slay_model_cache_write_model = bool(
		getattr(stage_config, "slay_model_cache_write_model", True)
	)
	model_path_raw = run_args.get("model_path", None)
	model_path_from_params = bool(
		model_path_raw is not None
		and str(model_path_raw).strip()
		and str(model_path_raw).strip().lower() != "none"
	)
	if (not model_path_from_params) and model_cache_path is not None:
		should_set_cache_model_path = bool(
			(slay_model_cache_use_cached_model and model_cache_path.exists())
			or slay_model_cache_write_model
		)
		if should_set_cache_model_path:
			run_args["model_path"] = str(model_cache_path)
			model_path_raw = run_args.get("model_path", None)

	resolved_model_path: Path | None = None
	if model_path_raw is not None and str(model_path_raw).strip() and str(model_path_raw).strip().lower() != "none":
		resolved_model_path = Path(str(model_path_raw)).expanduser().resolve()
		resolved_model_path.parent.mkdir(parents=True, exist_ok=True)
		run_args["model_path"] = str(resolved_model_path)

	using_stage_managed_model_cache_path = bool(
		(not model_path_from_params)
		and model_cache_path is not None
		and resolved_model_path is not None
		and resolved_model_path == model_cache_path.resolve()
	)
	model_deleted_to_disable_cache_use = False
	if (
		using_stage_managed_model_cache_path
		and (not slay_model_cache_use_cached_model)
		and resolved_model_path is not None
		and resolved_model_path.exists()
	):
		if resolved_model_path.is_dir():
			shutil.rmtree(resolved_model_path, ignore_errors=True)
		else:
			resolved_model_path.unlink(missing_ok=True)
		model_deleted_to_disable_cache_use = True

	slay_force_restart_retrain_model = bool(getattr(stage_config, "slay_force_restart_retrain_model", False))
	model_deleted_on_force_restart = False
	if bool(force_restart) and bool(slay_force_restart_retrain_model) and resolved_model_path is not None and resolved_model_path.exists():
		if resolved_model_path.is_dir():
			shutil.rmtree(resolved_model_path, ignore_errors=True)
		else:
			resolved_model_path.unlink(missing_ok=True)
		model_deleted_on_force_restart = True

	run_args["KS_folder"] = str(ks_dir)
	run_args["auto_accept_merges"] = bool(getattr(stage_config, "slay_auto_accept_merges", False))
	run_args["plot_merges"] = bool(getattr(stage_config, "slay_plot_merges", False))
	run_args["output_json"] = str(run_output_json)

	_log_phase_step_start(
		"SLAy execution step start",
		well_out_dir=well_out_dir,
		ks_dir=ks_dir,
		auto_accept_merges=bool(run_args.get("auto_accept_merges", False)),
		plot_merges=bool(run_args.get("plot_merges", False)),
	)

	run_slay(run_args)

	automerge_dir = (ks_dir / "automerge").resolve()
	automerge_snapshot_dir = merge_out_dir / "automerge"
	if bool(getattr(stage_config, "slay_copy_automerge_artifacts", True)) and automerge_dir.exists():
		if automerge_snapshot_dir.exists():
			shutil.rmtree(automerge_snapshot_dir, ignore_errors=True)
		shutil.copytree(automerge_dir, automerge_snapshot_dir)

	automerge_plots_dir = (automerge_dir / "merges").resolve()
	automerge_snapshot_plots_dir = (automerge_snapshot_dir / "merges").resolve()
	plot_files_generated = 0
	plot_files_generated_in_snapshot = 0
	if automerge_plots_dir.exists():
		plot_files_generated = int(sum(1 for p in automerge_plots_dir.iterdir() if p.is_file()))
	if automerge_snapshot_plots_dir.exists():
		plot_files_generated_in_snapshot = int(sum(1 for p in automerge_snapshot_plots_dir.iterdir() if p.is_file()))

	merge_groups_src = automerge_dir / "new2old.json"
	merge_groups_payload: dict[str, list[Any]] = {}
	if merge_groups_src.exists():
		merge_groups_payload = json.loads(merge_groups_src.read_text(encoding="utf-8"))

	merge_groups_out = merge_out_dir / str(getattr(stage_config, "slay_merge_groups_relpath", "recommended_merge_groups.json"))
	_write_json(
		merge_groups_out,
		{
			"n_groups": int(len(merge_groups_payload)),
			"merge_groups": merge_groups_payload,
		},
	)

	metrics_lookup = _load_metrics_lookup(automerge_dir / "metrics.tsv")
	candidate_rows = _build_candidate_pair_rows(
		merge_groups=merge_groups_payload,
		metrics_lookup=metrics_lookup,
	)
	candidates_out = merge_out_dir / str(getattr(stage_config, "slay_candidate_pairs_relpath", "recommended_merge_candidates.tsv"))
	_write_candidate_pairs_tsv(candidates_out, candidate_rows)

	outputs: dict[str, str] = {
		"slay.summary_json": str(summary_json),
		"slay.run_output_json": str(run_output_json),
		"slay.recommended_merge_groups_json": str(merge_groups_out),
		"slay.recommended_merge_candidates_tsv": str(candidates_out),
	}
	if resolved_model_path is not None:
		outputs["slay.model_cache_path"] = str(resolved_model_path)
	if automerge_dir.exists():
		outputs["slay.automerge_dir"] = str(automerge_dir)
	if automerge_plots_dir.exists():
		outputs["slay.automerge_plots_dir"] = str(automerge_plots_dir)
	if automerge_snapshot_dir.exists():
		outputs["slay.automerge_snapshot_dir"] = str(automerge_snapshot_dir)
	if automerge_snapshot_plots_dir.exists():
		outputs["slay.automerge_snapshot_plots_dir"] = str(automerge_snapshot_plots_dir)

	plot_merges_requested = bool(run_args.get("plot_merges", False))
	auto_accept_merges = bool(run_args.get("auto_accept_merges", False))
	plot_generation_note: str | None = None
	if plot_merges_requested and auto_accept_merges and plot_files_generated <= 0:
		plot_generation_note = (
			"SLAy does not emit merge plots when auto_accept_merges=true; "
			"set auto_accept_merges=false to generate automerge/merges plot files."
		)

	payload = {
		"status": "ok",
		"well_out_dir": str(well_out_dir),
		"stage_output_root_dir": str(stage_output_root_dir),
		"merge_out_dir": str(merge_out_dir),
		"ks_dir": str(ks_dir),
		"n_merge_groups": int(len(merge_groups_payload)),
		"n_candidate_pairs": int(len(candidate_rows)),
		"run_args": {
			"KS_folder": str(ks_dir),
			"auto_accept_merges": bool(auto_accept_merges),
			"plot_merges": bool(plot_merges_requested),
			"output_json": str(run_output_json),
			"model_path": (str(resolved_model_path) if resolved_model_path is not None else None),
		},
		"slay_model_cache_use_cached_model": bool(slay_model_cache_use_cached_model),
		"slay_model_cache_write_model": bool(slay_model_cache_write_model),
		"slay_model_deleted_to_disable_cache_use": bool(model_deleted_to_disable_cache_use),
		"slay_force_restart_retrain_model": bool(slay_force_restart_retrain_model),
		"slay_model_deleted_on_force_restart": bool(model_deleted_on_force_restart),
		"plot_files_generated": int(plot_files_generated),
		"plot_files_generated_in_snapshot": int(plot_files_generated_in_snapshot),
		"force_restart": bool(force_restart),
		"slay_delete_outputs_on_force_restart": bool(slay_delete_outputs_on_force_restart),
		"removed_on_force_restart": list(removed_on_force_restart),
		"outputs": outputs,
	}
	if plot_generation_note is not None:
		payload["plot_generation_note"] = str(plot_generation_note)
	_write_json(summary_json, payload)

	return {
		"name": "slay",
		"status": "ok",
		"reason": None,
		"out_dir": str(merge_out_dir),
		"summary_json": str(summary_json),
		"outputs": outputs,
		"n_merge_groups": int(len(merge_groups_payload)),
		"n_candidate_pairs": int(len(candidate_rows)),
		"ks_dir": str(ks_dir),
		"applied_merges": bool(auto_accept_merges),
		"plot_files_generated": int(plot_files_generated),
		"plot_files_generated_in_snapshot": int(plot_files_generated_in_snapshot),
		"slay_model_cache_use_cached_model": bool(slay_model_cache_use_cached_model),
		"slay_model_cache_write_model": bool(slay_model_cache_write_model),
		"slay_model_deleted_to_disable_cache_use": bool(model_deleted_to_disable_cache_use),
		"slay_force_restart_retrain_model": bool(slay_force_restart_retrain_model),
		"slay_model_deleted_on_force_restart": bool(model_deleted_on_force_restart),
		"removed_on_force_restart": list(removed_on_force_restart),
	}


def _run_slay_analyzer_recompute(
	*,
	well_out_dir: Path,
	stage_output_root_dir: Path,
	stage_config: Any,
	sorter_output_dir: Path,
) -> dict[str, Any]:
	_log_phase_step_start(
		"SLAy analyzer recompute step start",
		well_out_dir=well_out_dir,
		sorter_output_dir=sorter_output_dir,
	)
	merge_rel_output_root = _as_optional_relpath(getattr(stage_config, "merge_rel_output_root", None))
	summary_dir = (
		(stage_output_root_dir / str(merge_rel_output_root)).resolve()
		if merge_rel_output_root is not None
		else stage_output_root_dir
	)
	summary_json = summary_dir / "slay_analyzer_recompute_summary.json"
	si_module = _import_spikeinterface_full_module()
	analyzer, analyzer_dir = _recompute_spikesort_analyzer(
		si_module=si_module,
		well_out_dir=well_out_dir,
		stage_output_root_dir=stage_output_root_dir,
		sorter_output_dir=sorter_output_dir,
		stage_config=stage_config,
	)
	payload = {
		"status": "ok",
		"well_out_dir": str(well_out_dir),
		"stage_output_root_dir": str(stage_output_root_dir),
		"sorter_output_dir": str(sorter_output_dir),
		"analyzer_dir": str(analyzer_dir),
		"unit_count": int(_unit_count(analyzer)),
	}
	_log_memory_usage(
		"SLAy analyzer recompute memory snapshot",
		well_out_dir=well_out_dir,
		sorter_output_dir=sorter_output_dir,
		analyzer_dir=analyzer_dir,
	)
	_write_json(summary_json, payload)
	return {
		"name": "slay_recompute_analyzer",
		"status": "ok",
		"reason": None,
		"out_dir": str(summary_dir),
		"summary_json": str(summary_json),
		"outputs": {
			"slay.recompute_analyzer.summary_json": str(summary_json),
			"slay.recompute_analyzer.analyzer_dir": str(analyzer_dir),
		},
	}


def _run_auto_merge_method(
	*,
	well_out_dir: Path,
	stage_output_root_dir: Path,
	output_rel_root: str,
	stage_config: Any,
	force_restart: bool,
	sorter_output_dir: Path | None,
) -> dict[str, Any]:
	merge_output_rel_root = _compose_output_rel_root(
		stage_output_rel_root=output_rel_root,
		child_rel_root=getattr(stage_config, "merge_rel_output_root", None),
	)
	auto_merge_out_dir = _resolve_under_spikesort_output_root(
		well_out_dir=well_out_dir,
		output_rel_root=merge_output_rel_root,
		relpath=str(getattr(stage_config, "auto_merge_relpath", "automerge_outputs")),
	)
	summary_json = auto_merge_out_dir / "auto_merge_method_summary.json"

	delete_on_force_restart = bool(getattr(stage_config, "auto_merge_delete_outputs_on_force_restart", True))
	removed_on_force_restart: list[str] = []
	if bool(force_restart) and bool(delete_on_force_restart) and auto_merge_out_dir.exists():
		removed_on_force_restart.append(str(auto_merge_out_dir))
		shutil.rmtree(auto_merge_out_dir, ignore_errors=True)
	auto_merge_out_dir.mkdir(parents=True, exist_ok=True)

	auto_merge_enabled = bool(getattr(stage_config, "auto_merge_enabled", False))
	if not auto_merge_enabled:
		payload = {
			"status": "skipped",
			"reason": "auto_merge_disabled",
			"well_out_dir": str(well_out_dir),
			"stage_output_root_dir": str(stage_output_root_dir),
			"auto_merge_out_dir": str(auto_merge_out_dir),
			"force_restart": bool(force_restart),
			"delete_outputs_on_force_restart": bool(delete_on_force_restart),
			"removed_on_force_restart": list(removed_on_force_restart),
		}
		_write_json(summary_json, payload)
		return {
			"name": "auto_merge",
			"status": "skipped",
			"reason": "auto_merge_disabled",
			"out_dir": str(auto_merge_out_dir),
			"summary_json": str(summary_json),
			"outputs": {
				"auto_merge.summary_json": str(summary_json),
			},
			"removed_on_force_restart": list(removed_on_force_restart),
		}

	effective_sorter_output_dir = sorter_output_dir
	if effective_sorter_output_dir is None:
		effective_sorter_output_dir = _resolve_sorter_output_dir(
			well_out_dir=well_out_dir,
			output_rel_root=output_rel_root,
			stage_config=stage_config,
		)

	si_module = _import_spikeinterface_full_module()
	current_analyzer, analyzer_dir, analyzer_rebuilt = _load_or_recompute_spikesort_analyzer(
		si_module=si_module,
		well_out_dir=well_out_dir,
		stage_output_root_dir=stage_output_root_dir,
		sorter_output_dir=effective_sorter_output_dir,
		stage_config=stage_config,
	)

	template_diff_thresholds_raw = getattr(stage_config, "auto_merge_template_diff_thresholds", (0.25,))
	template_diff_thresholds: tuple[float, ...]
	if isinstance(template_diff_thresholds_raw, (list, tuple, set)) and template_diff_thresholds_raw:
		template_diff_thresholds = tuple(float(v) for v in template_diff_thresholds_raw)
	else:
		template_diff_thresholds = (0.25,)

	auto_accept_merges = bool(getattr(stage_config, "auto_merge_auto_accept_merges", False))
	candidate_pairs_root = auto_merge_out_dir / str(getattr(stage_config, "auto_merge_candidate_pairs_reldir", "recommended_merge_candidates"))
	merged_units_root = auto_merge_out_dir / str(getattr(stage_config, "auto_merge_merged_units_reldir", "merged_units"))
	candidate_pairs_root.mkdir(parents=True, exist_ok=True)
	merged_units_root.mkdir(parents=True, exist_ok=True)

	_log_phase_step_start(
		"Auto-merge analyzer step start",
		well_out_dir=well_out_dir,
		analyzer_dir=analyzer_dir,
		thresholds=template_diff_thresholds,
		auto_accept_merges=bool(auto_accept_merges),
	)

	iteration_payloads: list[dict[str, Any]] = []
	iteration_index = 0
	total_merge_groups = 0
	total_candidate_pairs = 0
	total_applied_groups = 0
	final_unit_count = int(_unit_count(current_analyzer))

	for threshold in template_diff_thresholds:
		continue_iterations = True
		while continue_iterations:
			iteration_index += 1
			_log_phase_step_start(
				"Auto-merge iteration start",
				well_out_dir=well_out_dir,
				iteration=int(iteration_index),
				template_diff_thresh=float(threshold),
			)
			merge_groups = _compute_auto_merge_groups(
				sorting_analyzer=current_analyzer,
				template_diff_thresh=float(threshold),
			)
			pair_rows = _build_auto_merge_pair_rows(
				merge_groups=merge_groups,
				iteration_index=int(iteration_index),
				template_diff_thresh=float(threshold),
			)
			iter_prefix = f"iteration_{int(iteration_index):03d}"
			iter_groups_json = candidate_pairs_root / f"{iter_prefix}.json"
			iter_pairs_tsv = candidate_pairs_root / f"{iter_prefix}.tsv"

			_write_json(
				iter_groups_json,
				{
					"iteration": int(iteration_index),
					"template_diff_thresh": float(threshold),
					"n_groups": int(len(merge_groups)),
					"merge_groups": merge_groups,
				},
			)
			_write_auto_merge_candidate_pairs_tsv(iter_pairs_tsv, pair_rows)

			total_merge_groups += int(len(merge_groups))
			total_candidate_pairs += int(len(pair_rows))

			iteration_info: dict[str, Any] = {
				"iteration": int(iteration_index),
				"template_diff_thresh": float(threshold),
				"n_groups": int(len(merge_groups)),
				"n_candidate_pairs": int(len(pair_rows)),
				"groups_json": str(iter_groups_json),
				"pairs_tsv": str(iter_pairs_tsv),
				"applied": False,
			}

			if not merge_groups or not auto_accept_merges:
				iteration_payloads.append(iteration_info)
				continue_iterations = False
				continue

			units_before = int(_unit_count(current_analyzer))
			current_analyzer = current_analyzer.merge_units(
				merge_unit_groups=[list(group) for group in merge_groups],
				format="memory",
				merging_mode="soft",
				raise_error_if_overlap_fails=False,
			)
			units_after = int(_unit_count(current_analyzer))
			final_unit_count = int(units_after)
			total_applied_groups += int(len(merge_groups))

			merged_iter_dir = merged_units_root / iter_prefix
			merged_iter_dir.mkdir(parents=True, exist_ok=True)
			iter_apply_json = merged_iter_dir / "applied_merge_groups.json"
			_write_json(
				iter_apply_json,
				{
					"iteration": int(iteration_index),
					"template_diff_thresh": float(threshold),
					"units_before": int(units_before),
					"units_after": int(units_after),
					"n_applied_groups": int(len(merge_groups)),
					"applied_groups": merge_groups,
				},
			)

			iter_analyzer_dir = merged_iter_dir / "analyzer_output"
			analyzer_saved = False
			analyzer_save_error: str | None = None
			try:
				if iter_analyzer_dir.exists():
					shutil.rmtree(iter_analyzer_dir, ignore_errors=True)
				current_analyzer.save_as(format="binary_folder", folder=iter_analyzer_dir)
				analyzer_saved = True
			except Exception as exc:
				analyzer_save_error = f"{type(exc).__name__}: {exc}"

			iteration_info["applied"] = True
			iteration_info["applied_groups_json"] = str(iter_apply_json)
			if analyzer_saved:
				iteration_info["analyzer_output_dir"] = str(iter_analyzer_dir)
			if analyzer_save_error is not None:
				iteration_info["analyzer_output_error"] = str(analyzer_save_error)
			iteration_payloads.append(iteration_info)

			if units_after >= units_before:
				continue_iterations = False

	if auto_accept_merges and total_applied_groups > 0:
		try:
			canonical_analyzer_dir = (stage_output_root_dir / "analyzer_output").resolve()
			if canonical_analyzer_dir.exists():
				shutil.rmtree(canonical_analyzer_dir, ignore_errors=True)
			current_analyzer.save_as(format="binary_folder", folder=canonical_analyzer_dir)
		except Exception:
			pass

	outputs: dict[str, str] = {
		"auto_merge.summary_json": str(summary_json),
		"auto_merge.candidate_pairs_dir": str(candidate_pairs_root),
		"auto_merge.merged_units_dir": str(merged_units_root),
		"auto_merge.analyzer_dir": str(analyzer_dir),
	}

	payload = {
		"status": "ok",
		"well_out_dir": str(well_out_dir),
		"stage_output_root_dir": str(stage_output_root_dir),
		"auto_merge_out_dir": str(auto_merge_out_dir),
		"sorter_output_dir": str(effective_sorter_output_dir),
		"analyzer_dir": str(analyzer_dir),
		"analyzer_rebuilt": bool(analyzer_rebuilt),
		"force_restart": bool(force_restart),
		"delete_outputs_on_force_restart": bool(delete_on_force_restart),
		"removed_on_force_restart": list(removed_on_force_restart),
		"auto_accept_merges": bool(auto_accept_merges),
		"template_diff_thresholds": [float(v) for v in template_diff_thresholds],
		"n_iterations": int(len(iteration_payloads)),
		"n_candidate_groups_total": int(total_merge_groups),
		"n_candidate_pairs_total": int(total_candidate_pairs),
		"n_applied_groups_total": int(total_applied_groups),
		"final_unit_count": int(final_unit_count),
		"iterations": iteration_payloads,
		"outputs": outputs,
	}
	_write_json(summary_json, payload)

	return {
		"name": "auto_merge",
		"status": "ok",
		"reason": None,
		"out_dir": str(auto_merge_out_dir),
		"summary_json": str(summary_json),
		"outputs": outputs,
		"n_candidate_groups_total": int(total_merge_groups),
		"n_candidate_pairs_total": int(total_candidate_pairs),
		"n_applied_groups_total": int(total_applied_groups),
		"n_iterations": int(len(iteration_payloads)),
		"removed_on_force_restart": list(removed_on_force_restart),
	}


def run_spikesort_merge_stage(
	*,
	h5_path: Path,
	stream_id: str,
	mea_output_root: Path,
	output_rel_root: str,
	stage_config: Any,
	force_restart: bool,
	force_replot: bool = False,
) -> SpikesortMergeResult:
	well_out_dir = compute_mea_analysis_output_dir(
		output_root=mea_output_root,
		data_file=h5_path,
		well=stream_id,
	)
	stage_output_root_dir = _resolve_under_well(
		well_out_dir=well_out_dir,
		relpath=str(output_rel_root).strip() or "spikesort_outputs",
	)
	stage_output_root_dir.mkdir(parents=True, exist_ok=True)
	merge_rel_output_root = _as_optional_relpath(getattr(stage_config, "merge_rel_output_root", None))
	merge_output_rel_root = _compose_output_rel_root(
		stage_output_rel_root=output_rel_root,
		child_rel_root=merge_rel_output_root,
	)
	if merge_rel_output_root is not None:
		merge_phase_out_dir = _resolve_under_spikesort_output_root(
			well_out_dir=well_out_dir,
			output_rel_root=merge_output_rel_root,
			relpath="",
		)
	else:
		# Legacy default: stage-level merge artifacts live in SLAy output dir unless a merge root is configured.
		merge_phase_out_dir = _resolve_under_spikesort_output_root(
			well_out_dir=well_out_dir,
			output_rel_root=output_rel_root,
			relpath=str(getattr(stage_config, "slay_relpath", "SLAy_outputs")),
		)

	requested_sequence_raw = list(getattr(stage_config, "merge_sequence", ()) or [])
	if not requested_sequence_raw:
		requested_sequence_raw = ["SLAy", "auto_merge", "unitmatch"]
	requested_sequence_normalized = [_normalize_merge_method_token(token) for token in requested_sequence_raw]
	slay_requested = bool(bool(getattr(stage_config, "slay_enabled", False)) and ("slay" in requested_sequence_normalized))
	_log_phase_step_start(
		"Spikesort merge stage start",
		stream_id=str(stream_id),
		force_restart=bool(force_restart),
		force_replot=bool(force_replot),
		sequence=requested_sequence_raw,
	)
	merge_units_enabled = bool(getattr(stage_config, "merge_units_enabled", True))
	merge_delete_outputs_on_force_restart = bool(
		getattr(stage_config, "merge_delete_outputs_on_force_restart", False)
	)
	cache_sorting_outputs_before_merge = bool(getattr(stage_config, "cache_sorting_outputs_before_merge", False))
	merge_reports_enabled = bool(getattr(stage_config, "merge_reports_enabled", False))
	merge_reports_unit_diff_json_enabled = bool(
		getattr(stage_config, "merge_reports_unit_diff_json_enabled", False)
	)
	merge_reports_unit_diff_json_relpath = (
		str(
			getattr(
				stage_config,
				"merge_reports_unit_diff_json_relpath",
				"unit_diffs_after_merge.json",
			)
			or "unit_diffs_after_merge.json"
		)
		.strip()
		.lstrip("/")
		or "unit_diffs_after_merge.json"
	)
	merge_reports_unit_diff_map_enabled = bool(
		getattr(stage_config, "merge_reports_unit_diff_map_enabled", False)
	)
	merge_reports_unit_diff_map_relpath = (
		str(
			getattr(
				stage_config,
				"merge_reports_unit_diff_map_relpath",
				"unit_diff_map.json",
			)
			or "unit_diff_map.json"
		)
		.strip()
		.lstrip("/")
		or "unit_diff_map.json"
	)
	merge_reports_unit_diff_map_flat_enabled = bool(
		getattr(stage_config, "merge_reports_unit_diff_map_flat_enabled", False)
	)
	merge_reports_unit_diff_map_flat_relpath = (
		str(
			getattr(
				stage_config,
				"merge_reports_unit_diff_map_flat_relpath",
				"unit_diff_map_flat.json",
			)
			or "unit_diff_map_flat.json"
		)
		.strip()
		.lstrip("/")
		or "unit_diff_map_flat.json"
	)
	merge_reports_post_merge_unit_locations_enabled = bool(
		getattr(stage_config, "merge_reports_post_merge_unit_locations_enabled", False)
	)
	merge_reports_post_merge_unit_locations_relpath = (
		str(
			getattr(
				stage_config,
				"merge_reports_post_merge_unit_locations_relpath",
				"post_merge_unit_locations.json",
			)
			or "post_merge_unit_locations.json"
		)
		.strip()
		.lstrip("/")
		or "post_merge_unit_locations.json"
	)
	merge_reports_2panel_enabled = bool(getattr(stage_config, "merge_reports_2panel_enabled", False))
	merge_reports_template_heatmaps_enabled = bool(
		getattr(stage_config, "merge_reports_template_heatmaps_enabled", False)
	)
	merge_reports_mappings_enabled = bool(
		merge_reports_enabled
		and (
			merge_reports_unit_diff_json_enabled
			or merge_reports_unit_diff_map_enabled
			or merge_reports_unit_diff_map_flat_enabled
			or merge_reports_2panel_enabled
			or merge_reports_template_heatmaps_enabled
		)
	)
	merge_reports_require_snapshots = bool(
		merge_reports_enabled
		and (
			merge_reports_2panel_enabled
			or merge_reports_template_heatmaps_enabled
			or merge_reports_unit_diff_json_enabled
			or merge_reports_unit_diff_map_enabled
			or merge_reports_unit_diff_map_flat_enabled
			or merge_reports_post_merge_unit_locations_enabled
		)
	)
	merge_reports_require_unit_locations = bool(
		merge_reports_enabled
		and (
			merge_reports_2panel_enabled
			or merge_reports_post_merge_unit_locations_enabled
		)
	)
	merge_reports_any_enabled = bool(
		merge_reports_enabled
		and (
			merge_reports_2panel_enabled
			or merge_reports_template_heatmaps_enabled
			or merge_reports_unit_diff_json_enabled
			or merge_reports_unit_diff_map_enabled
			or merge_reports_unit_diff_map_flat_enabled
			or merge_reports_post_merge_unit_locations_enabled
		)
	)
	cache_sorting_outputs_before_merge_relpath = (
		str(getattr(stage_config, "cache_sorting_outputs_before_merge_relpath", "pre_merge_cache") or "pre_merge_cache")
		.strip()
		.lstrip("/")
		or "pre_merge_cache"
	)
	cache_sorting_outputs_before_merge_cleanup_on_success = bool(
		getattr(stage_config, "cache_sorting_outputs_before_merge_cleanup_on_success", False)
	)
	cache_sorting_outputs_before_merge_replace_sorting_with_cache_before_force_restart = bool(
		getattr(
			stage_config,
			"cache_sorting_outputs_before_merge_replace_sorting_with_cache_before_force_restart",
			getattr(stage_config, "cache_sorting_outputs_before_merge_use_cache_on_force_restart", False),
		)
	)
	cache_sorting_outputs_before_merge_use_cache_on_force_restart = bool(
		cache_sorting_outputs_before_merge_replace_sorting_with_cache_before_force_restart
	)
	cache_sorting_outputs_before_merge_refresh_on_run = bool(
		getattr(stage_config, "cache_sorting_outputs_before_merge_refresh_on_run", False)
	)
	cache_sorting_outputs_before_merge_strict_restore_on_force_restart = bool(
		getattr(stage_config, "cache_sorting_outputs_before_merge_strict_restore_on_force_restart", True)
	)
	cache_sorting_outputs_before_merge_use_canonical_workspace = bool(
		getattr(stage_config, "cache_sorting_outputs_before_merge_use_canonical_workspace", False)
	)
	cache_sorting_outputs_before_merge_canonical_workspace_relpath = (
		str(
			getattr(
				stage_config,
				"cache_sorting_outputs_before_merge_canonical_workspace_relpath",
				"cache/merge_workspace",
			)
			or "cache/merge_workspace"
		)
		.strip()
		.lstrip("/")
		or "cache/merge_workspace"
	)
	cache_sorting_outputs_before_merge_canonical_workspace_refresh_on_run = bool(
		getattr(stage_config, "cache_sorting_outputs_before_merge_canonical_workspace_refresh_on_run", True)
	)
	cache_sorting_outputs_before_merge_canonical_workspace_rebuild_analyzer = bool(
		getattr(stage_config, "cache_sorting_outputs_before_merge_canonical_workspace_rebuild_analyzer", False)
	)
	cache_sorting_outputs_before_merge_publish_canonical_to_stage_outputs_on_success = bool(
		getattr(
			stage_config,
			"cache_sorting_outputs_before_merge_publish_canonical_to_stage_outputs_on_success",
			False,
		)
	)
	cache_sorting_outputs_before_merge_publish_canonical_to_stage_outputs_on_failure = bool(
		getattr(
			stage_config,
			"cache_sorting_outputs_before_merge_publish_canonical_to_stage_outputs_on_failure",
			False,
		)
	)
	cache_sorting_outputs_before_merge_assert_slay_uses_canonical_workspace = bool(
		getattr(
			stage_config,
			"cache_sorting_outputs_before_merge_assert_slay_uses_canonical_workspace",
			True,
		)
	)
	cache_sorting_outputs_before_merge_assert_auto_merge_uses_canonical_workspace = bool(
		getattr(
			stage_config,
			"cache_sorting_outputs_before_merge_assert_auto_merge_uses_canonical_workspace",
			True,
		)
	)

	cache_root_dir = _resolve_under_spikesort_output_root(
		well_out_dir=well_out_dir,
		output_rel_root=merge_output_rel_root,
		relpath=str(cache_sorting_outputs_before_merge_relpath),
	)
	canonical_workspace_root_dir = _resolve_under_spikesort_output_root(
		well_out_dir=well_out_dir,
		output_rel_root=merge_output_rel_root,
		relpath=str(cache_sorting_outputs_before_merge_canonical_workspace_relpath),
	)
	slay_model_cache_path = _resolve_slay_model_cache_path(
		well_out_dir=well_out_dir,
		output_rel_root=output_rel_root,
		stage_config=stage_config,
	)

	merge_metadata_enabled = bool(getattr(stage_config, "merge_metadata_enabled", False))
	merge_metadata_write_json = bool(getattr(stage_config, "merge_metadata_write_json", True))
	merge_metadata_include_unit_locations = bool(getattr(stage_config, "merge_metadata_include_unit_locations", True))
	merge_metadata_log_summary_details = bool(getattr(stage_config, "merge_metadata_log_summary_details", False))
	merge_metadata_json_relpath = (
		str(getattr(stage_config, "merge_metadata_json_relpath", "merge_metadata_summary.json") or "merge_metadata_summary.json")
		.strip()
		.lstrip("/")
		or "merge_metadata_summary.json"
	)
	pre_merge_metadata_enabled = bool(getattr(stage_config, "pre_merge_metadata_enabled", False))
	pre_merge_metadata_write_json = bool(getattr(stage_config, "pre_merge_metadata_write_json", True))
	pre_merge_metadata_include_unit_locations = bool(
		getattr(stage_config, "pre_merge_metadata_include_unit_locations", True)
	)
	pre_merge_metadata_log_summary_details = bool(
		getattr(stage_config, "pre_merge_metadata_log_summary_details", False)
	)
	pre_merge_metadata_json_relpath = (
		str(
			getattr(
				stage_config,
				"pre_merge_metadata_json_relpath",
				"pre_merge_metadata_summary.json",
			)
			or "pre_merge_metadata_summary.json"
		)
		.strip()
		.lstrip("/")
		or "pre_merge_metadata_summary.json"
	)
	post_merge_metadata_enabled = bool(getattr(stage_config, "post_merge_metadata_enabled", False))
	post_merge_metadata_write_json = bool(getattr(stage_config, "post_merge_metadata_write_json", True))
	post_merge_metadata_include_unit_locations = bool(
		getattr(stage_config, "post_merge_metadata_include_unit_locations", True)
	)
	post_merge_metadata_log_summary_details = bool(
		getattr(stage_config, "post_merge_metadata_log_summary_details", False)
	)
	post_merge_metadata_json_relpath = (
		str(
			getattr(
				stage_config,
				"post_merge_metadata_json_relpath",
				"post_merge_metadata_summary.json",
			)
			or "post_merge_metadata_summary.json"
		)
		.strip()
		.lstrip("/")
		or "post_merge_metadata_summary.json"
	)
	pre_merge_workspace_requires_unit_locations = bool(
		((merge_metadata_enabled and merge_metadata_write_json) and merge_metadata_include_unit_locations)
		or ((pre_merge_metadata_enabled and pre_merge_metadata_write_json) and pre_merge_metadata_include_unit_locations)
		or bool(merge_reports_require_unit_locations)
	)
	pre_merge_workspace_requires_templates = bool(merge_reports_template_heatmaps_enabled)
	if not merge_units_enabled:
		primary_out_dir = merge_phase_out_dir
		primary_out_dir.mkdir(parents=True, exist_ok=True)
		summary_json = primary_out_dir / "merge_stage_summary.json"
		outputs: dict[str, str] = {
			"summary_json": str(summary_json),
		}
		payload: dict[str, Any] = {
			"status": "skipped",
			"reason": "merge_units_disabled",
			"well_out_dir": str(well_out_dir),
			"stage_output_root_dir": str(stage_output_root_dir),
			"merge_out_dir": str(primary_out_dir),
			"merge_rel_output_root": (str(merge_rel_output_root) if merge_rel_output_root is not None else None),
			"merge_output_rel_root": str(merge_output_rel_root),
			"merge_delete_outputs_on_force_restart": bool(merge_delete_outputs_on_force_restart),
			"force_restart": bool(force_restart),
			"force_replot": bool(force_replot),
			"replot_only": False,
			"requested_sequence": [str(token) for token in requested_sequence_raw],
			"methods": [],
			"merge_units_enabled": False,
			"bombcell_label_config": {
				"enabled": bool(getattr(stage_config, "bombcell_label_enabled", False)),
				"relpath": str(getattr(stage_config, "bombcell_label_relpath", "bombcell_label_outputs")),
				"delete_outputs_on_force_restart": bool(
					getattr(stage_config, "bombcell_label_delete_outputs_on_force_restart", True)
				),
				"label_non_somatic": bool(getattr(stage_config, "bombcell_label_label_non_somatic", True)),
				"split_non_somatic_good_mua": bool(
					getattr(stage_config, "bombcell_label_split_non_somatic_good_mua", True)
				),
				"apply_to_sorter_output": bool(getattr(stage_config, "bombcell_label_apply_to_sorter_output", True)),
				"write_cluster_group": bool(getattr(stage_config, "bombcell_label_write_cluster_group", True)),
				"fail_on_error": bool(getattr(stage_config, "bombcell_label_fail_on_error", False)),
			},
			"cache_sorting_outputs_before_merge": bool(cache_sorting_outputs_before_merge),
			"merge_reports_enabled": bool(merge_reports_any_enabled),
			"slay_model_cache_path": (str(slay_model_cache_path) if slay_model_cache_path is not None else None),
			"slay_model_cache_use_cached_model": bool(
				getattr(stage_config, "slay_model_cache_use_cached_model", True)
			),
			"slay_model_cache_write_model": bool(
				getattr(stage_config, "slay_model_cache_write_model", True)
			),
			"slay_force_restart_retrain_model": bool(getattr(stage_config, "slay_force_restart_retrain_model", False)),
			"cache_sorting_outputs_before_merge_config": {
				"enabled": bool(cache_sorting_outputs_before_merge),
				"relpath": str(cache_sorting_outputs_before_merge_relpath),
				"cleanup_on_success": bool(cache_sorting_outputs_before_merge_cleanup_on_success),
				"replace_sorting_with_cache_before_force_restart": bool(
					cache_sorting_outputs_before_merge_replace_sorting_with_cache_before_force_restart
				),
				"use_cache_on_force_restart": bool(cache_sorting_outputs_before_merge_use_cache_on_force_restart),
				"refresh_on_run": bool(cache_sorting_outputs_before_merge_refresh_on_run),
				"strict_restore_on_force_restart": bool(
					cache_sorting_outputs_before_merge_strict_restore_on_force_restart
				),
				"use_canonical_workspace": bool(cache_sorting_outputs_before_merge_use_canonical_workspace),
				"canonical_workspace_relpath": str(cache_sorting_outputs_before_merge_canonical_workspace_relpath),
				"canonical_workspace_refresh_on_run": bool(
					cache_sorting_outputs_before_merge_canonical_workspace_refresh_on_run
				),
				"canonical_workspace_rebuild_analyzer": bool(
					cache_sorting_outputs_before_merge_canonical_workspace_rebuild_analyzer
				),
				"publish_canonical_to_stage_outputs_on_success": bool(
					cache_sorting_outputs_before_merge_publish_canonical_to_stage_outputs_on_success
				),
				"publish_canonical_to_stage_outputs_on_failure": bool(
					cache_sorting_outputs_before_merge_publish_canonical_to_stage_outputs_on_failure
				),
			},
			"merge_metadata_enabled": False,
			"outputs": outputs,
		}
		_write_json(summary_json, payload)
		if merge_metadata_log_summary_details:
			_log_merge_summary_details(
				stream_id=str(stream_id),
				status="skipped",
				method_reports=[],
				summary_json=summary_json,
				merge_metadata_enabled=False,
				merge_metadata_json=None,
				merge_metadata_payload=None,
				merge_metadata_error=None,
			)
		return SpikesortMergeResult(
			well_out_dir=well_out_dir,
			merge_out_dir=primary_out_dir,
			summary_json=summary_json,
			outputs=outputs,
		)

	replot_only_mode = bool(force_replot) and (not bool(force_restart))
	if replot_only_mode:
		_log_phase_step_start(
			"Merge replot-only step start",
			stream_id=str(stream_id),
			merge_out_dir=merge_phase_out_dir,
		)
		merge_analyzer_regenerate_on_replot = bool(
			getattr(stage_config, "merge_analyzer_regenerate_on_replot", True)
		)
		merge_analyzer_check_if_regen_is_needed = bool(
			getattr(stage_config, "merge_analyzer_check_if_regen_is_needed", True)
		)
		primary_out_dir = merge_phase_out_dir
		primary_out_dir.mkdir(parents=True, exist_ok=True)
		summary_json = primary_out_dir / "merge_stage_summary.json"
		existing_summary = (_read_json_dict(summary_json) or {})

		existing_outputs_raw = existing_summary.get("outputs", {})
		combined_outputs: dict[str, str] = {}
		if isinstance(existing_outputs_raw, dict):
			for raw_key, raw_value in existing_outputs_raw.items():
				key = str(raw_key)
				val = str(raw_value)
				if key:
					combined_outputs[key] = val

		pre_merge_workspace_raw = existing_summary.get("pre_merge_workspace", {})
		pre_merge_workspace_policy = (
			dict(pre_merge_workspace_raw.get("analyzer_policy", {}))
			if isinstance(pre_merge_workspace_raw, dict)
			and isinstance(pre_merge_workspace_raw.get("analyzer_policy", {}), dict)
			else None
		)
		if pre_merge_workspace_policy is None:
			merge_analyzer_policy_raw = existing_summary.get("merge_analyzer_policy", {})
			if isinstance(merge_analyzer_policy_raw, dict):
				pre_merge_workspace_policy = dict(merge_analyzer_policy_raw)

		post_merge_workspace_raw = existing_summary.get("post_merge_workspace", {})
		post_merge_workspace_policy = (
			dict(post_merge_workspace_raw.get("analyzer_policy", {}))
			if isinstance(post_merge_workspace_raw, dict)
			and isinstance(post_merge_workspace_raw.get("analyzer_policy", {}), dict)
			else None
		)
		if post_merge_workspace_policy is None and pre_merge_workspace_policy is not None:
			post_merge_workspace_policy = dict(pre_merge_workspace_policy)

		pre_merge_workspace_relpath = str(
			getattr(stage_config, "pre_merge_workspace_relpath", "cache/merge_workspace")
			or "cache/merge_workspace"
		).strip().lstrip("/") or "cache/merge_workspace"
		pre_merge_workspace_dir = _resolve_under_spikesort_output_root(
			well_out_dir=well_out_dir,
			output_rel_root=merge_output_rel_root,
			relpath=pre_merge_workspace_relpath,
		)
		pre_merge_workspace_analyzer_output_dir = (
			pre_merge_workspace_dir / "pre_merge_analyzer_output"
		).resolve()
		# Replot must rebuild the pre-merge analyzer from the live pre-merge sorter output,
		# not from a previously-used workspace sorter path (which may already be post-merge).
		try:
			pre_merge_workspace_sorter_output_dir = _resolve_sorter_output_dir(
				well_out_dir=well_out_dir,
				output_rel_root=output_rel_root,
				stage_config=stage_config,
			)
		except Exception:
			pre_merge_workspace_sorter_output_dir = (stage_output_root_dir / "sorter_output").resolve()

		pre_merge_workspace_analyzer_built = False
		pre_merge_workspace_analyzer_error: str | None = None
		pre_merge_workspace_analyzer_policy: dict[str, Any] | None = None
		pre_merge_workspace_analyzer_regenerated = False
		pre_merge_workspace_analyzer_regen_reason: str | None = None
		pre_merge_workspace_analyzer: Any | None = None
		post_merge_workspace_sorter_output_dir = (pre_merge_workspace_dir / "sorter_output").resolve()
		post_merge_workspace_analyzer_output_dir = (pre_merge_workspace_dir / "analyzer_output").resolve()
		post_merge_workspace_analyzer_built = False
		post_merge_workspace_analyzer_error: str | None = None
		post_merge_workspace_analyzer_policy: dict[str, Any] | None = None
		post_merge_workspace_analyzer_regenerated = False
		post_merge_workspace_analyzer_regen_reason: str | None = None
		post_merge_workspace_analyzer: Any | None = None
		si_module: Any | None = None
		if pre_merge_workspace_sorter_output_dir.exists():
			try:
				_log_phase_step_start(
					"Merge replot analyzer prepare step start",
					stream_id=str(stream_id),
					sorter_output_dir=pre_merge_workspace_sorter_output_dir,
					analyzer_output_dir=pre_merge_workspace_analyzer_output_dir,
					regenerate_on_replot=bool(merge_analyzer_regenerate_on_replot),
					check_if_regen_is_needed=bool(merge_analyzer_check_if_regen_is_needed),
				)
				if si_module is None:
					si_module = _import_spikeinterface_full_module()
				(
					pre_merge_workspace_analyzer,
					pre_merge_workspace_analyzer_output_dir,
					pre_merge_workspace_analyzer_policy,
					pre_merge_workspace_analyzer_regenerated,
					pre_merge_workspace_analyzer_regen_reason,
				) = _prepare_replot_workspace_analyzer(
					si_module=si_module,
					well_out_dir=well_out_dir,
					sorter_output_dir=pre_merge_workspace_sorter_output_dir,
					stage_config=stage_config,
					analyzer_dir=pre_merge_workspace_analyzer_output_dir,
					regenerate_on_replot=bool(merge_analyzer_regenerate_on_replot),
					check_if_regen_is_needed=bool(merge_analyzer_check_if_regen_is_needed),
					fallback_policy=pre_merge_workspace_policy,
				)
				if pre_merge_workspace_requires_templates or pre_merge_workspace_requires_unit_locations:
					_ensure_merge_analyzer_extensions(
						analyzer=pre_merge_workspace_analyzer,
						stage_config=stage_config,
						include_unit_locations=pre_merge_workspace_requires_unit_locations,
					)
				pre_merge_workspace_analyzer_built = True
				_log_memory_usage(
					"Merge replot pre-merge analyzer memory snapshot",
					stream_id=str(stream_id),
					sorter_output_dir=pre_merge_workspace_sorter_output_dir,
					analyzer_output_dir=pre_merge_workspace_analyzer_output_dir,
					regenerated=bool(pre_merge_workspace_analyzer_regenerated),
					regen_reason=pre_merge_workspace_analyzer_regen_reason,
				)
			except Exception as exc:
				pre_merge_workspace_analyzer_error = (
					f"replot_pre_merge_workspace_analyzer_prepare_failed:{type(exc).__name__}:{exc}"
				)

		if (
			(merge_reports_template_heatmaps_enabled or merge_reports_require_unit_locations)
			and post_merge_workspace_sorter_output_dir.exists()
		):
			try:
				_log_phase_step_start(
					"Merge replot post-merge analyzer prepare step start",
					stream_id=str(stream_id),
					sorter_output_dir=post_merge_workspace_sorter_output_dir,
					analyzer_output_dir=post_merge_workspace_analyzer_output_dir,
					regenerate_on_replot=bool(merge_analyzer_regenerate_on_replot),
					check_if_regen_is_needed=bool(merge_analyzer_check_if_regen_is_needed),
				)
				if si_module is None:
					si_module = _import_spikeinterface_full_module()
				(
					post_merge_workspace_analyzer,
					post_merge_workspace_analyzer_output_dir,
					post_merge_workspace_analyzer_policy,
					post_merge_workspace_analyzer_regenerated,
					post_merge_workspace_analyzer_regen_reason,
				) = _prepare_replot_workspace_analyzer(
					si_module=si_module,
					well_out_dir=well_out_dir,
					sorter_output_dir=post_merge_workspace_sorter_output_dir,
					stage_config=stage_config,
					analyzer_dir=post_merge_workspace_analyzer_output_dir,
					regenerate_on_replot=bool(merge_analyzer_regenerate_on_replot),
					check_if_regen_is_needed=bool(merge_analyzer_check_if_regen_is_needed),
					fallback_policy=post_merge_workspace_policy,
				)
				_ensure_merge_analyzer_extensions(
					analyzer=post_merge_workspace_analyzer,
					stage_config=stage_config,
					include_unit_locations=merge_reports_require_unit_locations,
				)
				post_merge_workspace_analyzer_built = True
				_log_memory_usage(
					"Merge replot post-merge analyzer memory snapshot",
					stream_id=str(stream_id),
					sorter_output_dir=post_merge_workspace_sorter_output_dir,
					analyzer_output_dir=post_merge_workspace_analyzer_output_dir,
					regenerated=bool(post_merge_workspace_analyzer_regenerated),
					regen_reason=post_merge_workspace_analyzer_regen_reason,
				)
			except Exception as exc:
				post_merge_workspace_analyzer_error = (
					f"replot_post_merge_workspace_analyzer_prepare_failed:{type(exc).__name__}:{exc}"
				)

		combined_outputs["merge.pre_merge_workspace_dir"] = str(pre_merge_workspace_dir.resolve())
		combined_outputs["merge.pre_merge_workspace_sorter_output_dir"] = str(
			pre_merge_workspace_sorter_output_dir.resolve()
		)
		if pre_merge_workspace_analyzer_output_dir.exists():
			combined_outputs["merge.pre_merge_workspace_analyzer_output_dir"] = str(
				pre_merge_workspace_analyzer_output_dir.resolve()
			)

		combined_outputs["merge.post_merge_workspace_sorter_output_dir"] = str(
			post_merge_workspace_sorter_output_dir.resolve()
		)
		if post_merge_workspace_analyzer_output_dir.exists():
			combined_outputs["merge.post_merge_analyzer_output_dir"] = str(
				post_merge_workspace_analyzer_output_dir.resolve()
			)

		method_reports_raw = existing_summary.get("methods", [])
		method_reports = (
			list(method_reports_raw)
			if isinstance(method_reports_raw, list)
			else []
		)

		merge_metadata_payload: dict[str, Any] | None = None
		merge_metadata_json: Path | None = None
		merge_metadata_error: str | None = None
		if merge_metadata_enabled and merge_metadata_write_json:
			merge_metadata_json_raw = existing_summary.get("merge_metadata_summary_json", None)
			if merge_metadata_json_raw is None and isinstance(existing_outputs_raw, dict):
				merge_metadata_json_raw = existing_outputs_raw.get("merge.metadata_summary_json", None)
			if merge_metadata_json_raw is None:
				merge_metadata_json = (primary_out_dir / str(merge_metadata_json_relpath)).resolve()
			else:
				merge_metadata_json = Path(str(merge_metadata_json_raw)).resolve()
			merge_metadata_payload = _read_json_dict(merge_metadata_json)
			if merge_metadata_payload is None:
				merge_metadata_error = "merge_metadata_summary_missing_or_invalid_for_replot"

		merge_unit_diff_payload: dict[str, Any] | None = None
		merge_unit_diff_json: Path | None = None
		merge_unit_diff_error: str | None = None
		if merge_reports_enabled and merge_reports_unit_diff_json_enabled:
			merge_unit_diff_json_raw = existing_summary.get("merge_unit_diff_json", None)
			if merge_unit_diff_json_raw is None and isinstance(existing_outputs_raw, dict):
				merge_unit_diff_json_raw = existing_outputs_raw.get("merge.report.unit_diff_json", None)
			if merge_unit_diff_json_raw is None:
				merge_unit_diff_json = (primary_out_dir / str(merge_reports_unit_diff_json_relpath)).resolve()
			else:
				merge_unit_diff_json = Path(str(merge_unit_diff_json_raw)).resolve()
			merge_unit_diff_payload = _read_json_dict(merge_unit_diff_json)
			if merge_unit_diff_payload is None:
				merge_unit_diff_error = "merge_unit_diff_json_missing_or_invalid_for_replot"
			else:
				combined_outputs["merge.report.unit_diff_json"] = str(merge_unit_diff_json)

		merge_reports_payload: dict[str, Any] | None = None
		merge_reports_error: str | None = None
		merge_template_heatmaps_payload: dict[str, Any] | None = None
		merge_template_heatmaps_error: str | None = None
		if merge_reports_enabled and (merge_reports_2panel_enabled or merge_reports_template_heatmaps_enabled):
			_log_phase_step_start(
				"Merge replot reports step start",
				stream_id=str(stream_id),
				unit_location_reports=bool(merge_reports_2panel_enabled),
				template_heatmaps=bool(merge_reports_template_heatmaps_enabled),
			)
			before_snapshot_for_report: dict[str, Any] = {}
			after_snapshot_for_report: dict[str, Any] = {}
			applied_unit_mappings_for_report: list[dict[str, Any]] = []
			if merge_reports_unit_diff_json_enabled:
				if isinstance(merge_unit_diff_payload, dict):
					(
						before_snapshot_for_report,
						after_snapshot_for_report,
						applied_unit_mappings_for_report,
					) = _extract_plot_inputs_from_unit_diff_report(
						unit_diff_payload=merge_unit_diff_payload,
					)
				else:
					merge_reports_error = "merge_reports_missing_unit_diff_json_source_for_replot"
			elif isinstance(merge_metadata_payload, dict):
				before_raw = merge_metadata_payload.get("before", None)
				after_raw = merge_metadata_payload.get("after", None)
				if isinstance(before_raw, dict):
					before_snapshot_for_report = dict(before_raw)
				if isinstance(after_raw, dict):
					after_snapshot_for_report = dict(after_raw)
				applied_unit_mappings_for_report = _extract_applied_unit_mappings_for_report(
					merge_metadata_payload=merge_metadata_payload
				)

			preferred_pre_analyzer_raw: Any = combined_outputs.get(
				"merge.pre_merge_workspace_analyzer_output_dir",
				None,
			)
			if merge_reports_error is None:
				if preferred_pre_analyzer_raw is None:
					merge_reports_error = (
						"merge_reports_missing_pre_merge_workspace_analyzer_output_dir_for_replot"
					)
				else:
					preferred_pre_analyzer_dir = Path(str(preferred_pre_analyzer_raw)).expanduser().resolve()
					if not preferred_pre_analyzer_dir.exists():
						merge_reports_error = (
							"merge_reports_missing_pre_merge_workspace_analyzer_for_replot"
						)
					else:
						before_analyzer_raw = before_snapshot_for_report.get("analyzer", {})
						before_analyzer = (
							dict(before_analyzer_raw)
							if isinstance(before_analyzer_raw, dict)
							else {}
						)
						before_analyzer["source_dir"] = str(preferred_pre_analyzer_dir)
						before_snapshot_for_report["analyzer"] = before_analyzer

			preferred_post_analyzer_raw: Any = combined_outputs.get(
				"merge.post_merge_analyzer_output_dir",
				None,
			)
			if merge_reports_error is None and merge_reports_template_heatmaps_enabled:
				if preferred_post_analyzer_raw is None:
					merge_reports_error = (
						"merge_reports_missing_post_merge_analyzer_output_dir_for_replot"
					)
				else:
					preferred_post_analyzer_dir = Path(str(preferred_post_analyzer_raw)).expanduser().resolve()
					if not preferred_post_analyzer_dir.exists():
						merge_reports_error = (
							"merge_reports_missing_post_merge_analyzer_for_replot"
						)
					else:
						after_analyzer_raw = after_snapshot_for_report.get("analyzer", {})
						after_analyzer = (
							dict(after_analyzer_raw)
							if isinstance(after_analyzer_raw, dict)
							else {}
						)
						after_analyzer["source_dir"] = str(preferred_post_analyzer_dir)
						after_snapshot_for_report["analyzer"] = after_analyzer
			elif preferred_post_analyzer_raw is not None:
				preferred_post_analyzer_dir = Path(str(preferred_post_analyzer_raw)).expanduser().resolve()
				if preferred_post_analyzer_dir.exists():
					after_analyzer_raw = after_snapshot_for_report.get("analyzer", {})
					after_analyzer = (
						dict(after_analyzer_raw)
						if isinstance(after_analyzer_raw, dict)
						else {}
					)
					after_analyzer["source_dir"] = str(preferred_post_analyzer_dir)
					after_snapshot_for_report["analyzer"] = after_analyzer

			if merge_reports_error is None and merge_reports_require_unit_locations:
				if pre_merge_workspace_analyzer_built and pre_merge_workspace_analyzer is not None:
					before_snapshot_for_report = _refresh_snapshot_analyzer_payload_from_live_analyzer(
						snapshot=before_snapshot_for_report,
						analyzer=pre_merge_workspace_analyzer,
						stage_config=stage_config,
						include_unit_locations=True,
						analyzer_source_dir=pre_merge_workspace_analyzer_output_dir,
					)
				if post_merge_workspace_analyzer_built and post_merge_workspace_analyzer is not None:
					after_snapshot_for_report = _refresh_snapshot_analyzer_payload_from_live_analyzer(
						snapshot=after_snapshot_for_report,
						analyzer=post_merge_workspace_analyzer,
						stage_config=stage_config,
						include_unit_locations=True,
						analyzer_source_dir=post_merge_workspace_analyzer_output_dir,
					)

			if merge_reports_error is None and (not before_snapshot_for_report or not after_snapshot_for_report):
				merge_reports_error = "merge_reports_missing_before_after_snapshots_for_replot"
			if merge_reports_error is None and merge_reports_2panel_enabled:
				try:
					merge_reports_payload = _write_merge_unit_location_reports(
						merge_out_dir=primary_out_dir,
						before_snapshot=before_snapshot_for_report,
						after_snapshot=after_snapshot_for_report,
						applied_unit_mappings=applied_unit_mappings_for_report,
						stage_config=stage_config,
					)
					if str(merge_reports_payload.get("status", "")) == "ok":
						combined_outputs.update(dict(merge_reports_payload.get("outputs", {})))
					else:
						merge_reports_error = str(merge_reports_payload.get("error", "merge_reports_failed"))
				except Exception as exc:
					merge_reports_error = f"merge_reports_failed:{type(exc).__name__}:{exc}"

			if merge_reports_error is None and merge_reports_template_heatmaps_enabled:
				try:
					merge_template_heatmaps_payload = _write_merge_template_heatmap_reports(
						merge_out_dir=primary_out_dir,
						before_snapshot=before_snapshot_for_report,
						after_snapshot=after_snapshot_for_report,
						applied_unit_mappings=applied_unit_mappings_for_report,
						stage_config=stage_config,
						before_analyzer=(pre_merge_workspace_analyzer if pre_merge_workspace_analyzer_built else None),
						after_analyzer=(post_merge_workspace_analyzer if post_merge_workspace_analyzer_built else None),
					)
					if str(merge_template_heatmaps_payload.get("status", "")) == "ok":
						combined_outputs.update(dict(merge_template_heatmaps_payload.get("outputs", {})))
					elif str(merge_template_heatmaps_payload.get("status", "")) not in {"", "skipped"}:
						merge_template_heatmaps_error = str(
							merge_template_heatmaps_payload.get(
								"error",
								"merge_template_heatmaps_failed",
							)
						)
				except Exception as exc:
					merge_template_heatmaps_error = (
						f"merge_template_heatmaps_failed:{type(exc).__name__}:{exc}"
					)

		if merge_reports_enabled and (merge_reports_2panel_enabled or merge_reports_template_heatmaps_enabled):
			_log_memory_usage(
				"Merge replot report rendering memory snapshot",
				stream_id=str(stream_id),
				merge_out_dir=primary_out_dir,
				unit_location_reports=bool(merge_reports_2panel_enabled),
				template_heatmaps=bool(merge_reports_template_heatmaps_enabled),
			)

		released_pre_extensions: list[str] = []
		if pre_merge_workspace_analyzer is not None:
			released_pre_extensions = _release_loaded_analyzer_extensions(analyzer=pre_merge_workspace_analyzer)
			pre_merge_workspace_analyzer = None
			gc.collect()
		released_post_extensions: list[str] = []
		if post_merge_workspace_analyzer is not None:
			released_post_extensions = _release_loaded_analyzer_extensions(analyzer=post_merge_workspace_analyzer)
			post_merge_workspace_analyzer = None
			gc.collect()
		if released_pre_extensions or released_post_extensions:
			_log_memory_usage(
				"Merge replot analyzer release memory snapshot",
				stream_id=str(stream_id),
				released_pre_extensions=int(len(released_pre_extensions)),
				released_post_extensions=int(len(released_post_extensions)),
			)

		status = "ok"
		reason: str | None = "force_replot_only"
		if merge_reports_error is not None or merge_template_heatmaps_error is not None:
			status = "skipped"
			reason = "force_replot_failed"

		payload: dict[str, Any] = {
			"status": str(status),
			"reason": reason,
			"well_out_dir": str(well_out_dir),
			"stage_output_root_dir": str(stage_output_root_dir),
			"merge_out_dir": str(primary_out_dir),
			"merge_rel_output_root": (str(merge_rel_output_root) if merge_rel_output_root is not None else None),
			"merge_output_rel_root": str(merge_output_rel_root),
			"merge_delete_outputs_on_force_restart": bool(merge_delete_outputs_on_force_restart),
			"force_restart": bool(force_restart),
			"force_replot": bool(force_replot),
			"replot_only": True,
			"merge_units_enabled": bool(merge_units_enabled),
			"cache_sorting_outputs_before_merge": bool(cache_sorting_outputs_before_merge),
			"merge_reports_enabled": bool(merge_reports_any_enabled),
			"requested_sequence": [str(token) for token in requested_sequence_raw],
			"methods": method_reports,
			"merge_metadata_enabled": bool(merge_metadata_enabled and merge_metadata_write_json),
			"outputs": combined_outputs,
		}
		payload["pre_merge_workspace"] = {
			"relpath": str(pre_merge_workspace_relpath),
			"workspace_dir": str(pre_merge_workspace_dir.resolve()),
			"sorter_output_dir": str(pre_merge_workspace_sorter_output_dir.resolve()),
			"analyzer_output_dir": str(pre_merge_workspace_analyzer_output_dir.resolve()),
			"analyzer_built": bool(pre_merge_workspace_analyzer_built),
			"analyzer_regenerated": bool(pre_merge_workspace_analyzer_regenerated),
			"analyzer_regen_reason": pre_merge_workspace_analyzer_regen_reason,
			"analyzer_policy": (
				dict(pre_merge_workspace_analyzer_policy)
				if isinstance(pre_merge_workspace_analyzer_policy, dict)
				else _requested_merge_analyzer_policy(stage_config)
			),
		}
		payload["post_merge_workspace"] = {
			"workspace_dir": str(pre_merge_workspace_dir.resolve()),
			"sorter_output_dir": str(post_merge_workspace_sorter_output_dir.resolve()),
			"analyzer_output_dir": str(post_merge_workspace_analyzer_output_dir.resolve()),
			"analyzer_built": bool(post_merge_workspace_analyzer_built),
			"analyzer_regenerated": bool(post_merge_workspace_analyzer_regenerated),
			"analyzer_regen_reason": post_merge_workspace_analyzer_regen_reason,
			"analyzer_policy": (
				dict(post_merge_workspace_analyzer_policy)
				if isinstance(post_merge_workspace_analyzer_policy, dict)
				else _requested_merge_analyzer_policy(stage_config)
			),
		}
		payload["merge_analyzer_policy"] = _requested_merge_analyzer_policy(stage_config)
		if pre_merge_workspace_analyzer_error is not None:
			payload["pre_merge_workspace_analyzer_error"] = str(pre_merge_workspace_analyzer_error)
		if post_merge_workspace_analyzer_error is not None:
			payload["post_merge_workspace_analyzer_error"] = str(post_merge_workspace_analyzer_error)
		if merge_metadata_json is not None:
			payload["merge_metadata_summary_json"] = str(merge_metadata_json)
		if isinstance(merge_metadata_payload, dict):
			payload["merge_metadata"] = {
				"applied_merge_group_count": int(merge_metadata_payload.get("applied_merge_group_count", 0) or 0),
				"change_validation": merge_metadata_payload.get("change_validation", {}),
			}
		if merge_metadata_error is not None:
			payload["merge_metadata_error"] = str(merge_metadata_error)
		if merge_unit_diff_json is not None:
			payload["merge_unit_diff_json"] = str(merge_unit_diff_json)
		if isinstance(merge_unit_diff_payload, dict):
			payload["merge_unit_diff"] = {
				"applied_merge_group_count": int(merge_unit_diff_payload.get("applied_merge_group_count", 0) or 0),
				"change_validation": merge_unit_diff_payload.get("change_validation", {}),
			}
		if merge_unit_diff_error is not None:
			payload["merge_unit_diff_error"] = str(merge_unit_diff_error)
		if isinstance(merge_reports_payload, dict):
			payload["merge_reports"] = {
				"status": str(merge_reports_payload.get("status", "ok")),
				"before_unit_locations_count": int(merge_reports_payload.get("before_unit_locations_count", 0) or 0),
				"after_unit_locations_count": int(merge_reports_payload.get("after_unit_locations_count", 0) or 0),
				"before_highlighted_units_count": int(merge_reports_payload.get("before_highlighted_units_count", 0) or 0),
				"after_highlighted_units_count": int(merge_reports_payload.get("after_highlighted_units_count", 0) or 0),
				"after_highlighted_inferred_units_count": int(
					merge_reports_payload.get("after_highlighted_inferred_units_count", 0) or 0
				),
			}
		if merge_reports_error is not None:
			payload["merge_reports_error"] = str(merge_reports_error)
		if isinstance(merge_template_heatmaps_payload, dict):
			payload["merge_template_heatmaps"] = {
				"status": str(merge_template_heatmaps_payload.get("status", "ok")),
				"n_mappings_requested": int(
					merge_template_heatmaps_payload.get("n_mappings_requested", 0) or 0
				),
				"n_mappings_processed": int(
					merge_template_heatmaps_payload.get("n_mappings_processed", 0) or 0
				),
				"debug_json": merge_template_heatmaps_payload.get("debug_json", None),
			}
		if merge_template_heatmaps_error is not None:
			payload["merge_template_heatmaps_error"] = str(merge_template_heatmaps_error)

		combined_outputs["summary_json"] = str(summary_json)
		_write_json(summary_json, payload)

		return SpikesortMergeResult(
			well_out_dir=well_out_dir,
			merge_out_dir=primary_out_dir,
			summary_json=summary_json,
			outputs=combined_outputs,
		)

	merge_phase_removed_on_force_restart: list[str] = []
	if bool(force_restart) and bool(merge_delete_outputs_on_force_restart) and merge_phase_out_dir.exists():
		preserved_targets: list[Path] = []
		if cache_sorting_outputs_before_merge and cache_root_dir.exists():
			preserved_targets.append(cache_root_dir.resolve())
		if (
			cache_sorting_outputs_before_merge_use_canonical_workspace
			and (not cache_sorting_outputs_before_merge_canonical_workspace_refresh_on_run)
			and canonical_workspace_root_dir.exists()
		):
			preserved_targets.append(canonical_workspace_root_dir.resolve())
		if slay_model_cache_path is not None and slay_model_cache_path.exists():
			preserved_targets.append(slay_model_cache_path.resolve())

		if preserved_targets:
			for child in list(merge_phase_out_dir.iterdir()):
				_remove_path_preserving_targets(
					target=child,
					preserved_targets=preserved_targets,
					removed_paths=merge_phase_removed_on_force_restart,
				)
		else:
			shutil.rmtree(merge_phase_out_dir, ignore_errors=True)
			merge_phase_removed_on_force_restart.append(str(merge_phase_out_dir.resolve()))

	cache_summary: dict[str, Any] | None = None
	cache_restore_summary: dict[str, Any] | None = None
	cache_error: str | None = None
	cache_restored_from_existing = False
	cache_preserved_existing = False
	cache_reseeded_after_restore_miss = False
	cache_restore_precheck_missing_sources: list[str] = []
	cache_outputs: dict[str, str] = {}
	if cache_sorting_outputs_before_merge:
		_log_phase_step_start(
			"Merge pre-cache step start",
			stream_id=str(stream_id),
			cache_root_dir=cache_root_dir,
			force_restart=bool(force_restart),
			use_cache_on_force_restart=bool(
				cache_sorting_outputs_before_merge_use_cache_on_force_restart
			),
		)
		try:
			if bool(force_restart) and bool(cache_sorting_outputs_before_merge_use_cache_on_force_restart):
				expected_cache_sources = [
					(cache_root_dir / "sorter_output").resolve(),
					(cache_root_dir / "analyzer_output").resolve(),
				]
				cache_restore_precheck_missing_sources = [
					str(path)
					for path in expected_cache_sources
					if not path.exists()
				]
				if cache_restore_precheck_missing_sources:
					if cache_sorting_outputs_before_merge_strict_restore_on_force_restart:
						raise FileNotFoundError(
							"Missing required pre-merge cache sources for force-restart: "
							+ ", ".join(cache_restore_precheck_missing_sources)
						)
					cache_reseeded_after_restore_miss = True
				else:
					cache_restore_summary = _restore_sorting_outputs_from_pre_merge_cache(
						stage_output_root_dir=stage_output_root_dir,
						cache_root_dir=cache_root_dir,
					)
					missing_after_restore = list(cache_restore_summary.get("missing_cache_sources", []) or [])
					if missing_after_restore:
						if cache_sorting_outputs_before_merge_strict_restore_on_force_restart:
							raise FileNotFoundError(
								"Missing required pre-merge cache sources during restore: "
								+ ", ".join(missing_after_restore)
							)
						cache_reseeded_after_restore_miss = True
					else:
						cache_restored_from_existing = True

			if not cache_restored_from_existing:
				should_refresh_cache = bool(
					(not cache_root_dir.exists())
					or cache_sorting_outputs_before_merge_refresh_on_run
					or cache_reseeded_after_restore_miss
				)
				if should_refresh_cache:
					cache_summary = _cache_sorting_outputs_before_merge(
						stage_output_root_dir=stage_output_root_dir,
						cache_root_dir=cache_root_dir,
					)
				else:
					cache_preserved_existing = True

			cache_outputs["merge.pre_merge_cache_dir"] = str(cache_root_dir)
			sorter_cache_dir = (cache_root_dir / "sorter_output").resolve()
			analyzer_cache_dir = (cache_root_dir / "analyzer_output").resolve()
			if sorter_cache_dir.exists():
				cache_outputs["merge.pre_merge_cache_sorter_output_dir"] = str(sorter_cache_dir)
			if analyzer_cache_dir.exists():
				cache_outputs["merge.pre_merge_cache_analyzer_output_dir"] = str(analyzer_cache_dir)
			summary_json_obj = (cache_root_dir / "pre_merge_cache_summary.json").resolve()
			if summary_json_obj.exists():
				cache_outputs["merge.pre_merge_cache_summary_json"] = str(summary_json_obj)
		except Exception as exc:
			cache_error = f"pre_merge_cache_failed:{type(exc).__name__}:{exc}"
			if (
				bool(force_restart)
				and bool(cache_sorting_outputs_before_merge_use_cache_on_force_restart)
				and bool(cache_sorting_outputs_before_merge_strict_restore_on_force_restart)
			):
				raise RuntimeError(cache_error) from exc

	if slay_model_cache_path is not None:
		cache_outputs["slay.model_cache_path"] = str(slay_model_cache_path.resolve())

	canonical_workspace_summary: dict[str, Any] | None = None
	canonical_workspace_error: str | None = None
	canonical_workspace_prepared = False
	canonical_workspace_preserved_existing = False
	canonical_workspace_sorter_output_dir: Path | None = None
	canonical_workspace_analyzer_output_dir: Path | None = None
	canonical_workspace_publish_summary: dict[str, Any] | None = None
	canonical_workspace_publish_error: str | None = None
	canonical_workspace_published = False
	active_stage_output_root_dir: Path = stage_output_root_dir
	active_sorter_output_dir: Path | None = None
	slay_binary_input_preflight: dict[str, Any] | None = None
	bombcell_report: dict[str, Any] | None = None
	bombcell_report_error: str | None = None

	if cache_sorting_outputs_before_merge_use_canonical_workspace:
		_log_phase_step_start(
			"Merge working cache prepare step start",
			stream_id=str(stream_id),
			workspace_root_dir=canonical_workspace_root_dir,
			refresh_on_run=bool(
				cache_sorting_outputs_before_merge_canonical_workspace_refresh_on_run
			),
		)
		try:
			should_refresh_canonical_workspace = bool(
				(not canonical_workspace_root_dir.exists())
				or cache_sorting_outputs_before_merge_canonical_workspace_refresh_on_run
			)
			if should_refresh_canonical_workspace:
				canonical_workspace_summary = _cache_canonical_sorter_output_for_merge(
					stage_output_root_dir=stage_output_root_dir,
					cache_root_dir=canonical_workspace_root_dir,
				)
			else:
				canonical_workspace_preserved_existing = True

			canonical_workspace_sorter_output_dir = (canonical_workspace_root_dir / "sorter_output").resolve()
			if not canonical_workspace_sorter_output_dir.exists():
				raise FileNotFoundError(
					"Merge working cache is missing sorter output: "
					+ str(canonical_workspace_sorter_output_dir)
				)

			if slay_requested:
				slay_binary_input_preflight = _preflight_slay_binary_input(
					sorter_output_dir=canonical_workspace_sorter_output_dir,
					well_out_dir=well_out_dir,
					stage_config=stage_config,
				)
				cache_outputs["slay.preflight_data_filepath"] = str(
					slay_binary_input_preflight.get("data_filepath", "")
				)

			cache_outputs["merge.working_cache_dir"] = str(canonical_workspace_root_dir.resolve())
			cache_outputs["merge.working_cache_sorter_output_dir"] = str(
				canonical_workspace_sorter_output_dir.resolve()
			)
			canonical_workspace_summary_json = (canonical_workspace_root_dir / "working_cache_summary.json").resolve()
			if canonical_workspace_summary_json.exists():
				cache_outputs["merge.working_cache_summary_json"] = str(canonical_workspace_summary_json)

			canonical_workspace_prepared = True
			active_stage_output_root_dir = canonical_workspace_root_dir
			active_sorter_output_dir = canonical_workspace_sorter_output_dir
		except Exception as exc:
			canonical_workspace_error = (
				f"working_cache_prepare_failed:{type(exc).__name__}:{exc}"
			)
			raise RuntimeError(canonical_workspace_error) from exc

	resolved_sorter_output_dir: Path | None = active_sorter_output_dir
	pre_snapshot_sorter_output_dir: Path | None = active_sorter_output_dir

	if resolved_sorter_output_dir is None:
		resolved_sorter_output_dir = _resolve_sorter_output_dir(
			well_out_dir=well_out_dir,
			output_rel_root=output_rel_root,
			stage_config=stage_config,
		)
	if slay_requested and slay_binary_input_preflight is None:
		slay_binary_input_preflight = _preflight_slay_binary_input(
			sorter_output_dir=resolved_sorter_output_dir,
			well_out_dir=well_out_dir,
			stage_config=stage_config,
		)
		cache_outputs["slay.preflight_data_filepath"] = str(
			slay_binary_input_preflight.get("data_filepath", "")
		)
	workspace_sorter_output_dir = resolved_sorter_output_dir

	pre_merge_workspace_relpath = str(
		getattr(stage_config, "pre_merge_workspace_relpath", "cache/merge_workspace")
		or "cache/merge_workspace"
	).strip().lstrip("/") or "cache/merge_workspace"
	pre_merge_workspace_dir = _resolve_under_spikesort_output_root(
		well_out_dir=well_out_dir,
		output_rel_root=merge_output_rel_root,
		relpath=pre_merge_workspace_relpath,
	)
	pre_merge_workspace_analyzer_output_dir = (
		pre_merge_workspace_dir / "pre_merge_analyzer_output"
	).resolve()
	pre_merge_workspace_analyzer_error: str | None = None
	pre_merge_workspace_analyzer_built = False
	pre_merge_workspace_analyzer_policy: dict[str, Any] | None = None
	pre_merge_workspace_analyzer: Any | None = None
	pre_snapshot_capture_needed = bool(
		(merge_metadata_enabled and merge_metadata_write_json)
		or (pre_merge_metadata_enabled and pre_merge_metadata_write_json)
		or bool(merge_reports_require_snapshots)
	)
	pre_snapshot_include_unit_locations = bool(
		((merge_metadata_enabled and merge_metadata_write_json) and merge_metadata_include_unit_locations)
		or ((pre_merge_metadata_enabled and pre_merge_metadata_write_json) and pre_merge_metadata_include_unit_locations)
		or bool(merge_reports_require_unit_locations)
	)
	pre_merge_runtime_analyzer_needed = bool(
		pre_snapshot_capture_needed
		and (pre_snapshot_include_unit_locations or pre_merge_workspace_requires_templates)
	)

	if pre_merge_runtime_analyzer_needed:
		_log_phase_step_start(
			"Merge pre-merge analyzer prepare step start",
			stream_id=str(stream_id),
			sorter_output_dir=workspace_sorter_output_dir,
			analyzer_output_dir=pre_merge_workspace_analyzer_output_dir,
			requires_templates=bool(pre_merge_workspace_requires_templates),
			requires_unit_locations=bool(pre_merge_workspace_requires_unit_locations),
		)
		try:
			si_module = _import_spikeinterface_full_module()
			pre_merge_workspace_analyzer, pre_merge_workspace_analyzer_output_dir = _recompute_sorting_analyzer_to_dir(
				si_module=si_module,
				well_out_dir=well_out_dir,
				sorter_output_dir=workspace_sorter_output_dir,
				stage_config=stage_config,
				analyzer_dir=pre_merge_workspace_analyzer_output_dir,
			)

			_ensure_merge_analyzer_extensions(
				analyzer=pre_merge_workspace_analyzer,
				stage_config=stage_config,
				include_unit_locations=pre_merge_workspace_requires_unit_locations,
				include_templates=pre_merge_workspace_requires_templates,
			)
			pre_merge_workspace_analyzer_policy = _get_merge_analyzer_policy_info(
				pre_merge_workspace_analyzer
			)

			pre_merge_workspace_analyzer_built = True
			_log_memory_usage(
				"Merge pre-merge analyzer memory snapshot",
				stream_id=str(stream_id),
				sorter_output_dir=workspace_sorter_output_dir,
				analyzer_output_dir=pre_merge_workspace_analyzer_output_dir,
			)
		except Exception as exc:
			pre_merge_workspace_analyzer_error = (
				f"pre_merge_workspace_analyzer_prepare_failed:{type(exc).__name__}:{exc}"
			)

	pre_merge_snapshot: dict[str, Any] | None = None
	post_merge_snapshot: dict[str, Any] | None = None
	pre_merge_metadata_payload: dict[str, Any] | None = None
	post_merge_metadata_payload: dict[str, Any] | None = None
	pre_merge_metadata_json: Path | None = None
	post_merge_metadata_json: Path | None = None
	pre_merge_metadata_error: str | None = None
	post_merge_metadata_error: str | None = None
	merge_metadata_payload: dict[str, Any] | None = None
	merge_metadata_json: Path | None = None
	merge_metadata_error: str | None = None
	if pre_snapshot_capture_needed:
		_log_phase_step_start(
			"Merge pre-merge snapshot step start",
			stream_id=str(stream_id),
			include_unit_locations=bool(pre_snapshot_include_unit_locations),
			analyzer_output_dir=pre_merge_workspace_analyzer_output_dir,
		)
		try:
			if pre_merge_runtime_analyzer_needed and not bool(
				pre_merge_workspace_analyzer_built
				and pre_merge_workspace_analyzer_output_dir.exists()
			):
				raise RuntimeError("pre_merge_workspace_analyzer_unavailable")
			pre_snapshot_kwargs: dict[str, Any] = {
				"well_out_dir": well_out_dir,
				"stage_output_root_dir": active_stage_output_root_dir,
				"output_rel_root": output_rel_root,
				"stage_config": stage_config,
				"capture_label": "before_merge",
				"include_unit_locations": bool(pre_snapshot_include_unit_locations),
				"allow_analyzer_recompute": False,
			}
			if pre_snapshot_sorter_output_dir is not None:
				pre_snapshot_kwargs["sorter_output_dir"] = pre_snapshot_sorter_output_dir
			if pre_merge_runtime_analyzer_needed:
				pre_snapshot_kwargs["analyzer_source_dir"] = pre_merge_workspace_analyzer_output_dir
				pre_snapshot_kwargs["analyzer_obj"] = pre_merge_workspace_analyzer
			pre_merge_snapshot_result = _call_capture_merge_state_snapshot_compat(**pre_snapshot_kwargs)
			if isinstance(pre_merge_snapshot_result, tuple):
				pre_merge_snapshot = pre_merge_snapshot_result[0]
			else:
				pre_merge_snapshot = pre_merge_snapshot_result
		except Exception as exc:
			err = f"before_merge_snapshot_failed:{type(exc).__name__}:{exc}"
			if merge_metadata_enabled and merge_metadata_write_json:
				merge_metadata_error = err
			if pre_merge_metadata_enabled and pre_merge_metadata_write_json:
				pre_merge_metadata_error = err

	if pre_merge_workspace_analyzer is not None:
		_log_phase_step_start(
			"Merge pre-merge analyzer release step start",
			stream_id=str(stream_id),
			analyzer_output_dir=pre_merge_workspace_analyzer_output_dir,
		)
		released_pre_merge_extensions = _release_loaded_analyzer_extensions(analyzer=pre_merge_workspace_analyzer)
		pre_merge_workspace_analyzer = None
		gc.collect()
		_log_memory_usage(
			"Merge pre-merge analyzer release memory snapshot",
			stream_id=str(stream_id),
			analyzer_output_dir=pre_merge_workspace_analyzer_output_dir,
			released_extensions=int(len(released_pre_merge_extensions)),
		)

	method_reports: list[dict[str, Any]] = []
	combined_outputs: dict[str, str] = {}
	combined_outputs.update(cache_outputs)
	combined_outputs["merge.pre_merge_workspace_dir"] = str(pre_merge_workspace_dir.resolve())
	combined_outputs["merge.pre_merge_workspace_sorter_output_dir"] = str(workspace_sorter_output_dir.resolve())
	if pre_merge_workspace_analyzer_built and pre_merge_workspace_analyzer_output_dir.exists():
		combined_outputs["merge.pre_merge_workspace_analyzer_output_dir"] = str(
			pre_merge_workspace_analyzer_output_dir.resolve()
		)
	primary_out_dir: Path = merge_phase_out_dir
	resolved_sorter_output_dir: Path | None = pre_snapshot_sorter_output_dir
	if isinstance(bombcell_report, dict):
		combined_outputs.update(dict(bombcell_report.get("outputs", {})))

	for idx, raw_method in enumerate(requested_sequence_raw):
		method = _normalize_merge_method_token(raw_method)
		_log_phase_step_start(
			"Merge method step start",
			stream_id=str(stream_id),
			method=method,
			sequence_index=int(idx + 1),
			sequence_length=int(len(requested_sequence_raw)),
		)
		if method == "slay":
			if (
				cache_sorting_outputs_before_merge_use_canonical_workspace
				and cache_sorting_outputs_before_merge_assert_slay_uses_canonical_workspace
			):
				_assert_method_uses_working_cache_sorter_output(
					method_name="SLAy",
					sorter_output_dir=resolved_sorter_output_dir,
					working_cache_root_dir=canonical_workspace_root_dir,
					knob_name="assert_selected_sorter_output",
				)
			slay_call_kwargs: dict[str, Any] = {
				"well_out_dir": well_out_dir,
				"stage_output_root_dir": active_stage_output_root_dir,
				"output_rel_root": output_rel_root,
				"stage_config": stage_config,
				"force_restart": bool(force_restart),
			}
			if resolved_sorter_output_dir is not None:
				slay_call_kwargs["sorter_output_dir"] = resolved_sorter_output_dir
			report = _run_slay_merge_method(**slay_call_kwargs)
			method_reports.append(report)
			combined_outputs.update(dict(report.get("outputs", {})))
			if report.get("ks_dir"):
				resolved_sorter_output_dir = Path(str(report.get("ks_dir"))).resolve()
				if (
					cache_sorting_outputs_before_merge_use_canonical_workspace
					and cache_sorting_outputs_before_merge_assert_slay_uses_canonical_workspace
				):
					_assert_method_uses_working_cache_sorter_output(
						method_name="SLAy",
						sorter_output_dir=resolved_sorter_output_dir,
						working_cache_root_dir=canonical_workspace_root_dir,
						knob_name="assert_selected_sorter_output",
					)

			should_recompute_after_slay = (
				bool(getattr(stage_config, "slay_recompute_analyzer", False))
				and bool(report.get("status") == "ok")
				and bool(getattr(stage_config, "slay_auto_accept_merges", False))
				and bool(report.get("applied_merges", False))
			)
			if should_recompute_after_slay and resolved_sorter_output_dir is not None:
				recompute_report = _run_slay_analyzer_recompute(
					well_out_dir=well_out_dir,
					stage_output_root_dir=active_stage_output_root_dir,
					stage_config=stage_config,
					sorter_output_dir=resolved_sorter_output_dir,
				)
				method_reports.append(recompute_report)
				combined_outputs.update(dict(recompute_report.get("outputs", {})))
				recomputed_analyzer_dir_raw = (
					recompute_report.get("outputs", {}).get("slay.recompute_analyzer.analyzer_dir", None)
					if isinstance(recompute_report.get("outputs", {}), dict)
					else None
				)
				if recomputed_analyzer_dir_raw is not None:
					combined_outputs["merge.post_merge_analyzer_output_dir"] = str(
						Path(str(recomputed_analyzer_dir_raw)).expanduser().resolve()
					)
			continue

		if method == "auto_merge":
			if (
				cache_sorting_outputs_before_merge_use_canonical_workspace
				and cache_sorting_outputs_before_merge_assert_auto_merge_uses_canonical_workspace
			):
				_assert_method_uses_working_cache_sorter_output(
					method_name="auto_merge",
					sorter_output_dir=resolved_sorter_output_dir,
					working_cache_root_dir=canonical_workspace_root_dir,
					knob_name="assert_selected_sorter_output",
				)
			report = _run_auto_merge_method(
				well_out_dir=well_out_dir,
				stage_output_root_dir=active_stage_output_root_dir,
				output_rel_root=output_rel_root,
				stage_config=stage_config,
				force_restart=bool(force_restart),
				sorter_output_dir=resolved_sorter_output_dir,
			)
			method_reports.append(report)
			combined_outputs.update(dict(report.get("outputs", {})))
			continue

		if method == "unitmatch":
			report = {
				"name": "unitmatch",
				"status": "skipped",
				"reason": "unitmatch_not_implemented_in_v2_merge_phase",
				"out_dir": str(active_stage_output_root_dir),
				"summary_json": None,
				"outputs": {},
			}
			method_reports.append(report)
			continue

		report = {
			"name": str(method),
			"status": "skipped",
			"reason": "unknown_merge_method",
			"out_dir": str(active_stage_output_root_dir),
			"summary_json": None,
			"outputs": {},
		}
		method_reports.append(report)

	primary_out_dir.mkdir(parents=True, exist_ok=True)
	summary_json = primary_out_dir / "merge_stage_summary.json"

	if pre_merge_metadata_enabled and pre_merge_metadata_write_json:
		if isinstance(pre_merge_snapshot, dict):
			try:
				pre_merge_metadata_payload = _build_snapshot_metadata_summary(
					snapshot_label="before_merge",
					snapshot=pre_merge_snapshot,
				)
				pre_merge_metadata_json = primary_out_dir / str(pre_merge_metadata_json_relpath)
				_write_json(pre_merge_metadata_json, pre_merge_metadata_payload)
				combined_outputs["merge.pre_metadata_summary_json"] = str(pre_merge_metadata_json)
			except Exception as exc:
				pre_merge_metadata_error = f"pre_merge_metadata_summary_failed:{type(exc).__name__}:{exc}"
		else:
			if pre_merge_metadata_error is None:
				pre_merge_metadata_error = "pre_merge_snapshot_unavailable"

	ok_reports = [r for r in method_reports if str(r.get("status")) == "ok"]
	if ok_reports:
		status = "ok"
		reason = None
	else:
		status = "skipped"
		reason = None
		for report in method_reports:
			report_reason = report.get("reason")
			if report_reason:
				reason = str(report_reason)
				break

	canonical_workspace_publish_requested = False
	canonical_workspace_publish_skip_reason: str | None = None
	if cache_sorting_outputs_before_merge_use_canonical_workspace and canonical_workspace_prepared:
		if str(status) == "ok":
			canonical_workspace_publish_requested = bool(
				cache_sorting_outputs_before_merge_publish_canonical_to_stage_outputs_on_success
			)
			if not canonical_workspace_publish_requested:
				canonical_workspace_publish_skip_reason = "publish_working_cache_to_canonical_on_success_disabled"
		else:
			canonical_workspace_publish_requested = bool(
				cache_sorting_outputs_before_merge_publish_canonical_to_stage_outputs_on_failure
			)
			if not canonical_workspace_publish_requested:
				canonical_workspace_publish_skip_reason = "publish_working_cache_to_canonical_on_failure_disabled"
	elif not cache_sorting_outputs_before_merge_use_canonical_workspace:
		canonical_workspace_publish_skip_reason = "working_cache_disabled"
	else:
		canonical_workspace_publish_skip_reason = "working_cache_not_prepared"

	_log_phase_step_start(
		"Merge working cache publish decision",
		stream_id=str(stream_id),
		merge_status=str(status),
		requested=bool(canonical_workspace_publish_requested),
		workspace_root_dir=active_stage_output_root_dir,
		stage_output_root_dir=stage_output_root_dir,
		reason=(None if canonical_workspace_publish_requested else canonical_workspace_publish_skip_reason),
	)

	if canonical_workspace_publish_requested:
		_log_phase_step_start(
			"Merge working cache publish step start",
			stream_id=str(stream_id),
			workspace_root_dir=active_stage_output_root_dir,
			stage_output_root_dir=stage_output_root_dir,
		)
		try:
			canonical_workspace_publish_summary = _publish_working_sorter_output_to_canonical(
				stage_output_root_dir=stage_output_root_dir,
				cache_root_dir=active_stage_output_root_dir,
			)
			canonical_workspace_published = True
			restored_paths = list(canonical_workspace_publish_summary.get("restored_paths", []) or [])
			missing_workspace_sources = list(
				canonical_workspace_publish_summary.get("missing_cache_sources", []) or []
			)
			published_sorter_output_dir = (stage_output_root_dir / "sorter_output").resolve()
			published_analyzer_output_dir = (stage_output_root_dir / "analyzer_output").resolve()
			if published_sorter_output_dir.exists():
				combined_outputs["merge.published_sorter_output_dir"] = str(published_sorter_output_dir)
			removed_published_analyzer_output_dir: str | None = None
			if published_analyzer_output_dir.exists():
				shutil.rmtree(published_analyzer_output_dir, ignore_errors=True)
				removed_published_analyzer_output_dir = str(published_analyzer_output_dir)
			_log_phase_step_start(
				"Merge working cache publish complete",
				stream_id=str(stream_id),
				workspace_root_dir=active_stage_output_root_dir,
				stage_output_root_dir=stage_output_root_dir,
				restored_paths=(";".join(restored_paths) if restored_paths else None),
				missing_workspace_sources=(
					";".join(missing_workspace_sources) if missing_workspace_sources else None
				),
				published_sorter_output_dir=(
					published_sorter_output_dir if published_sorter_output_dir.exists() else None
				),
				removed_stale_analyzer_output_dir=removed_published_analyzer_output_dir,
			)
		except Exception as exc:
			canonical_workspace_publish_error = (
				f"working_cache_publish_failed:{type(exc).__name__}:{exc}"
			)
			_log_phase_step_start(
				"Merge working cache publish failed",
				stream_id=str(stream_id),
				workspace_root_dir=active_stage_output_root_dir,
				stage_output_root_dir=stage_output_root_dir,
				error=canonical_workspace_publish_error,
			)
			status = "error"
			reason = "working_cache_publish_failed"

	cache_cleaned_up = False
	cache_cleanup_removed: list[str] = []
	post_merge_report_analyzer: Any | None = None
	if (
		bool(cache_sorting_outputs_before_merge)
		and bool(cache_sorting_outputs_before_merge_cleanup_on_success)
		and str(status) == "ok"
		and cache_root_dir.exists()
	):
		cache_cleanup_removed.append(str(cache_root_dir))
		shutil.rmtree(cache_root_dir, ignore_errors=True)
		cache_cleaned_up = True
		for key in [
			"merge.pre_merge_cache_dir",
			"merge.pre_merge_cache_sorter_output_dir",
			"merge.pre_merge_cache_analyzer_output_dir",
			"merge.pre_merge_cache_summary_json",
		]:
			combined_outputs.pop(key, None)

	post_snapshot_capture_needed = bool(
		(merge_metadata_enabled and merge_metadata_write_json)
		or (post_merge_metadata_enabled and post_merge_metadata_write_json)
		or bool(merge_reports_require_snapshots)
	)
	post_snapshot_include_unit_locations = bool(
		((merge_metadata_enabled and merge_metadata_write_json) and merge_metadata_include_unit_locations)
		or ((post_merge_metadata_enabled and post_merge_metadata_write_json) and post_merge_metadata_include_unit_locations)
		or bool(merge_reports_require_unit_locations)
	)
	post_merge_runtime_analyzer_needed = bool(
		post_snapshot_capture_needed
		and (post_snapshot_include_unit_locations or merge_reports_template_heatmaps_enabled)
	)
	if post_snapshot_capture_needed:
		_log_phase_step_start(
			"Merge post-merge snapshot step start",
			stream_id=str(stream_id),
			include_unit_locations=bool(post_snapshot_include_unit_locations),
			sorter_output_dir=resolved_sorter_output_dir,
		)
		try:
			post_snapshot_kwargs: dict[str, Any] = {
				"well_out_dir": well_out_dir,
				"stage_output_root_dir": active_stage_output_root_dir,
				"output_rel_root": output_rel_root,
				"stage_config": stage_config,
				"capture_label": "after_merge",
				"include_unit_locations": bool(post_snapshot_include_unit_locations),
				"allow_analyzer_recompute": bool(post_merge_runtime_analyzer_needed),
			}
			if resolved_sorter_output_dir is not None:
				post_snapshot_kwargs["sorter_output_dir"] = resolved_sorter_output_dir
			if merge_reports_template_heatmaps_enabled:
				post_snapshot_kwargs["return_analyzer_obj"] = True
			post_merge_snapshot_result = _call_capture_merge_state_snapshot_compat(**post_snapshot_kwargs)
			if isinstance(post_merge_snapshot_result, tuple):
				post_merge_snapshot, post_merge_report_analyzer = post_merge_snapshot_result
			else:
				post_merge_snapshot = post_merge_snapshot_result

			post_analyzer_raw = (
				post_merge_snapshot.get("analyzer", {})
				if isinstance(post_merge_snapshot, dict)
				else {}
			)
			if isinstance(post_analyzer_raw, dict):
				post_source_raw = post_analyzer_raw.get("source_dir", None)
				if post_source_raw is not None:
					post_source_dir = Path(str(post_source_raw)).expanduser().resolve()
					if post_source_dir.exists():
						combined_outputs["merge.post_merge_analyzer_output_dir"] = str(
							post_source_dir
						)
		except Exception as exc:
			err = f"after_merge_snapshot_failed:{type(exc).__name__}:{exc}"
			if merge_metadata_enabled and merge_metadata_write_json:
				merge_metadata_error = err
			if post_merge_metadata_enabled and post_merge_metadata_write_json:
				post_merge_metadata_error = err
		else:
			if post_merge_metadata_enabled and post_merge_metadata_write_json:
				try:
					post_merge_metadata_payload = _build_snapshot_metadata_summary(
						snapshot_label="after_merge",
						snapshot=post_merge_snapshot,
					)
					post_merge_metadata_json = primary_out_dir / str(post_merge_metadata_json_relpath)
					_write_json(post_merge_metadata_json, post_merge_metadata_payload)
					combined_outputs["merge.post_metadata_summary_json"] = str(post_merge_metadata_json)
				except Exception as exc:
					post_merge_metadata_error = (
						f"post_merge_metadata_summary_failed:{type(exc).__name__}:{exc}"
					)

			if merge_metadata_enabled and merge_metadata_write_json:
				try:
					applied_operations = _extract_applied_merge_operations(
						method_reports=method_reports,
						stage_config=stage_config,
					)
					merge_metadata_payload = _build_merge_metadata_summary(
						requested_sequence_raw=list(requested_sequence_raw),
						stage_config=stage_config,
						pre_snapshot=(pre_merge_snapshot or {"sorter": {}, "analyzer": {}}),
						post_snapshot=(post_merge_snapshot or {"sorter": {}, "analyzer": {}}),
						applied_operations=applied_operations,
					)
					merge_metadata_json = primary_out_dir / str(merge_metadata_json_relpath)
					_write_json(merge_metadata_json, merge_metadata_payload)
					combined_outputs["merge.metadata_summary_json"] = str(merge_metadata_json)
				except Exception as exc:
					merge_metadata_error = f"merge_metadata_summary_failed:{type(exc).__name__}:{exc}"

			_log_memory_usage(
				"Merge post-merge snapshot memory snapshot",
				stream_id=str(stream_id),
				sorter_output_dir=resolved_sorter_output_dir,
				analyzer_output_dir=combined_outputs.get("merge.post_merge_analyzer_output_dir", None),
				include_unit_locations=bool(post_snapshot_include_unit_locations),
			)

	merge_unit_diff_payload: dict[str, Any] | None = None
	merge_unit_diff_json: Path | None = None
	merge_unit_diff_error: str | None = None
	merge_unit_diff_map_payload: dict[str, Any] | None = None
	merge_unit_diff_map_json: Path | None = None
	merge_unit_diff_map_error: str | None = None
	merge_unit_diff_map_flat_payload: dict[str, Any] | None = None
	merge_unit_diff_map_flat_json: Path | None = None
	merge_unit_diff_map_flat_error: str | None = None
	post_merge_unit_locations_payload: dict[str, Any] | None = None
	post_merge_unit_locations_json: Path | None = None
	post_merge_unit_locations_error: str | None = None
	if merge_reports_mappings_enabled:
		_log_phase_step_start(
			"Merge mapping report build step start",
			stream_id=str(stream_id),
			unit_diff_json=bool(merge_reports_unit_diff_json_enabled),
			unit_diff_map=bool(merge_reports_unit_diff_map_enabled),
			unit_diff_map_flat=bool(merge_reports_unit_diff_map_flat_enabled),
		)
		try:
			applied_operations_for_unit_diff: list[dict[str, Any]] = []
			if isinstance(merge_metadata_payload, dict):
				applied_operations_raw = merge_metadata_payload.get("applied_merge_operations", None)
				if isinstance(applied_operations_raw, list):
					applied_operations_for_unit_diff = [
						dict(item)
						for item in applied_operations_raw
						if isinstance(item, dict)
					]
			else:
				applied_operations_for_unit_diff = _extract_applied_merge_operations(
					method_reports=method_reports,
					stage_config=stage_config,
				)

			merge_unit_diff_payload = _build_merge_unit_diff_report_payload(
				requested_sequence_raw=list(requested_sequence_raw),
				method_reports=method_reports,
				merge_metadata_payload=merge_metadata_payload,
				before_snapshot=(pre_merge_snapshot if isinstance(pre_merge_snapshot, dict) else {}),
				after_snapshot=(post_merge_snapshot if isinstance(post_merge_snapshot, dict) else {}),
				applied_operations=applied_operations_for_unit_diff,
			)
			merge_unit_diff_map_payload = _build_unit_diff_map_payload(
				unit_diff_payload=merge_unit_diff_payload,
			)
			merge_unit_diff_map_flat_payload = _build_unit_diff_map_flat_payload(
				unit_diff_map_payload=merge_unit_diff_map_payload,
				unit_diff_payload=merge_unit_diff_payload,
			)
			merge_unit_diff_payload["unit_diff_map"] = merge_unit_diff_map_payload
			merge_unit_diff_payload["unit_diff_map_flat"] = merge_unit_diff_map_flat_payload

			if merge_reports_enabled and merge_reports_unit_diff_json_enabled:
				merge_unit_diff_json = primary_out_dir / str(merge_reports_unit_diff_json_relpath)
				_write_json(merge_unit_diff_json, merge_unit_diff_payload)
				combined_outputs["merge.report.unit_diff_json"] = str(merge_unit_diff_json)

			if merge_reports_enabled and merge_reports_unit_diff_map_enabled:
				merge_unit_diff_map_json = primary_out_dir / str(merge_reports_unit_diff_map_relpath)
				_write_json(merge_unit_diff_map_json, merge_unit_diff_map_payload)
				combined_outputs["merge.report.unit_diff_map_json"] = str(merge_unit_diff_map_json)

			if merge_reports_enabled and merge_reports_unit_diff_map_flat_enabled:
				merge_unit_diff_map_flat_json = primary_out_dir / str(merge_reports_unit_diff_map_flat_relpath)
				_write_json(merge_unit_diff_map_flat_json, merge_unit_diff_map_flat_payload)
				combined_outputs["merge.report.unit_diff_map_flat_json"] = str(merge_unit_diff_map_flat_json)
		except Exception as exc:
			merge_unit_diff_error = f"merge_unit_diff_json_failed:{type(exc).__name__}:{exc}"

	if merge_reports_enabled and merge_reports_post_merge_unit_locations_enabled:
		if isinstance(post_merge_snapshot, dict):
			try:
				post_merge_unit_locations_payload = _build_post_merge_unit_locations_payload(
					post_snapshot=post_merge_snapshot,
				)
				post_merge_unit_locations_json = primary_out_dir / str(
					merge_reports_post_merge_unit_locations_relpath
				)
				_write_json(post_merge_unit_locations_json, post_merge_unit_locations_payload)
				combined_outputs["merge.report.post_merge_unit_locations_json"] = str(
					post_merge_unit_locations_json
				)
			except Exception as exc:
				post_merge_unit_locations_error = (
					f"post_merge_unit_locations_json_failed:{type(exc).__name__}:{exc}"
				)
		else:
			post_merge_unit_locations_error = "post_merge_snapshot_unavailable"

	merge_reports_payload: dict[str, Any] | None = None
	merge_reports_error: str | None = None
	merge_template_heatmaps_payload: dict[str, Any] | None = None
	merge_template_heatmaps_error: str | None = None
	if merge_reports_enabled and (merge_reports_2panel_enabled or merge_reports_template_heatmaps_enabled):
		_log_phase_step_start(
			"Merge report rendering step start",
			stream_id=str(stream_id),
			unit_location_reports=bool(merge_reports_2panel_enabled),
			template_heatmaps=bool(merge_reports_template_heatmaps_enabled),
		)
		try:
			before_snapshot_for_report: dict[str, Any] = {}
			after_snapshot_for_report: dict[str, Any] = {}
			applied_unit_mappings_for_report: list[dict[str, Any]] = []
			if merge_reports_mappings_enabled:
				if isinstance(merge_unit_diff_payload, dict):
					(
						before_snapshot_for_report,
						after_snapshot_for_report,
						applied_unit_mappings_for_report,
					) = _extract_plot_inputs_from_unit_diff_report(
						unit_diff_payload=merge_unit_diff_payload,
					)
				else:
					merge_reports_error = "merge_reports_missing_unit_diff_source"
			else:
				before_snapshot_for_report_raw: Any = None
				after_snapshot_for_report_raw: Any = None
				if isinstance(merge_metadata_payload, dict):
					before_snapshot_for_report_raw = merge_metadata_payload.get("before", None)
					after_snapshot_for_report_raw = merge_metadata_payload.get("after", None)
					applied_unit_mappings_for_report = _extract_applied_unit_mappings_for_report(
						merge_metadata_payload=merge_metadata_payload
					)
				if before_snapshot_for_report_raw is None:
					before_snapshot_for_report_raw = pre_merge_snapshot
				if after_snapshot_for_report_raw is None:
					after_snapshot_for_report_raw = post_merge_snapshot
				before_snapshot_for_report = (
					dict(before_snapshot_for_report_raw)
					if isinstance(before_snapshot_for_report_raw, dict)
					else {}
				)
				after_snapshot_for_report = (
					dict(after_snapshot_for_report_raw)
					if isinstance(after_snapshot_for_report_raw, dict)
					else {}
				)

			preferred_pre_analyzer_raw = combined_outputs.get(
				"merge.pre_merge_workspace_analyzer_output_dir",
				None,
			)
			if preferred_pre_analyzer_raw is None:
				merge_reports_error = "merge_reports_missing_pre_merge_workspace_analyzer_output_dir"
			else:
				preferred_pre_analyzer_dir = Path(str(preferred_pre_analyzer_raw)).expanduser().resolve()
				if not preferred_pre_analyzer_dir.exists():
					merge_reports_error = "merge_reports_missing_pre_merge_workspace_analyzer"
				else:
					before_analyzer_raw = before_snapshot_for_report.get("analyzer", {})
					before_analyzer = (
						dict(before_analyzer_raw)
						if isinstance(before_analyzer_raw, dict)
						else {}
					)
					before_analyzer["source_dir"] = str(preferred_pre_analyzer_dir)
					before_snapshot_for_report["analyzer"] = before_analyzer

			preferred_post_analyzer_raw = combined_outputs.get(
				"merge.post_merge_analyzer_output_dir",
				None,
			)
			if merge_reports_error is None and merge_reports_template_heatmaps_enabled:
				if preferred_post_analyzer_raw is None:
					merge_reports_error = "merge_reports_missing_post_merge_analyzer_output_dir"
				else:
					preferred_post_analyzer_dir = Path(str(preferred_post_analyzer_raw)).expanduser().resolve()
					if not preferred_post_analyzer_dir.exists():
						merge_reports_error = "merge_reports_missing_post_merge_analyzer"
					else:
						after_analyzer_raw = after_snapshot_for_report.get("analyzer", {})
						after_analyzer = (
							dict(after_analyzer_raw)
							if isinstance(after_analyzer_raw, dict)
							else {}
						)
						after_analyzer["source_dir"] = str(preferred_post_analyzer_dir)
						after_snapshot_for_report["analyzer"] = after_analyzer
			elif preferred_post_analyzer_raw is not None:
				preferred_post_analyzer_dir = Path(str(preferred_post_analyzer_raw)).expanduser().resolve()
				if preferred_post_analyzer_dir.exists():
					after_analyzer_raw = after_snapshot_for_report.get("analyzer", {})
					after_analyzer = (
						dict(after_analyzer_raw)
						if isinstance(after_analyzer_raw, dict)
						else {}
					)
					after_analyzer["source_dir"] = str(preferred_post_analyzer_dir)
					after_snapshot_for_report["analyzer"] = after_analyzer

			if merge_reports_error is None and (not before_snapshot_for_report or not after_snapshot_for_report):
				merge_reports_error = "merge_reports_missing_before_after_snapshots"
			if merge_reports_error is None and merge_reports_2panel_enabled:
				merge_reports_payload = _write_merge_unit_location_reports(
					merge_out_dir=primary_out_dir,
					before_snapshot=before_snapshot_for_report,
					after_snapshot=after_snapshot_for_report,
					applied_unit_mappings=applied_unit_mappings_for_report,
					stage_config=stage_config,
				)
				if str(merge_reports_payload.get("status", "")) == "ok":
					combined_outputs.update(dict(merge_reports_payload.get("outputs", {})))
				else:
					merge_reports_error = str(merge_reports_payload.get("error", "merge_reports_failed"))

			if merge_reports_error is None and merge_reports_template_heatmaps_enabled:
				merge_template_heatmaps_payload = _write_merge_template_heatmap_reports(
					merge_out_dir=primary_out_dir,
					before_snapshot=before_snapshot_for_report,
					after_snapshot=after_snapshot_for_report,
					applied_unit_mappings=applied_unit_mappings_for_report,
					stage_config=stage_config,
					after_analyzer=post_merge_report_analyzer,
				)
				if str(merge_template_heatmaps_payload.get("status", "")) == "ok":
					combined_outputs.update(dict(merge_template_heatmaps_payload.get("outputs", {})))
				elif str(merge_template_heatmaps_payload.get("status", "")) not in {"", "skipped"}:
					merge_template_heatmaps_error = str(
						merge_template_heatmaps_payload.get(
							"error",
							"merge_template_heatmaps_failed",
						)
					)
		except Exception as exc:
			merge_reports_error = f"merge_reports_failed:{type(exc).__name__}:{exc}"

		if merge_reports_enabled and (merge_reports_2panel_enabled or merge_reports_template_heatmaps_enabled):
			_log_memory_usage(
				"Merge report rendering memory snapshot",
				stream_id=str(stream_id),
				merge_out_dir=primary_out_dir,
				unit_location_reports=bool(merge_reports_2panel_enabled),
				template_heatmaps=bool(merge_reports_template_heatmaps_enabled),
			)

	if post_merge_report_analyzer is not None:
		_log_phase_step_start(
			"Merge post-merge analyzer release step start",
			stream_id=str(stream_id),
			analyzer_output_dir=combined_outputs.get("merge.post_merge_analyzer_output_dir", None),
		)
		released_post_merge_extensions = _release_loaded_analyzer_extensions(analyzer=post_merge_report_analyzer)
		post_merge_report_analyzer = None
		gc.collect()
		_log_memory_usage(
			"Merge post-merge analyzer release memory snapshot",
			stream_id=str(stream_id),
			analyzer_output_dir=combined_outputs.get("merge.post_merge_analyzer_output_dir", None),
			released_extensions=int(len(released_post_merge_extensions)),
		)

	generated_analyzer_cleanup_enabled = bool(
		getattr(stage_config, "merge_cleanup_generated_analyzers_on_success", True)
	)
	generated_analyzer_cleanup_removed: list[str] = []
	if str(status) == "ok" and generated_analyzer_cleanup_enabled:
		generated_analyzer_candidates: list[Path] = []
		if pre_merge_workspace_analyzer_built:
			generated_analyzer_candidates.append(pre_merge_workspace_analyzer_output_dir)
		if post_merge_runtime_analyzer_needed:
			generated_analyzer_candidates.append((active_stage_output_root_dir / "analyzer_output").resolve())
		if bool(canonical_workspace_published) or not cache_sorting_outputs_before_merge_use_canonical_workspace:
			generated_analyzer_candidates.append((stage_output_root_dir / "analyzer_output").resolve())

		seen_cleanup_targets: set[Path] = set()
		for analyzer_dir in generated_analyzer_candidates:
			resolved_analyzer_dir = Path(analyzer_dir).expanduser().resolve()
			if resolved_analyzer_dir in seen_cleanup_targets or not resolved_analyzer_dir.exists():
				continue
			seen_cleanup_targets.add(resolved_analyzer_dir)
			shutil.rmtree(resolved_analyzer_dir, ignore_errors=True)
			generated_analyzer_cleanup_removed.append(str(resolved_analyzer_dir))

		if generated_analyzer_cleanup_removed:
			for key, value in list(combined_outputs.items()):
				if key.endswith("analyzer_output_dir") and str(value) in set(generated_analyzer_cleanup_removed):
					combined_outputs.pop(key, None)
			_log_phase_step_start(
				"Merge generated analyzer cleanup complete",
				stream_id=str(stream_id),
				removed_paths=";".join(generated_analyzer_cleanup_removed),
			)

	payload: dict[str, Any] = {
		"status": str(status),
		"reason": reason,
		"well_out_dir": str(well_out_dir),
		"stage_output_root_dir": str(stage_output_root_dir),
		"active_stage_output_root_dir": str(active_stage_output_root_dir),
		"merge_out_dir": str(primary_out_dir),
		"merge_rel_output_root": (str(merge_rel_output_root) if merge_rel_output_root is not None else None),
		"merge_output_rel_root": str(merge_output_rel_root),
		"merge_delete_outputs_on_force_restart": bool(merge_delete_outputs_on_force_restart),
		"merge_removed_on_force_restart": list(merge_phase_removed_on_force_restart),
		"force_restart": bool(force_restart),
		"force_replot": bool(force_replot),
		"replot_only": False,
		"merge_units_enabled": bool(merge_units_enabled),
		"cache_sorting_outputs_before_merge": bool(cache_sorting_outputs_before_merge),
		"merge_reports_enabled": bool(merge_reports_any_enabled),
		"slay_model_cache_path": (str(slay_model_cache_path) if slay_model_cache_path is not None else None),
		"slay_model_cache_use_cached_model": bool(
			getattr(stage_config, "slay_model_cache_use_cached_model", True)
		),
		"slay_model_cache_write_model": bool(
			getattr(stage_config, "slay_model_cache_write_model", True)
		),
		"slay_force_restart_retrain_model": bool(getattr(stage_config, "slay_force_restart_retrain_model", False)),
		"cache_sorting_outputs_before_merge_config": {
			"enabled": bool(cache_sorting_outputs_before_merge),
			"relpath": str(cache_sorting_outputs_before_merge_relpath),
			"cleanup_on_success": bool(cache_sorting_outputs_before_merge_cleanup_on_success),
			"cleanup_generated_analyzers_on_success": bool(generated_analyzer_cleanup_enabled),
			"replace_sorting_with_cache_before_force_restart": bool(
				cache_sorting_outputs_before_merge_replace_sorting_with_cache_before_force_restart
			),
			"use_cache_on_force_restart": bool(cache_sorting_outputs_before_merge_use_cache_on_force_restart),
			"refresh_on_run": bool(cache_sorting_outputs_before_merge_refresh_on_run),
			"strict_restore_on_force_restart": bool(
				cache_sorting_outputs_before_merge_strict_restore_on_force_restart
			),
			"use_working_cache": bool(cache_sorting_outputs_before_merge_use_canonical_workspace),
			"working_cache_relpath": str(cache_sorting_outputs_before_merge_canonical_workspace_relpath),
			"working_cache_refresh_on_run": bool(
				cache_sorting_outputs_before_merge_canonical_workspace_refresh_on_run
			),
			"publish_working_cache_to_canonical_on_success": bool(
				cache_sorting_outputs_before_merge_publish_canonical_to_stage_outputs_on_success
			),
			"publish_working_cache_to_canonical_on_failure": bool(
				cache_sorting_outputs_before_merge_publish_canonical_to_stage_outputs_on_failure
			),
			"use_canonical_workspace": bool(cache_sorting_outputs_before_merge_use_canonical_workspace),
			"canonical_workspace_relpath": str(cache_sorting_outputs_before_merge_canonical_workspace_relpath),
			"canonical_workspace_refresh_on_run": bool(
				cache_sorting_outputs_before_merge_canonical_workspace_refresh_on_run
			),
			"canonical_workspace_rebuild_analyzer": bool(
				cache_sorting_outputs_before_merge_canonical_workspace_rebuild_analyzer
			),
			"publish_canonical_to_stage_outputs_on_success": bool(
				cache_sorting_outputs_before_merge_publish_canonical_to_stage_outputs_on_success
			),
			"publish_canonical_to_stage_outputs_on_failure": bool(
				cache_sorting_outputs_before_merge_publish_canonical_to_stage_outputs_on_failure
			),
			"assert_slay_uses_canonical_workspace": bool(
				cache_sorting_outputs_before_merge_assert_slay_uses_canonical_workspace
			),
			"assert_auto_merge_uses_canonical_workspace": bool(
				cache_sorting_outputs_before_merge_assert_auto_merge_uses_canonical_workspace
			),
			"restored_from_existing_cache": bool(cache_restored_from_existing),
			"preserved_existing_cache": bool(cache_preserved_existing),
			"reseeded_after_restore_miss": bool(cache_reseeded_after_restore_miss),
			"restore_precheck_missing_sources": list(cache_restore_precheck_missing_sources),
			"cleaned_up": bool(cache_cleaned_up),
			"canonical_workspace_prepared": bool(canonical_workspace_prepared),
			"canonical_workspace_preserved_existing": bool(canonical_workspace_preserved_existing),
			"canonical_workspace_published": bool(canonical_workspace_published),
			"canonical_workspace_publish_requested": bool(canonical_workspace_publish_requested),
		},
		"pre_merge_workspace": {
			"relpath": str(pre_merge_workspace_relpath),
			"workspace_dir": str(pre_merge_workspace_dir.resolve()),
			"sorter_output_dir": str(workspace_sorter_output_dir.resolve()),
			"analyzer_output_dir": (
				str(pre_merge_workspace_analyzer_output_dir.resolve())
				if pre_merge_workspace_analyzer_built
				else None
			),
			"analyzer_built": bool(pre_merge_workspace_analyzer_built),
			"analyzer_needed": bool(pre_merge_runtime_analyzer_needed),
			"analyzer_policy": (
				dict(pre_merge_workspace_analyzer_policy)
				if isinstance(pre_merge_workspace_analyzer_policy, dict)
				else _requested_merge_analyzer_policy(stage_config)
			),
		},
		"merge_analyzer_policy": _requested_merge_analyzer_policy(stage_config),
		"requested_sequence": [str(token) for token in requested_sequence_raw],
		"bombcell_label_config": {
			"enabled": bool(getattr(stage_config, "bombcell_label_enabled", False)),
			"relpath": str(getattr(stage_config, "bombcell_label_relpath", "bombcell_label_outputs")),
			"delete_outputs_on_force_restart": bool(
				getattr(stage_config, "bombcell_label_delete_outputs_on_force_restart", True)
			),
			"label_non_somatic": bool(getattr(stage_config, "bombcell_label_label_non_somatic", True)),
			"split_non_somatic_good_mua": bool(
				getattr(stage_config, "bombcell_label_split_non_somatic_good_mua", True)
			),
			"apply_to_sorter_output": bool(getattr(stage_config, "bombcell_label_apply_to_sorter_output", True)),
			"write_cluster_group": bool(getattr(stage_config, "bombcell_label_write_cluster_group", True)),
			"fail_on_error": bool(getattr(stage_config, "bombcell_label_fail_on_error", False)),
		},
		"bombcell_label": (
			{
				"status": str(bombcell_report.get("status", "skipped")),
				"reason": bombcell_report.get("reason", None),
				"out_dir": bombcell_report.get("out_dir", None),
				"summary_json": bombcell_report.get("summary_json", None),
				"n_units_labeled": int(bombcell_report.get("n_units_labeled", 0) or 0),
				"counts_by_label": dict(bombcell_report.get("counts_by_label", {}) or {}),
			}
			if isinstance(bombcell_report, dict)
			else {
				"status": "skipped",
				"reason": "bombcell_label_not_invoked_by_merge_stage",
			}
		),
		"methods": method_reports,
		"pre_merge_metadata_config": {
			"enabled": bool(pre_merge_metadata_enabled),
			"write_json": bool(pre_merge_metadata_write_json),
			"json_relpath": str(pre_merge_metadata_json_relpath),
			"include_unit_locations": bool(pre_merge_metadata_include_unit_locations),
			"log_summary_details": bool(pre_merge_metadata_log_summary_details),
		},
		"post_merge_metadata_config": {
			"enabled": bool(post_merge_metadata_enabled),
			"write_json": bool(post_merge_metadata_write_json),
			"json_relpath": str(post_merge_metadata_json_relpath),
			"include_unit_locations": bool(post_merge_metadata_include_unit_locations),
			"log_summary_details": bool(post_merge_metadata_log_summary_details),
		},
		"pre_merge_metadata_enabled": bool(pre_merge_metadata_enabled and pre_merge_metadata_write_json),
		"post_merge_metadata_enabled": bool(post_merge_metadata_enabled and post_merge_metadata_write_json),
		"merge_metadata_enabled": bool(merge_metadata_enabled and merge_metadata_write_json),
		"outputs": combined_outputs,
	}
	if isinstance(cache_summary, dict):
		payload["pre_merge_cache"] = {
			"summary_json": str(cache_summary.get("summary_json", "")),
			"copied_paths": list(cache_summary.get("copied_paths", []) or []),
			"missing_sources": list(cache_summary.get("missing_sources", []) or []),
		}
	if isinstance(canonical_workspace_summary, dict):
		payload["working_cache"] = {
			"summary_json": str(canonical_workspace_summary.get("summary_json", "")),
			"copied_paths": list(canonical_workspace_summary.get("copied_paths", []) or []),
			"missing_sources": list(canonical_workspace_summary.get("missing_sources", []) or []),
			"preserved_existing": bool(canonical_workspace_preserved_existing),
			"workspace_root_dir": str(canonical_workspace_root_dir),
			"sorter_output_dir": (
				str(canonical_workspace_sorter_output_dir)
				if canonical_workspace_sorter_output_dir is not None
				else None
			),
		}
	elif cache_sorting_outputs_before_merge_use_canonical_workspace:
		payload["working_cache"] = {
			"preserved_existing": bool(canonical_workspace_preserved_existing),
			"workspace_root_dir": str(canonical_workspace_root_dir),
			"sorter_output_dir": (
				str(canonical_workspace_sorter_output_dir)
				if canonical_workspace_sorter_output_dir is not None
				else None
			),
		}
	if isinstance(cache_restore_summary, dict):
		payload["pre_merge_cache_restore"] = {
			"restored_paths": list(cache_restore_summary.get("restored_paths", []) or []),
			"missing_cache_sources": list(cache_restore_summary.get("missing_cache_sources", []) or []),
		}
	if isinstance(canonical_workspace_publish_summary, dict):
		payload["working_cache_publish"] = {
			"requested": bool(canonical_workspace_publish_requested),
			"published": bool(canonical_workspace_published),
			"restored_paths": list(canonical_workspace_publish_summary.get("restored_paths", []) or []),
			"missing_workspace_sources": list(
				canonical_workspace_publish_summary.get("missing_cache_sources", []) or []
			),
		}
	if cache_cleanup_removed:
		payload["pre_merge_cache_cleanup"] = {
			"removed_on_success": list(cache_cleanup_removed),
		}
	if generated_analyzer_cleanup_removed:
		payload["generated_analyzer_cleanup"] = {
			"enabled": bool(generated_analyzer_cleanup_enabled),
			"removed_on_success": list(generated_analyzer_cleanup_removed),
		}
	if cache_error is not None:
		payload["pre_merge_cache_error"] = str(cache_error)
	if pre_merge_workspace_analyzer_error is not None:
		payload["pre_merge_workspace_analyzer_error"] = str(pre_merge_workspace_analyzer_error)
	if canonical_workspace_error is not None:
		payload["working_cache_error"] = str(canonical_workspace_error)
	if canonical_workspace_publish_error is not None:
		payload["working_cache_publish_error"] = str(canonical_workspace_publish_error)
	if bombcell_report_error is not None:
		payload["bombcell_label_error"] = str(bombcell_report_error)
	if pre_merge_metadata_json is not None:
		payload["pre_merge_metadata_summary_json"] = str(pre_merge_metadata_json)
	if isinstance(pre_merge_metadata_payload, dict):
		payload["pre_merge_metadata"] = {
			"snapshot_label": str(pre_merge_metadata_payload.get("snapshot_label", "before_merge")),
			"summary": pre_merge_metadata_payload.get("summary", {}),
		}
	if pre_merge_metadata_error is not None:
		payload["pre_merge_metadata_error"] = str(pre_merge_metadata_error)
	if post_merge_metadata_json is not None:
		payload["post_merge_metadata_summary_json"] = str(post_merge_metadata_json)
	if isinstance(post_merge_metadata_payload, dict):
		payload["post_merge_metadata"] = {
			"snapshot_label": str(post_merge_metadata_payload.get("snapshot_label", "after_merge")),
			"summary": post_merge_metadata_payload.get("summary", {}),
		}
	if post_merge_metadata_error is not None:
		payload["post_merge_metadata_error"] = str(post_merge_metadata_error)
	if merge_metadata_json is not None:
		payload["merge_metadata_summary_json"] = str(merge_metadata_json)
	if isinstance(merge_metadata_payload, dict):
		payload["merge_metadata"] = {
			"applied_merge_group_count": int(merge_metadata_payload.get("applied_merge_group_count", 0) or 0),
			"change_validation": merge_metadata_payload.get("change_validation", {}),
		}
	if merge_metadata_error is not None:
		payload["merge_metadata_error"] = str(merge_metadata_error)
	if merge_unit_diff_json is not None:
		payload["merge_unit_diff_json"] = str(merge_unit_diff_json)
	if isinstance(merge_unit_diff_payload, dict):
		payload["merge_unit_diff"] = {
			"applied_merge_group_count": int(merge_unit_diff_payload.get("applied_merge_group_count", 0) or 0),
			"change_validation": merge_unit_diff_payload.get("change_validation", {}),
		}
	if merge_unit_diff_map_json is not None:
		payload["merge_unit_diff_map_json"] = str(merge_unit_diff_map_json)
	if isinstance(merge_unit_diff_map_payload, dict):
		payload["merge_unit_diff_map"] = {
			"summary": dict(merge_unit_diff_map_payload.get("summary", {})),
		}
	if merge_unit_diff_map_flat_json is not None:
		payload["merge_unit_diff_map_flat_json"] = str(merge_unit_diff_map_flat_json)
	if isinstance(merge_unit_diff_map_flat_payload, dict):
		payload["merge_unit_diff_map_flat"] = {
			"summary": dict(merge_unit_diff_map_flat_payload.get("summary", {})),
		}
	if merge_unit_diff_error is not None:
		payload["merge_unit_diff_error"] = str(merge_unit_diff_error)
	if merge_unit_diff_map_error is not None:
		payload["merge_unit_diff_map_error"] = str(merge_unit_diff_map_error)
	if merge_unit_diff_map_flat_error is not None:
		payload["merge_unit_diff_map_flat_error"] = str(merge_unit_diff_map_flat_error)
	if post_merge_unit_locations_json is not None:
		payload["post_merge_unit_locations_json"] = str(post_merge_unit_locations_json)
	if isinstance(post_merge_unit_locations_payload, dict):
		payload["post_merge_unit_locations"] = {
			"summary": dict(post_merge_unit_locations_payload.get("summary", {})),
		}
	if post_merge_unit_locations_error is not None:
		payload["post_merge_unit_locations_error"] = str(post_merge_unit_locations_error)
	if isinstance(merge_reports_payload, dict):
		payload["merge_reports"] = {
			"status": str(merge_reports_payload.get("status", "ok")),
			"before_unit_locations_count": int(merge_reports_payload.get("before_unit_locations_count", 0) or 0),
			"after_unit_locations_count": int(merge_reports_payload.get("after_unit_locations_count", 0) or 0),
			"before_highlighted_units_count": int(merge_reports_payload.get("before_highlighted_units_count", 0) or 0),
			"after_highlighted_units_count": int(merge_reports_payload.get("after_highlighted_units_count", 0) or 0),
			"after_highlighted_inferred_units_count": int(
				merge_reports_payload.get("after_highlighted_inferred_units_count", 0) or 0
			),
		}
	if merge_reports_error is not None:
		payload["merge_reports_error"] = str(merge_reports_error)
	if isinstance(merge_template_heatmaps_payload, dict):
		payload["merge_template_heatmaps"] = {
			"status": str(merge_template_heatmaps_payload.get("status", "ok")),
			"n_mappings_requested": int(
				merge_template_heatmaps_payload.get("n_mappings_requested", 0) or 0
			),
			"n_mappings_processed": int(
				merge_template_heatmaps_payload.get("n_mappings_processed", 0) or 0
			),
			"debug_json": merge_template_heatmaps_payload.get("debug_json", None),
		}
	if merge_template_heatmaps_error is not None:
		payload["merge_template_heatmaps_error"] = str(merge_template_heatmaps_error)

	slay_ok = next((report for report in method_reports if report.get("name") == "slay" and report.get("status") == "ok"), None)
	if isinstance(slay_ok, dict):
		payload["n_merge_groups"] = int(slay_ok.get("n_merge_groups", 0) or 0)
		payload["n_candidate_pairs"] = int(slay_ok.get("n_candidate_pairs", 0) or 0)

	auto_merge_ok = next((report for report in method_reports if report.get("name") == "auto_merge" and report.get("status") == "ok"), None)
	if isinstance(auto_merge_ok, dict):
		payload["auto_merge_n_candidate_groups_total"] = int(auto_merge_ok.get("n_candidate_groups_total", 0) or 0)
		payload["auto_merge_n_candidate_pairs_total"] = int(auto_merge_ok.get("n_candidate_pairs_total", 0) or 0)
		payload["auto_merge_n_applied_groups_total"] = int(auto_merge_ok.get("n_applied_groups_total", 0) or 0)
		payload["auto_merge_n_iterations"] = int(auto_merge_ok.get("n_iterations", 0) or 0)

	combined_outputs["summary_json"] = str(summary_json)
	_write_json(summary_json, payload)

	if merge_metadata_log_summary_details:
		_log_merge_summary_details(
			stream_id=str(stream_id),
			status=str(status),
			method_reports=method_reports,
			summary_json=summary_json,
			merge_metadata_enabled=bool(merge_metadata_enabled and merge_metadata_write_json),
			merge_metadata_json=merge_metadata_json,
			merge_metadata_payload=merge_metadata_payload,
			merge_metadata_error=merge_metadata_error,
		)

	return SpikesortMergeResult(
		well_out_dir=well_out_dir,
		merge_out_dir=primary_out_dir,
		summary_json=summary_json,
		outputs=combined_outputs,
	)


def run_spikesort_stage(inputs: SpikesortInputs) -> SpikesortResult:
	well_out_dir = compute_mea_analysis_output_dir(
		output_root=inputs.mea_output_root,
		data_file=inputs.h5_path,
		well=inputs.stream_id,
	)

	stage_output_root_dir = _resolve_under_well(
		well_out_dir=well_out_dir,
		relpath=(str(inputs.output_rel_root).strip() or "spikesort_outputs"),
	)
	stage_output_root_dir.mkdir(parents=True, exist_ok=True)
	summary_json = stage_output_root_dir / "spikesort_summary.json"
	effective_force_restart = bool(inputs.force_restart or inputs.force_replot)
	_log_phase_step_start(
		"Spikesort stage start",
		stream_id=str(inputs.stream_id),
		sorter=str(inputs.sorter),
		force_restart=bool(inputs.force_restart),
		force_replot=bool(inputs.force_replot),
		output_root=str(inputs.output_rel_root),
	)

	if not bool(inputs.sort_enabled):
		_write_json(
			summary_json,
			{
				"status": "skipped",
				"reason": "sort_disabled",
				"h5_path": str(inputs.h5_path),
				"stream_id": str(inputs.stream_id),
				"well_out_dir": str(well_out_dir),
				"spikesort_out_dir": str(stage_output_root_dir),
				"output_rel_root": str(inputs.output_rel_root),
				"inputs": {
					"sort_enabled": bool(inputs.sort_enabled),
					"sort_delete_outputs_on_force_restart": bool(inputs.sort_delete_outputs_on_force_restart),
					"force_restart": bool(inputs.force_restart),
					"force_replot": bool(inputs.force_replot),
				},
				"outputs": {
					"summary_json": str(summary_json),
				},
			},
		)
		return SpikesortResult(
			well_out_dir=well_out_dir,
			spikesort_out_dir=stage_output_root_dir,
			summary_json=summary_json,
			outputs={
				"summary_json": str(summary_json),
			},
		)

	removed_on_force_restart: list[str] = []
	if bool(effective_force_restart) and bool(inputs.sort_delete_outputs_on_force_restart):
		_log_phase_step_start(
			"Spikesort cleanup step start",
			stream_id=str(inputs.stream_id),
			stage_output_root=stage_output_root_dir,
		)
		removed_on_force_restart = _cleanup_spikesort_outputs_for_force_restart(
			stage_output_root_dir=stage_output_root_dir,
			um_kwargs=inputs.um_kwargs,
		)

	legacy_inputs = LegacySpikeSortingInputs(
		h5_path=inputs.h5_path,
		stream_id=inputs.stream_id,
		mea_output_root=inputs.mea_output_root,
		output_subdir_after_well=(str(inputs.output_rel_root).strip() or "spikesort_outputs"),
		preprocess_concat_recording_relpath=inputs.preprocess_concat_recording_relpath,
		sort_original_preprocess_concat_recording_relpath=inputs.sort_original_preprocess_concat_recording_relpath,
		sort_bootstrapped_concat_recording_relpath=inputs.sort_bootstrapped_concat_recording_relpath,
		sort_use_bootstrapped_concat_binary=bool(inputs.sort_use_bootstrapped_concat_binary),
		sort_use_lazy_source=bool(inputs.sort_use_lazy_source),
		sort_assert_one_source=bool(inputs.sort_assert_one_source),
		log_enabled=bool(inputs.logging_enabled),
		log_verbose=bool(inputs.logging_verbose),
		log_file_override=inputs.logging_file_relpath,
		sorter=inputs.sorter,
		docker_image=inputs.docker_image,
		recording_num=inputs.recording_num,
		verbose=inputs.verbose,
		ks_batch_duration_s=inputs.ks_batch_duration_s,
		ks_batch_size=inputs.ks_batch_size,
		ks_th_universal=inputs.ks_th_universal,
		ks_th_learned=inputs.ks_th_learned,
		ks_th_single_ch=inputs.ks_th_single_ch,
		ks_cluster_downsampling=inputs.ks_cluster_downsampling,
		ks_nearest_chans=inputs.ks_nearest_chans,
		ks_max_channel_distance=inputs.ks_max_channel_distance,
		n_jobs=inputs.n_jobs,
		chunk_duration=inputs.chunk_duration,
		cuda_visible_devices=inputs.cuda_visible_devices,
		run_analyzer=inputs.run_analyzer,
		run_reports=inputs.run_reports,
		plot_mode=inputs.plot_mode,
		plot_debug=inputs.plot_debug,
		raster_sort=inputs.raster_sort,
		fixed_y=inputs.fixed_y,
		no_curation=inputs.no_curation,
		export_to_phy=inputs.export_to_phy,
		force_rerun_analyzer=inputs.force_rerun_analyzer,
		um_kwargs=(dict(inputs.um_kwargs) if isinstance(inputs.um_kwargs, dict) else None),
		am_kwargs=(dict(inputs.am_kwargs) if isinstance(inputs.am_kwargs, dict) else None),
		option_kwargs=(dict(inputs.option_kwargs) if isinstance(inputs.option_kwargs, dict) else None),
		force_restart=bool(effective_force_restart),
		resume_from=inputs.resume_from,
	)
	_log_phase_step_start(
		"Spikesort sort phase start",
		stream_id=str(inputs.stream_id),
		preprocess_concat_recording_relpath=inputs.preprocess_concat_recording_relpath,
		use_bootstrapped_concat_binary=bool(inputs.sort_use_bootstrapped_concat_binary),
		use_lazy_source=bool(inputs.sort_use_lazy_source),
		assert_one_source=bool(inputs.sort_assert_one_source),
		run_analyzer=bool(inputs.run_analyzer),
		run_reports=bool(inputs.run_reports),
		plot_mode=str(inputs.plot_mode),
		resume_from=inputs.resume_from,
	)
	legacy_outputs = run_legacy_spikesorting_stage(inputs=legacy_inputs, logger=LOGGER)

	legacy_out_dir = Path(legacy_outputs.output_dir)
	spikesort_out_dir = legacy_out_dir
	summary_json = spikesort_out_dir / "spikesort_summary.json"

	outputs: dict[str, str] = {
		"legacy.spikesort_out_dir": str(legacy_out_dir),
		"recording_dir": str(legacy_outputs.recording_dir),
		"sorter_output_dir": str(legacy_outputs.sorter_output_dir),
		"analyzer_dir": str(legacy_outputs.analyzer_dir),
	}
	summarize_sort_report: dict[str, Any] = {
		"status": "skipped",
		"reason": "summarize_sort_disabled",
	}
	if bool(inputs.summarize_sort_enabled):
		summarize_sort_report = _run_summarize_sort_phase(
			stream_id=str(inputs.stream_id),
			stage_output_root_dir=spikesort_out_dir,
			sorter_name=str(inputs.sorter),
			emit_logs=bool(inputs.summarize_sort_emit_logs),
			generate_artifacts=bool(inputs.summarize_sort_generate_artifacts),
		)
		outputs.update(dict(summarize_sort_report.get("outputs", {}) or {}))
	if legacy_outputs.merged_sorting_dir is not None:
		outputs["merged_sorting_dir"] = str(legacy_outputs.merged_sorting_dir)
	if legacy_outputs.merged_sorter_output_dir is not None:
		outputs["merged_sorter_output_dir"] = str(legacy_outputs.merged_sorter_output_dir)

	_write_json(
		summary_json,
		{
			"h5_path": str(inputs.h5_path),
			"stream_id": str(inputs.stream_id),
			"well_out_dir": str(well_out_dir),
			"spikesort_out_dir": str(spikesort_out_dir),
			"legacy_spikesort_out_dir": str(legacy_out_dir),
			"output_rel_root": str(inputs.output_rel_root),
			"inputs": {
				"preprocess_concat_recording_relpath": inputs.preprocess_concat_recording_relpath,
				"sort_original_preprocess_concat_recording_relpath": inputs.sort_original_preprocess_concat_recording_relpath,
				"sort_bootstrapped_concat_recording_relpath": inputs.sort_bootstrapped_concat_recording_relpath,
				"sort_use_bootstrapped_concat_binary": bool(inputs.sort_use_bootstrapped_concat_binary),
				"sort_use_lazy_source": bool(inputs.sort_use_lazy_source),
				"sort_assert_one_source": bool(inputs.sort_assert_one_source),
				"logging_enabled": bool(inputs.logging_enabled),
				"logging_verbose": bool(inputs.logging_verbose),
				"logging_file_relpath": inputs.logging_file_relpath,
				"sorter": str(inputs.sorter),
				"docker_image": inputs.docker_image,
				"recording_num": str(inputs.recording_num),
				"verbose": bool(inputs.verbose),
				"n_jobs": inputs.n_jobs,
				"chunk_duration": inputs.chunk_duration,
				"cuda_visible_devices": inputs.cuda_visible_devices,
				"run_analyzer": bool(inputs.run_analyzer),
				"run_reports": bool(inputs.run_reports),
				"sort_enabled": bool(inputs.sort_enabled),
				"sort_delete_outputs_on_force_restart": bool(inputs.sort_delete_outputs_on_force_restart),
				"plot_enabled": bool(inputs.plot_enabled),
				"plot_mode": str(inputs.plot_mode),
				"plot_debug": bool(inputs.plot_debug),
				"raster_sort": inputs.raster_sort,
				"fixed_y": bool(inputs.fixed_y),
				"no_curation": bool(inputs.no_curation),
				"export_to_phy": bool(inputs.export_to_phy),
				"summarize_sort_enabled": bool(inputs.summarize_sort_enabled),
				"summarize_sort_emit_logs": bool(inputs.summarize_sort_emit_logs),
				"summarize_sort_generate_artifacts": bool(inputs.summarize_sort_generate_artifacts),
				"force_restart": bool(inputs.force_restart),
				"force_replot": bool(inputs.force_replot),
				"effective_force_restart": bool(effective_force_restart),
				"resume_from": inputs.resume_from,
				"merge_analyzer_compute_sparsity": bool(inputs.merge_analyzer_compute_sparsity),
				"merge_template_random_spikes_method": str(
					inputs.merge_template_random_spikes_method
				),
				"merge_template_random_spikes_percentage": inputs.merge_template_random_spikes_percentage,
				"merge_template_random_spikes_max_spikes_per_unit": inputs.merge_template_random_spikes_max_spikes_per_unit,
				"merge_template_random_spikes_log_before_after_spike_counts": inputs.merge_template_random_spikes_log_before_after_spike_counts,
				"merge_template_random_spikes_min_spikes_per_unit": inputs.merge_template_random_spikes_min_spikes_per_unit,
				"merge_template_random_spikes_margin_size": inputs.merge_template_random_spikes_margin_size,
				"merge_template_random_spikes_seed": inputs.merge_template_random_spikes_seed,
				"merge_analyzer_n_jobs": inputs.merge_analyzer_n_jobs,
				"merge_analyzer_chunk_duration": inputs.merge_analyzer_chunk_duration,
				"merge_analyzer_sparsity_method": str(inputs.merge_analyzer_sparsity_method),
				"merge_analyzer_sparsity_radius_um": inputs.merge_analyzer_sparsity_radius_um,
				"merge_analyzer_sparsity_num_channels": inputs.merge_analyzer_sparsity_num_channels,
				"merge_analyzer_sparsity_threshold": inputs.merge_analyzer_sparsity_threshold,
				"merge_analyzer_sparsity_peak_sign": str(inputs.merge_analyzer_sparsity_peak_sign),
				"merge_analyzer_sparsity_num_spikes_for_sparsity": inputs.merge_analyzer_sparsity_num_spikes_for_sparsity,
				"merge_analyzer_sparsity_by_property": inputs.merge_analyzer_sparsity_by_property,
				"merge_analyzer_waveforms_ms_before": inputs.merge_analyzer_waveforms_ms_before,
				"merge_analyzer_waveforms_ms_after": inputs.merge_analyzer_waveforms_ms_after,
				"merge_analyzer_waveforms_dtype": inputs.merge_analyzer_waveforms_dtype,
			},
			"cleanup": {
				"removed_on_force_restart": list(removed_on_force_restart),
			},
			"summarize_sort": summarize_sort_report,
			"outputs": outputs,
		},
	)

	return SpikesortResult(
		well_out_dir=well_out_dir,
		spikesort_out_dir=spikesort_out_dir,
		summary_json=summary_json,
		outputs=outputs,
	)


def run_spikesort_summarize_sort(inputs: SpikesortInputs) -> SpikesortResult:
	well_out_dir = compute_mea_analysis_output_dir(
		output_root=inputs.mea_output_root,
		data_file=inputs.h5_path,
		well=inputs.stream_id,
	)
	stage_output_root_dir = _resolve_under_well(
		well_out_dir=well_out_dir,
		relpath=(str(inputs.output_rel_root).strip() or "spikesort_outputs"),
	)
	summary_json = (stage_output_root_dir / "summarize_sort_summary.json").resolve()
	payload = _run_summarize_sort_phase(
		stream_id=str(inputs.stream_id),
		stage_output_root_dir=stage_output_root_dir,
		sorter_name=str(inputs.sorter),
		emit_logs=bool(inputs.summarize_sort_emit_logs),
		generate_artifacts=bool(inputs.summarize_sort_generate_artifacts),
	)
	outputs = dict(payload.get("outputs", {}) or {})
	if not bool(inputs.summarize_sort_generate_artifacts):
		_write_json(summary_json, payload)
		outputs.setdefault("summarize_sort.summary_json", str(summary_json))
	else:
		artifact_summary = outputs.get("summarize_sort.summary_json", None)
		if artifact_summary is not None:
			summary_json = Path(str(artifact_summary)).resolve()
	return SpikesortResult(
		well_out_dir=well_out_dir,
		spikesort_out_dir=stage_output_root_dir,
		summary_json=summary_json,
		outputs=outputs,
	)
