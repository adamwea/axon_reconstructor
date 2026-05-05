from __future__ import annotations

import gc
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from axon_recon.pipeline.output_paths import compute_mea_analysis_output_dir

from ..templates.core.build_templates import (
	build_templates_phase_from_payloads,
	build_templates_phase_from_unit_payloads,
)
from ..templates.core.unit_labels import (
	count_labels,
	filter_unit_ids_by_labels,
	load_unit_labels_from_spikesorting,
)
from ..templates.integrations.spikeinterface_extract import (
	build_unit_source_payload,
	discover_cached_spikeinterface_analyzer_source_names,
	load_cached_spikeinterface_analyzers,
)
from ..templates.io import (
	load_materialized_source_payload,
	resolve_materialized_source_payload_unit_dir,
	write_json,
	write_materialized_source_payload,
)
from ..templates.models.inputs import TemplatesInputs

LOGGER = logging.getLogger("axon_recon.templates.build_templates")


@dataclass(frozen=True)
class BuildTemplatesContext:
	well_out_dir: Path
	alternate_well_out_dirs: list[Path]
	templates_out_dir: Path
	analyzer_cache_dir: Path | None
	payload_root: Path
	payload_output_rel_root: str


def _positive_int_or_none(value: Any) -> int | None:
	if value is None:
		return None
	try:
		parsed = int(value)
	except (TypeError, ValueError):
		return None
	return parsed if parsed > 0 else None


def _build_templates_applied_debug_limits(inputs: TemplatesInputs) -> dict[str, Any]:
	limits = {
		"limit_datasets": _positive_int_or_none(getattr(inputs, "debug_limit_datasets", None)),
		"limit_wells": _positive_int_or_none(getattr(inputs, "debug_limit_wells", None)),
		"limit_wells_per_dataset": _positive_int_or_none(getattr(inputs, "debug_limit_wells_per_dataset", None)),
		"limit_units": _positive_int_or_none(getattr(inputs, "unit_limit", None)),
		"limit_segments": _positive_int_or_none(getattr(inputs, "limit_segments", None)),
	}
	return {
		"debug_mode_enabled": bool(getattr(inputs, "debug_mode_enabled", False))
		or any(value is not None for value in limits.values()),
		**limits,
	}


def _resolve_alternate_well_out_dirs(*, inputs: TemplatesInputs, primary_well_out_dir: Path) -> list[Path]:
	roots_to_probe: list[Path] = []
	if inputs.final_output_root is not None:
		roots_to_probe.append(Path(inputs.final_output_root).expanduser().resolve())
	for root in list(inputs.artifact_lookup_roots or ()):  # model-level alternates from data config
		try:
			roots_to_probe.append(Path(root).expanduser().resolve())
		except Exception:
			continue

	resolved_primary = primary_well_out_dir.resolve()
	seen: set[Path] = {resolved_primary}
	alternate_well_out_dirs: list[Path] = []
	for candidate_root in roots_to_probe:
		try:
			candidate_well_out_dir = compute_mea_analysis_output_dir(
				output_root=candidate_root,
				data_file=inputs.h5_path,
				well=inputs.stream_id,
			)
			resolved_candidate = candidate_well_out_dir.resolve()
		except Exception:
			continue
		if resolved_candidate in seen:
			continue
		seen.add(resolved_candidate)
		alternate_well_out_dirs.append(candidate_well_out_dir)
	return alternate_well_out_dirs


def _resolve_templates_analyzer_cache_dir(*, inputs: TemplatesInputs, well_out_dir: Path) -> Path | None:
	if not bool(inputs.analyzer_cache.enabled):
		return None
	cache_rel = Path(str(inputs.analyzer_cache.relpath or "analyzers")).expanduser()
	if cache_rel.is_absolute():
		cache_rel = Path(str(cache_rel).lstrip("/"))
	return well_out_dir / str(inputs.output_rel_root) / cache_rel


def _resolve_build_templates_context(inputs: TemplatesInputs) -> BuildTemplatesContext:
	well_out_dir = compute_mea_analysis_output_dir(
		output_root=inputs.mea_output_root,
		data_file=inputs.h5_path,
		well=inputs.stream_id,
	)
	alternate_well_out_dirs = _resolve_alternate_well_out_dirs(
		inputs=inputs,
		primary_well_out_dir=well_out_dir,
	)
	templates_out_dir = well_out_dir / str(inputs.output_rel_root)
	templates_out_dir.mkdir(parents=True, exist_ok=True)
	analyzer_cache_dir = _resolve_templates_analyzer_cache_dir(inputs=inputs, well_out_dir=well_out_dir)
	payload_output_rel_root = str(inputs.phases.per_unit_processing.extract_template_segments.output_rel_root)
	payload_root = templates_out_dir / Path(payload_output_rel_root).expanduser()
	return BuildTemplatesContext(
		well_out_dir=well_out_dir,
		alternate_well_out_dirs=alternate_well_out_dirs,
		templates_out_dir=templates_out_dir,
		analyzer_cache_dir=analyzer_cache_dir,
		payload_root=payload_root,
		payload_output_rel_root=payload_output_rel_root,
	)


def _payload_root_status(payload_root: Path) -> str:
	if not payload_root.exists():
		return "missing"
	try:
		if any(path.is_dir() for path in payload_root.iterdir()):
			return "ready"
	except Exception:
		return "unreadable"
	return "empty"


def _log_build_templates_start(*, inputs: TemplatesInputs, context: BuildTemplatesContext) -> None:
	LOGGER.info(
		"templates.build_templates start: well_out_dir=%s templates_out_dir=%s payload_root=%s force_restart=%s",
		str(context.well_out_dir),
		str(context.templates_out_dir),
		str(context.payload_root),
		bool(inputs.force_restart),
	)
	LOGGER.info(
		"templates.build_templates settings: merge_enable=%s merge_method=%s centering_method=%s max_waveforms_per_source_channel=%s upsampling_enabled=%s upsampling_factor=%d upsampling_method=%s lazy_load_analyzers=%s",
		bool(inputs.phases.build_templates.merge.enable),
		str(inputs.phases.build_templates.merge.method),
		str(inputs.phases.build_templates.merge.centering_method),
		(
			"unlimited"
			if inputs.phases.build_templates.merge.max_waveforms_per_source_channel is None
			else str(int(inputs.phases.build_templates.merge.max_waveforms_per_source_channel))
		),
		bool(inputs.phases.build_templates.execution_upsampling.enabled),
		int(max(1, int(inputs.phases.build_templates.execution_upsampling.factor))),
		str(inputs.phases.build_templates.execution_upsampling.method),
		bool(getattr(inputs.phases.build_templates, "lazy_load_analyzers", False)),
	)


def _clear_payload_root_for_force_restart(*, inputs: TemplatesInputs, context: BuildTemplatesContext) -> None:
	if not bool(inputs.force_restart) or not context.payload_root.exists():
		return
	LOGGER.info("templates.build_templates clearing persisted source payloads on force_restart: %s", str(context.payload_root))
	shutil.rmtree(context.payload_root)


def _discover_cached_analyzer_sources(
	*,
	inputs: TemplatesInputs,
	context: BuildTemplatesContext,
) -> tuple[Path, Path | None, list[str]]:
	include_concat = bool(inputs.include_concat) and bool(inputs.phases.analyzers.concat.enabled)
	include_segments = bool(inputs.include_segments) and bool(inputs.phases.analyzers.segments.enabled)
	primary_cache_dir = context.analyzer_cache_dir
	for candidate_well_out_dir in [context.well_out_dir, *list(context.alternate_well_out_dirs)]:
		candidate_cache_dir = _resolve_templates_analyzer_cache_dir(inputs=inputs, well_out_dir=candidate_well_out_dir)
		source_names = discover_cached_spikeinterface_analyzer_source_names(
			analyzer_cache_dir=candidate_cache_dir,
			analyzer_cache_concat_subdir=str(inputs.analyzer_cache.concat_analyzer_subdir or "concat"),
			analyzer_cache_segments_subdir=str(inputs.analyzer_cache.segment_analyzers_subdir or ""),
			include_concat=include_concat,
			include_segments=include_segments,
			limit_segments=inputs.limit_segments,
		)
		if source_names:
			if candidate_well_out_dir != context.well_out_dir:
				LOGGER.info(
					"templates.build_templates using alternate analyzer cache for cached build bootstrap: primary=%s selected=%s source_count=%d",
					str(context.well_out_dir),
					str(candidate_well_out_dir),
					int(len(source_names)),
				)
			return candidate_well_out_dir, candidate_cache_dir, source_names
	return context.well_out_dir, primary_cache_dir, []


def _load_requested_cached_source(
	*,
	inputs: TemplatesInputs,
	analyzer_well_out_dir: Path,
	analyzer_cache_dir: Path,
	requested_source_name: str,
	lazy_load_analyzers: bool,
) -> tuple[tuple[str, Any], list[tuple[str, Any]]]:
	analyzers = load_cached_spikeinterface_analyzers(
		well_out_dir=analyzer_well_out_dir,
		preprocessed_concat_reldir=(
			inputs.phases.analyzers.concat.preprocessed_recording_reldir or inputs.preprocessed_concat_reldir
		),
		preprocessed_segments_reldir=(
			inputs.phases.analyzers.segments.preprocessed_sources_reldir or inputs.preprocessed_segments_reldir
		),
		preproc_seg_sources_reldir=(
			inputs.phases.analyzers.segments.preprocessed_sources_reldir or inputs.preproc_seg_sources_reldir
		),
		analyzer_cache_dir=analyzer_cache_dir,
		analyzer_cache_concat_subdir=str(inputs.analyzer_cache.concat_analyzer_subdir or "concat"),
		analyzer_cache_segments_subdir=str(inputs.analyzer_cache.segment_analyzers_subdir or ""),
		include_concat=bool(inputs.include_concat) and bool(inputs.phases.analyzers.concat.enabled),
		include_segments=bool(inputs.include_segments) and bool(inputs.phases.analyzers.segments.enabled),
		requested_source_names=[str(requested_source_name)],
		limit_segments=inputs.limit_segments,
		load_extensions=(not bool(lazy_load_analyzers)),
		attach_recordings=(not bool(lazy_load_analyzers)),
	)
	source_match = next(
		((name, analyzer) for name, analyzer in analyzers if str(name) == str(requested_source_name)),
		None,
	)
	if source_match is None:
		raise FileNotFoundError(
			f"Failed loading requested templates analyzer source {requested_source_name!r} under {analyzer_well_out_dir}"
		)
	return source_match, analyzers


def _apply_build_templates_unit_label_filter(
	*,
	inputs: TemplatesInputs,
	unit_ids: list[Any],
	well_out_dir: Path,
) -> list[Any]:
	allowed_labels = tuple(str(label).strip().lower() for label in inputs.unit_label_filter_labels if str(label).strip())
	if not allowed_labels:
		return list(unit_ids)
	labels_by_unit = load_unit_labels_from_spikesorting(well_out_dir)
	if not labels_by_unit:
		if bool(inputs.unit_label_filter_required):
			raise RuntimeError(
				"Templates unit label filter is enabled, but no Bombcell/Kilosort unit labels were found under "
				f"{well_out_dir}."
			)
		LOGGER.warning(
			"Templates build_templates: unit label filter skipped because no labels were found under %s",
			well_out_dir,
		)
		return list(unit_ids)
	filtered = filter_unit_ids_by_labels(unit_ids, labels_by_unit, allowed_labels)
	LOGGER.info(
		"Templates build_templates: unit label filter allowed=%s kept=%d/%d counts=%s",
		list(allowed_labels),
		len(filtered),
		len(unit_ids),
		count_labels(labels_by_unit),
	)
	return filtered


def _collect_build_templates_unit_ids(
	*,
	inputs: TemplatesInputs,
	analyzers: list[tuple[str, Any]],
	well_out_dir: Path,
) -> list[Any]:
	if inputs.unit_ids is not None:
		unit_ids = list(inputs.unit_ids)
	elif analyzers:
		unit_ids = list(getattr(analyzers[0][1].sorting, "unit_ids", []))
	else:
		unit_ids = []
	if inputs.unit_limit is not None:
		unit_ids = unit_ids[: int(inputs.unit_limit)]
	return _apply_build_templates_unit_label_filter(inputs=inputs, unit_ids=unit_ids, well_out_dir=well_out_dir)


def _write_cached_analyzer_payload(
	*,
	context: BuildTemplatesContext,
	source_name: str,
	unit_id: Any,
	payload: tuple[Any, ...],
) -> None:
	write_materialized_source_payload(
		templates_out_dir=context.templates_out_dir,
		output_rel_root=context.payload_output_rel_root,
		source_name=str(source_name),
		unit_id=unit_id,
		template_c_by_t=payload[0],
		locations_xy=payload[1],
		electrode_ids=payload[2],
		channel_ids=payload[3],
		waveform_count=payload[4],
		sampling_rate_hz=payload[5],
		overlay_waveforms=None,
		top_electrode_id=None,
		total_waveforms_at_channel=None,
	)


def _build_payload_loader(*, context: BuildTemplatesContext, source_names: list[str]):
	def _payload_loader(unit_id: Any) -> list[tuple[str, tuple[Any, ...]]]:
		loaded: list[tuple[str, tuple[Any, ...]]] = []
		for source_name in source_names:
			payload = load_materialized_source_payload(
				source_payload_unit_dir=resolve_materialized_source_payload_unit_dir(
					templates_out_dir=context.templates_out_dir,
					output_rel_root=context.payload_output_rel_root,
					source_name=str(source_name),
					unit_id=unit_id,
				),
			)
			if payload is None:
				continue
			loaded.append((str(source_name), payload))
		return loaded

	return _payload_loader


def _materialize_cached_analyzers_by_unit(
	*,
	inputs: TemplatesInputs,
	context: BuildTemplatesContext,
	analyzer_well_out_dir: Path,
	analyzer_cache_dir: Path,
	source_names: list[str],
	unit_ids: list[Any] | None,
) -> tuple[list[Any], dict[str, Any]]:
	streamed_sources_summary = {
		str(source_name): {"units_materialized": [], "unit_count": 0}
		for source_name in source_names
	}
	if unit_ids is None:
		first_match, first_analyzers = _load_requested_cached_source(
			inputs=inputs,
			analyzer_well_out_dir=analyzer_well_out_dir,
			analyzer_cache_dir=analyzer_cache_dir,
			requested_source_name=str(source_names[0]),
			lazy_load_analyzers=True,
		)
		unit_ids = _collect_build_templates_unit_ids(
			inputs=inputs,
			analyzers=[first_match],
			well_out_dir=context.well_out_dir,
		)
		del first_match
		del first_analyzers
		gc.collect()

	for unit_id in list(unit_ids or []):
		materialized_source_count = 0
		for requested_source_name in source_names:
			source_match, analyzers = _load_requested_cached_source(
				inputs=inputs,
				analyzer_well_out_dir=analyzer_well_out_dir,
				analyzer_cache_dir=analyzer_cache_dir,
				requested_source_name=str(requested_source_name),
				lazy_load_analyzers=True,
			)
			source_name, analyzer = source_match
			payload = build_unit_source_payload(
				analyzer=analyzer,
				unit_id=unit_id,
				include_overlay_waveforms=False,
				allow_prepare=False,
				allow_waveforms_sparsity_fallback=False,
			)
			if payload is not None:
				_write_cached_analyzer_payload(
					context=context,
					source_name=str(source_name),
					unit_id=unit_id,
					payload=payload,
				)
				summary_entry = streamed_sources_summary.setdefault(
					str(source_name),
					{"units_materialized": [], "unit_count": 0},
				)
				summary_entry["units_materialized"].append(unit_id)
				summary_entry["unit_count"] = int(summary_entry.get("unit_count", 0)) + 1
				materialized_source_count += 1
				del payload
			del analyzer
			del analyzers
			del source_match
			gc.collect()
		LOGGER.info(
			"templates.build_templates lazy-materialized cached analyzer payloads: unit=%s source_count=%d",
			str(unit_id),
			int(materialized_source_count),
		)
	return list(unit_ids or []), streamed_sources_summary


def _materialize_cached_analyzers_by_source(
	*,
	inputs: TemplatesInputs,
	context: BuildTemplatesContext,
	analyzer_well_out_dir: Path,
	analyzer_cache_dir: Path,
	source_names: list[str],
	unit_ids: list[Any] | None,
) -> tuple[list[Any], dict[str, Any]]:
	streamed_sources_summary: dict[str, Any] = {}
	for requested_source_name in source_names:
		source_match, analyzers = _load_requested_cached_source(
			inputs=inputs,
			analyzer_well_out_dir=analyzer_well_out_dir,
			analyzer_cache_dir=analyzer_cache_dir,
			requested_source_name=str(requested_source_name),
			lazy_load_analyzers=False,
		)
		source_name, analyzer = source_match
		if unit_ids is None:
			unit_ids = _collect_build_templates_unit_ids(
				inputs=inputs,
				analyzers=[source_match],
				well_out_dir=context.well_out_dir,
			)
		materialized_units: list[Any] = []
		for unit_id in list(unit_ids or []):
			payload = build_unit_source_payload(
				analyzer=analyzer,
				unit_id=unit_id,
				include_overlay_waveforms=False,
				allow_prepare=False,
			)
			if payload is None:
				continue
			_write_cached_analyzer_payload(
				context=context,
				source_name=str(source_name),
				unit_id=unit_id,
				payload=payload,
			)
			materialized_units.append(unit_id)
			del payload
		streamed_sources_summary[str(source_name)] = {
			"units_materialized": [unit for unit in materialized_units],
			"unit_count": int(len(materialized_units)),
		}
		LOGGER.info(
			"templates.build_templates materialized cached analyzer payloads: source=%s unit_count=%d",
			str(source_name),
			int(len(materialized_units)),
		)
		del analyzer
		del analyzers
		del source_match
		gc.collect()
	return list(unit_ids or []), streamed_sources_summary


def _build_templates_from_cached_analyzers(
	*,
	inputs: TemplatesInputs,
	context: BuildTemplatesContext,
) -> dict[str, Any]:
	analyzer_well_out_dir, analyzer_cache_dir, source_names = _discover_cached_analyzer_sources(
		inputs=inputs,
		context=context,
	)
	if analyzer_cache_dir is None or not source_names:
		raise FileNotFoundError(
			"No cached templates analyzers found for build_templates. "
			f"checked analyzer_cache_dir={analyzer_cache_dir}; run templates.analyzers before templates.build_templates."
		)

	lazy_load_analyzers = bool(getattr(inputs.phases.build_templates, "lazy_load_analyzers", False))
	LOGGER.info(
		"templates.build_templates loading cached analyzers for build bootstrap: analyzer_well_out_dir=%s analyzer_cache_dir=%s source_count=%d lazy_load_analyzers=%s",
		str(analyzer_well_out_dir),
		str(analyzer_cache_dir),
		int(len(source_names)),
		bool(lazy_load_analyzers),
	)

	unit_ids: list[Any] | None = (None if inputs.unit_ids is None else list(inputs.unit_ids))
	if unit_ids is not None and inputs.unit_limit is not None:
		unit_ids = unit_ids[: int(inputs.unit_limit)]
	source_names = [str(source_name) for source_name in source_names]
	if lazy_load_analyzers:
		unit_ids, streamed_sources_summary = _materialize_cached_analyzers_by_unit(
			inputs=inputs,
			context=context,
			analyzer_well_out_dir=analyzer_well_out_dir,
			analyzer_cache_dir=analyzer_cache_dir,
			source_names=source_names,
			unit_ids=unit_ids,
		)
	else:
		unit_ids, streamed_sources_summary = _materialize_cached_analyzers_by_source(
			inputs=inputs,
			context=context,
			analyzer_well_out_dir=analyzer_well_out_dir,
			analyzer_cache_dir=analyzer_cache_dir,
			source_names=source_names,
			unit_ids=unit_ids,
		)

	summary = build_templates_phase_from_unit_payloads(
		inputs=inputs,
		well_out_dir=context.well_out_dir,
		templates_out_dir=context.templates_out_dir,
		unit_ids=list(unit_ids),
		source_names=source_names,
		payload_root=context.payload_root,
		payload_materialization_mode="analyzer_cache",
		payload_loader=_build_payload_loader(context=context, source_names=source_names),
	)
	summary["source_payload_well_out_dir"] = str(analyzer_well_out_dir)
	summary["analyzer_cache_dir"] = str(analyzer_cache_dir)
	summary["lazy_load_analyzers"] = bool(lazy_load_analyzers)
	summary["source_payload_sources"] = streamed_sources_summary
	return summary


def _build_templates_from_existing_payloads(*, inputs: TemplatesInputs, context: BuildTemplatesContext) -> dict[str, Any]:
	return build_templates_phase_from_payloads(
		inputs=inputs,
		well_out_dir=context.well_out_dir,
		templates_out_dir=context.templates_out_dir,
	)


def _write_build_templates_summary(
	*,
	inputs: TemplatesInputs,
	context: BuildTemplatesContext,
	summary: dict[str, Any],
	phase_started: float,
) -> dict[str, Any]:
	summary["timing"] = {"duration_seconds": float(perf_counter() - phase_started)}
	summary["applied_debug_limits"] = _build_templates_applied_debug_limits(inputs)
	summary_path = context.templates_out_dir / str(inputs.phases.build_templates.summary_json_relpath)
	LOGGER.info("templates.build_templates generating outputs: summary_json=%s", str(summary_path))
	write_json(summary_path, summary)
	summary["summary_json"] = str(summary_path)
	LOGGER.info("templates.build_templates wrote summary output: %s", str(summary_path))
	LOGGER.info(
		"templates.build_templates run stats: duration_seconds=%.3f unit_count=%d built_units=%d skipped_units=%d",
		float(summary["timing"]["duration_seconds"]),
		int(summary.get("unit_count", 0)),
		int(len(summary.get("built_units", []))),
		int(len(summary.get("skipped_units", []))),
	)
	return summary


def run_reconstruct_templates_build_templates_phase(inputs: TemplatesInputs) -> dict[str, Any]:
	phase_started = perf_counter()

	# 1. Resolve the build_templates target paths and source payload cache location.
	context = _resolve_build_templates_context(inputs)
	_log_build_templates_start(inputs=inputs, context=context)

	# 2. Apply force-restart cleanup for build_templates-owned source payloads.
	_clear_payload_root_for_force_restart(inputs=inputs, context=context)
	payload_status = _payload_root_status(context.payload_root)

	# 3. Ensure source payloads exist, bootstrapping from cached analyzers when needed.
	if payload_status != "ready":
		bootstrap_reason = "force_restart" if bool(inputs.force_restart) else payload_status
		LOGGER.info(
			"templates.build_templates loading source payloads from cached analyzers: payload_root=%s reason=%s",
			str(context.payload_root),
			bootstrap_reason,
		)
		summary = _build_templates_from_cached_analyzers(inputs=inputs, context=context)
		LOGGER.info(
			"templates.build_templates loaded source payloads from cached analyzers: payload_root=%s source_count=%d analyzer_well_out_dir=%s analyzer_cache_dir=%s",
			str(context.payload_root),
			int(summary.get("source_count", 0)),
			str(summary.get("source_payload_well_out_dir", "")),
			str(summary.get("analyzer_cache_dir", "")),
		)
	else:
		LOGGER.info(
			"templates.build_templates using existing source payloads: payload_root=%s",
			str(context.payload_root),
		)
		summary = _build_templates_from_existing_payloads(inputs=inputs, context=context)

	# 4. Persist the phase summary after the core builder has written per-unit outputs.
	return _write_build_templates_summary(
		inputs=inputs,
		context=context,
		summary=summary,
		phase_started=phase_started,
	)


__all__ = ["run_reconstruct_templates_build_templates_phase"]
