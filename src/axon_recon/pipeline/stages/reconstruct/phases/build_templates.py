from __future__ import annotations

import concurrent.futures
import errno
import gc
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from axon_recon.pipeline.execution import install_linux_parent_death_signal
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
    SOURCE_PAYLOADS_CACHE_RELPATH,
    load_materialized_source_payload,
    resolve_materialized_source_payload_unit_dir,
    write_json,
    write_materialized_source_payload,
)
from ..templates.models.inputs import TemplatesInputs
from ..templates.source_units import (
    extract_analyzer_unit_ids,
    load_analyzer_source_units,
    resolve_analyzer_source_units_path,
    unit_key,
    write_analyzer_source_units,
)

LOGGER = logging.getLogger("axon_recon.templates.build_templates")


@dataclass(frozen=True)
class BuildTemplatesContext:
    well_out_dir: Path
    alternate_well_out_dirs: list[Path]
    templates_out_dir: Path
    analyzer_cache_dir: Path | None
    payload_root: Path
    payload_output_rel_root: str


@dataclass(frozen=True)
class _CachedAnalyzerUnitMaterializationJob:
    inputs: TemplatesInputs
    context: BuildTemplatesContext
    analyzer_well_out_dir: Path
    analyzer_cache_dir: Path
    source_names: tuple[str, ...]
    unit_id: Any
    source_unit_keys_by_source: dict[str, frozenset[str]] | None = None


@dataclass(frozen=True)
class _CachedAnalyzerUnitMaterializationResult:
    unit_id: Any
    source_names: tuple[str, ...]
    source_results: tuple["_CachedAnalyzerUnitSourceMaterializationResult", ...]


@dataclass(frozen=True)
class _CachedAnalyzerUnitSourceMaterializationJob:
    inputs: TemplatesInputs
    context: BuildTemplatesContext
    analyzer_well_out_dir: Path
    analyzer_cache_dir: Path
    source_name: str
    unit_id: Any
    source_unit_keys_by_source: dict[str, frozenset[str]] | None = None


@dataclass(frozen=True)
class _CachedAnalyzerUnitSourceMaterializationResult:
    unit_id: Any
    source_name: str
    status: str
    duration_seconds: float
    channel_count: int | None
    waveform_count: int | None


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
        "limit_wells_per_dataset": _positive_int_or_none(
            getattr(inputs, "debug_limit_wells_per_dataset", None)
        ),
        "limit_units": _positive_int_or_none(getattr(inputs, "unit_limit", None)),
        "limit_segments": _positive_int_or_none(getattr(inputs, "limit_segments", None)),
    }
    return {
        "debug_mode_enabled": bool(getattr(inputs, "debug_mode_enabled", False))
        or any(value is not None for value in limits.values()),
        **limits,
    }


def _resolve_alternate_well_out_dirs(
    *, inputs: TemplatesInputs, primary_well_out_dir: Path
) -> list[Path]:
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


def _resolve_templates_analyzer_cache_dir(
    *, inputs: TemplatesInputs, well_out_dir: Path
) -> Path | None:
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
    analyzer_cache_dir = _resolve_templates_analyzer_cache_dir(
        inputs=inputs, well_out_dir=well_out_dir
    )
    payload_output_rel_root = str(SOURCE_PAYLOADS_CACHE_RELPATH)
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
        "templates.build_templates settings: merge_enable=%s merge_method=%s centering_method=%s max_waveforms_per_source_channel=%s upsampling_enabled=%s upsampling_factor=%d upsampling_method=%s lazy_load_analyzers=%s emit_unit_source_materialization_log=%s",
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
        bool(getattr(inputs.phases.build_templates, "emit_unit_source_materialization_log", False)),
    )


def _clear_payload_root_for_force_restart(
    *, inputs: TemplatesInputs, context: BuildTemplatesContext
) -> None:
    if not bool(inputs.force_restart) or not context.payload_root.exists():
        return
    LOGGER.info(
        "templates.build_templates clearing persisted source payloads on force_restart: %s",
        str(context.payload_root),
    )
    try:
        shutil.rmtree(context.payload_root)
    except OSError as exc:
        if exc.errno != errno.ENOTEMPTY or not context.payload_root.exists():
            raise
        LOGGER.warning(
            "templates.build_templates source payload cleanup hit a non-empty directory race; retrying: %s",
            str(context.payload_root),
        )
        shutil.rmtree(context.payload_root)


def _discover_cached_analyzer_sources(
    *,
    inputs: TemplatesInputs,
    context: BuildTemplatesContext,
) -> tuple[Path, Path | None, list[str]]:
    include_concat = bool(inputs.include_concat) and bool(inputs.phases.analyzers.concat.enabled)
    include_segments = bool(inputs.include_segments) and bool(
        inputs.phases.analyzers.segments.enabled
    )
    primary_cache_dir = context.analyzer_cache_dir
    for candidate_well_out_dir in [context.well_out_dir, *list(context.alternate_well_out_dirs)]:
        candidate_cache_dir = _resolve_templates_analyzer_cache_dir(
            inputs=inputs, well_out_dir=candidate_well_out_dir
        )
        source_names = discover_cached_spikeinterface_analyzer_source_names(
            analyzer_cache_dir=candidate_cache_dir,
            analyzer_cache_concat_subdir=str(
                inputs.analyzer_cache.concat_analyzer_subdir or "concat"
            ),
            analyzer_cache_segments_subdir=str(
                inputs.analyzer_cache.segment_analyzers_subdir or ""
            ),
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
            inputs.phases.analyzers.concat.preprocessed_recording_reldir
            or inputs.preprocessed_concat_reldir
        ),
        preprocessed_segments_reldir=(
            inputs.phases.analyzers.segments.preprocessed_sources_reldir
            or inputs.preprocessed_segments_reldir
        ),
        preproc_seg_sources_reldir=(
            inputs.phases.analyzers.segments.preprocessed_sources_reldir
            or inputs.preproc_seg_sources_reldir
        ),
        analyzer_cache_dir=analyzer_cache_dir,
        analyzer_cache_concat_subdir=str(inputs.analyzer_cache.concat_analyzer_subdir or "concat"),
        analyzer_cache_segments_subdir=str(inputs.analyzer_cache.segment_analyzers_subdir or ""),
        include_concat=bool(inputs.include_concat) and bool(inputs.phases.analyzers.concat.enabled),
        include_segments=bool(inputs.include_segments)
        and bool(inputs.phases.analyzers.segments.enabled),
        requested_source_names=[str(requested_source_name)],
        limit_segments=inputs.limit_segments,
        load_extensions=(not bool(lazy_load_analyzers)),
        attach_recordings=(not bool(lazy_load_analyzers)),
    )
    source_match = next(
        (
            (name, analyzer)
            for name, analyzer in analyzers
            if str(name) == str(requested_source_name)
        ),
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
    allowed_labels = tuple(
        str(label).strip().lower()
        for label in inputs.unit_label_filter_labels
        if str(label).strip()
    )
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
    return _apply_build_templates_unit_label_filter(
        inputs=inputs, unit_ids=unit_ids, well_out_dir=well_out_dir
    )


def _source_unit_manifest_unit_ids(
    *, context: BuildTemplatesContext, source_name: str
) -> list[Any] | None:
    payload = load_analyzer_source_units(
        templates_out_dir=context.templates_out_dir,
        source_name=str(source_name),
    )
    if payload is None:
        return None
    unit_ids = payload.get("unit_ids", None)
    if not isinstance(unit_ids, list):
        return None
    return list(unit_ids)


def _write_source_unit_manifest_from_analyzer(
    *, context: BuildTemplatesContext, source_name: str, analyzer: Any
) -> list[Any] | None:
    unit_ids = extract_analyzer_unit_ids(analyzer)
    if unit_ids is None:
        return None
    manifest_path = write_analyzer_source_units(
        templates_out_dir=context.templates_out_dir,
        source_name=str(source_name),
        source_kind=("concat" if str(source_name) == "concat" else "segment"),
        unit_ids=list(unit_ids),
    )
    LOGGER.info(
        "templates.build_templates wrote source unit manifest: source=%s unit_count=%d path=%s",
        str(source_name),
        int(len(unit_ids)),
        str(manifest_path),
    )
    return list(unit_ids)


def _load_or_create_source_unit_manifest(
    *,
    inputs: TemplatesInputs,
    context: BuildTemplatesContext,
    analyzer_well_out_dir: Path,
    analyzer_cache_dir: Path,
    source_name: str,
) -> list[Any] | None:
    existing = _source_unit_manifest_unit_ids(context=context, source_name=str(source_name))
    if existing is not None:
        return existing
    LOGGER.info(
        "templates.build_templates source unit manifest missing; "
        "loading cached analyzer to create it: source=%s path=%s",
        str(source_name),
        str(
            resolve_analyzer_source_units_path(
                templates_out_dir=context.templates_out_dir,
                source_name=str(source_name),
            )
        ),
    )
    source_match, analyzers = _load_requested_cached_source(
        inputs=inputs,
        analyzer_well_out_dir=analyzer_well_out_dir,
        analyzer_cache_dir=analyzer_cache_dir,
        requested_source_name=str(source_name),
        lazy_load_analyzers=True,
    )
    loaded_source_name, analyzer = source_match
    unit_ids = _write_source_unit_manifest_from_analyzer(
        context=context,
        source_name=str(loaded_source_name),
        analyzer=analyzer,
    )
    del analyzer
    del analyzers
    del source_match
    gc.collect()
    return unit_ids


def _load_or_create_source_unit_manifests(
    *,
    inputs: TemplatesInputs,
    context: BuildTemplatesContext,
    analyzer_well_out_dir: Path,
    analyzer_cache_dir: Path,
    source_names: list[str],
) -> dict[str, list[Any] | None]:
    manifests: dict[str, list[Any] | None] = {}
    for source_name in source_names:
        try:
            manifests[str(source_name)] = _load_or_create_source_unit_manifest(
                inputs=inputs,
                context=context,
                analyzer_well_out_dir=analyzer_well_out_dir,
                analyzer_cache_dir=analyzer_cache_dir,
                source_name=str(source_name),
            )
        except Exception:
            manifests[str(source_name)] = None
            LOGGER.warning(
                "templates.build_templates failed to create source unit manifest; "
                "dispatch will fall back to payload probing: source=%s",
                str(source_name),
                exc_info=True,
            )
    return manifests


def _source_unit_keys_by_source(
    source_unit_ids_by_source: dict[str, list[Any] | None],
) -> dict[str, frozenset[str]]:
    return {
        str(source_name): frozenset(unit_key(unit_id) for unit_id in list(unit_ids))
        for source_name, unit_ids in source_unit_ids_by_source.items()
        if unit_ids is not None
    }


def _collect_unit_ids_from_source_manifests(
    source_unit_ids_by_source: dict[str, list[Any] | None], source_names: list[str]
) -> list[Any]:
    unit_ids: list[Any] = []
    seen: set[str] = set()
    for source_name in source_names:
        for unit_id in list(source_unit_ids_by_source.get(str(source_name)) or []):
            key = unit_key(unit_id)
            if key in seen:
                continue
            seen.add(key)
            unit_ids.append(unit_id)
    return unit_ids


def _source_manifest_allows_unit(
    *,
    source_unit_keys_by_source: dict[str, frozenset[str]] | None,
    source_name: str,
    unit_id: Any,
) -> bool:
    if source_unit_keys_by_source is None:
        return True
    unit_keys = source_unit_keys_by_source.get(str(source_name), None)
    if unit_keys is None:
        return True
    return unit_key(unit_id) in unit_keys


def _source_payload_artifact_exists(
    *, context: BuildTemplatesContext, source_name: str, unit_id: Any
) -> bool:
    unit_dir = resolve_materialized_source_payload_unit_dir(
        templates_out_dir=context.templates_out_dir,
        output_rel_root=context.payload_output_rel_root,
        source_name=str(source_name),
        unit_id=unit_id,
    )
    required = (
        unit_dir / "template.npy",
        unit_dir / "channel_locations_xy.npy",
        unit_dir / "payload_meta.json",
    )
    try:
        return all(path.exists() and path.stat().st_size > 0 for path in required)
    except OSError:
        return False


def _preflight_unit_source_materialization_result(
    job: _CachedAnalyzerUnitSourceMaterializationJob,
) -> _CachedAnalyzerUnitSourceMaterializationResult | None:
    if not _source_manifest_allows_unit(
        source_unit_keys_by_source=job.source_unit_keys_by_source,
        source_name=str(job.source_name),
        unit_id=job.unit_id,
    ):
        return _CachedAnalyzerUnitSourceMaterializationResult(
            unit_id=job.unit_id,
            source_name=str(job.source_name),
            status="skipped_unit_absent_preflight",
            duration_seconds=0.0,
            channel_count=None,
            waveform_count=None,
        )
    if _source_payload_artifact_exists(
        context=job.context,
        source_name=str(job.source_name),
        unit_id=job.unit_id,
    ):
        return _CachedAnalyzerUnitSourceMaterializationResult(
            unit_id=job.unit_id,
            source_name=str(job.source_name),
            status="reused_payload",
            duration_seconds=0.0,
            channel_count=None,
            waveform_count=None,
        )
    return None


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


def _record_materialization_result(
    summary: dict[str, Any],
    result: _CachedAnalyzerUnitMaterializationResult,
) -> None:
    for source_result in result.source_results:
        source_name = str(source_result.source_name)
        summary_entry = summary.setdefault(
            str(source_name),
            {
                "units_materialized": [],
                "units_reused": [],
                "units_skipped_absent": [],
                "unit_count": 0,
            },
        )
        status = str(source_result.status)
        if status == "materialized":
            summary_entry.setdefault("units_materialized", []).append(result.unit_id)
        elif status == "reused_payload":
            summary_entry.setdefault("units_reused", []).append(result.unit_id)
        elif status == "skipped_unit_absent_preflight":
            summary_entry.setdefault("units_skipped_absent", []).append(result.unit_id)
        available_count = len(summary_entry.get("units_materialized", [])) + len(
            summary_entry.get("units_reused", [])
        )
        summary_entry["unit_count"] = int(available_count)


def _payload_channel_count(payload: tuple[Any, ...]) -> int | None:
    try:
        return int(len(payload[1]))
    except Exception:
        return None


def _payload_waveform_count(payload: tuple[Any, ...]) -> int | None:
    try:
        return int(payload[4])
    except Exception:
        return None


def _log_unit_source_materialization_if_requested(
    *,
    inputs: TemplatesInputs,
    result: _CachedAnalyzerUnitSourceMaterializationResult,
    lazy_load_analyzers: bool,
) -> None:
    if not bool(getattr(inputs.phases.build_templates, "emit_unit_source_materialization_log", False)):
        return
    LOGGER.info(
        "templates.build_templates unit-source payload materialization: unit=%s source=%s status=%s lazy_load_analyzers=%s duration_seconds=%.3f channel_count=%s waveform_count=%s",
        str(result.unit_id),
        str(result.source_name),
        str(result.status),
        bool(lazy_load_analyzers),
        float(result.duration_seconds),
        "unknown" if result.channel_count is None else str(int(result.channel_count)),
        "unknown" if result.waveform_count is None else str(int(result.waveform_count)),
    )


def _materialize_cached_analyzer_unit_source(
    job: _CachedAnalyzerUnitSourceMaterializationJob,
) -> _CachedAnalyzerUnitSourceMaterializationResult:
    preflight_result = _preflight_unit_source_materialization_result(job)
    if preflight_result is not None:
        return preflight_result
    source_started = perf_counter()
    source_match, analyzers = _load_requested_cached_source(
        inputs=job.inputs,
        analyzer_well_out_dir=job.analyzer_well_out_dir,
        analyzer_cache_dir=job.analyzer_cache_dir,
        requested_source_name=str(job.source_name),
        lazy_load_analyzers=True,
    )
    source_name, analyzer = source_match
    payload = build_unit_source_payload(
        analyzer=analyzer,
        unit_id=job.unit_id,
        include_overlay_waveforms=False,
        allow_prepare=False,
        allow_waveforms_sparsity_fallback=False,
    )
    status = "skipped_empty_payload"
    channel_count = None
    waveform_count = None
    if payload is not None:
        channel_count = _payload_channel_count(payload)
        waveform_count = _payload_waveform_count(payload)
        _write_cached_analyzer_payload(
            context=job.context,
            source_name=str(source_name),
            unit_id=job.unit_id,
            payload=payload,
        )
        status = "materialized"
        del payload
    del analyzer
    del analyzers
    del source_match
    gc.collect()
    return _CachedAnalyzerUnitSourceMaterializationResult(
        unit_id=job.unit_id,
        source_name=str(source_name),
        status=status,
        duration_seconds=float(perf_counter() - source_started),
        channel_count=channel_count,
        waveform_count=waveform_count,
    )


def _cached_analyzer_source_jobs_for_unit(
    job: _CachedAnalyzerUnitMaterializationJob,
) -> list[_CachedAnalyzerUnitSourceMaterializationJob]:
    return [
        _CachedAnalyzerUnitSourceMaterializationJob(
            inputs=job.inputs,
            context=job.context,
            analyzer_well_out_dir=job.analyzer_well_out_dir,
            analyzer_cache_dir=job.analyzer_cache_dir,
            source_name=str(source_name),
            unit_id=job.unit_id,
            source_unit_keys_by_source=job.source_unit_keys_by_source,
        )
        for source_name in job.source_names
    ]


def _cached_analyzer_unit_result_from_source_results(
    *,
    unit_id: Any,
    source_names: tuple[str, ...],
    source_results: list[_CachedAnalyzerUnitSourceMaterializationResult],
) -> _CachedAnalyzerUnitMaterializationResult:
    results_by_source = {str(result.source_name): result for result in source_results}
    ordered_results = [
        results_by_source[str(source_name)]
        for source_name in source_names
        if str(source_name) in results_by_source
    ]
    return _CachedAnalyzerUnitMaterializationResult(
        unit_id=unit_id,
        source_names=tuple(
            str(result.source_name)
            for result in ordered_results
            if result.status in {"materialized", "reused_payload"}
        ),
        source_results=tuple(ordered_results),
    )


def _materialize_cached_analyzer_unit(
    job: _CachedAnalyzerUnitMaterializationJob,
) -> _CachedAnalyzerUnitMaterializationResult:
    source_results = [
        _materialize_cached_analyzer_unit_source(source_job)
        for source_job in _cached_analyzer_source_jobs_for_unit(job)
    ]
    return _cached_analyzer_unit_result_from_source_results(
        unit_id=job.unit_id,
        source_names=job.source_names,
        source_results=source_results,
    )


def _run_cached_analyzer_unit_materialization_jobs_serial(
    *,
    jobs: list[_CachedAnalyzerUnitMaterializationJob],
) -> list[_CachedAnalyzerUnitMaterializationResult]:
    results: list[_CachedAnalyzerUnitMaterializationResult] = []
    for job in jobs:
        source_results: list[_CachedAnalyzerUnitSourceMaterializationResult] = []
        for source_job in _cached_analyzer_source_jobs_for_unit(job):
            source_result = _materialize_cached_analyzer_unit_source(source_job)
            source_results.append(source_result)
            _log_unit_source_materialization_if_requested(
                inputs=job.inputs,
                result=source_result,
                lazy_load_analyzers=True,
            )
        results.append(
            _cached_analyzer_unit_result_from_source_results(
                unit_id=job.unit_id,
                source_names=job.source_names,
                source_results=source_results,
            )
        )
    return results


def _run_cached_analyzer_unit_materialization_jobs_with_processes(
    *,
    jobs: list[_CachedAnalyzerUnitMaterializationJob],
    worker_count: int,
) -> list[_CachedAnalyzerUnitMaterializationResult]:
    if not jobs:
        return []
    source_results_by_unit: dict[str, list[_CachedAnalyzerUnitSourceMaterializationResult]] = {
        str(job.unit_id): [] for job in jobs
    }
    source_jobs: list[_CachedAnalyzerUnitSourceMaterializationJob] = []
    for source_name in jobs[0].source_names:
        for job in jobs:
            source_job = _CachedAnalyzerUnitSourceMaterializationJob(
                inputs=job.inputs,
                context=job.context,
                analyzer_well_out_dir=job.analyzer_well_out_dir,
                analyzer_cache_dir=job.analyzer_cache_dir,
                source_name=str(source_name),
                unit_id=job.unit_id,
                source_unit_keys_by_source=job.source_unit_keys_by_source,
            )
            preflight_result = _preflight_unit_source_materialization_result(source_job)
            if preflight_result is not None:
                source_results_by_unit.setdefault(str(preflight_result.unit_id), []).append(
                    preflight_result
                )
                _log_unit_source_materialization_if_requested(
                    inputs=jobs[0].inputs,
                    result=preflight_result,
                    lazy_load_analyzers=True,
                )
                continue
            source_jobs.append(source_job)
    if source_jobs:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max(1, int(worker_count)),
            initializer=install_linux_parent_death_signal,
        ) as pool:
            futures = {
                pool.submit(_materialize_cached_analyzer_unit_source, job): job
                for job in source_jobs
            }
            for future in concurrent.futures.as_completed(futures):
                source_result = future.result()
                source_results_by_unit.setdefault(str(source_result.unit_id), []).append(
                    source_result
                )
                _log_unit_source_materialization_if_requested(
                    inputs=jobs[0].inputs,
                    result=source_result,
                    lazy_load_analyzers=True,
                )
    return [
        _cached_analyzer_unit_result_from_source_results(
            unit_id=job.unit_id,
            source_names=job.source_names,
            source_results=source_results_by_unit.get(str(job.unit_id), []),
        )
        for job in jobs
    ]


def _materialize_cached_analyzers_by_unit(
    *,
    inputs: TemplatesInputs,
    context: BuildTemplatesContext,
    analyzer_well_out_dir: Path,
    analyzer_cache_dir: Path,
    source_names: list[str],
    unit_ids: list[Any] | None,
    source_unit_ids_by_source: dict[str, list[Any] | None],
) -> tuple[list[Any], dict[str, Any]]:
    streamed_sources_summary = {
        str(source_name): {
            "units_materialized": [],
            "units_reused": [],
            "units_skipped_absent": [],
            "unit_count": 0,
        }
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
    source_unit_keys_by_source = _source_unit_keys_by_source(source_unit_ids_by_source)

    jobs = [
        _CachedAnalyzerUnitMaterializationJob(
            inputs=inputs,
            context=context,
            analyzer_well_out_dir=analyzer_well_out_dir,
            analyzer_cache_dir=analyzer_cache_dir,
            source_names=tuple(str(source_name) for source_name in source_names),
            unit_id=unit_id,
            source_unit_keys_by_source=source_unit_keys_by_source,
        )
        for unit_id in list(unit_ids or [])
    ]
    worker_count = max(1, min(len(jobs), int(max(1, int(inputs.n_jobs))))) if jobs else 1
    executor_kind = "serial" if worker_count <= 1 or len(jobs) <= 1 else "process"
    LOGGER.info(
        "templates.build_templates lazy materialization start: requested_units=%d source_count=%d worker_count=%d executor=%s",
        len(jobs),
        len(source_names),
        int(worker_count),
        str(executor_kind),
    )
    results: list[_CachedAnalyzerUnitMaterializationResult] = []
    if executor_kind == "process":
        try:
            results = _run_cached_analyzer_unit_materialization_jobs_with_processes(
                jobs=jobs,
                worker_count=worker_count,
            )
        except Exception as exc:
            LOGGER.warning(
                "templates.build_templates lazy materialization process workers failed; falling back to serial execution: %s",
                exc,
            )
    if not results:
        results = _run_cached_analyzer_unit_materialization_jobs_serial(jobs=jobs)
    results_by_unit = {str(result.unit_id): result for result in results}
    for unit_id in list(unit_ids or []):
        result = results_by_unit.get(str(unit_id))
        if result is None:
            continue
        _record_materialization_result(streamed_sources_summary, result)
        LOGGER.info(
            "templates.build_templates lazy-materialized cached analyzer payloads: unit=%s source_count=%d",
            str(result.unit_id),
            int(len(result.source_names)),
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
    source_unit_ids_by_source: dict[str, list[Any] | None],
) -> tuple[list[Any], dict[str, Any]]:
    streamed_sources_summary: dict[str, Any] = {}
    source_unit_keys_by_source = _source_unit_keys_by_source(source_unit_ids_by_source)
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
        reused_units: list[Any] = []
        skipped_absent_units: list[Any] = []
        for unit_id in list(unit_ids or []):
            if not _source_manifest_allows_unit(
                source_unit_keys_by_source=source_unit_keys_by_source,
                source_name=str(source_name),
                unit_id=unit_id,
            ):
                skipped_absent_units.append(unit_id)
                _log_unit_source_materialization_if_requested(
                    inputs=inputs,
                    result=_CachedAnalyzerUnitSourceMaterializationResult(
                        unit_id=unit_id,
                        source_name=str(source_name),
                        status="skipped_unit_absent_preflight",
                        duration_seconds=0.0,
                        channel_count=None,
                        waveform_count=None,
                    ),
                    lazy_load_analyzers=False,
                )
                continue
            if _source_payload_artifact_exists(
                context=context,
                source_name=str(source_name),
                unit_id=unit_id,
            ):
                reused_units.append(unit_id)
                _log_unit_source_materialization_if_requested(
                    inputs=inputs,
                    result=_CachedAnalyzerUnitSourceMaterializationResult(
                        unit_id=unit_id,
                        source_name=str(source_name),
                        status="reused_payload",
                        duration_seconds=0.0,
                        channel_count=None,
                        waveform_count=None,
                    ),
                    lazy_load_analyzers=False,
                )
                continue
            source_started = perf_counter()
            payload = build_unit_source_payload(
                analyzer=analyzer,
                unit_id=unit_id,
                include_overlay_waveforms=False,
                allow_prepare=False,
            )
            if payload is None:
                _log_unit_source_materialization_if_requested(
                    inputs=inputs,
                    result=_CachedAnalyzerUnitSourceMaterializationResult(
                        unit_id=unit_id,
                        source_name=str(source_name),
                        status="skipped_empty_payload",
                        duration_seconds=float(perf_counter() - source_started),
                        channel_count=None,
                        waveform_count=None,
                    ),
                    lazy_load_analyzers=False,
                )
                continue
            channel_count = _payload_channel_count(payload)
            waveform_count = _payload_waveform_count(payload)
            _write_cached_analyzer_payload(
                context=context,
                source_name=str(source_name),
                unit_id=unit_id,
                payload=payload,
            )
            materialized_units.append(unit_id)
            _log_unit_source_materialization_if_requested(
                inputs=inputs,
                result=_CachedAnalyzerUnitSourceMaterializationResult(
                    unit_id=unit_id,
                    source_name=str(source_name),
                    status="materialized",
                    duration_seconds=float(perf_counter() - source_started),
                    channel_count=channel_count,
                    waveform_count=waveform_count,
                ),
                lazy_load_analyzers=False,
            )
            del payload
        streamed_sources_summary[str(source_name)] = {
            "units_materialized": [unit for unit in materialized_units],
            "units_reused": [unit for unit in reused_units],
            "units_skipped_absent": [unit for unit in skipped_absent_units],
            "unit_count": int(len(materialized_units) + len(reused_units)),
        }
        LOGGER.info(
            "templates.build_templates materialized cached analyzer payloads: "
            "source=%s unit_count=%d reused_units=%d skipped_absent_units=%d",
            str(source_name),
            int(len(materialized_units) + len(reused_units)),
            int(len(reused_units)),
            int(len(skipped_absent_units)),
        )
        del analyzer
        del analyzers
        del source_match
        gc.collect()
    return list(unit_ids or []), streamed_sources_summary


def _discover_partial_payload_unit_ids(
    *, context: BuildTemplatesContext, source_names: list[str]
) -> list[Any]:
    """Discover unit ids that have at least one partial payload artifact on disk.

    The build_templates phase merges partial payloads written by
    extract_partial_templates. It must NOT reopen segment analyzers; the only
    way to enumerate units here is to read the on-disk partial payload tree.
    """
    discovered: list[Any] = []
    seen: set[str] = set()
    for source_name in source_names:
        source_dir = context.payload_root / str(source_name)
        if not source_dir.exists():
            continue
        for unit_dir in sorted(source_dir.iterdir()):
            if not unit_dir.is_dir():
                continue
            token = unit_dir.name
            if token.startswith("unit_"):
                token = token.split("unit_", 1)[1]
            try:
                unit_id: Any = int(token)
            except (TypeError, ValueError):
                unit_id = token
            key = str(unit_id)
            if key in seen:
                continue
            seen.add(key)
            discovered.append(unit_id)
    return discovered


def _build_templates_from_existing_payloads(
    *, inputs: TemplatesInputs, context: BuildTemplatesContext
) -> dict[str, Any]:
    """Merge per-unit templates from previously written partial payloads.

    Reads the partial payload set produced by extract_partial_templates and
    fans out across units. Does NOT reopen any segment analyzers.
    """
    payload_root = context.payload_root
    if not payload_root.exists():
        raise FileNotFoundError(
            f"Missing materialized source payloads at {payload_root}; "
            "run templates.extract_partial_templates before templates.build_templates."
        )

    # Discover sources that actually have partials on disk. We rely on the
    # extract phase to have laid these out; we never re-open analyzers.
    source_names: list[str] = []
    for source_dir in sorted(payload_root.iterdir()):
        if source_dir.is_dir():
            source_names.append(source_dir.name)
    if not source_names:
        raise FileNotFoundError(
            f"No partial payload sources found under {payload_root}; "
            "run templates.extract_partial_templates before templates.build_templates."
        )

    unit_ids: list[Any] | None = None if inputs.unit_ids is None else list(inputs.unit_ids)
    if unit_ids is not None and inputs.unit_limit is not None:
        unit_ids = unit_ids[: int(inputs.unit_limit)]
    if unit_ids is not None:
        unit_ids = _apply_build_templates_unit_label_filter(
            inputs=inputs,
            unit_ids=unit_ids,
            well_out_dir=context.well_out_dir,
        )
    if unit_ids is None:
        discovered = _discover_partial_payload_unit_ids(
            context=context, source_names=source_names
        )
        if inputs.unit_limit is not None:
            discovered = discovered[: int(inputs.unit_limit)]
        unit_ids = _apply_build_templates_unit_label_filter(
            inputs=inputs,
            unit_ids=discovered,
            well_out_dir=context.well_out_dir,
        )

    LOGGER.info(
        "templates.build_templates merging partial payloads: payload_root=%s source_count=%d unit_count=%d",
        str(payload_root),
        int(len(source_names)),
        int(len(unit_ids)),
    )

    summary = build_templates_phase_from_unit_payloads(
        inputs=inputs,
        well_out_dir=context.well_out_dir,
        templates_out_dir=context.templates_out_dir,
        unit_ids=list(unit_ids),
        source_names=source_names,
        payload_root=context.payload_root,
        payload_materialization_mode="partial_payloads",
        payload_loader=_build_payload_loader(context=context, source_names=source_names),
    )
    summary["source_names"] = list(source_names)
    return summary


def _write_build_templates_summary(
    *,
    inputs: TemplatesInputs,
    context: BuildTemplatesContext,
    summary: dict[str, Any],
    phase_started: float,
) -> dict[str, Any]:
    summary["timing"] = {"duration_seconds": float(perf_counter() - phase_started)}
    summary["applied_debug_limits"] = _build_templates_applied_debug_limits(inputs)
    summary_path = context.templates_out_dir / str(
        inputs.phases.build_templates.summary_json_relpath
    )
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

    # 2. build_templates only consumes partial payloads written by the
    #    extract_partial_templates phase. It MUST NOT reopen segment analyzers.
    payload_status = _payload_root_status(context.payload_root)
    if payload_status != "ready":
        raise FileNotFoundError(
            "templates.build_templates requires partial payloads to be present at "
            f"{context.payload_root} (status={payload_status}). Run "
            "templates.extract_partial_templates before templates.build_templates."
        )

    LOGGER.info(
        "templates.build_templates using existing partial payloads: payload_root=%s",
        str(context.payload_root),
    )
    summary = _build_templates_from_existing_payloads(inputs=inputs, context=context)

    # 3. Persist the phase summary after the core builder has written per-unit outputs.
    return _write_build_templates_summary(
        inputs=inputs,
        context=context,
        summary=summary,
        phase_started=phase_started,
    )


__all__ = ["run_reconstruct_templates_build_templates_phase"]
