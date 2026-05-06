from __future__ import annotations

import gc
import shutil
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any

from axon_recon.pipeline.stages.reconstruct.templates import runner as templates_runner
from axon_recon.pipeline.stages.reconstruct.templates.integrations.spikeinterface_extract import (
    discover_cached_spikeinterface_analyzer_source_names,
    discover_spikeinterface_analyzer_source_names,
)
from axon_recon.pipeline.stages.reconstruct.templates.models.inputs import TemplatesInputs
from axon_recon.pipeline.stages.reconstruct.templates.source_units import (
    extract_analyzer_unit_ids,
    load_analyzer_source_units,
    resolve_analyzer_source_units_path,
    write_analyzer_source_units,
)


def _analyzer_policy_summary(policy: Any) -> dict[str, Any]:
    return {
        "sparsity_mode": str(policy.sparsity_mode),
        "compute_sparsity": bool(policy.compute_sparsity),
        "sparsity_method": str(policy.sparsity_method),
        "sparsity_radius_um": policy.sparsity_radius_um,
        "sparsity_num_channels": policy.sparsity_num_channels,
        "sparsity_threshold": policy.sparsity_threshold,
        "sparsity_peak_sign": str(policy.sparsity_peak_sign),
        "sparsity_num_spikes_for_sparsity": policy.sparsity_num_spikes_for_sparsity,
        "sparsity_by_property": policy.sparsity_by_property,
        "random_spikes_method": str(policy.random_spikes_method),
        "random_spikes_percentage": policy.random_spikes_percentage,
        "min_spikes_per_unit": policy.min_spikes_per_unit,
        "random_seed": policy.random_seed,
        "log_before_after_spike_counts": bool(policy.log_before_after_spike_counts),
        "margin_size": policy.margin_size,
        "ms_before": policy.ms_before,
        "ms_after": policy.ms_after,
        "dtype": policy.dtype,
        "max_spikes_per_unit": policy.max_spikes_per_unit,
        "n_jobs": policy.n_jobs,
        "chunk_duration": policy.chunk_duration,
    }


def _discover_analyzer_source_names(
    *,
    inputs: TemplatesInputs,
    well_out_dir: Path,
    analyzer_cache_dir: Path | None,
    source_scope: str | None,
) -> list[str]:
    include_concat = bool(inputs.include_concat) and bool(inputs.phases.analyzers.concat.enabled)
    include_segments = bool(inputs.include_segments) and bool(
        inputs.phases.analyzers.segments.enabled
    )
    if source_scope == "concat":
        include_segments = False
    elif source_scope == "segments":
        include_concat = False
    return discover_spikeinterface_analyzer_source_names(
        well_out_dir=well_out_dir,
        concat_analyzer_relpath=(
            inputs.phases.analyzers.concat.analyzer_relpath or inputs.concat_analyzer_relpath
        ),
        concat_sorting_relpath=(
            inputs.phases.analyzers.concat.sorting_relpath or inputs.concat_sorting_relpath
        ),
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
        include_concat=include_concat,
        include_segments=include_segments,
        concat_use_existing_analyzer=bool(inputs.phases.analyzers.concat.use_existing_analyzer),
        concat_build_if_missing=bool(inputs.phases.analyzers.concat.build_if_missing),
        segments_use_existing_analyzer=bool(inputs.phases.analyzers.segments.use_existing_analyzer),
        segments_build_if_missing=bool(inputs.phases.analyzers.segments.build_if_missing),
        limit_segments=inputs.limit_segments,
    )


def _discover_cached_analyzer_source_names(
    *,
    inputs: TemplatesInputs,
    analyzer_cache_dir: Path | None,
    source_scope: str | None,
) -> list[str]:
    include_concat = bool(inputs.include_concat) and bool(inputs.phases.analyzers.concat.enabled)
    include_segments = bool(inputs.include_segments) and bool(
        inputs.phases.analyzers.segments.enabled
    )
    if source_scope == "concat":
        include_segments = False
    elif source_scope == "segments":
        include_concat = False
    return discover_cached_spikeinterface_analyzer_source_names(
        analyzer_cache_dir=analyzer_cache_dir,
        analyzer_cache_concat_subdir=str(inputs.analyzer_cache.concat_analyzer_subdir or "concat"),
        analyzer_cache_segments_subdir=str(inputs.analyzer_cache.segment_analyzers_subdir or ""),
        include_concat=include_concat,
        include_segments=include_segments,
        limit_segments=inputs.limit_segments,
    )


def _source_summary_from_manifest(
    *,
    inputs: TemplatesInputs,
    templates_out_dir: Path,
    source_name: str,
) -> dict[str, Any] | None:
    payload = load_analyzer_source_units(
        templates_out_dir=templates_out_dir,
        source_name=str(source_name),
    )
    if payload is None:
        return None
    unit_ids = payload.get("unit_ids", [])
    if not isinstance(unit_ids, list):
        return None
    unit_count = payload.get("unit_count", None)
    if not isinstance(unit_count, int):
        unit_count = int(len(unit_ids))
    policy = templates_runner._templates_analyzer_policy_for_source(inputs, source_name)
    return {
        "has_sparsity": None,
        "num_channels": None,
        "num_units": int(unit_count),
        "unit_manifest_json": str(
            resolve_analyzer_source_units_path(
                templates_out_dir=templates_out_dir,
                source_name=str(source_name),
            )
        ),
        "policy": _analyzer_policy_summary(policy),
        "manifest_reused": True,
    }


def _inputs_with_analyzer_build_if_missing(
    inputs: TemplatesInputs,
    *,
    concat_build_if_missing: bool,
    segments_build_if_missing: bool,
) -> TemplatesInputs:
    analyzers_phase = inputs.phases.analyzers
    return replace(
        inputs,
        phases=replace(
            inputs.phases,
            analyzers=replace(
                analyzers_phase,
                concat=replace(
                    analyzers_phase.concat,
                    build_if_missing=bool(concat_build_if_missing),
                ),
                segments=replace(
                    analyzers_phase.segments,
                    build_if_missing=bool(segments_build_if_missing),
                ),
            ),
        ),
    )


def _inputs_with_analyzer_use_existing(
    inputs: TemplatesInputs,
    *,
    concat_use_existing_analyzer: bool,
    segments_use_existing_analyzer: bool,
) -> TemplatesInputs:
    analyzers_phase = inputs.phases.analyzers
    return replace(
        inputs,
        phases=replace(
            inputs.phases,
            analyzers=replace(
                analyzers_phase,
                concat=replace(
                    analyzers_phase.concat,
                    use_existing_analyzer=bool(concat_use_existing_analyzer),
                ),
                segments=replace(
                    analyzers_phase.segments,
                    use_existing_analyzer=bool(segments_use_existing_analyzer),
                ),
            ),
        ),
    )


def _analyzer_cache_paths_for_source(
    *,
    inputs: TemplatesInputs,
    analyzer_cache_dir: Path | None,
    source_name: str,
) -> list[Path]:
    if analyzer_cache_dir is None:
        return []
    source_name = str(source_name)
    cache_root = Path(analyzer_cache_dir).expanduser()
    if source_name == "concat":
        concat_subdir = str(inputs.analyzer_cache.concat_analyzer_subdir or "concat").strip().strip("/")
        return [cache_root / (concat_subdir or "concat")]
    segments_subdir = str(inputs.analyzer_cache.segment_analyzers_subdir or "").strip().strip("/")
    paths = []
    if segments_subdir:
        paths.append(cache_root / segments_subdir / source_name)
    paths.append(cache_root / source_name)
    deduped: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def _delete_path_if_exists(path: Path) -> bool:
    if not path.exists():
        return False
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
    return True


def _clear_force_restart_analyzer_artifacts(
    *,
    inputs: TemplatesInputs,
    templates_out_dir: Path,
    analyzer_cache_dir: Path | None,
    source_names: list[str],
) -> dict[str, list[str]]:
    cleared_cache_paths: list[str] = []
    cleared_manifest_paths: list[str] = []
    if source_names:
        for source_name in source_names:
            for cache_path in _analyzer_cache_paths_for_source(
                inputs=inputs,
                analyzer_cache_dir=analyzer_cache_dir,
                source_name=str(source_name),
            ):
                if _delete_path_if_exists(cache_path):
                    cleared_cache_paths.append(str(cache_path))
            manifest_path = resolve_analyzer_source_units_path(
                templates_out_dir=templates_out_dir,
                source_name=str(source_name),
            )
            if _delete_path_if_exists(manifest_path):
                cleared_manifest_paths.append(str(manifest_path))
        return {
            "cleared_cache_paths": cleared_cache_paths,
            "cleared_manifest_paths": cleared_manifest_paths,
        }

    if analyzer_cache_dir is not None and _delete_path_if_exists(Path(analyzer_cache_dir)):
        cleared_cache_paths.append(str(analyzer_cache_dir))
    manifest_dir = resolve_analyzer_source_units_path(
        templates_out_dir=templates_out_dir,
        source_name="__force_restart_inventory_fallback__",
    ).parent
    if _delete_path_if_exists(manifest_dir):
        cleared_manifest_paths.append(str(manifest_dir))
    return {
        "cleared_cache_paths": cleared_cache_paths,
        "cleared_manifest_paths": cleared_manifest_paths,
    }


def _source_summary_from_loaded_analyzer(
    *,
    inputs: TemplatesInputs,
    templates_out_dir: Path,
    source_name: str,
    analyzer: Any,
) -> tuple[dict[str, Any], Path | None, int | None]:
    policy = templates_runner._templates_analyzer_policy_for_source(inputs, source_name)
    num_channels: int | None = None
    get_num_channels = getattr(analyzer, "get_num_channels", None)
    if callable(get_num_channels):
        try:
            num_channels = int(get_num_channels())
        except Exception:
            num_channels = None
    elif hasattr(getattr(analyzer, "recording", None), "get_num_channels"):
        try:
            num_channels = int(analyzer.recording.get_num_channels())
        except Exception:
            num_channels = None
    analyzer_unit_ids = extract_analyzer_unit_ids(analyzer)
    unit_manifest_path: Path | None = None
    unit_count: int | None = None
    if analyzer_unit_ids is not None:
        unit_count = int(len(analyzer_unit_ids))
        unit_manifest_path = write_analyzer_source_units(
            templates_out_dir=templates_out_dir,
            source_name=str(source_name),
            source_kind=("concat" if str(source_name) == "concat" else "segment"),
            unit_ids=list(analyzer_unit_ids),
        )
    return (
        {
            "has_sparsity": bool(getattr(analyzer, "sparsity", None) is not None),
            "num_channels": num_channels,
            "num_units": unit_count,
            "unit_manifest_json": (None if unit_manifest_path is None else str(unit_manifest_path)),
            "policy": _analyzer_policy_summary(policy),
        },
        unit_manifest_path,
        unit_count,
    )


def run_reconstruct_templates_analyzers_phase(
    inputs: TemplatesInputs, *, source_scope: str | None = None
) -> dict[str, Any]:
    phase_started = perf_counter()
    well_out_dir, alternate_well_out_dirs, templates_out_dir, analyzer_cache_dir = (
        templates_runner._resolve_templates_phase_environment(inputs)
    )
    include_concat = bool(inputs.include_concat) and bool(inputs.phases.analyzers.concat.enabled)
    include_segments = bool(inputs.include_segments) and bool(
        inputs.phases.analyzers.segments.enabled
    )
    require_concat = bool(inputs.require_concat_analyzer) and include_concat
    require_segments = bool(inputs.require_segment_analyzers) and include_segments
    if source_scope == "concat":
        include_segments = False
        require_segments = False
    elif source_scope == "segments":
        include_concat = False
        require_concat = False
    templates_runner.LOGGER.info(
        "templates.analyzers start: source_scope=%s well_out_dir=%s templates_out_dir=%s analyzer_cache_dir=%s include_concat=%s include_segments=%s require_concat=%s require_segments=%s force_restart=%s",
        source_scope,
        str(well_out_dir),
        str(templates_out_dir),
        (None if analyzer_cache_dir is None else str(analyzer_cache_dir)),
        bool(include_concat),
        bool(include_segments),
        bool(require_concat),
        bool(require_segments),
        bool(inputs.force_restart),
    )
    templates_runner.LOGGER.info(
        "templates.analyzers concat settings: use_existing=%s build_if_missing=%s analyzer_relpath=%s sorting_relpath=%s preprocessed_recording_reldir=%s settings=%s",
        bool(inputs.phases.analyzers.concat.use_existing_analyzer),
        bool(inputs.phases.analyzers.concat.build_if_missing),
        (inputs.phases.analyzers.concat.analyzer_relpath or inputs.concat_analyzer_relpath),
        (inputs.phases.analyzers.concat.sorting_relpath or inputs.concat_sorting_relpath),
        (
            inputs.phases.analyzers.concat.preprocessed_recording_reldir
            or inputs.preprocessed_concat_reldir
        ),
        templates_runner._format_templates_log_fields(
            templates_runner._templates_analyzer_policy_log_fields(
                templates_runner._resolve_analyzer_policy_runtime_n_jobs(
                    inputs, inputs.phases.analyzers.concat.policy
                )
            )
        ),
    )
    templates_runner.LOGGER.info(
        "templates.analyzers segments settings: use_existing=%s build_if_missing=%s preprocessed_sources_reldir=%s settings=%s",
        bool(inputs.phases.analyzers.segments.use_existing_analyzer),
        bool(inputs.phases.analyzers.segments.build_if_missing),
        (
            inputs.phases.analyzers.segments.preprocessed_sources_reldir
            or inputs.preprocessed_segments_reldir
            or inputs.preproc_seg_sources_reldir
        ),
        templates_runner._format_templates_log_fields(
            templates_runner._templates_analyzer_policy_log_fields(
                templates_runner._resolve_analyzer_policy_runtime_n_jobs(
                    inputs, inputs.phases.analyzers.segments.policy
                )
            )
        ),
    )
    templates_runner.LOGGER.info("templates.analyzers streaming analyzer sources (one at a time)")
    load_stats: dict[str, Any] = {}
    sources_summary: dict[str, Any] = {}
    reused_manifest_count = 0
    generated_manifest_count = 0
    force_restart_artifacts: dict[str, list[str]] = {
        "cleared_cache_paths": [],
        "cleared_manifest_paths": [],
    }
    iteration_alternate_well_out_dirs = list(alternate_well_out_dirs)
    if bool(inputs.force_restart) and not bool(inputs.analyzer_cache.reuse_on_force_restart):
        if iteration_alternate_well_out_dirs:
            templates_runner.LOGGER.info(
                "templates.analyzers force_restart suppressing artifact lookup fallbacks: %s",
                [str(path) for path in iteration_alternate_well_out_dirs],
            )
        iteration_alternate_well_out_dirs = []
    try:
        discovered_source_names = _discover_analyzer_source_names(
            inputs=inputs,
            well_out_dir=well_out_dir,
            analyzer_cache_dir=analyzer_cache_dir,
            source_scope=source_scope,
        )
    except Exception:
        discovered_source_names = []
        templates_runner.LOGGER.warning(
            "templates.analyzers source discovery failed; falling back to full analyzer iteration",
            exc_info=True,
        )
    if bool(inputs.force_restart):
        force_restart_artifacts = _clear_force_restart_analyzer_artifacts(
            inputs=inputs,
            templates_out_dir=templates_out_dir,
            analyzer_cache_dir=analyzer_cache_dir,
            source_names=[str(source_name) for source_name in discovered_source_names],
        )
        templates_runner.LOGGER.info(
            "templates.analyzers force_restart cleared scoped analyzer artifacts: source_count=%d cache_paths=%d manifest_paths=%d",
            int(len(discovered_source_names)),
            int(len(force_restart_artifacts.get("cleared_cache_paths", []))),
            int(len(force_restart_artifacts.get("cleared_manifest_paths", []))),
        )
    try:
        cached_source_names = set(
            _discover_cached_analyzer_source_names(
                inputs=inputs,
                analyzer_cache_dir=analyzer_cache_dir,
                source_scope=source_scope,
            )
        )
    except Exception:
        cached_source_names = set()
        templates_runner.LOGGER.warning(
            "templates.analyzers cached source discovery failed; existing manifests will not be reused",
            exc_info=True,
        )
    missing_manifest_source_names: list[str] = []
    if discovered_source_names:
        for source_name in discovered_source_names:
            if bool(inputs.force_restart):
                missing_manifest_source_names.append(str(source_name))
                continue
            existing_summary = _source_summary_from_manifest(
                inputs=inputs,
                templates_out_dir=templates_out_dir,
                source_name=str(source_name),
            )
            if existing_summary is None:
                missing_manifest_source_names.append(str(source_name))
                continue
            sources_summary[str(source_name)] = existing_summary
            reused_manifest_count += 1
        templates_runner.LOGGER.info(
            "templates.analyzers manifest resume check: discovered_source_count=%d cached_source_count=%d reused_manifest_count=%d missing_manifest_count=%d",
            int(len(discovered_source_names)),
            int(len(cached_source_names)),
            int(reused_manifest_count),
            int(len(missing_manifest_source_names)),
        )
        if missing_manifest_source_names:
            templates_runner.LOGGER.info(
                "templates.analyzers loading analyzers only for missing source manifests: sources=%s",
                list(missing_manifest_source_names),
            )
        else:
            templates_runner.LOGGER.info(
                "templates.analyzers all cached source unit manifests already present; skipping analyzer loads"
            )
    else:
        templates_runner.LOGGER.info(
            "templates.analyzers source discovery returned no resumable inventory; falling back to analyzer iteration"
        )
    requested_source_names = (
        None if not discovered_source_names else list(missing_manifest_source_names)
    )

    def _consume_loaded_analyzers(
        iter_inputs: TemplatesInputs,
        *,
        requested_names: list[str] | None,
    ) -> set[str]:
        nonlocal generated_manifest_count
        loaded_source_names: set[str] = set()
        for source_name, analyzer in templates_runner._iter_templates_phase_analyzers(
            inputs=iter_inputs,
            well_out_dir=well_out_dir,
            alternate_well_out_dirs=iteration_alternate_well_out_dirs,
            analyzer_cache_dir=analyzer_cache_dir,
            source_scope=source_scope,
            requested_source_names=requested_names,
            load_stats=load_stats,
        ):
            source_name = str(source_name)
            loaded_source_names.add(source_name)
            source_summary, unit_manifest_path, unit_count = _source_summary_from_loaded_analyzer(
                inputs=inputs,
                templates_out_dir=templates_out_dir,
                source_name=source_name,
                analyzer=analyzer,
            )
            if unit_manifest_path is not None:
                generated_manifest_count += 1
                templates_runner.LOGGER.info(
                    "templates.analyzers wrote source unit manifest: source=%s unit_count=%d path=%s",
                    source_name,
                    int(unit_count or 0),
                    str(unit_manifest_path),
                )
            sources_summary[source_name] = source_summary
            del analyzer
            gc.collect()
        return loaded_source_names

    if requested_source_names is None or requested_source_names:
        if bool(inputs.force_restart):
            restart_inputs = _inputs_with_analyzer_use_existing(
                inputs,
                concat_use_existing_analyzer=False,
                segments_use_existing_analyzer=False,
            )
            _consume_loaded_analyzers(restart_inputs, requested_names=requested_source_names)
        elif requested_source_names is None:
            _consume_loaded_analyzers(inputs, requested_names=None)
        elif requested_source_names:
            templates_runner.LOGGER.info(
                "templates.analyzers backfilling missing manifests from existing analyzers before rebuild: sources=%s",
                list(requested_source_names),
            )
            no_build_inputs = _inputs_with_analyzer_build_if_missing(
                inputs,
                concat_build_if_missing=False,
                segments_build_if_missing=False,
            )
            resolved_existing_sources = _consume_loaded_analyzers(
                no_build_inputs,
                requested_names=list(requested_source_names),
            )
            unresolved_sources = [
                source_name
                for source_name in requested_source_names
                if str(source_name) not in resolved_existing_sources
            ]
            if unresolved_sources:
                templates_runner.LOGGER.info(
                    "templates.analyzers existing-analyzer backfill unresolved; allowing analyzer rebuild for remaining sources: sources=%s",
                    list(unresolved_sources),
                )
                _consume_loaded_analyzers(inputs, requested_names=unresolved_sources)
    source_count = int(len(sources_summary))
    concat_count = int(sum(1 for source_name in sources_summary.keys() if str(source_name) == "concat"))
    segment_count = int(source_count - concat_count)
    source_unit_manifest_count = int(
        sum(
            1
            for source_summary in sources_summary.values()
            if source_summary.get("unit_manifest_json", None) is not None
        )
    )
    summary = {
        "phase": ("analyzers" if source_scope is None else f"analyzers.{source_scope}"),
        "stream_id": str(inputs.stream_id),
        "well_out_dir": str(well_out_dir),
        "templates_out_dir": str(templates_out_dir),
        "applied_debug_limits": templates_runner._templates_applied_debug_limits(inputs),
        "analyzer_cache_dir": (None if analyzer_cache_dir is None else str(analyzer_cache_dir)),
        "source_scope": source_scope,
        "source_count": int(source_count),
        "source_unit_manifest_count": int(source_unit_manifest_count),
        "manifest_resume": {
            "discovered_source_count": int(len(discovered_source_names)),
            "reused_manifest_count": int(reused_manifest_count),
            "generated_manifest_count": int(generated_manifest_count),
            "missing_manifest_sources": list(missing_manifest_source_names),
        },
        "force_restart_artifacts": force_restart_artifacts,
        "load_stats": load_stats,
        "sources": sources_summary,
    }
    summary["timing"] = {"duration_seconds": float(perf_counter() - phase_started)}
    summary_path = templates_out_dir / str(inputs.phases.analyzers.summary_json_relpath)
    templates_runner.LOGGER.info(
        "templates.analyzers generating outputs: summary_json=%s", str(summary_path)
    )
    templates_runner.write_json(summary_path, summary)
    summary["summary_json"] = str(summary_path)
    templates_runner.LOGGER.info("templates.analyzers wrote summary output: %s", str(summary_path))
    templates_runner.LOGGER.info(
        "templates.analyzers run stats: duration_seconds=%.3f source_count=%d concat_count=%d segment_count=%d load_stats=%s",
        float(summary["timing"]["duration_seconds"]),
        int(source_count),
        int(concat_count),
        int(segment_count),
        load_stats,
    )
    return summary
