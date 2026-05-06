from __future__ import annotations

import gc
import shutil
from time import perf_counter
from typing import Any

from axon_recon.pipeline.stages.reconstruct.templates import runner as templates_runner
from axon_recon.pipeline.stages.reconstruct.templates.models.inputs import TemplatesInputs


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
    if (
        bool(inputs.force_restart)
        and analyzer_cache_dir is not None
        and analyzer_cache_dir.exists()
    ):
        templates_runner.LOGGER.info(
            "templates.analyzers clearing analyzer cache on force_restart: %s",
            str(analyzer_cache_dir),
        )
        shutil.rmtree(analyzer_cache_dir)
    templates_runner.LOGGER.info("templates.analyzers streaming analyzer sources (one at a time)")
    load_stats: dict[str, Any] = {}
    sources_summary: dict[str, Any] = {}
    source_count = 0
    concat_count = 0
    segment_count = 0
    for source_name, analyzer in templates_runner._iter_templates_phase_analyzers(
        inputs=inputs,
        well_out_dir=well_out_dir,
        alternate_well_out_dirs=alternate_well_out_dirs,
        analyzer_cache_dir=analyzer_cache_dir,
        source_scope=source_scope,
        load_stats=load_stats,
    ):
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
        sources_summary[str(source_name)] = {
            "has_sparsity": bool(getattr(analyzer, "sparsity", None) is not None),
            "num_channels": num_channels,
            "num_units": int(len(list(getattr(analyzer.sorting, "unit_ids", []))))
            if hasattr(analyzer, "sorting")
            else None,
            "policy": {
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
            },
        }
        source_count += 1
        if str(source_name) == "concat":
            concat_count += 1
        else:
            segment_count += 1
        del analyzer
        gc.collect()
    summary = {
        "phase": ("analyzers" if source_scope is None else f"analyzers.{source_scope}"),
        "stream_id": str(inputs.stream_id),
        "well_out_dir": str(well_out_dir),
        "templates_out_dir": str(templates_out_dir),
        "applied_debug_limits": templates_runner._templates_applied_debug_limits(inputs),
        "analyzer_cache_dir": (None if analyzer_cache_dir is None else str(analyzer_cache_dir)),
        "source_scope": source_scope,
        "source_count": int(source_count),
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
