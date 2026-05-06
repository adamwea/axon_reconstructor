from __future__ import annotations

from typing import Any

from axon_recon.pipeline.output_paths import compute_mea_analysis_output_dir
from axon_recon.pipeline.stages.reconstruct.templates import runner as templates_runner
from axon_recon.pipeline.stages.reconstruct.templates.models.inputs import TemplatesInputs


def run_reconstruct_templates_resolve_sources_phase(inputs: TemplatesInputs) -> dict[str, Any]:
    phase_cfg = inputs.resolve_sources_phase
    well_out_dir = compute_mea_analysis_output_dir(
        output_root=inputs.mea_output_root,
        data_file=inputs.h5_path,
        well=inputs.stream_id,
    )
    alternate_well_out_dirs = (
        templates_runner._resolve_alternate_well_out_dirs(
            inputs=inputs, primary_well_out_dir=well_out_dir
        )
        if bool(phase_cfg.include_alternate_well_dirs)
        else []
    )
    well_dirs = [well_out_dir, *list(alternate_well_out_dirs)]

    concat_analyzer_tokens = templates_runner._dedupe_string_tokens(
        [
            str(inputs.concat_analyzer_relpath or ""),
            "/spikesort_outputs/analyzer_output",
            "/stg2_spikesorting_outputs/analyzer_output",
            "spikesort_outputs/analyzer_output",
            "stg2_spikesorting_outputs/analyzer_output",
        ]
    )
    concat_sorting_tokens = templates_runner._dedupe_string_tokens(
        [
            str(inputs.concat_sorting_relpath or ""),
            "/spikesort_outputs/sorter_output",
            "/stg2_spikesorting_outputs/sorter_output",
            "spikesort_outputs/sorter_output",
            "stg2_spikesorting_outputs/sorter_output",
        ]
    )
    preprocessed_concat_tokens = templates_runner._dedupe_string_tokens(
        [
            str(inputs.preprocessed_concat_reldir or ""),
            "/preprocess_outputs/preprocessed_recording",
            "preprocess_outputs/preprocessed_recording",
        ]
    )
    preprocessed_segments_tokens = templates_runner._dedupe_string_tokens(
        [
            str(inputs.preprocessed_segments_reldir or ""),
            str(inputs.preproc_seg_sources_reldir or ""),
            "/preprocess_outputs/per_segment_preprocessed",
            "/preprocess_outputs/per_segment_recordings",
            "preprocess_outputs/per_segment_preprocessed",
            "preprocess_outputs/per_segment_recordings",
        ]
    )

    concat_analyzer_candidates = templates_runner._resolve_well_relative_path_candidates(
        well_dirs=well_dirs,
        relpath_tokens=concat_analyzer_tokens,
    )
    concat_sorting_candidates = templates_runner._resolve_well_relative_path_candidates(
        well_dirs=well_dirs,
        relpath_tokens=concat_sorting_tokens,
    )
    preprocessed_concat_candidates = templates_runner._resolve_well_relative_path_candidates(
        well_dirs=well_dirs,
        relpath_tokens=preprocessed_concat_tokens,
    )
    preprocessed_segments_candidates = templates_runner._resolve_well_relative_path_candidates(
        well_dirs=well_dirs,
        relpath_tokens=preprocessed_segments_tokens,
    )

    source_summaries = {
        "concat_analyzer": templates_runner._summarize_source_candidates(
            name="concat_analyzer",
            tokens=concat_analyzer_tokens,
            candidate_paths=concat_analyzer_candidates,
            check_path_exists=bool(phase_cfg.check_path_exists),
            max_candidates_per_source=int(phase_cfg.max_candidates_per_source),
        ),
        "concat_sorting": templates_runner._summarize_source_candidates(
            name="concat_sorting",
            tokens=concat_sorting_tokens,
            candidate_paths=concat_sorting_candidates,
            check_path_exists=bool(phase_cfg.check_path_exists),
            max_candidates_per_source=int(phase_cfg.max_candidates_per_source),
        ),
        "preprocessed_concat": templates_runner._summarize_source_candidates(
            name="preprocessed_concat",
            tokens=preprocessed_concat_tokens,
            candidate_paths=preprocessed_concat_candidates,
            check_path_exists=bool(phase_cfg.check_path_exists),
            max_candidates_per_source=int(phase_cfg.max_candidates_per_source),
        ),
        "preprocessed_segments": templates_runner._summarize_source_candidates(
            name="preprocessed_segments",
            tokens=preprocessed_segments_tokens,
            candidate_paths=preprocessed_segments_candidates,
            check_path_exists=bool(phase_cfg.check_path_exists),
            max_candidates_per_source=int(phase_cfg.max_candidates_per_source),
        ),
    }

    unit_label_probe: dict[str, Any] = {
        "allowed_labels": list(inputs.unit_label_filter_labels),
        "required": bool(inputs.unit_label_filter_required),
        "probe_attempted": False,
        "available": None,
        "counts_by_label": {},
    }
    if bool(phase_cfg.probe_unit_labels) and inputs.unit_label_filter_labels:
        unit_label_probe["probe_attempted"] = True
        labels_by_unit = templates_runner.load_unit_labels_from_spikesorting(well_out_dir)
        if labels_by_unit is not None:
            unit_label_probe["available"] = True
            unit_label_probe["counts_by_label"] = templates_runner.count_labels(labels_by_unit)
        else:
            unit_label_probe["available"] = False

    summary: dict[str, Any] = {
        "phase": "resolve_sources",
        "stream_id": str(inputs.stream_id),
        "h5_path": str(inputs.h5_path),
        "well_out_dir": str(well_out_dir),
        "alternate_well_out_dirs": [str(path) for path in alternate_well_out_dirs],
        "run_intent": {
            "force_restart": bool(inputs.force_restart),
            "force_replot": bool(inputs.force_replot),
            "force_replot_per_unit": bool(inputs.force_replot_per_unit),
            "force_rereport": bool(inputs.force_rereport),
        },
        "applied_debug_limits": templates_runner._templates_applied_debug_limits(inputs),
        "unit_scope": {
            "unit_ids": (None if inputs.unit_ids is None else list(inputs.unit_ids)),
            "unit_limit": inputs.unit_limit,
            "unit_label_filter": unit_label_probe,
        },
        "source_requirements": {
            "include_concat": bool(inputs.include_concat),
            "include_segments": bool(inputs.include_segments),
            "require_concat_analyzer": bool(inputs.require_concat_analyzer),
            "require_segment_analyzers": bool(inputs.require_segment_analyzers),
        },
        "sources": source_summaries,
    }

    if bool(phase_cfg.fail_if_required_sources_missing):
        missing_required: list[str] = []
        if bool(inputs.include_concat) and bool(inputs.require_concat_analyzer):
            if source_summaries["concat_analyzer"].get("first_existing", None) is None:
                missing_required.append("concat_analyzer")
        if bool(inputs.include_segments) and bool(inputs.require_segment_analyzers):
            if source_summaries["preprocessed_segments"].get("first_existing", None) is None:
                missing_required.append("preprocessed_segments")
        if missing_required:
            raise RuntimeError(
                "resolve_sources required inputs missing: " + ", ".join(missing_required)
            )

    if bool(phase_cfg.enabled):
        if bool(phase_cfg.show_header):
            header_title = f"templates.resolve_sources [{inputs.stream_id}]"
            header_line = "=" * max(24, len(header_title))
            templates_runner.LOGGER.info(header_line)
            templates_runner.LOGGER.info(header_title)
            templates_runner.LOGGER.info(header_line)
        templates_runner.LOGGER.info(
            "resolve_sources: stream=%s run_intent=%s",
            str(inputs.stream_id),
            summary["run_intent"],
        )
        templates_runner.LOGGER.info(
            "resolve_sources: well_out_dir=%s alternate_well_out_dirs=%s",
            str(well_out_dir),
            ["%s" % path for path in alternate_well_out_dirs],
        )
        for source_name, payload in source_summaries.items():
            templates_runner.LOGGER.info(
                "resolve_sources: %s first_existing=%s candidates=%d",
                source_name,
                payload.get("first_existing", None),
                int(payload.get("candidate_count", 0)),
            )
            if bool(phase_cfg.log_candidates):
                for row in list(payload.get("candidates", [])):
                    templates_runner.LOGGER.info(
                        "resolve_sources: %s candidate path=%s exists=%s",
                        source_name,
                        row.get("path", None),
                        row.get("exists", None),
                    )
        templates_runner.LOGGER.info("resolve_sources: unit_scope=%s", summary["unit_scope"])

    if bool(phase_cfg.write_json):
        templates_out_dir = well_out_dir / str(inputs.output_rel_root)
        json_path = templates_out_dir / str(phase_cfg.json_relpath)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        templates_runner.write_json(json_path, summary)
        summary["summary_json"] = str(json_path)

    return summary
