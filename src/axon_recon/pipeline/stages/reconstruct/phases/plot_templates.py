from __future__ import annotations

from dataclasses import replace
from time import perf_counter
from typing import Any

from axon_recon.pipeline.stages.reconstruct.templates import runner as templates_runner
from axon_recon.pipeline.stages.reconstruct.templates.models.inputs import TemplatesInputs
from axon_recon.pipeline.stages.reconstruct.templates.models.results import TemplatesResult


def run_reconstruct_templates_plot_templates_phase(inputs: TemplatesInputs) -> dict[str, Any]:
    phase_started = perf_counter()
    well_out_dir, _, templates_out_dir, _ = templates_runner._resolve_templates_phase_environment(
        inputs
    )
    try:
        merged_units_dir, _ = templates_runner._resolve_templates_dirs(
            well_out_dir=well_out_dir,
            templates_out_dir=templates_out_dir,
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "Missing built template artifacts for plot_templates; run templates.build_templates first"
        ) from exc
    unit_ids = templates_runner._build_unit_ids(inputs, merged_units_dir)
    unit_ids = templates_runner._apply_unit_label_filter(
        inputs, unit_ids, well_out_dir, context="plot_templates"
    )
    if not unit_ids:
        raise FileNotFoundError(
            f"No built template artifacts found under {merged_units_dir}; run templates.build_templates first"
        )
    phase_inputs = templates_runner.build_plot_templates_phase_inputs(inputs)
    requested_outputs = templates_runner.requested_plot_output_keys(phase_inputs.per_unit_outputs)
    force_replot_requested = (
        bool(inputs.force_restart)
        or bool(inputs.force_replot)
        or bool(inputs.force_replot_per_unit)
    )
    templates_runner._cleanup_unit_output_artifacts(
        templates_out_dir=templates_out_dir,
        unit_ids=unit_ids,
        per_unit_outputs=phase_inputs.per_unit_outputs,
        output_keys=templates_runner.excluded_plot_output_keys(),
    )
    units_to_render: list[Any] = []
    skipped_units: list[Any] = []
    for unit_id in unit_ids:
        existing_outputs = templates_runner._collect_existing_unit_output_paths(
            templates_out_dir=templates_out_dir,
            unit_id=unit_id,
            per_unit_outputs=phase_inputs.per_unit_outputs,
            output_keys=requested_outputs,
        )
        if (not force_replot_requested) and len(existing_outputs) == len(requested_outputs):
            templates_runner._persist_unit_summary_output_paths(
                templates_out_dir=templates_out_dir,
                unit_id=unit_id,
                per_unit_outputs=phase_inputs.per_unit_outputs,
                output_paths=existing_outputs,
            )
            skipped_units.append(unit_id)
        else:
            units_to_render.append(unit_id)
    templates_runner.LOGGER.info(
        "templates.plot_templates start: templates_out_dir=%s units=%d units_to_render=%d skipped_units=%d force_restart=%s",
        str(templates_out_dir),
        len(unit_ids),
        len(units_to_render),
        len(skipped_units),
        bool(inputs.force_restart),
    )
    result = TemplatesResult(
        well_out_dir=well_out_dir,
        templates_out_dir=templates_out_dir,
        summary_json=templates_out_dir / "templates_summary.json",
        units=[],
    )
    if units_to_render:
        result = _run_reconstruct_templates_plot_batches(
            inputs=phase_inputs,
            well_out_dir=well_out_dir,
            templates_out_dir=templates_out_dir,
            unit_ids=units_to_render,
        )
    summary = templates_runner.build_plot_templates_phase_summary(
        inputs=phase_inputs,
        result=result,
        skipped_units=skipped_units,
        duration_seconds=float(perf_counter() - phase_started),
    )
    summary["applied_debug_limits"] = templates_runner._templates_applied_debug_limits(phase_inputs)
    summary_path = templates_out_dir / str(inputs.phases.plot_templates.summary_json_relpath)
    templates_runner.write_json(summary_path, summary)
    summary["summary_json"] = str(summary_path)
    templates_runner.LOGGER.info(
        "templates.plot_templates wrote summary output: %s", str(summary_path)
    )
    templates_runner.LOGGER.info(
        "templates.plot_templates run stats: duration_seconds=%.3f unit_count=%d rendered_units=%d skipped_units=%d failed_units=%d",
        float(summary["duration_seconds"]),
        int(summary.get("unit_count", 0)),
        int(len(summary.get("rendered_units", []))),
        int(len(summary.get("skipped_units", []))),
        int(len(summary.get("failed_units", []))),
    )
    return summary


def _resolve_plot_templates_execution_plan(
    *,
    inputs: TemplatesInputs,
    unit_ids: list[Any],
) -> tuple[int, int, int, list[list[Any]]]:
    unit_count = len(unit_ids)
    if unit_count <= 0:
        return 1, 1, 1, []
    return 1, 1, int(unit_count), [list(unit_ids)]


def _run_reconstruct_templates_plot_batches(
    *,
    inputs: TemplatesInputs,
    well_out_dir: Any,
    templates_out_dir: Any,
    unit_ids: list[Any],
) -> TemplatesResult:
    plot_unit_workers, unit_procs, unit_batch_size, batches = (
        _resolve_plot_templates_execution_plan(
            inputs=inputs,
            unit_ids=unit_ids,
        )
    )
    templates_runner.LOGGER.info(
        "templates.plot_templates execution plan: requested_units=%d derived_unit_workers=%d plot_unit_workers=%d unit_procs=%d unit_batch_size=%d unit_batches=%d parallel=false",
        len(unit_ids),
        int(max(1, int(inputs.n_jobs))),
        int(plot_unit_workers),
        int(unit_procs),
        int(unit_batch_size),
        len(batches),
    )
    with templates_runner._quiet_unexpected_plot_logs(inputs):
        return templates_runner._run_reconstruct_templates_pipeline_monolithic(
            replace(inputs, unit_ids=list(unit_ids), n_jobs=1)
        )
