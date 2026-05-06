from __future__ import annotations

from pathlib import Path
from typing import Any

from axon_recon.pipeline.shared.grid_sorting import normalize_grid_sort_by
from axon_recon.pipeline.stages.reconstruct import runner as reconstruct_runner
from axon_recon.pipeline.stages.reconstruct.models.inputs import ReconstructionInputs
from axon_recon.pipeline.stages.reconstruct.models.results import UnitReconstructionResult


def run_reconstruct_report_summaries_phase(inputs: ReconstructionInputs) -> dict[str, Any]:
    env = reconstruct_runner._prepare_reconstruct_phase_environment(
        inputs=inputs, clear_output_root=False
    )
    unit_results = reconstruct_runner._load_reconstruct_unit_results(
        reconstruction_out_dir=env.reconstruction_out_dir,
        inputs=inputs,
        unit_ids=env.unit_ids,
    )
    stage_outputs = _run_reconstruct_report_summaries_phase_impl(
        inputs=inputs,
        env=env,
        unit_results=unit_results,
        stage_outputs=dict(env.existing_stage_outputs),
    )
    summary_json = (
        env.reconstruction_out_dir
        / Path(str(inputs.phases.report_summaries.summary_json_relpath)).expanduser()
    )
    return reconstruct_runner._write_reconstruct_phase_summary(
        phase_name="report_summaries",
        summary_json=summary_json,
        inputs=inputs,
        well_out_dir=env.well_out_dir,
        reconstruction_out_dir=env.reconstruction_out_dir,
        unit_results=unit_results,
        stage_outputs=stage_outputs,
        failed_units_summary_json=reconstruct_runner._current_failed_units_summary_json(
            inputs=inputs,
            reconstruction_out_dir=env.reconstruction_out_dir,
        ),
        preserve_stage_reports=env.preserve_stage_reports,
        extra_fields={
            "report_sort_by": normalize_grid_sort_by(inputs.report_sort_by, default="unit_id"),
        },
    )


def _run_reconstruct_report_summaries_phase_impl(
    *,
    inputs: ReconstructionInputs,
    env: Any,
    unit_results: list[UnitReconstructionResult],
    stage_outputs: dict[str, str],
) -> dict[str, str]:
    report_sort_by = normalize_grid_sort_by(inputs.report_sort_by, default="unit_id")
    unit_results_for_reports = reconstruct_runner._sort_reconstruct_units_for_reports(
        unit_results=unit_results,
        reconstruction_out_dir=env.reconstruction_out_dir,
        inputs=inputs,
        sort_by=report_sort_by,
    )
    return reconstruct_runner.run_report_summaries_core_phase(
        inputs=inputs,
        reconstruction_out_dir=env.reconstruction_out_dir,
        unit_results=unit_results,
        unit_results_for_reports=unit_results_for_reports,
        preserve_stage_reports=env.preserve_stage_reports,
        existing_stage_outputs=stage_outputs,
        resolve_report_output_paths_fn=reconstruct_runner.resolve_report_output_paths,
        write_reconstruct_summary_slides_pdf_fn=reconstruct_runner.write_reconstruct_summary_slides_pdf,
        logger=reconstruct_runner.LOGGER,
    )
