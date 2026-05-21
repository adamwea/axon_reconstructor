from __future__ import annotations

from pathlib import Path
from typing import Any

from axon_recon.pipeline.checkpoint import with_checkpoint_marker
from axon_recon.pipeline.shared.grid_sorting import normalize_grid_sort_by
from axon_recon.pipeline.stages.reconstruct import runner as reconstruct_runner
from axon_recon.pipeline.stages.reconstruct.models.inputs import ReconstructionInputs
from axon_recon.pipeline.stages.reconstruct.models.results import UnitReconstructionResult


def run_reconstruct_report_recons_phase(inputs: ReconstructionInputs) -> dict[str, Any]:
    from axon_recon.pipeline.config import get_dry_run_override, get_no_plot_override

    env = reconstruct_runner._prepare_reconstruct_phase_environment(
        inputs=inputs, clear_output_root=False
    )
    summary_json = (
        env.reconstruction_out_dir
        / Path(str(inputs.phases.report_recons.summary_json_relpath)).expanduser()
    )

    if get_dry_run_override():
        return reconstruct_runner.reconstruct_phase_dry_run_short_circuit(
            inputs=inputs, phase_name="report_recons", summary_json=summary_json
        )

    with with_checkpoint_marker(
        summary_json,
        phase_name="report_recons",
        stage_name="reconstruct",
    ):
        if get_no_plot_override() is True:
            return reconstruct_runner.reconstruct_phase_plots_disabled_skip(
                inputs=inputs, phase_name="report_recons", summary_json=summary_json
            )
        unit_results = reconstruct_runner._load_reconstruct_unit_results(
            reconstruction_out_dir=env.reconstruction_out_dir,
            inputs=inputs,
            unit_ids=env.unit_ids,
        )
        stage_outputs = _run_reconstruct_report_recons_phase_impl(
            inputs=inputs, env=env, unit_results=unit_results
        )
        return reconstruct_runner._write_reconstruct_phase_summary(
            phase_name="report_recons",
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


def _run_reconstruct_report_recons_phase_impl(
    *,
    inputs: ReconstructionInputs,
    env: Any,
    unit_results: list[UnitReconstructionResult],
) -> dict[str, str]:
    report_sort_by = normalize_grid_sort_by(inputs.report_sort_by, default="unit_id")
    unit_results_for_reports = reconstruct_runner._sort_reconstruct_units_for_reports(
        unit_results=unit_results,
        reconstruction_out_dir=env.reconstruction_out_dir,
        inputs=inputs,
        sort_by=report_sort_by,
    )
    return reconstruct_runner.run_report_recons_core_phase(
        inputs=inputs,
        reconstruction_out_dir=env.reconstruction_out_dir,
        unit_results=unit_results,
        unit_results_for_reports=unit_results_for_reports,
        preserve_stage_reports=env.preserve_stage_reports,
        existing_stage_outputs=env.existing_stage_outputs,
        resolve_report_output_paths_fn=reconstruct_runner.resolve_report_output_paths,
        write_amplitude_map_summary_png_fn=reconstruct_runner.write_amplitude_map_summary_png,
        render_template_report_pdf_fn=reconstruct_runner.render_template_report_pdf,
        write_reconstruct_report_markdown_fn=reconstruct_runner.write_reconstruct_report_markdown,
        logger=reconstruct_runner.LOGGER,
    )
